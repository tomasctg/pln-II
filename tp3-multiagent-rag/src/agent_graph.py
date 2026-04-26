import os
import json
from typing import List, Dict, Any, Literal
from typing_extensions import TypedDict
from langgraph.graph import StateGraph, END, START
from langchain_groq import ChatGroq
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage

class AgentState(TypedDict):
    query: str
    chat_history: List[Dict[str, str]]
    names_to_query: List[str]
    agent_responses: List[str]
    final_response: str
    retrieved_chunks: List[Dict[str, Any]]
    groq_api_key: str
    pinecone_api_key: str
    modelo_llm: str
    top_k: int
    available_profiles: List[str]
    revision_count: int
    reviewer_feedback: str

def supervisor_node(state: AgentState) -> dict:
    """
    Analiza la consulta y determina qué perfiles (CVs) se están consultando.
    Si no se menciona ninguno, por defecto se usa 'Corteggiano'.
    """
    llm = ChatGroq(model=state.get("modelo_llm", "llama-3.1-8b-instant"), api_key=state["groq_api_key"])
    
    historial_texto = ""
    if state.get("chat_history"):
        historial_texto = "Historial reciente de la conversación:\n"
        for msg in state["chat_history"][-3:]:
            if "pregunta" in msg and "respuesta" in msg:
                historial_texto += f"- Usuario: {msg['pregunta']}\n- Asistente: {msg['respuesta']}\n"
            elif "role" in msg and "content" in msg:
                historial_texto += f"- {msg['role']}: {msg['content']}\n"
            
    prompt = f"""Eres un Analista de Intenciones (Intent Classifier). Tu tarea es clasificar la consulta del usuario en una de las siguientes categorías para enrutarla a los perfiles correctos.
Actualmente tenemos los siguientes perfiles exactos en la base de datos: {state.get('available_profiles', [])}

CATEGORÍAS DE INTENCIÓN (Elige UNA):
- "all_profiles": La consulta está explícitamente en PLURAL (verbos como "estudiaron", "trabajaron", "tienen", etc.) y hace referencia a todos los perfiles, o pide compararlos.
- "owner_profile": La consulta NO menciona ningún nombre, está explícitamente en SINGULAR y pide información personal/profesional del dueño principal (ej. "¿cuál es mi email?", "¿tu teléfono?", "trayectoria").
- "specific_names": La consulta menciona explícitamente a una o más personas, o la consulta es una continuación directa (por pronombres o sujeto tácito) del mensaje anterior donde ya se estaba hablando de personas específicas.
- "general": Consultas fuera de contexto.

REGLA VITAL: Si eliges "specific_names" debido a un sujeto tácito del historial (ej. el usuario acaba de preguntar por "Tomas y Rodrigo" y ahora dice "¿en qué empresas trabajaron?"), DEBES incluir obligatoriamente esos nombres en la lista "extracted_names".

{historial_texto}
Consulta actual del usuario: {state['query']}

Responde ÚNICAMENTE con un JSON válido con esta estructura.
{{
  "intent": "owner_profile",
  "extracted_names": [] 
}}

NOTA: "extracted_names" solo debe contener nombres si la intención es "specific_names". En caso contrario, déjalo vacío `[]`.
"""
    
    try:
        response = llm.invoke(prompt)
        content = response.content.replace("```json", "").replace("```", "").strip()
        result = json.loads(content)
        intent = result.get("intent", "owner_profile")
        nombres = result.get("extracted_names", [])
    except Exception as e:
        print(f"Error parseando supervisor: {e}")
        intent = "owner_profile"
        nombres = []
        
    perfiles = state.get("available_profiles", [])
    nombres_validos = []
    
    # Lógica determinística basada en la intención
    if intent == "all_profiles":
        nombres_validos = perfiles
    elif intent == "owner_profile":
        # Siempre busca al dueño principal (Tomas)
        for p in perfiles:
            if "tomas" in p.lower() or "corteggiano" in p.lower():
                nombres_validos = [p]
                break
        if not nombres_validos and perfiles:
            nombres_validos = [perfiles[0]]
    elif intent == "specific_names":
        nombres_validos = [n for n in nombres if n in perfiles]
    else: # general
        nombres_validos = []
        
    return {"names_to_query": list(set(nombres_validos))}

def person_agents_node(state: AgentState) -> dict:
    """
    Este nodo actúa como el orquestador de los 'Agentes por Persona'.
    Recorre los nombres, busca en Pinecone con filtro de metadata y genera una respuesta
    específica para esa persona.
    """
    import app # Re-use the existing logic from app.py for retrieval
    
    pc = app.inicializar_pinecone(state["pinecone_api_key"])
    indice = pc.Index(app.NOMBRE_INDICE)
    
    modelo_emb = app.cargar_modelo_embeddings()
    emb_pregunta = app.generar_embeddings(modelo_emb, [state["query"]])[0]
    
    from groq import Groq
    cliente_groq = Groq(api_key=state["groq_api_key"])
    
    respuestas = []
    all_chunks_recuperados = []
    
    for nombre in state["names_to_query"]:
        # Buscar chunks en Pinecone filtrando por person_name
        resultados = indice.query(
            vector=emb_pregunta,
            top_k=state.get("top_k", 3),
            include_metadata=True,
            filter={"person_name": {"$eq": nombre}}
        )
        
        chunks_encontrados = []
        for match in resultados["matches"]:
            chunks_encontrados.append({
                "id": match["id"],
                "score": round(match["score"], 4),
                "texto": match["metadata"]["texto"],
            })
            
        if not chunks_encontrados:
            respuestas.append(f"No encontré información en el CV de {nombre} relacionada con la pregunta.")
            continue
            
        # Generar respuesta RAG para esta persona
        respuesta_rag = app.generar_respuesta_rag(
            cliente_groq,
            state.get("modelo_llm", "llama-3.1-8b-instant"),
            state["query"],
            chunks_encontrados,
            state["chat_history"],
            agent_name=nombre
        )
        
        respuestas.append(f"**Respuesta del Agente de {nombre}:**\n{respuesta_rag}")
        all_chunks_recuperados.extend(chunks_encontrados)
        
    return {"agent_responses": respuestas, "retrieved_chunks": all_chunks_recuperados}

def combiner_node(state: AgentState) -> dict:
    """
    Si hubo más de un perfil consultado, este nodo combina las respuestas.
    Si fue solo uno, devuelve esa misma respuesta.
    """
    if not state["agent_responses"]:
        return {"final_response": "La consulta no menciona a ninguna de las personas en mi base de datos ni parece estar dirigida a mis perfiles. Por favor, especifica sobre qué CV deseas preguntar (ej. Corteggiano o Juan Perez)."}
        
    llm = ChatGroq(model=state.get("modelo_llm", "llama-3.1-8b-instant"), api_key=state["groq_api_key"])
    
    respuestas_texto = "\n\n---\n\n".join(state["agent_responses"])
    
    prompt = f"""El usuario hizo la siguiente pregunta: '{state['query']}'
    
Hemos consultado a los agentes de las diferentes personas involucradas y han respondido lo siguiente:
{respuestas_texto}

Tu tarea es tomar estas respuestas individuales y combinarlas en una respuesta única, coherente y natural para el usuario.
REGLAS ESTRICTAS:
1. No menciones que "hemos consultado a los agentes" ni suenes robótico. 
2. Si la pregunta es sobre varias personas (ej. "dónde trabajaron"), DEBES detallar la información de CADA UNA por separado. NO asumas que el usuario busca únicamente los trabajos "en común" a menos que lo diga explícitamente.
3. Mantén toda la información valiosa aportada por cada agente sin resumirla en exceso.
"""

    feedback = state.get("reviewer_feedback", "")
    if feedback:
        prompt += f"\nATENCIÓN: Tu respuesta anterior recibió la siguiente crítica de un supervisor: '{feedback}'. Debes reescribir y mejorar tu respuesta tomando en cuenta esta crítica."

    response = llm.invoke(prompt)
    return {"final_response": response.content}

def reviewer_node(state: AgentState) -> dict:
    """
    Evalúa la respuesta generada por el combiner. Si es buena, el ciclo termina.
    Si falta información o la calidad es mala, provee feedback para que el combiner reintente.
    """
    if len(state["agent_responses"]) == 0:
        return {"revision_count": state.get("revision_count", 0) + 1} # No loop si no hay respuestas
        
    llm = ChatGroq(model=state.get("modelo_llm", "llama-3.1-8b-instant"), api_key=state["groq_api_key"])
    
    prompt = f"""Eres un revisor de calidad experto. Debes evaluar si la Respuesta Final contesta de forma directa, natural y completa a la Pregunta del Usuario.
    
Pregunta del Usuario: {state['query']}
Respuesta Final a evaluar: {state['final_response']}

Si la respuesta responde a la pregunta de forma adecuada y natural, devuelve "is_good": true y un feedback vacío.
Si la respuesta tiene alucinaciones, está incompleta, parece robótica (ej. "según la información provista..."), o ignora la pregunta original, devuelve "is_good": false y detalla la crítica en "feedback".

Responde ÚNICAMENTE con un JSON válido:
{{
  "is_good": true,
  "feedback": ""
}}
"""
    try:
        response = llm.invoke(prompt)
        content = response.content.replace("```json", "").replace("```", "").strip()
        result = json.loads(content)
        is_good = result.get("is_good", True)
        feedback = result.get("feedback", "")
    except Exception as e:
        print(f"Error parseando reviewer: {e}")
        is_good = True
        feedback = ""
        
    current_count = state.get("revision_count", 0)
    
    # Si es buena o ya alcanzamos 2 revisiones, terminamos el loop.
    if is_good or current_count >= 2:
        return {"revision_count": current_count + 1, "reviewer_feedback": ""}
        
    return {"revision_count": current_count + 1, "reviewer_feedback": feedback}

def should_continue(state: AgentState) -> Literal["combiner", "end"]:
    """
    Lógica de la arista condicional.
    Si reviewer_feedback tiene texto, significa que falló la revisión y debemos volver al combiner.
    """
    if state.get("reviewer_feedback"):
        return "combiner"
    return "end"

# Grafo de LangGraph
workflow = StateGraph(AgentState)

# Agregar Nodos
workflow.add_node("supervisor", supervisor_node)
workflow.add_node("person_agents", person_agents_node)
workflow.add_node("combiner", combiner_node)
workflow.add_node("reviewer", reviewer_node)

# Agregar Edges
workflow.add_edge(START, "supervisor")
workflow.add_edge("supervisor", "person_agents")
workflow.add_edge("person_agents", "combiner")
workflow.add_edge("combiner", "reviewer")
workflow.add_conditional_edges("reviewer", should_continue, {"combiner": "combiner", "end": END})

# Compilar
app_graph = workflow.compile()
