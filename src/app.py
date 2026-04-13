"""
RAG Chatbot sobre CV - Ejercicio Clase VI
==========================================

Flujo completo:
1. Carga un CV en formato .docx
2. Divide el texto en chunks semanticos usando un LLM (Groq)
3. Genera embeddings con sentence-transformers (all-MiniLM-L6-v2)
4. Sube los vectores a Pinecone (metrica coseno)
5. Ante una pregunta, busca los chunks mas relevantes por similitud coseno
6. Genera una respuesta con Groq (Llama 3) usando el contexto recuperado

Ejecucion:
    export PINECONE_API_KEY="tu-key"
    export GROQ_API_KEY="tu-key"
    streamlit run rag_cv_chatbot.py

Autor: Tomas Corteggiano - Clase VI - CEIA LLMIAG
"""

import os
import json
import time
import numpy as np
import streamlit as st
from typing import List, Dict
from docx import Document as DocxDocument
from sentence_transformers import SentenceTransformer
from pinecone import Pinecone, ServerlessSpec
from groq import Groq


# =============================================================================
# 1. LECTURA Y CHUNKING DEL CV
# =============================================================================

def leer_cv_docx(path: str) -> str:
    """Lee un archivo .docx y retorna todo su texto concatenado."""
    doc = DocxDocument(path)
    parrafos = [p.text.strip() for p in doc.paragraphs if p.text.strip()]
    texto_completo = "\n".join(parrafos)
    return texto_completo


def chunking_semantico_con_llm(texto: str, cliente_groq: Groq, modelo_llm: str) -> List[Dict]:
    """
    Usa un LLM (Groq con JSON mode) para dividir el CV en chunks semánticos.
    
    A diferencia del chunking naive (por cantidad de caracteres), este método produce 
    chunks que respetan los límites lógicos y semánticos del documento.
    
    Args:
        texto: Texto completo del CV.
        cliente_groq: Cliente instanciado de Groq.
        modelo_llm: Nombre del modelo a utilizar (recomendado llama-3.3-70b-versatile para mejor reasoning).

    Returns:
        List[Dict]: Lista estructurada de chunks.
    """
    prompt = f"""Eres un experto en procesamiento de texto para sistemas RAG. 
Tu tarea es dividir el siguiente Currículum Vitae en fragmentos (chunks) semánticos óptimos para indexación vectorial.

REGLAS DE CHUNKING:
1. Unidad temática: Cada chunk debe representar un concepto o bloque unitario (ej. un rol laboral específico, la sección completa de educación, un conjunto afín de habilidades).
2. Autosuficiencia de contexto: Cada chunk debe entenderse por sí mismo. Menciona explícitamente el rol, la empresa o la institución dentro del texto del chunk si es necesario para dar contexto.
3. Fidelidad: NO inventes ni resumas excesivamente. Mantén la mayor cantidad de detalles del texto original.
4. Tipo de salida: Tu respuesta DEBE ser un objeto JSON válido con la clave "chunks" que contenga una lista de objetos, cada uno con "seccion" y "texto".

Estructura JSON esperada:
{{
  "chunks": [
    {{"seccion": "experiencia_empresaX", "texto": "contenido descriptivo de la experiencia"}},
    {{"seccion": "educacion", "texto": "contenido sobre estudios"}}
  ]
}}

CV A PROCESAR:
{texto}
"""

    respuesta = cliente_groq.chat.completions.create(
        model=modelo_llm,
        messages=[
            {"role": "system", "content": "You are an assistant designed to output strictly valid JSON."},
            {"role": "user", "content": prompt},
        ],
        temperature=0.1,
        max_tokens=4096,
        response_format={"type": "json_object"}
    )

    contenido = respuesta.choices[0].message.content.strip()

    # Como usamos response_format={"type": "json_object"}, el modelo nos garantiza JSON válido
    resultado = json.loads(contenido)
    secciones = resultado.get("chunks", [])

    chunks = []
    for idx, sec in enumerate(secciones):
        if not isinstance(sec, dict) or "texto" not in sec:
            continue
        chunks.append({
            "id": f"cv_chunk_{idx:03d}",
            "texto": sec["texto"].strip(),
            "seccion": sec.get("seccion", "general"),
            "inicio": 0,
            "fin": len(sec["texto"]),
        })

    if not chunks:
        raise ValueError("El LLM no generó ningún chunk válido en el JSON.")

    return chunks


def dividir_en_chunks_naive(texto: str, tamano: int = 300, overlap: int = 50) -> List[Dict]:
    """
    Fallback: divide el texto por cantidad de caracteres con overlap.
    Se usa si el chunking semantico con LLM falla.
    """
    chunks = []
    inicio = 0
    idx = 0

    while inicio < len(texto):
        fin = inicio + tamano
        fragmento = texto[inicio:fin]

        if fin < len(texto):
            ultimo_espacio = fragmento.rfind(" ")
            if ultimo_espacio > tamano // 2:
                fin = inicio + ultimo_espacio
                fragmento = texto[inicio:fin]

        chunks.append({
            "id": f"cv_chunk_{idx:03d}",
            "texto": fragmento.strip(),
            "seccion": "naive",
            "inicio": inicio,
            "fin": fin,
        })

        inicio = fin - overlap
        idx += 1

    return chunks


# =============================================================================
# 2. GENERACION DE EMBEDDINGS
# =============================================================================

@st.cache_resource
def cargar_modelo_embeddings():
    """Carga el modelo de embeddings (se cachea para no recargar)."""
    modelo = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
    return modelo


def generar_embeddings(modelo: SentenceTransformer, textos: List[str]) -> List[List[float]]:
    """Genera embeddings para una lista de textos."""
    embeddings = modelo.encode(textos, show_progress_bar=False)
    return [emb.tolist() for emb in embeddings]


# =============================================================================
# 3. PINECONE - INDICE VECTORIAL
# =============================================================================

NOMBRE_INDICE = "cv-rag-ceia"
DIMENSION = 384  # dimension de all-MiniLM-L6-v2


def inicializar_pinecone(api_key: str) -> Pinecone:
    """Crea el cliente de Pinecone."""
    pc = Pinecone(api_key=api_key)
    return pc


def crear_o_conectar_indice(pc: Pinecone) -> object:
    """
    Crea el indice si no existe, o se conecta al existente.
    Usa ServerlessSpec (compatible con plan Starter gratuito).
    """
    indices_existentes = [idx.name for idx in pc.list_indexes()]

    if NOMBRE_INDICE not in indices_existentes:
        pc.create_index(
            name=NOMBRE_INDICE,
            dimension=DIMENSION,
            metric="cosine",
            spec=ServerlessSpec(
                cloud="aws",
                region="us-east-1"
            )
        )
        while not pc.describe_index(NOMBRE_INDICE).status["ready"]:
            time.sleep(1)

    indice = pc.Index(NOMBRE_INDICE)
    return indice


def subir_chunks_a_pinecone(indice, chunks: List[Dict], embeddings: List[List[float]]):
    """
    Sube los vectores de los chunks al indice de Pinecone.
    Cada vector incluye el texto del chunk como metadata.
    """
    vectores = []
    for chunk, emb in zip(chunks, embeddings):
        vectores.append({
            "id": chunk["id"],
            "values": emb,
            "metadata": {
                "texto": chunk["texto"],
                "seccion": chunk.get("seccion", ""),
            }
        })

    # Upsert en lotes de 50
    for i in range(0, len(vectores), 50):
        lote = vectores[i:i + 50]
        indice.upsert(vectors=lote)


def buscar_chunks_similares(
    indice,
    query_embedding: List[float],
    top_k: int = 3
) -> List[Dict]:
    """
    Busca los chunks mas similares a la query usando similitud coseno.

    Args:
        indice: Indice de Pinecone
        query_embedding: Vector de la pregunta
        top_k: Numero de resultados a devolver

    Returns:
        Lista de chunks con score de similitud
    """
    resultados = indice.query(
        vector=query_embedding,
        top_k=top_k,
        include_metadata=True
    )

    chunks_encontrados = []
    for match in resultados["matches"]:
        chunks_encontrados.append({
            "id": match["id"],
            "score": round(match["score"], 4),
            "texto": match["metadata"]["texto"],
        })

    return chunks_encontrados


# =============================================================================
# 4. GROQ - GENERACION DE RESPUESTAS (RAG)
# =============================================================================

def generar_respuesta_rag(
    cliente_groq: Groq,
    modelo_llm: str,
    pregunta: str,
    contexto_chunks: List[Dict],
    historial: List[Dict]
) -> str:
    """
    Genera una respuesta usando Groq con el contexto de los chunks recuperados.

    Args:
        cliente_groq: Cliente de Groq
        modelo_llm: Nombre del modelo (ej: llama-3.1-8b-instant)
        pregunta: Pregunta del usuario
        contexto_chunks: Chunks relevantes recuperados de Pinecone
        historial: Historial de conversacion previo

    Returns:
        Respuesta generada por el LLM
    """
    contexto_texto = "\n---\n".join([
        f"[Fragmento {i+1} | Similitud: {c['score']}]\n{c['texto']}"
        for i, c in enumerate(contexto_chunks)
    ])

    system_prompt = f"""Eres un asistente conversacional con memoria de los mensajes anteriores.
Tenes acceso al contenido de un CV (curriculum vitae) como fuente de informacion.

REGLAS:
- Responde de forma directa y natural, como si conocieras a la persona.
- NUNCA digas "segun el CV", "segun el contexto", "en el documento" ni frases similares.
  Simplemente da la respuesta directamente.
- Si la pregunta se puede responder con el contenido del CV, usa esa informacion.
- Si la informacion NO esta en el CV, responde con tu conocimiento general, pero aclara
  que no tenes esa informacion especifica sobre la persona.
- Responde en el mismo idioma en que se hace la pregunta.
- Se conciso pero informativo.
- Tenes memoria de la conversacion: podes referenciar preguntas y respuestas anteriores.

CONTEXTO DEL CV:
{contexto_texto}"""

    messages = [{"role": "system", "content": system_prompt}]

    if historial:
        for msg in historial[-5:]:
            messages.append({"role": "user", "content": msg["pregunta"]})
            messages.append({"role": "assistant", "content": msg["respuesta"]})

    messages.append({"role": "user", "content": pregunta})

    respuesta = cliente_groq.chat.completions.create(
        model=modelo_llm,
        messages=messages,
        temperature=0.3,
        max_tokens=1024,
    )

    return respuesta.choices[0].message.content


# =============================================================================
# 6. INTERFAZ STREAMLIT
# =============================================================================

def main():
    st.set_page_config(
        page_title="RAG Chatbot - CV",
        page_icon="",
        layout="wide",
    )

    st.title("RAG Chatbot - Preguntas sobre CV")
    st.caption("Clase VI - CEIA LLMIAG | Pinecone + Sentence-Transformers + Groq")

    # -- Sidebar: Configuracion --
    with st.sidebar:
        st.header("Configuracion")
        st.markdown("---")

        pinecone_key = st.text_input(
            "Pinecone API Key",
            value=os.getenv("PINECONE_API_KEY", os.getenv("PINECONE_TOKEN", "")),
            type="password",
        )
        groq_key = st.text_input(
            "Groq API Key",
            value=os.getenv("GROQ_API_KEY", ""),
            type="password",
        )

        st.markdown("---")

        st.subheader("Modelo LLM")
        modelo_llm = st.selectbox(
            "Modelo Groq:",
            [
                "llama-3.1-8b-instant",
                "llama-3.3-70b-versatile",
                "gemma2-9b-it",
                "mixtral-8x7b-32768",
            ],
            index=0,
        )

        st.subheader("Cargar CV")
        cv_path = st.text_input(
            "Ruta al archivo .docx",
            value="",
        )

        st.subheader("Chunking (fallback naive)")
        tamano_chunk = st.slider("Tamano del chunk (caracteres)", 100, 800, 300)
        overlap_chunk = st.slider("Overlap (caracteres)", 0, 150, 50)

        st.subheader("Busqueda")
        top_k = st.slider("Top-K resultados", 1, 10, 3)

        st.markdown("---")

        procesar = st.button("Procesar CV y subir a Pinecone", type="primary", use_container_width=True)

        if st.button("Limpiar conversacion", use_container_width=True):
            st.session_state.historial_chat = []
            st.session_state.messages = []
            st.rerun()

    # -- Estado de sesion --
    if "chunks" not in st.session_state:
        st.session_state.chunks = []
    if "indice_listo" not in st.session_state:
        st.session_state.indice_listo = False
    if "historial_chat" not in st.session_state:
        st.session_state.historial_chat = []
    if "messages" not in st.session_state:
        st.session_state.messages = []
    if "cv_texto" not in st.session_state:
        st.session_state.cv_texto = ""

    # -- Proceso de carga del CV --
    if procesar:
        if not pinecone_key or not groq_key:
            st.error("Se deben configurar ambas API keys en la barra lateral.")
            st.stop()

        if not os.path.exists(cv_path):
            st.error(f"No se encontro el archivo: {cv_path}")
            st.stop()

        with st.status("Procesando CV...", expanded=True) as status:

            # Paso 1: Leer CV
            st.write("Paso 1: Leyendo CV...")
            texto_cv = leer_cv_docx(cv_path)
            st.session_state.cv_texto = texto_cv
            st.write(f"  Leidos {len(texto_cv)} caracteres.")

            # Crear cliente Groq
            cliente_groq = Groq(api_key=groq_key)

            # Paso 2: Chunking semantico con LLM
            st.write("Paso 2: Chunking semantico con LLM (Groq)...")
            try:
                chunks = chunking_semantico_con_llm(texto_cv, cliente_groq, modelo_llm)
                st.write(f"  {len(chunks)} chunks semanticos generados.")
                for c in chunks:
                    st.write(f"  - [{c['seccion']}] {len(c['texto'])} caracteres")
            except Exception as e:
                st.warning(f"Chunking semantico fallo ({e}). Usando chunking naive como fallback.")
                chunks = dividir_en_chunks_naive(texto_cv, tamano=tamano_chunk, overlap=overlap_chunk)
                st.write(f"  {len(chunks)} chunks naive generados (fallback).")

            st.session_state.chunks = chunks

            # Paso 3: Embeddings
            st.write("Paso 3: Generando embeddings con all-MiniLM-L6-v2...")
            modelo_emb = cargar_modelo_embeddings()
            textos_chunks = [c["texto"] for c in chunks]
            embeddings = generar_embeddings(modelo_emb, textos_chunks)
            st.write(f"  {len(embeddings)} embeddings generados (dim={len(embeddings[0])}).")

            # Paso 5: Subir a Pinecone
            st.write("Paso 5: Subiendo vectores a Pinecone...")
            pc = inicializar_pinecone(pinecone_key)
            indice = crear_o_conectar_indice(pc)
            subir_chunks_a_pinecone(indice, chunks, embeddings)
            stats = indice.describe_index_stats()
            st.write(f"  Indice '{NOMBRE_INDICE}' con {stats['total_vector_count']} vectores.")

            # Paso 6: Groq listo
            st.write(f"Paso 6: Groq configurado (modelo: {modelo_llm}).")

            st.session_state.indice_listo = True
            st.session_state.pinecone_key = pinecone_key
            st.session_state.groq_key = groq_key
            st.session_state.modelo_llm = modelo_llm

            status.update(label="CV procesado y cargado en Pinecone", state="complete")

    # -- Mostrar chunks --
    if st.session_state.chunks:
        with st.expander(f"Ver chunks del CV ({len(st.session_state.chunks)} fragmentos)", expanded=False):
            for c in st.session_state.chunks:
                seccion = c.get("seccion", "")
                st.markdown(f"**{c['id']}** [{seccion}]")
                st.text(c["texto"])
                st.markdown("---")

    # -- Chat --
    if st.session_state.indice_listo:
        st.markdown("---")
        st.subheader("Pregunta sobre el CV")

        for msg in st.session_state.messages:
            with st.chat_message(msg["role"]):
                st.markdown(msg["content"])
                if "chunks" in msg:
                    with st.expander("Chunks recuperados"):
                        for c in msg["chunks"]:
                            st.markdown(f"**{c['id']}** (coseno: {c['score']})")
                            st.text(c["texto"])

        if pregunta := st.chat_input("Ej: Donde estudio? Que tecnologias maneja?"):
            st.session_state.messages.append({"role": "user", "content": pregunta})
            with st.chat_message("user"):
                st.markdown(pregunta)

            with st.chat_message("assistant"):
                with st.spinner("Buscando en el CV y generando respuesta..."):
                    cliente_groq = Groq(api_key=st.session_state.groq_key)
                    pc = inicializar_pinecone(st.session_state.pinecone_key)
                    indice = pc.Index(NOMBRE_INDICE)

                    modelo_emb = cargar_modelo_embeddings()
                    emb_pregunta = generar_embeddings(modelo_emb, [pregunta])[0]

                    chunks_relevantes = buscar_chunks_similares(indice, emb_pregunta, top_k=top_k)

                    respuesta = generar_respuesta_rag(
                        cliente_groq,
                        st.session_state.get("modelo_llm", modelo_llm),
                        pregunta,
                        chunks_relevantes,
                        st.session_state.historial_chat,
                    )

                    st.markdown(respuesta)

                    with st.expander("Chunks recuperados"):
                        for c in chunks_relevantes:
                            st.markdown(f"**{c['id']}** (coseno: {c['score']})")
                            st.text(c["texto"])

                    st.session_state.historial_chat.append({
                        "pregunta": pregunta,
                        "respuesta": respuesta,
                    })
                    st.session_state.messages.append({
                        "role": "assistant",
                        "content": respuesta,
                        "chunks": chunks_relevantes,
                    })

    elif not st.session_state.indice_listo:
        st.info(
            "Configurar las API keys en la barra lateral y hacer click en "
            "'Procesar CV y subir a Pinecone' para comenzar."
        )

    # -- Footer --
    st.markdown("---")
    st.caption("Clase VI - CEIA LLMIAG | RAG: Pinecone + Sentence-Transformers + Groq")


if __name__ == "__main__":
    main()
