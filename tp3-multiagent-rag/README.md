# RAG Chatbot sobre CVs - TP3 Multi-Agente
 
Chatbot conversacional basado en **Retrieval-Augmented Generation (RAG)** y **Sistemas Multi-Agente** que permite hacer preguntas sobre el contenido de uno o más Currículums Vitae (.docx).

Este proyecto implementa los requerimientos del **Trabajo Práctico 3 (TP3)**, utilizando LangGraph para orquestar agentes dinámicos.

## Arquitectura Multi-Agente (LangGraph)
 
```
        Pregunta del usuario
                |
                v
      +-------------------+ 
      | Supervisor Node   | (Enruta dinámicamente según la consulta y contexto histórico)
      +-------------------+
                |
                v
      +-------------------+ 
      |   Person Agents   | ---> Busca chunks en Pinecone filtrando por perfil exacto
      |                   | ---> Genera RAG asumiendo el rol del agente (Groq)
      +-------------------+
                |
                v
      +-------------------+
      | Combiner Node     | (Fusiona respuestas y aplica formato)
      +-------------------+
                |
                v
      +-------------------+
      | Reviewer Node     | ---> Evalúa calidad y falta de alucinaciones
      +-------------------+
             |      |
         [Fail]    [Pass]
           |          |
      (Loop)          v
           |      Respuesta
           +-------+
 
### Flujo completo
 
1. **Ingestión masiva**: Lee todos los CVs en formato `.docx` y `.pdf` ubicados en el directorio `data/`.
2. **Extracción de Identidad**: Un LLM lee los primeros caracteres del archivo y deduce automáticamente el nombre y apellido del dueño del CV, independientemente de cómo se llame el archivo.
3. **Chunking Estructural**: Se utiliza `RecursiveCharacterTextSplitter` para dividir el texto de manera determinística, lógica y sin costo de tokens, previniendo errores de límite de API en documentos muy grandes.
4. **Embeddings & Metadatos**: Genera vectores con `all-MiniLM-L6-v2` y les asigna el campo `person_name` deducido por el LLM en Pinecone para aislar el contexto.
5. **Orquestación LangGraph con Reflexión**:
   - Analiza a quién se está preguntando, resolviendo pronombres de la conversación (ej: "su correo").
   - Distribuye la consulta a un agente específico en primera persona.
   - Combina los resultados y ejecuta un **Reviewer Node** para auto-corregir respuestas robóticas o incompletas antes de devolver la respuesta final al usuario.

## Requisitos previos

- Python 3.10+
- Cuenta gratuita en [Pinecone](https://www.pinecone.io/) (plan Starter)
- Cuenta gratuita en [Groq](https://console.groq.com/) (API key)

## Instalacion

```bash
# Clonar o navegar al directorio del proyecto
cd tp3-multiagent-rag

# Crear entorno virtual
python3 -m venv venv
source venv/bin/activate

# Instalar dependencias
pip install -r requirements.txt
```

## Configuracion

Exportar las API keys como variables de entorno:

```bash
export PINECONE_API_KEY="tu-pinecone-api-key"
export GROQ_API_KEY="tu-groq-api-key"
```

Alternativamente, se pueden crear en un archivo `.env` (ver `.env.example`).

## Ejecucion

```bash
streamlit run src/app.py
```

La aplicacion se abre en `http://localhost:8501`.

### Uso
 
1. Completar las API keys en la barra lateral (o usar variables de entorno).
2. Indicar el directorio que contiene los `.docx` de los CVs (por defecto `data/`).
3. Hacer click en **"Procesar CV y subir a Pinecone"**.
4. Escribir preguntas en el chat.

## Estructura del proyecto

rag-multiagent-cv/
├── README.md               # Este archivo
├── requirements.txt        # Dependencias Python
├── .env.example            # Plantilla de variables de entorno
├── .gitignore              # Archivos excluidos del repositorio
├── src/
│   ├── app.py              # Aplicacion principal (Streamlit)
│   └── agent_graph.py      # Lógica de agentes multi-perfil (LangGraph)
├── data/
│   └── *.docx              # CVs de ejemplo
└── docs/
    ├── arquitectura.md     # Documentacion tecnica
    └── *.excalidraw        # Diagramas editables
```

## Tecnologias

| Componente         | Tecnologia                        |
|--------------------|-----------------------------------|
| Interfaz           | Streamlit                         |
| Orquestación       | LangGraph                         |
| Lectura de CVs     | python-docx, pypdf                |
| Embeddings         | sentence-transformers (MiniLM)    |
| Base vectorial     | Pinecone (Serverless, AWS)        |
| LLM                | Groq (Llama 3.1 / 3.3 / Gemma2)  |
| Chunking           | RecursiveCharacterTextSplitter |

## Autor

Tomas Corteggiano 
