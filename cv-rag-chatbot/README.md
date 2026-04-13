# RAG Chatbot sobre CV

Chatbot conversacional basado en **Retrieval-Augmented Generation (RAG)** que permite
hacer preguntas sobre el contenido de un Curriculum Vitae (.docx).

## Arquitectura

```
Pregunta del usuario
        |
        v
+-------------------+     +-------------------+     +-------------------+
|  SentenceTransf.  | --> |     Pinecone      | --> |    Groq LLM       |
|  all-MiniLM-L6-v2 |     | Similitud coseno  |     | Llama 3.1 / 3.3   |
|  (embedding 384d) |     |   Top-K chunks    |     | (genera respuesta)|
+-------------------+     +-------------------+     +-------------------+
                                                              |
                                                              v
                                                        Respuesta
```

### Flujo completo

1. **Ingestion**: Lee el CV en formato `.docx` y extrae el texto.
2. **Chunking semantico**: Un LLM (Groq) divide el texto en secciones
   semanticas coherentes (experiencia, educacion, skills, etc.). Si falla,
   se usa un chunking naive por caracteres como fallback.
3. **Embeddings**: Genera vectores de 384 dimensiones con `all-MiniLM-L6-v2`.
4. **Almacenamiento**: Sube los vectores a un indice de Pinecone (metrica coseno).
5. **Query**: Ante una pregunta, genera el embedding, busca los Top-K chunks
   mas similares en Pinecone, y construye un prompt con el contexto recuperado.
6. **Generacion**: Groq genera la respuesta usando el contexto + historial
   de conversacion (memoria de los ultimos 5 intercambios).

## Requisitos previos

- Python 3.10+
- Cuenta gratuita en [Pinecone](https://www.pinecone.io/) (plan Starter)
- Cuenta gratuita en [Groq](https://console.groq.com/) (API key)

## Instalacion

```bash
# Clonar o navegar al directorio del proyecto
cd ClaseVI/rag-cv-chatbot

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
2. Indicar la ruta al archivo `.docx` del CV (hay uno de ejemplo en `data/`).
3. Hacer click en **"Procesar CV y subir a Pinecone"**.
4. Escribir preguntas en el chat.

## Estructura del proyecto

```
rag-cv-chatbot/
├── README.md               # Este archivo
├── requirements.txt        # Dependencias Python
├── .env.example            # Plantilla de variables de entorno
├── .gitignore              # Archivos excluidos del repositorio
├── src/
│   └── app.py              # Aplicacion principal (Streamlit)
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
| Lectura de CVs     | python-docx                       |
| Embeddings         | sentence-transformers (MiniLM)    |
| Base vectorial     | Pinecone (Serverless, AWS)        |
| LLM                | Groq (Llama 3.1 / 3.3 / Gemma2)  |
| Chunking           | LLM-based semantico + naive fallback |

## Autor

Tomas Corteggiano 
