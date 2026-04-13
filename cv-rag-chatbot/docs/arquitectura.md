# Arquitectura del Sistema RAG

## Descripcion general

El sistema implementa un pipeline de **Retrieval-Augmented Generation (RAG)**
que permite consultar el contenido de un CV mediante lenguaje natural.
La arquitectura se divide en dos fases: una de **ingestion** (offline) y una
de **consulta** (online, en tiempo real).

## Fase de Ingestion

Esta fase se ejecuta una vez al cargar el CV. Transforma el documento en
vectores almacenados en Pinecone.

### 1. Lectura del documento

- Entrada: archivo `.docx`
- Libreria: `python-docx`
- Salida: texto plano concatenado (todos los parrafos no vacios)

### 2. Chunking semantico con LLM

A diferencia del chunking tradicional (cortar cada N caracteres), este sistema
usa el propio LLM para dividir el CV en **secciones semanticas coherentes**.

**Ventajas sobre chunking naive:**
- Cada chunk es una unidad tematica completa (un puesto de trabajo, formacion, etc.)
- No se cortan oraciones ni se pierde contexto entre chunks
- El LLM agrega metadata de seccion (experiencia, educacion, habilidades...)

**Pipeline de parsing:**
1. Se envia el texto completo al LLM con instrucciones de devolver JSON
2. Se intenta `json.loads()` sobre la respuesta
3. Si falla (JSON malformado), se intenta extraer con regex
4. Si tambien falla, se cae al chunking naive como ultimo fallback

### 3. Generacion de embeddings

- Modelo: `sentence-transformers/all-MiniLM-L6-v2`
- Dimension: 384
- Se cachea con `@st.cache_resource` para no recargar en cada interaccion

### 4. Almacenamiento en Pinecone

- Indice: `cv-rag-ceia`
- Metrica: coseno
- Spec: Serverless (AWS us-east-1, compatible con plan Starter gratuito)
- Metadata por vector: texto del chunk + nombre de seccion
- Upsert en lotes de 50 vectores

## Fase de Consulta (RAG)

Esta fase se ejecuta en cada pregunta del usuario.

### 1. Embedding de la pregunta

Se genera el embedding de la pregunta con el mismo modelo (all-MiniLM-L6-v2)
para asegurar que vive en el mismo espacio vectorial.

### 2. Busqueda por similitud coseno

Se envian los embeddings a Pinecone que retorna los Top-K chunks mas similares,
ordenados por score de similitud coseno descendente.

**Similitud coseno manual** (implementada para el ejercicio):

```
cos(theta) = (A . B) / (||A|| x ||B||)
```

Donde A y B son los vectores de embedding. Un score de 1.0 indica
vectores identicos, 0.0 indica ortogonalidad.

### 3. Construccion del prompt

Se construye un prompt estructurado con:
- **System prompt**: instrucciones de comportamiento del asistente
- **Contexto**: los Top-K chunks recuperados con sus scores
- **Historial**: los ultimos 5 intercambios (memoria conversacional)
- **Pregunta**: la pregunta actual del usuario

### 4. Generacion de respuesta

- Proveedor: Groq API
- Modelos disponibles: Llama 3.1 8B, Llama 3.3 70B, Gemma2 9B, Mixtral 8x7B
- Temperatura: 0.3 (respuestas mas deterministas)
- Max tokens: 1024

## Memoria conversacional

El sistema mantiene un historial de conversacion en `st.session_state`.
Los ultimos 5 intercambios (pregunta + respuesta) se incluyen en cada
llamada al LLM, permitiendo:
- Preguntas de follow-up ("y que mas hizo en ese puesto?")
- Referencias a respuestas anteriores
- Conversacion coherente multi-turno

## Diagrama de componentes

Ver `diagrama_rag_system.excalidraw` para el diagrama editable completo.

```
+------------------+
|   Streamlit UI   |  <-- Interfaz web (sidebar + chat)
+--------+---------+
         |
    +----v----+   +------------------+   +---------------+
    | python- |   | SentenceTransf.  |   |   Groq API    |
    |  docx   |   | (embeddings)     |   | (LLM chunking |
    +---------+   +--------+---------+   |  + respuestas)|
                           |             +-------+-------+
                  +--------v---------+           |
                  |     Pinecone     |           |
                  |  (vector store)  | <---------+
                  +------------------+
```
