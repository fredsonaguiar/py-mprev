# py-mprev: Python Preventive Maitenance Document Explorer

This project focuses on the discussion of applicability, requirements, and employment of ML-Based techniques for preventive maintenance in the context of civil aviation, with a focus on Preventive and Health Management (PHM) tools. This is justified by the potential impact of Artificial Intelligence and Machine Learning in this sector, which requires strict regulations, currently not compatible with the most recent technologies.


# GraphRAG

GraphRAG (Graph-enhanced Retrieval-Augmented Generation) is a a methodology that extends traditional (vector) RAG (Retrieval-Augmented Generation) by including the use of a Knowledge Base (KB) to generate verifiable, structured answers to queries. 

## Architecture

Lightweight, Verifiable and Scalable are keyords in the proposed project. Our initial arquitecture is develloped to run locally, which avoids exposing

 * LlamaIndex: Orchestration, manages chunking, routes semantic data to the vector store, and maps relationship data into the graph.

 * Ollama: Local LLM Engine, offline background daemon that executes models locally with hardware acceleration.

 * Kùzu DB: In-Process Graph, a serverless, embedded property graph database. It allows running high-speed Cypher queries to explicitly discover, trace, and dynamically update missing standard dependencies.

 * ChromaDB: A lightweight, embedded vector store that manages dense mathematical sentence embeddings for classic semantic text retrieval.

 * PyMuPDF (fitz): Document Parsing. It extracts raw text and critical structural metadata (like page numbers and document properties) from dense aerospace PDFs completely offline.

## Instalation

First, install ollama and depencencies:

```
curl -fsSL https://ollama.com/install.sh | sh
ollama pull llama3:8b
ollama pull mxbai-embed-large
```

For the python libraries, using a virtual environment is advised:

```
python -m venv path/to/env-mprev
source env-mprev/bin/activate
```

Then, install the required python libraries:

```
# install llama libraries
pip install llama-index
pip install llama-index-llms-ollama
pip install llama-index-embeddings-ollama
pip install llama-index-graph-stores-kuzu
pip install llama-index-vector-stores-chroma

# install other libraries
pip install kuzu
pip install chromadb
pip install pymupdf
```
