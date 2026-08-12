# py-mprev: Python Preventive Maitenance Document Explorer

This project focuses on the discussion of applicability, requirements, and employment of ML-Based techniques for preventive maintenance in the context of civil aviation, with a focus on Preventive and Health Management (PHM) tools. This is justified by the potential impact of Artificial Intelligence and Machine Learning in this sector, which requires strict regulations, currently not compatible with the most recent technologies.


# GraphRAG

GraphRAG (Graph-enhanced Retrieval-Augmented Generation) is a a methodology that extends traditional (vector) RAG (Retrieval-Augmented Generation) by including the use of a Knowledge Base (KB) to generate verifiable, structured answers to queries. 

## Architecture

The keyords in this proposed are: Lightweight; Verifiable; Scalable. Our initial arquitecture is develloped to run locally or in server, which allows scallability but also avoids exposition of potentially sensible documents, if necessary:

 * **[LangChain](https://www.langchain.com/):** Pipeline orchestration, deterministic document chunking (`RecursiveCharacterTextSplitter`), and prompt management.
* **[Neo4j](https://neo4j.com/):** Unified graph and vector database. Stores document hierarchies (`:Source` -> `:Chunk`), sequential relationships (`:NEXT_CHUNK`), domain entities, dense vector embeddings, and full-text indexes in a single engine.
<!-- * **[Ollama](https://ollama.com/):** Local LLM inference engine providing offline hardware acceleration for entity extraction and query synthesis without data leakage. -->
<!-- * **[Sentence-Transformers](https://www.sbert.net/):** Generates dense semantic vector embeddings for chunk texts and entity resolution (e.g., `all-MiniLM-L6-v2`). -->
<!-- * **[APOC (Awesome Procedures on Cypher)](https://neo4j.com/labs/apoc/):** Graph refactoring procedures used for automated entity deduplication and synonym node merging. -->
* **[PyMuPDF (fitz)](https://pymupdf.readthedocs.io/):** Fast, offline PDF parser for extracting clean text and structural metadata from complex technical documents.

## Instalation

For simple experiments using cloud-based services. Installation in a virtual environment is strongly advisable. Include the alias `--editable` for development mode:

```
python -m venv path/to/venv-mprev/
source path/to/venv-mprev/bin/activate
pip install --editable path/to/py-mprev/
```

where `path/to/venv-mprev/` is to be replace by the path one intends to create the virtual environment, and `path/to/py-mprev/` is to be replaced by the where the library is downloaded.