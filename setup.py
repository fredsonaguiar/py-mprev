from setuptools import setup

setup(
    name='pymprev',
    version='0.0',
    description='Python Preventive Maitenance Document Explorer',
    # author='',
    # author_email='',
    packages=['pymprev'],
    install_requires=[
        "tqdm",
        "tenacity",
        "networkx",
        "numpy",
        # "graspologic",
        "graspologic-native",
        # "kuzu",
        # "chromadb",
        "pymupdf",
        # "llama-index",
        # "llama-index-llms-ollama",
        # "llama-index-embeddings-ollama",
        # "llama-index-graph-stores-kuzu",
        # "llama-index-vector-stores-chroma",
        # "install llama-index-llms-gemini",
        # "llama-index-embeddings-gemini",
        "langchain",
        "langchain-neo4j",
        "langchain-google-genai",
        "langchain-text-splitters",
        "langchain-huggingface",
        "sentence-transformers"
    ],
)
