from setuptools import setup, find_packages

setup(
    name='pymprev',
    version='0.0',
    description='Python Preventive Maitenance Document Explorer',
    # author='',
    # author_email='',
    packages=find_packages(),
    install_requires=[
        "kuzu",
        "chromadb",
        "pymupdf",
        "llama-index",
        "llama-index-llms-ollama",
        "llama-index-embeddings-ollama",
        "llama-index-graph-stores-kuzu",
        "llama-index-vector-stores-chroma"
    ],
)