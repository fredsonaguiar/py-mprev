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
        "numpy",
        "tenacity",
        "pymupdf",
        "langchain",
        "langchain-neo4j",
        "langchain-google-genai",
        "langchain-text-splitters",
        "langchain-huggingface",
        "sentence-transformers",
        #
        "networkx",
        # "graspologic",
        "graspologic-native",
        #
        "streamlit",
    ],
)
