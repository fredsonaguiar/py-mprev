# LLM
from langchain_core.language_models.llms import BaseLLM
from langchain_google_genai import ChatGoogleGenerativeAI

# Embeddings
from langchain_core.embeddings import Embeddings
# from langchain_google_genai import GoogleGenerativeAIEmbeddings
from huggingface_hub import login as hf_login
from langchain_huggingface import HuggingFaceEmbeddings

# Graph and Vector store
from langchain_neo4j.graphs.graph_store import GraphStore
from langchain_neo4j import Neo4jGraph


def connect_llm(api_key:str, model_name:str) -> BaseLLM:
    llm = ChatGoogleGenerativeAI(model=model_name, api_key=api_key)
     
    return llm


def connect_embeder(api_key:str, model_name:str) -> Embeddings:
    # embeder = GoogleGenerativeAIEmbeddings(model=model_name, api_key=api_key)
    hf_login(api_key)
    embeder = HuggingFaceEmbeddings(model=model_name)

    return embeder


def connect_database(uri, database, username, password) -> GraphStore:
    graph = Neo4jGraph(url=uri, username=username, password=password, database=database)
    
    return graph
