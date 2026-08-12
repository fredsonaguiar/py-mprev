import getpass

from connection import connect_llm, connect_embeder, connect_database

# configuration llm and embed
google_llm_model = "gemma-4-31b-it"
google_embeder_model = "gemini-embedding-2"
google_api_key = getpass.getpass("Enter your Google AI API key: ")

neo4j_uri="neo4j+s://312151bc.databases.neo4j.io"
neo4j_username="312151bc"
neo4j_password = getpass.getpass("Enter your Neo4j password: ")
neo4j_database="312151bc"

# connect services
llm = connect_llm(
    google_api_key, 
    google_llm_model)
embeder = connect_embeder(
    google_api_key, 
    google_embeder_model)
graph = connect_database(
    neo4j_uri,
    neo4j_database,
    neo4j_username,
    # sanitize=True,
    # enhanced_schema=True,
    neo4j_password)


# ingest documents
from util import Ingestor, remove_document
from ontology import Ontology

ontology = Ontology()
# reset_databasis(graph)

doc_ingestor = Ingestor(llm, embeder, graph, ontology)
doc_ingestor.ingest_documents("/home/fredson-aguiar/Downloads/resources/")
