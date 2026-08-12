import os
import tqdm
import pymupdf

from tenacity import retry, stop_after_attempt, wait_exponential

from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_neo4j.graph_transformers.llm import LLMGraphTransformer

# querying
from langchain_neo4j import GraphCypherQAChain


class GraphRAG:
    def __init__(self, llm, embedder, graph, ontology):
        self.llm = llm
        self.embedder = embedder
        self.graph = graph
        self.ontology = ontology

        # splits text into chunks
        self.splitter = RecursiveCharacterTextSplitter(
            # chunk_size=1000,
            # chunk_overlap=200,
            # separators=["\n\n", "\n", " ", ""]
        )

        # recovers structured data
        self.graph_transformer = LLMGraphTransformer(
            llm=self.llm,
            allowed_nodes = self.ontology.nodes,
            allowed_relationships = self.ontology.relations,
            node_properties = self.ontology.node_properties,
            relationship_properties = self.ontology.relationship_properties,
            additional_instructions = "\n\n".join([
                "Follow STRICTLY the Ontology:" + self.ontology.description,
                "Answers should be always given in english."
                ])
        )


    def _get_filenames(self, path:str):
        # we assume only valid files in the directory
        if os.path.isfile(path):
            return [path]

        if os.path.isdir(path):
            filenames = []
            for subpath in os.listdir(path):
                subpath = os.path.join(path, subpath)
                filenames += self._get_filenames(subpath)

            return filenames

        # raise Exception("Invalid path type found.")


    @retry(stop=stop_after_attempt(10), wait=wait_exponential(multiplier=2))
    def _transform_chunk(self, chunk:str):
        document = Document(page_content=chunk)
        return self.graph_transformer.convert_to_graph_documents([document])


    def _add_source_file_node(self, filename: str, metadata:dict = {}):
        query = """
        MERGE (f:Source {filename: $filename})
        SET f += $metadata
        """
        params = {"filename": filename, "metadata":metadata}
        self.graph.query(query, params=params)


    def _add_source_chunk_node(self, chunk_text:str, chunk_embedding, chunk_index:int, source_filename:str):
        query = """
        MERGE (c:Chunk {text: $text, embedding: $embedding, index: $index, source: $filename})
        """
        params = {
            "text": chunk_text,
            "embedding": chunk_embedding,
            "index": chunk_index,
            "filename": source_filename}
        self.graph.query(query, params=params)


    def _add_mentions_relations(self, node_ids, chunk_index:int, source_filename:str):
        query = """
        MATCH (c:Chunk {source: $filename, index: $index})
        UNWIND $node_ids AS node_id
        MATCH (n) WHERE n.id = node_id
        MERGE (c)-[:MENTIONS]->(n)
        """
        params = {"node_ids": node_ids, "index": chunk_index, "filename": source_filename}
        self.graph.query(query, params=params)


    def _add_next_chunk_relations(self):
        query = """
        MATCH (c1:Chunk)
        MATCH (c2:Chunk {source: c1.source, index: c1.index + 1})
        MERGE (c1)-[:NEXT_CHUNK]->(c2)
        """
        self.graph.query(query)


    def _add_source_chunk_relations(self):
        query = """
        MATCH (c:Chunk)
        MATCH (f:Source {filename: c.source})
        MERGE (f)-[:HAS_CHUNK]->(c)
        """
        self.graph.query(query)


    def ingest_documents(self, input_path:str, asyncronous=False):
        filenames = self._get_filenames(input_path)

        # ingest one document at a time
        for file_index, filename in enumerate(filenames, start=1):
            print(f"Processing file ({file_index}/{len(filenames)}):",  filename)

            doc = pymupdf.open(filename) # doc pages
            full_text = "\n".join([page.get_text() for page in doc])

            # add source file to graph
            self._add_source_file_node(filename, doc.metadata)

            # splits document in chunks
            chunks = self.splitter.split_text(full_text)
            for chunk_index, chunk in enumerate(tqdm.tqdm(chunks)):
                # add shource chunk to graph
                embedding = self.embedder.embed_query(chunk)
                self._add_source_chunk_node(
                    chunk_text=chunk,
                    chunk_index=chunk_index,
                    chunk_embedding=embedding,
                    source_filename=filename,)

                # get structured graph from chunk
                chunk_graph = self._transform_chunk(chunk)
                # Attach chunk properties to nodes + relationships
                for element in chunk_graph[0].nodes + chunk_graph[0].relationships:
                    element.properties["chunk_index"] = chunk_index
                    element.properties["chunk_source"] = filename

                self.graph.add_graph_documents(chunk_graph)

                # crates chunk mentioning relation to nodes
                node_ids = [node.id for node in chunk_graph[0].nodes]
                self._add_mentions_relations(node_ids, chunk_index, filename)

        # add global "NEXT_CHUNK" and "HAS_CHUNK" relations 
        self._add_next_chunk_relations() 
        self._add_source_chunk_relations()


    def run_leiden_clustering(self):
        pass


    def reset_databasis(self, graph):
        graph.query("MATCH (n) DETACH DELETE n")


    def remove_document(self, filename: str):
        # removes relations based on file
        query_remove_relations = """
        MATCH ()-[r {chunk_source: $filename}]->() DELETE r """
            
        # removes nodes created based on file
        query_remove_nodes = """
        MATCH (n)
        WHERE n.chunk_source = $filename AND NOT n:Chunk AND NOT n:Source
        DETACH DELETE n """
            
        # removes chunks and connections
        query_remove_chunks = """
        MATCH (c:Chunk {source: $filename}) DETACH DELETE c """

        # removes source node
        query_remove_source = """
        MATCH (f:Source {filename: $filename}) DETACH DELETE f """

        params = {"filename": filename}
        self.graph.query(query_remove_relations, params=params)
        self.graph.query(query_remove_nodes, params=params)
        self.graph.query(query_remove_chunks, params=params)
        self.graph.query(query_remove_source, params=params)


    def custom_query_RAG(self, text_query:str):
        # # chain used for queries
        # self.chain = GraphCypherQAChain.from_llm(
        #     llm=self.llm,
        #     )
        pass


    def custom_query_Graph(self, text_query:str):
        # # chain used for queries
        # self.chain = GraphCypherQAChain.from_llm(
        #     llm=self.llm,
        #     )
        pass


    def generate_reports(self):
        #   reports clarifying clusters of concepts: ML for Corrosion, etc
        #   also report how the documents are related
        #   answers to relevant questions already in documents
        pass