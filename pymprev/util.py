import os
import tqdm
import pymupdf

from tenacity import retry, stop_after_attempt, wait_exponential

from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_neo4j.graph_transformers.llm import LLMGraphTransformer

# community detection
# import networkx
import graspologic_native

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


    @retry(stop=stop_after_attempt(10), wait=wait_exponential(multiplier=2, min=4, max=60))
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


    def _get_ontology_subgraph(self):
        # get ontology-based subgraph
        node_types = " | ".join(self.ontology.nodes)
        relation_types = " | ".join(self.ontology.relations_schema.keys())
            
        query = """
        MATCH (source: {})-[r: {}]->(target: {})
        RETURN
            source.id AS source_id,
            target.id AS target_id
        """.format(node_types, relation_types, node_types)
        
        result = self.graph.query(query)
        return result
    

    def _get_leiden_clusters(self, edges, max_cluster_size=10, resolution=0.1):
        clusters = graspologic_native.hierarchical_leiden(
             edges, max_cluster_size=max_cluster_size, resolution=resolution)
                
        # find depth and width
        level_cluster_map = dict()
        for row in clusters:
            if row.level not in level_cluster_map:
                level_cluster_map[row.level] = set()
            level_cluster_map[row.level].add(int(row.cluster))

        return clusters, level_cluster_map
        

    def _add_cluster_node_relations(self, clusters):
        query_create = """
        UNWIND $clusters AS row
        // match the node by id
        MATCH (e {id: row.node_id})
        // MERGE ONLY on the unique identifier
        MERGE (c:Cluster {id: row.cluster_id})
        // SET properties safely (handles null/None gracefully)
        SET c.level = row.level,
            c.parent = row.parent
        // create relationship
        MERGE (e)-[r:IN_CLUSTER]->(c)
        SET r.final = row.final
        """
        
        params = {
            "clusters":[{
                "node_id": row.node,
                "cluster_id": int(row.cluster),
                "level": int(row.level),
                "parent": row.parent_cluster,
                "final": bool(row.is_final_cluster)
            } for row in clusters]}

        self.graph.query(query_create, params=params)


    def _add_cluster_subcluster_relations(self):
        query = """
        // match the child first
        MATCH (c:Cluster)
        WHERE c.parent IS NOT NULL
        // look up the parent
        MATCH (p:Cluster {id: c.parent})
        // connect them
        MERGE (c)-[:IN_CLUSTER]->(p)
        """

        self.graph.query(query)


    @retry(stop=stop_after_attempt(10), wait=wait_exponential(multiplier=2, min=4, max=60))
    def _get_cluster_summary(self, prompt):
        json_schema = {
            "title": "cluster_summary",
            "description": "Cluster name and summary",
            "type": "object",
            "properties": {
                "name": {
                    "type": "string",
                    "description": "Short, descriptive title for the cluster (3-6 words)"
                },
                "summary": {
                    "type": "string",
                    "description": "Comprehensive synthesis of the cluster contents"
                }
            },
            "required": ["name", "summary"]
        }

        # Pass the dict schema directly
        structured_llm = self.llm.with_structured_output(json_schema)
        result = structured_llm.invoke(prompt)

        return result
    

    def _add_cluster_summary(self, cluster_id, leaf_only):
        # non-cluster entity details
        query_entities = """
        MATCH (e)-[r:IN_CLUSTER]->(c:Cluster {id: $cluster_id})
        WHERE NOT e:Cluster AND r.final = true
        RETURN e.id as node_id, labels(e) AS labels, e.name as name, e.description as description
        """
        entities = self.graph.query(query_entities, params={"cluster_id": cluster_id})
        entity_texts = [f"Node ID/Type: {r['node_id']}/{r['labels']} \n\t* name: {r["name"]}\n\t* description: {r["description"]}" for r in entities]

        # child cluster summaries
        if leaf_only:
            subcluster_texts = []
        else:
            query_subclusters = """
            MATCH (child:Cluster)-[:IN_CLUSTER]->(parent:Cluster {id: $cluster_id})
            RETURN child.id AS child_id, child.name AS name, child.summary AS summary
            """
            subclusters = self.graph.query(query_subclusters, params={"cluster_id": cluster_id})
            subcluster_texts = [f"Cluster ID: {r['child_id']} \n\t* name: {r["name"]}\n\t* description: {r["summary"]}" for r in subclusters]

        # Combine into context
        context_parts = []
        if subcluster_texts:
            context_parts.append("### Sub-Community Summaries:\n- " + "\n\n - ".join(subcluster_texts))
        if entity_texts:
            context_parts.append("### Direct Entity Details:\n- " + "\n\n - ".join(entity_texts))

        context = "\n\n".join(context_parts)

        prompt = f"""
        Synthesize the following information for Cluster {cluster_id} into a name and a cohesive summary. 
        It contains both broader sub-community summaries and direct entity/concept details.

        {context}

        When relevant, take in consideration the base ontology described in the following:

        {self.ontology.description}
        """

        # request cluster summaries
        try:
            result = self._get_cluster_summary(prompt)
        except Exception as e:
            print(f"Error while parsing cluster '{cluster_id}': ", e)
            print("Prompt: ", prompt)
        else:
            # Store summary on the Cluster node
            query_save = """
            MATCH (c:Cluster {id: $cluster_id})
            SET c.name = $name, c.summary = $summary
            """
            self.graph.query(query_save, params={
                "cluster_id": cluster_id,
                "name": result["name"],
                "summary": result["summary"]
            })


    def run_leiden_clustering(self, max_cluster_size=10, resolution=1.0):
        # access only ontology, ignoring document structure
        subgraph = self._get_ontology_subgraph()
        
        # get and format edges
        edges = []
        for entry in subgraph:
            source_id = entry["source_id"]
            target_id = entry["target_id"]
            edges.append((source_id, target_id, 1.0))

        # determine cluster structure
        clusters, level_cluster_map = self._get_leiden_clusters(
            edges, max_cluster_size, resolution)

        # create clusters and summarizations
        print("Adding Leiden cluster structure to Database")
        self._add_cluster_node_relations(clusters)
        self._add_cluster_subcluster_relations()

        for level in range(len(level_cluster_map)-1, -1, -1):
            print(f"Creating bottom-up cluster summarization: level {level}")
            for cluster_id in tqdm.tqdm(level_cluster_map[level]):
                leaf_only = level==len(level_cluster_map)-1
                self._add_cluster_summary(cluster_id, leaf_only)


    def reset_databasis(self):
        self.graph.query("MATCH (n) DETACH DELETE n")


    def delete_clustering(self):
        self.graph.query("MATCH (c:Cluster) DETACH DELETE c")


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