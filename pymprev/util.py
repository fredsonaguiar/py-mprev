import os
import tqdm
import pymupdf

from tenacity import retry, stop_after_attempt, wait_exponential

from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_neo4j.graph_transformers.llm import LLMGraphTransformer

# community detection
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


    def _get_nodes_by_similarity(
            self, node_type,
            low_threshold=0.95, high_threshold=0.995, text_threshold=0.95):
        query = """
        MATCH (target:{node_type})
        MATCH (duplicate:{node_type})
        WHERE target.id < duplicate.id
        // 1. Vector similarity check
        WITH target, duplicate, 
            vector.similarity.cosine(target.embedding, duplicate.embedding) AS vec_score
        WHERE vec_score >= $low_threshold
        // 2. Lexical Levenshtein similarity check using APOC
        WITH target, duplicate, vec_score,
            apoc.text.levenshteinSimilarity(toLower(target.id), toLower(duplicate.id)) AS text_score
        // Require EITHER ultra-high vector similarity OR high vector + strong text match
        WHERE (vec_score >= $high_threshold) 
        OR (text_score >= $text_threshold)

        RETURN target.id AS target, 
            duplicate.id AS duplicate, 
            vec_score,
            text_score
        ORDER BY vec_score DESC
        """.format(node_type=node_type)

        params={"low_threshold":low_threshold,
                "high_threshold":high_threshold,
                "text_threshold":text_threshold}
        
        unifiable = self.graph.query(query, params=params)
        return unifiable


    def unify_nodes_by_similarity(
            self,
            low_threshold=0.95, high_threshold=0.995, text_threshold=0.95):

        query = """
        UNWIND $candidates AS row
        MATCH (target:{node_type}) WHERE target.id = row.target_id
        MATCH (duplicate:{node_type}) WHERE duplicate.id = row.duplicate_id
        MERGE (target)-[:TEMP_SAME_AS]-(duplicate)
        """

        merge_query = """
        MATCH (n:{node_type})-[:TEMP_SAME_AS]-()
        CALL apoc.path.subgraphNodes(n, {{relationshipFilter: "TEMP_SAME_AS"}}) YIELD node AS m
        // group by n so each component is collected individually
        WITH n, collect(DISTINCT m) AS cluster
        WHERE size(cluster) > 1
        // identify the representative ID for each component
        WITH coll.min([x IN cluster | x.id]) AS rep_id, cluster
        // deduplicate to execute 1 merge component
        WITH DISTINCT rep_id, cluster
        WITH
            [x IN cluster WHERE x.id = rep_id][0] AS target,
            [x IN cluster WHERE x.id <> rep_id] AS duplicates
        CALL apoc.refactor.mergeNodes([target] + duplicates, {{
            properties: {{ name: "discard", description: "combine", `.*`: "override" }},
            mergeRels: true
        }}) YIELD node
        RETURN count(node) AS merged_count
        """

        cleanup_query = "MATCH ()-[r:TEMP_SAME_AS]-() DELETE r"

        for node_type in self.ontology.nodes:
            unifiable = self._get_nodes_by_similarity(
                node_type, low_threshold, high_threshold, text_threshold)
            
            candidates = [{"target_id":row["target"],
                           "duplicate_id":row["duplicate"]} for row in unifiable]

            # creates temporary relations
            self.graph.query(query.format(node_type=node_type),
                             params={"candidates":candidates})

            # merges temporary component
            res = self.graph.query(merge_query.format(node_type=node_type))
            merged_count = res[0]["merged_count"]
            print(f"total merged nodes of type {node_type}: {merged_count}")

            # removes temporary relations
            self.graph.query(cleanup_query)


    def add_knn_similarity_relations(self, score_threshold=0.8, top_k=10):
        close_neighbors_query = """
        MATCH (target:{target_type})
        MATCH (duplicate:{duplicate_type}) 
        WHERE duplicate.id <> target.id
        WITH target, duplicate,
            vector.similarity.cosine(target.embedding, duplicate.embedding) AS score
        WHERE score >= $score_threshold
        RETURN target.id as target, duplicate.id AS duplicate, score
        """

        link_query = """
        UNWIND $candidates AS row
        MATCH (a:{target_type} {{id: row.target_id}})
        MATCH (b:{duplicate_type} {{id: row.duplicate_id}})
        MERGE (a)-[r:SIMILAR_TO]-(b)
        SET r.score = row.score
        """

        node_types = " | ".join(self.ontology.nodes)

        for target_type in self.ontology.nodes:
            print("finding similar neighbors per type:", target_type)
            # get neighbors
            query = close_neighbors_query.format(
                target_type=target_type, duplicate_type=node_types)
            results = self.graph.query(
                query, params={"score_threshold":score_threshold})

            neighbors = dict()
            for row in results:
                if row["target"] not in neighbors:
                    neighbors[row["target"]] = []
                neighbors[row["target"]].append(row)

            # get most similar k
            candidates = []
            for target_id, row in neighbors.items():
                top_k_row = sorted(row, key=lambda x: x["score"], reverse=True)
                candidates += [{"target_id":target_id,
                                "duplicate_id":row["duplicate"],
                                "score":row["score"]} for row in top_k_row[:top_k]]

            query = link_query.format(
                target_type=target_type, duplicate_type=node_types)
            self.graph.query(query, params={"candidates": candidates})


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
                self._add_source_chunk_node(
                    chunk_text=chunk,
                    chunk_index=chunk_index,
                    source_filename=filename)

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

        # computing chunk and node embeddings
        self.reset_node_embeddings()
        self.reset_chunk_embeddings()

        # unify entities
        self.unify_nodes_by_similarity()
        self.add_knn_similarity_relations(10)


    def _get_aiml_ontology_subgraph(self):
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


    def _get_node_similarity_subgraph(self):
        node_types = " | ".join(self.ontology.nodes)
        query = """
        MATCH (source: {node_types})-[r: SIMILAR_TO]->(target: {node_types})
        RETURN
            source.id AS source_id,
            target.id AS target_id,
            r.score as weight
        """.format(node_types=node_types)
                
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
        entities = self.get_children_entities(cluster_id)
        entity_texts = [self._node_as_text(r['node_id'], r['labels'], r['name'], r['description']) for r in entities]

        # child cluster summaries
        if leaf_only:
            subcluster_texts = []
        else:
            subclusters = self.get_children_clusters(cluster_id)
            subcluster_texts = [self._cluster_as_text(r['child_id'], r['name'], r['summary']) for r in subclusters]

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


    def run_leiden_clustering(
            self, max_cluster_size=10, resolution=1.0, ontology_weight=1.0):
        # access only ontology, ignoring document structure
        ontology_subgraph = self._get_aiml_ontology_subgraph()
        similarity_subgraph = self._get_node_similarity_subgraph()

        edges = []
        for entry in ontology_subgraph:
            edges.append((entry["source_id"], entry["target_id"], ontology_weight))

        for entry in similarity_subgraph:
            edges.append((entry["source_id"], entry["target_id"], entry["weight"]))

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


    def reset_chunk_embeddings(self, batch_size=1):

        fetch_query = """
        MATCH (c:Chunk)
        RETURN elementId(c) AS id, c.text AS text
        """
        records = self.graph.query(fetch_query)
        
        # Processa em lotes
        chunk_prefix = "title: none | text: "
        print("seting text chunk embeddings")
        for i in tqdm.tqdm(range(0, len(records), batch_size)):
            batch_records = records[i : i + batch_size]

            # Prepara textos com o prefixo oficial de documento
            texts = [f"{chunk_prefix}{r['text']}" for r in batch_records]

            # Vetorização em lote na GPU
            embeddings = self.embedder.embed_documents(texts)

            # Monta o payload para a atualização no Cypher
            payload = [
                {"id": r["id"], "embedding": emb}
                for r, emb in zip(batch_records, embeddings)]

            # Atualização em massa no Neo4j
            update_query = """
            UNWIND $batch AS row
            MATCH (c) WHERE elementId(c) = row.id
            SET c.embedding = row.embedding
            """
            self.graph.query(update_query, params={"batch": payload})


    def reset_node_embeddings(self, batch_size=1):
        fetch_query = """
        MATCH (n : {node_type})
        RETURN elementId(n) AS id,
            n.id AS name_id,
            labels(n) as type,
            n.name AS name,
            n.description AS description
        """

        update_query = """
        UNWIND $batch AS row
        MATCH (n) WHERE elementId(n) = row.id
        SET n.embedding = row.embedding
        """

        print("seting node embeddings")
        # Busca todos os nós que não são Chunks
        for node_type in self.ontology.nodes:
            print("seting node embeddings type: ", node_type)
            fetch_query_type = fetch_query.format(node_type=node_type)
            records = self.graph.query(fetch_query_type)
            node_prefix = "task: clustering | query: "
            for i in tqdm(range(0, len(records), batch_size)):
                batch_records = records[i : i + batch_size]

                # Formata o texto de cada nó e aplica o prefixo de clustering
                texts = [
                    f"{node_prefix}{
                        self._node_as_text(r['name_id'], r['type'], r['name'], r['description'])}"
                        for r in batch_records]

                # Vetorização em lote na GPU
                embeddings = self.embedder.embed_documents(texts)

                payload = [
                    {"id": r["id"], "embedding": emb}
                    for r, emb in zip(batch_records, embeddings)]

                self.graph.query(update_query, params={"batch": payload})
                # print(f"  └─ Entidades {i} até {i + len(batch_records)} atualizadas.")


    def get_children_clusters(self, cluster_id):
        query_subclusters = """
        MATCH (child:Cluster)-[:IN_CLUSTER]->(parent:Cluster {id: $cluster_id})
        RETURN child.id AS child_id, child.name AS name, child.summary AS summary
        """
        subclusters = self.graph.query(query_subclusters, params={"cluster_id": cluster_id})
        return subclusters
    

    def get_children_entities(self, cluster_id):
        query_entities = """
        MATCH (e)-[r:IN_CLUSTER]->(c:Cluster {id: $cluster_id})
        WHERE NOT e:Cluster AND r.final = true
        RETURN e.id as node_id, labels(e) AS labels, e.name as name, e.description as description
        """
        entities = self.graph.query(query_entities, params={"cluster_id": cluster_id})
        return entities


    def _cluster_as_text(self, cluster_id, name, descripion):
        return f"Cluster ID: {cluster_id} \n\t* name: {name}\n\t* description: {descripion}"


    def _node_as_text(self, node_id, labels, name, description):
        return f"Node ID/Type: {node_id}/{labels} \n\t* name: {name}\n\t* description: {description}"


    def get_cluster_details(self, cluster_id):
        query = """
        MATCH (c:Cluster {id: $cluster_id})
        RETURN c.name as name, c.summary as summary, c.level as level
        """
        result = self.graph.query(query, params={"cluster_id":cluster_id})

        name = result[0]["name"]
        summary = result[0]["summary"]
        level = result[0]["level"]

        return name, summary, level


    def get_cluster_ids_by_level(self, level=0):
        query = """
        MATCH (c:Cluster {level: $level})
        RETURN c.id as id
        """
        results = self.graph.query(query, params={"level":level})
        return [row["id"] for row in results]


    def custom_query_RAG(self, text_query:str):
        "task: search result | query: "
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