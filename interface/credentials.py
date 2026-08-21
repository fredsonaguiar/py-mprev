import streamlit as st
# from streamlit_agraph import agraph, Node, Edge, Config

# Initialize GraphRAG backend
from pymprev.util import GraphRAG
from pymprev.ontology import Ontology
from pymprev.connection import connect_llm, connect_embeder, connect_database


# credentials page
def render_credentials_page():
    st.title("Graph RAG Setup & Credentials")
    st.caption("Enter your model configurations and database connection details to initialize the system.")

    with st.form(key="connection_form"):
        # LLM Settings
        st.subheader("LLM Configuration")
        llm_model = st.text_input(
            "LLM Model name",
            value="gemma-4-31b-it")
        llm_api_key = st.text_input(
            "LLM API Key", 
            type="password",
            help="Your LLM key/password")

        st.divider()

        # Embedder Settings
        st.subheader("Embedder Configuration")
        embedder_model = st.text_input(
            "Embedder Model", 
            value="google/embeddinggemma-300m")
        embedder_key = st.text_input(
            "Embedder Model key/password", 
            type="password",
            help="your Embedder Model key/password")

        st.divider()

        # Database Settings
        st.subheader("Neo4j Database Connection")
        database_uri = st.text_input(
            "Database URI",
            value="neo4j+s://312151bc.databases.neo4j.io")
        database_name = st.text_input(
            "Database name",
            value="312151bc")
        database_username = st.text_input(
            "Database Username",
            value="312151bc")
        database_password = st.text_input(
            "Database Password",
            type="password",
            help="Your database access key/password")

        submit_button = st.form_submit_button(label="Connect & Start Session", type="primary")

    if submit_button:
        # Validate missing fields
        if not llm_api_key or not embedder_key or not database_password:
            st.error("Please fill in all secret key and password fields.")
            return

        # Save configuration in session state
        st.session_state.config = {
            "llm_model": llm_model,
            "llm_api_key": llm_api_key,
            "embedder_model": embedder_model,
            "embedder_key": embedder_key,
            "database_uri": database_uri,
            "database_name": database_name,
            "database_username": database_username,
            "database_password": database_password
        }
        st.session_state.cluster_details = dict()

        with st.spinner("Connecting and initializing models..."):
            try:
                # connect to GraphRag
                ontology = Ontology()
                llm = connect_llm(
                    model_name=llm_model, api_key=llm_api_key)
                embedder = connect_embeder(
                    model_name=embedder_model, api_key=embedder_key) 
                graph = connect_database(
                    uri=database_uri, database=database_name,
                    username=database_username, password=database_password)

                graphRAG = GraphRAG(llm, embedder, graph, ontology)
                
                st.session_state.connected = True
                st.session_state.config["graphRAG"] = graphRAG
                st.success("Connection established successfully!")
                st.rerun()

            except Exception as e:
                st.error(f"Failed to connect to database or initialize models: {e}")


def render_qa_page():
    st.title("Vector Search Q&A Assistant")
    st.caption("Ask natural language questions to query community clusters and source chunks.")

    # get graphRAG from session state
    graphRAG = st.session_state.config["graphRAG"]


    with st.form(key="rag_form"):
        user_query = st.text_input(
            "Enter your query:", 
            value="What safety problems are associated to Neural Nets in Aviation?")
        top_k_global = st.number_input("Top K Global", min_value=1, max_value=20, value=10)
        top_k_local = st.number_input("Top K Local", min_value=1, max_value=20, value=10)
                
        submit_button = st.form_submit_button(label="Query RAG", type="primary")

    if submit_button and user_query:
        with st.spinner("Retrieving vector matches and generating answer..."):
            # call your backend GraphRAG
            result = graphRAG.query_RAG(
                user_query,
                top_k_global=top_k_global,
                top_k_local=top_k_local)

        st.markdown(f"### {result['title']}")
        st.success(result["answer"])

        st.subheader("Retrieved Context")
        with st.expander(f"Matched Cluster Summaries ({len(result['clusters'])})"):
            for cluster in result["clusters"]:
                st.markdown(f"• **[{cluster['name']} - Level {cluster['level']}]** *(Similarity: {cluster['score']:.2f})*")
                st.caption(f"\"{cluster['summary']}\"")

        with st.expander(f"Matched Text Chunks ({len(result['chunks'])})"):
            for chunk in result["chunks"]:
                st.markdown(f"• **Excerpt [`{chunk['index']}`] - Document: {chunk['source']}** *(Similarity: {chunk['score']:.2f})*")
                st.caption(f"\"{chunk['text']}\"")


def _load_cluster_information(cluster_id):
    # get graphRAG from session state
    graphRAG = st.session_state.config["graphRAG"]

    name, summary, level = graphRAG.get_cluster_details(cluster_id)
    childnodes = graphRAG.get_children_entities(cluster_id)
    subclusters = graphRAG.get_children_clusters(cluster_id)

    st.session_state.cluster_details[cluster_id] = {
        "name": name,
        "level":level,
        "summary": summary,
        "childnodes": childnodes,
        "subclusters": subclusters}


def _expand_cluster_tree_id(cluster_id):
    # get graphRAG from session state
    graphRAG = st.session_state.config["graphRAG"]

    if cluster_id not in st.session_state.cluster_details:
        name, summary, level = graphRAG.get_cluster_details(cluster_id)
    else:
        name = st.session_state.cluster_details[cluster_id]["name"]
        level = st.session_state.cluster_details[cluster_id]["level"]
        summary = st.session_state.cluster_details[cluster_id]["summary"]

    st.markdown(f"**Cluster {cluster_id}:** {name}")
    with st.expander(f"See More: summary and subnodes", expanded=False):
        st.markdown(f"**Summary:** {summary}")

        # load further details
        st.markdown("---")
        if cluster_id not in st.session_state.cluster_details:
            st.button(
                "Load Cluster Details",
                key=f"btn_{cluster_id}",
                on_click=_load_cluster_information,
                kwargs={"cluster_id":cluster_id})
        else:
            childnodes = st.session_state.cluster_details[cluster_id]["childnodes"]
            subclusters = st.session_state.cluster_details[cluster_id]["subclusters"]

            if childnodes:
                st.markdown(f"**Child Nodes: {len(childnodes)}**")
                for childnode in childnodes:
                    st.markdown(f" - **{childnode['node_id']} {childnode['labels']}**: {childnode['description']}")

            if subclusters:
                st.markdown(f"**Sub-Clusters: {len(subclusters)}**")
                for subcluster in subclusters:
                    subcluster_id = subcluster["child_id"]
                    _expand_cluster_tree_id(subcluster_id)


def render_hierarchy_page():
    st.title("Top-Bottom Summarization Hierarchy")
    st.caption("Explore how general community topics branch down into granular sub-summaries.")

    # get graphRAG from session state
    graphRAG = st.session_state.config["graphRAG"]

    # top-bottom tree
    cluster_ids = graphRAG.get_cluster_ids_by_level(0)
    for cluster_id in cluster_ids:
        _expand_cluster_tree_id(cluster_id)


def welcome_page():
    st.write("# Welcome to Py-MPREV")

    st.markdown(
    """
    ## Python Preventive Maitenance Document Explorer
        
    This project focuses on the discussion of applicability, requirements, and employment of ML-Based techniques for preventive maintenance in the context of civil aviation, with a focus on Preventive and Health Management (PHM) tools. This is justified by the potential impact of Artificial Intelligence and Machine Learning in this sector, which requires strict regulations, currently not compatible with the most recent technologies.
    """)

# session initialization
if "connected" not in st.session_state:
    st.session_state.connected = False
if "config" not in st.session_state:
    st.session_state.config = {}

# routing map
page_routing = {
    "Welcome": welcome_page,
    "Vector Similarity Q&A": render_qa_page,
    "Topic Hierarchy Navigator": render_hierarchy_page,
}

def main():
    st.sidebar.title("Navigation")

    if not st.session_state.connected:
        st.sidebar.warning("⚠️ Disconnected")
        render_credentials_page()
    else:
        st.sidebar.success("🟢 Connected")
        
        # Sidebar Menu Selection
        page_name = st.sidebar.selectbox(
            "Select Page:",
            page_routing
        )
        try:
            page_routing[page_name]()
        except:
            pass

        st.sidebar.divider()
        st.sidebar.caption(f"**LLM:** `{st.session_state.config['llm_model']}`")
        st.sidebar.caption(f"**Embedder:** `{st.session_state.config['embedder_model']}`")
        st.sidebar.caption(f"**DB URI:** `{st.session_state.config['database_uri']}`")
                
        if st.sidebar.button("Disconnect / Edit Keys"):
            st.session_state.connected = False
            st.session_state.config = {}
            st.rerun()


if __name__ == "__main__":
    main()