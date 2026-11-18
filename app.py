import streamlit as st
from rag_engine import GraphRAGEngine
from streamlit_agraph import agraph, Node, Edge, Config

st.set_page_config(
    page_title="GraphRAG Nexus | Hybrid Knowledge Engine",
    page_icon="🧬",
    layout="wide"
)

# Bind the GraphRAGEngine strictly to the user's private session state
if "engine" not in st.session_state:
    st.session_state.engine = GraphRAGEngine()

engine = st.session_state.engine

st.title("🧬 GraphRAG Nexus")
st.markdown("**Hybrid Vector-Graph Multi-Agent Knowledge Synthesis Engine** — Moving beyond flat RAG with structural entity mapping and multi-hop reasoning.")

# Sidebar controls
st.sidebar.header("1. Document Ingestion")
uploaded_file = st.sidebar.file_uploader("Upload a PDF research paper or study text", type=["pdf"])

if uploaded_file is not None:
    if st.sidebar.button("Process & Build Graph Index"):
        with st.spinner("Extracting text, chunking vectors, and synthesizing Knowledge Graph..."):
            try:
                num_chunks, num_nodes, num_edges = engine.process_document(uploaded_file, uploaded_file.name)
                st.sidebar.success(f"Successfully processed! Stored {num_chunks} vector chunks. Knowledge graph has {num_nodes} nodes and {num_edges} edges.")
            except Exception as e:
                st.sidebar.error(f"Error processing file: {e}")

st.sidebar.markdown("---")
st.sidebar.info("Each student session runs in an isolated, secure memory sandbox. Uploading a new file automatically replaces your current session index.")

# Main Layout Tabs
tab1, tab2 = st.tabs(["💬 Knowledge Query & Synthesis", "🕸️ Interactive Knowledge Graph"])

with tab1:
    st.subheader("Multi-Agent Dual-Retrieval Query")
    user_query = st.text_input("Ask a complex question spanning your ingested documents:", placeholder="e.g., How do the core architectural concepts connect across documents?")
    
    if st.button("Synthesize Answer") and user_query:
        with st.spinner("Executing multi-agent dual retrieval (Vector Search + Graph Traversal)..."):
            result = engine.synthesize_answer(user_query)
            
            st.markdown("### 🧠 Synthesized AI Response")
            st.write(result["answer"])
            
            col1, col2 = st.columns(2)
            with col1:
                st.markdown("#### 📄 Retrieved Vector Chunks")
                for i, chunk in enumerate(result["chunks"]):
                    st.info(f"**Chunk {i+1}:** {chunk[:300]}...")
                    
            with col2:
                st.markdown("#### 🔗 Graph Relational Context")
                if result["graph_context"]:
                    for edge in result["graph_context"]:
                        st.success(edge)
                else:
                    st.warning("No explicit graph edge matches found for this query term.")

with tab2:
    st.subheader("Global Knowledge Graph Visualization")
    st.markdown("Visualizing extracted entities (nodes) and relational links (edges) stored in the graph database.")
    
    if len(engine.graph.nodes) > 0:
        nodes = []
        edges = []
        
        displayed_nodes = list(engine.graph.nodes())[:50]
        
        for node in displayed_nodes:
            nodes.append(Node(id=node, label=node, size=20, color="#4F46E5"))
            
        for u, v, data in engine.graph.edges(data=True):
            if u in displayed_nodes and v in displayed_nodes:
                label = data.get("relation", "relates")
                edges.append(Edge(source=u, target=v, label=label))
                
        config = Config(
            width=1100,
            height=600,
            directed=True,
            nodeHighlightBehavior=True,
            highlightColor="#F59E0B",
            collapsible=False,
            physics=True
        )
        
        try:
            agraph(nodes=nodes, edges=edges, config=config)
        except Exception as e:
            st.error(f"Graph rendering error: {e}")
    else:
        st.info("Knowledge graph is currently empty. Upload and process a PDF document using the sidebar to populate nodes and edges.")