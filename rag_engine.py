import os
import json
import chromadb
import networkx as nx
from dotenv import load_dotenv
import google.generativeai as genai
from PyPDF2 import PdfReader
from chromadb.utils import embedding_functions

# Load environment variables
load_dotenv()
api_key = os.getenv("GEMINI_API_KEY")

if not api_key:
    raise ValueError("GEMINI_API_KEY not found in environment variables or .env file!")

genai.configure(api_key=api_key)

# Global persistent storage paths
CHROMA_PATH = "./chroma_db"
GRAPH_PATH = "./knowledge_graph.graphml"

class GraphRAGEngine:
    def __init__(self):
        # Initialize Gemini Model for generation
        self.model = genai.GenerativeModel("gemini-2.5-flash")
        
        # Initialize Local Embedding Function for ChromaDB (Runs 100% locally and reliably)
        self.ef = embedding_functions.SentenceTransformerEmbeddingFunction(
            model_name="all-MiniLM-L6-v2"
        )
        
        # Initialize ChromaDB Vector Store with local embedding function
        self.chroma_client = chromadb.PersistentClient(path=CHROMA_PATH)
        self.collection = self.chroma_client.get_or_create_collection(
            name="graphrag_chunks",
            embedding_function=self.ef
        )
        
        # Initialize NetworkX Knowledge Graph
        self.graph = nx.Graph()
        if os.path.exists(GRAPH_PATH):
            try:
                self.graph = nx.read_graphml(GRAPH_PATH)
            except Exception:
                self.graph = nx.Graph()

    def save_graph(self):
        nx.write_graphml(self.graph, GRAPH_PATH)

    def extract_text_from_pdf(self, uploaded_file) -> str:
        reader = PdfReader(uploaded_file)
        text = ""
        for page in reader.pages:
            page_text = page.extract_text()
            if page_text:
                text += page_text + "\n"
        return text

    def chunk_text(self, text: str, chunk_size: int = 1000, overlap: int = 200) -> list:
        chunks = []
        start = 0
        while start < len(text):
            end = start + chunk_size
            chunks.append(text[start:end])
            start += chunk_size - overlap
        return chunks

    def process_document(self, uploaded_file, filename: str):
        raw_text = self.extract_text_from_pdf(uploaded_file)
        if not raw_text.strip():
            raise ValueError("Could not extract any text from the uploaded document.")

        chunks = self.chunk_text(raw_text)
        
        # 1. Upsert chunks to ChromaDB (Chroma handles local embedding generation automatically)
        for idx, chunk in enumerate(chunks):
            doc_id = f"{filename}_chunk_{idx}"
            self.collection.upsert(
                documents=[chunk],
                ids=[doc_id],
                metadatas=[{"source": filename, "chunk_index": idx}]
            )

        # 2. Extract Entities and Relationships using Gemini for Knowledge Graph
        sample_text = raw_text[:4000] 
        prompt = f"""
        Extract key entities (nodes) and their relationships (edges) from the following text. 
        Return ONLY valid JSON in this exact format without any markdown wrappers:
        {{
            "entities": ["EntityA", "EntityB", "EntityC"],
            "relationships": [
                {{"source": "EntityA", "target": "EntityB", "relation": "uses"}},
                {{"source": "EntityB", "target": "EntityC", "relation": "depends_on"}}
            ]
        }}
        
        Text:
        {sample_text}
        """
        
        try:
            response = self.model.generate_content(prompt)
            clean_text = response.text.strip()
            if clean_text.startswith("```json"):
                clean_text = clean_text[7:]
            if clean_text.endswith("```"):
                clean_text = clean_text[:-3]
                
            data = json.loads(clean_text.strip())
            
            for ent in data.get("entities", []):
                self.graph.add_node(ent, label=ent)
                
            for rel in data.get("relationships", []):
                s = rel.get("source")
                t = rel.get("target")
                r = rel.get("relation", "relates_to")
                if s and t:
                    self.graph.add_edge(s, t, relation=r)
                    
            self.save_graph()
        except Exception as e:
            print(f"Graph extraction warning: {e}")

        return len(chunks), self.graph.number_of_nodes(), self.graph.number_of_edges()

    def dual_index_retrieve(self, query: str, top_k: int = 3):
        # 1. Vector Search using local embedding function
        results = self.collection.query(
            query_texts=[query],
            n_results=top_k
        )
        retrieved_chunks = results.get("documents", [[]])[0]
        
        # 2. Graph Traversal Context
        graph_context = []
        query_words = set(query.lower().split())
        for node in self.graph.nodes():
            if any(w in node.lower() for w in query_words):
                neighbors = list(self.graph.neighbors(node))
                for n in neighbors:
                    edge_data = self.graph.get_edge_data(node, n)
                    rel = edge_data.get("relation", "connected_to") if edge_data else "connected_to"
                    graph_context.append(f"Graph Link: {node} --[{rel}]--> {n}")
                
        return retrieved_chunks, graph_context[:10]

    def synthesize_answer(self, query: str) -> dict:
        chunks, graph_edges = self.dual_index_retrieve(query)
        
        context_str = "\n\n".join([f"[Vector Chunk]: {c}" for c in chunks])
        if graph_edges:
            context_str += "\n\n" + "\n".join([f"[Graph Relation]: {e}" for e in graph_edges])
            
        prompt = f"""
        You are GraphRAG Nexus, an advanced multi-agent research assistant. Answer the user's query thoroughly using ONLY the provided Vector Chunks and Graph Relations. If the information is insufficient, state so clearly.
        
        Context:
        {context_str}
        
        User Query: {query}
        
        Provide a structured, deeply analytical answer with citations where appropriate.
        """
        
        response = self.model.generate_content(prompt)
        return {
            "answer": response.text,
            "chunks": chunks,
            "graph_context": graph_edges
        }