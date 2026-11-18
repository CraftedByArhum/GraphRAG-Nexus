import os
import json
import chromadb
import networkx as nx
import streamlit as st
from dotenv import load_dotenv
import google.generativeai as genai
from PyPDF2 import PdfReader
from chromadb.utils import embedding_functions

# Load environment variables (Local .env or Streamlit Secrets)
load_dotenv()
try:
    api_key = st.secrets["GEMINI_API_KEY"]
except Exception:
    api_key = os.getenv("GEMINI_API_KEY")

if not api_key:
    raise ValueError("GEMINI_API_KEY not found in Streamlit Secrets or environment variables!")

genai.configure(api_key=api_key)

class GraphRAGEngine:
    def __init__(self):
        # Using Gemini 3.1 Flash Lite for efficient, high-limit text generation
        self.model = genai.GenerativeModel("gemini-3.1-flash-lite")
        
        # Initialize Local Embedding Function for ChromaDB
        self.ef = embedding_functions.SentenceTransformerEmbeddingFunction(
            model_name="all-MiniLM-L6-v2"
        )
        
        # Ephemeral In-Memory Client ensures complete isolation per session
        self.chroma_client = chromadb.Client()
        self.collection = self.chroma_client.get_or_create_collection(
            name="graphrag_session_chunks",
            embedding_function=self.ef
        )
        
        # Isolated NetworkX Knowledge Graph for this session
        self.graph = nx.Graph()

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
        # Clear previous session data if a new file is uploaded
        try:
            existing_ids = self.collection.get()["ids"]
            if existing_ids:
                self.collection.delete(ids=existing_ids)
        except Exception:
            pass
        self.graph.clear()

        raw_text = self.extract_text_from_pdf(uploaded_file)
        if not raw_text.strip():
            raise ValueError("Could not extract any text from the uploaded document.")

        chunks = self.chunk_text(raw_text)
        
        # Upsert chunks into session-isolated ChromaDB collection
        for idx, chunk in enumerate(chunks):
            doc_id = f"{filename}_chunk_{idx}"
            self.collection.upsert(
                documents=[chunk],
                ids=[doc_id],
                metadatas=[{"source": filename, "chunk_index": idx}]
            )

        # Extract Entities and Relationships using Gemini for Knowledge Graph
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
        except Exception as e:
            print(f"Graph extraction warning: {e}")

        return len(chunks), self.graph.number_of_nodes(), self.graph.number_of_edges()

    def dual_index_retrieve(self, query: str, top_k: int = 3):
        results = self.collection.query(
            query_texts=[query],
            n_results=top_k
        )
        retrieved_chunks = results.get("documents", [[]])[0]
        
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