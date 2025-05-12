import os
import json
import numpy as np
import faiss
import pickle
from sentence_transformers import SentenceTransformer
from typing import List, Dict, Any, Tuple

def load_rag_resources(resource_dir: str) -> Tuple[List[Dict[str, Any]], np.ndarray, faiss.Index]:
    """Load RAG resources from disk."""
    # Load chunks
    with open(os.path.join(resource_dir, "chunks.json"), 'r', encoding='utf-8') as f:
        chunks = json.load(f)
    
    # Load embeddings
    with open(os.path.join(resource_dir, "embeddings.pkl"), 'rb') as f:
        embeddings = pickle.load(f)
    
    # Load FAISS index
    index = faiss.read_index(os.path.join(resource_dir, "faiss_index.bin"))
    
    return chunks, embeddings, index

def query_rag(query: str, chunks: List[Dict[str, Any]], index: faiss.Index, 
              model_name: str = "all-MiniLM-L6-v2", top_k: int = 3) -> List[Dict[str, Any]]:
    """Query the RAG system and return the top k relevant chunks."""
    # Create model
    model = SentenceTransformer(model_name)
    
    # Encode query
    query_embedding = model.encode([query])[0].reshape(1, -1).astype(np.float32)
    
    # Search in the index
    distances, indices = index.search(query_embedding, top_k)
    
    # Get the top k chunks
    results = []
    for i, idx in enumerate(indices[0]):
        if idx < len(chunks):
            result = chunks[idx].copy()
            result["score"] = float(1.0 / (1.0 + distances[0][i]))  # Convert distance to a score
            results.append(result)
    
    return results

def print_results(results: List[Dict[str, Any]], query: str):
    """Print RAG results in a readable format."""
    print(f"\n===== RESULTS FOR QUERY: '{query}' =====\n")
    
    for i, result in enumerate(results):
        print(f"RESULT #{i+1} (Score: {result['score']:.4f}, Type: {result['metadata']['type']})")
        print("-" * 80)
        
        # Print metadata
        for key, value in result['metadata'].items():
            print(f"{key}: {value}")
        
        print("\nCONTENT:")
        print("-" * 80)
        
        # Print first 10 lines of content or all if less than 10
        content_lines = result['content'].split('\n')
        for j, line in enumerate(content_lines[:min(10, len(content_lines))]):
            print(line)
        
        if len(content_lines) > 10:
            print("...")
            print(f"[{len(content_lines) - 10} more lines]")
        
        print("\n" + "=" * 80 + "\n")

def test_queries():
    """Run test queries on the RAG system."""
    # Load resources
    resource_dir = "sec_embeddings"
    chunks, embeddings, index = load_rag_resources(resource_dir)
    
    # Test queries
    queries = [
        "What is the adsh field?",
        "How are the NUM and SUB tables related?",
        "What is the format of the ddate field in the NUM table?",
        "Tell me about the TAG table and its fields",
        "What is the Field Name in the SUB table that contains company name?",
        "How does the PRE table link to other tables?",
        "What is the maximum size of the tag field?",
        "Which fields in NUM table are nullable?",
        "What does the format ALPHANUMERIC mean in SEC data?"
    ]
    
    # Run each query
    for query in queries:
        results = query_rag(query, chunks, index, top_k=3)
        print_results(results, query)
        
        # Ask if user wants to continue
        if query != queries[-1]:
            input("Press Enter to continue to the next query...")

def interactive_mode():
    """Run in interactive mode where users can input their own queries."""
    # Load resources
    resource_dir = "sec_embeddings"
    chunks, embeddings, index = load_rag_resources(resource_dir)
    
    print("\n===== SEC Data Interactive RAG System =====")
    print("Type 'exit' to quit")
    
    while True:
        query = input("\nEnter your query: ")
        if query.lower() in ['exit', 'quit']:
            break
        
        results = query_rag(query, chunks, index, top_k=3)
        print_results(results, query)

def main():
    # Check if resources exist
    if not os.path.exists("sec_embeddings/chunks.json"):
        print("Error: Embeddings not found. Run create_embeddings.py first.")
        return
    
    # Ask user which mode to run
    mode = input("Choose mode (1: Test predefined queries, 2: Interactive): ")
    
    if mode == "1":
        test_queries()
    else:
        interactive_mode()

if __name__ == "__main__":
    main()