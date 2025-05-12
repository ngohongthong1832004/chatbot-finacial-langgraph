import os
import json
import numpy as np
from sentence_transformers import SentenceTransformer
import pickle
import faiss
from typing import List, Dict, Any

def load_chunks(jsonl_path: str) -> List[Dict[str, Any]]:
    """Load chunks from a JSONL file."""
    chunks = []
    with open(jsonl_path, 'r', encoding='utf-8') as f:
        for line in f:
            chunks.append(json.loads(line))
    return chunks

def create_embeddings(chunks: List[Dict[str, Any]], model_name: str = "all-MiniLM-L6-v2") -> np.ndarray:
    """Create embeddings for chunks using the specified model."""
    model = SentenceTransformer(model_name)
    texts = [chunk["content"] for chunk in chunks]
    embeddings = model.encode(texts, show_progress_bar=True)
    return embeddings

def build_faiss_index(embeddings: np.ndarray) -> faiss.IndexFlatL2:
    """Build a FAISS index from embeddings."""
    # Get embedding dimension
    dimension = embeddings.shape[1]
    
    # Create index
    index = faiss.IndexFlatL2(dimension)
    
    # Add embeddings to index
    index.add(embeddings)
    
    return index

def save_rag_resources(chunks: List[Dict[str, Any]], embeddings: np.ndarray, index: faiss.Index, output_dir: str):
    """Save all RAG resources to disk."""
    os.makedirs(output_dir, exist_ok=True)
    
    # Save chunks
    with open(os.path.join(output_dir, "chunks.json"), 'w', encoding='utf-8') as f:
        json.dump(chunks, f, ensure_ascii=False, indent=2)
    
    # Save embeddings
    with open(os.path.join(output_dir, "embeddings.pkl"), 'wb') as f:
        pickle.dump(embeddings, f)
    
    # Save FAISS index
    faiss.write_index(index, os.path.join(output_dir, "faiss_index.bin"))
    
    print(f"Saved RAG resources to {output_dir}")

def main():
    # Directory containing flattened data
    input_dir = "flattened_sec_data"
    output_dir = "sec_embeddings"
    
    # Load the different types of chunks
    print("Loading chunks...")
    rag_chunks = load_chunks(os.path.join(input_dir, "rag_chunks.jsonl"))
    combined_chunks = load_chunks(os.path.join(input_dir, "combined_chunks.jsonl"))
    
    # Combine all chunks
    all_chunks = rag_chunks + combined_chunks
    
    # Create embeddings
    print(f"Creating embeddings for {len(all_chunks)} chunks...")
    embeddings = create_embeddings(all_chunks)
    
    # Build FAISS index
    print("Building FAISS index...")
    index = build_faiss_index(embeddings)
    
    # Save everything
    print("Saving RAG resources...")
    save_rag_resources(all_chunks, embeddings, index, output_dir)
    
    print(f"Done! Created embeddings for {len(all_chunks)} chunks")
    print(f"Embedding dimension: {embeddings.shape[1]}")
    print(f"Index size: {index.ntotal}")

if __name__ == "__main__":
    main()