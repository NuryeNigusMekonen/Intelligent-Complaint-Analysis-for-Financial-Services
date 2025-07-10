import faiss
import pickle
import numpy as np
from sentence_transformers import SentenceTransformer

def load_faiss_index(index_path, metadata_path):
    index = faiss.read_index(index_path)
    with open(metadata_path, "rb") as f:
        metadata = pickle.load(f)
    return index, metadata

def retrieve_top_k(query, index, metadata, embedding_model, k=5):
    embedding = embedding_model.encode([query])[0]
    D, I = index.search(np.array([embedding]).astype("float32"), k)
    results = []
    for idx in I[0]:
        results.append(metadata[idx])
    return results

