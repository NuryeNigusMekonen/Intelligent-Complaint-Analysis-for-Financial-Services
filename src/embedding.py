# src/embedding.py

import os
import numpy as np
import pandas as pd
import faiss
import torch
import pickle
from tqdm import tqdm
from sentence_transformers import SentenceTransformer
from langchain.text_splitter import RecursiveCharacterTextSplitter
import gc

def create_vector_store(
    input_path="../data/filtered_complaints.csv",
    vector_dir="../vector_store",
    chunk_size=5000,
    chunk_overlap=100,
    embedding_model_name="sentence-transformers/all-MiniLM-L6-v2",
    use_gpu=True,
    slice_size=5000  # Number of chunks to embed per slice
):
    os.makedirs(vector_dir, exist_ok=True)
    print(" Loading cleaned complaint data...")
    df = pd.read_csv(input_path)
    print(f" Loaded {len(df):,} records. Starting chunking...")

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap
    )

    all_chunks = []
    all_metadata = []

    for idx, row in tqdm(df.iterrows(), total=len(df), desc=" Chunking rows"):
        chunks = text_splitter.split_text(row['cleaned_narrative'])
        all_chunks.extend(chunks)
        all_metadata.extend([{
            "complaint_id": row["complaint_id"],
            "product": row["product"],
            "chunk_id": i,
            "chunk_text": chunks[i]  
        } for i in range(len(chunks))])

    print(f"\n Total chunks generated: {len(all_chunks):,}")

    print(" Loading embedding model...")
    model = SentenceTransformer(embedding_model_name)
    device = "cuda" if use_gpu and torch.cuda.is_available() else "cpu"
    model = model.to(device)
    print(f" Embedding on device: {device}")

    index = faiss.IndexFlatL2(384)
    metadata_store = []

    total_chunks = len(all_chunks)
    total_slices = (total_chunks // slice_size) + 1

    for slice_idx in tqdm(range(total_slices), desc=" Encoding slices"):
        start = slice_idx * slice_size
        end = min((slice_idx + 1) * slice_size, total_chunks)
        chunk_slice = all_chunks[start:end]
        meta_slice = all_metadata[start:end]

        if not chunk_slice:
            continue

        embeddings = model.encode(
            chunk_slice,
            batch_size=8,
            show_progress_bar=True,
            device=device,
            convert_to_numpy=True
        ).astype("float32")

        index.add(embeddings)
        metadata_store.extend(meta_slice)

        # Cleanup
        del embeddings
        torch.cuda.empty_cache()
        gc.collect()

        # Optional checkpoint
        if (slice_idx + 1) % 10 == 0 or end == total_chunks:
            faiss.write_index(index, os.path.join(vector_dir, "faiss_index_checkpoint.idx"))
            with open(os.path.join(vector_dir, "metadata_checkpoint.pkl"), "wb") as f:
                pickle.dump(metadata_store, f)
            print(f" Saved checkpoint at slice {slice_idx + 1}")

    # Final save
    faiss.write_index(index, os.path.join(vector_dir, "faiss_index.idx"))
    with open(os.path.join(vector_dir, "metadata.pkl"), "wb") as f:
        pickle.dump(metadata_store, f)

    print(f"\n Finished! FAISS index with {index.ntotal} vectors saved.")
    return index, metadata_store
