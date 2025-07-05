# src/evaluate_rag.py

from retriever import load_faiss_index, retrieve_top_k
from prompt_template import build_prompt
from generator import load_generator_model, generate_answer
from sentence_transformers import SentenceTransformer
import pandas as pd

# Max number of characters allowed per chunk to stay under token limit
MAX_CHUNK_CHAR_LEN = 1800

def run_rag_pipeline(question, index, metadata, embed_model, gen_model, top_k=5):
    chunks = retrieve_top_k(question, index, metadata, embed_model, k=top_k)

    if len(chunks) > 0:
        print("Chunks sample keys:", chunks[0].keys())
        print("Chunks sample content:", chunks[0])
    else:
        print("No chunks retrieved for question:", question)

    # Truncate chunk_text safely before generating the prompt
    context = "\n\n".join([
        f"[{c['product']}] {c['chunk_text'][:MAX_CHUNK_CHAR_LEN]}"
        for c in chunks
    ])

    prompt = build_prompt(context, question)
    answer = generate_answer(prompt, gen_model)
    return answer, chunks


def evaluate_rag(questions, index_path, metadata_path):
    embed_model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
    gen_model = load_generator_model()
    index, metadata = load_faiss_index(index_path, metadata_path)

    rows = []
    for q in questions:
        answer, chunks = run_rag_pipeline(q, index, metadata, embed_model, gen_model)
        rows.append({
            "Question": q,
            "Answer": answer,
            "Top_1_Context": chunks[0]['chunk_text'][:MAX_CHUNK_CHAR_LEN] + "...",
            "Score": "",      # To be filled manually
            "Comment": ""     # Manual observations or analysis
        })

    return pd.DataFrame(rows)
