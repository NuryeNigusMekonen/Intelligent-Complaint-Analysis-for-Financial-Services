from src.retriever import load_faiss_index, retrieve_top_k
from src.prompt_template import build_prompt
from src.generator import load_generator_model
from sentence_transformers import SentenceTransformer

import os

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
INDEX_PATH = os.path.join(PROJECT_ROOT, "vector_store", "faiss_index.idx")
METADATA_PATH = os.path.join(PROJECT_ROOT, "vector_store", "metadata.pkl")

embed_model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
gen_model = load_generator_model()
index, metadata = load_faiss_index(INDEX_PATH, METADATA_PATH)

MAX_CHUNK_CHAR_LEN = 1800

def stream_answer(question, top_k=5):
    chunks = retrieve_top_k(question, index, metadata, embed_model, k=top_k)
    context = "\n\n".join([
        f"[{c['product']}] {c['chunk_text'][:MAX_CHUNK_CHAR_LEN]}" for c in chunks
    ])
    prompt = build_prompt(context, question)

    #  Proper handling of model output
    full_answer = gen_model(prompt)  # This is likely a list
    if isinstance(full_answer, list):
        full_answer = full_answer[0]['generated_text']  # Adjust based on  model output

    current = ""
    for word in full_answer.split():
        current += word + " "
        yield current.strip(), chunks
