from sentence_transformers import SentenceTransformer, util
from src.retriever import load_faiss_index, retrieve_top_k
from src.evaluate_rag import evaluate_rag, run_rag_pipeline
from src.generator import load_generator_model
from src.prompt_template import build_prompt
import pandas as pd

def auto_score_and_comment(question, answer, top_chunks, embed_model):
    question_emb = embed_model.encode(question, convert_to_tensor=True)
    chunk_texts = [c['chunk_text'] for c in top_chunks]
    chunk_embs = embed_model.encode(chunk_texts, convert_to_tensor=True)
    sims = util.pytorch_cos_sim(question_emb, chunk_embs).cpu().numpy()[0]

    avg_sim = sims.mean() if len(sims) > 0 else 0

    answer_lower = answer.lower()
    if "don't have enough information" in answer_lower or len(top_chunks) == 0:
        score = 1
        comment = "No relevant information found in retrieved context."
    elif avg_sim > 0.7:
        score = 5
        comment = "Answer well supported by highly relevant context."
    elif avg_sim > 0.5:
        score = 3
        comment = "Answer somewhat supported by context, moderate relevance."
    else:
        score = 2
        comment = "Low context relevance, answer may be unreliable."

    return score, comment


def evaluate_and_generate_report_auto(questions, index_path, metadata_path, output_path, top_k=5):
    # Load models
    embed_model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
    gen_model = load_generator_model()
    index, metadata = load_faiss_index(index_path, metadata_path)

    rows = []

    for q in questions:
        answer, chunks = run_rag_pipeline(q, index, metadata, embed_model, gen_model, top_k=top_k)

        score, comment = auto_score_and_comment(q, answer, chunks, embed_model)

        # Extract top 2 contexts (with truncation)
        top2 = [c.get('chunk_text', '') for c in chunks[:2]]
        while len(top2) < 2:
            top2.append("")

        rows.append({
            "Question": q,
            "Answer": answer,
            "Top_1_Context": top2[0][:300] + ("..." if len(top2[0]) > 300 else ""),
            "Top_2_Context": top2[1][:300] + ("..." if len(top2[1]) > 300 else ""),
            "Score": score,
            "Comment": comment
        })

    df = pd.DataFrame(rows)

    # Generate markdown report
    generate_evaluation_report(df, output_path)

    return df


def generate_evaluation_report(df_eval, output_path):
    md_lines = ["#  RAG Evaluation Report\n"]
    md_lines.append("| Question | Generated Answer | Top-1 Retrieved Chunk | Top-2 Retrieved Chunk | Score (1-5) | Comments |")
    md_lines.append("|----------|------------------|-----------------------|-----------------------|-------------|----------|")

    for _, row in df_eval.iterrows():
        q = row['Question']
        ans = row['Answer'].replace("\n", " ").replace("|", "\\|")
        top1 = row.get('Top_1_Context', '').replace("\n", " ").replace("|", "\\|")
        top2 = row.get('Top_2_Context', '').replace("\n", " ").replace("|", "\\|")
        score = row.get('Score', '')
        comment = row.get('Comment', '')

        md_lines.append(f"| {q} | {ans} | {top1} | {top2} | {score} | {comment} |")

    md_text = "\n".join(md_lines)

    import os
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(md_text)

    print(f"Markdown evaluation report saved to {output_path}")
