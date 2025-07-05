import gradio as gr
from src.rag_chat import stream_answer

def chat_interface(user_question):
    for partial_answer, sources in stream_answer(user_question):
        sources_md = "\n\n".join([
            f" {i+1}. {src['product']} — {src['chunk_text'][:300]}..." for i, src in enumerate(sources)
        ])
        yield partial_answer, sources_md


with gr.Blocks(
    title="CrediTrust Complaint Assistant",
    css="""
        .gradio-container {
            background-color: #f8f9fa;
            font-family: 'Segoe UI', sans-serif;
        }
        .title {
            font-size: 28px;
            font-weight: bold;
            color: #004085;
            margin-bottom: 0.5rem;
        }
        .description {
            font-size: 16px;
            color: #333;
            margin-bottom: 1rem;
        }
        .gr-button {
            border-radius: 10px !important;
            padding: 10px 20px;
        }
    """
) as demo:

    gr.Markdown("<div class='title'>Intelligent Complaint Analysis Chat developed by Nurye.</div>", elem_classes="title")
    gr.Markdown("<div class='description'>Ask questions about financial complaints. Our assistant retrieves real complaint excerpts and gives contextual answers.</div>", elem_classes="description")

    with gr.Row():
        with gr.Column(scale=1):
            question = gr.Textbox(
                placeholder="Type your question about complaints...",
                label=" Your Question",
                lines=2
            )
            with gr.Row():
                submit = gr.Button(" Ask")
                clear = gr.Button(" Clear")

        with gr.Column(scale=2):
            answer_output = gr.Textbox(label=" AI Answer", lines=4)
            source_output = gr.Textbox(label=" Retrieved Sources", lines=12)

    submit.click(fn=chat_interface, inputs=question, outputs=[answer_output, source_output])
    clear.click(fn=lambda: ("", ""), inputs=None, outputs=[answer_output, source_output])

demo.launch()
