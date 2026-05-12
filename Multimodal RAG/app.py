"""
app.py — Upgraded Gradio UI with:
  1. Arbitrary PDF upload (ingest any paper, not just the hardcoded one)
  2. RAGAS evaluation scores (faithfulness, answer relevancy, context precision)
  3. Gemini-as-VLM design choice toggle
"""

import os
import time
import threading

import gradio as gr

from generation import generate_rag_response
from evaluation import evaluate_rag_response, format_scores_for_display
from ingestion import ingest_pdf


# ── Helper: stream response ───────────────────────────────────────────────────
def process_query_stream(query, search_type, model_type):
    full_response = ""
    for chunk in generate_rag_response(query, search_type, 5, model_type, stream=True):
        full_response += chunk
        if len(chunk) > 10 or chunk.endswith((".", "!", "?", "\n")):
            time.sleep(0.01)
            yield full_response
    yield full_response


def process_query_normal(query, search_type, model_type):
    return generate_rag_response(query, search_type, 5, model_type, stream=False)


# ── Tab 1: Q&A ────────────────────────────────────────────────────────────────
def on_submit(query, search_type, model_type, stream, run_eval):
    if not query.strip():
        yield "Please enter a question.", ""
        return

    yield "Retrieving relevant information...", ""

    if stream:
        full_answer = ""
        for chunk in process_query_stream(query, search_type, model_type):
            full_answer = chunk
            yield full_answer, ""
    else:
        full_answer = process_query_normal(query, search_type, model_type)
        yield full_answer, ""

    # Run RAGAS evaluation after answer is complete
    # Run RAGAS evaluation after answer is complete
    if run_eval:
        yield full_answer, "⏳ Running RAGAS evaluation..."
        try:
            result = generate_rag_response(
                query, search_type, 5, model_type,
                stream=False, return_contexts=True
            )

            # Safely unpack — handle empty or unexpected result
            if isinstance(result, tuple) and len(result) == 2:
                answer, contexts = result
            else:
                contexts = []

            if not contexts:
                yield full_answer, "⚠️ Could not retrieve contexts for evaluation. Try a different search method."
            else:
                scores = evaluate_rag_response(
                    question=query,
                    answer=full_answer,
                    contexts=contexts,
                )
                eval_display = format_scores_for_display(scores)
                yield full_answer, eval_display
        except Exception as e:
            yield full_answer, f"Evaluation error: {str(e)}"


# ── Tab 2: PDF Upload & Ingestion ─────────────────────────────────────────────
def on_ingest(pdf_file, use_gemini_vlm, progress=gr.Progress()):
    if pdf_file is None:
        return "❌ Please upload a PDF file first."

    pdf_path = pdf_file.name
    pdf_name = os.path.basename(pdf_path)

    yield f"📄 Starting ingestion of: **{pdf_name}**\n\nThis may take 10-20 minutes depending on document size..."

    try:
        count = ingest_pdf(
            pdf_file_path=pdf_path,
            index_name="localrag",
            use_gemini_vlm=use_gemini_vlm,
        )
        vlm_note = (
            "Gemini VLM was used for image captioning (better semantic quality, uses API credits)."
            if use_gemini_vlm
            else "Raw OCR text was used for images (faster, no API cost)."
        )
        yield f"✅ Successfully ingested **{count} chunks** from `{pdf_name}`\n\n{vlm_note}\n\nYou can now ask questions about this document in the Q&A tab!"
    except Exception as e:
        yield f"❌ Ingestion failed:\n```\n{str(e)}\n```\n\nMake sure OpenSearch and Ollama are running."


# ── Build UI ──────────────────────────────────────────────────────────────────
with gr.Blocks(title="Multimodal RAG System", theme="soft") as demo:
    gr.Markdown("# 📚 Multimodal RAG System")
    gr.Markdown("Upload any PDF and ask questions using RAG — with RAGAS evaluation scores.")

    with gr.Tabs():

        # ── Tab 1: Ask Questions ──────────────────────────────────────────────
        with gr.Tab("💬 Ask Questions"):
            with gr.Row():
                with gr.Column(scale=1):
                    query_input = gr.Textbox(
                        label="Your Question",
                        placeholder="Ask anything about your uploaded document...",
                        lines=4,
                    )

                    with gr.Row():
                        search_type = gr.Radio(
                            ["keyword", "semantic", "hybrid"],
                            label="Search Method",
                            value="hybrid",
                            info="hybrid = best results",
                        )
                        model_type = gr.Radio(
                            ["gemini", "ollama"],
                            label="AI Model",
                            value="gemini",
                        )

                    with gr.Row():
                        stream_checkbox = gr.Checkbox(
                            label="Stream Response", value=True
                        )
                        eval_checkbox = gr.Checkbox(
                            label="Run RAGAS Evaluation",
                            value=False,
                            info="Scores faithfulness, relevancy & context precision",
                        )

                    submit_btn = gr.Button("Generate Answer", variant="primary")

                with gr.Column(scale=2):
                    output = gr.Markdown(label="Answer")
                    eval_output = gr.Markdown(label="Evaluation Scores")

            submit_btn.click(
                on_submit,
                inputs=[query_input, search_type, model_type, stream_checkbox, eval_checkbox],
                outputs=[output, eval_output],
                show_progress="minimal",
            )

            gr.Examples(
                examples=[
                    ["What is RAG?", "hybrid", "gemini", True, False],
                    ["What are the benefits of RAG vs fine-tuning?", "semantic", "gemini", True, True],
                    ["Explain the RAG architecture diagram", "hybrid", "gemini", False, True],
                    ["What are common challenges in RAG?", "keyword", "gemini", False, False],
                ],
                inputs=[query_input, search_type, model_type, stream_checkbox, eval_checkbox],
            )

            gr.Markdown("""
### 📘 Search Methods
| Method | How it works |
|--------|-------------|
| **Keyword** | Traditional text matching |
| **Semantic** | Vector similarity (meaning-based) |
| **Hybrid** | Best of both — recommended |

### 📊 RAGAS Metrics
| Metric | What it measures |
|--------|----------------|
| **Faithfulness** | Is the answer grounded in retrieved context? |
| **Answer Relevancy** | Does the answer address the question? |
| **Context Precision** | Are retrieved chunks relevant to the question? |
""")

        # ── Tab 2: Upload PDF ─────────────────────────────────────────────────
        with gr.Tab("📤 Upload & Ingest PDF"):
            gr.Markdown("### Upload any PDF to index it into the RAG system")
            gr.Markdown("The system will extract text, images, and tables — then make everything searchable.")

            with gr.Row():
                with gr.Column(scale=1):
                    pdf_upload = gr.File(
                        label="Upload PDF",
                        file_types=[".pdf"],
                        type="filepath",
                    )

                    use_gemini_vlm = gr.Checkbox(
                        label="Use Gemini VLM for image captioning",
                        value=True,
                        info="✅ Better semantic understanding of figures | ❌ Uses API credits (~$0.0002/image) + ~2-4s per image",
                    )

                    ingest_btn = gr.Button("🚀 Start Ingestion", variant="primary")

                with gr.Column(scale=1):
                    ingest_output = gr.Markdown(label="Ingestion Status")

            ingest_btn.click(
                on_ingest,
                inputs=[pdf_upload, use_gemini_vlm],
                outputs=ingest_output,
            )

            gr.Markdown("""
### 🔍 Design Choice: Gemini as VLM for Image Captioning

This system uses **Gemini-1.5-Flash as a Vision-Language Model (VLM)** to generate rich semantic descriptions of figures, diagrams, and charts found in PDFs.

| | Gemini VLM ✅ | Raw OCR Only ❌ |
|---|---|---|
| **Quality** | Rich natural language descriptions | Raw extracted text only |
| **Search** | Figures become semantically searchable | Poor searchability for visual content |
| **Cost** | ~$0.0002 per image (Gemini API) | Free |
| **Speed** | ~2-4 seconds per image | Instant |
| **Best for** | Research papers, technical docs | Text-heavy documents |

**Verdict**: For documents with meaningful figures (architecture diagrams, charts, results tables), Gemini VLM significantly improves retrieval quality and is worth the small API cost.
""")

# ── Launch ────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    demo.queue().launch()
