"""
generation.py — Upgraded to return retrieved contexts alongside
the answer, enabling RAGAS evaluation scores.
"""

import json
import os

import google.generativeai as genai
import requests
from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate

from retrieval import hybrid_search, keyword_search, semantic_search

load_dotenv()

gemini_api_key = os.getenv("GEMINI_API_KEY")
if not gemini_api_key:
    print("ERROR: GEMINI_API_KEY not found in environment variables")
else:
    print(f"Configuring Gemini with API key: {gemini_api_key[:5]}...")
    genai.configure(api_key=gemini_api_key)

RAG_PROMPT_TEMPLATE = """
You are an AI assistant helping answer questions about documents indexed in this RAG system.
Use the following retrieved documents to answer the user's question.
If the retrieved documents don't contain relevant information, say that you don't know.

RETRIEVED DOCUMENTS:
{context}

USER QUESTION:
{question}

YOUR ANSWER (be comprehensive, accurate, and helpful):
"""

prompt = PromptTemplate(
    input_variables=["context", "question"],
    template=RAG_PROMPT_TEMPLATE,
)


def generate_with_gemini(prompt_text, model_name="gemini-2.0-flash", stream=False):
    """Generate response using Google's Gemini model."""
    try:
        model = genai.GenerativeModel(model_name)

        if len(prompt_text) > 30000:
            prompt_text = prompt_text[:30000] + "...[truncated due to length]"

        generation_config = {
            "temperature": 0.7,
            "top_p": 0.95,
            "top_k": 40,
            "max_output_tokens": 2048,
        }

        safety_settings = {
            "harassment": "block_none",
            "hate": "block_none",
            "sexual": "block_none",
            "dangerous": "block_none",
        }

        if stream:
            response_generator = model.generate_content(
                contents=prompt_text,
                generation_config=generation_config,
                safety_settings=safety_settings,
                stream=True,
            )
            for chunk in response_generator:
                if hasattr(chunk, "text") and chunk.text:
                    yield chunk.text
                elif hasattr(chunk, "parts"):
                    for part in chunk.parts:
                        if hasattr(part, "text") and part.text:
                            yield part.text
        else:
            response = model.generate_content(
                contents=prompt_text,
                generation_config=generation_config,
                safety_settings=safety_settings,
            )
            if hasattr(response, "text"):
                return response.text
            elif hasattr(response, "parts") and response.parts:
                return "".join([p.text for p in response.parts if hasattr(p, "text")])
            else:
                return f"Response received but couldn't extract text: {str(response)}"

    except Exception as e:
        import traceback
        error_msg = f"Error with Gemini generation: {str(e)}\n{traceback.format_exc()}"
        print(error_msg)
        if stream:
            yield error_msg
        else:
            return error_msg


def generate_with_ollama(prompt_text, model_name="deepseek-r1:1.5b", stream=False):
    """Generate response using Ollama with Deepseek model."""
    try:
        url = "http://localhost:11434/api/generate"
        data = {
            "model": model_name,
            "prompt": prompt_text,
            "stream": stream,
            "options": {"temperature": 0.7},
        }

        if stream:
            response = requests.post(url, json=data, stream=True)
            response.raise_for_status()
            for line in response.iter_lines():
                if line:
                    try:
                        chunk = json.loads(line.decode("utf-8"))
                        if "response" in chunk:
                            yield chunk["response"]
                    except json.JSONDecodeError:
                        continue
        else:
            response = requests.post(url, json=data)
            response.raise_for_status()
            return response.json().get("response", "No response generated")
    except Exception as e:
        error_msg = f"Error generating response with Ollama: {str(e)}"
        if stream:
            yield error_msg
        else:
            return error_msg


def generate_rag_response(
    query, search_type="hybrid", top_k=5, model_type="gemini", stream=False,
    return_contexts=False   # NEW: set True to also get contexts back for evaluation
):
    """
    Generate RAG response using retrieved chunks.

    Args:
        query:           User query
        search_type:     Type of search (keyword, semantic, hybrid)
        top_k:           Number of chunks to retrieve
        model_type:      Type of model to use (gemini, ollama)
        stream:          Whether to stream the response
        return_contexts: If True, returns (answer, contexts) tuple instead of just answer
                         Used by evaluation.py to compute RAGAS metrics

    Returns:
        answer string, or (answer, contexts) if return_contexts=True
        Generator if stream=True
    """
    try:
        # Step 1: Retrieve
        if search_type == "keyword":
            results = keyword_search(query, top_k=top_k)
        elif search_type == "semantic":
            results = semantic_search(query, top_k=top_k)
        else:
            results = hybrid_search(query, top_k=top_k)

        if not results:
            message = "No relevant information found. Please try a different search type or refine your question."
            if stream:
                yield message
                return
            else:
                return (message, []) if return_contexts else message

        # Step 2: Format contexts
        contexts = []
        context_strings = []
        for i, hit in enumerate(results):
            source = hit["_source"]
            content = source.get("content", "")
            content_type = source.get("content_type", "unknown")
            source_pdf = source.get("source_pdf", "")

            metadata_info = ""
            if "metadata" in source and source["metadata"]:
                if source["metadata"].get("caption"):
                    metadata_info += f"\nCaption: {source['metadata']['caption']}"

            context_entry = f"[Document {i+1} - {content_type}]{metadata_info}\n{content}"
            contexts.append(context_entry)
            context_strings.append(content)   # clean text for evaluation

        # Step 3: Build prompt
        context_text = "\n\n---\n\n".join(contexts)
        prompt_text = prompt.format(context=context_text, question=query)

        # Step 4: Generate
        if stream:
            # Streaming can't return contexts tuple, yield answer only
            if model_type == "gemini":
                yield from generate_with_gemini(prompt_text, stream=True)
            else:
                yield from generate_with_ollama(prompt_text, stream=True)
        else:
            if model_type == "gemini":
                answer = generate_with_gemini(prompt_text, stream=False)
            else:
                answer = generate_with_ollama(prompt_text, stream=False)

            if return_contexts:
                return answer, context_strings
            else:
                return answer

    except Exception as e:
        error_message = f"Error in RAG process: {str(e)}"
        if stream:
            yield error_message
        else:
            return (error_message, []) if return_contexts else error_message


if __name__ == "__main__":
    query = "How does RAG work?"
    print("Streaming response:")
    for chunk in generate_rag_response(query, "hybrid", 3, "gemini", True):
        print(chunk, end="", flush=True)
