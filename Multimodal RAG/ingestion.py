"""
ingestion.py — Upgraded to accept arbitrary PDF uploads
instead of hardcoding a single paper path.

Design choice: Gemini-as-VLM for image captioning
  ✅ Pros: Rich semantic understanding of figures, diagrams, architecture visuals
  ❌ Cons: API cost per image (~$0.0002/image) + latency (~2-4s/image)
  Tradeoff: Worth it for technical PDFs with meaningful figures.
             For cost-sensitive use, set use_gemini=False to fall back to raw OCR text.
"""

import os


def create_index_if_not_exists(client, index_name):
    """
    Create an OpenSearch index with proper mapping for vector search if it doesn't exist.
    """
    if client.indices.exists(index=index_name):
        print(f"Deleting existing index '{index_name}' to recreate with proper mappings...")
        client.indices.delete(index=index_name)

    from helper import get_embedding
    sample_embedding = get_embedding("Sample text for dimension detection")
    dimension = len(sample_embedding)
    print(f"Using embedding dimension: {dimension}")

    mappings = {
        "mappings": {
            "properties": {
                "content": {"type": "text"},
                "content_type": {"type": "keyword"},
                "token_count": {"type": "integer"},
                "embedding": {"type": "knn_vector", "dimension": dimension},
                "base64_image": {"type": "binary", "doc_values": False, "index": False},
                "table_html": {"type": "text", "index": False},
                "source_pdf": {"type": "keyword"},   # NEW: track which PDF this came from
                "metadata": {
                    "properties": {
                        "filename": {"type": "keyword"},
                        "caption": {"type": "text"},
                        "image_text": {"type": "text"},
                    }
                },
            }
        },
        "settings": {
            "index": {
                "knn": True,
                "knn.space_type": "cosinesimil",
            }
        },
    }

    try:
        client.indices.create(index=index_name, body=mappings)
        print(f"Created index '{index_name}' with vector search capabilities.")
    except Exception as e:
        print(f"Error creating index: {e}")
        raise


def prepare_chunks_for_ingestion(chunks, source_pdf=""):
    """
    Prepare chunks for ingestion by adding embeddings, token counts,
    and source PDF tracking.
    """
    from helper import get_embedding, get_token_count

    prepared_chunks = []

    for i, chunk in enumerate(chunks):
        try:
            if not chunk.get("content"):
                continue

            embedding = get_embedding(chunk["content"])
            token_count = get_token_count(chunk["content"])

            ingestion_doc = {
                "content": chunk["content"],
                "content_type": chunk.get("content_type", "text"),
                "token_count": token_count,
                "embedding": embedding,
                "source_pdf": source_pdf,   # NEW: track source PDF
                "metadata": {
                    "filename": chunk.get("filename", ""),
                    "caption": chunk.get("caption", ""),
                    "image_text": chunk.get("image_text", ""),
                },
            }

            if chunk.get("content_type") == "image" and "base64_image" in chunk:
                ingestion_doc["base64_image"] = chunk["base64_image"]

            if chunk.get("content_type") == "table" and "table_as_html" in chunk:
                ingestion_doc["table_html"] = chunk["table_as_html"]

            prepared_chunks.append(ingestion_doc)

            if (i + 1) % 10 == 0:
                print(f"Prepared {i+1}/{len(chunks)} chunks")

        except Exception as e:
            print(f"Error preparing chunk: {str(e)}")

    print(f"Successfully prepared {len(prepared_chunks)} chunks for ingestion")
    return prepared_chunks


def ingest_chunks_into_opensearch(client, index_name, chunks):
    """Ingest prepared chunks into OpenSearch using bulk API."""
    from opensearchpy.helpers import bulk

    successful = 0
    failed = 0
    operations = []

    for i, chunk in enumerate(chunks):
        operations.append({"_index": index_name, "_source": chunk})

        if (i + 1) % 100 == 0 or i == len(chunks) - 1:
            try:
                success, failed_items = bulk(client, operations, stats_only=True)
                successful += success
                failed += len(operations) - success
                operations = []
                print(f"Ingested {successful} chunks so far ({failed} failed)")
            except Exception as e:
                print(f"Bulk ingestion error: {str(e)}")
                failed += len(operations)
                operations = []

    if operations:
        try:
            success, failed_items = bulk(client, operations, stats_only=True)
            successful += success
            failed += len(operations) - success
        except Exception as e:
            print(f"Bulk ingestion error: {str(e)}")
            failed += len(operations)

    print(f"Ingestion complete: {successful} successful, {failed} failed")
    return successful


def ingest_pdf(pdf_file_path, index_name="localrag", use_gemini_vlm=True):
    """
    NEW: Main entry point — accepts any PDF file path.

    Args:
        pdf_file_path: Path to any PDF file (not hardcoded anymore)
        index_name: OpenSearch index name (default: localrag)
        use_gemini_vlm: Use Gemini for image captioning (True = better quality,
                        False = use raw OCR text only, cheaper & faster)

    Design choice — Gemini as VLM for image captioning:
        ✅ Better semantic understanding of technical figures and architecture diagrams
        ✅ Generates searchable natural language descriptions of visual content
        ❌ Costs ~$0.0002 per image via Gemini API
        ❌ Adds ~2-4 seconds latency per image during ingestion
        Verdict: Use True for research papers / technical docs with meaningful figures.
                 Use False for text-heavy documents or cost-sensitive pipelines.
    """
    from unstructured.partition.pdf import partition_pdf
    from chunking import (
        create_semantic_chunks,
        process_images_with_captions,
        process_tables_with_descriptions,
    )
    from helper import get_opensearch_client

    if not os.path.exists(pdf_file_path):
        raise FileNotFoundError(f"PDF not found: {pdf_file_path}")

    source_name = os.path.basename(pdf_file_path)
    print(f"\n{'='*60}")
    print(f"Processing PDF: {source_name}")
    print(f"Gemini VLM for images: {use_gemini_vlm}")
    print(f"{'='*60}\n")

    # 1. Extract raw chunks (hi_res strategy for images + tables)
    print("Step 1/6: Extracting raw elements from PDF...")
    raw_chunks = partition_pdf(
        filename=pdf_file_path,
        strategy="hi_res",
        infer_table_structure=True,
        extract_image_block_types=["Image", "Figure", "Table"],
        extract_image_block_to_payload=True,
        chunking_strategy=None,
    )
    print(f"  Extracted {len(raw_chunks)} raw elements")

    # 2. Process images with Gemini VLM (or fallback to OCR text)
    print(f"\nStep 2/6: Processing images (Gemini VLM={use_gemini_vlm})...")
    if use_gemini_vlm:
        print("  [Design choice] Using Gemini-1.5-Flash as VLM for rich figure descriptions.")
        print("  Tradeoff: Better semantic search quality vs. API cost + ~2-4s per image.")
    else:
        print("  [Design choice] Skipping Gemini VLM — using raw OCR text only.")
        print("  Tradeoff: Faster & cheaper, but figures may have poor search quality.")

    processed_images, image_errors = process_images_with_captions(
        raw_chunks, use_gemini=use_gemini_vlm
    )
    print(f"  Processed {len(processed_images)} images, {len(image_errors)} errors")

    # 3. Process tables
    print("\nStep 3/6: Processing tables...")
    processed_tables, table_errors = process_tables_with_descriptions(
        raw_chunks, use_gemini=True, use_ollama=False
    )
    print(f"  Processed {len(processed_tables)} tables, {len(table_errors)} errors")

    # 4. Create semantic text chunks
    print("\nStep 4/6: Creating semantic text chunks...")
    chunks = partition_pdf(
        filename=pdf_file_path,
        strategy="hi_res",
        chunking_strategy="by_title",
        max_characters=2000,
        min_chars_to_combine=500,
        chars_before_new_chunk=1500,
    )
    semantic_chunks = create_semantic_chunks(chunks)
    print(f"  Created {len(semantic_chunks)} semantic chunks")

    # 5. Connect to OpenSearch and set up index
    print("\nStep 5/6: Connecting to OpenSearch...")
    client = get_opensearch_client("localhost", 9200)
    create_index_if_not_exists(client, index_name)

    # 6. Prepare and ingest
    print("\nStep 6/6: Ingesting into OpenSearch...")
    all_chunks = processed_images + processed_tables + semantic_chunks
    print(f"  Total chunks: {len(all_chunks)}")

    prepared_chunks = prepare_chunks_for_ingestion(all_chunks, source_pdf=source_name)
    successful_count = ingest_chunks_into_opensearch(client, index_name, prepared_chunks)

    print(f"\n{'='*60}")
    print(f"✅ Done! Ingested {successful_count} chunks from '{source_name}'")
    print(f"{'='*60}\n")
    return successful_count


# ── Legacy compatibility: still works if called directly ──────────────────────
if __name__ == "__main__":
    import sys

    if len(sys.argv) > 1:
        # Accept PDF path as command line argument:
        # python ingestion.py path/to/your/paper.pdf
        pdf_path = sys.argv[1]
    else:
        # Default fallback for backwards compatibility
        pdf_path = "files/2312.10997v5.pdf"

    ingest_pdf(pdf_path)
