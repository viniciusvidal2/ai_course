import argparse
import os
from pathlib import Path
import fitz  # PyMuPDF
import chromadb
from chromadb.utils import embedding_functions


def chunk_text(text: str, chunk_size: int, overlap: int) -> list[str]:
    chunks = []
    start = 0
    while start < len(text):
        end = start + chunk_size
        chunks.append(text[start:end])
        # Move forward by chunk_size minus overlap
        start += chunk_size - overlap
    return chunks


def main():
    parser = argparse.ArgumentParser(
        description="Create a ChromaDB database from a folder of PDFs.")
    parser.add_argument("--pdf-dir", required=True, type=str,
                        help="Path to the directory containing PDF files.")
    parser.add_argument("--db-path", required=True, type=str,
                        help="Path to save the ChromaDB database.")
    parser.add_argument("--chunk-size", type=int, default=500,
                        help="Chunk size for splitting text (default: 500 characters).")
    args = parser.parse_args()

    pdf_dir = Path(args.pdf_dir)
    db_path = Path(args.db_path)
    chunk_size = args.chunk_size
    overlap = max(1, chunk_size // 10)

    if not pdf_dir.exists() or not pdf_dir.is_dir():
        print(
            f"Error: PDF directory '{pdf_dir}' does not exist or is not a directory.")
        return

    # Initialize Chroma client
    client = chromadb.PersistentClient(path=str(db_path))

    # Use default embedding function (sentence-transformers/all-MiniLM-L6-v2)
    emb_fn = embedding_functions.DefaultEmbeddingFunction()

    collection = client.get_or_create_collection(
        name="pdf_documents",
        embedding_function=emb_fn
    )

    pdf_files = list(pdf_dir.glob("*.pdf"))
    if not pdf_files:
        print(f"No PDF files found in '{pdf_dir}'.")
        return

    print(f"Found {len(pdf_files)} PDF files to process.")

    doc_id_counter = 0
    for pdf_path in pdf_files:
        print(f"Processing '{pdf_path.name}'...")
        try:
            with fitz.open(pdf_path) as doc:
                for page_num, page in enumerate(doc, start=1):
                    text = page.get_text("text").strip()
                    if not text:
                        continue

                    # Split page text into chunks
                    chunks = chunk_text(text, chunk_size, overlap)
                    for chunk_idx, chunk in enumerate(chunks):
                        doc_id = f"{pdf_path.stem}_p{page_num}_c{chunk_idx}_{doc_id_counter}"
                        doc_id_counter += 1

                        collection.add(
                            documents=[chunk],
                            metadatas=[{
                                "source": pdf_path.name,
                                "page": page_num,
                                "chunk": chunk_idx
                            }],
                            ids=[doc_id]
                        )
        except Exception as e:
            print(f"Failed to process '{pdf_path.name}': {e}")

    print(
        f"Successfully processed PDFs. Database created at '{db_path}'. Total chunks indexed: {doc_id_counter}")


if __name__ == "__main__":
    main()
