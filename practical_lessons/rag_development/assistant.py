import argparse
from pathlib import Path
import requests
import chromadb
from chromadb.utils import embedding_functions


def query_ollama(api_url: str, model: str, system_prompt: str, user_prompt: str) -> str:
    url = f"{api_url.rstrip('/')}/api/generate"
    payload = {
        "model": model,
        "prompt": f"System: {system_prompt}\n\nUser: {user_prompt}",
        "stream": False
    }
    try:
        response = requests.post(url, json=payload, timeout=60)
        response.raise_for_status()
        return response.json().get("response", "")
    except requests.exceptions.RequestException as e:
        return f"Error querying Ollama API: {e}"


def main():
    parser = argparse.ArgumentParser(
        description="RAG Assistant querying ChromaDB and Ollama.")
    parser.add_argument("--db-path", required=True, type=str,
                        help="Path to the ChromaDB database.")
    parser.add_argument("--model", required=True, type=str,
                        help="Ollama model to use.")
    parser.add_argument("--query", required=True, type=str, help="User query.")
    parser.add_argument("--ollama-url", type=str, default="http://localhost:11434",
                        help="Ollama API base URL (default: http://localhost:11434).")
    parser.add_argument("--num-results", type=int, default=3,
                        help="Number of retrieved chunks (default: 3).")
    args = parser.parse_args()

    db_path = Path(args.db_path)
    if not db_path.exists():
        print(f"Error: ChromaDB directory '{db_path}' does not exist.")
        return

    # Initialize Chroma client
    client = chromadb.PersistentClient(path=str(db_path))
    emb_fn = embedding_functions.DefaultEmbeddingFunction()

    try:
        collection = client.get_collection(
            name="pdf_documents", embedding_function=emb_fn)
    except Exception as e:
        print(
            f"Error: Could not retrieve collection 'pdf_documents' from database. Has it been initialized? Detail: {e}")
        return

    # Query ChromaDB
    print(f"Retrieving context for query: '{args.query}'...")
    results = collection.query(
        query_texts=[args.query],
        n_results=args.num_results
    )

    retrieved_chunks = []
    if results and 'documents' in results and results['documents']:
        docs = results['documents'][0]
        metadatas = results['metadatas'][0] if 'metadatas' in results and results['metadatas'] else [
        ]
        for i, doc in enumerate(docs):
            meta = metadatas[i] if i < len(metadatas) else {}
            source = meta.get("source", "Unknown")
            page = meta.get("page", "Unknown")
            retrieved_chunks.append({
                "text": doc,
                "source": source,
                "page": page
            })

    # Prepare context for the prompt
    context_str = ""
    for i, chunk in enumerate(retrieved_chunks, start=1):
        context_str += f"[Chunk {i}] (Source: {chunk['source']}, Page: {chunk['page']})\n{chunk['text']}\n\n"

    system_prompt = (
        "You are a helpful assistant. Use the following retrieved document context chunks to answer the user query. "
        "If the answer cannot be found in the context, state that you do not know based on the provided documents. "
        "Do not invent information outside the context."
    )
    user_prompt = f"Context:\n{context_str}\nQuery: {args.query}"

    print("Querying Ollama model...")
    response_text = query_ollama(
        args.ollama_url, args.model, system_prompt, user_prompt)

    # Output details at the end
    print("\n" + "="*80)
    print("USER QUERY:")
    print(args.query)
    print("="*80)
    print("RETRIEVED CHUNKS:")
    if not retrieved_chunks:
        print("No context chunks retrieved.")
    for i, chunk in enumerate(retrieved_chunks, start=1):
        print(f"\n--- Chunk {i} ---")
        print(f"Source: {chunk['source']} | Page: {chunk['page']}")
        print(chunk['text'])
    print("="*80)
    print("ASSISTANT RESPONSE:")
    print(response_text)
    print("="*80)


if __name__ == "__main__":
    main()
