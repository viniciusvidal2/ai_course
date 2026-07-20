# RAG Development Practice

This folder contains scripts to build and query a local Retrieval-Augmented Generation (RAG) assistant using **ChromaDB** as the vector database, **PyMuPDF** (`fitz`) to parse PDFs, and a local **Ollama** instance to run LLMs.

---

## 1) Project Structure

- `create_db.py`: Extracts text from PDF files, chunks the text, and stores/indexes the chunks in a ChromaDB database using ChromaDB's default embedding model.
- `assistant.py`: Retrieves relevant context chunks from the ChromaDB database for a user query, constructs a prompt, queries an Ollama LLM, and prints the outputs.
- `requirements.txt`: Python package dependencies.
- `Dockerfile`: Docker image template to package and run these scripts.

---

## 2) Running Locally

### A. Install Dependencies

Install the required python packages:
```bash
pip install -r requirements.txt
```

### B. Index your PDFs

To load, parse, and index PDF documents into the ChromaDB collection:
```bash
python create_db.py \
  --pdf-dir /path/to/pdfs \
  --db-path ./chroma_db \
  --chunk-size 500
```

- `--pdf-dir`: Directory containing your `.pdf` documents.
- `--db-path`: Target folder where ChromaDB will persist its database files.
- `--chunk-size` *(Optional, default: 500)*: Text chunk size limit in characters.

### C. Run the Assistant Query

Ask a question using Ollama:
```bash
python assistant.py \
  --db-path ./chroma_db \
  --model llama3.2:1b \
  --query "What are the main findings in the document?"
```

- `--db-path`: The directory containing your ChromaDB files.
- `--model`: The model name you have loaded in Ollama.
- `--query`: Your question.
- `--ollama-url` *(Optional, default: http://localhost:11434)*: The endpoint for Ollama API.

---

## 3) Docker Usage

### Build the Image
```bash
docker build -t rag-assistant .
```

### Run the Container
```bash
docker run -d --name rag-assistant-container rag-assistant
```

### Exec into the Container
```bash
docker exec -it rag-assistant-container bash
```

From inside the container, you can run the same scripts. 

> **Docker networking note**:
> - On Linux, if your Ollama service is running on the host, you can run the container using `--network host` to allow easy connection to `http://localhost:11434`.
> - On macOS or Windows, use `--ollama-url http://host.docker.internal:11434` when invoking `assistant.py`.
