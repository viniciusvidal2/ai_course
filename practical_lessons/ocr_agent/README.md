# OCR Agent (Visual OCR with Ollama)

This folder contains a Python script that executes an OCR task on a target image utilizing Ollama's vision/multimodal models.

## 1) Create Python Environment and Install Dependencies

```bash
# Example with Anaconda
conda create -n ocr-agent python=3.10 -y
conda activate ocr-agent
pip install -r requirements.txt
```

## 2) Running the Script

### Arguments
- `--image-path` (required): The path to the input image file (supports `.png`, `.jpg`, `.jpeg`, `.webp`).
- `--model` (required): The Ollama multimodal/vision model name (e.g., `llama3.2-vision` or `llava`).
- `--host` (optional, default `http://localhost:11434`): The Ollama server endpoint.

### Run Example

```bash
python ocr_agent.py \
  --image-path /absolute/path/to/receipt.png \
  --model llama3.2-vision \
  --host http://localhost:11434
```

The script will:
1. Load the image and encode it to base64 format.
2. Send the image to the designated multimodal Ollama model.
3. Save the returned OCR output formatted in clean Markdown to `/path/to/receipt_ocr.md`.

## 3) Docker Usage

### Build Image
```bash
docker build -t ocr-agent .
```

### Run Container in Background
```bash
docker run -d --name ocr-agent-container ocr-agent
```

### Exec into Container and Run
```bash
docker exec -it ocr-agent-container bash
```

```bash
python ocr_agent.py \
  --image-path /data/receipt.png \
  --model llama3.2-vision
```
