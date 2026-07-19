# Tool Usage Practice (PDF Agent with Ollama)

This folder contains a Python script that runs an agentic loop using Ollama and two PDF tools:
- extract relevant PDF content and save it as Markdown in the same PDF folder.
- extract PDF images and save them as PNG files in the same PDF folder.

## 1) Create Python 3.10 environment with Anaconda

```bash
conda create -n tool-usage python=3.10 -y
conda activate tool-usage
pip install -r requirements.txt
```

## 2) Install and run Ollama (Linux and Windows)

### Linux
1. Install:
   ```bash
   curl -fsSL https://ollama.com/install.sh | sh
   ```
2. Start Ollama service:
   ```bash
   ollama serve
   ```
3. Pull model (`llama3.2:1b`, suitable for ~8GB RAM environments):
   ```bash
   ollama pull llama3.2:1b
   ```

### Windows
1. Download and install Ollama from: https://ollama.com/download/windows
2. Open PowerShell and verify:
   ```powershell
   ollama --version
   ```
3. Start Ollama (if not already running from tray/app):
   ```powershell
   ollama serve
   ```
4. Pull model:
   ```powershell
   ollama pull llama3.2:1b
   ```

## 3) Script: `pdf_tool_agent.py`

### Intent
Run an agentic main LLM loop that decides when to execute tools until the user goal is reached.

### Arguments
- `--model` (optional, default `llama3.2:1b`): main agent model
- `--selector-model` (optional, default `llama3.2:1b`): internal model for section selection tool
- `--pdf-path` (required): path to input PDF
- `--query` (required): user objective
- `--host` (optional, default `http://localhost:11434`): Ollama URL
- `--max-steps` (optional, default `8`): max tool loop iterations

### Tool behavior
- `extract_content_to_markdown(pdf_path, formatted_query)`:
  - reads and formats PDF text by page.
  - calls an internal LLM with a system prompt + user query + PDF content.
  - saves selected content to `<pdf_name>_selected_content.md` in the same folder.
- `extract_images_from_pdf(pdf_path)`:
  - extracts all PDF images.
  - saves each image as `<pdf_name>_page_<n>_image_<m>.png` in the same folder.

### Agentic loop details
- Runs in a `while`-style iterative loop (`for` bounded by `--max-steps`).
- Tracks and prints function execution results every step.
- Always prints the model last response before the next action.
- Stops when model returns `done` action.

### Run example

```bash
python pdf_tool_agent.py \
  --model llama3.2:1b \
  --selector-model llama3.2:1b \
  --pdf-path /absolute/path/to/file.pdf \
  --query "Find the exact section that explains the evaluation methodology and extract all figures." \
  --host http://localhost:11434 \
  --max-steps 8
```

## 4) Docker usage

### Build image

```bash
docker build -t tool-usage .
```

### Run container in background

```bash
docker run -d --name tool-usage-container tool-usage
```

### Exec into running container

```bash
docker exec -it tool-usage-container bash
```

### Run script inside container

```bash
python pdf_tool_agent.py \
  --model llama3.2:1b \
  --selector-model llama3.2:1b \
  --pdf-path /data/your.pdf \
  --query "Extract the exact part about deployment constraints and all images."
```

> Notes:
> - Ollama must be reachable from the container.
> - On Linux, you can add `--network host` if Ollama is running on the host.
> - On Windows/macOS Docker Desktop, use `--host http://host.docker.internal:11434` if needed.
