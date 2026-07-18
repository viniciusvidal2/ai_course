# Basic Inference Practice

This folder contains two simple Python scripts for LLM inference:
- `provider_inference.py`: inference with OpenAI or Anthropic (chosen by argument).
- `ollama_inference.py`: local inference using Ollama with the lightest Llama 3.2 model (`llama3.2:1b` by default).

## 1) Create Python 3.10 environment with Anaconda

```bash
conda create -n basic-inference python=3.10 -y
conda activate basic-inference
pip install -r requirements.txt
```

## 2) Configure `.env` file

Create a `.env` file in this folder:

```env
OPENAI_API_KEY=your_openai_key_here
ANTHROPIC_API_KEY=your_anthropic_key_here
```

The script loads these values using `python-dotenv`.

## 3) Script: `provider_inference.py`

### Intent
Run one inference call using either OpenAI or Anthropic.

### Arguments
- `--provider` (required): `openai` or `anthropic`
- `--prompt` (required): prompt text
- `--model` (optional): provider model name  
  - default for OpenAI: `gpt-4o-mini`
  - default for Anthropic: `claude-3-5-haiku-latest`
- `--max-tokens` (optional, default `256`)
- `--temperature` (optional, default `0.7`)

### Run examples

```bash
python provider_inference.py --provider openai --prompt "Explain transformers in one paragraph."
python provider_inference.py --provider anthropic --prompt "Give me 3 practical AI project ideas."
```

## 4) Install and run Ollama (Linux and Windows)

### Linux
1. Install:
   ```bash
   curl -fsSL https://ollama.com/install.sh | sh
   ```
2. Start Ollama service:
   ```bash
   ollama serve
   ```
3. Pull model:
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

## 5) Script: `ollama_inference.py`

### Intent
Run local Llama inference through the Ollama Python API.

### Arguments
- `--prompt` (required): prompt text
- `--model` (optional, default `llama3.2:1b`)
- `--host` (optional, default `http://localhost:11434`)

### Run example

```bash
python ollama_inference.py --prompt "Summarize retrieval-augmented generation in 5 bullet points."
```

## 6) Docker usage

### Build image

```bash
docker build -t basic-inference .
```

### Run container in background

```bash
docker run -d --name basic-inference-container --env-file .env basic-inference
```

### Exec into running container

```bash
docker exec -it basic-inference-container bash
```

### Run scripts inside container

```bash
python provider_inference.py --provider openai --prompt "What is zero-shot prompting?"
python ollama_inference.py --prompt "What is chain-of-thought?"
```

> Notes:
> - For `ollama_inference.py`, Ollama must be reachable from the container.  
> - On Linux, you can add `--network host` when running the container if Ollama is on the host machine.
> - On Windows/macOS Docker Desktop, set `--host http://host.docker.internal:11434` when running the script if needed.
