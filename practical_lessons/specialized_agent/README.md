# Specialized Agent Practice

This folder contains a Python chatbot script that uses the OpenAI API client against a local Ollama OpenAI-compatible endpoint.

## 1) Create Python 3.10 environment with Anaconda

```bash
conda create -n specialized-agent python=3.10 -y
conda activate specialized-agent
pip install -r requirements.txt
```

## 2) Configure `.env` file

Create a `.env` file in this folder:

```env
OPENAI_API_KEY=ollama
```

For local Ollama, this key is only a placeholder for the OpenAI client.

## 3) Install and run Ollama (Linux and Windows)

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

## 4) Script: `specialized_agent.py`

### Intent
Run an interactive chatbot loop with custom personality and language, while keeping chat history and tracking user input tokens.

### Arguments
- `--personality` (required): personality/instructions for the assistant
- `--temperature` (optional, default `0.7`): sampling temperature
- `--output-tokens` (optional, default `256`): max output tokens per response
- `--input-tokens` (optional, default `512`): max allowed input tokens per user message
- `--language` (optional, default `english`): response language
- `--model` (optional, default `llama3.2:1b`): local Ollama model
- `--base-url` (optional, default `http://localhost:11434/v1`): OpenAI-compatible Ollama endpoint

### Run example

```bash
python specialized_agent.py \
  --personality "You are a patient Python mentor" \
  --temperature 0.6 \
  --output-tokens 300 \
  --input-tokens 400 \
  --language english
```

### Behavior notes
- The first system message combines personality, language, and base assistant behavior.
- The script keeps a full `messages` history (`system`, `user`, `assistant`) in memory.
- At each user step, input token count is displayed before sending to the model.
- Press `Ctrl+C` to exit the loop safely.

## 5) Docker usage

### Build image

```bash
docker build -t specialized-agent .
```

### Run container in background

```bash
docker run -d --name specialized-agent-container --env-file .env specialized-agent
```

### Exec into running container

```bash
docker exec -it specialized-agent-container bash
```

### Run script inside container

```bash
python specialized_agent.py \
  --personality "You are a concise AI tutor" \
  --temperature 0.7 \
  --output-tokens 256 \
  --input-tokens 512 \
  --language english
```

> Notes:
> - Ollama must be reachable from the container.
> - On Linux, you can add `--network host` when running the container if Ollama is on the host machine.
> - On Windows/macOS Docker Desktop, use `--base-url http://host.docker.internal:11434/v1` when running the script if needed.
