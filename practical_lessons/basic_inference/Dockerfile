FROM python:3.10-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY provider_inference.py ollama_inference.py ./

CMD ["sleep", "infinity"]
