FROM python:3.10-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY create_db.py assistant.py ./

CMD ["sleep", "infinity"]
