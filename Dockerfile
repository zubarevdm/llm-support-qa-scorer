FROM python:3.12-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

EXPOSE 8000

# Serve the scoring API. Set LLM_API_KEY at runtime to enable the real LLM judge.
CMD ["uvicorn", "support_qa.api:app", "--host", "0.0.0.0", "--port", "8000", "--app-dir", "src"]
