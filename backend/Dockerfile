FROM python:3.12-slim

WORKDIR /app

# outils build + wget (panns_inference télécharge des assets via wget)
RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc \
    g++ \
    make \
    wget \
    ca-certificates \
    && rm -rf /var/lib/apt/lists/*

# Dossier où panns_inference stockera labels + checkpoint
ENV PANNS_DATA_DIR=/app/panns_data
RUN mkdir -p /app/panns_data

COPY requirements.txt .

# Torch CPU uniquement (sinon pip part sur CUDA -> énorme)
RUN pip install --no-cache-dir --index-url https://download.pytorch.org/whl/cpu \
    torch==2.5.1

RUN pip install --no-cache-dir -r requirements.txt

COPY . .

CMD ["python", "-m", "app.main"]
