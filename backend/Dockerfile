FROM python:3.12-slim

WORKDIR /app

# outils build pour webrtcvad (C ext)
RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc \
    g++ \
    make \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .

# 1) Installer torch CPU uniquement (sinon pip prend CUDA -> énorme)
RUN pip install --no-cache-dir --index-url https://download.pytorch.org/whl/cpu \
    torch==2.5.1

# 2) Installer le reste
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

CMD ["python", "-m", "app.main"]
