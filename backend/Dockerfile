FROM python:3.12-slim

WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY app/ /app/app
ENV PYTHONUNBUFFERED=1

EXPOSE 5000 8000
CMD ["python", "-m", "app.main"]
