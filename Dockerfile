FROM python:3.12-slim

# Evitar problemas con matplotlib en modo headless
ENV MPLBACKEND=Agg \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

WORKDIR /app

# Instalar dependencias del sistema mínimas para pandas/matplotlib
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    gcc \
    && rm -rf /var/lib/apt/lists/*

# Copiar requirements e instalar
COPY requirements.txt .
RUN pip install -r requirements.txt

COPY src ./src

RUN mkdir -p data reports

CMD ["python", "src/eda_flakiness.py", "--help"]
