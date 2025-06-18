FROM python:3.11-slim

WORKDIR /app

RUN apt-get update && apt-get install -y \
    curl \
    ffmpeg \
    build-essential \
    gcc \
    g++ \
    libffi-dev \
    libssl-dev \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    libgl1-mesa-glx \
    libglib2.0-0 \
    wget \
    git \
    && rm -rf /var/lib/apt/lists/*

# Copiar archivos de la aplicación
COPY . .

# Instalar dependencias Python básicas
RUN pip install --no-cache-dir -r requirements.txt

# INSTALACIÓN DE RKNN TOOLKIT LITE (para Radxa Rock 5T)
RUN echo "🔧 Instalando RKNN Toolkit Lite..." && \
    pip install --no-cache-dir rknn-toolkit-lite2 || \
    pip install --no-cache-dir rknn-toolkit2 || \
    echo "⚠️ RKNN no instalado - continuando sin soporte RKNN"

EXPOSE 8000

CMD ["python3", "main.py"]