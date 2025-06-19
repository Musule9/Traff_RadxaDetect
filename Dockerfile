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

# Instalar Ultralytics actualizado con soporte RKNN
RUN pip install --no-cache-dir ultralytics>=8.3.0

RUN apt-get update && apt-get install -y \
    wget \
    && rm -rf /var/lib/apt/lists/*

# Descargar e instalar las librerías RKNN para RK3588
RUN cd /tmp && \
    wget https://github.com/airockchip/rknn-toolkit2/raw/master/rknpu2/runtime/Linux/librknn_api/aarch64/librknnrt.so && \
    cp librknnrt.so /usr/lib/ && \
    ldconfig

# Instalar RKNN Toolkit Lite2
RUN pip install --no-cache-dir rknn-toolkit-lite2==2.3.2 || \
    pip install --no-cache-dir rknn-toolkit2 || \
    echo "⚠️ RKNN no instalado - continuando sin soporte RKNN"

EXPOSE 8000

CMD ["python3", "main.py"]