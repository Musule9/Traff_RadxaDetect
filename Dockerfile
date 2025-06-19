FROM python:3.11-slim

WORKDIR /app

# Instalar dependencias del sistema (incluyendo las necesarias para OpenCV y RKNN)
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
    libgstreamer1.0-0 \
    libgstreamer-plugins-base1.0-0 \
    wget \
    git \
    pkg-config \
    libgtk-3-dev \
    libavcodec-dev \
    libavformat-dev \
    libswscale-dev \
    libjpeg-dev \
    libpng-dev \
    libtiff5-dev \
    && rm -rf /var/lib/apt/lists/*

# Copiar archivos de la aplicación
COPY . .

# Instalar dependencias Python básicas primero
RUN pip install --no-cache-dir --upgrade pip setuptools wheel

# Instalar dependencias Python desde requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

# Instalar Ultralytics ACTUALIZADO con soporte RKNN
RUN pip install --no-cache-dir "ultralytics>=8.3.0"

# Instalar OpenCV optimizado para ARM64 si es posible
RUN pip install --no-cache-dir opencv-python-headless==4.8.1.78

# Para Radxa Rock 5T/RK3588: instalar librerías RKNN
RUN apt-get update && apt-get install -y \
    wget \
    && rm -rf /var/lib/apt/lists/*

# Descargar e instalar las librerías RKNN para RK3588 (si están disponibles)
RUN cd /tmp && \
    (wget -q https://github.com/airockchip/rknn-toolkit2/raw/master/rknpu2/runtime/Linux/librknn_api/aarch64/librknnrt.so && \
     cp librknnrt.so /usr/lib/ && \
     ldconfig ) 

# Instalar RKNN Toolkit Lite2 (para dispositivos ARM64)
RUN pip install --no-cache-dir rknn-toolkit-lite2==2.3.2 || \
    pip install --no-cache-dir rknn-toolkit2 

# Crear directorios necesarios
RUN mkdir -p /app/models /app/data /app/config /app/logs

# Copiar script de conversión de modelo
#COPY scripts/convert_yolo11_rknn.py /app/scripts/

# Variables de entorno
ENV PYTHONPATH="/app:$PYTHONPATH"
ENV USE_RKNN=1
ENV OPENCV_LOG_LEVEL=ERROR

# Puerto de la aplicación
EXPOSE 8000

# Script de inicio
CMD ["python3", "main.py"]