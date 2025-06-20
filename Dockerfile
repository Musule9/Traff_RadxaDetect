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

# RKNN: Crear directorio y descargar librknnrt.so para RK3588
RUN if [ -f /proc/device-tree/model ]; then \
        MODEL=$(cat /proc/device-tree/model 2>/dev/null || echo "Unknown"); \
        if [[ "$MODEL" == *"Radxa"* ]] || [[ "$MODEL" == *"ROCK"* ]]; then \
            echo "✅ Radxa Rock detectada - configurando NPU"; \
            # Verificar librknnrt.so
            if [ ! -f "/usr/lib/librknnrt.so" ]; then \
                wget -q https://github.com/airockchip/rknn-toolkit2/raw/master/rknpu2/runtime/Linux/librknn_api/aarch64/librknnrt.so \
                -O /usr/lib/librknnrt.so; \
                chmod +x /usr/lib/librknnrt.so; \
            fi; \
        fi; \
    fi

# Copiar archivos de la aplicación
COPY . .

# Instalar dependencias Python básicas primero
RUN pip install --no-cache-dir --upgrade pip setuptools wheel

# Instalar dependencias Python desde requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

# Instalar Ultralytics ACTUALIZADO con soporte RKNN
RUN pip install --no-cache-dir "ultralytics>=8.3.0"

# Instalar OpenCV optimizado para ARM64
RUN pip install --no-cache-dir opencv-python-headless==4.8.1.78

# Instalar RKNN Toolkit Lite2 para RK3588
RUN pip install --no-cache-dir rknn-toolkit-lite2==2.3.2

# Crear directorios necesarios
RUN mkdir -p /app/models /app/data /app/config /app/logs

# Descargar modelo YOLO11n base si no existe
RUN cd /app/models && \
    if [ ! -f "yolo11n.pt" ]; then \
        wget -q https://github.com/ultralytics/assets/releases/download/v8.3.0/yolo11n.pt; \
    fi

# Variables de entorno
ENV PYTHONPATH="/app:$PYTHONPATH"
ENV USE_RKNN=1
ENV OPENCV_LOG_LEVEL=ERROR
ENV OPENCV_DNN_BACKEND=DEFAULT
ENV LD_LIBRARY_PATH="/usr/lib:$LD_LIBRARY_PATH"
ENV RKNN_TARGET_PLATFORM=rk3588
ENV RKNN_QUANTIZE=i8
ENV NPU_ENABLED=1
ENV PROCESSING_WIDTH=640
ENV PROCESSING_HEIGHT=640
ENV FORCE_RESOLUTION=640x640
# Puerto de la aplicación
EXPOSE 8000

# Script de inicio
CMD ["python3", "main.py"]