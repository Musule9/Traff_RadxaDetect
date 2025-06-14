FROM node:18-slim AS frontend-builder

WORKDIR /app/frontend
COPY frontend/package*.json ./
RUN npm ci --only=production

COPY frontend/ ./
RUN npm run build

# =============================================================================
# STAGE 2: Base System
# =============================================================================
FROM ubuntu:22.04 AS base-system

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1

# Instalar dependencias del sistema
RUN apt-get update && apt-get install -y \
    python3 \
    python3-pip \
    python3-dev \
    python3-venv \
    cmake \
    build-essential \
    pkg-config \
    libopencv-dev \
    libgstreamer1.0-dev \
    libgstreamer-plugins-base1.0-dev \
    libv4l-dev \
    wget \
    curl \
    git \
    sqlite3 \
    ffmpeg \
    && rm -rf /var/lib/apt/lists/*

# =============================================================================
# STAGE 3: Platform Detection and Setup
# =============================================================================
FROM base-system AS platform-setup

# Script para detectar plataforma
RUN echo '#!/bin/bash\n\
if [ -f "/proc/device-tree/model" ]; then\n\
  MODEL=$(cat /proc/device-tree/model 2>/dev/null || echo "")\n\
    if echo "$MODEL" | grep -qi "jetson"; then\n\
        echo "jetson"\n\
        exit 0\n\
    fi\n\
    if echo "$MODEL" | grep -qi "radxa\|rock"; then\n\
        echo "radxa"\n\
        exit 0\n\
    fi\n\
fi\n\
if [ -f "/proc/cpuinfo" ]; then\n\
    if grep -qi "rockchip\|rk3588" /proc/cpuinfo; then\n\
        echo "radxa"\n\
        exit 0\n\
    fi\n\
fi\n\
echo "generic"\n' > /usr/local/bin/detect_platform.sh \
    && chmod +x /usr/local/bin/detect_platform.sh

# Instalar dependencias específicas de Radxa
RUN PLATFORM=$(bash /usr/local/bin/detect_platform.sh) && \
    echo "Detected platform: $PLATFORM" && \
    if [ "$PLATFORM" = "radxa" ]; then \
        echo "Installing Radxa-specific packages..."; \
        # Intentar instalar RKNN si está disponible
        apt-get update && apt-get install -y \
            rockchip-mpp-dev \
            rockchip-rga-dev \
        || echo "Radxa packages not available, continuing..."; \
    fi && \
    rm -rf /var/lib/apt/lists/*

# =============================================================================
# STAGE 4: Python Dependencies
# =============================================================================
FROM platform-setup AS python-deps

WORKDIR /app

# Copiar requirements
COPY requirements.txt ./

# Instalar dependencias Python
RUN pip3 install --no-cache-dir --upgrade pip && \
    pip3 install --no-cache-dir -r requirements.txt

# Instalar dependencias opcionales por plataforma
RUN PLATFORM=$(bash /usr/local/bin/detect_platform.sh) && \
    if [ "$PLATFORM" = "radxa" ]; then \
        pip3 install --no-cache-dir \
            rknn-toolkit-lite2 \
        || echo "RKNN Python packages not available"; \
    fi

# =============================================================================
# STAGE 5: Final Application
# =============================================================================
FROM python-deps AS final

# Metadata
LABEL maintainer="Vehicle Detection System"
LABEL description="Sistema avanzado de detección y conteo vehicular"
LABEL version="1.0.0"

# Variables de entorno de la aplicación
ENV APP_ENV=production \
    LOG_LEVEL=INFO \
    MAX_CAMERAS=1 \
    API_HOST=0.0.0.0 \
    API_PORT=8000 \
    DATA_PATH=/app/data \
    CONFIG_PATH=/app/config \
    MODELS_PATH=/app/models \
    LOGS_PATH=/app/logs

# Crear directorios necesarios
RUN mkdir -p \
    $DATA_PATH \
    $CONFIG_PATH \
    $MODELS_PATH \
    $LOGS_PATH \
    /app/frontend/build

# Copiar aplicación backend
COPY app/ ./app/
COPY main.py ./
COPY scripts/ ./scripts/

# Copiar frontend construido
COPY --from=frontend-builder /app/frontend/build ./frontend/build

# Copiar archivos de configuración
COPY config/ ./config/

# Crear directorio para modelos y descargar modelo por defecto
RUN mkdir -p $MODELS_PATH && \
    echo "Downloading default YOLOv8n model..." && \
    wget -q -O $MODELS_PATH/yolov8n.onnx \
        https://github.com/ultralytics/assets/releases/download/v0.0.0/yolov8n.onnx \
    || echo "Model download failed, will be downloaded on first run"

# Script de inicialización
RUN echo '#!/bin/bash\n\
set -e\n\
\n\
echo "🚀 Iniciando Sistema de Detección Vehicular..."\n\
echo "Platform: $(detect_platform.sh)"\n\
echo "Environment: $APP_ENV"\n\
echo "Max Cameras: $MAX_CAMERAS"\n\
\n\
# Verificar directorios\n\
mkdir -p $DATA_PATH $CONFIG_PATH $MODELS_PATH $LOGS_PATH\n\
\n\
# Verificar permisos\n\
chown -R $(whoami) $DATA_PATH $CONFIG_PATH $MODELS_PATH $LOGS_PATH || true\n\
\n\
# Inicializar configuración si no existe\n\
if [ ! -f "$CONFIG_PATH/system.json" ]; then\n\
    echo "Inicializando configuración por defecto..."\n\
    python3 scripts/init_config.py\n\
fi\n\
\n\
# Verificar modelo\n\
if [ ! -f "$MODELS_PATH/yolov8n.onnx" ]; then\n\
    echo "Descargando modelo YOLOv8n..."\n\
    wget -O $MODELS_PATH/yolov8n.onnx \\\n\
        https://github.com/ultralytics/assets/releases/download/v0.0.0/yolov8n.onnx\n\
fi\n\
\n\
# Convertir a RKNN si es Radxa\n\
PLATFORM=$(detect_platform.sh)\n\
if [ "$PLATFORM" = "radxa" ] && [ ! -f "$MODELS_PATH/yolov8n.rknn" ]; then\n\
    echo "Convirtiendo modelo a RKNN..."\n\
    python3 scripts/convert_model.py || echo "Conversión RKNN falló, usando ONNX"\n\
fi\n\
\n\
echo "✅ Inicialización completada"\n\
echo "🌐 Iniciando servidor en puerto $API_PORT..."\n\
\n\
# Iniciar aplicación\n\
exec python3 main.py\n' > /app/start.sh && chmod +x /app/start.sh

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=10s --start-period=40s --retries=3 \
    CMD curl -f http://localhost:8000/api/camera_health || exit 1

CMD ["/app/start.sh"]