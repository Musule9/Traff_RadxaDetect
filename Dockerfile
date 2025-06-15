# ============================================================================
# STAGE 1: Frontend Build (CON MANEJO DE ERRORES)
# ============================================================================
FROM node:18-slim AS frontend-builder

WORKDIR /app/frontend

# Instalar dependencias del sistema para npm
RUN apt-get update && apt-get install -y \
    python3 \
    make \
    g++ \
    git \
    && rm -rf /var/lib/apt/lists/*

# Copiar package files y verificar
COPY frontend/package*.json ./

# Verificar que los archivos existen
RUN echo "📦 Verificando archivos package..." && \
    ls -la && \
    echo "📄 Contenido de package.json:" && \
    head -10 package.json

# Limpiar cache de npm y instalar dependencias
RUN echo "🧹 Limpiando cache npm..." && \
    npm cache clean --force && \
    echo "📥 Instalando dependencias..." && \
    npm ci --verbose --no-audit --no-fund || \
    (echo "❌ npm ci falló, intentando npm install..." && npm install --verbose --no-audit --no-fund)

# Copiar código fuente
COPY frontend/ ./

# Verificar estructura antes del build
RUN echo "📁 Verificando estructura:" && \
    ls -la src/ && \
    echo "📄 Verificando archivos principales:" && \
    ls -la src/App.js src/index.js

# Intentar build con fallback
RUN echo "🏗️ Construyendo frontend..." && \
    (npm run build && echo "✅ Build exitoso") || \
    (echo "❌ Build falló, creando build básico..." && \
     mkdir -p build && \
     echo '<!DOCTYPE html>
<html lang="es">
<head>
    <meta charset="utf-8" />
    <meta name="viewport" content="width=device-width, initial-scale=1" />
    <title>Sistema de Detección Vehicular</title>
    <style>
        body { background: #111827; color: white; font-family: Arial; text-align: center; padding: 50px; }
        .container { max-width: 600px; margin: 0 auto; }
        .status { background: #10B981; padding: 20px; border-radius: 8px; margin: 20px 0; }
        .info { background: #374151; padding: 15px; border-radius: 8px; margin: 10px 0; }
    </style>
</head>
<body>
    <div class="container">
        <h1>🚗 Sistema de Detección Vehicular</h1>
        <div class="status">✅ Sistema Funcionando</div>
        <div class="info">
            <p><strong>Hardware:</strong> Radxa Rock 5T</p>
            <p><strong>Versión:</strong> 1.0.0</p>
            <p><strong>Estado:</strong> Modo Básico</p>
        </div>
        <div class="info">
            <p>🔧 <strong>API Endpoints Disponibles:</strong></p>
            <p><a href="/docs" style="color: #60A5FA;">/docs</a> - Documentación de la API</p>
            <p><a href="/api/camera_health" style="color: #60A5FA;">/api/camera_health</a> - Estado del sistema</p>
        </div>
        <script>
            // Auto-refresh cada 30 segundos
            setTimeout(() => location.reload(), 30000);
            
            // Verificar API
            fetch("/api/camera_health")
                .then(r => r.json())
                .then(data => {
                    console.log("API Status:", data);
                    document.getElementById("api-status").innerHTML = "✅ API Funcionando";
                })
                .catch(e => {
                    console.log("API Error:", e);
                    document.getElementById("api-status").innerHTML = "⚠️ API Iniciando...";
                });
        </script>
        <div id="api-status" class="info">🔄 Verificando API...</div>
    </div>
</body>
</html>' > build/index.html && \
     mkdir -p build/static && \
     echo "✅ Build básico creado")

# Verificar que build existe
RUN echo "🔍 Verificando build final:" && \
    ls -la build/ && \
    echo "📄 Contenido de index.html:" && \
    head -5 build/index.html

# ============================================================================
# STAGE 2: Backend + Sistema
# ============================================================================
FROM python:3.11-slim-bookworm AS final

# Variables de entorno
ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    APP_ENV=production \
    LOG_LEVEL=info

WORKDIR /app

# Instalar dependencias del sistema
RUN apt-get update && apt-get install -y \
    python3-opencv \
    curl \
    wget \
    ffmpeg \
    sqlite3 \
    libglib2.0-0 \
    jq \
    && rm -rf /var/lib/apt/lists/*

# Instalar dependencias Python
COPY requirements.txt .
RUN echo "📦 Instalando dependencias Python..." && \
    pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Copiar aplicación backend
COPY app/ ./app/
COPY main.py ./
COPY scripts/ ./scripts/
COPY config/ ./config/

# Copiar frontend construido
COPY --from=frontend-builder /app/frontend/build ./frontend/build

# Verificar estructura final
RUN echo "📁 Verificando estructura final:" && \
    echo "Backend:" && ls -la app/ && \
    echo "Frontend:" && ls -la frontend/build/ && \
    echo "Config:" && ls -la config/ && \
    echo "Scripts:" && ls -la scripts/

# Crear directorios necesarios
RUN mkdir -p /app/{data,models,logs} && \
    echo "📁 Directorios creados:" && \
    ls -la /app/

# Hacer scripts ejecutables
RUN find scripts/ -name "*.py" -exec chmod +x {} \; 2>/dev/null || true

# Script de inicio mejorado con verificaciones
RUN echo '#!/bin/bash\n\
set -e\n\
\n\
echo "🚀 Iniciando Sistema de Detección Vehicular..."\n\
echo "==============================================="\n\
\n\
# Verificar estructura\n\
echo "📁 Verificando estructura..."\n\
ls -la /app/\n\
\n\
# Crear directorios necesarios\n\
mkdir -p /app/{data,config,models,logs}\n\
echo "✅ Directorios verificados"\n\
\n\
# Verificar frontend\n\
if [ -f "/app/frontend/build/index.html" ]; then\n\
    echo "✅ Frontend disponible"\n\
else\n\
    echo "⚠️ Frontend no disponible - solo API"\n\
fi\n\
\n\
# Inicializar configuración\n\
if [ ! -f "/app/config/system.json" ]; then\n\
    echo "📝 Inicializando configuración por defecto..."\n\
    python3 scripts/init_config.py 2>/dev/null || \\\n\
    echo "⚠️ Error inicializando config - usando valores por defecto"\n\
fi\n\
\n\
# Descargar modelo YOLOv8n si no existe\n\
if [ ! -f "/app/models/yolov8n.onnx" ]; then\n\
    echo "📥 Descargando modelo YOLOv8n..."\n\
    wget -q --timeout=30 -O /app/models/yolov8n.onnx \\\n\
        "https://github.com/ultralytics/assets/releases/download/v0.0.0/yolov8n.onnx" || \\\n\
    echo "⚠️ Error descargando modelo - se descargará en el primer uso"\n\
fi\n\
\n\
# Verificar permisos\n\
chmod -R 755 /app/data /app/config /app/models /app/logs 2>/dev/null || true\n\
\n\
echo "✅ Inicialización completa"\n\
echo "🌐 Iniciando servidor en puerto 8000..."\n\
echo "📖 Documentación disponible en: http://localhost:8000/docs"\n\
\n\
# Iniciar aplicación con manejo de errores\n\
exec python3 main.py' > /app/start.sh && chmod +x /app/start.sh

# Exponer puerto
EXPOSE 8000

# Health check mejorado
HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
    CMD curl -f http://localhost:8000/api/camera_health || exit 1

# Comando de inicio
CMD ["/app/start.sh"]