set -e

echo "🚀 Iniciando Sistema de Detección Vehicular para Radxa Rock 5T..."

# Detectar plataforma
if [ -f /proc/device-tree/model ]; then
    MODEL=$(cat /proc/device-tree/model 2>/dev/null || echo "Unknown")
    echo "Hardware detectado: $MODEL"
fi

# Verificar si es Radxa Rock
if [[ "$MODEL" == *"Radxa"* ]] || [[ "$MODEL" == *"ROCK"* ]]; then
    echo "✅ Radxa Rock detectada - habilitando optimizaciones NPU"
    export USE_RKNN=1
else
    echo "⚠️  Hardware no reconocido como Radxa - usando CPU/OpenCV"
    export USE_RKNN=0
fi

# Crear directorios necesarios
mkdir -p /app/data /app/config /app/models /app/logs

# Crear archivos de configuración por defecto si no existen
echo "📁 Verificando archivos de configuración..."

if [ ! -f "/app/config/analysis.json" ]; then
    echo "📝 Creando analysis.json por defecto..."
    cat > /app/config/analysis.json << 'EOF'
{
  "lines": {},
  "zones": {}
}
EOF
fi

if [ ! -f "/app/config/cameras.json" ]; then
    echo "📝 Creando cameras.json por defecto..."
    cat > /app/config/cameras.json << 'EOF'
{
  "camera_1": {
    "id": "camera_1",
    "name": "Cámara Principal",
    "rtsp_url": "",
    "fase": "fase1",
    "direccion": "norte",
    "controladora_id": "CTRL_001",
    "controladora_ip": "192.168.1.200",
    "enabled": false
  }
}
EOF
fi

if [ ! -f "/app/config/controllers.json" ]; then
    echo "📝 Creando controllers.json por defecto..."
    cat > /app/config/controllers.json << 'EOF'
{
  "controllers": {
    "CTRL_001": {
      "id": "CTRL_001",
      "name": "Controladora Principal",
      "network": {
        "ip": "192.168.1.200",
        "port": 8080
      },
      "endpoints": {
        "analytic": "/api/analitico",
        "status": "/api/analiticos"
      }
    }
  }
}
EOF
fi

echo "✅ Archivos de configuración verificados"

# Configuración de permisos
chown -R $(whoami) /app/data /app/config /app/models /app/logs 2>/dev/null || true

# Inicializar configuración si no existe
if [ ! -f "/app/config/system.json" ]; then
    echo "📝 Inicializando configuración por defecto..."
    python3 /app/scripts/init_config.py
fi

# Verificar y convertir modelo a RKNN si es necesario
if [ "$USE_RKNN" = "1" ] && [ ! -f "/app/models/yolov8n.rknn" ]; then
    echo "🔧 Convirtiendo modelo YOLOv8n a RKNN..."
    python3 /app/scripts/convert_model.py
fi

# Configurar variables de entorno
export PYTHONPATH="/app:$PYTHONPATH"
export DATA_RETENTION_DAYS=${DATA_RETENTION_DAYS:-30}
export LOG_LEVEL=${LOG_LEVEL:-INFO}

echo "📊 Configuración:"
echo "  - Retención de datos: $DATA_RETENTION_DAYS días"
echo "  - Nivel de log: $LOG_LEVEL"
echo "  - Cámaras máximas: $MAX_CAMERAS"
echo "  - Uso de RKNN: $USE_RKNN"

# Limpiar bases de datos antiguas al inicio
echo "🧹 Limpiando bases de datos antiguas..."
python3 -c "
import asyncio
from app.core.database import DatabaseManager
async def cleanup():
    db = DatabaseManager(retention_days=int('$DATA_RETENTION_DAYS'))
    await db.cleanup_old_databases()
asyncio.run(cleanup())
"

echo "🌐 Iniciando servidor web..."

# Iniciar aplicación principal
exec python3 main.py