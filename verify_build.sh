#!/bin/bash
set -e

echo "🔍 SCRIPT DE VERIFICACIÓN DEL SISTEMA"
echo "====================================="

PROJECT_DIR="/opt/vehicle-detection"
cd "$PROJECT_DIR"

echo ""
echo "📁 1. VERIFICANDO ESTRUCTURA EN HOST:"
echo "------------------------------------"

# Verificar archivos principales
check_file() {
    if [ -f "$1" ]; then
        echo "✅ $1"
    else
        echo "❌ $1 - FALTA"
    fi
}

check_dir() {
    if [ -d "$1" ]; then
        echo "✅ $1/ ($(ls -1 "$1" | wc -l) archivos)"
    else
        echo "❌ $1/ - FALTA"
    fi
}

check_file "main.py"
check_file "requirements.txt"
check_file "Dockerfile"
check_file "docker-compose.yml"
check_dir "app"
check_dir "app/core"
check_dir "app/services"
check_dir "frontend"
check_dir "config"
check_dir "scripts"

echo ""
echo "📦 2. CONSTRUYENDO IMAGEN:"
echo "-------------------------"
docker-compose build --no-cache

echo ""
echo "🚀 3. INICIANDO CONTENEDOR:"
echo "---------------------------"
docker-compose up -d

echo ""
echo "⏳ 4. ESPERANDO INICIALIZACIÓN (60 segundos)..."
sleep 60

echo ""
echo "🔍 5. VERIFICANDO ESTRUCTURA EN CONTENEDOR:"
echo "-------------------------------------------"
docker exec vehicle-detection-prod /bin/bash -c "
echo '📁 Estructura principal:'
ls -la /app/

echo ''
echo '📂 Directorio app/:'
ls -la /app/app/ 2>/dev/null || echo '❌ /app/app/ NO existe'

echo ''
echo '📂 Subdirectorios de app/:'
ls -la /app/app/core/ 2>/dev/null || echo '❌ /app/app/core/ NO existe'
ls -la /app/app/services/ 2>/dev/null || echo '❌ /app/app/services/ NO existe'

echo ''
echo '📂 Frontend:'
ls -la /app/frontend/build/ 2>/dev/null || echo '❌ /app/frontend/build/ NO existe'

echo ''
echo '📄 Archivos Python encontrados:'
find /app -name '*.py' | head -10

echo ''
echo '📋 Configuraciones:'
ls -la /app/config/ 2>/dev/null || echo '❌ /app/config/ NO existe'

echo ''
echo '🗄️ Directorio de datos:'
ls -la /app/data/ 2>/dev/null || echo '❌ /app/data/ NO existe'
"

echo ""
echo "🌐 6. VERIFICANDO API:"
echo "---------------------"
curl -s http://localhost:8000/api/camera_health | jq . 2>/dev/null || curl -s http://localhost:8000/api/camera_health

echo ""
echo "📊 7. VERIFICANDO ENDPOINTS:"
echo "----------------------------"
echo "📹 Configuración de cámara:"
curl -s http://localhost:8000/api/camera/config | jq . 2>/dev/null || echo "Error en endpoint"

echo ""
echo "📏 Líneas de análisis:"
curl -s http://localhost:8000/api/analysis/lines | jq . 2>/dev/null || echo "Error en endpoint"

echo ""
echo "🎯 Zonas de análisis:"
curl -s http://localhost:8000/api/analysis/zones | jq . 2>/dev/null || echo "Error en endpoint"

echo ""
echo "✅ VERIFICACIÓN COMPLETADA"
echo "=========================="
echo ""
echo "🌐 URLs disponibles:"
echo "  Frontend: http://$(hostname -I | awk '{print $1}'):8000"
echo "  API Docs: http://$(hostname -I | awk '{print $1}'):8000/docs"
echo "  Health:   http://$(hostname -I | awk '{print $1}'):8000/api/camera_health"
echo ""
echo "🔧 Comandos útiles:"
echo "  docker logs vehicle-detection-prod"
echo "  docker exec -it vehicle-detection-prod /bin/bash"
echo "  docker-compose restart"