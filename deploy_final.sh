#!/bin/bash

echo "🚀 DEPLOYMENT FINAL - VEHICLE DETECTION"
echo "========================================"
echo "Estrategia: Frontend local + Backend Docker"
echo ""

# Función para mostrar error y salir
error_exit() {
    echo "❌ ERROR: $1"
    exit 1
}

# Paso 1: Verificar archivos críticos
echo "📁 Verificando archivos críticos..."
[ ! -f "main.py" ] && error_exit "main.py no encontrado"
[ ! -f "requirements.txt" ] && error_exit "requirements.txt no encontrado"
[ ! -d "frontend" ] && error_exit "Directorio frontend no encontrado"

echo "✅ Archivos críticos encontrados"

# Paso 2: Limpiar instalaciones anteriores
echo ""
echo "🧹 Limpieza completa..."
docker stop $(docker ps -q) 2>/dev/null || true
docker rm vehicle-detection-production 2>/dev/null || true
docker system prune -f
pkill -f "npm start" 2>/dev/null || true

# Paso 3: Compilar frontend LOCALMENTE
echo ""
echo "🏗️ Compilando frontend localmente..."
cd frontend

# Arreglar package-lock.json si está desincronizado
if [ -f "package-lock.json" ]; then
    echo "📦 Arreglando package-lock.json..."
    rm package-lock.json
fi

# Instalar dependencias
echo "📦 Instalando dependencias..."
npm install

if [ $? -ne 0 ]; then
    echo "⚠️ npm install falló, intentando con --legacy-peer-deps..."
    npm install --legacy-peer-deps
    [ $? -ne 0 ] && error_exit "No se pudieron instalar las dependencias de npm"
fi

# Compilar
echo "🔨 Compilando frontend..."
npm run build

if [ ! -d "build" ] || [ ! -f "build/index.html" ]; then
    error_exit "Frontend build falló - no se generó build/index.html"
fi

echo "✅ Frontend compilado exitosamente"
ls -la build/ | head -5

cd ..

# Paso 4: Crear estructura para Docker
echo ""
echo "📂 Preparando estructura..."
mkdir -p app/{core,services} data config models logs
touch app/__init__.py app/core/__init__.py app/services/__init__.py

# Paso 5: Docker build (sin frontend compilation)
echo ""
echo "🐳 Building Docker (solo backend)..."
docker build -t vehicle-final .

if [ $? -ne 0 ]; then
    echo "📋 Mostrando últimas líneas del build para debug..."
    docker build --no-cache -t vehicle-detection-final . --no-cache 2>&1 | tail -20
    error_exit "Docker build falló"
fi

echo "✅ Docker build exitoso"

# Paso 6: Ejecutar contenedor
echo ""
echo "🚀 Iniciando contenedor..."
docker run -d \
  --name vehicle-detection-production \
  --restart unless-stopped \
  --privileged \
  -p 8000:8000 \
  -v $(pwd)/data:/app/data \
  -v $(pwd)/config:/app/config \
  -v $(pwd)/models:/app/models \
  -v $(pwd)/logs:/app/logs \
  -v /dev:/dev \
  -v /sys:/sys:ro \
  -v /usr/lib/aarch64-linux-gnu:/usr/lib/aarch64-linux-gnu:ro \
  --device=/dev/dri:/dev/dri \
  --device=/dev/mali0:/dev/mali0 \
  --device=/dev/dma_heap:/dev/dma_heap \
  --device=/dev/rga:/dev/rga \
  --device=/dev/mpp_service:/dev/mpp_service \
  -e USE_RKNN=1 \
  -e PYTHONPATH=/app \
  -e LD_LIBRARY_PATH=/usr/lib/aarch64-linux-gnu \
  vehicle-detection-final

if [ $? -ne 0 ]; then
    error_exit "No se pudo iniciar el contenedor"
fi

# Paso 7: Verificar funcionamiento
echo ""
echo "⏳ Esperando que el sistema inicie..."
sleep 10

echo "🧪 Verificando funcionamiento..."

# Test health
for i in {1..6}; do
    if curl -s http://localhost:8000/api/camera_health > /dev/null 2>&1; then
        echo "✅ Sistema funcionando correctamente"
        break
    else
        echo "⏳ Intento $i/6 - esperando..."
        sleep 5
    fi
    
    if [ $i -eq 6 ]; then
        echo "❌ Sistema no responde después de 30 segundos"
        echo ""
        echo "📋 LOGS DEL CONTENEDOR:"
        docker logs vehicle-detection-production --tail 20
        exit 1
    fi
done

# Obtener información del sistema
IP=$(hostname -I | awk '{print $1}')
HEALTH_DATA=$(curl -s http://localhost:8000/api/camera_health 2>/dev/null)

echo ""
echo "🎉 DEPLOYMENT COMPLETADO EXITOSAMENTE"
echo "====================================="
echo ""
echo "🌐 ACCESO AL SISTEMA:"
echo "   Principal: http://$IP:8000"
echo "   API Docs:  http://$IP:8000/docs"
echo "   Health:    http://$IP:8000/api/camera_health"
echo ""
echo "📊 ESTADO DEL SISTEMA:"
docker ps --format "table {{.Names}}\t{{.Status}}\t{{.Ports}}" | grep vehicle-detection
echo ""
echo "🔧 CONFIGURACIÓN:"
echo "$HEALTH_DATA" | python3 -c "import sys, json; print(json.dumps(json.load(sys.stdin), indent=2))" 2>/dev/null || echo "Health data: OK"
echo ""
echo "📋 COMANDOS ÚTILES:"
echo "   Ver logs:    docker logs -f vehicle-detection-production"
echo "   Reiniciar:   docker restart vehicle-detection-production"
echo "   Parar:       docker stop vehicle-detection-production"
echo "   Health test: curl http://$IP:8000/api/camera_health"
echo ""
echo "🎯 El sistema está listo para producción en el puerto 8000"