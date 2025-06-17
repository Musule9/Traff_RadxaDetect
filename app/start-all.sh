#!/bin/bash

echo "🏠 DEPLOY TODO EN UN CONTENEDOR"
echo "==============================="
echo "Backend + Frontend + Supervisor en un solo container"
echo ""

# Limpiar contenedores anteriores
echo "🧹 Limpiando..."
docker stop $(docker ps -q) 2>/dev/null || true
docker rm vehicle-detection-all-in-one 2>/dev/null || true

# Crear archivos necesarios
mkdir -p app/{core,services} data config models logs
touch app/__init__.py app/core/__init__.py app/services/__init__.py

# Build usando Dockerfile especial
echo "🏗️ Building imagen all-in-one..."
docker build -f Dockerfile-all-in-one -t vehicle-detection-all-in-one .

if [ $? -ne 0 ]; then
    echo "❌ Build falló"
    exit 1
fi

# Ejecutar contenedor
echo "🚀 Iniciando contenedor all-in-one..."
docker run -d \
  --name vehicle-detection-all-in-one \
  --restart unless-stopped \
  --privileged \
  -p 8000:8000 \
  -p 3001:3001 \
  -p 9001:9001 \
  -v $(pwd)/data:/app/data \
  -v $(pwd)/config:/app/config \
  -v $(pwd)/models:/app/models \
  -v $(pwd)/logs:/app/logs \
  -v /dev:/dev \
  -v /sys:/sys:ro \
  -v /usr/lib/aarch64-linux-gnu:/usr/lib/aarch64-linux-gnu:ro \
  --device=/dev/dri --device=/dev/mali0 --device=/dev/dma_heap \
  -e USE_RKNN=1 \
  vehicle-detection-all-in-one

# Esperar inicio
echo "⏳ Esperando que los servicios inicien..."
sleep 10

# Verificar estado
echo ""
echo "📊 ESTADO DEL CONTENEDOR:"
docker ps | grep vehicle-detection-all-in-one

echo ""
echo "🧪 PRUEBAS DE CONECTIVIDAD:"

# Test backend
BACKEND_RESPONSE=$(curl -s -o /dev/null -w "%{http_code}" http://localhost:8000/api/camera_health --connect-timeout 5)
if [ "$BACKEND_RESPONSE" == "200" ]; then
    echo "✅ Backend (8000) responde"
else
    echo "❌ Backend (8000) no responde"
fi

# Test frontend
FRONTEND_RESPONSE=$(curl -s -o /dev/null -w "%{http_code}" http://localhost:3001 --connect-timeout 5)
if [ "$FRONTEND_RESPONSE" == "200" ]; then
    echo "✅ Frontend (3001) responde"
else
    echo "⚠️ Frontend (3001) no responde (puede tardar más)"
fi

# Test supervisor
SUPERVISOR_RESPONSE=$(curl -s -o /dev/null -w "%{http_code}" http://localhost:9001 --connect-timeout 3)
if [ "$SUPERVISOR_RESPONSE" == "200" ]; then
    echo "✅ Supervisor (9001) responde"
else
    echo "⚠️ Supervisor (9001) no responde"
fi

IP=$(hostname -I | awk '{print $1}')

echo ""
echo "🎉 ALL-IN-ONE DEPLOYADO"
echo "======================="
echo ""
echo "🌐 ACCESO:"
echo "Frontend:   http://$IP:3001"
echo "Backend:    http://$IP:8000"
echo "API Docs:   http://$IP:8000/docs"
echo "Supervisor: http://$IP:9001 (admin/admin)"
echo ""
echo "📋 COMANDOS:"
echo "Ver logs todo:     docker logs -f vehicle-detection-all-in-one"
echo "Ver logs backend:  docker exec vehicle-detection-all-in-one tail -f /var/log/supervisor/backend.out.log"
echo "Ver logs frontend: docker exec vehicle-detection-all-in-one tail -f /var/log/supervisor/frontend.out.log"
echo "Acceder container: docker exec -it vehicle-detection-all-in-one /bin/bash"
echo "Parar todo:        docker stop vehicle-detection-all-in-one"
echo ""

# Mostrar logs si hay errores
if [ "$BACKEND_RESPONSE" != "200" ] || [ "$FRONTEND_RESPONSE" != "200" ]; then
    echo "🚨 LOGS RECIENTES:"
    docker logs vehicle-detection-all-in-one --tail 20
fi