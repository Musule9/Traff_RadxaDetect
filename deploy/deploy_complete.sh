#!/bin/bash

echo "🚀 DEPLOYMENT COMPLETO - VEHICLE DETECTION RKNN"
echo "==============================================="
echo "Frontend: Puerto 3001 | Backend: Puerto 8000"
echo ""

# Función para mostrar estado
show_status() {
    echo "📊 ESTADO ACTUAL:"
    echo "Contenedores:"
    docker ps --format "table {{.Names}}\t{{.Status}}\t{{.Ports}}"
    echo ""
    echo "Puertos activos:"
    sudo netstat -tulpn | grep -E ":3001|:8000" || echo "No hay puertos activos"
    echo ""
}

# Paso 1: Limpieza
echo "🧹 PASO 1: Limpieza completa..."
chmod +x cleanup-all.sh
./cleanup-all.sh

echo ""
echo "🔍 PASO 2: Verificación de archivos..."

# Verificar estructura
if [ ! -f "main.py" ]; then
    echo "❌ main.py no encontrado"
    exit 1
fi

if [ ! -d "frontend" ]; then
    echo "❌ Directorio frontend no encontrado"
    exit 1
fi

if [ ! -f "frontend/package.json" ]; then
    echo "❌ frontend/package.json no encontrado"
    exit 1
fi

echo "✅ Archivos principales encontrados"

# Paso 3: Crear archivos necesarios
echo ""
echo "📁 PASO 3: Creando estructura..."
mkdir -p app/{core,services} data config models logs
touch app/__init__.py app/core/__init__.py app/services/__init__.py

# Paso 4: Verificar/configurar CORS en main.py
echo ""
echo "🔧 PASO 4: Configurando CORS..."
if grep -q "CORSMiddleware" main.py; then
    echo "✅ CORS ya configurado en main.py"
else
    echo "⚠️ CORS no encontrado en main.py"
    echo "   Asegúrate de agregar la configuración CORS del artifact"
fi

# Paso 5: Build y deployment
echo ""
echo "🏗️ PASO 5: Building y deployment..."

# Verificar Docker
if ! command -v docker &> /dev/null; then
    echo "❌ Docker no está instalado"
    exit 1
fi

if ! command -v docker-compose &> /dev/null; then
    echo "❌ Docker Compose no está instalado"
    exit 1
fi

# Build y run
echo "📦 Building servicios..."
docker-compose build --no-cache

if [ $? -eq 0 ]; then
    echo "✅ Build exitoso"
else
    echo "❌ Build falló"
    exit 1
fi

echo ""
echo "🚀 PASO 6: Iniciando servicios..."
docker-compose up -d

# Esperar que los servicios inicien
echo "⏳ Esperando que los servicios inicien..."
sleep 10

# Verificar estado
echo ""
show_status

# Paso 7: Pruebas de conectividad
echo "🧪 PASO 7: Pruebas de conectividad..."

# Test backend
echo "Testing backend (puerto 8000)..."
BACKEND_RESPONSE=$(curl -s -o /dev/null -w "%{http_code}" http://localhost:8000/api/camera_health --connect-timeout 5)
if [ "$BACKEND_RESPONSE" == "200" ]; then
    echo "✅ Backend responde correctamente"
else
    echo "❌ Backend no responde (código: $BACKEND_RESPONSE)"
    echo "Logs del backend:"
    docker logs vehicle-detection-backend --tail 10
fi

# Test frontend
echo "Testing frontend (puerto 3001)..."
FRONTEND_RESPONSE=$(curl -s -o /dev/null -w "%{http_code}" http://localhost:3001 --connect-timeout 5)
if [ "$FRONTEND_RESPONSE" == "200" ]; then
    echo "✅ Frontend responde correctamente"
else
    echo "⚠️ Frontend no responde (código: $FRONTEND_RESPONSE)"
    echo "Logs del frontend:"
    docker logs vehicle-detection-frontend --tail 10
fi

# Obtener IP
IP=$(hostname -I | awk '{print $1}')

echo ""
echo "🎉 DEPLOYMENT COMPLETADO"
echo "========================"
echo ""
echo "🌐 ACCESO AL SISTEMA:"
echo "   Frontend: http://$IP:3001"
echo "   Backend API: http://$IP:8000"
echo "   API Docs: http://$IP:8000/docs"
echo "   Health Check: http://$IP:8000/api/camera_health"
echo ""
echo "📋 COMANDOS ÚTILES:"
echo "   Ver logs backend: docker logs -f vehicle-detection-backend"
echo "   Ver logs frontend: docker logs -f vehicle-detection-frontend"
echo "   Parar servicios: docker-compose down"
echo "   Reiniciar: docker-compose restart"
echo "   Estado: docker-compose ps"
echo ""
echo "🔧 VERIFICAR RKNN:"
echo "   docker exec vehicle-detection-backend python3 -c \"from rknnlite.api import RKNNLite; print('RKNN OK')\""
echo ""

# Mostrar logs recientes si hay errores
if [ "$BACKEND_RESPONSE" != "200" ] || [ "$FRONTEND_RESPONSE" != "200" ]; then
    echo "🚨 HAY ERRORES - LOGS RECIENTES:"
    echo ""
    echo "Backend logs:"
    docker logs vehicle-detection-backend --tail 15
    echo ""
    echo "Frontend logs:"
    docker logs vehicle-detection-frontend --tail 15
fi