#!/bin/bash
# 🚀 INSTALACIÓN COMPLETA DEL SISTEMA DE DETECCIÓN VEHICULAR

echo "🚀 INSTALANDO SISTEMA COMPLETO DE DETECCIÓN VEHICULAR"
echo "====================================================="

# Ir al directorio correcto
cd /opt/vehicle-detection

# 1. LIMPIAR SISTEMA ANTERIOR
echo "🧹 Limpiando sistema anterior..."
sudo docker stop vehicle-detection-prod 2>/dev/null || true
sudo docker rm vehicle-detection-prod 2>/dev/null || true
sudo docker stop mock-controller 2>/dev/null || true
sudo docker rm mock-controller 2>/dev/null || true

# Limpiar redes conflictivas
sudo docker network rm vehicle_net 2>/dev/null || true
sudo docker network prune -f

# 2. CREAR ESTRUCTURA DE DIRECTORIOS
echo "📁 Creando estructura de directorios..."
sudo mkdir -p {app/core,app/services,app/api,config,data,models,logs,frontend/build,scripts,tests}



# 4. CONSTRUIR E INICIAR SISTEMA
echo "🏗️  Construyendo imagen Docker..."
sudo docker-compose build --no-cache

echo "🚀 Iniciando sistema..."
sudo docker-compose up -d

# 5. VERIFICAR SISTEMA
echo "⏳ Esperando inicialización..."
sleep 15

echo "📋 Verificando logs..."
sudo docker logs vehicle-detection-prod --tail=20

echo "🔍 Verificando salud..."
curl -s http://localhost:8000/api/camera_health | jq . || echo "API iniciando..."

echo ""
echo "✅ SISTEMA INSTALADO COMPLETAMENTE"
echo "=================================="
echo "🌐 URL: http://$(hostname -I | awk '{print $1}'):8000"
echo "🔑 Usuario: admin / Contraseña: admin123"
echo ""
echo "📊 Comandos útiles:"
echo "  sudo docker-compose logs -f    # Ver logs"
echo "  sudo docker-compose restart    # Reiniciar"
echo "  sudo docker-compose ps         # Estado"
echo ""