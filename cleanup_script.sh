#!/bin/bash

echo "🧹 LIMPIEZA COMPLETA DE CONTENEDORES"
echo "==================================="

# Parar todos los contenedores relacionados
echo "📱 Parando contenedores..."
docker stop $(docker ps -q) 2>/dev/null || echo "No hay contenedores corriendo"

# Remover contenedores específicos
echo "🗑️ Removiendo contenedores anteriores..."
docker rm -f vehicle-detection-prod 2>/dev/null || true
docker rm -f vehicle-detection-simple 2>/dev/null || true
docker rm -f traff_radxadetect_vehicle-detection 2>/dev/null || true
docker rm -f vehicle-detection-rknn 2>/dev/null || true
docker rm -f vehicle-detection-smart 2>/dev/null || true
docker rm -f vehicle-detection-final 2>/dev/null || true

# Limpiar imágenes no usadas
echo "🖼️ Limpiando imágenes no usadas..."
docker image prune -f

# Limpiar redes no usadas
echo "🌐 Limpiando redes..."
docker network prune -f

# Parar procesos npm/node si existen
echo "🔄 Parando procesos Node/NPM..."
pkill -f "npm start" 2>/dev/null || true
pkill -f "node.*3000" 2>/dev/null || true
pkill -f "node.*3001" 2>/dev/null || true

echo ""
echo "✅ LIMPIEZA COMPLETADA"
echo "Estado actual:"
docker ps -a