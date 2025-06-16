#!/bin/bash
set -e

echo "🔧 REPARANDO ESTRUCTURA DEL PROYECTO"
echo "====================================="

PROJECT_DIR="/opt/vehicle-detection"
cd "$PROJECT_DIR"

# 1. Detener contenedor actual
echo "🛑 Deteniendo contenedor actual..."
sudo -u vehicle-detection docker-compose down 2>/dev/null || true

# 2. Crear estructura correcta de directorios
echo "📁 Creando estructura correcta..."
sudo mkdir -p app/{core,services,api,utils}
sudo mkdir -p frontend/{src,public}
sudo mkdir -p {config,data,models,logs,scripts,tests}

# 3. Crear archivos __init__.py necesarios
echo "🐍 Creando archivos __init__.py..."
sudo touch app/__init__.py
sudo touch app/core/__init__.py
sudo touch app/services/__init__.py
sudo touch app/api/__init__.py
sudo touch app/utils/__init__.py

# 4. Verificar que todos los archivos .py están en su lugar
echo "📋 Verificando archivos Python existentes..."
if [ -f "app/core/analyzer.py" ]; then
    echo "✅ app/core/analyzer.py"
else
    echo "❌ app/core/analyzer.py - FALTA"
fi

if [ -f "app/core/database.py" ]; then
    echo "✅ app/core/database.py"
else
    echo "❌ app/core/database.py - FALTA"
fi

if [ -f "app/core/detector.py" ]; then
    echo "✅ app/core/detector.py"
else
    echo "❌ app/core/detector.py - FALTA"
fi

if [ -f "app/core/tracker.py" ]; then
    echo "✅ app/core/tracker.py"
else
    echo "❌ app/core/tracker.py - FALTA"
fi

if [ -f "app/core/video_processor.py" ]; then
    echo "✅ app/core/video_processor.py"
else
    echo "❌ app/core/video_processor.py - FALTA"
fi

if [ -f "app/services/auth_service.py" ]; then
    echo "✅ app/services/auth_service.py"
else
    echo "❌ app/services/auth_service.py - FALTA"
fi

if [ -f "app/services/controller_service.py" ]; then
    echo "✅ app/services/controller_service.py"
else
    echo "❌ app/services/controller_service.py - FALTA"
fi

# 5. Configurar permisos
echo "🔐 Configurando permisos..."
sudo chown -R vehicle-detection:vehicle-detection "$PROJECT_DIR"

echo "✅ Estructura del proyecto reparada"