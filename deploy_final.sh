#!/bin/bash

# EJECUTAR ESTE SCRIPT: ./install_rknn_lib.sh

echo "🔧 INSTALANDO librknnrt.so PARA RK3588"
echo "======================================"

# Verificar si ya existe
if [ -f "/usr/lib/librknnrt.so" ]; then
    echo "✅ librknnrt.so ya está instalado"
    ls -la /usr/lib/librknnrt.so
    exit 0
fi

echo "📥 Descargando librknnrt.so para RK3588..."

# Crear directorio temporal
mkdir -p /tmp/rknn_install
cd /tmp/rknn_install

# Descargar librknnrt.so para aarch64 (RK3588)
echo "🌐 Descargando desde GitHub oficial..."
curl -L -o librknnrt.so \
    "https://github.com/airockchip/rknn-toolkit2/raw/master/rknpu2/runtime/Linux/librknn_api/aarch64/librknnrt.so"

# Verificar descarga
if [ ! -f "librknnrt.so" ]; then
    echo "❌ Error: No se pudo descargar librknnrt.so"
    echo "💡 Intente descarga manual desde:"
    echo "   https://github.com/airockchip/rknn-toolkit2/tree/master/rknpu2/runtime/Linux/librknn_api/aarch64"
    exit 1
fi

# Verificar tamaño del archivo
SIZE=$(stat -c%s "librknnrt.so")
if [ $SIZE -lt 100000 ]; then
    echo "❌ Error: Archivo descargado muy pequeño ($SIZE bytes)"
    exit 1
fi

echo "✅ Descarga exitosa: $(stat -c%s librknnrt.so) bytes"

# Copiar a /usr/lib
echo "📂 Instalando en /usr/lib..."
sudo cp librknnrt.so /usr/lib/
sudo chmod +x /usr/lib/librknnrt.so

# Verificar instalación
if [ -f "/usr/lib/librknnrt.so" ]; then
    echo "✅ librknnrt.so instalado exitosamente"
    ls -la /usr/lib/librknnrt.so
    
    # Verificar que es el correcto
    echo "🔍 Verificando librería..."
    ldd /usr/lib/librknnrt.so | head -5
    
    echo ""
    echo "🎉 INSTALACIÓN COMPLETADA"
    echo "   Reinicie el contenedor Docker para aplicar cambios"
    echo ""
    echo "🔄 Comandos para reiniciar:"
    echo "   docker stop vehicle-detection-rknn"
    echo "   docker start vehicle-detection-rknn"
    
else
    echo "❌ Error: No se pudo instalar librknnrt.so"
    exit 1
fi

# Limpiar
cd /
rm -rf /tmp/rknn_install

echo "✅ Instalación de librknnrt.so completada"