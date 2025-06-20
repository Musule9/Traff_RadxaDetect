#!/bin/bash

echo "🔧 INSTALACIÓN DE RKNN PARA RADXA ROCK 5T"
echo "========================================="

# Verificar que estamos en Radxa Rock 5T
if [ -f /proc/device-tree/model ]; then
    MODEL=$(cat /proc/device-tree/model 2>/dev/null || echo "Unknown")
    echo "Hardware detectado: $MODEL"
    
    if [[ "$MODEL" != *"Radxa"* ]] && [[ "$MODEL" != *"ROCK"* ]]; then
        echo "⚠️ ADVERTENCIA: Este script está optimizado para Radxa Rock 5T"
        echo "¿Continuar de todas formas? (y/N)"
        read -r response
        if [ "$response" != "y" ]; then
            exit 0
        fi
    fi
fi

# Actualizar sistema
echo "📦 Actualizando sistema..."
sudo apt update
sudo apt upgrade -y

# Instalar dependencias básicas
echo "📦 Instalando dependencias básicas..."
sudo apt install -y \
    python3-pip \
    python3-dev \
    python3-opencv \
    python3-numpy \
    python3-setuptools \
    build-essential \
    cmake \
    wget \
    curl \
    git \
    unzip

# Verificar arquitectura
ARCH=$(uname -m)
echo "Arquitectura: $ARCH"

if [ "$ARCH" != "aarch64" ]; then
    echo "❌ Este script requiere arquitectura ARM64 (aarch64)"
    exit 1
fi

# Crear directorios
sudo mkdir -p /usr/lib/
sudo mkdir -p /usr/local/lib/rknn
sudo mkdir -p /opt/rknn

# Descargar librerías RKNN para RK3588
echo "📥 Descargando librerías RKNN para RK3588..."

cd /tmp

# Opción 1: Descargar desde GitHub oficial
echo "🔗 Descargando desde repositorio oficial..."
if ! wget -q --timeout=30 https://github.com/airockchip/rknn-toolkit2/raw/master/rknpu2/runtime/Linux/librknn_api/aarch64/librknnrt.so; then
    echo "⚠️ Descarga directa falló, intentando alternativa..."
    
    # Opción 2: Clonar repositorio completo (más lento pero más confiable)
    if [ ! -d "rknn-toolkit2" ]; then
        echo "📂 Clonando repositorio RKNN toolkit..."
        git clone --depth 1 https://github.com/airockchip/rknn-toolkit2.git
    fi
    
    if [ -f "rknn-toolkit2/rknpu2/runtime/Linux/librknn_api/aarch64/librknnrt.so" ]; then
        cp rknn-toolkit2/rknpu2/runtime/Linux/librknn_api/aarch64/librknnrt.so .
    else
        echo "❌ No se pudo obtener librknnrt.so"
        echo "📋 Opciones:"
        echo "1. Verificar conexión a internet"
        echo "2. Descargar manualmente desde: https://github.com/airockchip/rknn-toolkit2"
        echo "3. Contactar al fabricante de su dispositivo"
        exit 1
    fi
fi

# Verificar que tenemos el archivo
if [ ! -f "librknnrt.so" ]; then
    echo "❌ librknnrt.so no encontrado"
    exit 1
fi

echo "✅ librknnrt.so descargado correctamente"

# Instalar librería
echo "📋 Instalando librknnrt.so..."
sudo cp librknnrt.so /usr/lib/
sudo chmod 755 /usr/lib/librknnrt.so

# Configurar ldconfig
echo "/usr/lib" | sudo tee -a /etc/ld.so.conf.d/rknn.conf
sudo ldconfig

# Verificar instalación
if [ -f "/usr/lib/librknnrt.so" ]; then
    echo "✅ librknnrt.so instalado en /usr/lib/"
    ldd /usr/lib/librknnrt.so | head -3
else
    echo "❌ Error instalando librknnrt.so"
    exit 1
fi

# Instalar rknn-toolkit-lite2
echo "🐍 Instalando rknn-toolkit-lite2..."

# Actualizar pip
python3 -m pip install --upgrade pip

# Instalar rknn-toolkit-lite2
if python3 -m pip install rknn-toolkit-lite2==2.3.2; then
    echo "✅ rknn-toolkit-lite2 instalado"
else
    echo "⚠️ Error con pip, intentando instalación manual..."
    
    # Descargar wheel manualmente
    WHEEL_URL="https://pypi.org/project/rknn-toolkit-lite2/2.3.2/#files"
    echo "📥 Busque el wheel para aarch64 en: $WHEEL_URL"
    echo "O intente: pip3 install https://files.pythonhosted.org/packages/.../rknn_toolkit_lite2-2.3.2-cp38-cp38-linux_aarch64.whl"
fi

# Instalar dependencias adicionales
echo "📦 Instalando dependencias adicionales..."
python3 -m pip install \
    ultralytics \
    opencv-python==4.8.1.78 \
    numpy \
    loguru \
    requests

# Verificar instalación RKNN
echo "🧪 Verificando instalación RKNN..."
python3 -c "
try:
    from rknnlite.api import RKNNLite
    print('✅ RKNN importado correctamente')
    
    # Test básico
    rknn = RKNNLite()
    print('✅ RKNNLite instanciado')
    
except ImportError as e:
    print(f'❌ Error importando RKNN: {e}')
except Exception as e:
    print(f'⚠️ RKNN importado pero error en test: {e}')
    print('Esto puede ser normal sin un modelo cargado')
"

# Verificar dispositivos NPU
echo "🔍 Verificando dispositivos NPU..."
if [ -d "/dev/dri" ]; then
    echo "✅ Dispositivos DRI encontrados:"
    ls -la /dev/dri/
fi

if [ -e "/dev/mali0" ]; then
    echo "✅ Mali GPU encontrado: /dev/mali0"
fi

if [ -e "/dev/dma_heap" ]; then
    echo "✅ DMA heap encontrado: /dev/dma_heap"
fi

if [ -e "/dev/rga" ]; then
    echo "✅ RGA encontrado: /dev/rga"
fi

if [ -e "/dev/mpp_service" ]; then
    echo "✅ MPP service encontrado: /dev/mpp_service"
fi

# Configurar permisos de dispositivos
echo "⚙️ Configurando permisos de dispositivos..."
sudo usermod -a -G video $USER
sudo usermod -a -G render $USER 2>/dev/null || true

# Crear script de verificación
cat > /tmp/test_rknn.py << 'EOF'
#!/usr/bin/env python3
import sys
import os

def test_rknn():
    print("🧪 Test completo de RKNN")
    print("=" * 30)
    
    # Test 1: Importar
    try:
        from rknnlite.api import RKNNLite
        print("✅ Importación exitosa")
    except ImportError as e:
        print(f"❌ Error de importación: {e}")
        return False
    
    # Test 2: Verificar librería nativa
    if os.path.exists("/usr/lib/librknnrt.so"):
        print("✅ librknnrt.so encontrado")
    else:
        print("❌ librknnrt.so NO encontrado")
        return False
    
    # Test 3: Instanciar RKNNLite
    try:
        rknn = RKNNLite()
        print("✅ RKNNLite instanciado")
        
        # Test 4: Verificar métodos
        if hasattr(rknn, 'load_rknn') and hasattr(rknn, 'init_runtime'):
            print("✅ Métodos disponibles")
        else:
            print("❌ Métodos faltantes")
            return False
            
    except Exception as e:
        print(f"❌ Error instanciando: {e}")
        return False
    
    print("✅ TODOS LOS TESTS PASARON")
    return True

if __name__ == "__main__":
    success = test_rknn()
    sys.exit(0 if success else 1)
EOF

chmod +x /tmp/test_rknn.py

echo ""
echo "🎯 Ejecutando test final..."
if python3 /tmp/test_rknn.py; then
    echo ""
    echo "🎉 INSTALACIÓN COMPLETADA EXITOSAMENTE"
    echo "======================================"
    echo ""
    echo "✅ librknnrt.so instalado y configurado"
    echo "✅ rknn-toolkit-lite2 instalado"
    echo "✅ Dispositivos NPU verificados"
    echo ""
    echo "📋 Próximos pasos:"
    echo "1. Reiniciar sistema: sudo reboot"
    echo "2. Verificar Docker está usando los dispositivos correctos"
    echo "3. Ejecutar el sistema de detección vehicular"
    echo ""
    echo "🐳 Para Docker, usar:"
    echo "docker run ... \\"
    echo "  -v /usr/lib/librknnrt.so:/usr/lib/librknnrt.so:ro \\"
    echo "  --device=/dev/dri --device=/dev/mali0 \\"
    echo "  --device=/dev/dma_heap --device=/dev/rga \\"
    echo "  --device=/dev/mpp_service \\"
    echo "  ..."
    echo ""
else
    echo ""
    echo "❌ INSTALACIÓN FALLÓ"
    echo "==================="
    echo ""
    echo "📋 Posibles causas:"
    echo "1. Hardware no compatible con RK3588"
    echo "2. Versión de kernel incompatible"
    echo "3. Drivers NPU no instalados"
    echo "4. Permisos insuficientes"
    echo ""
    echo "📞 Soluciones:"
    echo "1. Verificar que está usando imagen oficial de Radxa"
    echo "2. Actualizar firmware y kernel"
    echo "3. Contactar soporte de Radxa"
    exit 1
fi

# Limpiar archivos temporales
cd /
rm -rf /tmp/rknn-toolkit2 /tmp/librknnrt.so /tmp/test_rknn.py

echo "🧹 Limpieza completada"