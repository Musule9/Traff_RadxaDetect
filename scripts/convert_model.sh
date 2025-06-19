#!/bin/bash

set -e

echo "🚀 Actualizando repositorios e instalando dependencias..."
sudo apt update
sudo apt install -y python3-venv python3-pip python3-opencv python3-numpy python3-setuptools git wget

echo "🛠️ Creando entorno virtual..."
python3 -m venv ~/rknn-env
source ~/rknn-env/bin/activate

echo "⬆️ Actualizando pip y setuptools..."
pip install --upgrade pip setuptools

echo "📦 Instalando paquetes necesarios..."
pip install ultralytics rknn-toolkit-lite2==2.3.2 onnx onnxruntime onnxsim

echo "📥 Descargando modelo YOLOv8n preentrenado..."
wget -nc https://github.com/ultralytics/assets/releases/download/v8.3.0/yolov8n.pt -P ~/models/

echo "✅ Entorno preparado."
echo "Para activar el entorno virtual usa: source ~/rknn-env/bin/activate"
echo "Luego ejecuta tu script de exportación y conversión."

# Opcional: ejemplo mínimo para exportar y convertir
cat << 'EOF' > ~/export_convert.py
import os
from ultralytics import YOLO
from rknnlite.api import RKNNLite
from loguru import logger

def export_onnx(model_path, onnx_path):
    logger.info("Exportando modelo ONNX...")
    model = YOLO(model_path)
    model.export(format='onnx')
    if not os.path.exists("yolov8n.onnx"):
        logger.error("No se pudo exportar ONNX")
        return False
    os.rename("yolov8n.onnx", onnx_path)
    logger.info(f"Modelo ONNX guardado en {onnx_path}")
    return True

def convert_rknn(onnx_path, rknn_path):
    rknn = RKNNLite()
    logger.info("Cargando modelo ONNX...")
    ret = rknn.load_onnx(onnx_path)
    if ret != 0:
        logger.error("Error cargando ONNX")
        return False
    logger.info("Construyendo modelo RKNN...")
    ret = rknn.build_rknn(target_platform="rk3588", do_quantization=True, mean_values=[[0,0,0]], std_values=[[255,255,255]])
    if ret != 0:
        logger.error("Error construyendo RKNN")
        return False
    logger.info(f"Exportando modelo RKNN a {rknn_path}...")
    ret = rknn.export_rknn(rknn_path)
    if ret != 0:
        logger.error("Error exportando RKNN")
        return False
    rknn.release()
    logger.info("Modelo RKNN creado exitosamente")
    return True

def main():
    os.makedirs("models", exist_ok=True)
    onnx_path = "models/yolov8n.onnx"
    rknn_path = "models/yolov8n.rknn"

    if not export_onnx("models/yolov8n.pt", onnx_path):
        return
    if not convert_rknn(onnx_path, rknn_path):
        return

if __name__ == "__main__":
    main()
EOF

echo "📄 Script de ejemplo 'export_convert.py' creado en tu home."
echo "Ejecuta: source ~/rknn-env/bin/activate && python ~/export_convert.py"
