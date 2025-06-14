import os
import sys
import subprocess
from loguru import logger

def download_onnx_model():
    """Descargar modelo ONNX si no existe"""
    onnx_path = "/app/models/yolov8n.onnx"
    
    if not os.path.exists(onnx_path):
        logger.info("Descargando modelo YOLOv8n ONNX...")
        try:
            import urllib.request
            url = "https://github.com/ultralytics/assets/releases/download/v0.0.0/yolov8n.onnx"
            urllib.request.urlretrieve(url, onnx_path)
            logger.info(f"Modelo descargado: {onnx_path}")
        except Exception as e:
            logger.error(f"Error descargando modelo: {e}")
            return False
    
    return True

def convert_to_rknn():
    """Convertir modelo ONNX a RKNN"""
    try:
        from rknn.api import RKNN
        
        onnx_path = "/app/models/yolov8n.onnx"
        rknn_path = "/app/models/yolov8n.rknn"
        
        if not os.path.exists(onnx_path):
            logger.error(f"Modelo ONNX no encontrado: {onnx_path}")
            return False
        
        logger.info("Iniciando conversión ONNX -> RKNN...")
        
        # Crear instancia RKNN
        rknn = RKNN(verbose=True)
        
        # Configurar modelo
        logger.info("Configurando modelo...")
        ret = rknn.config(
            mean_values=[[0, 0, 0]],
            std_values=[[255, 255, 255]],
            target_platform='rk3588'
        )
        
        if ret != 0:
            logger.error("Error en configuración RKNN")
            return False
        
        # Cargar modelo ONNX
        logger.info("Cargando modelo ONNX...")
        ret = rknn.load_onnx(model=onnx_path)
        
        if ret != 0:
            logger.error("Error cargando modelo ONNX")
            return False
        
        # Construir modelo
        logger.info("Construyendo modelo RKNN...")
        ret = rknn.build(do_quantization=True)
        
        if ret != 0:
            logger.error("Error construyendo modelo")
            return False
        
        # Exportar modelo
        logger.info(f"Exportando modelo a {rknn_path}...")
        ret = rknn.export_rknn(rknn_path)
        
        if ret != 0:
            logger.error("Error exportando modelo")
            return False
        
        rknn.release()
        
        logger.info("✅ Conversión completada exitosamente")
        return True
        
    except ImportError:
        logger.error("RKNN toolkit no disponible - usando modelo ONNX")
        return False
    except Exception as e:
        logger.error(f"Error en conversión: {e}")
        return False

def main():
    """Función principal"""
    logger.info("🔧 Iniciando conversión de modelo...")
    
    # Crear directorio de modelos
    os.makedirs("/app/models", exist_ok=True)
    
    # Descargar modelo ONNX
    if not download_onnx_model():
        logger.error("No se pudo descargar el modelo ONNX")
        sys.exit(1)
    
    # Convertir a RKNN
    if not convert_to_rknn():
        logger.warning("Conversión RKNN falló - usando ONNX como fallback")
    
    logger.info("🎉 Proceso de modelo completado")

if __name__ == "__main__":
    main()