import requests
import sys
import json
from loguru import logger

def check_api_health():
    """Verificar salud de la API"""
    try:
        response = requests.get("http://localhost:8000/api/camera_health", timeout=5)
        if response.status_code == 200:
            data = response.json()
            if data.get("healthy", False):
                logger.info("✅ API saludable")
                return True
            else:
                logger.error("❌ API reporta problemas")
                return False
        else:
            logger.error(f"❌ API responde con código: {response.status_code}")
            return False
    except Exception as e:
        logger.error(f"❌ Error conectando con API: {e}")
        return False

def check_camera_status():
    """Verificar estado de cámara"""
    try:
        response = requests.get("http://localhost:8000/api/camera/status", timeout=5)
        if response.status_code == 200:
            data = response.json()
            if data.get("connected", False) and data.get("fps", 0) > 0:
                logger.info(f"✅ Cámara conectada - FPS: {data['fps']}")
                return True
            else:
                logger.error("❌ Cámara desconectada o sin FPS")
                return False
        else:
            logger.error(f"❌ Error verificando cámara: {response.status_code}")
            return False
    except Exception as e:
        logger.error(f"❌ Error verificando cámara: {e}")
        return False

def main():
    """Función principal de health check"""
    logger.info("🔍 Iniciando verificación de salud del sistema...")
    
    health_checks = [
        ("API", check_api_health),
        ("Cámara", check_camera_status)
    ]
    
    failed_checks = []
    
    for name, check_func in health_checks:
        logger.info(f"Verificando {name}...")
        if not check_func():
            failed_checks.append(name)
    
    if failed_checks:
        logger.error(f"❌ Fallos en: {', '.join(failed_checks)}")
        sys.exit(1)
    else:
        logger.info("✅ Todos los sistemas funcionando correctamente")
        sys.exit(0)

if __name__ == "__main__":
    main()