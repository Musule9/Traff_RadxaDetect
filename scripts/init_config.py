import json
import os
from datetime import datetime

def create_default_config():
    """Crear configuración por defecto"""
    
    # Configuración del sistema
    system_config = {
        "app_name": "Vehicle Detection System",
        "version": "1.0.0",
        "model_path": "/app/models/yolov8n.rknn",
        "confidence_threshold": 0.5,
        "high_threshold": 0.6,
        "low_threshold": 0.1,
        "max_age": 30,
        "night_vision_enhancement": True,
        "show_overlay": True,
        "data_retention_days": 30,
        "max_cameras": 1,
        "target_fps": 30,
        "stream_resolution": {
            "input": [1920, 1080],
            "display": [1280, 720]
        },
        "authentication": {
            "enabled": True,
            "default_username": "admin",
            "default_password": "admin123",
            "session_timeout": 3600
        }
    }
    
    # Configuración de cámara por defecto
    camera_config = {
        "camera_1": {
            "id": "camera_1",
            "name": "Cámara Principal",
            "rtsp_url": "rtsp://admin:password@192.168.1.100:554/stream1",
            "fase": "fase1",
            "direccion": "norte",
            "controladora_id": "CTRL_001",
            "controladora_ip": "192.168.1.200",
            "enabled": True,
            "lines": [],
            "zones": []
        }
    }
    
    # Configuración de análisis
    analysis_config = {
        "lines": {
            "line_1": {
                "id": "line_1",
                "name": "Línea Carril 1 - Conteo",
                "points": [[100, 300], [400, 300]],
                "lane": "carril_1",
                "line_type": "counting",
                "distance_to_next": 10.0
            },
            "line_2": {
                "id": "line_2", 
                "name": "Línea Carril 1 - Velocidad",
                "points": [[100, 250], [400, 250]],
                "lane": "carril_1",
                "line_type": "speed",
                "distance_to_next": None
            }
        },
        "zones": {
            "red_zone_1": {
                "id": "red_zone_1",
                "name": "Zona Semáforo Rojo",
                "points": [[150, 200], [350, 200], [350, 400], [150, 400]],
                "zone_type": "red_light"
            }
        }
    }
    
    # Configuración de controladora
    controller_config = {
        "controllers": {
            "CTRL_001": {
                "id": "CTRL_001",
                "name": "Controladora Principal",
                "ip": "192.168.1.200",
                "port": 8080,
                "endpoints": {
                    "analytic": "/api/analitico",
                    "status": "/api/analiticos"
                },
                "phases": ["fase1", "fase2", "fase3", "fase4"]
            }
        }
    }
    
    # Crear archivos de configuración
    config_files = {
        "system.json": system_config,
        "cameras.json": camera_config,
        "analysis.json": analysis_config,
        "controllers.json": controller_config
    }
    
    config_dir = "/app/config"
    os.makedirs(config_dir, exist_ok=True)
    
    for filename, config in config_files.items():
        filepath = os.path.join(config_dir, filename)
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(config, f, indent=2, ensure_ascii=False)
        print(f"✅ Configuración creada: {filepath}")

if __name__ == "__main__":
    create_default_config()
    print("🎉 Configuración inicial completada")