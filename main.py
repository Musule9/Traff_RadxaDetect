import os
import asyncio
import json
from datetime import datetime, timedelta
from typing import Optional, Dict, List
from contextlib import asynccontextmanager
import uvicorn

from fastapi import FastAPI, HTTPException, Depends, status, Request
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse, JSONResponse, FileResponse
import uvicorn
from pydantic import BaseModel
from loguru import logger
import cv2
import numpy as np
import threading
import time
import sys
from pathlib import Path


# ============================================================================
# CONFIGURACIÓN DE LOGGING CORREGIDA
# ============================================================================
def setup_logging():
    """Configurar sistema de logging"""
    log_level = os.getenv('LOG_LEVEL', 'info').lower()
    
    # Mapear niveles válidos para uvicorn
    valid_levels = {
        'debug': 'debug',
        'info': 'info', 
        'warning': 'warning',
        'error': 'error',
        'critical': 'critical'
    }
    
    # Usar nivel válido o por defecto 'info'
    uvicorn_level = valid_levels.get(log_level, 'info')
    
    logger.remove()
    
    os.makedirs("/app/logs", exist_ok=True)
    logger.add(
        "/app/logs/app.log",
        rotation="10 MB",
        retention="7 days",
        level=log_level.upper(),
        format="{time:YYYY-MM-DD HH:mm:ss} | {level} | {name}:{function}:{line} - {message}"
    )
    
    logger.add(
        sys.stdout,
        level=log_level.upper(),
        format="<green>{time:HH:mm:ss}</green> | <level>{level}</level> | {message}"
    )
    
    return uvicorn_level

LOG_LEVEL = setup_logging()

# ============================================================================
# IMPORTAR MÓDULOS DE LA APLICACIÓN CON MANEJO ROBUSTO DE ERRORES
# ============================================================================
video_processor = None
db_manager = None
auth_service = None
controller_service = None
MODULES_AVAILABLE = False

def import_app_modules():
    """Importar módulos de la aplicación con manejo robusto de errores"""
    global video_processor, db_manager, auth_service, controller_service, MODULES_AVAILABLE
    
    try:
        # Verificar que los archivos existen
        required_files = [
            "/app/app/__init__.py",
            "/app/app/core/__init__.py", 
            "/app/app/core/database.py",
            "/app/app/services/__init__.py",
            "/app/app/services/auth_service.py"
        ]
        
        missing_files = []
        for file_path in required_files:
            if not os.path.exists(file_path):
                missing_files.append(file_path)
        
        if missing_files:
            logger.error(f"❌ Archivos faltantes: {missing_files}")
            logger.warning("🔄 Continuando con funcionalidad básica...")
            return False
        
        # Importar módulos principales
        from app.core.database import DatabaseManager
        from app.services.auth_service import AuthService
        from app.services.controller_service import ControllerService
        
        # Inicializar servicios básicos
        db_manager = DatabaseManager()
        auth_service = AuthService()
        controller_service = ControllerService()
        
        # Intentar importar video processor (específico para Radxa + RKNN)
        try:
            from app.core.video_processor import VideoProcessor
            logger.info("✅ VideoProcessor con soporte RKNN disponible")
        except ImportError as e:
            logger.warning(f"⚠️ VideoProcessor no disponible: {e}")
        
        MODULES_AVAILABLE = True
        logger.info("✅ Módulos de aplicación cargados correctamente")
        return True
        
    except Exception as e:
        logger.error(f"❌ Error importando módulos: {e}")
        logger.info("🔄 Continuando con funcionalidad básica...")
        return False

# Intentar importar módulos al inicio
import_app_modules()

# ============================================================================
# MODELOS PYDANTIC (MANTIENEN TODA LA FUNCIONALIDAD)
# ============================================================================
class LoginRequest(BaseModel):
    username: str
    password: str

class CameraConfig(BaseModel):
    # Básicos (originales)
    rtsp_url: str
    fase: str = "fase1"
    direccion: str = "norte"
    controladora_id: str = "CTRL_001"
    controladora_ip: str = "192.168.1.200"
    
    # Identificación de cámara
    camera_name: Optional[str] = ""
    camera_model: Optional[str] = ""
    camera_location: Optional[str] = ""
    camera_serial: Optional[str] = ""
    
    # Configuración de red
    camera_ip: Optional[str] = ""
    username: str = "admin"
    password: Optional[str] = ""
    port: str = "554"
    stream_path: str = "/stream1"
    
    # Configuración de video
    resolution: str = "1920x1080"
    frame_rate: str = "30"
    bitrate: str = "4000"
    encoding: str = "H264"
    stream_quality: str = "high"
    
    # Configuraciones avanzadas
    night_vision: bool = False
    motion_detection: bool = False
    recording_enabled: bool = False
    audio_enabled: bool = False
    
    # Configuración de análisis
    detection_zones: bool = True
    speed_calculation: bool = True
    vehicle_counting: bool = True
    license_plate_recognition: bool = False
    
    # Estado
    enabled: bool = False

class LineConfig(BaseModel):
    id: str
    name: str
    points: List[List[int]]
    lane: str
    line_type: str
    distance_to_next: Optional[float] = None
    speed_line_id: Optional[str] = None
    counting_line_id: Optional[str] = None
    speed_line_distance: Optional[float] = None
    direction: Optional[str] = None

class ZoneConfig(BaseModel):
    id: str
    name: str
    points: List[List[int]]
    zone_type: str = "red_light"

class SystemConfig(BaseModel):
    confidence_threshold: float = 0.5
    night_vision_enhancement: bool = True
    show_overlay: bool = True
    data_retention_days: int = 30
    target_fps: int = 30
    log_level: str = "INFO"

# ============================================================================
# FUNCIONES AUXILIARES (MANTIENEN FUNCIONALIDAD ESPECÍFICA RADXA)
# ============================================================================
def load_system_config() -> Dict:
    """Cargar configuración del sistema"""
    try:
        with open("/app/config/system.json", "r") as f:
            return json.load(f)
    except Exception as e:
        logger.error(f"Error cargando configuración del sistema: {e}")
        return {
            "confidence_threshold": 0.5,
            "night_vision_enhancement": True,
            "show_overlay": True,
            "data_retention_days": 30,
            "target_fps": 30,
            "use_rknn": True,
            "hardware": "radxa-rock-5t"
        }

def load_camera_config() -> Dict:
    """Cargar configuración de cámara"""
    try:
        with open("/app/config/cameras.json", "r") as f:
            cameras = json.load(f)
            for camera in cameras.values():
                if camera.get("enabled", False):
                    return camera
            return {}
    except Exception as e:
        logger.error(f"Error cargando configuración de cámaras: {e}")
        return {
            "rtsp_url": "",
            "fase": "fase1",
            "direccion": "norte",
            "enabled": False,
            "lane_detection": True,
            "speed_calculation": True,
            "red_zone_detection": True
        }

async def controller_callback(action: str, data: Dict):
    """Callback para comunicación con controladora TICSA"""
    if action == "send_analytic" and controller_service:
        await controller_service.send_analytic(data)

# ============================================================================
# LIFESPAN MANAGER (NUEVO SISTEMA SIN DEPRECATION WARNINGS)
# ============================================================================
@asynccontextmanager
async def lifespan(app: FastAPI):
    """Gestor de ciclo de vida de la aplicación - SIN DEPRECATION WARNINGS"""
    global video_processor
    
    try:
        logger.info("🚀 Iniciando servicios del sistema...")
        
        # Detectar hardware Radxa Rock 5T
        hardware_info = "Unknown"
        if os.path.exists("/proc/device-tree/model"):
            with open("/proc/device-tree/model", "rb") as f:
                hardware_info = f.read().decode('utf-8', errors='ignore').strip('\x00')
        
        logger.info(f"📋 Hardware detectado: {hardware_info}")
        
        # Inicializar base de datos si está disponible
        if db_manager:
            await db_manager.init_daily_database()
            logger.info("✅ Base de datos SQLite inicializada")
        
        # Inicializar video processor con soporte RKNN para Radxa Rock 5T
        if MODULES_AVAILABLE:
            try:
                from app.core.video_processor import VideoProcessor
                camera_config = load_camera_config()
                system_config = load_system_config()
                
                video_processor = VideoProcessor(
                    camera_config=camera_config,
                    system_config=system_config,
                    db_manager=db_manager,
                    callback_func=controller_callback
                )
                
                await video_processor.initialize()
                
                if camera_config.get("rtsp_url"):
                    video_processor.start_processing()
                    logger.info("✅ Procesamiento de video con RKNN iniciado")
                else:
                    logger.info("⚠️ URL RTSP no configurada - esperando configuración")
                    
            except Exception as e:
                logger.warning(f"⚠️ Video processor no disponible: {e}")
                logger.info("🔄 Sistema funcionará sin procesamiento de video")
        
        # Tareas en background para sistema completo
        asyncio.create_task(daily_cleanup_task())
        if MODULES_AVAILABLE:
            asyncio.create_task(traffic_light_update_task())
        
        logger.info("✅ Sistema completo inicializado correctamente")
        logger.info("🌐 API disponible en puerto 8000")
        
        yield
        
    except Exception as e:
        logger.error(f"Error en inicialización: {e}")
        yield
    finally:
        # Limpieza al cerrar
        if video_processor:
            video_processor.stop_processing()
        if controller_service:
            await controller_service.close()
        logger.info("🔽 Servicios finalizados")

# ============================================================================
# CREAR APLICACIÓN FASTAPI (SIN DEPRECATION WARNINGS)
# ============================================================================
app = FastAPI()

# Middleware CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:8000",
        "http://127.0.0.1:8000",
        "http://0.0.0.0:8000"
    ],
    allow_credentials=True,
    allow_methods=["GET", "POST", "PUT", "DELETE"],
    allow_headers=["*"],
)


# Seguridad
security = HTTPBearer()

async def verify_token(credentials: HTTPAuthorizationCredentials = Depends(security)):
    """Verificar token de autenticación"""
    if not auth_service or not auth_service.verify_token(credentials.credentials):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Token inválido",
            headers={"WWW-Authenticate": "Bearer"},
        )
    return credentials.credentials

# ============================================================================
# CONFIGURACIÓN DE FRONTEND COMPLETO
# ============================================================================
if os.path.exists("/app/frontend/build"):
    app.mount("/static", StaticFiles(directory="/app/frontend/build/static"), name="static")

# ============================================================================
# RUTAS DE API COMPLETAS (MANTIENEN TODA LA FUNCIONALIDAD)
# ============================================================================

# Autenticación
@app.post("/api/auth/login")
async def login(request: LoginRequest):
    """Iniciar sesión"""
    if not auth_service:
        if request.username == "admin" and request.password == "admin123":
            return {"token": "development_token", "message": "Login exitoso"}
        raise HTTPException(status_code=401, detail="Credenciales inválidas")
    
    token = await auth_service.authenticate(request.username, request.password)
    if not token:
        raise HTTPException(status_code=401, detail="Credenciales inválidas")
    
    return {"token": token, "message": "Login exitoso"}

@app.post("/api/auth/logout")
async def logout(token: str = Depends(verify_token)):
    """Cerrar sesión"""
    if auth_service:
        auth_service.revoke_token(token)
    return {"message": "Logout exitoso"}

# Health check con información específica de Radxa Rock 5T
@app.get("/api/camera_health")
async def health():
    """Health check con información completa del sistema"""
    
    # Verificar estado de la cámara
    camera_connected = False
    camera_fps = 0
    if video_processor and video_processor.is_running:
        camera_connected = True
        camera_fps = video_processor.current_fps
    
    # Verificar estado de la controladora
    controller_connected = False
    if controller_service:
        try:
            status = await controller_service.get_traffic_light_status()
            controller_connected = status is not None
        except:
            controller_connected = False
    
    # Información del hardware
    hardware_info = "Unknown"
    rknn_available = False
    
    try:
        if os.path.exists("/proc/device-tree/model"):
            with open("/proc/device-tree/model", "rb") as f:
                hardware_info = f.read().decode('utf-8', errors='ignore').strip('\x00')
        
        # Verificar RKNN
        rknn_available = os.getenv("USE_RKNN", "0") == "1"
        if rknn_available:
            try:
                from rknnlite.api import RKNNLite
                rknn_available = True
            except ImportError:
                rknn_available = False
    except:
        pass
    
    return {
        "status": "healthy" if camera_connected else "warning",
        "timestamp": datetime.now().isoformat(),
        "camera_connected": camera_connected,
        "camera_fps": camera_fps,
        "controller_connected": controller_connected,
        "processing_active": video_processor.is_running if video_processor else False,
        "hardware": hardware_info,
        "rknn_enabled": rknn_available,
        "modules_available": MODULES_AVAILABLE,
        "frontend_available": HAS_FRONTEND,
        "version": "1.0.0",
        "system_time": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    }

@app.get("/api/info")
async def info():
    return {
        "name": "Vehicle Detection System",
        "version": "1.0.0",
        "rknn_enabled": os.getenv("USE_RKNN", "0") == "1"
    }

# Estado de cámara
@app.get("/api/camera/status")
async def get_camera_status():
    """Obtener estado de la cámara"""
    if not video_processor:
        camera_config = load_camera_config()
        return {
            "connected": False, 
            "fps": 0,
            "rtsp_url": camera_config.get("rtsp_url", ""),
            "fase": camera_config.get("fase", "fase1"),
            "direccion": camera_config.get("direccion", "norte")
        }
    
    return {
        "connected": video_processor.is_running,
        "fps": video_processor.current_fps,
        "rtsp_url": video_processor.camera_config.get("rtsp_url", ""),
        "fase": video_processor.camera_config.get("fase", ""),
        "direccion": video_processor.camera_config.get("direccion", "")
    }

# Configuración de cámara
@app.get("/api/camera/config")
async def get_camera_config():
    """Obtener configuración actual de cámara"""
    try:
        os.makedirs("/app/config", exist_ok=True)
        
        try:
            with open("/app/config/cameras.json", "r") as f:
                cameras = json.load(f)
        except:
            cameras = {
                "camera_1": {
                    "id": "camera_1",
                    "name": "Cámara Principal",
                    "rtsp_url": "",
                    "fase": "fase1",
                    "direccion": "norte",
                    "controladora_id": "CTRL_001",
                    "controladora_ip": "192.168.1.200",
                    "enabled": False
                }
            }
            with open("/app/config/cameras.json", "w") as f:
                json.dump(cameras, f, indent=2)
        
        for camera in cameras.values():
            if camera.get("enabled", False):
                return camera
        
        if cameras:
            return list(cameras.values())[0]
        
        return {
            "rtsp_url": "",
            "fase": "fase1",
            "direccion": "norte",
            "controladora_id": "CTRL_001",
            "controladora_ip": "192.168.1.200",
            "enabled": False
        }
        
    except Exception as e:
        logger.error(f"Error obteniendo configuración de cámara: {e}")
        return {
            "rtsp_url": "",
            "fase": "fase1",
            "direccion": "norte", 
            "controladora_id": "CTRL_001",
            "controladora_ip": "192.168.1.200",
            "enabled": False
        }

@app.post("/api/camera/config")
async def update_camera_config(config: CameraConfig):
    """Actualizar configuración de cámara"""
    try:
        os.makedirs("/app/config", exist_ok=True)
        
        try:
            with open("/app/config/cameras.json", "r") as f:
                cameras = json.load(f)
        except:
            cameras = {}
        
        if not cameras:
            cameras = {"camera_1": {"id": "camera_1", "name": "Cámara Principal"}}
        
        camera_key = list(cameras.keys())[0]
        cameras[camera_key].update(config.dict())
        cameras[camera_key]["enabled"] = True
        
        with open("/app/config/cameras.json", "w") as f:
            json.dump(cameras, f, indent=2)
        
        logger.info(f"Configuración de cámara actualizada: {config.dict()}")
        
        global video_processor
        if video_processor and video_processor.is_running:
            logger.info("Reiniciando procesador de video con nueva configuración...")
            video_processor.stop_processing()
            await asyncio.sleep(2)
        
        if video_processor:
            video_processor.camera_config = config.dict()
            if config.rtsp_url:
                video_processor.start_processing()
                logger.info("Procesador de video reiniciado exitosamente")
        
        return {"message": "Configuración actualizada exitosamente", "config": config.dict()}
        
    except Exception as e:
        logger.error(f"Error actualizando configuración de cámara: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/camera/stream")
async def get_camera_stream():
    """Stream de video HTTP optimizado para navegadores web"""
    def generate_frames():
        while True:
            try:
                if video_processor and video_processor.is_running:
                    # Obtener frame procesado con análisis
                    frame = video_processor.get_latest_frame()
                    if frame is not None:
                        # Redimensionar para web (opcional)
                        height, width = frame.shape[:2]
                        if width > 1280:
                            scale = 1280 / width
                            new_width = 1280
                            new_height = int(height * scale)
                            frame = cv2.resize(frame, (new_width, new_height))
                        
                        # Comprimir para web
                        encode_params = [
                            cv2.IMWRITE_JPEG_QUALITY, 85,  # Calidad 85%
                            cv2.IMWRITE_JPEG_OPTIMIZE, 1   # Optimizar
                        ]
                        
                        ret, buffer = cv2.imencode('.jpg', frame, encode_params)
                        if ret:
                            yield (b'--frame\r\n'
                                   b'Content-Type: image/jpeg\r\n'
                                   b'Content-Length: ' + str(len(buffer)).encode() + b'\r\n\r\n' + 
                                   buffer.tobytes() + b'\r\n')
                    else:
                        # Frame placeholder si no hay video
                        yield _generate_placeholder_frame()
                else:
                    # Placeholder cuando no hay cámara configurada
                    yield _generate_placeholder_frame()
                
                # Control de FPS para web (15 FPS es suficiente)
                time.sleep(1/15)
                
            except Exception as e:
                logger.error(f"Error en streaming: {e}")
                yield _generate_error_frame()
                time.sleep(1)
    
    return StreamingResponse(
        generate_frames(),
        media_type="multipart/x-mixed-replace; boundary=frame",
        headers={
            "Cache-Control": "no-cache, no-store, must-revalidate",
            "Pragma": "no-cache",
            "Expires": "0",
            "Connection": "close"
        }
    )

def _generate_placeholder_frame():
    """Generar frame placeholder"""
    placeholder = np.zeros((480, 640, 3), dtype=np.uint8)
    
    # Fondo degradado
    for i in range(480):
        placeholder[i, :] = [20 + (i//10), 25 + (i//10), 35 + (i//10)]
    
    # Texto informativo
    cv2.putText(placeholder, "SISTEMA DE DETECCION VEHICULAR", (80, 200), 
               cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
    cv2.putText(placeholder, "Radxa Rock 5T", (230, 240), 
               cv2.FONT_HERSHEY_SIMPLEX, 0.7, (100, 150, 255), 2)
    cv2.putText(placeholder, "Configure la camara para comenzar", (130, 280), 
               cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1)
    
    ret, buffer = cv2.imencode('.jpg', placeholder, [cv2.IMWRITE_JPEG_QUALITY, 80])
    if ret:
        return (b'--frame\r\n'
                b'Content-Type: image/jpeg\r\n'
                b'Content-Length: ' + str(len(buffer)).encode() + b'\r\n\r\n' + 
                buffer.tobytes() + b'\r\n')
    return b''

def _generate_error_frame():
    """Generar frame de error"""
    error_frame = np.zeros((480, 640, 3), dtype=np.uint8)
    error_frame[:] = [40, 20, 20]  # Fondo rojizo
    
    cv2.putText(error_frame, "ERROR DE CONEXION", (180, 220), 
               cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
    cv2.putText(error_frame, "Verificar configuracion RTSP", (150, 260), 
               cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1)
    
    ret, buffer = cv2.imencode('.jpg', error_frame, [cv2.IMWRITE_JPEG_QUALITY, 80])
    if ret:
        return (b'--frame\r\n'
                b'Content-Type: image/jpeg\r\n'
                b'Content-Length: ' + str(len(buffer)).encode() + b'\r\n\r\n' + 
                buffer.tobytes() + b'\r\n')
    return b''

# AGREGAR TAMBIÉN ESTE ENDPOINT PARA PREVIEW
@app.get("/api/camera/preview")
async def get_camera_preview():
    """Stream de preview sin análisis (más rápido)"""
    def generate_preview():
        while True:
            try:
                if video_processor and video_processor.is_running:
                    # Obtener frame original sin overlay
                    frame = video_processor.get_raw_frame()  # Necesitarás agregar este método
                    if frame is not None:
                        # Redimensionar más pequeño para preview
                        frame = cv2.resize(frame, (320, 240))
                        ret, buffer = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 70])
                        if ret:
                            yield (b'--frame\r\n'
                                   b'Content-Type: image/jpeg\r\n\r\n' + 
                                   buffer.tobytes() + b'\r\n')
                
                time.sleep(1/10)  # 10 FPS para preview
            except:
                time.sleep(1)
    
    return StreamingResponse(
        generate_preview(),
        media_type="multipart/x-mixed-replace; boundary=frame"
    )
     
# Configuración del sistema
@app.get("/api/config/system")
async def get_system_config():
    """Obtener configuración del sistema"""
    return load_system_config()

@app.post("/api/config/system")
async def update_system_config(config: SystemConfig):
    """Actualizar configuración del sistema"""
    try:
        os.makedirs("/app/config", exist_ok=True)
        
        try:
            with open("/app/config/system.json", "r") as f:
                system_config = json.load(f)
        except:
            system_config = {}
        
        system_config.update(config.dict())
        
        with open("/app/config/system.json", "w") as f:
            json.dump(system_config, f, indent=2)
        
        return {"message": "Configuración del sistema actualizada"}
        
    except Exception as e:
        logger.error(f"Error actualizando configuración: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# Análisis - Líneas
@app.get("/api/analysis/lines")
async def get_lines():
    """Obtener todas las líneas de análisis configuradas"""
    try:
        os.makedirs("/app/config", exist_ok=True)
        
        try:
            with open("/app/config/analysis.json", "r") as f:
                analysis = json.load(f)
        except:
            analysis = {"lines": {}, "zones": {}}
            with open("/app/config/analysis.json", "w") as f:
                json.dump(analysis, f, indent=2)
        
        return {"lines": analysis.get("lines", {})}
        
    except Exception as e:
        logger.error(f"Error obteniendo líneas: {e}")
        return {"lines": {}}

@app.post("/api/analysis/lines")
async def add_line(line: LineConfig):
    """Agregar línea de análisis"""
    try:
        os.makedirs("/app/config", exist_ok=True)
        
        try:
            with open("/app/config/analysis.json", "r") as f:
                analysis = json.load(f)
        except:
            analysis = {"lines": {}, "zones": {}}
        
        analysis["lines"][line.id] = line.dict()
        
        with open("/app/config/analysis.json", "w") as f:
            json.dump(analysis, f, indent=2)
        
        if video_processor and video_processor.analyzer and MODULES_AVAILABLE:
            from app.core.analyzer import Line, LineType
            new_line = Line(
                id=line.id,
                name=line.name,
                points=[(p[0], p[1]) for p in line.points],
                lane=line.lane,
                line_type=LineType.COUNTING if line.line_type == "counting" else LineType.SPEED,
                distance_to_next=line.distance_to_next
            )
            video_processor.analyzer.add_line(new_line)
        
        return {"message": "Línea agregada exitosamente"}
        
    except Exception as e:
        logger.error(f"Error agregando línea: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.delete("/api/analysis/lines/{line_id}")
async def delete_line(line_id: str):
    """Eliminar línea de análisis"""
    try:
        os.makedirs("/app/config", exist_ok=True)
        
        try:
            with open("/app/config/analysis.json", "r") as f:
                analysis = json.load(f)
        except:
            analysis = {"lines": {}, "zones": {}}
        
        if line_id in analysis.get("lines", {}):
            del analysis["lines"][line_id]
            
            with open("/app/config/analysis.json", "w") as f:
                json.dump(analysis, f, indent=2)
            
            logger.info(f"Línea eliminada: {line_id}")
            return {"message": "Línea eliminada exitosamente"}
        else:
            raise HTTPException(status_code=404, detail="Línea no encontrada")
            
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error eliminando línea: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# Análisis - Zonas
@app.get("/api/analysis/zones")
async def get_zones():
    """Obtener todas las zonas de análisis configuradas"""
    try:
        os.makedirs("/app/config", exist_ok=True)
        
        try:
            with open("/app/config/analysis.json", "r") as f:
                analysis = json.load(f)
        except:
            analysis = {"lines": {}, "zones": {}}
            with open("/app/config/analysis.json", "w") as f:
                json.dump(analysis, f, indent=2)
        
        return {"zones": analysis.get("zones", {})}
        
    except Exception as e:
        logger.error(f"Error obteniendo zonas: {e}")
        return {"zones": {}}

@app.post("/api/analysis/zones")
async def add_zone(zone: ZoneConfig):
    """Agregar zona de análisis"""
    try:
        os.makedirs("/app/config", exist_ok=True)
        
        try:
            with open("/app/config/analysis.json", "r") as f:
                analysis = json.load(f)
        except:
            analysis = {"lines": {}, "zones": {}}
        
        analysis["zones"][zone.id] = zone.dict()
        
        with open("/app/config/analysis.json", "w") as f:
            json.dump(analysis, f, indent=2)
        
        if video_processor and video_processor.analyzer and MODULES_AVAILABLE:
            from app.core.analyzer import Zone
            new_zone = Zone(
                id=zone.id,
                name=zone.name,
                points=[(p[0], p[1]) for p in zone.points],
                zone_type=zone.zone_type
            )
            video_processor.analyzer.add_zone(new_zone)
        
        return {"message": "Zona agregada exitosamente"}
        
    except Exception as e:
        logger.error(f"Error agregando zona: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.delete("/api/analysis/zones/{zone_id}")
async def delete_zone(zone_id: str):
    """Eliminar zona de análisis"""
    try:
        os.makedirs("/app/config", exist_ok=True)
        
        try:
            with open("/app/config/analysis.json", "r") as f:
                analysis = json.load(f)
        except:
            analysis = {"lines": {}, "zones": {}}
        
        if zone_id in analysis.get("zones", {}):
            del analysis["zones"][zone_id]
            
            with open("/app/config/analysis.json", "w") as f:
                json.dump(analysis, f, indent=2)
            
            logger.info(f"Zona eliminada: {zone_id}")
            return {"message": "Zona eliminada exitosamente"}
        else:
            raise HTTPException(status_code=404, detail="Zona no encontrada")
            
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error eliminando zona: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/analysis/clear")
async def clear_analysis():
    """Limpiar todas las líneas y zonas"""
    try:
        os.makedirs("/app/config", exist_ok=True)
        
        analysis = {"lines": {}, "zones": {}}
        with open("/app/config/analysis.json", "w") as f:
            json.dump(analysis, f, indent=2)
        
        logger.info("Configuración de análisis limpiada")
        return {"message": "Todas las líneas y zonas eliminadas"}
        
    except Exception as e:
        logger.error(f"Error limpiando análisis: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# Exportar datos
@app.get("/api/data/export")
async def export_data(date: str, type: str = "vehicle", fase: str = None):
    """Exportar datos por fecha"""
    if not db_manager:
        return {
            "date": date,
            "type": type,
            "fase": fase,
            "data": [],
            "exported_at": datetime.now().isoformat(),
            "message": "Base de datos no disponible"
        }
    
    try:
        export_date = datetime.strptime(date, "%Y_%m_%d")
        
        if type == "vehicle":
            data = await db_manager.export_vehicle_crossings(date, fase)
        elif type == "red_light":
            data = await db_manager.export_red_light_counts(date, fase)
        elif type == "all":
            vehicle_data = await db_manager.export_vehicle_crossings(date, fase)
            red_light_data = await db_manager.export_red_light_counts(date, fase)
            data = {
                "vehicle_crossings": vehicle_data,
                "red_light_counts": red_light_data
            }
        else:
            raise HTTPException(status_code=400, detail="Tipo de exportación no válido")
        
        return {
            "date": date,
            "type": type,
            "fase": fase,
            "data": data,
            "exported_at": datetime.now().isoformat()
        }
    
    except ValueError:
        raise HTTPException(status_code=400, detail="Formato de fecha inválido (YYYY_MM_DD)")
    except Exception as e:
        logger.error(f"Error exportando datos: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# Controladora
@app.post("/api/rojo_status")
async def update_traffic_light_status(request: Request):
    """Recibir estado de semáforos de la controladora"""
    try:
        data = await request.json()
        fases = data.get("fases", {})
        
        if controller_service:
            controller_service.update_traffic_light_status(fases)
        
        logger.info(f"Estado de semáforos actualizado: {fases}")
        return {"status": "updated", "fases": fases}
        
    except Exception as e:
        logger.error(f"Error actualizando estado de semáforos: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/rojo_status")
async def get_traffic_light_status():
    """Obtener estado actual de semáforos"""
    if controller_service:
        return {"fases": controller_service.current_status}
    return {"fases": {"fase1": False, "fase2": False, "fase3": False, "fase4": False}}

@app.post("/api/analitico_camara")
async def receive_analytic_confirmation(request: Request):
    """Recibir confirmación de analítico de la controladora"""
    try:
        data = await request.json()
        logger.info(f"Confirmación de analítico recibida: {data}")
        return {"status": "received", "message": "Confirmación procesada"}
        
    except Exception as e:
        logger.error(f"Error procesando confirmación: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# ============================================================================
# TAREAS EN BACKGROUND (MANTIENEN FUNCIONALIDAD COMPLETA)
# ============================================================================
async def daily_cleanup_task():
    """Tarea diaria de limpieza de base de datos"""
    while True:
        try:
            now = datetime.now()
            next_cleanup = now.replace(hour=2, minute=0, second=0, microsecond=0)
            if next_cleanup <= now:
                next_cleanup += timedelta(days=1)
            
            wait_seconds = (next_cleanup - now).total_seconds()
            await asyncio.sleep(wait_seconds)
            
            logger.info("🧹 Ejecutando limpieza diaria...")
            if db_manager:
                await db_manager.cleanup_old_databases()
                await db_manager.init_daily_database()
            
        except Exception as e:
            logger.error(f"Error en tarea de limpieza: {e}")
            await asyncio.sleep(3600)

async def traffic_light_update_task():
    """Tarea de actualización de estado de semáforo con controladora TICSA"""
    while True:
        try:
            if controller_service:
                status = await controller_service.get_traffic_light_status()
                if status and video_processor:
                    camera_config = load_camera_config()
                    camera_phase = camera_config.get("fase", "fase1")
                    is_red = status.get(camera_phase, False)
                    video_processor.update_red_light_status(is_red)
            
            await asyncio.sleep(1)
            
        except Exception as e:
            logger.error(f"Error actualizando estado de semáforo: {e}")
            await asyncio.sleep(5)

# ============================================================================
# RUTAS DEL FRONTEND (MANTIENEN FUNCIONALIDAD COMPLETA)
# ============================================================================
# REEMPLAZA TODO desde línea 854 hasta el final con esto:

FRONTEND_BUILD_PATH = "/app/frontend/build"
HAS_FRONTEND = os.path.exists(FRONTEND_BUILD_PATH) and os.path.exists(f"{FRONTEND_BUILD_PATH}/index.html")

if HAS_FRONTEND:
    logger.info("✅ Frontend encontrado - configurando rutas")
    
    # Montar archivos estáticos
    app.mount("/static", StaticFiles(directory=f"{FRONTEND_BUILD_PATH}/static"), name="static")
    
    @app.get("/")
    async def root():
        return FileResponse(f"{FRONTEND_BUILD_PATH}/index.html")

    @app.get("/{path:path}")
    async def catch_all(path: str):
        # Skip API routes
        if path.startswith(("api/", "docs", "redoc", "openapi.json")):
            raise HTTPException(404, "Not found")
        
        file_path = f"{FRONTEND_BUILD_PATH}/{path}"
        if os.path.exists(file_path) and os.path.isfile(file_path):
            return FileResponse(file_path)
        return FileResponse(f"{FRONTEND_BUILD_PATH}/index.html")

else:
    @app.get("/")
    async def fallback_root():
        """Fallback cuando no hay frontend"""
        return {
            "message": "Sistema de Detección Vehicular - Radxa Rock 5T",
            "status": "running",
            "version": "1.0.0",
            "api_docs": "/docs",
            "endpoints": {
                "health": "/api/camera_health",
                "camera": "/api/camera/status"
            }
        }

# ============================================================================
# INICIO DEL SERVIDOR (CORREGIDO)
# ============================================================================
if __name__ == "__main__":
    print("🚀 Vehicle Detection System Starting")
    print(f"🌐 Server: http://0.0.0.0:8000")
    print(f"📚 Docs: http://0.0.0.0:8000/docs")
    print(f"🎯 Frontend: {'Available' if HAS_FRONTEND else 'Not available'}")
    print(f"⚡ RKNN: {'Enabled' if os.getenv('USE_RKNN', '0') == '1' else 'Disabled'}")
    
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=8000,
        reload=False
    )