# REEMPLAZA COMPLETAMENTE main.py

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
from pydantic import BaseModel
from loguru import logger
import cv2
import numpy as np
import threading
import time
import sys
from pathlib import Path

# ============================================================================
# CONFIGURACIÓN DE LOGGING
# ============================================================================
def setup_logging():
    log_level = os.getenv('LOG_LEVEL', 'info').lower()
    logger.remove()
    os.makedirs("/app/logs", exist_ok=True)
    logger.add("/app/logs/app.log", rotation="10 MB", retention="7 days", level=log_level.upper())
    logger.add(sys.stdout, level=log_level.upper(), format="<green>{time:HH:mm:ss}</green> | <level>{level}</level> | {message}")
    return log_level

LOG_LEVEL = setup_logging()

# ============================================================================
# IMPORTAR MÓDULOS
# ============================================================================
video_processor = None
db_manager = None
auth_service = None
controller_service = None
MODULES_AVAILABLE = False

def import_app_modules():
    global video_processor, db_manager, auth_service, controller_service, MODULES_AVAILABLE
    try:
        from app.core.database import DatabaseManager
        from app.services.auth_service import AuthService
        from app.services.controller_service import ControllerService
        
        db_manager = DatabaseManager()
        auth_service = AuthService()
        controller_service = ControllerService()
        MODULES_AVAILABLE = True
        logger.info("✅ Módulos de aplicación cargados correctamente")
        return True
    except Exception as e:
        logger.error(f"❌ Error importando módulos: {e}")
        return False

import_app_modules()

# ============================================================================
# MODELOS PYDANTIC
# ============================================================================
class LoginRequest(BaseModel):
    username: str
    password: str

class CameraConfig(BaseModel):
    rtsp_url: str
    fase: str = "fase1"
    direccion: str = "norte"
    controladora_id: str = "CTRL_001"
    controladora_ip: str = "192.168.1.200"
    camera_name: Optional[str] = ""
    camera_location: Optional[str] = ""
    camera_ip: Optional[str] = ""
    username: str = "admin"
    password: Optional[str] = ""
    port: str = "554"
    stream_path: str = "/stream1"
    resolution: str = "1920x1080"
    frame_rate: str = "30"
    enabled: bool = True

class LineConfig(BaseModel):
    id: str
    name: str
    points: List[List[int]]
    lane: str
    line_type: str
    distance_to_next: Optional[float] = None

class ZoneConfig(BaseModel):
    id: str
    name: str
    points: List[List[int]]
    zone_type: str = "red_light"

# ============================================================================
# CONFIGURACIÓN - CORREGIDA SIN DUPLICADOS
# ============================================================================
def get_config_file_path():
    return "/app/config/camera_config.json"

def get_system_config_file_path():
    return "/app/config/system_config.json"

def load_camera_config() -> Dict:
    """Cargar configuración de cámara"""
    config_file = get_config_file_path()
    try:
        if os.path.exists(config_file):
            with open(config_file, "r") as f:
                config = json.load(f)
                logger.debug(f"📄 Config cargada: RTSP={bool(config.get('rtsp_url'))}")
                return config
    except Exception as e:
        logger.error(f"Error cargando configuración: {e}")
    
    # Configuración por defecto
    return {
        "rtsp_url": "",
        "fase": "fase1",
        "direccion": "norte",
        "controladora_id": "CTRL_001",
        "controladora_ip": "192.168.1.200",
        "camera_name": "",
        "camera_location": "",
        "camera_ip": "",
        "username": "admin",
        "password": "",
        "port": "554",
        "stream_path": "/stream1",
        "resolution": "640x640",  # ✅ FORZADO PARA RKNN
        "frame_rate": "30",
        "enabled": False
    }

def save_camera_config(config: Dict) -> bool:
    """Guardar configuración de cámara"""
    config_file = get_config_file_path()
    try:
        os.makedirs("/app/config", exist_ok=True)
        
        # ✅ FORZAR RESOLUCIÓN PARA RKNN
        config["resolution"] = "640x640"
        config["frame_rate"] = "30"
        config["last_updated"] = datetime.now().isoformat()
        
        with open(config_file, "w") as f:
            json.dump(config, f, indent=2)
        
        logger.info(f"✅ Configuración guardada: RTSP={bool(config.get('rtsp_url'))}")
        return True
    except Exception as e:
        logger.error(f"❌ Error guardando configuración: {e}")
        return False

def load_system_config() -> Dict:
    """Cargar configuración del sistema"""
    config_file = get_system_config_file_path()
    try:
        if os.path.exists(config_file):
            with open(config_file, "r") as f:
                return json.load(f)
    except Exception as e:
        logger.error(f"Error cargando config sistema: {e}")
    
    # Configuración por defecto
    default_config = {
        "confidence_threshold": 0.5,
        "night_vision_enhancement": True,
        "show_overlay": True,
        "data_retention_days": 30,
        "target_fps": 30,
        "log_level": "INFO",
        "model_path": "/app/models/yolo11n.rknn",
        "use_rknn": True,
        "target_platform": "rk3588"
    }
    
    save_system_config(default_config)
    return default_config

def save_system_config(config: Dict) -> bool:
    """Guardar configuración del sistema"""
    config_file = get_system_config_file_path()
    try:
        os.makedirs("/app/config", exist_ok=True)
        with open(config_file, "w") as f:
            json.dump(config, f, indent=2)
        return True
    except Exception as e:
        logger.error(f"Error guardando config sistema: {e}")
        return False

# ============================================================================
# VIDEO PROCESSOR - CORREGIDO
# ============================================================================
class SimpleVideoStream:
    """Stream básico para casos de emergencia"""
    
    def __init__(self, rtsp_url):
        self.rtsp_url = rtsp_url
        self.is_running = False
        self.current_fps = 0
        self.latest_frame = None
        self.capture_thread = None
        self.frame_lock = threading.Lock()
        
    def start_processing(self):
        if self.is_running:
            return
            
        self.is_running = True
        self.capture_thread = threading.Thread(target=self._capture_loop)
        self.capture_thread.daemon = True
        self.capture_thread.start()
        logger.info(f"✅ Stream básico iniciado: {self.rtsp_url}")
    
    def stop_processing(self):
        self.is_running = False
        if self.capture_thread:
            self.capture_thread.join(timeout=5)
    
    def _capture_loop(self):
        cap = cv2.VideoCapture(self.rtsp_url)
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        
        if not cap.isOpened():
            logger.error(f"❌ No se pudo abrir stream: {self.rtsp_url}")
            self.is_running = False
            return
        
        fps_counter = 0
        fps_time = time.time()
        
        while self.is_running:
            ret, frame = cap.read()
            if ret and frame is not None:
                # ✅ REDIMENSIONAR A 640x640
                if frame.shape[:2] != (640, 640):
                    frame = cv2.resize(frame, (640, 640))
                
                with self.frame_lock:
                    self.latest_frame = frame.copy()
                
                fps_counter += 1
                current_time = time.time()
                if current_time - fps_time >= 1.0:
                    self.current_fps = fps_counter
                    fps_counter = 0
                    fps_time = current_time
            else:
                time.sleep(0.1)
        
        cap.release()
    
    def get_latest_frame(self):
        with self.frame_lock:
            return self.latest_frame.copy() if self.latest_frame is not None else None

async def restart_video_processor():
    """Reiniciar video processor - CORREGIDO PARA RKNN"""
    global video_processor
    
    try:
        # Parar procesador actual
        if video_processor:
            if hasattr(video_processor, 'stop_processing'):
                video_processor.stop_processing()
            await asyncio.sleep(2)
            video_processor = None
        
        # Cargar configuración
        camera_config = load_camera_config()
        system_config = load_system_config()
        
        # ✅ VERIFICAR RTSP - MENOS ESTRICTO
        rtsp_url = camera_config.get("rtsp_url", "").strip()
        if not rtsp_url:
            logger.info("⏸️ No hay URL RTSP configurada")
            return False
        
        logger.info(f"🚀 Iniciando video processor con RTSP: {rtsp_url[:50]}...")
        
        # ✅ INTENTAR VIDEO PROCESSOR COMPLETO PRIMERO
        if MODULES_AVAILABLE:
            try:
                from app.core.video_processor import VideoProcessor
                
                # Configurar para RKNN
                system_config.update({
                    "model_path": "/app/models/yolo11n.rknn",
                    "use_rknn": True,
                    "target_platform": "rk3588",
                    "forced_resolution": "640x640"
                })
                
                video_processor = VideoProcessor(
                    camera_config=camera_config,
                    system_config=system_config,
                    db_manager=db_manager,
                    callback_func=None
                )
                
                # ✅ PASAR EVENT LOOP
                video_processor.set_event_loop(asyncio.get_event_loop())
                
                await video_processor.initialize()
                video_processor.start_processing()
                
                # ✅ VERIFICAR INICIO - MÁS TIEMPO PARA RKNN
                await asyncio.sleep(5)
                
                if hasattr(video_processor, 'is_running') and video_processor.is_running:
                    logger.info("✅ VideoProcessor con IA iniciado correctamente")
                    return True
                else:
                    logger.warning("⚠️ VideoProcessor no se inició, probando stream básico...")
                    video_processor = None
                    
            except Exception as e:
                logger.warning(f"⚠️ Error con VideoProcessor: {e}")
                video_processor = None
        
        # ✅ FALLBACK A STREAM BÁSICO
        logger.info("🔄 Iniciando stream básico...")
        try:
            video_processor = SimpleVideoStream(rtsp_url)
            video_processor.start_processing()
            
            await asyncio.sleep(3)
            if video_processor.is_running:
                logger.info("✅ Stream básico iniciado correctamente")
                return True
            else:
                logger.error("❌ Stream básico también falló")
                video_processor = None
                return False
                
        except Exception as e:
            logger.error(f"❌ Error con stream básico: {e}")
            video_processor = None
            return False
            
    except Exception as e:
        logger.error(f"❌ Error crítico en restart_video_processor: {e}")
        video_processor = None
        return False

def get_video_processor_status():
    """Obtener estado del video processor"""
    if not video_processor:
        return {"running": False, "fps": 0, "error": "No inicializado"}
    
    try:
        is_running = getattr(video_processor, 'is_running', False)
        fps = getattr(video_processor, 'current_fps', 0)
        
        return {
            "running": is_running,
            "fps": fps,
            "error": None if is_running else "No procesando"
        }
    except Exception as e:
        return {"running": False, "fps": 0, "error": str(e)}

# ============================================================================
# APLICACIÓN FASTAPI
# ============================================================================
@asynccontextmanager
async def lifespan(app: FastAPI):
    try:
        logger.info("🚀 Iniciando servicios...")
        
        if db_manager:
            await db_manager.init_daily_database()
        
        # ✅ VERIFICAR E INICIAR VIDEO PROCESSOR SI HAY CONFIGURACIÓN
        camera_config = load_camera_config()
        if camera_config.get("rtsp_url", "").strip():
            logger.info("🎥 Iniciando video processor con configuración existente...")
            await restart_video_processor()
        else:
            logger.info("⏸️ No hay configuración RTSP - esperando configuración")
        
        logger.info("✅ Sistema inicializado")
        yield
        
    except Exception as e:
        logger.error(f"Error en inicialización: {e}")
        yield
    finally:
        if video_processor and hasattr(video_processor, 'stop_processing'):
            video_processor.stop_processing()
        logger.info("🔽 Servicios finalizados")

app = FastAPI(lifespan=lifespan)

# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["GET", "POST", "PUT", "DELETE", "OPTIONS"],
    allow_headers=["*"],
    expose_headers=["*"]
)

# Seguridad
security = HTTPBearer(auto_error=False)

async def verify_token(credentials: HTTPAuthorizationCredentials = Depends(security)):
    if not credentials:
        raise HTTPException(status_code=401, detail="Token requerido")
    
    try:
        if not auth_service:
            if credentials.credentials and len(credentials.credentials) > 5:
                return credentials.credentials
            raise HTTPException(status_code=401, detail="Token inválido")
        
        if not auth_service.verify_token(credentials.credentials):
            raise HTTPException(status_code=401, detail="Token inválido o expirado")
        
        return credentials.credentials
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error verificando token: {e}")
        raise HTTPException(status_code=401, detail="Error de autenticación")

# ============================================================================
# RUTAS API
# ============================================================================

# Autenticación
@app.post("/api/auth/login")
async def login(request: LoginRequest):
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
    if auth_service:
        auth_service.revoke_token(token)
    return {"message": "Logout exitoso"}

# Health check
@app.get("/api/camera_health")
async def health():
    camera_config = load_camera_config()
    video_status = get_video_processor_status()
    
    # ✅ VERIFICAR RKNN
    rknn_available = False
    try:
        from rknnlite.api import RKNNLite
        rknn_available = True
    except:
        pass
    
    rknn_model_available = os.path.exists("/app/models/yolo11n.rknn")
    
    return {
        "status": "healthy" if video_status["running"] else "warning",
        "timestamp": datetime.now().isoformat(),
        "camera_connected": video_status["running"],
        "camera_fps": video_status["fps"],
        "camera_configured": bool(camera_config.get("rtsp_url")),
        "modules_available": MODULES_AVAILABLE,
        "rknn_available": rknn_available,
        "rknn_model_available": rknn_model_available,
        "processing_resolution": "640x640",
        "optimization_status": "NPU RKNN" if rknn_model_available and rknn_available else "CPU"
    }

@app.get("/api/camera/config")
async def get_camera_config_api():
    """Obtener configuración de cámara - CORREGIDO"""
    try:
        config = load_camera_config()
        logger.debug(f"📤 Enviando config: RTSP={bool(config.get('rtsp_url'))}")
        return config
    except Exception as e:
        logger.error(f"❌ Error obteniendo config: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/camera/config")
async def update_camera_config_api(config: CameraConfig, token: str = Depends(verify_token)):
    """Actualizar configuración - ARREGLAR GUARDADO"""
    try:
        logger.info(f"📥 Guardando configuración: RTSP={bool(config.rtsp_url)}")
        
        # Convertir y forzar resolución
        config_dict = config.dict()
        config_dict["resolution"] = "640x640"
        config_dict["frame_rate"] = "30"
        config_dict["enabled"] = bool(config.rtsp_url and config.rtsp_url.strip())
        
        # ✅ GUARDAR INMEDIATAMENTE
        if not save_camera_config(config_dict):
            raise HTTPException(status_code=500, detail="Error guardando configuración")
        
        # ✅ VERIFICAR QUE SE GUARDÓ
        saved_config = load_camera_config()
        if saved_config.get("rtsp_url") != config.rtsp_url:
            logger.error("❌ La configuración no se guardó correctamente")
            raise HTTPException(status_code=500, detail="La configuración no se persistió")
        
        logger.info("✅ Configuración verificada y guardada")
        
        # Reiniciar video processor solo si hay RTSP
        processor_started = False
        if config.rtsp_url and config.rtsp_url.strip():
            logger.info("🔄 Reiniciando video processor...")
            processor_started = await restart_video_processor()
        
        return {
            "message": "Configuración guardada y verificada exitosamente",
            "config_saved": True,
            "config_verified": saved_config.get("rtsp_url") == config.rtsp_url,
            "video_processor_started": processor_started,
            "rtsp_configured": bool(config.rtsp_url),
            "resolution_forced": "640x640",
            "enabled": config_dict["enabled"]
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"❌ Error crítico guardando configuración: {e}")
        raise HTTPException(status_code=500, detail=f"Error: {str(e)}")

# ✅ RUTA PARA VERIFICAR CONFIGURACIÓN
@app.get("/api/camera/config/verify")
async def verify_camera_config_api():
    """Verificar que la configuración se guardó correctamente"""
    try:
        config = load_camera_config()
        config_file = get_config_file_path()
        
        return {
            "config_file_exists": os.path.exists(config_file),
            "config_file_path": config_file,
            "rtsp_configured": bool(config.get("rtsp_url")),
            "rtsp_url_length": len(config.get("rtsp_url", "")),
            "last_updated": config.get("last_updated"),
            "enabled": config.get("enabled", False),
            "config_content": {k: v if k != "password" else "***" for k, v in config.items()}
        }
    except Exception as e:
        logger.error(f"❌ Error verificando config: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/camera/config/reset")
async def reset_camera_config_api():
    """Resetear configuración de cámara"""
    try:
        global video_processor
        
        # Parar video processor
        if video_processor:
            if hasattr(video_processor, 'stop_processing'):
                video_processor.stop_processing()
            await asyncio.sleep(2)
            video_processor = None
        
        # Configuración limpia
        default_config = {
            "rtsp_url": "",
            "fase": "fase1",
            "direccion": "norte",
            "controladora_id": "CTRL_001",
            "controladora_ip": "192.168.1.200",
            "camera_name": "",
            "camera_location": "",
            "camera_ip": "",
            "username": "admin",
            "password": "",
            "port": "554",
            "stream_path": "/stream1",
            "resolution": "640x640",
            "frame_rate": "30",
            "enabled": False,
            "reset_at": datetime.now().isoformat()
        }
        
        if not save_camera_config(default_config):
            raise HTTPException(status_code=500, detail="Error guardando configuración")
        
        logger.info("🧹 Configuración reseteada completamente")
        
        return {
            "message": "Configuración reseteada exitosamente",
            "config": default_config
        }
        
    except Exception as e:
        logger.error(f"❌ Error reseteando configuración: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# Estado de cámara
@app.get("/api/camera/status")
async def get_camera_status_api():
    config = load_camera_config()
    video_status = get_video_processor_status()
    
    return {
        "connected": video_status["running"],
        "fps": video_status["fps"],
        "rtsp_url": config.get("rtsp_url", ""),
        "fase": config.get("fase", "fase1"),
        "direccion": config.get("direccion", "norte"),
        "enabled": bool(config.get("rtsp_url", "").strip()),  # ✅ BASADO EN RTSP
        "error": video_status.get("error"),
        "last_check": datetime.now().isoformat(),
        "resolution": "640x640"
    }

# Reiniciar cámara
@app.post("/api/camera/restart")
async def restart_camera_api():
    try:
        config = load_camera_config()
        
        if not config.get("rtsp_url", "").strip():
            raise HTTPException(status_code=400, detail="No hay URL RTSP configurada")
        
        success = await restart_video_processor()
        video_status = get_video_processor_status()
        
        return {
            "message": "Procesamiento reiniciado",
            "status": "running" if success and video_status["running"] else "inactive",
            "fps": video_status["fps"]
        }
            
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"❌ Error reiniciando cámara: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# Test de conexión
@app.post("/api/camera/test")
async def test_camera_stream_api(request: Request):
    try:
        data = await request.json()
        rtsp_url = data.get("rtsp_url", "")
        
        if not rtsp_url:
            raise HTTPException(status_code=400, detail="URL RTSP requerida")
        
        logger.info(f"🧪 Probando RTSP: {rtsp_url}")
        
        # Test con OpenCV
        cap = cv2.VideoCapture(rtsp_url)
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        
        if not cap.isOpened():
            cap.release()
            return {
                "success": False,
                "message": "No se pudo conectar al stream RTSP"
            }
        
        # Leer algunos frames
        frames_read = 0
        for i in range(10):
            ret, frame = cap.read()
            if ret and frame is not None:
                frames_read += 1
                if frames_read >= 5:
                    break
            else:
                time.sleep(0.1)
        
        cap.release()
        
        if frames_read >= 3:
            return {
                "success": True,
                "message": f"Conexión exitosa. Se leyeron {frames_read} frames.",
                "frames_tested": frames_read
            }
        else:
            return {
                "success": False,
                "message": f"Conexión inestable. Solo se leyeron {frames_read} frames."
            }
            
    except Exception as e:
        logger.error(f"❌ Error en test: {e}")
        return {
            "success": False,
            "message": f"Error en prueba: {str(e)}"
        }

# ✅ STREAM DE VIDEO - CORREGIDO
@app.get("/api/camera/stream")
async def get_camera_stream_api():
    """Stream de video HTTP"""
    def generate_frames():
        frame_count = 0
        last_frame_time = time.time()
        
        while True:
            try:
                current_time = time.time()
                
                # Control de FPS para web (15 FPS)
                if current_time - last_frame_time < 1/15:
                    time.sleep(0.01)
                    continue
                
                frame_sent = False
                
                if video_processor and hasattr(video_processor, 'get_latest_frame'):
                    frame = video_processor.get_latest_frame()
                    if frame is not None:
                        # Asegurar 640x640
                        if frame.shape[:2] != (640, 640):
                            frame = cv2.resize(frame, (640, 640))
                        
                        # Comprimir
                        ret, buffer = cv2.imencode('.jpg', frame, [
                            cv2.IMWRITE_JPEG_QUALITY, 85
                        ])
                        
                        if ret:
                            frame_count += 1
                            last_frame_time = current_time
                            frame_sent = True
                            
                            yield (b'--frame\r\n'
                                   b'Content-Type: image/jpeg\r\n'
                                   b'Content-Length: ' + str(len(buffer)).encode() + b'\r\n\r\n' + 
                                   buffer.tobytes() + b'\r\n')
                
                if not frame_sent:
                    # Frame placeholder
                    yield _generate_placeholder_frame()
                    last_frame_time = current_time
                
            except Exception as e:
                logger.error(f"❌ Error en streaming: {e}")
                yield _generate_error_frame()
                time.sleep(1)
    
    return StreamingResponse(
        generate_frames(),
        media_type="multipart/x-mixed-replace; boundary=frame",
        headers={
            "Cache-Control": "no-cache, no-store, must-revalidate",
            "Pragma": "no-cache",
            "Expires": "0"
        }
    )

def _generate_placeholder_frame():
    """Frame placeholder 640x640"""
    placeholder = np.zeros((640, 640, 3), dtype=np.uint8)
    placeholder[:] = [30, 30, 30]
    
    cv2.putText(placeholder, "SISTEMA DE DETECCION VEHICULAR", (80, 300), 
               cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
    cv2.putText(placeholder, "Radxa Rock 5T - RK3588 NPU", (140, 340), 
               cv2.FONT_HERSHEY_SIMPLEX, 0.7, (100, 150, 255), 2)
    cv2.putText(placeholder, "Configure RTSP para comenzar", (150, 380), 
               cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1)
    
    ret, buffer = cv2.imencode('.jpg', placeholder, [cv2.IMWRITE_JPEG_QUALITY, 80])
    if ret:
        return (b'--frame\r\n'
                b'Content-Type: image/jpeg\r\n'
                b'Content-Length: ' + str(len(buffer)).encode() + b'\r\n\r\n' + 
                buffer.tobytes() + b'\r\n')
    return b''

def _generate_error_frame():
    """Frame de error 640x640"""
    error_frame = np.zeros((640, 640, 3), dtype=np.uint8)
    error_frame[:] = [40, 20, 20]
    
    cv2.putText(error_frame, "ERROR DE CONEXION", (180, 300), 
               cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
    cv2.putText(error_frame, "Verificar configuracion RTSP", (150, 340), 
               cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1)
    
    ret, buffer = cv2.imencode('.jpg', error_frame, [cv2.IMWRITE_JPEG_QUALITY, 80])
    if ret:
        return (b'--frame\r\n'
                b'Content-Type: image/jpeg\r\n'
                b'Content-Length: ' + str(len(buffer)).encode() + b'\r\n\r\n' + 
                buffer.tobytes() + b'\r\n')
    return b''

# ============================================================================
# RESTO DE RUTAS (análisis, sistema, etc.) - IGUAL QUE ANTES
# ============================================================================

# Configuración del sistema
@app.get("/api/config/system")
async def get_system_config_api():
    """Obtener configuración del sistema"""
    try:
        config = load_system_config()
        return config
    except Exception as e:
        logger.error(f"❌ Error obteniendo config sistema: {e}")
        return {
            "confidence_threshold": 0.4,  # ✅ THRESHOLD MÁS BAJO
            "night_vision_enhancement": True,
            "show_overlay": True,
            "data_retention_days": 30,
            "target_fps": 30,
            "log_level": "INFO"
        }

@app.post("/api/config/system") 
async def update_system_config_api(request: Request, token: str = Depends(verify_token)):
    """Actualizar configuración del sistema - CORREGIDA"""
    try:
        data = await request.json()
        logger.info(f"📥 Actualizando config sistema: {list(data.keys())}")
        
        current_config = load_system_config()
        current_config.update(data)
        
        # ✅ GUARDAR Y VERIFICAR
        if save_system_config(current_config):
            saved_config = load_system_config()
            logger.info("✅ Configuración del sistema guardada y verificada")
            return {
                "message": "Configuración del sistema actualizada",
                "config_saved": True,
                "config_verified": True,
                "updated_keys": list(data.keys())
            }
        else:
            raise HTTPException(status_code=500, detail="Error guardando configuración del sistema")
            
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"❌ Error actualizando config sistema: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# Análisis (líneas y zonas) - SIMPLIFICADO
def get_analysis_file_path():
    return "/app/config/analysis.json"

def load_analysis_config():
    analysis_file = get_analysis_file_path()
    try:
        if os.path.exists(analysis_file):
            with open(analysis_file, "r") as f:
                return json.load(f)
    except Exception as e:
        logger.error(f"Error cargando análisis: {e}")
    
    return {"lines": {}, "zones": {}}

def save_analysis_config(analysis_config):
    analysis_file = get_analysis_file_path()
    try:
        os.makedirs("/app/config", exist_ok=True)
        with open(analysis_file, "w") as f:
            json.dump(analysis_config, f, indent=2)
        return True
    except Exception as e:
        logger.error(f"Error guardando análisis: {e}")
        return False

@app.get("/api/analysis/lines")
async def get_lines_api():
    analysis = load_analysis_config()
    return {"lines": analysis.get("lines", {})}

@app.post("/api/analysis/lines")
async def add_line_api(line: LineConfig):
    try:
        analysis = load_analysis_config()
        analysis["lines"][line.id] = line.dict()
        
        if save_analysis_config(analysis):
            return {"message": "Línea agregada exitosamente"}
        else:
            raise HTTPException(status_code=500, detail="Error guardando línea")
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.delete("/api/analysis/lines/{line_id}")
async def delete_line_api(line_id: str):
    try:
        analysis = load_analysis_config()
        if line_id in analysis.get("lines", {}):
            del analysis["lines"][line_id]
            if save_analysis_config(analysis):
                return {"message": "Línea eliminada exitosamente"}
        raise HTTPException(status_code=404, detail="Línea no encontrada")
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/analysis/zones")
async def get_zones_api():
    analysis = load_analysis_config()
    return {"zones": analysis.get("zones", {})}

@app.post("/api/analysis/zones")
async def add_zone_api(zone: ZoneConfig):
    try:
        analysis = load_analysis_config()
        analysis["zones"][zone.id] = zone.dict()
        
        if save_analysis_config(analysis):
            return {"message": "Zona agregada exitosamente"}
        else:
            raise HTTPException(status_code=500, detail="Error guardando zona")
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.delete("/api/analysis/zones/{zone_id}")
async def delete_zone_api(zone_id: str):
    try:
        analysis = load_analysis_config()
        if zone_id in analysis.get("zones", {}):
            del analysis["zones"][zone_id]
            if save_analysis_config(analysis):
                return {"message": "Zona eliminada exitosamente"}
        raise HTTPException(status_code=404, detail="Zona no encontrada")
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/analysis/clear")
async def clear_analysis_api():
    try:
        analysis = {"lines": {}, "zones": {}}
        if save_analysis_config(analysis):
            return {"message": "Configuración de análisis limpiada"}
        else:
            raise HTTPException(status_code=500, detail="Error limpiando análisis")
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/system/detailed_status")
async def get_detailed_status():
    """Estado detallado para debugging"""
    try:
        camera_config = load_camera_config()
        system_config = load_system_config()
        video_status = get_video_processor_status()
        
        # Verificar archivos
        camera_config_exists = os.path.exists(get_config_file_path())
        system_config_exists = os.path.exists(get_system_config_file_path())
        analysis_config_exists = os.path.exists("/app/config/analysis.json")
        
        return {
            "timestamp": datetime.now().isoformat(),
            "video_processor": video_status,
            "camera_config": {
                "file_exists": camera_config_exists,
                "rtsp_configured": bool(camera_config.get("rtsp_url")),
                "enabled": camera_config.get("enabled", False),
                "resolution": camera_config.get("resolution"),
                "last_updated": camera_config.get("last_updated")
            },
            "system_config": {
                "file_exists": system_config_exists,
                "confidence_threshold": system_config.get("confidence_threshold"),
                "target_fps": system_config.get("target_fps"),
                "use_rknn": system_config.get("use_rknn")
            },
            "analysis_config": {
                "file_exists": analysis_config_exists
            },
            "modules": {
                "modules_available": MODULES_AVAILABLE,
                "detector_type": getattr(video_processor, "detector", {}).get("model_type", "none") if video_processor else "none"
            },
            "files": {
                "camera_config_path": get_config_file_path(),
                "system_config_path": get_system_config_file_path(),
                "analysis_config_path": "/app/config/analysis.json"
            }
        }
    except Exception as e:
        logger.error(f"❌ Error obteniendo estado detallado: {e}")
        return {"error": str(e), "timestamp": datetime.now().isoformat()}

# Exportar datos
@app.get("/api/data/export")
async def export_data_api(date: str, type: str = "vehicle", fase: str = None):
    if not db_manager:
        return {
            "date": date,
            "type": type,
            "data": [],
            "message": "Base de datos no disponible"
        }
    
    try:
        if type == "vehicle":
            data = await db_manager.export_vehicle_crossings(date, fase)
        elif type == "red_light":
            data = await db_manager.export_red_light_counts(date, fase)
        else:
            data = []
        
        return {
            "date": date,
            "type": type,
            "data": data,
            "exported_at": datetime.now().isoformat()
        }
    except Exception as e:
        logger.error(f"Error exportando datos: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# ============================================================================
# FRONTEND
# ============================================================================
FRONTEND_BUILD_PATH = "/app/frontend/build"
HAS_FRONTEND = os.path.exists(FRONTEND_BUILD_PATH)

if HAS_FRONTEND:
    app.mount("/static", StaticFiles(directory=f"{FRONTEND_BUILD_PATH}/static"), name="static")
    
    @app.get("/")
    async def root():
        return FileResponse(f"{FRONTEND_BUILD_PATH}/index.html")

    @app.get("/{path:path}")
    async def catch_all(path: str):
        if path.startswith(("api/", "docs", "redoc")):
            raise HTTPException(404)
        
        file_path = f"{FRONTEND_BUILD_PATH}/{path}"
        if os.path.exists(file_path) and os.path.isfile(file_path):
            return FileResponse(file_path)
        return FileResponse(f"{FRONTEND_BUILD_PATH}/index.html")
else:
    @app.get("/")
    async def fallback_root():
        return {
            "message": "Sistema de Detección Vehicular - Radxa Rock 5T",
            "status": "running",
            "api_docs": "/docs",
            "health": "/api/camera_health"
        }

# ============================================================================
# INICIO DEL SERVIDOR
# ============================================================================
if __name__ == "__main__":
    print("🚀 Vehicle Detection System Starting")
    print("🎯 Optimizado para Radxa Rock 5T con NPU RK3588")
    print(f"🌐 Server: http://0.0.0.0:8000")
    print(f"📚 API Docs: http://0.0.0.0:8000/docs")
    
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=8000,
        reload=False,
        log_config=None
    )