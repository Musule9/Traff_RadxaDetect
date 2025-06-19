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
# IMPORTAR MÓDULOS CON MANEJO DE ERRORES
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
 #
# ============================================================================
# MODELOS PYDANTIC SIMPLIFICADOS
# ============================================================================
class LoginRequest(BaseModel):
    username: str
    password: str

class CameraConfig(BaseModel):
    # Configuración básica RTSP
    rtsp_url: str
    fase: str = "fase1"
    direccion: str = "norte"
    controladora_id: str = "CTRL_001"
    controladora_ip: str = "192.168.1.200"
    
    # Identificación
    camera_name: Optional[str] = ""
    camera_location: Optional[str] = ""
    
    # Red
    camera_ip: Optional[str] = ""
    username: str = "admin"
    password: Optional[str] = ""
    port: str = "554"
    stream_path: str = "/stream1"
    
    # Video
    resolution: str = "1920x1080"
    frame_rate: str = "30"
    
    # Estado
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
# FUNCIONES DE CONFIGURACIÓN SIMPLIFICADAS
# ============================================================================
def get_config_file_path():
    """Obtener ruta del archivo de configuración"""
    return "/app/config/camera_config.json"

def load_camera_config() -> Dict:
    """Cargar configuración de cámara - SIMPLIFICADO"""
    config_file = get_config_file_path()
    try:
        if os.path.exists(config_file):
            with open(config_file, "r") as f:
                config = json.load(f)
                logger.info(f"📄 Configuración cargada: RTSP={bool(config.get('rtsp_url'))}")
                return config
    except Exception as e:
        logger.error(f"Error cargando configuración: {e}")
    
    # Configuración por defecto
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
        "resolution": "1920x1080",
        "frame_rate": "30",
        "enabled": False
    }
    return default_config

def save_camera_config(config: Dict) -> bool:
    """Guardar configuración de cámara - SIMPLIFICADO"""
    config_file = get_config_file_path()
    try:
        os.makedirs("/app/config", exist_ok=True)
        
        # Agregar timestamp
        config["last_updated"] = datetime.now().isoformat()
        
        with open(config_file, "w") as f:
            json.dump(config, f, indent=2)
        
        logger.info(f"✅ Configuración guardada: {config_file}")
        return True
    except Exception as e:
        logger.error(f"❌ Error guardando configuración: {e}")
        return False

# ============================================================================
# GESTOR DE VIDEO PROCESSOR SIMPLIFICADO
# ============================================================================
# ============================================================================
# GESTOR DE VIDEO PROCESSOR SIMPLIFICADO
# ============================================================================
class SimpleVideoStream:
    """Stream de video básico sin IA para cuando no hay modelos disponibles"""
    
    def __init__(self, rtsp_url):
        self.rtsp_url = rtsp_url
        self.is_running = False
        self.current_fps = 0
        self.latest_frame = None
        self.capture_thread = None
        self.frame_lock = threading.Lock()
        
    def start_processing(self):
        """Iniciar captura de video"""
        if self.is_running:
            return
            
        self.is_running = True
        self.capture_thread = threading.Thread(target=self._capture_loop)
        self.capture_thread.daemon = True
        self.capture_thread.start()
        logger.info(f"✅ Stream básico iniciado: {self.rtsp_url}")
    
    def stop_processing(self):
        """Detener captura de video"""
        self.is_running = False
        if self.capture_thread:
            self.capture_thread.join(timeout=5)
        logger.info("⏹️ Stream básico detenido")
    
    def _capture_loop(self):
        """Loop de captura de video"""
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
                with self.frame_lock:
                    self.latest_frame = frame.copy()
                
                # Calcular FPS
                fps_counter += 1
                current_time = time.time()
                if current_time - fps_time >= 1.0:
                    self.current_fps = fps_counter
                    fps_counter = 0
                    fps_time = current_time
                
            else:
                logger.warning("⚠️ No se pudo leer frame del stream")
                time.sleep(0.1)
        
        cap.release()
    
    def get_latest_frame(self):
        """Obtener último frame capturado"""
        with self.frame_lock:
            return self.latest_frame.copy() if self.latest_frame is not None else None

async def restart_video_processor():
    """Reiniciar video processor con nueva configuración - CON FALLBACK"""
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
        
        # Solo inicializar si hay URL RTSP válida
        if not camera_config.get("rtsp_url") or not camera_config.get("rtsp_url").strip():
            logger.info("⏸️ No hay URL RTSP - video processor en espera")
            return False
        
        rtsp_url = camera_config.get("rtsp_url")
        
        # Intentar importar e inicializar VideoProcessor completo
        if MODULES_AVAILABLE:
            try:
                from app.core.video_processor import VideoProcessor
                
                system_config = load_system_config()
                
                video_processor = VideoProcessor(
                    camera_config=camera_config,
                    system_config=system_config,
                    db_manager=db_manager,
                    callback_func=None
                )
                
                await video_processor.initialize()
                video_processor.start_processing()
                
                # Verificar que se inició correctamente
                await asyncio.sleep(3)
                
                if hasattr(video_processor, 'is_running') and video_processor.is_running:
                    logger.info("✅ Video processor completo iniciado correctamente")
                    return True
                else:
                    logger.warning("⚠️ Video processor completo falló, usando stream básico")
                    raise Exception("VideoProcessor no se inició correctamente")
                    
            except Exception as e:
                logger.warning(f"⚠️ Error con VideoProcessor completo: {e}")
                logger.info("🔄 Intentando con stream básico...")
                
                # Fallback a stream básico
                try:
                    video_processor = SimpleVideoStream(rtsp_url)
                    video_processor.start_processing()
                    
                    # Verificar que funciona
                    await asyncio.sleep(2)
                    if video_processor.is_running:
                        logger.info("✅ Stream básico iniciado correctamente (sin IA)")
                        return True
                    else:
                        logger.error("❌ Stream básico también falló")
                        video_processor = None
                        return False
                        
                except Exception as e2:
                    logger.error(f"❌ Error con stream básico: {e2}")
                    video_processor = None
                    return False
        else:
            # Usar directamente stream básico si no hay módulos
            logger.info("ℹ️ Módulos no disponibles, usando stream básico")
            try:
                video_processor = SimpleVideoStream(rtsp_url)
                video_processor.start_processing()
                
                await asyncio.sleep(2)
                if video_processor.is_running:
                    logger.info("✅ Stream básico iniciado correctamente")
                    return True
                else:
                    logger.error("❌ Stream básico falló")
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
# CREAR APLICACIÓN FASTAPI
# ============================================================================
@asynccontextmanager
async def lifespan(app: FastAPI):
    """Gestor de ciclo de vida"""
    try:
        logger.info("🚀 Iniciando servicios...")
        
        # Inicializar base de datos
        if db_manager:
            await db_manager.init_daily_database()
        
        # Intentar inicializar video processor si hay configuración
        camera_config = load_camera_config()
        if camera_config.get("rtsp_url") and camera_config.get("rtsp_url").strip():
            await restart_video_processor()
        
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
    allow_methods=["*"],
    allow_headers=["*"],
)

# Seguridad
security = HTTPBearer()

async def verify_token(credentials: HTTPAuthorizationCredentials = Depends(security)):
    if not auth_service or not auth_service.verify_token(credentials.credentials):
        raise HTTPException(status_code=401, detail="Token inválido")
    return credentials.credentials

# ============================================================================
# RUTAS DE API CORREGIDAS
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
    """Health check completo"""
    camera_config = load_camera_config()
    video_status = get_video_processor_status()
    
    # Información del hardware
    hardware_info = "Unknown"
    try:
        if os.path.exists("/proc/device-tree/model"):
            with open("/proc/device-tree/model", "rb") as f:
                hardware_info = f.read().decode('utf-8', errors='ignore').strip('\x00')
    except:
        pass
    
    return {
        "status": "healthy" if video_status["running"] else "warning",
        "timestamp": datetime.now().isoformat(),
        "camera_connected": video_status["running"],
        "camera_fps": video_status["fps"],
        "camera_configured": bool(camera_config.get("rtsp_url")),
        "hardware": hardware_info,
        "modules_available": MODULES_AVAILABLE,
        "version": "1.0.0"
    }

# CONFIGURACIÓN DE CÁMARA - CORREGIDA
@app.get("/api/camera/config")
async def get_camera_config_api():
    """Obtener configuración de cámara"""
    config = load_camera_config()
    logger.info(f"📤 Enviando configuración: RTSP={bool(config.get('rtsp_url'))}")
    return config

@app.post("/api/camera/config")
async def update_camera_config_api(config: CameraConfig):
    """Actualizar configuración de cámara - SIMPLIFICADO"""
    try:
        logger.info(f"📥 Recibiendo configuración: RTSP={bool(config.rtsp_url)}")
        
        # Convertir a dict y guardar
        config_dict = config.dict()
        
        if not save_camera_config(config_dict):
            raise HTTPException(status_code=500, detail="Error guardando configuración")
        
        # Reiniciar video processor
        processor_started = await restart_video_processor()
        
        return {
            "message": "Configuración guardada exitosamente",
            "config_saved": True,
            "video_processor_started": processor_started,
            "rtsp_configured": bool(config.rtsp_url)
        }
        
    except Exception as e:
        logger.error(f"❌ Error actualizando configuración: {e}")
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
        
        # Crear configuración limpia
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
            "resolution": "1920x1080",
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

# ESTADO DE CÁMARA
@app.get("/api/camera/status")
async def get_camera_status_api():
    """Obtener estado de cámara"""
    config = load_camera_config()
    video_status = get_video_processor_status()
    
    return {
        "connected": video_status["running"],
        "fps": video_status["fps"],
        "rtsp_url": config.get("rtsp_url", ""),
        "fase": config.get("fase", "fase1"),
        "direccion": config.get("direccion", "norte"),
        "enabled": config.get("enabled", False),
        "error": video_status.get("error"),
        "last_check": datetime.now().isoformat()
    }

# REINICIAR CÁMARA
@app.post("/api/camera/restart")
async def restart_camera_api():
    """Reiniciar procesamiento de cámara"""
    try:
        config = load_camera_config()
        
        if not config.get("rtsp_url"):
            raise HTTPException(status_code=400, detail="No hay URL RTSP configurada")
        
        success = await restart_video_processor()
        video_status = get_video_processor_status()
        
        if success and video_status["running"]:
            return {
                "message": "Cámara reiniciada exitosamente",
                "status": "running",
                "fps": video_status["fps"]
            }
        else:
            return {
                "message": "Cámara reiniciada pero no está procesando",
                "status": "inactive",
                "fps": 0,
                "error": video_status.get("error")
            }
            
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"❌ Error reiniciando cámara: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# TEST DE CONEXIÓN
@app.post("/api/camera/test")
async def test_camera_stream_api(request: Request):
    """Probar conexión RTSP - CORREGIDO"""
    try:
        data = await request.json()
        rtsp_url = data.get("rtsp_url", "")
        
        if not rtsp_url:
            raise HTTPException(status_code=400, detail="URL RTSP requerida")
        
        logger.info(f"🧪 Probando conexión RTSP: {rtsp_url}")
        
        # Test básico con OpenCV - SIN CAP_PROP_TIMEOUT
        cap = cv2.VideoCapture(rtsp_url)
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        
        if not cap.isOpened():
            cap.release()
            return {
                "success": False,
                "message": "No se pudo conectar al stream RTSP. Verifique URL, credenciales y conectividad."
            }
        
        # Intentar leer algunos frames con timeout manual
        frames_read = 0
        max_attempts = 10
        
        for i in range(max_attempts):
            ret, frame = cap.read()
            if ret and frame is not None:
                frames_read += 1
                if frames_read >= 3:  # Si leemos 3 frames exitosos, es suficiente
                    break
            else:
                # Esperar un poco entre intentos
                import time
                time.sleep(0.1)
        
        cap.release()
        
        if frames_read >= 3:
            return {
                "success": True,
                "message": f"Conexión exitosa. Se leyeron {frames_read} frames.",
                "frames_tested": frames_read
            }
        elif frames_read > 0:
            return {
                "success": True,
                "message": f"Conexión inestable pero funcional. Se leyeron {frames_read} frames de {max_attempts} intentos.",
                "frames_tested": frames_read
            }
        else:
            return {
                "success": False,
                "message": f"No se pudieron leer frames del stream. Verifique la URL RTSP y la configuración de la cámara."
            }
            
    except Exception as e:
        logger.error(f"❌ Error en test de conexión: {e}")
        return {
            "success": False,
            "message": f"Error en prueba de conexión: {str(e)}"
        }

# STREAM DE VIDEO - CORREGIDO
@app.get("/api/camera/stream")
async def get_camera_stream_api():
    """Stream de video HTTP"""
    def generate_frames():
        frame_count = 0
        last_frame_time = time.time()
        
        while True:
            try:
                current_time = time.time()
                
                # Control de FPS para web (15 FPS máximo)
                if current_time - last_frame_time < 1/15:
                    time.sleep(0.01)
                    continue
                
                if video_processor and hasattr(video_processor, 'get_latest_frame'):
                    frame = video_processor.get_latest_frame()
                    if frame is not None:
                        # Redimensionar para web
                        height, width = frame.shape[:2]
                        if width > 1280:
                            scale = 1280 / width
                            new_width = 1280
                            new_height = int(height * scale)
                            frame = cv2.resize(frame, (new_width, new_height))
                        
                        # Comprimir
                        ret, buffer = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 85])
                        if ret:
                            frame_count += 1
                            last_frame_time = current_time
                            yield (b'--frame\r\n'
                                   b'Content-Type: image/jpeg\r\n'
                                   b'Content-Length: ' + str(len(buffer)).encode() + b'\r\n\r\n' + 
                                   buffer.tobytes() + b'\r\n')
                            continue
                
                # Frame placeholder si no hay video
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
            "Expires": "0",
            "Connection": "close"
        }
    )

def _generate_placeholder_frame():
    """Frame placeholder"""
    placeholder = np.zeros((480, 640, 3), dtype=np.uint8)
    
    # Fondo degradado
    for i in range(480):
        placeholder[i, :] = [20 + (i//15), 25 + (i//15), 35 + (i//15)]
    
    # Texto
    cv2.putText(placeholder, "SISTEMA DE DETECCION VEHICULAR", (80, 200), 
               cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
    cv2.putText(placeholder, "Radxa Rock 5T", (230, 240), 
               cv2.FONT_HERSHEY_SIMPLEX, 0.7, (100, 150, 255), 2)
    cv2.putText(placeholder, "Configure la camara para comenzar", (130, 280), 
               cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1)
    cv2.putText(placeholder, f"FPS: 0 | Estado: Esperando configuracion", (160, 320),
               cv2.FONT_HERSHEY_SIMPLEX, 0.5, (150, 150, 150), 1)
    
    ret, buffer = cv2.imencode('.jpg', placeholder, [cv2.IMWRITE_JPEG_QUALITY, 80])
    if ret:
        return (b'--frame\r\n'
                b'Content-Type: image/jpeg\r\n'
                b'Content-Length: ' + str(len(buffer)).encode() + b'\r\n\r\n' + 
                buffer.tobytes() + b'\r\n')
    return b''

def _generate_error_frame():
    """Frame de error"""
    error_frame = np.zeros((480, 640, 3), dtype=np.uint8)
    error_frame[:] = [40, 20, 20]
    
    cv2.putText(error_frame, "ERROR DE CONEXION", (180, 220), 
               cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
    cv2.putText(error_frame, "Verificar configuracion RTSP", (150, 260), 
               cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1)
    cv2.putText(error_frame, "URL, credenciales y conectividad", (140, 300),
               cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1)
    
    ret, buffer = cv2.imencode('.jpg', error_frame, [cv2.IMWRITE_JPEG_QUALITY, 80])
    if ret:
        return (b'--frame\r\n'
                b'Content-Type: image/jpeg\r\n'
                b'Content-Length: ' + str(len(buffer)).encode() + b'\r\n\r\n' + 
                buffer.tobytes() + b'\r\n')
    return b''

# ============================================================================
# CONFIGURACIÓN DEL SISTEMA
# ============================================================================
def get_system_config_file_path():
    return "/app/config/system_config.json"

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
    return {
        "confidence_threshold": 0.5,
        "night_vision_enhancement": True,
        "show_overlay": True,
        "data_retention_days": 30,
        "target_fps": 30,
        "log_level": "INFO"
    }

def save_system_config(config: Dict) -> bool:
    """Guardar configuración del sistema"""
    config_file = get_system_config_file_path()
    try:
        os.makedirs("/app/config", exist_ok=True)
        with open(config_file, "w") as f:
            json.dump(config, f, indent=2)
        logger.info(f"✅ Config sistema guardada: {config_file}")
        return True
    except Exception as e:
        logger.error(f"❌ Error guardando config sistema: {e}")
        return False

@app.get("/api/config/system")
async def get_system_config_api():
    """Obtener configuración del sistema"""
    config = load_system_config()
    return config

@app.post("/api/config/system") 
async def update_system_config_api(request: Request):
    """Actualizar configuración del sistema"""
    try:
        data = await request.json()
        
        current_config = load_system_config()
        current_config.update(data)
        
        if save_system_config(current_config):
            return {"message": "Configuración del sistema actualizada"}
        else:
            raise HTTPException(status_code=500, detail="Error guardando configuración")
            
    except Exception as e:
        logger.error(f"❌ Error actualizando config sistema: {e}")
        raise HTTPException(status_code=500, detail=str(e))
def get_analysis_file_path():
    return "/app/config/analysis.json"

def load_analysis_config():
    """Cargar configuración de análisis"""
    analysis_file = get_analysis_file_path()
    try:
        if os.path.exists(analysis_file):
            with open(analysis_file, "r") as f:
                return json.load(f)
    except Exception as e:
        logger.error(f"Error cargando análisis: {e}")
    
    return {"lines": {}, "zones": {}}

def save_analysis_config(analysis_config):
    """Guardar configuración de análisis"""
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
        logger.error(f"Error agregando línea: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.delete("/api/analysis/lines/{line_id}")
async def delete_line_api(line_id: str):
    try:
        analysis = load_analysis_config()
        if line_id in analysis.get("lines", {}):
            del analysis["lines"][line_id]
            if save_analysis_config(analysis):
                return {"message": "Línea eliminada exitosamente"}
            else:
                raise HTTPException(status_code=500, detail="Error guardando cambios")
        else:
            raise HTTPException(status_code=404, detail="Línea no encontrada")
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error eliminando línea: {e}")
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
        logger.error(f"Error agregando zona: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.delete("/api/analysis/zones/{zone_id}")
async def delete_zone_api(zone_id: str):
    try:
        analysis = load_analysis_config()
        if zone_id in analysis.get("zones", {}):
            del analysis["zones"][zone_id]
            if save_analysis_config(analysis):
                return {"message": "Zona eliminada exitosamente"}
            else:
                raise HTTPException(status_code=500, detail="Error guardando cambios")
        else:
            raise HTTPException(status_code=404, detail="Zona no encontrada")
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error eliminando zona: {e}")
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
        logger.error(f"Error limpiando análisis: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# ============================================================================
# EXPORTAR DATOS (SIMPLIFICADO)
# ============================================================================
@app.get("/api/data/export")
async def export_data_api(date: str, type: str = "vehicle", fase: str = None):
    """Exportar datos por fecha"""
    if not db_manager:
        return {
            "date": date,
            "type": type,
            "fase": fase,
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
            "fase": fase,
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
            "version": "1.0.0",
            "api_docs": "/docs",
            "health": "/api/camera_health"
        }

# ============================================================================
# INICIO DEL SERVIDOR
# ============================================================================
if __name__ == "__main__":
    print("🚀 Vehicle Detection System Starting")
    print(f"🌐 Server: http://0.0.0.0:8000")
    print(f"📚 API Docs: http://0.0.0.0:8000/docs")
    print(f"🎯 Frontend: {'Available' if HAS_FRONTEND else 'Not available'}")
    
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=8000,
        reload=False
    )