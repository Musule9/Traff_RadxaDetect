import os
import asyncio
import json
from datetime import datetime, timedelta
from typing import Optional, Dict, List
from contextlib import asynccontextmanager

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

# ============================================================================
# CONFIGURACIÓN DE LOGGING
# ============================================================================
def setup_logging():
    """Configurar sistema de logging"""
    log_level = os.getenv('LOG_LEVEL', 'info').lower()
    
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
    
    return log_level

LOG_LEVEL = setup_logging()

# Importar módulos de la aplicación
try:
    from app.core.video_processor import VideoProcessor
    from app.core.database import DatabaseManager
    from app.services.auth_service import AuthService
    from app.services.controller_service import ControllerService
    MODULES_AVAILABLE = True
    logger.info("✅ Módulos de aplicación cargados correctamente")
except ImportError as e:
    logger.warning(f"⚠️ Módulos de aplicación no disponibles: {e}")
    MODULES_AVAILABLE = False

# Modelos Pydantic
class LoginRequest(BaseModel):
    username: str
    password: str

class CameraConfig(BaseModel):
    rtsp_url: str
    fase: str
    direccion: str
    controladora_id: str
    controladora_ip: str

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

class SystemConfig(BaseModel):
    confidence_threshold: float = 0.5
    night_vision_enhancement: bool = True
    show_overlay: bool = True
    data_retention_days: int = 30
    target_fps: int = 30
    log_level: str = "INFO"

# Variables globales
video_processor = None
db_manager = None
auth_service = None
controller_service = None
security = HTTPBearer()

# Funciones auxiliares
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
            "target_fps": 30
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
            "enabled": False
        }

async def controller_callback(action: str, data: Dict):
    """Callback para comunicación con controladora"""
    if action == "send_analytic" and controller_service:
        await controller_service.send_analytic(data)

async def verify_token(credentials: HTTPAuthorizationCredentials = Depends(security)):
    """Verificar token de autenticación"""
    if not auth_service or not auth_service.verify_token(credentials.credentials):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Token inválido",
            headers={"WWW-Authenticate": "Bearer"},
        )
    return credentials.credentials

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Gestor de ciclo de vida de la aplicación"""
    global video_processor, db_manager, auth_service, controller_service
    
    try:
        logger.info("🚀 Inicializando servicios...")
        
        # FORZAR INICIALIZACIÓN DE BASE DE DATOS
        db_manager = DatabaseManager()
        await db_manager.init_daily_database()
        logger.info("✅ Base de datos inicializada")
        
        # Autenticación
        auth_service = AuthService()
        
        # Servicio de controladora
        controller_service = ControllerService()
        
        # SOLO inicializar video processor si los módulos están disponibles
        if MODULES_AVAILABLE:
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
                logger.info("✅ Procesamiento de video iniciado")
        
        # Tareas en background SIEMPRE
        asyncio.create_task(daily_cleanup_task())
        if MODULES_AVAILABLE:
            asyncio.create_task(traffic_light_update_task())
        
        logger.info("✅ Servicios inicializados correctamente")
        
        yield
        
    except Exception as e:
        logger.error(f"Error en inicialización: {e}")
        yield
    finally:
        if video_processor:
            video_processor.stop_processing()
        if controller_service:
            await controller_service.close()
        logger.info("🔽 Servicios finalizados")
# Crear aplicación FastAPI
app = FastAPI(
    title="Sistema de Detección Vehicular",
    description="Sistema avanzado para Radxa Rock 5T",
    version="1.0.0",
    lifespan=lifespan
)

# Middleware CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ============================================================================
# SERVIR FRONTEND - CONFIGURACIÓN CRÍTICA
# ============================================================================

# Verificar si existe el build del frontend
FRONTEND_BUILD_PATH = "/app/frontend/build"
HAS_FRONTEND = os.path.exists(FRONTEND_BUILD_PATH) and os.path.exists(f"{FRONTEND_BUILD_PATH}/index.html")

if HAS_FRONTEND:
    logger.info("✅ Frontend build encontrado - configurando rutas")
    
    # Montar archivos estáticos
    app.mount("/static", StaticFiles(directory=f"{FRONTEND_BUILD_PATH}/static"), name="static")
    
    # Servir archivos estáticos adicionales
    @app.get("/favicon.ico")
    async def get_favicon():
        favicon_path = f"{FRONTEND_BUILD_PATH}/favicon.ico"
        if os.path.exists(favicon_path):
            return FileResponse(favicon_path)
        raise HTTPException(404)
    
    @app.get("/manifest.json")
    async def get_manifest():
        manifest_path = f"{FRONTEND_BUILD_PATH}/manifest.json"
        if os.path.exists(manifest_path):
            return FileResponse(manifest_path)
        raise HTTPException(404)
else:
    logger.warning("⚠️ Frontend build no encontrado")

# ============================================================================
# RUTAS DE API
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

# ============================================================================
# CONFIGURACIÓN DE CÁMARA - COMPLETA
# ============================================================================

@app.get("/api/camera/config")
async def get_camera_config():
    """Obtener configuración actual de cámara"""
    try:
        # Asegurar que el directorio existe
        os.makedirs("/app/config", exist_ok=True)
        
        # Intentar cargar configuración existente
        try:
            with open("/app/config/cameras.json", "r") as f:
                cameras = json.load(f)
        except:
            # Crear configuración por defecto si no existe
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
        
        # Retornar la primera cámara habilitada o la primera disponible
        for camera in cameras.values():
            if camera.get("enabled", False):
                return camera
        
        # Si no hay cámara habilitada, retornar la primera
        if cameras:
            return list(cameras.values())[0]
        
        # Configuración por defecto como último recurso
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
        
        # Cargar configuración existente o crear nueva
        try:
            with open("/app/config/cameras.json", "r") as f:
                cameras = json.load(f)
        except:
            cameras = {}
        
        # Asegurar que existe al menos camera_1
        if not cameras:
            cameras = {"camera_1": {"id": "camera_1", "name": "Cámara Principal"}}
        
        # Actualizar la primera cámara disponible
        camera_key = list(cameras.keys())[0]
        cameras[camera_key].update(config.dict())
        cameras[camera_key]["enabled"] = True
        
        # Guardar configuración
        with open("/app/config/cameras.json", "w") as f:
            json.dump(cameras, f, indent=2)
        
        logger.info(f"Configuración de cámara actualizada: {config.dict()}")
        
        # Reiniciar procesador si está ejecutándose
        global video_processor
        if video_processor and video_processor.is_running:
            logger.info("Reiniciando procesador de video con nueva configuración...")
            video_processor.stop_processing()
            await asyncio.sleep(2)
        
        if video_processor:
            video_processor.camera_config = config.dict()
            if config.rtsp_url:  # Solo iniciar si hay URL RTSP
                video_processor.start_processing()
                logger.info("Procesador de video reiniciado exitosamente")
        
        return {"message": "Configuración actualizada exitosamente", "config": config.dict()}
        
    except Exception as e:
        logger.error(f"Error actualizando configuración de cámara: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/camera/stream")
async def get_camera_stream():
    """Stream de video de la cámara"""
    def generate_frames():
        while True:
            if video_processor and video_processor.is_running:
                frame = video_processor.get_latest_frame()
                if frame is not None:
                    height, width = frame.shape[:2]
                    if width > 1280:
                        scale = 1280 / width
                        new_width = 1280
                        new_height = int(height * scale)
                        frame = cv2.resize(frame, (new_width, new_height))
                    
                    ret, buffer = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 80])
                    if ret:
                        yield (b'--frame\r\n'
                               b'Content-Type: image/jpeg\r\n\r\n' + buffer.tobytes() + b'\r\n')
            else:
                # Frame placeholder
                placeholder = np.zeros((480, 640, 3), dtype=np.uint8)
                cv2.putText(placeholder, "Camara no configurada", (50, 240), 
                           cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
                ret, buffer = cv2.imencode('.jpg', placeholder)
                if ret:
                    yield (b'--frame\r\n'
                           b'Content-Type: image/jpeg\r\n\r\n' + buffer.tobytes() + b'\r\n')
            
            time.sleep(1/15)
    
    return StreamingResponse(
        generate_frames(),
        media_type="multipart/x-mixed-replace; boundary=frame"
    )

# Health check
@app.get("/api/camera_health")
async def get_camera_health():
    """Verificar salud de la cámara"""
    if video_processor:
        return {
            "healthy": video_processor.is_running,
            "fps": video_processor.current_fps,
            "last_frame": video_processor.latest_frame is not None,
            "modules_available": MODULES_AVAILABLE
        }
    return {
        "healthy": False, 
        "fps": 0, 
        "last_frame": False,
        "modules_available": MODULES_AVAILABLE
    }

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

# Análisis
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

# AGREGAR estos endpoints después de los existentes:

@app.get("/api/analysis/lines")
async def get_lines():
    """Obtener todas las líneas de análisis configuradas"""
    try:
        os.makedirs("/app/config", exist_ok=True)
        
        try:
            with open("/app/config/analysis.json", "r") as f:
                analysis = json.load(f)
        except:
            # Crear archivo por defecto si no existe
            analysis = {"lines": {}, "zones": {}}
            with open("/app/config/analysis.json", "w") as f:
                json.dump(analysis, f, indent=2)
        
        return {"lines": analysis.get("lines", {})}
        
    except Exception as e:
        logger.error(f"Error obteniendo líneas: {e}")
        return {"lines": {}}

@app.get("/api/analysis/zones")
async def get_zones():
    """Obtener todas las zonas de análisis configuradas"""
    try:
        os.makedirs("/app/config", exist_ok=True)
        
        try:
            with open("/app/config/analysis.json", "r") as f:
                analysis = json.load(f)
        except:
            # Crear archivo por defecto si no existe
            analysis = {"lines": {}, "zones": {}}
            with open("/app/config/analysis.json", "w") as f:
                json.dump(analysis, f, indent=2)
        
        return {"zones": analysis.get("zones", {})}
        
    except Exception as e:
        logger.error(f"Error obteniendo zonas: {e}")
        return {"zones": {}}

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
# ============================================================================
# TAREAS EN BACKGROUND
# ============================================================================

async def daily_cleanup_task():
    """Tarea diaria de limpieza"""
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
    """Tarea de actualización de estado de semáforo"""
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
# RUTAS DEL FRONTEND - DEBEN IR AL FINAL
# ============================================================================

if HAS_FRONTEND:
    @app.get("/")
    async def serve_frontend():
        """Servir frontend React"""
        return FileResponse(f"{FRONTEND_BUILD_PATH}/index.html")
    
    @app.get("/{path:path}")
    async def serve_frontend_routes(path: str):
        """Servir rutas del frontend React y archivos estáticos"""
        # Si es una ruta de API, no interferir
        if path.startswith("api/"):
            raise HTTPException(404)
        
        # Verificar si existe el archivo estático
        file_path = f"{FRONTEND_BUILD_PATH}/{path}"
        if os.path.exists(file_path) and os.path.isfile(file_path):
            return FileResponse(file_path)
        
        # Para rutas de React Router, servir index.html
        return FileResponse(f"{FRONTEND_BUILD_PATH}/index.html")
else:
    @app.get("/")
    async def fallback_root():
        """Fallback cuando no hay frontend"""
        return {
            "message": "Sistema de Detección Vehicular",
            "status": "running",
            "modules_available": MODULES_AVAILABLE,
            "frontend_available": False,
            "api_docs": "/docs"
        }

# ============================================================================
# INICIO DEL SERVIDOR
# ============================================================================

if __name__ == "__main__":
    uvicorn_log_level = LOG_LEVEL.lower()
    
    logger.info(f"🚀 Iniciando servidor en puerto 8000 con log level: {uvicorn_log_level}")
    logger.info(f"📁 Frontend disponible: {HAS_FRONTEND}")
    logger.info(f"🔧 Módulos disponibles: {MODULES_AVAILABLE}")
    
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=8000,
        reload=False,
        log_level=uvicorn_log_level,
        access_log=False
    )