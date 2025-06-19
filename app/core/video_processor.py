import cv2
import asyncio
import numpy as np
from typing import Optional, Dict, Callable, List
import time
import threading
import json
from queue import Queue
from loguru import logger
import os
import re

from .detector import VehicleDetector
from .tracker import BYTETracker
from .analyzer import TrafficAnalyzer
from .database import DatabaseManager

class VideoProcessor:
    """Procesador principal de video con RTSP - CORREGIDO para OpenCV moderno"""
    
    def __init__(self, 
                 camera_config: Dict,
                 system_config: Dict,
                 db_manager: DatabaseManager,
                 callback_func: Optional[Callable] = None):
        
        self.camera_config = camera_config
        self.system_config = system_config
        self.db_manager = db_manager
        self.callback_func = callback_func
        
        # Componentes principales
        self.detector = None
        self.tracker = None
        self.analyzer = None
        
        # Estado del procesamiento
        self.is_running = False
        self.video_capture = None
        self.processing_thread = None
        
        # Métricas
        self.fps_counter = 0
        self.fps_time = time.time()
        self.current_fps = 0
        
        # Frame sharing para web
        self.latest_frame = None
        self.latest_raw_frame = None
        self.frame_lock = threading.Lock()
        
        # Control de errores y reconexión
        self.connection_retry_count = 0
        self.max_retries = 5
        self.last_successful_frame = time.time()
        
    async def initialize(self):
        """Inicializar todos los componentes"""
        try:
            # Inicializar detector con YOLO11n
            model_path = self.system_config.get('model_path')
            confidence = self.system_config.get('confidence_threshold', 0.5)
            self.detector = VehicleDetector(model_path, confidence)
            
            # Log información del modelo
            model_info = self.detector.get_model_info()
            logger.info(f"🤖 Detector: {model_info['model_type']} | RKNN: {model_info['use_rknn']}")
            
            # Inicializar tracker
            self.tracker = BYTETracker(
                high_thresh=self.system_config.get('high_threshold', 0.6),
                low_thresh=self.system_config.get('low_threshold', 0.1),
                max_age=self.system_config.get('max_age', 30)
            )
            
            # Inicializar analizador
            self.analyzer = TrafficAnalyzer()
            
            # Cargar configuración de líneas y zonas
            await self._load_analysis_config()
            
            # Inicializar base de datos del día
            await self.db_manager.init_daily_database()
            
            logger.info("VideoProcessor inicializado correctamente")
            
        except Exception as e:
            logger.error(f"Error inicializando VideoProcessor: {e}")
            raise
    
    async def _load_analysis_config(self):
        """Cargar configuración de análisis desde archivo"""
        try:
            os.makedirs("/app/config", exist_ok=True)
            
            # Asegurar que el archivo existe
            analysis_file = "/app/config/analysis.json"
            if not os.path.exists(analysis_file):
                logger.info("📝 Creando archivo de análisis por defecto")
                default_analysis = {"lines": {}, "zones": {}}
                with open(analysis_file, "w") as f:
                    json.dump(default_analysis, f, indent=2)
                return
            
            # Cargar configuración de líneas y zonas
            with open(analysis_file, "r") as f:
                analysis_config = json.load(f)
            
            # Cargar líneas
            lines_config = analysis_config.get("lines", {})
            lines_loaded = 0
            for line_id, line_data in lines_config.items():
                if line_data.get("enabled", True):
                    try:
                        from .analyzer import Line, LineType
                        line = Line(
                            id=line_data["id"],
                            name=line_data["name"],
                            points=[(p[0], p[1]) for p in line_data["points"]],
                            lane=line_data["lane"],
                            line_type=LineType.COUNTING if line_data["line_type"] == "counting" else LineType.SPEED,
                            distance_to_next=line_data.get("distance_to_next"),
                            speed_line_id=line_data.get("speed_line_id"),
                            counting_line_id=line_data.get("counting_line_id"),
                            direction=line_data.get("direction", self.camera_config.get("direccion", "norte"))
                        )
                        self.analyzer.add_line(line)
                        lines_loaded += 1
                        logger.info(f"✅ Línea cargada: {line.name}")
                    except Exception as e:
                        logger.error(f"❌ Error cargando línea {line_id}: {e}")
            
            # Cargar zonas
            zones_config = analysis_config.get("zones", {})
            zones_loaded = 0
            for zone_id, zone_data in zones_config.items():
                if zone_data.get("enabled", True):
                    try:
                        from .analyzer import Zone
                        zone = Zone(
                            id=zone_data["id"],
                            name=zone_data["name"],
                            points=[(p[0], p[1]) for p in zone_data["points"]],
                            zone_type=zone_data["zone_type"]
                        )
                        self.analyzer.add_zone(zone)
                        zones_loaded += 1
                        logger.info(f"✅ Zona cargada: {zone.name}")
                    except Exception as e:
                        logger.error(f"❌ Error cargando zona {zone_id}: {e}")
            
            logger.info(f"📊 Configuración de análisis cargada: {lines_loaded} líneas, {zones_loaded} zonas")
                        
        except Exception as e:
            logger.error(f"❌ Error cargando configuración de análisis: {e}")

    def start_processing(self):
        """Iniciar procesamiento de video"""
        if self.is_running:
            return
        
        self.is_running = True
        self.processing_thread = threading.Thread(target=self._processing_loop)
        self.processing_thread.daemon = True
        self.processing_thread.start()
        
        logger.info("Procesamiento de video iniciado")
    
    def stop_processing(self):
        """Detener procesamiento de video"""
        self.is_running = False
        
        if self.video_capture:
            try:
                self.video_capture.release()
            except:
                pass
        
        if self.processing_thread:
            self.processing_thread.join(timeout=5)
        
        logger.info("Procesamiento de video detenido")
    
    def get_raw_frame(self) -> Optional[np.ndarray]:
        """Obtener frame original sin overlay de análisis"""
        with self.frame_lock:
            if hasattr(self, 'latest_raw_frame') and self.latest_raw_frame is not None:
                return self.latest_raw_frame.copy()
            return None

    def get_latest_frame(self) -> Optional[np.ndarray]:
        """Obtener último frame procesado con análisis"""
        with self.frame_lock:
            return self.latest_frame.copy() if self.latest_frame is not None else None

    def get_frame_info(self) -> Dict:
        """Obtener información del frame actual"""
        return {
            'fps': self.current_fps,
            'resolution': f"{self.latest_frame.shape[1]}x{self.latest_frame.shape[0]}" if self.latest_frame is not None else "0x0",
            'processing_time': getattr(self, 'last_processing_time', 0),
            'tracks_count': len(getattr(self, 'current_tracks', [])),
            'detections_count': getattr(self, 'last_detections_count', 0),
            'connection_retries': self.connection_retry_count
        }

    def _validate_rtsp_url(self, rtsp_url: str) -> tuple[bool, str]:
        """Validar URL RTSP y corregir problemas comunes"""
        if not rtsp_url or not rtsp_url.strip():
            return False, "URL RTSP vacía"
        
        # Verificar formato básico
        if not rtsp_url.startswith('rtsp://'):
            return False, "URL debe empezar con rtsp://"
        
        # Verificar IP inválida (problema común)
        if '0.0.0.0' in rtsp_url:
            return False, "IP 0.0.0.0 no es válida. Configure la IP real de la cámara."
        
        # Verificar formato básico de IP
        ip_pattern = r'rtsp://[^@]*@([0-9]{1,3}\.[0-9]{1,3}\.[0-9]{1,3}\.[0-9]{1,3})'
        match = re.search(ip_pattern, rtsp_url)
        if match:
            ip = match.group(1)
            # Verificar que no sean IPs reservadas problemáticas
            if ip.startswith('127.') or ip == '255.255.255.255':
                return False, f"IP {ip} no es válida para cámara remota"
        
        return True, "URL válida"

    def _processing_loop(self):
        """Loop principal de procesamiento - CORREGIDO"""
        rtsp_url = self.camera_config.get('rtsp_url')
        
        # Validar URL RTSP
        valid, error_msg = self._validate_rtsp_url(rtsp_url)
        if not valid:
            logger.error(f"❌ URL RTSP inválida: {error_msg}")
            logger.error(f"URL recibida: {rtsp_url}")
            self.is_running = False
            return
        
        logger.info(f"📡 Conectando a RTSP: {rtsp_url}")
        
        # Intentar conectar con reintentos
        while self.is_running and self.connection_retry_count < self.max_retries:
            try:
                if self._connect_to_stream(rtsp_url):
                    self._main_processing_loop()
                    break
                else:
                    self.connection_retry_count += 1
                    if self.connection_retry_count < self.max_retries:
                        retry_delay = min(5 * self.connection_retry_count, 30)
                        logger.warning(f"⏳ Reintento {self.connection_retry_count}/{self.max_retries} en {retry_delay}s...")
                        time.sleep(retry_delay)
                    else:
                        logger.error("❌ Máximo de reintentos alcanzado")
                        
            except Exception as e:
                logger.error(f"❌ Error en loop de procesamiento: {e}")
                self.connection_retry_count += 1
                time.sleep(2)
        
        self.is_running = False

    def _connect_to_stream(self, rtsp_url: str) -> bool:
        """Conectar al stream RTSP con configuración optimizada"""
        try:
            # Cerrar conexión anterior si existe
            if self.video_capture:
                self.video_capture.release()
                time.sleep(1)
            
            # Crear nueva captura con configuración optimizada
            self.video_capture = cv2.VideoCapture(rtsp_url)
            
            # CONFIGURACIÓN CORREGIDA para OpenCV moderno
            # No usar CAP_PROP_BUFFER_SIZE que fue deprecado
            try:
                # Configuraciones que SÍ funcionan en OpenCV moderno
                self.video_capture.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
                self.video_capture.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
                self.video_capture.set(cv2.CAP_PROP_FPS, 30)
                
                # Estas configuraciones son opcionales y pueden no estar disponibles
                # Se intentan pero no fallan si no están soportadas
                try:
                    # Intentar configurar buffer si está disponible (backend dependiente)
                    buffer_set = self.video_capture.set(cv2.CAP_PROP_BUFFERSIZE, 1)
                    if buffer_set:
                        logger.info("✅ Buffer size configurado")
                    else:
                        logger.info("ℹ️ Buffer size no soportado por este backend")
                except:
                    logger.info("ℹ️ Buffer size no disponible en esta versión de OpenCV")
                
                # Timeout si está disponible
                try:
                    self.video_capture.set(cv2.CAP_PROP_OPEN_TIMEOUT_MSEC, 10000)
                except:
                    pass
                    
            except Exception as e:
                logger.warning(f"⚠️ Algunas configuraciones de captura no se aplicaron: {e}")
            
            # Verificar conexión
            if not self.video_capture.isOpened():
                logger.error("❌ No se pudo abrir el stream RTSP")
                return False
            
            # Test de lectura
            logger.info("🧪 Probando lectura de frames...")
            test_frames = 0
            for i in range(5):
                ret, frame = self.video_capture.read()
                if ret and frame is not None:
                    test_frames += 1
                else:
                    time.sleep(0.2)
            
            if test_frames >= 2:
                logger.info(f"✅ Stream RTSP conectado exitosamente ({test_frames}/5 frames)")
                self.connection_retry_count = 0  # Reset contador en conexión exitosa
                return True
            else:
                logger.error(f"❌ Stream inestable ({test_frames}/5 frames)")
                return False
                
        except Exception as e:
            logger.error(f"❌ Error conectando stream: {e}")
            return False

    def _main_processing_loop(self):
        """Loop principal de procesamiento una vez conectado"""
        logger.info("🎬 Iniciando procesamiento de frames...")
        
        # Variables para control de FPS
        target_fps = self.system_config.get('target_fps', 30)
        frame_time = 1.0 / target_fps
        last_frame_time = time.time()
        consecutive_failures = 0
        max_failures = 30  # Máximo frames fallidos consecutivos
        
        while self.is_running:
            try:
                current_time = time.time()
                
                # Control de FPS
                if current_time - last_frame_time < frame_time:
                    time.sleep(0.001)
                    continue
                
                # Leer frame
                ret, frame = self.video_capture.read()
                
                if ret and frame is not None:
                    # Frame exitoso
                    consecutive_failures = 0
                    self.last_successful_frame = current_time
                    
                    # Procesar frame
                    processed_frame = self._process_frame(frame)
                    
                    # Actualizar frame compartido para web
                    with self.frame_lock:
                        self.latest_frame = processed_frame
                        self.latest_raw_frame = frame.copy()
                    
                    # Actualizar FPS
                    self._update_fps()
                    
                    last_frame_time = current_time
                    
                else:
                    # Frame fallido
                    consecutive_failures += 1
                    logger.warning(f"⚠️ No se pudo leer frame ({consecutive_failures}/{max_failures})")
                    
                    if consecutive_failures >= max_failures:
                        logger.error("❌ Demasiados frames fallidos consecutivos")
                        break
                    
                    time.sleep(0.1)
                
                # Verificar timeout de conexión
                if current_time - self.last_successful_frame > 30:  # 30 segundos sin frames
                    logger.error("❌ Timeout de conexión - sin frames por 30 segundos")
                    break
                    
            except Exception as e:
                logger.error(f"❌ Error en loop principal: {e}")
                consecutive_failures += 1
                if consecutive_failures >= max_failures:
                    break
                time.sleep(0.1)
        
        logger.info("🔚 Loop de procesamiento terminado")
    
    def _process_frame(self, frame: np.ndarray) -> np.ndarray:
        """Procesar frame individual"""
        start_time = time.time()
        
        try:
            # Mejorar imagen si es necesario (modo nocturno)
            if self.system_config.get('night_vision_enhancement', False):
                frame = self.detector.enhance_night_vision(frame)
            
            # Detección
            detections = self.detector.detect(frame)
            self.last_detections_count = len(detections)
            
            # Tracking
            tracks = self.tracker.update(detections)
            self.current_tracks = tracks
            
            # Análisis de tráfico
            analysis_results = self.analyzer.analyze_frame(tracks, frame.shape)
            
            # Procesar resultados de forma asíncrona
            asyncio.create_task(self._process_analysis_results(analysis_results, tracks))
            
            # Dibujar overlay si está habilitado
            if self.system_config.get('show_overlay', True):
                frame = self.analyzer.draw_analysis_overlay(frame, tracks)
            
            # Guardar tiempo de procesamiento
            self.last_processing_time = time.time() - start_time
            
            return frame
            
        except Exception as e:
            logger.error(f"❌ Error procesando frame: {e}")
            return frame

    async def _process_analysis_results(self, results: Dict, tracks: List):
        """Procesar resultados del análisis"""
        try:
            # Procesar cruces de líneas
            for crossing in results['line_crossings']:
                # Encontrar track correspondiente
                track = next((t for t in tracks if t.track_id == crossing['vehicle_id']), None)
                if track:
                    # Preparar datos para base de datos
                    calculated_speed = None
                    
                    # Buscar velocidad calculada
                    for speed_calc in results.get('speed_calculations', []):
                        if speed_calc['vehicle_id'] == crossing['vehicle_id']:
                            calculated_speed = speed_calc['speed_kmh']
                            break

                    # Fallback a velocidad del tracker
                    if not calculated_speed and hasattr(track, 'average_velocity'):
                        if track.average_velocity > 0:
                            calculated_speed = track.average_velocity * 3.6  # convertir a km/h
                    
                    crossing_data = {
                        'vehicle_id': crossing['vehicle_id'],
                        'line_id': crossing['line_id'],
                        'line_name': crossing['line_name'],
                        'fase': self.camera_config.get('fase', 'fase1'),
                        'semaforo_estado': 'rojo' if self.analyzer.red_light_active else 'verde',
                        'velocidad': calculated_speed,
                        'direccion': self.camera_config.get('direccion', 'norte'),
                        'No_Controladora': self.camera_config.get('controladora_id', 'CTRL_001'),
                        'confianza': track.confidence,
                        'carril': crossing['lane'],
                        'clase_vehiculo': track.class_id,
                        'bbox_x': int(track.bbox[0]),
                        'bbox_y': int(track.bbox[1]),
                        'bbox_w': int(track.bbox[2]),
                        'bbox_h': int(track.bbox[3]),
                        'metadata': {
                            'center': crossing['center'],
                            'timestamp': crossing['timestamp'],
                            'model_type': self.detector.model_type if self.detector else 'unknown'
                        }
                    }
                    
                    await self.db_manager.insert_vehicle_crossing(crossing_data)
            
            # Enviar analítico si es necesario
            if results['send_analytic'] and self.callback_func is not None:
                analytic_data = {
                    'fase': self.camera_config.get('fase', 'fase1'),
                    'puntos': len(results['vehicles_in_red_zone']),
                    'vehiculos': True
                }
                await self.callback_func('send_analytic', analytic_data)
        
        except Exception as e:
            logger.error(f"❌ Error procesando resultados: {e}")
    
    def _update_fps(self):
        """Actualizar contador de FPS"""
        self.fps_counter += 1
        current_time = time.time()
        
        if current_time - self.fps_time >= 1.0:
            self.current_fps = self.fps_counter
            self.fps_counter = 0
            self.fps_time = current_time
    
    def update_red_light_status(self, is_red: bool):
        """Actualizar estado del semáforo"""
        if self.analyzer:
            self.analyzer.update_red_light_status(is_red)
    
    def get_connection_status(self) -> Dict:
        """Obtener estado de conexión detallado"""
        return {
            'connected': self.is_running and self.video_capture and self.video_capture.isOpened(),
            'fps': self.current_fps,
            'retry_count': self.connection_retry_count,
            'max_retries': self.max_retries,
            'last_successful_frame': self.last_successful_frame,
            'processing_time_ms': getattr(self, 'last_processing_time', 0) * 1000,
            'rtsp_url_configured': bool(self.camera_config.get('rtsp_url')),
            'model_info': self.detector.get_model_info() if self.detector else {}
        }