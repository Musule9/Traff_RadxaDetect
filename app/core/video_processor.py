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
    """Procesador principal de video con RTSP - FORZANDO 640x640 EN TODA LA PIPELINE"""
    
    def __init__(self, 
                 camera_config: Dict,
                 system_config: Dict,
                 db_manager: DatabaseManager,
                 callback_func: Optional[Callable] = None):
        
        self.camera_config = camera_config
        self.system_config = system_config
        self.db_manager = db_manager
        self.callback_func = callback_func
        
        # ✅ FORZAR RESOLUCIÓN 640x640 EN TODA LA PIPELINE - CRÍTICO
        self.TARGET_WIDTH = 640
        self.TARGET_HEIGHT = 640
        self.PROCESSING_SIZE = (self.TARGET_WIDTH, self.TARGET_HEIGHT)
        
        logger.info(f"📐 VideoProcessor inicializado con resolución FORZADA: {self.TARGET_WIDTH}x{self.TARGET_HEIGHT}")
        
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
        
        # Frame sharing para web - ✅ SIEMPRE 640x640
        self.latest_frame = None
        self.latest_raw_frame = None
        self.frame_lock = threading.Lock()
        
        # Control de errores y reconexión
        self.connection_retry_count = 0
        self.max_retries = 5
        self.last_successful_frame = time.time()
        
        # Estadísticas de procesamiento
        self.last_processing_time = 0
        self.last_detections_count = 0
        self.current_tracks = []
        self.loop = None

    async def initialize(self):
        """Inicializar todos los componentes - CON FORZADO 640x640"""
        try:
            logger.info("🔄 Inicializando VideoProcessor con resolución forzada 640x640...")
            
            # 1. Inicializar detector con resolución forzada
            try:
                model_path = self.system_config.get('model_path', '/app/models/yolo11n.rknn')
                confidence = self.system_config.get('confidence_threshold', 0.5)
                
                logger.info(f"🤖 Inicializando detector YOLO11n RKNN (640x640) con confianza: {confidence}")
                logger.info(f"📂 Modelo: {model_path}")
                
                self.detector = VehicleDetector(model_path, confidence)
                
                if self.detector is None:
                    raise Exception("Detector no se pudo inicializar")
                
                # ✅ VERIFICAR QUE EL DETECTOR USE 640x640
                detector_info = self.detector.get_model_info()
                logger.info(f"🤖 Detector: {detector_info.get('model_type', 'unknown')} | RKNN: {detector_info.get('use_rknn', False)}")
                logger.info(f"📐 Input size configurado: {detector_info.get('input_size', 'unknown')}")
                
                # ✅ VERIFICAR ESPECÍFICAMENTE QUE USE RKNN
                if detector_info.get('use_rknn', False):
                    logger.info("✅ Detector usando RKNN NPU - rendimiento esperado: ~99.5ms")
                else:
                    logger.warning("⚠️ Detector NO está usando RKNN - rendimiento reducido")
                
            except Exception as e:
                logger.error(f"❌ Error inicializando detector: {e}")
                raise Exception(f"Error inicializando detector: {e}")
            
            # 2. Inicializar tracker
            try:
                logger.info("📍 Inicializando tracker...")
                self.tracker = BYTETracker(
                    high_thresh=self.system_config.get('high_threshold', 0.6),
                    low_thresh=self.system_config.get('low_threshold', 0.1),
                    max_age=self.system_config.get('max_age', 30)
                )
                if self.tracker is None:
                    raise Exception("Tracker no se pudo inicializar")
                logger.info("✅ Tracker inicializado")
            except Exception as e:
                logger.error(f"❌ Error inicializando tracker: {e}")
                raise Exception(f"Error inicializando tracker: {e}")
            
            # 3. Inicializar analizador
            try:
                logger.info("📊 Inicializando analizador...")
                self.analyzer = TrafficAnalyzer()
                if self.analyzer is None:
                    raise Exception("Analyzer no se pudo inicializar")
                logger.info("✅ Analizador inicializado")
            except Exception as e:
                logger.error(f"❌ Error inicializando analizador: {e}")
                raise Exception(f"Error inicializando analizador: {e}")
            
            # 4. Cargar configuración de líneas y zonas
            try:
                await self._load_analysis_config()
            except Exception as e:
                logger.warning(f"⚠️ Error cargando configuración de análisis: {e}")
            
            # 5. Inicializar base de datos del día
            try:
                if self.db_manager:
                    await self.db_manager.init_daily_database()
                    logger.info("✅ Base de datos inicializada")
            except Exception as e:
                logger.warning(f"⚠️ Error inicializando base de datos: {e}")
            
            logger.info("✅ VideoProcessor inicializado correctamente con pipeline 640x640")
            
        except Exception as e:
            logger.error(f"❌ Error inicializando VideoProcessor: {e}")
            raise
    
    def set_event_loop(self, loop: asyncio.AbstractEventLoop):
        self.loop = loop

    async def _load_analysis_config(self):
        """Cargar configuración de análisis desde archivo"""
        try:
            os.makedirs("/app/config", exist_ok=True)
            
            analysis_file = "/app/config/analysis.json"
            if not os.path.exists(analysis_file):
                logger.info("📝 Creando archivo de análisis por defecto")
                default_analysis = {"lines": {}, "zones": {}}
                with open(analysis_file, "w") as f:
                    json.dump(default_analysis, f, indent=2)
                return
            
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
        
        logger.info("✅ Procesamiento de video iniciado con resolución 640x640")
    
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
        
        logger.info("⏹️ Procesamiento de video detenido")
    
    def get_raw_frame(self) -> Optional[np.ndarray]:
        """Obtener frame original - SIEMPRE 640x640"""
        with self.frame_lock:
            if hasattr(self, 'latest_raw_frame') and self.latest_raw_frame is not None:
                frame = self.latest_raw_frame.copy()
                # ✅ ASEGURAR 640x640
                if frame.shape[:2] != (640, 640):
                    frame = cv2.resize(frame, (640, 640))
                return frame
            return None

    def get_latest_frame(self) -> Optional[np.ndarray]:
        """Obtener último frame procesado con análisis - SIEMPRE 640x640"""
        with self.frame_lock:
            if self.latest_frame is not None:
                frame = self.latest_frame.copy()
                # ✅ ASEGURAR 640x640
                if frame.shape[:2] != (640, 640):
                    frame = cv2.resize(frame, (640, 640))
                return frame
            return None

    def get_frame_info(self) -> Dict:
        """Obtener información del frame actual"""
        return {
            'fps': self.current_fps,
            'resolution': f"{self.TARGET_WIDTH}x{self.TARGET_HEIGHT}",  # ✅ SIEMPRE 640x640
            'processing_time': getattr(self, 'last_processing_time', 0),
            'tracks_count': len(getattr(self, 'current_tracks', [])),
            'detections_count': getattr(self, 'last_detections_count', 0),
            'connection_retries': self.connection_retry_count,
            'input_size': self.PROCESSING_SIZE,
            'target_platform': 'rk3588',
            'forced_resolution': True
        }

    def _validate_rtsp_url(self, rtsp_url: str) -> tuple[bool, str]:
        """Validar URL RTSP y corregir problemas comunes"""
        if not rtsp_url or not rtsp_url.strip():
            return False, "URL RTSP vacía"
        
        if not rtsp_url.startswith('rtsp://'):
            return False, "URL debe empezar con rtsp://"
        
        if '0.0.0.0' in rtsp_url:
            return False, "IP 0.0.0.0 no es válida. Configure la IP real de la cámara."
        
        ip_pattern = r'rtsp://[^@]*@([0-9]{1,3}\.[0-9]{1,3}\.[0-9]{1,3}\.[0-9]{1,3})'
        match = re.search(ip_pattern, rtsp_url)
        if match:
            ip = match.group(1)
            if ip.startswith('127.') or ip == '255.255.255.255':
                return False, f"IP {ip} no es válida para cámara remota"
        
        return True, "URL válida"

    def _processing_loop(self):
        """Loop principal de procesamiento - CON RESOLUCIÓN FORZADA 640x640"""
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
        """Conectar al stream RTSP - FORZANDO RESOLUCIÓN 640x640"""
        try:
            # Cerrar conexión anterior si existe
            if self.video_capture:
                self.video_capture.release()
                time.sleep(1)
            
            # Crear nueva captura
            self.video_capture = cv2.VideoCapture(rtsp_url)
            
            # ✅ CONFIGURACIÓN FORZADA PARA 640x640 - CRÍTICO
            try:
                # Forzar resolución exacta 640x640
                self.video_capture.set(cv2.CAP_PROP_FRAME_WIDTH, self.TARGET_WIDTH)
                self.video_capture.set(cv2.CAP_PROP_FRAME_HEIGHT, self.TARGET_HEIGHT)
                self.video_capture.set(cv2.CAP_PROP_FPS, 30)
                
                # Configurar buffer pequeño para baja latencia
                try:
                    buffer_set = self.video_capture.set(cv2.CAP_PROP_BUFFERSIZE, 1)
                    if buffer_set:
                        logger.info("✅ Buffer size configurado a 1")
                except:
                    logger.info("ℹ️ Buffer size no disponible")
                
                # Timeout si está disponible
                try:
                    self.video_capture.set(cv2.CAP_PROP_OPEN_TIMEOUT_MSEC, 10000)
                except:
                    pass
                    
                logger.info(f"📐 Configurando captura a {self.TARGET_WIDTH}x{self.TARGET_HEIGHT}")
                    
            except Exception as e:
                logger.warning(f"⚠️ Algunas configuraciones de captura no se aplicaron: {e}")
            
            # Verificar conexión
            if not self.video_capture.isOpened():
                logger.error("❌ No se pudo abrir el stream RTSP")
                return False
            
            # Test de lectura con verificación de resolución
            logger.info("🧪 Probando lectura de frames...")
            test_frames = 0
            for i in range(5):
                ret, frame = self.video_capture.read()
                if ret and frame is not None:
                    test_frames += 1
                    
                    # ✅ VERIFICAR Y FORZAR RESOLUCIÓN EN EL PRIMER FRAME
                    if i == 0:
                        original_h, original_w = frame.shape[:2]
                        logger.info(f"📐 Frame original: {original_w}x{original_h}")
                        
                        # Si no es 640x640, lo forzamos AQUÍ
                        if original_w != self.TARGET_WIDTH or original_h != self.TARGET_HEIGHT:
                            logger.info(f"🔄 Redimensionando de {original_w}x{original_h} a {self.TARGET_WIDTH}x{self.TARGET_HEIGHT}")
                else:
                    time.sleep(0.2)
            
            if test_frames >= 2:
                logger.info(f"✅ Stream RTSP conectado exitosamente ({test_frames}/5 frames) - Pipeline 640x640")
                self.connection_retry_count = 0
                return True
            else:
                logger.error(f"❌ Stream inestable ({test_frames}/5 frames)")
                return False
                
        except Exception as e:
            logger.error(f"❌ Error conectando stream: {e}")
            return False

    def _main_processing_loop(self):
        """Loop principal de procesamiento - CON REDIMENSIONADO FORZADO"""
        logger.info("🎬 Iniciando procesamiento de frames con análisis completo (640x640)...")
        
        # Variables para control de FPS
        target_fps = self.system_config.get('target_fps', 30)
        frame_time = 1.0 / target_fps
        last_frame_time = time.time()
        consecutive_failures = 0
        max_failures = 30
        
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
                    
                    # ✅ FORZAR REDIMENSIONADO A 640x640 INMEDIATAMENTE - CRÍTICO
                    original_h, original_w = frame.shape[:2]
                    if original_w != self.TARGET_WIDTH or original_h != self.TARGET_HEIGHT:
                        frame = cv2.resize(frame, self.PROCESSING_SIZE)
                        if self.fps_counter % 100 == 0:  # Log cada 100 frames para no saturar
                            logger.debug(f"🔄 Frame redimensionado: {original_w}x{original_h} → {self.TARGET_WIDTH}x{self.TARGET_HEIGHT}")
                    
                    # PROCESAR FRAME CON ANÁLISIS COMPLETO
                    try:
                        processed_frame = self._process_frame_with_analysis(frame)
                        
                        # ✅ VERIFICAR QUE EL FRAME PROCESADO SEA 640x640
                        if processed_frame.shape[:2] != (self.TARGET_HEIGHT, self.TARGET_WIDTH):
                            processed_frame = cv2.resize(processed_frame, self.PROCESSING_SIZE)
                        
                        # Actualizar frame compartido para web - SIEMPRE 640x640
                        with self.frame_lock:
                            self.latest_frame = processed_frame
                            self.latest_raw_frame = frame.copy()  # También 640x640
                        
                        # Actualizar FPS
                        self._update_fps()
                        
                        last_frame_time = current_time
                        
                    except Exception as processing_error:
                        logger.error(f"❌ Error procesando frame: {processing_error}")
                        # Usar frame redimensionado si falla el procesamiento
                        with self.frame_lock:
                            self.latest_frame = frame
                            self.latest_raw_frame = frame.copy()
                
                else:
                    # Frame fallido
                    consecutive_failures += 1
                    logger.warning(f"⚠️ No se pudo leer frame ({consecutive_failures}/{max_failures})")
                    
                    if consecutive_failures >= max_failures:
                        logger.error("❌ Demasiados frames fallidos consecutivos")
                        break
                    
                    time.sleep(0.1)
                
                # Verificar timeout de conexión
                if current_time - self.last_successful_frame > 30:
                    logger.error("❌ Timeout de conexión - sin frames por 30 segundos")
                    break
                    
            except Exception as e:
                logger.error(f"❌ Error en loop principal: {e}")
                consecutive_failures += 1
                if consecutive_failures >= max_failures:
                    break
                time.sleep(0.1)
        
        logger.info("🔚 Loop de procesamiento terminado")
    
    def _process_frame_with_analysis(self, frame: np.ndarray) -> np.ndarray:
        """Procesar frame con análisis completo de IA - FRAME YA ES 640x640"""
        start_time = time.time()
        
        try:
            # ✅ VERIFICAR QUE EL FRAME SEA EXACTAMENTE 640x640
            if frame.shape[:2] != (self.TARGET_HEIGHT, self.TARGET_WIDTH):
                frame = cv2.resize(frame, self.PROCESSING_SIZE)
                logger.debug(f"🔄 Frame forzado a 640x640 en processing")
            
            # Mejorar imagen si es necesario (modo nocturno)
            if self.system_config.get('night_vision_enhancement', False):
                frame = self.detector.enhance_night_vision(frame)
            
            # DETECCIÓN - CRÍTICO CON FRAME 640x640
            detections = []
            if self.detector:
                try:
                    # ✅ EL DETECTOR RECIBE FRAME 640x640 EXACTO
                    detections = self.detector.detect(frame)
                    self.last_detections_count = len(detections)
                    
                    if len(detections) > 0:
                        logger.debug(f"🔍 Detecciones (640x640): {len(detections)}")
                    
                except Exception as e:
                    logger.error(f"❌ Error en detección: {e}")
                    self.last_detections_count = 0
            
            # TRACKING - CRÍTICO
            tracks = []
            if self.tracker and detections:
                try:
                    tracks = self.tracker.update(detections)
                    self.current_tracks = tracks
                    
                    if len(tracks) > 0:
                        logger.debug(f"📍 Tracks activos: {len(tracks)}")
                        
                except Exception as e:
                    logger.error(f"❌ Error en tracking: {e}")
                    self.current_tracks = []
            
            # ANÁLISIS DE TRÁFICO - CRÍTICO
            if self.analyzer and tracks:
                try:
                    # ✅ PASAR SHAPE 640x640 AL ANALYZER
                    analysis_results = self.analyzer.analyze_frame(tracks, (self.TARGET_HEIGHT, self.TARGET_WIDTH))
                    
                    # Procesar resultados de forma asíncrona
                    if analysis_results:
                        if self.loop is not None:
                            asyncio.run_coroutine_threadsafe(
                                self._process_analysis_results(analysis_results, tracks),
                                self.loop
                            )
                        else:
                            logger.error("❌ No hay event loop disponible para análisis")
                            
                except Exception as e:
                    logger.error(f"❌ Error en análisis: {e}")
            
            # DIBUJAR OVERLAY CON DETECCIONES Y TRACKING
            frame = self._draw_detection_overlay(frame, detections, tracks)
            
            # Dibujar overlay de análisis si está habilitado
            if self.system_config.get('show_overlay', True) and self.analyzer:
                try:
                    frame = self.analyzer.draw_analysis_overlay(frame, tracks)
                except Exception as e:
                    logger.error(f"❌ Error dibujando overlay de análisis: {e}")
            
            # ✅ ASEGURAR QUE EL FRAME FINAL SEA 640x640
            if frame.shape[:2] != (self.TARGET_HEIGHT, self.TARGET_WIDTH):
                frame = cv2.resize(frame, self.PROCESSING_SIZE)
            
            # Guardar tiempo de procesamiento
            processing_time = time.time() - start_time
            self.last_processing_time = processing_time
            
            # Log cada 100 frames para no saturar
            if self.fps_counter % 100 == 0:
                logger.debug(f"📊 Frame procesado (640x640): {processing_time*1000:.1f}ms | Det: {len(detections)} | Tracks: {len(tracks)}")
            
            return frame
            
        except Exception as e:
            logger.error(f"❌ Error procesando frame: {e}")
            # Devolver frame 640x640 en caso de error
            if frame.shape[:2] != (self.TARGET_HEIGHT, self.TARGET_WIDTH):
                frame = cv2.resize(frame, self.PROCESSING_SIZE)
            return frame
    
    def _draw_detection_overlay(self, frame: np.ndarray, detections: List, tracks: List) -> np.ndarray:
        """Dibujar overlay de detecciones y tracking - OPTIMIZADO PARA 640x640"""
        try:
            # Dibujar detecciones (bounding boxes)
            for detection in detections:
                bbox = detection['bbox']
                confidence = detection['confidence']
                class_name = detection['class_name']
                
                x, y, w, h = bbox
                
                # Color según clase
                colors = {
                    'car': (0, 255, 0),      # Verde
                    'truck': (0, 0, 255),    # Rojo  
                    'bus': (255, 0, 0),      # Azul
                    'motorcycle': (0, 255, 255)  # Amarillo
                }
                color = colors.get(class_name, (255, 255, 255))
                
                # Dibujar bounding box
                cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
                
                # Etiqueta
                label = f"{class_name} {confidence:.2f}"
                label_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)[0]
                
                # Fondo para etiqueta
                cv2.rectangle(frame, (x, y - label_size[1] - 5), 
                             (x + label_size[0], y), color, -1)
                
                # Texto
                cv2.putText(frame, label, (x, y - 3), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)
            
            # Dibujar tracks (IDs y trayectorias)
            for track in tracks:
                center = track.center
                track_id = track.track_id
                
                # Punto central
                cv2.circle(frame, (int(center[0]), int(center[1])), 3, (255, 0, 255), -1)
                
                # ID del track
                cv2.putText(frame, f"ID:{track_id}", 
                           (int(center[0]) + 5, int(center[1]) - 5),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
                
                # Trayectoria (últimas 5 posiciones para no saturar)
                if len(track.history) > 1:
                    points = []
                    for hist_bbox in track.history[-5:]:
                        hist_center = (hist_bbox[0] + hist_bbox[2]//2, 
                                     hist_bbox[1] + hist_bbox[3]//2)
                        points.append(hist_center)
                    
                    # Dibujar línea de trayectoria
                    for i in range(1, len(points)):
                        cv2.line(frame, points[i-1], points[i], (255, 0, 255), 1)
            
            # Información del sistema - AJUSTADA PARA 640x640
            info_y = 20
            cv2.putText(frame, f"Det: {len(detections)} | Tracks: {len(tracks)}", (5, info_y),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            
            # FPS
            if hasattr(self, 'current_fps'):
                cv2.putText(frame, f"FPS: {self.current_fps}", (5, info_y + 20),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            
            # Resolución para verificar
            cv2.putText(frame, f"640x640", (5, info_y + 40),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 0), 1)
            
            # Indicador RKNN si está activo
            if self.detector and hasattr(self.detector, 'use_rknn') and self.detector.use_rknn:
                cv2.putText(frame, f"RKNN", (5, info_y + 60),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 255), 1)
            
            return frame
            
        except Exception as e:
            logger.error(f"❌ Error dibujando overlay: {e}")
            return frame

    async def _process_analysis_results(self, results: Dict, tracks: List):
        """Procesar resultados del análisis"""
        try:
            # Procesar cruces de líneas
            for crossing in results.get('line_crossings', []):
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
                            calculated_speed = track.average_velocity * 3.6
                    
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
                            'model_type': getattr(self.detector, 'model_type', 'unknown'),
                            'frame_size': '640x640',  # ✅ REGISTRAR RESOLUCIÓN
                            'target_platform': 'rk3588'
                        }
                    }
                    
                    if self.db_manager:
                        await self.db_manager.insert_vehicle_crossing(crossing_data)
            
            # Enviar analítico si es necesario
            if results.get('send_analytic') and self.callback_func is not None:
                analytic_data = {
                    'fase': self.camera_config.get('fase', 'fase1'),
                    'puntos': len(results.get('vehicles_in_red_zone', [])),
                    'vehiculos': True
                }
                try:
                    await self.callback_func('send_analytic', analytic_data)
                except Exception as e:
                    logger.error(f"❌ Error en callback: {e}")
        
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
            'detector_available': self.detector is not None,
            'tracker_available': self.tracker is not None,
            'analyzer_available': self.analyzer is not None,
            'resolution': f"{self.TARGET_WIDTH}x{self.TARGET_HEIGHT}",  # ✅ SIEMPRE 640x640
            'input_size': self.PROCESSING_SIZE,
            'target_platform': 'rk3588',
            'forced_resolution': True,
            'detector_type': getattr(self.detector, 'model_type', 'unknown') if self.detector else 'none',
            'using_rknn': getattr(self.detector, 'use_rknn', False) if self.detector else False
        }