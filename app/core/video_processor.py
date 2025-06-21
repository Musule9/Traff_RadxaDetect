# REEMPLAZA app/core/video_processor.py COMPLETAMENTE

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
    """Procesador optimizado para 30 FPS con RKNN RK3588"""
    
    def __init__(self, 
                 camera_config: Dict,
                 system_config: Dict,
                 db_manager: DatabaseManager,
                 callback_func: Optional[Callable] = None):
        
        self.camera_config = camera_config
        self.system_config = system_config
        self.db_manager = db_manager
        self.callback_func = callback_func
        
        # ✅ OPTIMIZACIÓN: RESOLUCIÓN Y FPS FIJOS
        self.TARGET_WIDTH = 640
        self.TARGET_HEIGHT = 640
        self.TARGET_FPS = 30  # ✅ OBJETIVO: 30 FPS REAL
        self.PROCESSING_SIZE = (self.TARGET_WIDTH, self.TARGET_HEIGHT)
        
        logger.info(f"📐 VideoProcessor optimizado: 640x640 @ {self.TARGET_FPS} FPS")
        
        # Componentes principales
        self.detector = None
        self.tracker = None
        self.analyzer = None
        
        # ✅ OPTIMIZACIÓN: CONTROL DE ESTADO MÁS ROBUSTO
        self.is_running = False
        self.video_capture = None
        self.processing_thread = None
        self.should_stop = False
        
        # ✅ OPTIMIZACIÓN: MÉTRICAS DE RENDIMIENTO
        self.fps_counter = 0
        self.fps_time = time.time()
        self.current_fps = 0
        self.processing_fps = 0
        self.detection_count = 0
        self.track_count = 0
        
        # ✅ OPTIMIZACIÓN: FRAME SHARING MEJORADO
        self.latest_frame = None
        self.latest_raw_frame = None
        self.frame_lock = threading.Lock()
        self.frame_queue = Queue(maxsize=2)  # Buffer pequeño para baja latencia
        
        # ✅ OPTIMIZACIÓN: CONTROL DE ERRORES MEJORADO
        self.connection_retry_count = 0
        self.max_retries = 3  # Menos reintentos
        self.last_successful_frame = time.time()
        self.consecutive_failures = 0
        self.max_consecutive_failures = 10  # Menos tolerancia a fallos
        
        # ✅ OPTIMIZACIÓN: ESTADÍSTICAS DETALLADAS
        self.last_processing_time = 0
        self.last_detection_time = 0
        self.last_tracking_time = 0
        self.last_analysis_time = 0
        
        self.loop = None

    async def initialize(self):
        """Inicializar con optimizaciones para 30 FPS"""
        try:
            logger.info("🔄 Inicializando VideoProcessor optimizado para 30 FPS...")
            
            # 1. Detector optimizado
            try:
                model_path = self.system_config.get('model_path', '/app/models/yolo11n.rknn')
                confidence = self.system_config.get('confidence_threshold', 0.4)  # ✅ BAJAMOS THRESHOLD
                
                logger.info(f"🤖 Inicializando detector RKNN optimizado con confianza: {confidence}")
                
                self.detector = VehicleDetector(model_path, confidence)
                
                if self.detector is None:
                    raise Exception("Detector no se pudo inicializar")
                
                detector_info = self.detector.get_model_info()
                if detector_info.get('use_rknn', False):
                    logger.info("✅ Detector RKNN activo - procesamiento optimizado")
                else:
                    logger.warning("⚠️ Detector NO usa RKNN - rendimiento limitado")
                
            except Exception as e:
                logger.error(f"❌ Error inicializando detector: {e}")
                raise
            
            # 2. Tracker optimizado
            try:
                logger.info("📍 Inicializando tracker optimizado...")
                self.tracker = BYTETracker(
                    high_thresh=0.6,
                    low_thresh=0.2,  # ✅ THRESHOLD MÁS PERMISIVO
                    max_age=15,      # ✅ VIDA MÁS CORTA PARA TRACKS
                    min_hits=2       # ✅ MENOS HITS REQUERIDOS
                )
                logger.info("✅ Tracker optimizado inicializado")
            except Exception as e:
                logger.error(f"❌ Error inicializando tracker: {e}")
                raise
            
            # 3. Analizador
            try:
                logger.info("📊 Inicializando analizador...")
                self.analyzer = TrafficAnalyzer()
                logger.info("✅ Analizador inicializado")
            except Exception as e:
                logger.error(f"❌ Error inicializando analizador: {e}")
                raise
            
            # 4. Cargar configuración de análisis
            try:
                await self._load_analysis_config()
            except Exception as e:
                logger.warning(f"⚠️ Error cargando análisis: {e}")
            
            # 5. DB
            try:
                if self.db_manager:
                    await self.db_manager.init_daily_database()
                    logger.info("✅ Base de datos inicializada")
            except Exception as e:
                logger.warning(f"⚠️ Error con base de datos: {e}")
            
            logger.info("✅ VideoProcessor optimizado inicializado correctamente")
            
        except Exception as e:
            logger.error(f"❌ Error crítico inicializando VideoProcessor: {e}")
            raise
    
    def set_event_loop(self, loop: asyncio.AbstractEventLoop):
        self.loop = loop

    async def _load_analysis_config(self):
        """Cargar configuración de análisis"""
        try:
            os.makedirs("/app/config", exist_ok=True)
            
            analysis_file = "/app/config/analysis.json"
            if not os.path.exists(analysis_file):
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
        """Iniciar procesamiento optimizado"""
        if self.is_running:
            logger.warning("⚠️ Procesamiento ya está ejecutándose")
            return
        
        self.is_running = True
        self.should_stop = False
        self.processing_thread = threading.Thread(target=self._processing_loop_optimized)
        self.processing_thread.daemon = True
        self.processing_thread.start()
        
        logger.info("✅ Procesamiento optimizado iniciado")
    
    def stop_processing(self):
        """Detener procesamiento"""
        logger.info("⏹️ Deteniendo procesamiento...")
        self.should_stop = True
        self.is_running = False
        
        if self.video_capture:
            try:
                self.video_capture.release()
            except:
                pass
        
        if self.processing_thread:
            self.processing_thread.join(timeout=10)
        
        logger.info("✅ Procesamiento detenido")
    
    def _processing_loop_optimized(self):
        """Loop principal OPTIMIZADO para 30 FPS"""
        rtsp_url = self.camera_config.get('rtsp_url')
        
        if not rtsp_url or not rtsp_url.strip():
            logger.error("❌ No hay URL RTSP configurada")
            self.is_running = False
            return
        
        logger.info(f"📡 Conectando a RTSP optimizado: {rtsp_url}")
        
        # ✅ OPTIMIZACIÓN: CONEXIÓN MÁS ROBUSTA
        retry_count = 0
        while self.is_running and not self.should_stop and retry_count < self.max_retries:
            try:
                if self._connect_to_stream_optimized(rtsp_url):
                    self._main_processing_loop_optimized()
                    break
                else:
                    retry_count += 1
                    if retry_count < self.max_retries:
                        wait_time = min(3 * retry_count, 10)
                        logger.warning(f"⏳ Reintento {retry_count}/{self.max_retries} en {wait_time}s...")
                        time.sleep(wait_time)
                    else:
                        logger.error("❌ Máximo de reintentos alcanzado")
                        
            except Exception as e:
                logger.error(f"❌ Error en loop de procesamiento: {e}")
                retry_count += 1
                time.sleep(2)
        
        self.is_running = False
        logger.info("🔚 Loop de procesamiento terminado")

    def _connect_to_stream_optimized(self, rtsp_url: str) -> bool:
        """Conexión optimizada para 30 FPS"""
        try:
            if self.video_capture:
                self.video_capture.release()
                time.sleep(0.5)
            
            # ✅ OPTIMIZACIÓN: CONFIGURACIÓN ESPECÍFICA PARA ALTA VELOCIDAD
            self.video_capture = cv2.VideoCapture(rtsp_url)
            
            # Configuración optimizada para 30 FPS
            self.video_capture.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
            self.video_capture.set(cv2.CAP_PROP_FRAME_HEIGHT, 640)
            self.video_capture.set(cv2.CAP_PROP_FPS, 30)
            self.video_capture.set(cv2.CAP_PROP_BUFFERSIZE, 1)  # Buffer mínimo
            
            # Configuraciones adicionales para rendimiento
            try:
                self.video_capture.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc('H', '2', '6', '4'))
                self.video_capture.set(cv2.CAP_PROP_OPEN_TIMEOUT_MSEC, 5000)
                self.video_capture.set(cv2.CAP_PROP_READ_TIMEOUT_MSEC, 5000)
            except:
                pass
            
            if not self.video_capture.isOpened():
                logger.error("❌ No se pudo abrir stream RTSP")
                return False
            
            # Test optimizado
            logger.info("🧪 Test de conexión optimizado...")
            test_frames = 0
            start_test = time.time()
            
            for i in range(8):  # Test más rápido
                ret, frame = self.video_capture.read()
                if ret and frame is not None:
                    test_frames += 1
                    if i == 0:
                        original_h, original_w = frame.shape[:2]
                        logger.info(f"📐 Frame original: {original_w}x{original_h}")
                else:
                    time.sleep(0.1)
            
            test_time = time.time() - start_test
            if test_frames >= 5:
                fps_test = test_frames / test_time if test_time > 0 else 0
                logger.info(f"✅ Stream conectado: {test_frames}/8 frames en {test_time:.1f}s ({fps_test:.1f} FPS)")
                self.connection_retry_count = 0
                return True
            else:
                logger.error(f"❌ Stream inestable: {test_frames}/8 frames")
                return False
                
        except Exception as e:
            logger.error(f"❌ Error conectando: {e}")
            return False

    def _main_processing_loop_optimized(self):
        """Loop principal OPTIMIZADO para máximo rendimiento"""
        logger.info("🎬 Iniciando procesamiento optimizado para 30 FPS...")
        
        # ✅ OPTIMIZACIÓN: CONTROL DE TIMING PRECISO
        frame_interval = 1.0 / self.TARGET_FPS  # ~33ms por frame
        last_frame_time = time.time()
        last_fps_update = time.time()
        frame_count = 0
        
        # ✅ OPTIMIZACIÓN: PROCESAMIENTO INTELIGENTE
        process_every_nth = 1  # Procesar todos los frames inicialmente
        detection_interval = 0.1  # 10 detecciones por segundo máximo
        last_detection_time = 0
        
        while self.is_running and not self.should_stop:
            try:
                current_time = time.time()
                
                # ✅ CONTROL DE FPS OPTIMIZADO
                time_since_last_frame = current_time - last_frame_time
                if time_since_last_frame < frame_interval:
                    sleep_time = frame_interval - time_since_last_frame
                    if sleep_time > 0.001:  # Solo dormir si vale la pena
                        time.sleep(sleep_time)
                    continue
                
                # Leer frame
                ret, frame = self.video_capture.read()
                
                if ret and frame is not None:
                    self.consecutive_failures = 0
                    self.last_successful_frame = current_time
                    
                    # ✅ REDIMENSIONAR INMEDIATAMENTE
                    if frame.shape[:2] != (640, 640):
                        frame = cv2.resize(frame, (640, 640))
                    
                    # ✅ PROCESAMIENTO INTELIGENTE
                    should_process = (current_time - last_detection_time) >= detection_interval
                    
                    if should_process:
                        try:
                            processed_frame = self._process_frame_optimized(frame, current_time)
                            last_detection_time = current_time
                        except Exception as e:
                            logger.error(f"❌ Error procesando frame: {e}")
                            processed_frame = frame
                    else:
                        # Solo dibujar overlay sin detección
                        processed_frame = self._draw_simple_overlay(frame)
                    
                    # Actualizar frames compartidos
                    with self.frame_lock:
                        self.latest_frame = processed_frame
                        self.latest_raw_frame = frame.copy()
                    
                    # ✅ ACTUALIZAR MÉTRICAS
                    frame_count += 1
                    last_frame_time = current_time
                    
                    # Actualizar FPS cada segundo
                    if current_time - last_fps_update >= 1.0:
                        self.current_fps = frame_count
                        frame_count = 0
                        last_fps_update = current_time
                        
                        # ✅ LOG DE RENDIMIENTO CADA 10 SEGUNDOS
                        if int(current_time) % 10 == 0:
                            logger.debug(f"📊 FPS: {self.current_fps} | Detecciones: {self.detection_count} | Tracks: {self.track_count}")
                
                else:
                    # Frame fallido
                    self.consecutive_failures += 1
                    if self.consecutive_failures >= self.max_consecutive_failures:
                        logger.error(f"❌ Demasiados frames fallidos: {self.consecutive_failures}")
                        break
                    time.sleep(0.05)
                
                # ✅ VERIFICAR TIMEOUT
                if current_time - self.last_successful_frame > 30:
                    logger.error("❌ Timeout - sin frames por 30 segundos")
                    break
                    
            except Exception as e:
                logger.error(f"❌ Error en loop principal: {e}")
                self.consecutive_failures += 1
                if self.consecutive_failures >= self.max_consecutive_failures:
                    break
                time.sleep(0.1)
        
        # Cleanup
        if self.video_capture:
            self.video_capture.release()
        
        logger.info("🏁 Loop de procesamiento optimizado terminado")

    def _process_frame_optimized(self, frame: np.ndarray, current_time: float) -> np.ndarray:
        """Procesamiento de frame OPTIMIZADO"""
        try:
            start_time = time.time()
            
            # ✅ DETECCIÓN OPTIMIZADA
            detections = []
            if self.detector:
                detection_start = time.time()
                detections = self.detector.detect(frame)
                self.last_detection_time = (time.time() - detection_start) * 1000
                self.detection_count = len(detections)
            
            # ✅ TRACKING OPTIMIZADO
            tracks = []
            if self.tracker and detections:
                tracking_start = time.time()
                tracks = self.tracker.update(detections)
                self.last_tracking_time = (time.time() - tracking_start) * 1000
                self.track_count = len(tracks)
            
            # ✅ ANÁLISIS LIGERO
            if self.analyzer and tracks and len(tracks) > 0:
                try:
                    analysis_start = time.time()
                    analysis_results = self.analyzer.analyze_frame(tracks, (640, 640))
                    self.last_analysis_time = (time.time() - analysis_start) * 1000
                    
                    # Procesar resultados de forma asíncrona
                    if analysis_results and self.loop:
                        asyncio.run_coroutine_threadsafe(
                            self._process_analysis_results(analysis_results, tracks),
                            self.loop
                        )
                except Exception as e:
                    logger.error(f"❌ Error en análisis: {e}")
            
            # ✅ DIBUJAR OVERLAY OPTIMIZADO
            frame = self._draw_optimized_overlay(frame, detections, tracks)
            
            # ✅ MÉTRICAS DE RENDIMIENTO
            self.last_processing_time = (time.time() - start_time) * 1000
            
            return frame
            
        except Exception as e:
            logger.error(f"❌ Error en procesamiento optimizado: {e}")
            return frame

    def _draw_optimized_overlay(self, frame: np.ndarray, detections: List, tracks: List) -> np.ndarray:
        """Overlay optimizado para alto rendimiento"""
        try:
            # ✅ DIBUJAR SOLO LO ESENCIAL
            
            # Detecciones con bounding boxes simples
            for detection in detections:
                bbox = detection['bbox']
                confidence = detection['confidence']
                class_name = detection['class_name']
                
                x, y, w, h = bbox
                
                # Colores simples por clase
                color = (0, 255, 0) if class_name == 'car' else (0, 0, 255) if class_name == 'truck' else (255, 0, 0)
                
                # Box simple
                cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
                
                # Etiqueta mínima
                label = f"{class_name[:3]} {confidence:.1f}"
                cv2.putText(frame, label, (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
            
            # Tracks con IDs simples
            for track in tracks:
                center = track.center
                track_id = track.track_id
                
                # Punto central
                cv2.circle(frame, (int(center[0]), int(center[1])), 3, (255, 0, 255), -1)
                
                # ID
                cv2.putText(frame, f"{track_id}", (int(center[0]) + 5, int(center[1]) - 5),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
            
            # ✅ INFO DE RENDIMIENTO EN ESQUINA
            info_y = 20
            cv2.putText(frame, f"FPS: {self.current_fps}", (10, info_y),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 1)
            
            cv2.putText(frame, f"Det: {len(detections)} | Trk: {len(tracks)}", (10, info_y + 25),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            
            if hasattr(self, 'last_detection_time') and self.last_detection_time > 0:
                cv2.putText(frame, f"RKNN: {self.last_detection_time:.1f}ms", (10, info_y + 50),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 255), 1)
            
            return frame
            
        except Exception as e:
            logger.error(f"❌ Error dibujando overlay: {e}")
            return frame

    def _draw_simple_overlay(self, frame: np.ndarray) -> np.ndarray:
        """Overlay simple sin procesamiento de IA"""
        try:
            cv2.putText(frame, f"FPS: {self.current_fps}", (10, 20),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 1)
            cv2.putText(frame, "OPTIMIZADO RKNN", (10, 45),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 255), 1)
            return frame
        except:
            return frame

    async def _process_analysis_results(self, results: Dict, tracks: List):
        """Procesar resultados de análisis de forma asíncrona"""
        try:
            # Solo procesar cruces importantes para no sobrecargar
            for crossing in results.get('line_crossings', []):
                track = next((t for t in tracks if t.track_id == crossing['vehicle_id']), None)
                if track:
                    crossing_data = {
                        'vehicle_id': crossing['vehicle_id'],
                        'line_id': crossing['line_id'],
                        'line_name': crossing['line_name'],
                        'fase': self.camera_config.get('fase', 'fase1'),
                        'semaforo_estado': 'rojo' if self.analyzer.red_light_active else 'verde',
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
                            'resolution': '640x640',
                            'fps': self.current_fps
                        }
                    }
                    
                    if self.db_manager:
                        await self.db_manager.insert_vehicle_crossing(crossing_data)
        
        except Exception as e:
            logger.error(f"❌ Error procesando resultados: {e}")

    def get_latest_frame(self) -> Optional[np.ndarray]:
        """Obtener último frame procesado"""
        with self.frame_lock:
            return self.latest_frame.copy() if self.latest_frame is not None else None

    def get_raw_frame(self) -> Optional[np.ndarray]:
        """Obtener frame original"""
        with self.frame_lock:
            return self.latest_raw_frame.copy() if self.latest_raw_frame is not None else None

    def get_connection_status(self) -> Dict:
        """Estado detallado de conexión"""
        return {
            'connected': self.is_running and self.video_capture and self.video_capture.isOpened(),
            'fps': self.current_fps,
            'target_fps': self.TARGET_FPS,
            'detection_fps': 1000 / self.last_detection_time if self.last_detection_time > 0 else 0,
            'processing_time_ms': self.last_processing_time,
            'detection_time_ms': self.last_detection_time,
            'tracking_time_ms': self.last_tracking_time,
            'analysis_time_ms': self.last_analysis_time,
            'retry_count': self.connection_retry_count,
            'consecutive_failures': self.consecutive_failures,
            'last_successful_frame': self.last_successful_frame,
            'detector_available': self.detector is not None,
            'tracker_available': self.tracker is not None,
            'analyzer_available': self.analyzer is not None,
            'resolution': f"{self.TARGET_WIDTH}x{self.TARGET_HEIGHT}",
            'optimized': True,
            'using_rknn': getattr(self.detector, 'use_rknn', False) if self.detector else False,
            'detection_count': self.detection_count,
            'track_count': self.track_count
        }

    def update_red_light_status(self, is_red: bool):
        """Actualizar estado del semáforo"""
        if self.analyzer:
            self.analyzer.update_red_light_status(is_red)