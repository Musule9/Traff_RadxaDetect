import os
import subprocess
from pathlib import Path
from loguru import logger
import cv2
import numpy as np
import time

try:
    from ultralytics import YOLO
    ULTRALYTICS_AVAILABLE = True
except ImportError:
    ULTRALYTICS_AVAILABLE = False
    logger.warning("Ultralytics no disponible")

# ✅ IMPORTACIÓN CORRECTA PARA RK3588
RKNN_AVAILABLE = False
try:
    from rknnlite.api import RKNNLite  # ✅ CORRECTO para Radxa Rock 5T
    RKNN_AVAILABLE = True
    logger.info("✅ RKNNLite disponible para RK3588")
except Exception as e:
    logger.warning(f"❌ RKNNLite no disponible: {e}")

class VehicleDetector:
    """Detector de vehículos YOLO11n + RKNN optimizado ESPECÍFICAMENTE para Radxa Rock 5T"""
    
    def __init__(self, model_path: str = None, confidence_threshold: float = 0.5):
        self.confidence_threshold = confidence_threshold
        self.model = None
        self.rknn_model = None
        self.use_rknn = False
        self.model_type = "none"
        
        # ✅ RESOLUCIÓN FIJA 640x640 SIEMPRE
        self.input_size = (640, 640)
        self.input_width = 640
        self.input_height = 640
        
        # ✅ PARÁMETROS ESPECÍFICOS PARA RK3588
        self.platform = "rk3588"
        self.target_platform = "rk3588"
        
        # Asegurar directorio de modelos
        os.makedirs("/app/models", exist_ok=True)
        
        # Inicializar modelo
        self._initialize_model(model_path)
    
    def _initialize_model(self, model_path: str = None):
        """Inicializar modelo con PRIORIDAD: RKNN > PyTorch"""
        
        # ✅ 1. INTENTAR RKNN PRIMERO (ÚNICO PARA RK3588)
        if RKNN_AVAILABLE and self._check_rk3588_hardware():
            if self._init_yolo11n_rknn():
                return
        
        # 2. Fallback a YOLO11n PyTorch
        if ULTRALYTICS_AVAILABLE:
            if self._init_yolo11n_pytorch():
                return
        
        # 3. Fallback final
        logger.error("❌ No se pudo inicializar ningún modelo")
        self.model_type = "none"
    
    def _check_rk3588_hardware(self) -> bool:
        """Verificar hardware RK3588 específico"""
        
        # ✅ 1. VERIFICAR HARDWARE RK3588
        try:
            if os.path.exists("/proc/device-tree/model"):
                with open("/proc/device-tree/model", "rb") as f:
                    model = f.read().decode('utf-8', errors='ignore').strip('\x00')
                    if "RK3588" not in model:
                        logger.warning(f"⚠️ Hardware no es RK3588: {model}")
                        return False
                    logger.info(f"✅ Hardware RK3588 confirmado: {model}")
        except Exception as e:
            logger.warning(f"⚠️ No se pudo verificar hardware: {e}")
        
        # ✅ 2. VERIFICAR DRIVER NPU
        try:
            result = subprocess.run(['dmesg'], capture_output=True, text=True)
            if "rknpu" not in result.stdout:
                logger.warning("⚠️ Driver NPU no encontrado en dmesg")
                return False
            logger.info("✅ Driver NPU encontrado")
        except Exception as e:
            logger.warning(f"⚠️ Error verificando driver NPU: {e}")
        
        # ✅ 3. VERIFICAR MODELO YOLO11n.rknn
        rknn_path = "/app/models/yolo11n.rknn"
        if os.path.exists(rknn_path):
            logger.info(f"✅ Modelo RKNN encontrado: {rknn_path}")
            return True
        
        logger.warning("❌ yolo11n.rknn no encontrado en /app/models/")
        return False
    
    def _init_yolo11n_rknn(self) -> bool:
        """Inicializar YOLO11n RKNN para RK3588 - ESPECÍFICO PARA RADXA 5T"""
        try:
            logger.info("🚀 Inicializando YOLO11n RKNN específico para RK3588...")
            
            # ✅ 1. RUTA ESPECÍFICA DEL MODELO
            rknn_model_path = "/app/models/yolo11n.rknn"
            if not os.path.exists(rknn_model_path):
                logger.error(f"❌ No se encontró: {rknn_model_path}")
                return False
            
            # ✅ 2. INICIALIZAR RKNNLite ESPECÍFICO PARA RK3588
            self.rknn_model = RKNNLite()
            
            logger.info(f"📁 Cargando modelo RKNN: {rknn_model_path}")
            ret = self.rknn_model.load_rknn(rknn_model_path)
            if ret != 0:
                logger.error(f"❌ Error cargando modelo RKNN: {ret}")
                return False
            
            # ✅ 3. INICIALIZAR RUNTIME ESPECÍFICO PARA RK3588
            logger.info("🔧 Inicializando runtime para RK3588...")
            # ✅ CRÍTICO: Para RK3588, init_runtime() SIN parámetros
            ret = self.rknn_model.init_runtime()
            if ret != 0:
                logger.error(f"❌ Error inicializando runtime RKNN: {ret}")
                self.rknn_model.release()
                self.rknn_model = None
                return False
            
            # ✅ 4. TEST DE FUNCIONAMIENTO CON RESOLUCIÓN 640x640
            logger.info("🧪 Probando inferencia RKNN con 640x640...")
            test_input = np.random.rand(1, 3, 640, 640).astype(np.float32)
            
            start_time = time.time()
            outputs = self.rknn_model.inference(inputs=[test_input])
            inference_time = (time.time() - start_time) * 1000
            
            if outputs is None or len(outputs) == 0:
                logger.error("❌ Test de inferencia RKNN falló")
                self.rknn_model.release()
                self.rknn_model = None
                return False
            
            # ✅ 5. VERIFICAR FORMATO DE SALIDA YOLO11n
            if len(outputs) > 0:
                output_shape = outputs[0].shape
                logger.info(f"📊 Output shape: {output_shape}")
                
                # ✅ YOLO11n RKNN debe tener formato [1, 84, 8400] o similar
                if len(output_shape) == 3 and output_shape[1] >= 84:
                    logger.info("✅ Formato de salida YOLO11n correcto")
                else:
                    logger.warning(f"⚠️ Formato de salida inesperado: {output_shape}")
            
            logger.info(f"✅ YOLO11n RKNN RK3588 inicializado exitosamente!")
            logger.info(f"⚡ Tiempo de inferencia test: {inference_time:.1f}ms (esperado: ~99.5ms)")
            
            self.use_rknn = True
            self.model_type = "yolo11n_rknn_rk3588"
            
            return True
            
        except Exception as e:
            logger.error(f"❌ Error crítico inicializando YOLO11n RKNN: {e}")
            if self.rknn_model:
                try:
                    self.rknn_model.release()
                except:
                    pass
                self.rknn_model = None
            return False
    
    def _init_yolo11n_pytorch(self) -> bool:
        """Inicializar YOLO11n con PyTorch (fallback)"""
        try:
            logger.info("🔄 Inicializando YOLO11n PyTorch fallback...")
            
            # ✅ BUSCAR MODELO PT
            model_paths = [
                "/app/models/yolo11n.pt",
                "yolo11n.pt"  # Descarga automática
            ]
            
            for model_path in model_paths:
                try:
                    self.model = YOLO(model_path, task="detect")
                    self.model_type = "yolo11n_pytorch"
                    logger.info(f"✅ YOLO11n PyTorch inicializado: {model_path}")
                    return True
                except Exception as e:
                    logger.warning(f"⚠️ Error con {model_path}: {e}")
                    continue
            
            return False
            
        except Exception as e:
            logger.error(f"❌ Error con YOLO11n PyTorch: {e}")
            return False
    
    def detect(self, frame):
        """Detectar vehículos - MÉTODO PRINCIPAL CON RESOLUCIÓN FORZADA"""
        try:
            # ✅ FORZAR RESOLUCIÓN 640x640 INMEDIATAMENTE
            if frame.shape[:2] != (640, 640):
                frame = cv2.resize(frame, (640, 640))
            
            if self.use_rknn and self.rknn_model:
                return self._detect_rknn(frame)
            elif self.model:
                return self._detect_pytorch(frame)
            else:
                return []
                
        except Exception as e:
            logger.error(f"❌ Error en detección: {e}")
            return []
    
    def _detect_rknn(self, frame):
        """Detectar con RKNN YOLO11n - REESCRITO COMPLETAMENTE PARA RK3588"""
        try:
            start_time = time.time()
            
            # ✅ 1. PREPROCESAR ESPECÍFICO PARA YOLO11n RKNN
            input_data = self._preprocess_frame_yolo11n_rknn(frame)
            
            # ✅ 2. INFERENCIA RKNN RK3588
            outputs = self.rknn_model.inference(inputs=[input_data])
            
            if outputs is None or len(outputs) == 0:
                logger.warning("⚠️ RKNN inference devolvió outputs vacíos")
                return []
            
            # ✅ 3. POSTPROCESAR ESPECÍFICO PARA YOLO11n RKNN
            detections = self._postprocess_yolo11n_rknn_rk3588(outputs, frame.shape)
            
            inference_time = (time.time() - start_time) * 1000
            
            # ✅ LOG DE RENDIMIENTO
            if len(detections) > 0:
                logger.debug(f"🔍 RKNN RK3588: {inference_time:.1f}ms, {len(detections)} detecciones")
            
            # ✅ VERIFICAR RENDIMIENTO (debería ser ~99.5ms)
            if inference_time > 150:
                logger.warning(f"⚠️ Inferencia lenta: {inference_time:.1f}ms (esperado: ~99.5ms)")
            
            return detections
            
        except Exception as e:
            logger.error(f"❌ Error en detección RKNN RK3588: {e}")
            return []
    
    def _preprocess_frame_yolo11n_rknn(self, frame):
        """Preprocesar frame para YOLO11n RKNN RK3588 - ESPECÍFICO"""
        
        # ✅ 1. VERIFICAR RESOLUCIÓN 640x640
        if frame.shape[:2] != (640, 640):
            frame = cv2.resize(frame, (640, 640))
        
        # ✅ 2. BGR A RGB
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # ✅ 3. NORMALIZAR ESPECÍFICO PARA YOLO11n
        input_data = rgb.astype(np.float32) / 255.0
        
        # ✅ 4. TRANSPONER: HWC → CHW → NCHW
        input_data = np.transpose(input_data, (2, 0, 1))  # HWC to CHW
        input_data = np.expand_dims(input_data, axis=0)   # Add batch dimension
        
        # ✅ 5. VERIFICAR SHAPE FINAL
        expected_shape = (1, 3, 640, 640)
        if input_data.shape != expected_shape:
            logger.error(f"❌ Shape incorrecto: {input_data.shape}, esperado: {expected_shape}")
            raise ValueError(f"Shape incorrecto para YOLO11n RKNN")
        
        return input_data
    
    def _postprocess_yolo11n_rknn_rk3588(self, outputs, original_frame_shape):
        """Postprocesar YOLO11n RKNN RK3588 - REESCRITO COMPLETAMENTE"""
        
        detections = []
        
        try:
            if len(outputs) == 0:
                return detections
            
            # ✅ YOLO11n RKNN OUTPUT: [1, 84, 8400] o similar
            # 84 = 4 (bbox coordinates) + 80 (COCO classes)
            output = outputs[0]
            
            logger.debug(f"📊 YOLO11n RKNN Output shape: {output.shape}")
            
            # ✅ MANEJAR DIFERENTES FORMATOS DE BATCH
            if len(output.shape) == 3 and output.shape[0] == 1:
                # Formato [1, 84, 8400] → [84, 8400]
                output = output[0]
            
            if len(output.shape) != 2:
                logger.error(f"❌ Output shape inesperado: {output.shape}")
                return detections
            
            num_features, num_detections = output.shape
            logger.debug(f"📊 Features: {num_features}, Detecciones: {num_detections}")
            
            # ✅ VERIFICAR QUE TENGAMOS SUFICIENTES FEATURES PARA YOLO11n
            if num_features < 84:  # 4 bbox + 80 clases mínimo
                logger.error(f"❌ Features insuficientes: {num_features} < 84")
                return detections
            
            # ✅ EXTRAER BBOX Y CLASES
            boxes = output[:4, :]      # Primeras 4 filas: x_center, y_center, width, height
            scores = output[4:84, :]   # Siguientes 80 filas: confianzas de clases COCO
            
            # ✅ CLASES DE VEHÍCULOS COCO
            vehicle_classes = {
                2: 'car', 
                3: 'motorcycle', 
                5: 'bus', 
                7: 'truck'
            }
            
            # ✅ PROCESAR CADA DETECCIÓN
            for i in range(num_detections):
                # Obtener confianza máxima entre clases de vehículos
                max_score = 0
                max_class_id = -1
                
                for class_id in vehicle_classes.keys():
                    if class_id < scores.shape[0]:  # Verificar índice válido
                        score = scores[class_id, i]
                        if score > max_score:
                            max_score = score
                            max_class_id = class_id
                
                # ✅ FILTRAR POR CONFIANZA
                if max_score < self.confidence_threshold:
                    continue
                
                # ✅ EXTRAER COORDENADAS (formato YOLO: center format)
                center_x = float(boxes[0, i])
                center_y = float(boxes[1, i])
                width = float(boxes[2, i])
                height = float(boxes[3, i])
                
                # ✅ CONVERTIR A COORDENADAS REALES (640x640 → original)
                orig_h, orig_w = original_frame_shape[:2]
                
                # Las coordenadas ya están normalizadas para 640x640
                center_x_real = center_x * orig_w
                center_y_real = center_y * orig_h
                width_real = width * orig_w
                height_real = height * orig_h
                
                # ✅ CONVERTIR A FORMATO BBOX [x, y, width, height]
                x = int(center_x_real - width_real / 2)
                y = int(center_y_real - height_real / 2)
                w = int(width_real)
                h = int(height_real)
                
                # ✅ VALIDAR COORDENADAS
                x = max(0, min(x, orig_w - 1))
                y = max(0, min(y, orig_h - 1))
                w = max(1, min(w, orig_w - x))
                h = max(1, min(h, orig_h - y))
                
                # ✅ AGREGAR DETECCIÓN VÁLIDA
                detections.append({
                    'bbox': [x, y, w, h],
                    'confidence': float(max_score),
                    'class_id': int(max_class_id),
                    'class_name': vehicle_classes[max_class_id]
                })
            
            # ✅ APLICAR NMS
            detections = self._apply_nms_yolo11n(detections)
            
            logger.debug(f"🎯 YOLO11n RKNN RK3588 - Detecciones finales: {len(detections)}")
            
            return detections
            
        except Exception as e:
            logger.error(f"❌ Error en postprocesamiento YOLO11n RKNN RK3588: {e}")
            logger.error(f"   Output info: {[o.shape if hasattr(o, 'shape') else type(o) for o in outputs]}")
            return []
    
    def _apply_nms_yolo11n(self, detections, iou_threshold=0.45):
        """Aplicar NMS específico para YOLO11n"""
        if len(detections) == 0:
            return detections
        
        try:
            boxes = []
            confidences = []
            class_ids = []
            
            for det in detections:
                boxes.append(det['bbox'])
                confidences.append(det['confidence'])
                class_ids.append(det['class_id'])
            
            # ✅ NMS usando OpenCV
            indices = cv2.dnn.NMSBoxes(
                boxes, confidences, 
                self.confidence_threshold, 
                iou_threshold
            )
            
            if len(indices) > 0:
                indices = indices.flatten()
                return [detections[i] for i in indices]
            else:
                return []
                
        except Exception as e:
            logger.error(f"❌ Error en NMS YOLO11n: {e}")
            return detections
    
    def _detect_pytorch(self, frame):
        """Detectar con PyTorch YOLO11n - RESOLUCIÓN FORZADA"""
        try:
            # ✅ VERIFICAR RESOLUCIÓN 640x640
            if frame.shape[:2] != (640, 640):
                frame = cv2.resize(frame, (640, 640))
            
            results = self.model(
                frame, 
                conf=self.confidence_threshold,
                iou=0.45,
                verbose=False,
                classes=[2, 3, 5, 7],  # Solo vehículos
                imgsz=640  # ✅ FORZAR 640x640
            )
            
            detections = []
            for result in results:
                if result.boxes is not None:
                    for box in result.boxes:
                        xyxy = box.xyxy[0].cpu().numpy()
                        confidence = float(box.conf[0])
                        class_id = int(box.cls[0])
                        
                        x1, y1, x2, y2 = xyxy
                        width = x2 - x1
                        height = y2 - y1
                        
                        if confidence >= self.confidence_threshold:
                            detections.append({
                                'bbox': [int(x1), int(y1), int(width), int(height)],
                                'confidence': confidence,
                                'class_id': class_id,
                                'class_name': self._get_class_name(class_id)
                            })
            
            return detections
            
        except Exception as e:
            logger.error(f"❌ Error en detección PyTorch: {e}")
            return []
    
    def _get_class_name(self, class_id: int) -> str:
        """Obtener nombre de clase COCO"""
        class_names = {2: 'car', 3: 'motorcycle', 5: 'bus', 7: 'truck'}
        return class_names.get(class_id, 'vehicle')
    
    def enhance_night_vision(self, frame: np.ndarray) -> np.ndarray:
        """Mejorar visión nocturna - MANTENIENDO 640x640"""
        try:
            # ✅ ASEGURAR 640x640
            if frame.shape[:2] != (640, 640):
                frame = cv2.resize(frame, (640, 640))
            
            lab = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)
            l, a, b = cv2.split(lab)
            
            clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
            l = clahe.apply(l)
            
            enhanced = cv2.merge([l, a, b])
            enhanced = cv2.cvtColor(enhanced, cv2.COLOR_LAB2BGR)
            
            gamma = 1.2
            enhanced = np.power(enhanced / 255.0, gamma)
            enhanced = (enhanced * 255).astype(np.uint8)
            
            return enhanced
            
        except Exception as e:
            logger.error(f"Error en mejora nocturna: {e}")
            return frame
    
    def get_model_info(self) -> dict:
        """Información del modelo - ACTUALIZADA PARA RK3588"""
        return {
            "model_type": self.model_type,
            "use_rknn": self.use_rknn,
            "rknn_available": RKNN_AVAILABLE,
            "ultralytics_available": ULTRALYTICS_AVAILABLE,
            "confidence_threshold": self.confidence_threshold,
            "input_size": self.input_size,
            "target_platform": self.target_platform,
            "forced_resolution": "640x640",
            "vehicle_classes": ["car", "motorcycle", "bus", "truck"],
            "expected_performance_rknn": "99.5ms per image (~10 FPS) para YOLO11n",
            "expected_performance_pytorch": "200-300ms per image (~3-5 FPS)",
            "rknn_version": "rknnlite2 para RK3588",
            "hardware_acceleration": "NPU RK3588 Radxa Rock 5T",
            "model_file": "/app/models/yolo11n.rknn",
            "coco_classes_used": [2, 3, 5, 7],
            "nms_threshold": 0.45
        }
    
    def __del__(self):
        """Limpiar recursos RK3588"""
        if self.rknn_model:
            try:
                self.rknn_model.release()
                logger.info("🧹 YOLO11n RKNN RK3588 model released")
            except:
                pass