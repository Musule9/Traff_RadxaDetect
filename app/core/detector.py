# REEMPLAZA COMPLETAMENTE app/core/detector.py

import os
import subprocess
import time
import numpy as np
import cv2
from pathlib import Path
from loguru import logger

# ✅ IMPORTS ESPECÍFICOS PARA RK3588
RKNN_AVAILABLE = False
try:
    from rknnlite.api import RKNNLite
    RKNN_AVAILABLE = True
    logger.info("✅ RKNNLite importado correctamente para RK3588")
except Exception as e:
    logger.error(f"❌ RKNNLite no disponible: {e}")

# ULTRALYTICS como fallback
ULTRALYTICS_AVAILABLE = False
try:
    from ultralytics import YOLO
    ULTRALYTICS_AVAILABLE = True
    logger.info("✅ Ultralytics disponible como fallback")
except Exception as e:
    logger.warning(f"⚠️ Ultralytics no disponible: {e}")

class VehicleDetector:
    """Detector YOLO11n optimizado ESPECÍFICAMENTE para Radxa Rock 5T RK3588"""
    
    def __init__(self, model_path: str = None, confidence_threshold: float = 0.5):
        self.confidence_threshold = confidence_threshold
        self.rknn_model = None
        self.torch_model = None
        self.use_rknn = False
        self.model_type = "none"
        
        # ✅ RESOLUCIÓN FIJA 640x640 PARA RKNN
        self.input_size = (640, 640)
        
        # Inicializar modelo
        self._initialize_model(model_path)
    
    def _initialize_model(self, model_path: str = None):
        """Inicializar modelo - RKNN PRIMERO"""
        
        # ✅ 1. VERIFICAR HARDWARE RK3588
        if not self._check_rk3588():
            logger.warning("⚠️ Hardware no es RK3588, usando fallback")
            self._init_ultralytics_fallback()
            return
        
        # ✅ 2. INTENTAR RKNN PRIMERO
        if RKNN_AVAILABLE:
            rknn_path = "/app/models/yolo11n.rknn"
            if os.path.exists(rknn_path):
                if self._init_rknn_model(rknn_path):
                    return
            else:
                logger.warning(f"❌ Modelo RKNN no encontrado: {rknn_path}")
        
        # ✅ 3. FALLBACK A ULTRALYTICS
        logger.info("🔄 Usando fallback Ultralytics...")
        self._init_ultralytics_fallback()
    
    def _check_rk3588(self) -> bool:
        """Verificar hardware RK3588"""
        try:
            if os.path.exists("/proc/device-tree/model"):
                with open("/proc/device-tree/model", "rb") as f:
                    model = f.read().decode('utf-8', errors='ignore').strip('\x00')
                    if "RK3588" in model or "Radxa" in model:
                        logger.info(f"✅ Hardware RK3588 confirmado: {model}")
                        return True
                    else:
                        logger.warning(f"⚠️ Hardware diferente: {model}")
                        return False
        except Exception as e:
            logger.warning(f"⚠️ No se pudo verificar hardware: {e}")
        return False
    
    def _init_rknn_model(self, model_path: str) -> bool:
        """Inicializar modelo RKNN - ESPECÍFICO RK3588"""
        try:
            logger.info(f"🚀 Inicializando YOLO11n RKNN: {model_path}")
            
            # ✅ 1. CREAR INSTANCIA RKNN
            self.rknn_model = RKNNLite()
            
            # ✅ 2. CARGAR MODELO
            logger.info("📁 Cargando modelo RKNN...")
            ret = self.rknn_model.load_rknn(model_path)
            if ret != 0:
                logger.error(f"❌ Error cargando modelo RKNN: código {ret}")
                self.rknn_model = None
                return False
            
            # ✅ 3. INICIALIZAR RUNTIME SIN PARÁMETROS (CRÍTICO PARA RK3588)
            logger.info("🔧 Inicializando runtime RK3588...")
            ret = self.rknn_model.init_runtime()
            if ret != 0:
                logger.error(f"❌ Error inicializando runtime RKNN: código {ret}")
                self.rknn_model.release()
                self.rknn_model = None
                return False
            
            # ✅ 4. TEST DE INFERENCIA
            logger.info("🧪 Probando inferencia RKNN...")
            test_input = np.random.rand(1, 3, 640, 640).astype(np.float32)
            
            start_time = time.time()
            outputs = self.rknn_model.inference(inputs=[test_input])
            inference_time = (time.time() - start_time) * 1000
            
            if outputs is None or len(outputs) == 0:
                logger.error("❌ Test de inferencia falló - outputs vacíos")
                self.rknn_model.release()
                self.rknn_model = None
                return False
            
            # ✅ 5. VERIFICAR FORMATO YOLO11n
            output_shape = outputs[0].shape
            logger.info(f"📊 Output shape: {output_shape}")
            
            if len(output_shape) >= 2 and output_shape[-2] >= 80:  # YOLO11n: [1, 84, 8400]
                logger.info("✅ Formato YOLO11n correcto")
            else:
                logger.warning(f"⚠️ Formato inesperado: {output_shape}")
            
            # ✅ ÉXITO
            self.use_rknn = True
            self.model_type = "yolo11n_rknn_rk3588"
            
            logger.info(f"✅ YOLO11n RKNN inicializado exitosamente!")
            logger.info(f"⚡ Tiempo inferencia: {inference_time:.1f}ms")
            logger.info(f"🎯 Rendimiento esperado: ~99.5ms por imagen (~10 FPS)")
            
            return True
            
        except Exception as e:
            logger.error(f"❌ ERROR CRÍTICO RKNN: {e}")
            if self.rknn_model:
                try:
                    self.rknn_model.release()
                except:
                    pass
                self.rknn_model = None
            return False
    
    def _init_ultralytics_fallback(self):
        """Inicializar Ultralytics como fallback"""
        if not ULTRALYTICS_AVAILABLE:
            logger.error("❌ Ni RKNN ni Ultralytics disponibles")
            self.model_type = "none"
            return
        
        try:
            logger.info("🔄 Inicializando YOLO11n Ultralytics...")
            self.torch_model = YOLO("yolo11n.pt")
            self.model_type = "yolo11n_ultralytics"
            logger.info("✅ YOLO11n Ultralytics inicializado")
        except Exception as e:
            logger.error(f"❌ Error Ultralytics: {e}")
            self.model_type = "none"
    
    def detect(self, frame):
        """Detectar vehículos - MÉTODO PRINCIPAL"""
        try:
            # ✅ FORZAR 640x640
            if frame.shape[:2] != (640, 640):
                frame = cv2.resize(frame, (640, 640))
            
            if self.use_rknn and self.rknn_model:
                return self._detect_rknn(frame)
            elif self.torch_model:
                return self._detect_ultralytics(frame)
            else:
                logger.warning("⚠️ No hay modelo disponible para detección")
                return []
                
        except Exception as e:
            logger.error(f"❌ Error en detección: {e}")
            return []
    
    def _detect_rknn(self, frame):
        """Detectar con RKNN"""
        try:
            start_time = time.time()
            
            # ✅ PREPROCESAR
            input_data = self._preprocess_for_rknn(frame)
            
            # ✅ INFERENCIA
            outputs = self.rknn_model.inference(inputs=[input_data])
            
            if outputs is None or len(outputs) == 0:
                logger.warning("⚠️ RKNN outputs vacíos")
                return []
            
            # ✅ POSTPROCESAR
            detections = self._postprocess_rknn(outputs, frame.shape)
            
            inference_time = (time.time() - start_time) * 1000
            
            if len(detections) > 0:
                logger.debug(f"🔍 RKNN: {inference_time:.1f}ms, {len(detections)} detecciones")
            
            return detections
            
        except Exception as e:
            logger.error(f"❌ Error RKNN detection: {e}")
            return []
    
    def _preprocess_for_rknn(self, frame):
        """Preprocesar para RKNN"""
        # BGR a RGB
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        # Normalizar
        input_data = rgb.astype(np.float32) / 255.0
        # HWC a CHW
        input_data = np.transpose(input_data, (2, 0, 1))
        # Agregar batch dimension
        input_data = np.expand_dims(input_data, axis=0)
        
        return input_data
    
    def _postprocess_rknn(self, outputs, original_shape):
        """Postprocesar RKNN outputs"""
        detections = []
        
        try:
            output = outputs[0]
            
            # Manejar batch dimension
            if len(output.shape) == 3 and output.shape[0] == 1:
                output = output[0]
            
            if len(output.shape) != 2:
                logger.error(f"❌ Formato output incorrecto: {output.shape}")
                return detections
            
            num_features, num_detections = output.shape
            
            if num_features < 84:
                logger.error(f"❌ Features insuficientes: {num_features}")
                return detections
            
            # Extraer boxes y scores
            boxes = output[:4, :]  # x_center, y_center, width, height
            scores = output[4:84, :]  # 80 clases COCO
            
            # Clases de vehículos
            vehicle_classes = {2: 'car', 3: 'motorcycle', 5: 'bus', 7: 'truck'}
            
            for i in range(num_detections):
                # Encontrar clase con mayor confianza
                max_score = 0
                max_class = -1
                
                for class_id in vehicle_classes.keys():
                    if class_id < scores.shape[0]:
                        score = scores[class_id, i]
                        if score > max_score:
                            max_score = score
                            max_class = class_id
                
                if max_score < self.confidence_threshold:
                    continue
                
                # Extraer coordenadas
                center_x = float(boxes[0, i])
                center_y = float(boxes[1, i])
                width = float(boxes[2, i])
                height = float(boxes[3, i])
                
                # Convertir a píxeles
                orig_h, orig_w = original_shape[:2]
                center_x_real = center_x * orig_w
                center_y_real = center_y * orig_h
                width_real = width * orig_w
                height_real = height * orig_h
                
                # Convertir a bbox [x, y, w, h]
                x = int(center_x_real - width_real / 2)
                y = int(center_y_real - height_real / 2)
                w = int(width_real)
                h = int(height_real)
                
                # Validar coordenadas
                x = max(0, min(x, orig_w - 1))
                y = max(0, min(y, orig_h - 1))
                w = max(1, min(w, orig_w - x))
                h = max(1, min(h, orig_h - y))
                
                detections.append({
                    'bbox': [x, y, w, h],
                    'confidence': float(max_score),
                    'class_id': int(max_class),
                    'class_name': vehicle_classes[max_class]
                })
            
            # Aplicar NMS
            detections = self._apply_nms(detections)
            
            return detections
            
        except Exception as e:
            logger.error(f"❌ Error postprocesando RKNN: {e}")
            return []
    
    def _detect_ultralytics(self, frame):
        """Detectar con Ultralytics"""
        try:
            results = self.torch_model(
                frame,
                conf=self.confidence_threshold,
                classes=[2, 3, 5, 7],  # Solo vehículos
                verbose=False
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
                        
                        detections.append({
                            'bbox': [int(x1), int(y1), int(width), int(height)],
                            'confidence': confidence,
                            'class_id': class_id,
                            'class_name': self._get_class_name(class_id)
                        })
            
            return detections
            
        except Exception as e:
            logger.error(f"❌ Error Ultralytics: {e}")
            return []
    
    def _apply_nms(self, detections, iou_threshold=0.45):
        """Aplicar Non-Maximum Suppression"""
        if len(detections) == 0:
            return detections
        
        try:
            boxes = []
            confidences = []
            
            for det in detections:
                boxes.append(det['bbox'])
                confidences.append(det['confidence'])
            
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
            logger.error(f"❌ Error NMS: {e}")
            return detections
    
    def _get_class_name(self, class_id: int) -> str:
        """Obtener nombre de clase"""
        class_names = {2: 'car', 3: 'motorcycle', 5: 'bus', 7: 'truck'}
        return class_names.get(class_id, 'vehicle')
    
    def get_model_info(self) -> dict:
        """Información del modelo"""
        return {
            "model_type": self.model_type,
            "use_rknn": self.use_rknn,
            "rknn_available": RKNN_AVAILABLE,
            "ultralytics_available": ULTRALYTICS_AVAILABLE,
            "confidence_threshold": self.confidence_threshold,
            "input_size": self.input_size,
            "hardware_optimized": self.use_rknn,
            "expected_fps": "~10 FPS" if self.use_rknn else "~3-5 FPS"
        }
    
    def enhance_night_vision(self, frame: np.ndarray) -> np.ndarray:
        """Mejorar visión nocturna - MÉTODO FALTANTE AGREGADO"""
        try:
            # Asegurar 640x640
            if frame.shape[:2] != (640, 640):
                frame = cv2.resize(frame, (640, 640))
            
            # Convertir a LAB
            lab = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)
            l, a, b = cv2.split(lab)
            
            # Aplicar CLAHE al canal L
            clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
            l = clahe.apply(l)
            
            # Recombinar
            enhanced = cv2.merge([l, a, b])
            enhanced = cv2.cvtColor(enhanced, cv2.COLOR_LAB2BGR)
            
            # Ajuste gamma
            gamma = 1.2
            enhanced = np.power(enhanced / 255.0, gamma)
            enhanced = (enhanced * 255).astype(np.uint8)
            
            return enhanced
            
        except Exception as e:
            logger.error(f"Error en mejora nocturna: {e}")
            return frame
    
    def __del__(self):
        """Limpiar recursos"""
        if self.rknn_model:
            try:
                self.rknn_model.release()
            except:
                pass