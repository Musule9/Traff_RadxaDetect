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
        """Método principal optimizado"""
        try:
            if frame.shape[:2] != (640, 640):
                frame = cv2.resize(frame, (640, 640))
            
            if self.use_rknn and self.rknn_model:
                return self._detect_rknn_optimized(frame, skip_nms=False)
            elif self.torch_model:
                return self._detect_ultralytics(frame)
            else:
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
    
    def detect_optimized(self, frame, skip_nms=False):
        """Detección optimizada para 30 FPS"""
        try:
            # Verificar resolución
            if frame.shape[:2] != (640, 640):
                frame = cv2.resize(frame, (640, 640))
            
            if self.use_rknn and self.rknn_model:
                return self._detect_rknn_optimized(frame, skip_nms)
            elif self.torch_model:
                return self._detect_ultralytics_optimized(frame)
            else:
                return []
                    
        except Exception as e:
            logger.error(f"❌ Error en detección optimizada: {e}")
            return []
        
    def _detect_rknn_optimized(self, frame, skip_nms=False):
        """Detección RKNN optimizada"""
        try:
            # ✅ PREPROCESSING OPTIMIZADO
            input_data = self._preprocess_optimized(frame)
            
            # ✅ INFERENCIA RKNN
            outputs = self.rknn_model.inference(inputs=[input_data])
            
            if outputs is None or len(outputs) == 0:
                return []
            
            # ✅ POSTPROCESAMIENTO OPTIMIZADO
            detections = self._postprocess_rknn_optimized(outputs, frame.shape, skip_nms)
            
            return detections
            
        except Exception as e:
            logger.error(f"❌ Error RKNN optimizado: {e}")
            return []
        
    def _preprocess_optimized(self, frame):
        """Preprocessing optimizado"""
        # BGR a RGB (más rápido que cvtColor en algunos casos)
        rgb = frame[:, :, ::-1]
        
        # Normalización optimizada
        input_data = rgb.astype(np.float32, copy=False) * (1.0 / 255.0)
        
        # Transpose optimizado
        input_data = np.transpose(input_data, (2, 0, 1))
        input_data = np.expand_dims(input_data, axis=0)
        
        return input_data

    def _apply_nms_fast(self, detections, iou_threshold=0.5):
        """NMS ultra-rápido"""
        if len(detections) <= 1:
            return detections
        
        try:
            # Ordenar por confianza descendente
            detections.sort(key=lambda x: x['confidence'], reverse=True)
            
            # NMS simple pero rápido
            keep = []
            boxes = np.array([det['bbox'] for det in detections])
            
            for i, det in enumerate(detections):
                if i in keep:
                    continue
                    
                keep.append(i)
                
                # Calcular IoU con detecciones restantes
                current_box = boxes[i]
                for j in range(i + 1, len(detections)):
                    if j in keep:
                        continue
                        
                    other_box = boxes[j]
                    iou = self._calculate_iou_fast(current_box, other_box)
                    
                    if iou > iou_threshold:
                        keep.append(j)  # Marcar para eliminar
            
            return [detections[i] for i in range(len(detections)) if i not in keep[1:]]
            
        except Exception as e:
            logger.error(f"❌ Error NMS rápido: {e}")
            return detections

    def _calculate_iou_fast(self, box1, box2):
        """Cálculo IoU optimizado"""
        x1, y1, w1, h1 = box1
        x2, y2, w2, h2 = box2
        
        # Intersección
        xi1 = max(x1, x2)
        yi1 = max(y1, y2)
        xi2 = min(x1 + w1, x2 + w2)
        yi2 = min(y1 + h1, y2 + h2)
        
        if xi2 <= xi1 or yi2 <= yi1:
            return 0.0
        
        inter_area = (xi2 - xi1) * (yi2 - yi1)
        union_area = w1 * h1 + w2 * h2 - inter_area
        
        return inter_area / union_area if union_area > 0 else 0.0

    
    def _postprocess_rknn_optimized(self, outputs, original_shape, skip_nms=False):
        """Postprocesamiento RKNN ultra-optimizado"""
        detections = []
        
        try:
            output = outputs[0]
            
            if len(output.shape) == 3 and output.shape[0] == 1:
                output = output[0]
            
            if len(output.shape) != 2 or output.shape[0] < 84:
                return detections
            
            # ✅ OPTIMIZACIÓN EXTREMA: THRESHOLD MÁS ALTO PARA REDUCIR CÁLCULOS
            high_conf_threshold = max(self.confidence_threshold, 0.3)
            
            # ✅ PROCESAMIENTO ULTRA-RÁPIDO
            boxes = output[:4, :]
            scores = output[4:84, :]
            
            # Solo clases de vehículos
            vehicle_scores = scores[[2, 3, 5, 7], :]  # car, motorcycle, bus, truck
            max_scores = np.max(vehicle_scores, axis=0)
            
            # Filtro rápido por confianza
            valid_mask = max_scores >= high_conf_threshold
            valid_indices = np.where(valid_mask)[0]
            
            if len(valid_indices) == 0:
                return detections
            
            # Limitar número de detecciones para mantener FPS
            if len(valid_indices) > 20:
                # Tomar las 20 con mayor confianza
                top_indices = np.argsort(max_scores[valid_indices])[-20:]
                valid_indices = valid_indices[top_indices]
            
            # Conversión rápida
            orig_h, orig_w = original_shape[:2]
            vehicle_classes = {0: 'car', 1: 'motorcycle', 2: 'bus', 3: 'truck'}
            vehicle_ids = [2, 3, 5, 7]
            
            for i in valid_indices:
                class_scores = vehicle_scores[:, i]
                max_class_idx = np.argmax(class_scores)
                confidence = class_scores[max_class_idx]
                
                if confidence < self.confidence_threshold:
                    continue
                
                # Coordenadas
                center_x = boxes[0, i] * orig_w
                center_y = boxes[1, i] * orig_h
                width = boxes[2, i] * orig_w
                height = boxes[3, i] * orig_h
                
                x = int(max(0, center_x - width / 2))
                y = int(max(0, center_y - height / 2))
                w = int(min(width, orig_w - x))
                h = int(min(height, orig_h - y))
                
                if w > 0 and h > 0:
                    detections.append({
                        'bbox': [x, y, w, h],
                        'confidence': float(confidence),
                        'class_id': vehicle_ids[max_class_idx],
                        'class_name': vehicle_classes[max_class_idx]
                    })
            
            # NMS ligero solo si hay muchas detecciones
            if not skip_nms and len(detections) > 8:
                detections = self._apply_nms_fast(detections)
            
            return detections
            
        except Exception as e:
            logger.error(f"❌ Error postprocesamiento ultra-optimizado: {e}")
            return []

    def _postprocess_rknn(self, outputs, original_shape):
        """Postprocesar RKNN - OPTIMIZADO para 30 FPS"""
        detections = []
        
        try:
            if len(outputs) == 0:
                return detections
            
            output = outputs[0]
            
            # Manejar batch dimension
            if len(output.shape) == 3 and output.shape[0] == 1:
                output = output[0]
            
            if len(output.shape) != 2:
                return detections
            
            num_features, num_detections = output.shape
            
            if num_features < 84:
                return detections
            
            # ✅ OPTIMIZACIÓN: PROCESAMIENTO VECTORIZADO
            boxes = output[:4, :]  # x_center, y_center, width, height
            scores = output[4:84, :]  # 80 clases COCO
            
            # ✅ OPTIMIZACIÓN: CLASES DE VEHÍCULOS CON ÍNDICES OPTIMIZADOS
            vehicle_classes = {2: 'car', 3: 'motorcycle', 5: 'bus', 7: 'truck'}
            vehicle_indices = list(vehicle_classes.keys())
            
            # ✅ OPTIMIZACIÓN: PROCESAMIENTO BATCH
            # Extraer solo scores de vehículos
            vehicle_scores = scores[vehicle_indices, :]  # Shape: [4, num_detections]
            
            # Encontrar clase con mayor confianza para cada detección
            max_scores = np.max(vehicle_scores, axis=0)
            max_class_indices = np.argmax(vehicle_scores, axis=0)
            
            # ✅ OPTIMIZACIÓN: FILTRAR POR CONFIANZA INMEDIATAMENTE
            valid_mask = max_scores >= self.confidence_threshold
            valid_indices = np.where(valid_mask)[0]
            
            if len(valid_indices) == 0:
                return detections
            
            # ✅ OPTIMIZACIÓN: PROCESAMIENTO SOLO DE DETECCIONES VÁLIDAS
            valid_boxes = boxes[:, valid_indices]
            valid_scores = max_scores[valid_indices]
            valid_classes = max_class_indices[valid_indices]
            
            # ✅ OPTIMIZACIÓN: CONVERSIÓN VECTORIZADA A PÍXELES
            orig_h, orig_w = original_shape[:2]
            
            center_x = valid_boxes[0, :] * orig_w
            center_y = valid_boxes[1, :] * orig_h
            width = valid_boxes[2, :] * orig_w
            height = valid_boxes[3, :] * orig_h
            
            # Convertir a bbox [x, y, w, h]
            x = (center_x - width / 2).astype(np.int32)
            y = (center_y - height / 2).astype(np.int32)
            w = width.astype(np.int32)
            h = height.astype(np.int32)
            
            # ✅ OPTIMIZACIÓN: CLIP VECTORIZADO
            x = np.clip(x, 0, orig_w - 1)
            y = np.clip(y, 0, orig_h - 1)
            w = np.clip(w, 1, orig_w - x)
            h = np.clip(h, 1, orig_h - y)
            
            # ✅ OPTIMIZACIÓN: CREAR DETECCIONES EN BATCH
            for i in range(len(valid_indices)):
                class_idx = vehicle_indices[valid_classes[i]]
                detections.append({
                    'bbox': [x[i], y[i], w[i], h[i]],
                    'confidence': float(valid_scores[i]),
                    'class_id': int(class_idx),
                    'class_name': vehicle_classes[class_idx]
                })
            
            # ✅ OPTIMIZACIÓN: NMS SOLO SI HAY MUCHAS DETECCIONES
            if len(detections) > 10:
                detections = self._apply_nms_optimized(detections)
            
            return detections
            
        except Exception as e:
            logger.error(f"❌ Error postprocesando RKNN optimizado: {e}")
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