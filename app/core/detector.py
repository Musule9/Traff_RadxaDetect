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
    """Detector YOLO11n optimizado para Radxa Rock 5T RK3588"""
    
    def __init__(self, model_path: str = None, confidence_threshold: float = 0.25):  # ✅ THRESHOLD MÁS BAJO
        self.confidence_threshold = confidence_threshold
        self.rknn_model = None
        self.torch_model = None
        self.use_rknn = False
        self.model_type = "none"
        
        # ✅ RESOLUCIÓN FIJA 640x640 PARA RKNN
        self.input_size = (640, 640)
        
        # ✅ CLASES DE VEHÍCULOS COCO
        self.vehicle_classes = {2: 'car', 3: 'motorcycle', 5: 'bus', 7: 'truck'}
        self.vehicle_indices = list(self.vehicle_classes.keys())
        
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
            rknn_path = model_path or "/app/models/yolo11n.rknn"
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
            
            # ✅ 3. INICIALIZAR RUNTIME
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
            logger.info(f"📊 Output dtype: {outputs[0].dtype}")
            logger.info(f"📊 Output range: [{outputs[0].min():.2f}, {outputs[0].max():.2f}]")
            
            # ✅ ÉXITO
            self.use_rknn = True
            self.model_type = "yolo11n_rknn_rk3588"
            
            logger.info(f"✅ YOLO11n RKNN inicializado exitosamente!")
            logger.info(f"⚡ Tiempo inferencia: {inference_time:.1f}ms")
            logger.info(f"🎯 Rendimiento esperado: ~10-15 FPS")
            
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
        """Método principal de detección"""
        try:
            # Asegurar 640x640
            if frame.shape[:2] != (640, 640):
                frame = cv2.resize(frame, (640, 640))
            
            if self.use_rknn and self.rknn_model:
                return self._detect_rknn(frame)
            elif self.torch_model:
                return self._detect_ultralytics(frame)
            else:
                return []
                    
        except Exception as e:
            logger.error(f"❌ Error en detección: {e}")
            return []

    def _detect_rknn(self, frame):
        """Detectar con RKNN - CORREGIDO PARA YOLO11"""
        try:
            start_time = time.time()
            
            # ✅ PREPROCESAR
            input_data = self._preprocess_for_rknn(frame)
            
            # ✅ INFERENCIA
            outputs = self.rknn_model.inference(inputs=[input_data])
            
            if outputs is None or len(outputs) == 0:
                logger.warning("⚠️ RKNN outputs vacíos")
                return []
            
            # ✅ POSTPROCESAR - CRÍTICO PARA YOLO11
            detections = self._postprocess_yolo11_rknn(outputs[0], frame.shape)
            
            inference_time = (time.time() - start_time) * 1000
            
            if len(detections) > 0:
                logger.debug(f"🔍 RKNN: {inference_time:.1f}ms, {len(detections)} detecciones")
            
            return detections
            
        except Exception as e:
            logger.error(f"❌ Error RKNN detection: {e}")
            return []
    
    def _preprocess_for_rknn(self, frame):
        """Preprocesar para RKNN - FORMATO CORRECTO"""
        # BGR a RGB
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # Normalizar a [0, 1]
        input_data = rgb.astype(np.float32) / 255.0
        
        # HWC a CHW
        input_data = np.transpose(input_data, (2, 0, 1))
        
        # Agregar batch dimension
        input_data = np.expand_dims(input_data, axis=0)
        
        # Asegurar tipo correcto
        input_data = input_data.astype(np.float32)
        
        return input_data

    def _postprocess_yolo11_rknn(self, output, original_shape):
        """Postprocesar salida YOLO11 RKNN - VERSIÓN CORREGIDA"""
        detections = []
        
        try:
            # ✅ YOLO11 RKNN típicamente retorna shape [1, 84, 8400] o [84, 8400]
            logger.debug(f"Output shape: {output.shape}, dtype: {output.dtype}")
            
            # Manejar diferentes formatos de salida
            if len(output.shape) == 3:
                # [1, 84, 8400] -> [84, 8400]
                output = output[0]
            
            if len(output.shape) != 2:
                logger.error(f"❌ Formato inesperado: {output.shape}")
                return detections
            
            # ✅ VERIFICAR SI NECESITA TRANSPUESTA
            if output.shape[0] == 8400 and output.shape[1] == 84:
                # Transponer si es [8400, 84] -> [84, 8400]
                output = output.T
                logger.debug("📊 Output transpuesto a [84, 8400]")
            
            num_features, num_detections = output.shape
            
            if num_features != 84:
                logger.error(f"❌ Se esperaban 84 features, se obtuvieron {num_features}")
                return detections
            
            # ✅ EXTRAER COMPONENTES
            # YOLO11 format: [cx, cy, w, h, obj_conf, class1, class2, ..., class80]
            boxes = output[:4, :]      # Coordenadas bbox
            scores = output[4:, :]      # Confidence + clases
            
            # ✅ PROCESAR CADA DETECCIÓN
            for i in range(num_detections):
                # Obtener objectness/confidence
                obj_conf = scores[0, i]
                
                # Si la confianza es muy baja, saltar
                if obj_conf < 0.1:  # Pre-filtro muy bajo
                    continue
                
                # Obtener scores de clases
                class_scores = scores[1:, i]  # 80 clases COCO
                
                # Encontrar mejor clase
                class_id = np.argmax(class_scores)
                class_conf = class_scores[class_id]
                
                # Confianza final = objectness * class_confidence
                final_conf = obj_conf * class_conf
                
                # Verificar si es vehículo y cumple threshold
                if class_id in self.vehicle_indices and final_conf >= self.confidence_threshold:
                    # ✅ CONVERTIR COORDENADAS
                    # De formato YOLO (normalized) a píxeles
                    cx = boxes[0, i]
                    cy = boxes[1, i]
                    w = boxes[2, i]
                    h = boxes[3, i]
                    
                    # Escalar a tamaño original
                    orig_h, orig_w = original_shape[:2]
                    cx *= orig_w
                    cy *= orig_h
                    w *= orig_w
                    h *= orig_h
                    
                    # Convertir a formato [x, y, w, h]
                    x = int(max(0, cx - w / 2))
                    y = int(max(0, cy - h / 2))
                    w = int(min(w, orig_w - x))
                    h = int(min(h, orig_h - y))
                    
                    if w > 10 and h > 10:  # Filtrar detecciones muy pequeñas
                        detections.append({
                            'bbox': [x, y, w, h],
                            'confidence': float(final_conf),
                            'class_id': int(class_id),
                            'class_name': self.vehicle_classes.get(class_id, 'vehicle')
                        })
            
            # ✅ APLICAR NMS SI HAY MUCHAS DETECCIONES
            if len(detections) > 0:
                detections = self._apply_nms(detections, iou_threshold=0.45)
                logger.debug(f"✅ Detecciones después de NMS: {len(detections)}")
            
            return detections
            
        except Exception as e:
            logger.error(f"❌ Error en postprocesamiento YOLO11: {e}")
            import traceback
            logger.error(traceback.format_exc())
            return []

    def _apply_nms(self, detections, iou_threshold=0.45):
        """Non-Maximum Suppression mejorado"""
        if len(detections) <= 1:
            return detections
        
        try:
            # Convertir a formato para cv2.dnn.NMSBoxes
            boxes = []
            confidences = []
            class_ids = []
            
            for det in detections:
                boxes.append(det['bbox'])
                confidences.append(det['confidence'])
                class_ids.append(det['class_id'])
            
            # Aplicar NMS
            indices = cv2.dnn.NMSBoxes(
                boxes, 
                confidences, 
                self.confidence_threshold, 
                iou_threshold
            )
            
            # Extraer detecciones filtradas
            if len(indices) > 0:
                if isinstance(indices, tuple):
                    indices = indices[0]
                indices = indices.flatten()
                return [detections[i] for i in indices]
            
            return []
            
        except Exception as e:
            logger.error(f"❌ Error en NMS: {e}")
            # Si NMS falla, usar método simple
            return self._simple_nms(detections, iou_threshold)
    
    def _simple_nms(self, detections, iou_threshold=0.45):
        """NMS simple como fallback"""
        if len(detections) <= 1:
            return detections
        
        # Ordenar por confianza
        detections = sorted(detections, key=lambda x: x['confidence'], reverse=True)
        
        keep = []
        for i, det1 in enumerate(detections):
            should_keep = True
            
            for j in range(len(keep)):
                det2 = keep[j]
                
                # Misma clase y IoU alto = suprimir
                if det1['class_id'] == det2['class_id']:
                    iou = self._calculate_iou(det1['bbox'], det2['bbox'])
                    if iou > iou_threshold:
                        should_keep = False
                        break
            
            if should_keep:
                keep.append(det1)
        
        return keep
    
    def _calculate_iou(self, box1, box2):
        """Calcular Intersection over Union"""
        x1, y1, w1, h1 = box1
        x2, y2, w2, h2 = box2
        
        # Coordenadas de intersección
        xi1 = max(x1, x2)
        yi1 = max(y1, y2)
        xi2 = min(x1 + w1, x2 + w2)
        yi2 = min(y1 + h1, y2 + h2)
        
        if xi2 <= xi1 or yi2 <= yi1:
            return 0.0
        
        # Áreas
        inter_area = (xi2 - xi1) * (yi2 - yi1)
        box1_area = w1 * h1
        box2_area = w2 * h2
        union_area = box1_area + box2_area - inter_area
        
        return inter_area / union_area if union_area > 0 else 0.0

    def _detect_ultralytics(self, frame):
        """Detectar con Ultralytics"""
        try:
            results = self.torch_model(
                frame,
                conf=self.confidence_threshold,
                classes=list(self.vehicle_classes.keys()),
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
                            'class_name': self.vehicle_classes.get(class_id, 'vehicle')
                        })
            
            return detections
            
        except Exception as e:
            logger.error(f"❌ Error Ultralytics: {e}")
            return []
    
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
            "expected_fps": "10-15 FPS" if self.use_rknn else "1-3 FPS"
        }
    
    def enhance_night_vision(self, frame: np.ndarray) -> np.ndarray:
        """Mejorar visión nocturna"""
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
            gamma = 1.5  # Más brillo para nocturno
            enhanced = np.power(enhanced / 255.0, 1/gamma)
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
                logger.info("✅ Modelo RKNN liberado")
            except:
                pass