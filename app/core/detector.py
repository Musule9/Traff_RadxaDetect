import cv2
import numpy as np
from typing import List, Tuple, Optional
import time
from loguru import logger

try:
    from rknnlite.api import RKNNLite
    RKNN_AVAILABLE = True
except ImportError:
    RKNN_AVAILABLE = False
    logger.warning("RKNN no disponible, usando OpenCV como fallback")

class VehicleDetector:
    """Detector de vehículos optimizado para Radxa Rock 5T"""
    
    def __init__(self, model_path: str, confidence_threshold: float = 0.5):
        self.model_path = model_path
        self.confidence_threshold = confidence_threshold
        self.input_size = (640, 640)
        self.classes = self._get_vehicle_classes()
        
        # Inicializar detector
        if RKNN_AVAILABLE and model_path.endswith('.rknn'):
            self._init_rknn()
        else:
            self._init_opencv()
    
    def _get_vehicle_classes(self) -> List[str]:
        """Clases de vehículos de COCO dataset"""
        return [
            'car', 'motorcycle', 'bus', 'truck'
        ]
    
    def _init_rknn(self):
        """Inicializar RKNN para NPU de Radxa"""
        try:
            self.rknn = RKNNLite()
            
            # Cargar modelo
            ret = self.rknn.load_rknn(self.model_path)
            if ret != 0:
                raise Exception(f"Error cargando modelo RKNN: {ret}")
            
            # Inicializar runtime
            ret = self.rknn.init_runtime()
            if ret != 0:
                raise Exception(f"Error inicializando RKNN runtime: {ret}")
            
            self.use_rknn = True
            logger.info("RKNN inicializado correctamente")
            
        except Exception as e:
            logger.error(f"Error inicializando RKNN: {e}")
            self._init_opencv()
    
    def _init_opencv(self):
        """Inicializar OpenCV como fallback"""
        try:
            self.net = cv2.dnn.readNetFromONNX(self.model_path.replace('.rknn', '.onnx'))
            self.use_rknn = False
            logger.info("OpenCV DNN inicializado correctamente")
        except Exception as e:
            logger.error(f"Error inicializando OpenCV: {e}")
            raise
    
    def preprocess_frame(self, frame: np.ndarray) -> np.ndarray:
        """Preprocesar frame para detección"""
        # Redimensionar manteniendo aspect ratio
        h, w = frame.shape[:2]
        scale = min(self.input_size[0] / w, self.input_size[1] / h)
        new_w, new_h = int(w * scale), int(h * scale)
        
        # Redimensionar
        resized = cv2.resize(frame, (new_w, new_h))
        
        # Crear imagen con padding
        padded = np.full((self.input_size[1], self.input_size[0], 3), 114, dtype=np.uint8)
        padded[:new_h, :new_w] = resized
        
        # Normalizar para modelo
        blob = padded.astype(np.float32) / 255.0
        blob = np.transpose(blob, (2, 0, 1))  # HWC to CHW
        blob = np.expand_dims(blob, axis=0)   # Add batch dimension
        
        return blob, scale
    
    def detect(self, frame: np.ndarray) -> List[dict]:
        """Detectar vehículos en frame"""
        try:
            # Preprocesar
            blob, scale = self.preprocess_frame(frame)
            
            # Inferencia
            if self.use_rknn:
                outputs = self.rknn.inference(inputs=[blob])
            else:
                self.net.setInput(blob)
                outputs = self.net.forward()
            
            # Postprocesar
            detections = self.postprocess(outputs[0], frame.shape, scale)
            
            return detections
            
        except Exception as e:
            logger.error(f"Error en detección: {e}")
            return []
    
    def postprocess(self, output: np.ndarray, original_shape: Tuple, scale: float) -> List[dict]:
        """Postprocesar salidas del modelo"""
        detections = []
        
        # Reshape output si es necesario
        if len(output.shape) == 3:
            output = output[0]
        
        # Filtrar por confianza
        scores = output[:, 4]
        valid_indices = scores > self.confidence_threshold
        
        if not np.any(valid_indices):
            return detections
        
        valid_output = output[valid_indices]
        
        for detection in valid_output:
            x_center, y_center, width, height, confidence = detection[:5]
            class_scores = detection[5:]
            
            # Encontrar clase con mayor score
            class_id = np.argmax(class_scores)
            class_score = class_scores[class_id]
            
            # Solo vehículos (clases 2, 3, 5, 7 en COCO)
            if class_id not in [2, 3, 5, 7]:
                continue
            
            final_confidence = confidence * class_score
            if final_confidence < self.confidence_threshold:
                continue
            
            # Convertir a coordenadas originales
            orig_h, orig_w = original_shape[:2]
            
            x_center = x_center / self.input_size[0] * orig_w
            y_center = y_center / self.input_size[1] * orig_h
            width = width / self.input_size[0] * orig_w
            height = height / self.input_size[1] * orig_h
            
            x1 = int(x_center - width / 2)
            y1 = int(y_center - height / 2)
            x2 = int(x_center + width / 2)
            y2 = int(y_center + height / 2)
            
            # Limitar a dimensiones del frame
            x1 = max(0, min(x1, orig_w))
            y1 = max(0, min(y1, orig_h))
            x2 = max(0, min(x2, orig_w))
            y2 = max(0, min(y2, orig_h))
            
            detection_dict = {
                'bbox': [x1, y1, x2 - x1, y2 - y1],
                'confidence': float(final_confidence),
                'class_id': int(class_id),
                'class_name': self._get_class_name(class_id)
            }
            
            detections.append(detection_dict)
        
        return detections
    
    def _get_class_name(self, class_id: int) -> str:
        """Obtener nombre de clase de vehículo"""
        coco_to_vehicle = {
            2: 'car',
            3: 'motorcycle', 
            5: 'bus',
            7: 'truck'
        }
        return coco_to_vehicle.get(class_id, 'vehicle')
    
    def enhance_night_vision(self, frame: np.ndarray) -> np.ndarray:
        """Mejorar imagen para visión nocturna"""
        # Convertir a LAB
        lab = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)
        
        # Aplicar CLAHE al canal L
        clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
        l = clahe.apply(l)
        
        # Recombinar canales
        enhanced = cv2.merge([l, a, b])
        enhanced = cv2.cvtColor(enhanced, cv2.COLOR_LAB2BGR)
        
        # Ajuste de gamma para mejor visibilidad
        gamma = 1.2
        inv_gamma = 1.0 / gamma
        table = np.array([((i / 255.0) ** inv_gamma) * 255 for i in np.arange(0, 256)]).astype("uint8")
        enhanced = cv2.LUT(enhanced, table)
        
        return enhanced