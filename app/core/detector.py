# Actualizar app/core/detector.py

import os
import subprocess
from pathlib import Path
from loguru import logger

try:
    from ultralytics import YOLO
    ULTRALYTICS_AVAILABLE = True
except ImportError:
    ULTRALYTICS_AVAILABLE = False
    logger.warning("Ultralytics no disponible")

try:
    from rknnlite.api import RKNNLite
    RKNN_AVAILABLE = True
except ImportError:
    RKNN_AVAILABLE = False
    logger.warning("RKNN no disponible, usando CPU como fallback")

class VehicleDetector:
    """Detector de vehículos optimizado para YOLO11n + RKNN en Radxa Rock 5T"""
    
    def __init__(self, model_path: str = None, confidence_threshold: float = 0.5):
        self.confidence_threshold = confidence_threshold
        self.model = None
        self.use_rknn = False
        self.model_type = "none"
        
        # Asegurar directorio de modelos
        os.makedirs("/app/models", exist_ok=True)
        
        # Inicializar modelo
        self._initialize_model(model_path)
    
    def _initialize_model(self, model_path: str = None):
        """Inicializar modelo con prioridad: RKNN > ONNX > PyTorch"""
        
        # 1. Intentar usar YOLO11n con RKNN (método moderno)
        if ULTRALYTICS_AVAILABLE and self._check_rknn_support():
            if self._init_yolo11n_rknn():
                return
        
        # 2. Fallback a ONNX si está disponible
        onnx_path = "/app/models/yolo11n.onnx"
        if os.path.exists(onnx_path) and ULTRALYTICS_AVAILABLE:
            if self._init_yolo11n_onnx(onnx_path):
                return
        
        # 3. Fallback a PyTorch
        if ULTRALYTICS_AVAILABLE:
            if self._init_yolo11n_pytorch():
                return
        
        # 4. Fallback final: OpenCV con modelo básico
        logger.warning("⚠️ Todos los métodos fallaron, usando detección básica")
        self.model_type = "basic"
    
    def _check_rknn_support(self) -> bool:
        """Verificar si RKNN está disponible y funcional"""
        if not RKNN_AVAILABLE:
            return False
        
        try:
            # Test básico de RKNN
            test_rknn = RKNNLite()
            # No cargar modelo, solo verificar que la librería funciona
            logger.info("✅ RKNN runtime disponible")
            return True
        except Exception as e:
            logger.error(f"❌ Error verificando RKNN: {e}")
            return False
    
    def _init_yolo11n_rknn(self) -> bool:
        """Inicializar YOLO11n con RKNN usando Ultralytics nativo"""
        try:
            logger.info("🚀 Inicializando YOLO11n con soporte RKNN nativo...")
            
            # Verificar si ya existe modelo RKNN
            rknn_model_dir = "/app/models/yolo11n-rk3588.rknn"
            
            if not os.path.exists(rknn_model_dir):
                logger.info("📥 Descargando y convirtiendo YOLO11n a RKNN...")
                
                # Crear modelo y exportar a RKNN
                model = YOLO("yolo11n.pt")  # Descarga automáticamente
                
                # Exportar a RKNN para RK3588
                export_path = model.export(
                    format="rknn", 
                    name="rk3588",
                    imgsz=640,
                    half=False,
                    int8=True  # Cuantización INT8 para mejor rendimiento
                )
                
                # Mover a ubicación estándar
                if os.path.exists("yolo11n_rknn_model"):
                    import shutil
                    shutil.move("yolo11n_rknn_model", rknn_model_dir)
                
                logger.info(f"✅ Modelo RKNN exportado: {rknn_model_dir}")
            
            # Cargar modelo RKNN
            self.model = YOLO(rknn_model_dir)
            self.use_rknn = True
            self.model_type = "yolo11n_rknn"
            
            logger.info("✅ YOLO11n + RKNN inicializado correctamente")
            return True
            
        except Exception as e:
            logger.error(f"❌ Error inicializando YOLO11n + RKNN: {e}")
            return False
    
    def _init_yolo11n_onnx(self, onnx_path: str) -> bool:
        """Inicializar YOLO11n con ONNX"""
        try:
            logger.info("🔄 Inicializando YOLO11n con ONNX...")
            self.model = YOLO(onnx_path)
            self.model_type = "yolo11n_onnx"
            logger.info("✅ YOLO11n ONNX inicializado")
            return True
        except Exception as e:
            logger.error(f"❌ Error con YOLO11n ONNX: {e}")
            return False
    
    def _init_yolo11n_pytorch(self) -> bool:
        """Inicializar YOLO11n con PyTorch"""
        try:
            logger.info("🔄 Inicializando YOLO11n con PyTorch...")
            self.model = YOLO("yolo11n.pt")  # Descarga automáticamente
            self.model_type = "yolo11n_pytorch"
            logger.info("✅ YOLO11n PyTorch inicializado")
            return True
        except Exception as e:
            logger.error(f"❌ Error con YOLO11n PyTorch: {e}")
            return False
    
    def detect(self, frame):
        """Detectar vehículos en frame"""
        try:
            if self.model is None:
                return []
            
            if self.model_type == "basic":
                return self._basic_detection(frame)
            
            # Usar YOLO11n (cualquier backend)
            results = self.model(frame, conf=self.confidence_threshold, verbose=False)
            
            detections = []
            for result in results:
                if result.boxes is not None:
                    for box in result.boxes:
                        # Filtrar solo vehículos (clases COCO: 2=car, 3=motorcycle, 5=bus, 7=truck)
                        class_id = int(box.cls[0])
                        if class_id in [2, 3, 5, 7]:
                            x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                            confidence = float(box.conf[0])
                            
                            detections.append({
                                'bbox': [int(x1), int(y1), int(x2-x1), int(y2-y1)],
                                'confidence': confidence,
                                'class_id': class_id,
                                'class_name': self._get_class_name(class_id)
                            })
            
            return detections
            
        except Exception as e:
            logger.error(f"❌ Error en detección: {e}")
            return []
    
    def _basic_detection(self, frame):
        """Detección básica de fallback"""
        # Implementación básica que siempre devuelve una lista vacía
        # En producción, podrías usar OpenCV con un modelo básico
        return []
    
    def _get_class_name(self, class_id: int) -> str:
        """Obtener nombre de clase de vehículo"""
        class_names = {
            2: 'car',
            3: 'motorcycle', 
            5: 'bus',
            7: 'truck'
        }
        return class_names.get(class_id, 'vehicle')
    
    def get_model_info(self) -> dict:
        """Obtener información del modelo actual"""
        return {
            "model_type": self.model_type,
            "use_rknn": self.use_rknn,
            "rknn_available": RKNN_AVAILABLE,
            "ultralytics_available": ULTRALYTICS_AVAILABLE,
            "confidence_threshold": self.confidence_threshold
        }