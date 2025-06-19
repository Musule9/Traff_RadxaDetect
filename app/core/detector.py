import os
import subprocess
from pathlib import Path
from loguru import logger
import cv2
import numpy as np

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
        self.input_size = (640, 640)
        
        # Asegurar directorio de modelos
        os.makedirs("/app/models", exist_ok=True)
        
        # Inicializar modelo
        self._initialize_model(model_path)
    
    def _initialize_model(self, model_path: str = None):
        """Inicializar modelo con prioridad: YOLO11n RKNN > ONNX > PyTorch"""
        
        # 1. Intentar YOLO11n con RKNN (método moderno Ultralytics 8.3+)
        if ULTRALYTICS_AVAILABLE and self._check_rknn_support():
            if self._init_yolo11n_rknn_native(model_path):
                return
        
        # 2. Fallback a PyTorch/ONNX
        if ULTRALYTICS_AVAILABLE:
            if self._init_yolo11n_pytorch():
                return
        
        # 3. Fallback final: detección básica
        logger.warning("⚠️ Todos los métodos avanzados fallaron, usando detección básica")
        self.model_type = "basic"
    
    def _check_rknn_support(self) -> bool:
        """Verificar si RKNN está disponible y funcional"""
        if not RKNN_AVAILABLE:
            return False
        
        try:
            # Test básico de RKNN runtime
            test_rknn = RKNNLite()
            logger.info("✅ RKNN runtime disponible")
            return True
        except Exception as e:
            logger.error(f"❌ Error verificando RKNN: {e}")
            return False
    
    def _init_yolo11n_rknn_native(self, model_path: str = None) -> bool:
        """Inicializar YOLO11n con RKNN usando Ultralytics nativo (método recomendado)"""
        try:
            logger.info("🚀 Inicializando YOLO11n con soporte RKNN nativo...")
            
            # Buscar modelo RKNN disponible
            possible_paths = [
                model_path,
                "/app/models/yolo11n-rk3588.rknn",
                "/app/models/yolo11n_rknn_model",
                "/models/yolo11n-rk3588.rknn",  # Ruta mencionada por el usuario
                "/models/yolo11n_rknn_model"
            ]
            
            rknn_model_path = None
            for path in possible_paths:
                if path and (os.path.exists(path) or os.path.isdir(path)):
                    rknn_model_path = path
                    logger.info(f"📁 Modelo RKNN encontrado: {path}")
                    break
            
            # Si no existe, crear modelo RKNN
            if not rknn_model_path:
                logger.info("📥 Modelo RKNN no encontrado, creando desde PyTorch...")
                
                # Cargar modelo YOLO11n y exportar a RKNN
                try:
                    model = YOLO("yolo11n.pt")  # Descarga automáticamente si no existe
                    
                    # Exportar a RKNN para RK3588 (método actualizado)
                    logger.info("🔄 Exportando YOLO11n a formato RKNN...")
                    export_path = model.export(
                        format="rknn", 
                        name="rk3588",  # Plataforma específica
                        task="detect",  # Evitar warning automático
                        imgsz=640,
                        half=False,
                        int8=True  # Cuantización INT8 para mejor rendimiento
                    )
                    
                    rknn_model_path = "/app/models/yolo11n_rknn_model"
                    
                    # Mover a ubicación estándar si es necesario
                    if export_path != rknn_model_path and os.path.exists(export_path):
                        import shutil
                        if os.path.isdir(export_path):
                            shutil.move(export_path, rknn_model_path)
                        else:
                            shutil.move(export_path, f"{rknn_model_path}.rknn")
                    
                    logger.info(f"✅ Modelo RKNN exportado: {rknn_model_path}")
                    
                except Exception as e:
                    logger.error(f"❌ Error exportando modelo RKNN: {e}")
                    return False
            
            # Cargar modelo RKNN con Ultralytics
            try:
                self.model = YOLO(rknn_model_path, task="detect")  # Especificar task explícitamente
                self.use_rknn = True
                self.model_type = "yolo11n_rknn_native"
                
                logger.info("✅ YOLO11n + RKNN inicializado correctamente")
                return True
                
            except Exception as e:
                logger.error(f"❌ Error cargando modelo RKNN: {e}")
                return False
            
        except Exception as e:
            logger.error(f"❌ Error en inicialización RKNN nativa: {e}")
            return False
    
    def _init_yolo11n_pytorch(self) -> bool:
        """Inicializar YOLO11n con PyTorch (fallback)"""
        try:
            logger.info("🔄 Inicializando YOLO11n con PyTorch...")
            self.model = YOLO("yolo11n.pt", task="detect")  # Especificar task
            self.model_type = "yolo11n_pytorch"
            logger.info("✅ YOLO11n PyTorch inicializado")
            return True
        except Exception as e:
            logger.error(f"❌ Error con YOLO11n PyTorch: {e}")
            return False
    
    def detect(self, frame):
        """Detectar vehículos en frame con YOLO11n moderno"""
        try:
            if self.model is None:
                return []
            
            if self.model_type == "basic":
                return self._basic_detection(frame)
            
            # Usar YOLO11n moderno (RKNN o PyTorch)
            # Parámetros optimizados según documentación Ultralytics
            results = self.model(
                frame, 
                conf=self.confidence_threshold,
                iou=0.7,  # NMS IoU threshold
                verbose=False,  # Silenciar logs verbosos
                device='cpu',  # Forzar CPU para RKNN
                classes=[2, 3, 5, 7]  # Solo vehículos: car, motorcycle, bus, truck
            )
            
            detections = []
            for result in results:
                if result.boxes is not None:
                    for box in result.boxes:
                        # Extraer datos de detección
                        xyxy = box.xyxy[0].cpu().numpy()
                        confidence = float(box.conf[0])
                        class_id = int(box.cls[0])
                        
                        # Convertir coordenadas
                        x1, y1, x2, y2 = xyxy
                        width = x2 - x1
                        height = y2 - y1
                        
                        # Validar detección
                        if confidence >= self.confidence_threshold and width > 10 and height > 10:
                            detections.append({
                                'bbox': [int(x1), int(y1), int(width), int(height)],
                                'confidence': confidence,
                                'class_id': class_id,
                                'class_name': self._get_class_name(class_id)
                            })
            
            return detections
            
        except Exception as e:
            logger.error(f"❌ Error en detección: {e}")
            return []
    
    def _basic_detection(self, frame):
        """Detección básica de fallback (placeholder)"""
        return []
    
    def _get_class_name(self, class_id: int) -> str:
        """Obtener nombre de clase de vehículo (COCO dataset)"""
        class_names = {
            2: 'car',
            3: 'motorcycle', 
            5: 'bus',
            7: 'truck'
        }
        return class_names.get(class_id, 'vehicle')
    
    def enhance_night_vision(self, frame: np.ndarray) -> np.ndarray:
        """Mejorar visión nocturna - Optimizado para Radxa Rock 5T"""
        try:
            # Convertir a LAB para mejor manipulación de luminancia
            lab = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)
            l, a, b = cv2.split(lab)
            
            # Aplicar CLAHE (Contrast Limited Adaptive Histogram Equalization)
            clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
            l = clahe.apply(l)
            
            # Recombinar canales
            enhanced = cv2.merge([l, a, b])
            enhanced = cv2.cvtColor(enhanced, cv2.COLOR_LAB2BGR)
            
            # Ajuste adicional de gamma para condiciones nocturnas
            gamma = 1.2
            enhanced = np.power(enhanced / 255.0, gamma)
            enhanced = (enhanced * 255).astype(np.uint8)
            
            return enhanced
            
        except Exception as e:
            logger.error(f"Error en mejora nocturna: {e}")
            return frame
    
    def get_model_info(self) -> dict:
        """Obtener información del modelo actual"""
        return {
            "model_type": self.model_type,
            "use_rknn": self.use_rknn,
            "rknn_available": RKNN_AVAILABLE,
            "ultralytics_available": ULTRALYTICS_AVAILABLE,
            "confidence_threshold": self.confidence_threshold,
            "input_size": self.input_size,
            "vehicle_classes": ["car", "motorcycle", "bus", "truck"]
        }
    
    def benchmark_model(self, test_image_path: str = None) -> dict:
        """Hacer benchmark del modelo (útil para optimización)"""
        if not self.model:
            return {"error": "Modelo no inicializado"}
        
        try:
            import time
            
            # Crear imagen de prueba si no se proporciona
            if test_image_path and os.path.exists(test_image_path):
                test_frame = cv2.imread(test_image_path)
            else:
                test_frame = np.random.randint(0, 255, (640, 640, 3), dtype=np.uint8)
            
            # Warm-up
            for _ in range(3):
                self.detect(test_frame)
            
            # Benchmark
            start_time = time.time()
            num_runs = 10
            
            for _ in range(num_runs):
                detections = self.detect(test_frame)
            
            end_time = time.time()
            avg_time = (end_time - start_time) / num_runs
            fps = 1.0 / avg_time if avg_time > 0 else 0
            
            return {
                "model_type": self.model_type,
                "avg_inference_time_ms": avg_time * 1000,
                "fps": fps,
                "num_detections": len(detections) if 'detections' in locals() else 0,
                "rknn_enabled": self.use_rknn
            }
            
        except Exception as e:
            return {"error": str(e)}