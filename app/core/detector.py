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
        self.rknn_model = None
        self.use_rknn = False
        self.model_type = "none"
        self.input_size = (640, 640)
        
        # Asegurar directorio de modelos
        os.makedirs("/app/models", exist_ok=True)
        
        # Inicializar modelo
        self._initialize_model(model_path)
    
    def _initialize_model(self, model_path: str = None):
        """Inicializar modelo con prioridad: RKNN > YOLO11n PyTorch > OpenCV"""
        
        # 1. Intentar RKNN primero
        if RKNN_AVAILABLE and self._check_rknn_support():
            if self._init_rknn_model():
                return
        
        # 2. Fallback a YOLO11n PyTorch
        if ULTRALYTICS_AVAILABLE:
            if self._init_yolo11n_pytorch():
                return
        
        # 3. Fallback final: detección básica
        logger.warning("⚠️ Todos los métodos avanzados fallaron, usando detección básica")
        self.model_type = "basic"
    
    def _check_rknn_support(self) -> bool:
        """Verificar si RKNN está disponible"""
        if not os.path.exists("/usr/lib/librknnrt.so"):
            logger.error("❌ librknnrt.so no encontrada en /usr/lib/")
            return False
        if not any(Path("/app/models").glob("*.rknn")):
            logger.error("❌ No se encontró ningún modelo .rknn en /app/models")
            return False
        logger.info("✅ RKNN parece estar disponible y hay modelo .rknn")
        return True
    
    def _init_rknn_model(self) -> bool:
        """Inicializar modelo RKNN"""
        try:
            logger.info("🚀 Inicializando modelo RKNN...")
            
            # Buscar modelo RKNN existente
            rknn_paths = [
                "/app/models/yolo11n_rk3588.rknn",
                "/app/models/yolo11n.rknn",
                "/app/models/yolov8n.rknn"
            ]
            
            rknn_model_path = None
            for path in rknn_paths:
                if os.path.exists(path):
                    rknn_model_path = path
                    logger.info(f"📁 Modelo RKNN encontrado: {path}")
                    break
            
            # Si no existe, crear modelo RKNN
            if not rknn_model_path:
                logger.info("📥 Modelo RKNN no encontrado, creando desde YOLO11n...")
                rknn_model_path = self._create_rknn_model()
                if not rknn_model_path:
                    return False
            
            # Inicializar RKNN
            self.rknn_model = RKNNLite()
            
            # Cargar modelo
            ret = self.rknn_model.load_rknn(rknn_model_path)
            if ret != 0:
                logger.error(f"❌ Error cargando modelo RKNN: {ret}")
                return False
            
            # Inicializar runtime
            ret = self.rknn_model.init_runtime()
            if ret != 0:
                logger.error(f"❌ Error inicializando RKNN runtime: {ret}")
                return False
            
            self.use_rknn = True
            self.model_type = "yolo11n_rknn"
            
            logger.info("✅ YOLO11n + RKNN inicializado correctamente")
            return True
            
        except Exception as e:
            logger.error(f"❌ Error inicializando RKNN: {e}")
            if self.rknn_model:
                try:
                    self.rknn_model.release()
                except:
                    pass
                self.rknn_model = None
            return False
    
    def _create_rknn_model(self) -> str:
        """Crear modelo RKNN desde YOLO11n"""
        try:
            logger.info("🔄 Creando modelo RKNN desde YOLO11n...")
            
            # Verificar si existe modelo base
            base_models = [
                "/app/models/yolo11n.pt",
                "/app/models/yolov8n.pt"
            ]
            
            base_model_path = None
            for path in base_models:
                if os.path.exists(path):
                    base_model_path = path
                    break
            
            if not base_model_path:
                logger.info("📥 Descargando YOLO11n base...")
                model = YOLO("yolo11n.pt")  # Descarga automáticamente
                base_model_path = "/app/models/yolo11n.pt"
                model.save(base_model_path)
            
            # Exportar a ONNX primero
            logger.info("🔄 Exportando a ONNX...")
            model = YOLO(base_model_path)
            onnx_path = model.export(format="onnx", imgsz=640)
            
            if not os.path.exists(onnx_path):
                logger.error("❌ Error exportando a ONNX")
                return None
            
            # Convertir ONNX a RKNN
            logger.info("🔄 Convirtiendo ONNX a RKNN...")
            rknn_path = "/app/models/yolo11n.rknn"
            
            rknn = RKNNLite()
            
            # Configurar RKNN
            ret = rknn.config(target_platform='rk3588')
            if ret != 0:
                logger.error(f"❌ Error configurando RKNN: {ret}")
                return None
            
            # Cargar ONNX
            ret = rknn.load_onnx(onnx_path)
            if ret != 0:
                logger.error(f"❌ Error cargando ONNX: {ret}")
                return None
            
            # Build modelo con cuantización
            ret = rknn.build(do_quantization=True, dataset=None)
            if ret != 0:
                logger.error(f"❌ Error building RKNN: {ret}")
                return None
            
            # Exportar
            ret = rknn.export_rknn(rknn_path)
            if ret != 0:
                logger.error(f"❌ Error exportando RKNN: {ret}")
                return None
            
            rknn.release()
            
            # Limpiar ONNX temporal
            if os.path.exists(onnx_path):
                os.remove(onnx_path)
            
            logger.info(f"✅ Modelo RKNN creado: {rknn_path}")
            return rknn_path
            
        except Exception as e:
            logger.error(f"❌ Error creando modelo RKNN: {e}")
            return None
    
    def _init_yolo11n_pytorch(self) -> bool:
        """Inicializar YOLO11n con PyTorch (fallback)"""
        try:
            logger.info("🔄 Inicializando YOLO11n con PyTorch...")
            
            # Buscar modelo existente
            model_paths = [
                "/app/models/yolo11n.pt",
                "/app/models/yolov8n.pt",
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
        """Detectar vehículos en frame"""
        try:
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
        """Detectar con RKNN"""
        try:
            # Preprocesar frame
            input_data = self._preprocess_frame_rknn(frame)
            
            # Inferencia
            outputs = self.rknn_model.inference(inputs=[input_data])
            
            # Postprocesar
            detections = self._postprocess_rknn(outputs[0], frame.shape)
            
            return detections
            
        except Exception as e:
            logger.error(f"❌ Error en detección RKNN: {e}")
            return []
    
    def _detect_pytorch(self, frame):
        """Detectar con PyTorch"""
        try:
            # Usar YOLO11n moderno
            results = self.model(
                frame, 
                conf=self.confidence_threshold,
                iou=0.7,
                verbose=False,
                classes=[2, 3, 5, 7]  # Solo vehículos
            )
            
            detections = []
            for result in results:
                if result.boxes is not None:
                    for box in result.boxes:
                        # Extraer datos
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
            logger.error(f"❌ Error en detección PyTorch: {e}")
            return []
    
    def _preprocess_frame_rknn(self, frame):
        """Preprocesar frame para RKNN"""
        # Redimensionar a 640x640
        resized = cv2.resize(frame, self.input_size)
        
        # Convertir BGR a RGB
        rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
        
        # Normalizar y cambiar dimensiones
        input_data = rgb.astype(np.float32)
        input_data = input_data / 255.0
        input_data = np.transpose(input_data, (2, 0, 1))  # HWC to CHW
        input_data = np.expand_dims(input_data, axis=0)   # Add batch dimension
        
        return input_data
    
    def _postprocess_rknn(self, output, frame_shape):
        """Postprocesar salida de RKNN"""
        detections = []
        
        # Configuración para YOLO
        anchors = [10, 13, 16, 30, 33, 23, 30, 61, 62, 45, 59, 119, 116, 90, 156, 198, 373, 326]
        num_classes = 80
        conf_threshold = self.confidence_threshold
        
        # Postprocesamiento simplificado
        # Esto es una implementación básica - se puede optimizar
        
        h, w = frame_shape[:2]
        scale_x = w / 640
        scale_y = h / 640
        
        # Procesar output (simplificado)
        if len(output.shape) == 3:
            output = output[0]  # Remove batch dimension
        
        # Filtrar por confianza
        for detection in output:
            if len(detection) >= 6:  # [x, y, w, h, conf, class_scores...]
                conf = detection[4]
                if conf >= conf_threshold:
                    # Coordenadas
                    x_center, y_center, width, height = detection[:4]
                    
                    # Convertir a coordenadas absolutas
                    x1 = int((x_center - width/2) * scale_x)
                    y1 = int((y_center - height/2) * scale_y)
                    w = int(width * scale_x)
                    h = int(height * scale_y)
                    
                    # Clase con mayor probabilidad
                    class_scores = detection[5:]
                    class_id = np.argmax(class_scores)
                    
                    # Solo vehículos
                    if class_id in [2, 3, 5, 7]:
                        detections.append({
                            'bbox': [x1, y1, w, h],
                            'confidence': float(conf),
                            'class_id': class_id,
                            'class_name': self._get_class_name(class_id)
                        })
        
        return detections
    
    def _get_class_name(self, class_id: int) -> str:
        """Obtener nombre de clase"""
        class_names = {
            2: 'car',
            3: 'motorcycle', 
            5: 'bus',
            7: 'truck'
        }
        return class_names.get(class_id, 'vehicle')
    
    def enhance_night_vision(self, frame: np.ndarray) -> np.ndarray:
        """Mejorar visión nocturna"""
        try:
            # Convertir a LAB
            lab = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)
            l, a, b = cv2.split(lab)
            
            # CLAHE
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
    
    def get_model_info(self) -> dict:
        """Información del modelo"""
        return {
            "model_type": self.model_type,
            "use_rknn": self.use_rknn,
            "rknn_available": RKNN_AVAILABLE,
            "ultralytics_available": ULTRALYTICS_AVAILABLE,
            "confidence_threshold": self.confidence_threshold,
            "input_size": self.input_size,
            "vehicle_classes": ["car", "motorcycle", "bus", "truck"],
            "rknn_lib_found": os.path.exists("/usr/lib/librknnrt.so")
        }
    
    def __del__(self):
        """Limpiar recursos"""
        if self.rknn_model:
            try:
                self.rknn_model.release()
            except:
                pass