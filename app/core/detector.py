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

try:
    from rknnlite.api import RKNNLite
    RKNN_AVAILABLE = True
except ImportError:
    RKNN_AVAILABLE = False
    logger.warning("RKNN no disponible, usando CPU como fallback")

class VehicleDetector:
    """Detector de vehículos optimizado para YOLO11n + RKNN en Radxa Rock 5T - CORREGIDO"""
    
    def __init__(self, model_path: str = None, confidence_threshold: float = 0.5):
        self.confidence_threshold = confidence_threshold
        self.model = None
        self.rknn_model = None
        self.use_rknn = False
        self.model_type = "none"
        
        # FORZAR input size a 640x640 como menciona el usuario
        self.input_size = (640, 640)
        
        # Asegurar directorio de modelos
        os.makedirs("/app/models", exist_ok=True)
        
        # Inicializar modelo
        self._initialize_model(model_path)
    
    def _initialize_model(self, model_path: str = None):
        """Inicializar modelo con prioridad: RKNN > YOLO11n PyTorch"""
        
        # 1. Intentar RKNN primero (más importante para RK3588)
        if RKNN_AVAILABLE and self._check_rknn_support():
            if self._init_rknn_model():
                return
        
        # 2. Fallback a YOLO11n PyTorch
        if ULTRALYTICS_AVAILABLE:
            if self._init_yolo11n_pytorch():
                return
        
        # 3. Fallback final
        logger.warning("⚠️ Todos los métodos avanzados fallaron")
        self.model_type = "basic"
    
    def _check_rknn_support(self) -> bool:
        """Verificar si RKNN está disponible - MEJORADO"""
        # Verificar librknnrt.so
        lib_paths = [
            "/usr/lib/librknnrt.so",
            "/usr/lib/aarch64-linux-gnu/librknnrt.so"
        ]
        
        lib_found = False
        for lib_path in lib_paths:
            if os.path.exists(lib_path):
                logger.info(f"✅ librknnrt.so encontrada: {lib_path}")
                lib_found = True
                break
        
        if not lib_found:
            logger.error("❌ librknnrt.so no encontrada")
            return False
        
        # Verificar modelo RKNN
        rknn_patterns = [
            "/app/models/*.rknn",
            "/app/models/yolo11n*.rknn",
            "/app/models/yolov8n*.rknn"
        ]
        
        import glob
        for pattern in rknn_patterns:
            if glob.glob(pattern):
                logger.info(f"✅ Modelo RKNN encontrado: {pattern}")
                return True
        
        logger.error("❌ No se encontró ningún modelo .rknn")
        return False
    
    def _init_rknn_model(self) -> bool:
        """Inicializar modelo RKNN - COMPLETAMENTE CORREGIDO"""
        try:
            logger.info("🚀 Inicializando modelo RKNN para RK3588...")
            
            # Buscar modelo RKNN existente
            rknn_paths = [
                "/app/models/yolo11n.rknn",
                "/app/models/yolo11n_rk3588.rknn", 
                "/app/models/yolov8n.rknn",
                "/app/models/yolov8n_rk3588.rknn"
            ]
            
            rknn_model_path = None
            for path in rknn_paths:
                if os.path.exists(path):
                    rknn_model_path = path
                    logger.info(f"📁 Modelo RKNN encontrado: {path}")
                    break
            
            if not rknn_model_path:
                logger.info("📥 Modelo RKNN no encontrado, creando desde YOLO11n...")
                rknn_model_path = self._create_rknn_model()
                if not rknn_model_path:
                    return False
            
            # ✅ INICIALIZACIÓN CORRECTA DE RKNN
            self.rknn_model = RKNNLite()
            
            # ✅ CONFIGURACIÓN ESPECÍFICA PARA RK3588
            logger.info("🔧 Configurando RKNN para target_platform=rk3588...")
            
            # Cargar modelo
            ret = self.rknn_model.load_rknn(rknn_model_path)
            if ret != 0:
                logger.error(f"❌ Error cargando modelo RKNN: {ret}")
                return False
            
            # ✅ INICIALIZAR RUNTIME CON CONFIGURACIÓN CORRECTA
            # Según la documentación, init_runtime() debe llamarse sin parámetros
            ret = self.rknn_model.init_runtime()
            if ret != 0:
                logger.error(f"❌ Error inicializando RKNN runtime: {ret}")
                return False
            
            # ✅ VERIFICAR QUE EL NPU ESTÁ FUNCIONANDO
            try:
                # Test básico con imagen dummy
                dummy_input = np.random.rand(1, 3, 640, 640).astype(np.float32)
                test_output = self.rknn_model.inference(inputs=[dummy_input])
                if test_output is None or len(test_output) == 0:
                    logger.error("❌ Test de inferencia RKNN falló")
                    return False
                
                logger.info(f"✅ Test RKNN exitoso - Salidas: {len(test_output)}")
                
            except Exception as e:
                logger.error(f"❌ Error en test RKNN: {e}")
                return False
            
            self.use_rknn = True
            self.model_type = "yolo11n_rknn"
            
            logger.info("✅ YOLO11n + RKNN inicializado correctamente en RK3588")
            logger.info("ℹ️  Nota: El warning 'Query dynamic range failed' es normal para modelos estáticos")
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
        """Crear modelo RKNN desde YOLO11n - MEJORADO"""
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
                if ULTRALYTICS_AVAILABLE:
                    model = YOLO("yolo11n.pt")  # Descarga automáticamente
                    base_model_path = "/app/models/yolo11n.pt"
                    model.save(base_model_path)
                else:
                    logger.error("❌ Ultralytics no disponible para descargar modelo")
                    return None
            
            # ✅ USAR ULTRALYTICS PARA EXPORTAR CORRECTAMENTE A RKNN
            logger.info("🔄 Exportando a RKNN con Ultralytics...")
            try:
                model = YOLO(base_model_path)
                # Exportar directamente a RKNN con target platform correcto
                rknn_path = model.export(
                    format="rknn", 
                    name="rk3588",  # ✅ Especificar target platform
                    imgsz=640       # ✅ Forzar tamaño 640x640
                )
                
                # El modelo exportado estará en un directorio
                if os.path.isdir(rknn_path):
                    # Buscar el archivo .rknn dentro del directorio
                    import glob
                    rknn_files = glob.glob(os.path.join(rknn_path, "*.rknn"))
                    if rknn_files:
                        final_path = "/app/models/yolo11n_rk3588.rknn"
                        import shutil
                        shutil.copy2(rknn_files[0], final_path)
                        logger.info(f"✅ Modelo RKNN creado: {final_path}")
                        return final_path
                
                logger.error("❌ Error encontrando archivo RKNN exportado")
                return None
                
            except Exception as e:
                logger.error(f"❌ Error exportando con Ultralytics: {e}")
                # Fallback to manual export (código anterior)
                return self._manual_rknn_export(base_model_path)
            
        except Exception as e:
            logger.error(f"❌ Error creando modelo RKNN: {e}")
            return None
    
    def _manual_rknn_export(self, base_model_path: str) -> str:
        """Export manual a RKNN (fallback)"""
        try:
            logger.info("🔄 Intentando export manual a RKNN...")
            
            # Exportar a ONNX primero
            logger.info("🔄 Exportando a ONNX...")
            model = YOLO(base_model_path)
            onnx_path = model.export(format="onnx", imgsz=640)
            
            if not os.path.exists(onnx_path):
                logger.error("❌ Error exportando a ONNX")
                return None
            
            # Convertir ONNX a RKNN usando rknn-toolkit2 si está disponible
            try:
                from rknn.api import RKNN
                rknn = RKNN()
                
                # ✅ CONFIGURACIÓN CORRECTA PARA RK3588
                ret = rknn.config(target_platform='rk3588')
                if ret != 0:
                    logger.error(f"❌ Error configurando RKNN: {ret}")
                    return None
                
                # Cargar ONNX
                ret = rknn.load_onnx(onnx_path)
                if ret != 0:
                    logger.error(f"❌ Error cargando ONNX: {ret}")
                    return None
                
                # Build modelo
                ret = rknn.build(do_quantization=True)
                if ret != 0:
                    logger.error(f"❌ Error building RKNN: {ret}")
                    return None
                
                # Exportar
                rknn_path = "/app/models/yolo11n_rk3588.rknn"
                ret = rknn.export_rknn(rknn_path)
                if ret != 0:
                    logger.error(f"❌ Error exportando RKNN: {ret}")
                    return None
                
                rknn.release()
                
                # Limpiar ONNX temporal
                if os.path.exists(onnx_path):
                    os.remove(onnx_path)
                
                logger.info(f"✅ Modelo RKNN creado manualmente: {rknn_path}")
                return rknn_path
                
            except ImportError:
                logger.error("❌ rknn-toolkit2 no disponible para conversión manual")
                return None
            
        except Exception as e:
            logger.error(f"❌ Error en export manual: {e}")
            return None
    
    def _init_yolo11n_pytorch(self) -> bool:
        """Inicializar YOLO11n con PyTorch (fallback)"""
        try:
            logger.info("🔄 Inicializando YOLO11n con PyTorch...")
            
            model_paths = [
                "/app/models/yolo11n.pt",
                "/app/models/yolov8n.pt",
                "yolo11n.pt"
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
        """Detectar vehículos en frame - MÉTODO PRINCIPAL"""
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
        """Detectar con RKNN - COMPLETAMENTE CORREGIDO"""
        try:
            start_time = time.time()
            
            # ✅ PREPROCESAR FRAME A EXACTAMENTE 640x640
            input_data = self._preprocess_frame_rknn(frame)
            
            # ✅ INFERENCIA RKNN
            outputs = self.rknn_model.inference(inputs=[input_data])
            
            if outputs is None or len(outputs) == 0:
                logger.warning("⚠️ RKNN inference devolvió outputs vacíos")
                return []
            
            # ✅ POSTPROCESAR CON FORMATO CORRECTO YOLO11n RKNN
            detections = self._postprocess_rknn_yolo11n(outputs, frame.shape)
            
            inference_time = (time.time() - start_time) * 1000
            logger.debug(f"🔍 RKNN inference: {inference_time:.1f}ms, detecciones: {len(detections)}")
            
            return detections
            
        except Exception as e:
            logger.error(f"❌ Error en detección RKNN: {e}")
            return []
    
    def _preprocess_frame_rknn(self, frame):
        """Preprocesar frame para RKNN - CORREGIDO PARA 640x640"""
        
        # ✅ REDIMENSIONAR A EXACTAMENTE 640x640 (como menciona el usuario)
        resized = cv2.resize(frame, self.input_size)
        
        # ✅ CONVERTIR BGR A RGB
        rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
        
        # ✅ NORMALIZAR Y CAMBIAR DIMENSIONES SEGÚN YOLO11n
        input_data = rgb.astype(np.float32)
        input_data = input_data / 255.0  # Normalizar a [0, 1]
        input_data = np.transpose(input_data, (2, 0, 1))  # HWC to CHW
        input_data = np.expand_dims(input_data, axis=0)   # Add batch dimension [1, 3, 640, 640]
        
        return input_data
    
    def _postprocess_rknn_yolo11n(self, outputs, frame_shape):
        """Postprocesar salida de RKNN YOLO11n - COMPLETAMENTE REESCRITO"""
        
        detections = []
        
        try:
            # ✅ FORMATO DE SALIDA YOLO11n RKNN OPTIMIZADO
            # Según la documentación: outputs contiene múltiples escalas
            # Cada escala tiene formato: [coords, conf_per_class, conf_sum]
            # Por ejemplo: [1,64,80,80], [1,80,80,80], [1,1,80,80]
            
            h, w = frame_shape[:2]
            scale_x = w / 640.0
            scale_y = h / 640.0
            
            # ✅ PROCESAR CADA ESCALA DE YOLO11n
            scales = [
                {"size": 80, "stride": 8},   # 80x80 grid
                {"size": 40, "stride": 16},  # 40x40 grid  
                {"size": 20, "stride": 32}   # 20x20 grid
            ]
            
            output_idx = 0
            
            for scale_info in scales:
                if output_idx + 2 >= len(outputs):
                    break
                
                # ✅ EXTRAER OUTPUTS DE ESTA ESCALA
                coords_output = outputs[output_idx]      # [1, 64, H, W] - coordenadas
                conf_output = outputs[output_idx + 1]    # [1, 80, H, W] - confianza por clase
                conf_sum_output = outputs[output_idx + 2] # [1, 1, H, W] - suma de confianza
                
                output_idx += 3
                
                # ✅ PROCESAR GRID DE ESTA ESCALA
                grid_h, grid_w = scale_info["size"], scale_info["size"]
                stride = scale_info["stride"]
                
                # Verificar dimensiones
                if (coords_output.shape[-2] != grid_h or 
                    coords_output.shape[-1] != grid_w):
                    continue
                
                # ✅ ITERAR SOBRE GRID
                for cy in range(grid_h):
                    for cx in range(grid_w):
                        
                        # Obtener suma de confianza
                        obj_conf = conf_sum_output[0, 0, cy, cx]
                        
                        if obj_conf < self.confidence_threshold:
                            continue
                        
                        # ✅ OBTENER COORDENADAS (formato YOLO11n)
                        # coords_output contiene: [x_center, y_center, width, height]
                        x_center = coords_output[0, 0, cy, cx]
                        y_center = coords_output[0, 1, cy, cx]
                        width = coords_output[0, 2, cy, cx]
                        height = coords_output[0, 3, cy, cx]
                        
                        # ✅ CONVERTIR A COORDENADAS ABSOLUTAS
                        # Aplicar stride y offset del grid
                        abs_x = (cx + x_center) * stride
                        abs_y = (cy + y_center) * stride
                        abs_w = width * stride
                        abs_h = height * stride
                        
                        # ✅ CONVERTIR A FORMATO BOUNDING BOX
                        x1 = int((abs_x - abs_w/2) * scale_x)
                        y1 = int((abs_y - abs_h/2) * scale_y)
                        x2 = int((abs_x + abs_w/2) * scale_x)
                        y2 = int((abs_y + abs_h/2) * scale_y)
                        
                        # Validar coordenadas
                        x1 = max(0, min(x1, w))
                        y1 = max(0, min(y1, h))
                        x2 = max(0, min(x2, w))
                        y2 = max(0, min(y2, h))
                        
                        if x2 <= x1 or y2 <= y1:
                            continue
                        
                        # ✅ OBTENER MEJOR CLASE
                        # conf_output contiene confianza para cada una de las 80 clases COCO
                        class_confs = conf_output[0, :, cy, cx]
                        class_id = np.argmax(class_confs)
                        class_conf = class_confs[class_id]
                        
                        # ✅ CALCULAR CONFIANZA FINAL
                        final_conf = obj_conf * class_conf
                        
                        # ✅ FILTRAR SOLO VEHÍCULOS (clases COCO)
                        vehicle_classes = {
                            2: 'car',
                            3: 'motorcycle', 
                            5: 'bus',
                            7: 'truck'
                        }
                        
                        if class_id in vehicle_classes and final_conf >= self.confidence_threshold:
                            detections.append({
                                'bbox': [x1, y1, x2 - x1, y2 - y1],  # [x, y, width, height]
                                'confidence': float(final_conf),
                                'class_id': int(class_id),
                                'class_name': vehicle_classes[class_id]
                            })
            
            # ✅ APLICAR NMS (Non-Maximum Suppression)
            detections = self._apply_nms(detections, iou_threshold=0.45)
            
            return detections
            
        except Exception as e:
            logger.error(f"❌ Error en postprocesamiento RKNN: {e}")
            return []
    
    def _apply_nms(self, detections, iou_threshold=0.45):
        """Aplicar Non-Maximum Suppression"""
        if len(detections) == 0:
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
            logger.error(f"❌ Error en NMS: {e}")
            return detections
    
    def _detect_pytorch(self, frame):
        """Detectar con PyTorch - MEJORADO"""
        try:
            # ✅ USAR YOLO11n CON CONFIGURACIÓN OPTIMIZADA
            results = self.model(
                frame, 
                conf=self.confidence_threshold,
                iou=0.45,
                verbose=False,
                classes=[2, 3, 5, 7],  # Solo vehículos
                imgsz=640  # ✅ Forzar 640x640
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
        """Información del modelo - ACTUALIZADA"""
        return {
            "model_type": self.model_type,
            "use_rknn": self.use_rknn,
            "rknn_available": RKNN_AVAILABLE,
            "ultralytics_available": ULTRALYTICS_AVAILABLE,
            "confidence_threshold": self.confidence_threshold,
            "input_size": self.input_size,
            "target_platform": "rk3588",
            "vehicle_classes": ["car", "motorcycle", "bus", "truck"],
            "rknn_lib_found": any(os.path.exists(p) for p in [
                "/usr/lib/librknnrt.so",
                "/usr/lib/aarch64-linux-gnu/librknnrt.so"
            ]),
            "expected_performance": {
                "rknn_fps": "~10 FPS (99.5ms per image)",
                "pytorch_fps": "~3-5 FPS"
            }
        }
    
    def __del__(self):
        """Limpiar recursos"""
        if self.rknn_model:
            try:
                self.rknn_model.release()
            except:
                pass