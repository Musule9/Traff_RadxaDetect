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

# ✅ IMPORTACIÓN CORRECTA PARA RADXA ROCK 5T
RKNN_AVAILABLE = False
try:
    # Verificar si python3-rknnlite2 está instalado
    import subprocess
    result = subprocess.run(['python3', '-c', 'import rknnlite2'], 
                          capture_output=True, text=True)
    if result.returncode == 0:
        from rknnlite2 import RKNNLite
        RKNN_AVAILABLE = True
        logger.info("✅ RKNN-Toolkit2-Lite disponible para RK3588")
    else:
        logger.warning("❌ python3-rknnlite2 no instalado")
except Exception as e:
    logger.warning(f"❌ RKNN no disponible: {e}")

class VehicleDetector:
    """Detector de vehículos YOLO11n + RKNN optimizado para Radxa Rock 5T"""
    
    def __init__(self, model_path: str = None, confidence_threshold: float = 0.5):
        self.confidence_threshold = confidence_threshold
        self.model = None
        self.rknn_model = None
        self.use_rknn = False
        self.model_type = "none"
        
        # ✅ FORZAR RESOLUCIÓN 640x640 SIEMPRE
        self.input_size = (640, 640)
        self.input_width = 640
        self.input_height = 640
        
        # Asegurar directorio de modelos
        os.makedirs("/app/models", exist_ok=True)
        
        # Inicializar modelo
        self._initialize_model(model_path)
    
    def _initialize_model(self, model_path: str = None):
        """Inicializar modelo con prioridad: RKNN > YOLO11n PyTorch"""
        
        # 1. ✅ INTENTAR RKNN PRIMERO (ESPECÍFICO PARA RK3588)
        if RKNN_AVAILABLE and self._check_rk3588_support():
            if self._init_rknn_yolo11n():
                return
        
        # 2. Fallback a YOLO11n PyTorch
        if ULTRALYTICS_AVAILABLE:
            if self._init_yolo11n_pytorch():
                return
        
        # 3. Fallback final
        logger.warning("⚠️ Todos los métodos fallaron")
        self.model_type = "basic"
    
    def _check_rk3588_support(self) -> bool:
        """Verificar soporte RK3588 específico"""
        
        # ✅ 1. VERIFICAR HARDWARE RK3588
        try:
            if os.path.exists("/proc/device-tree/model"):
                with open("/proc/device-tree/model", "rb") as f:
                    model = f.read().decode('utf-8', errors='ignore').strip('\x00')
                    if "RK3588" not in model and "Radxa" not in model:
                        logger.warning(f"⚠️ Hardware no es RK3588: {model}")
                        return False
                    logger.info(f"✅ Hardware RK3588 detectado: {model}")
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
        
        # ✅ 3. VERIFICAR MODELO RKNN
        rknn_paths = [
            "/app/models/yolo11n.rknn",
            "/app/models/yolo11n_rk3588.rknn"
        ]
        
        for path in rknn_paths:
            if os.path.exists(path):
                logger.info(f"✅ Modelo RKNN encontrado: {path}")
                return True
        
        logger.info("📥 Modelo RKNN no encontrado, será creado automáticamente...")
        return True  # Permitir creación automática
    
    def _init_rknn_yolo11n(self) -> bool:
        """Inicializar YOLO11n RKNN para RK3588 - COMPLETAMENTE CORREGIDO"""
        try:
            logger.info("🚀 Inicializando YOLO11n RKNN para RK3588...")
            
            # ✅ 1. BUSCAR/CREAR MODELO RKNN
            rknn_model_path = self._get_or_create_rknn_model()
            if not rknn_model_path:
                logger.error("❌ No se pudo obtener modelo RKNN")
                return False
            
            # ✅ 2. INICIALIZAR RKNN CON CONFIGURACIÓN RK3588
            self.rknn_model = RKNNLite()
            
            logger.info(f"📁 Cargando modelo RKNN: {rknn_model_path}")
            ret = self.rknn_model.load_rknn(rknn_model_path)
            if ret != 0:
                logger.error(f"❌ Error cargando modelo RKNN: {ret}")
                return False
            
            # ✅ 3. CONFIGURAR RUNTIME PARA RK3588
            logger.info("🔧 Configurando runtime para RK3588...")
            # Según documentación, init_runtime() para RK3588 no necesita parámetros
            ret = self.rknn_model.init_runtime()
            if ret != 0:
                logger.error(f"❌ Error inicializando runtime RKNN: {ret}")
                self.rknn_model.release()
                self.rknn_model = None
                return False
            
            # ✅ 4. TEST DE FUNCIONAMIENTO
            logger.info("🧪 Probando inferencia RKNN...")
            test_input = np.random.rand(1, 3, 640, 640).astype(np.float32)
            
            start_time = time.time()
            outputs = self.rknn_model.inference(inputs=[test_input])
            inference_time = (time.time() - start_time) * 1000
            
            if outputs is None or len(outputs) == 0:
                logger.error("❌ Test de inferencia RKNN falló")
                self.rknn_model.release()
                self.rknn_model = None
                return False
            
            logger.info(f"✅ RKNN inicializado exitosamente!")
            logger.info(f"⚡ Tiempo de inferencia test: {inference_time:.1f}ms")
            logger.info(f"📊 Outputs: {len(outputs)} tensores")
            
            # Mostrar información de outputs para debug
            for i, output in enumerate(outputs):
                logger.info(f"   Output {i}: shape {output.shape}, dtype {output.dtype}")
            
            self.use_rknn = True
            self.model_type = "yolo11n_rknn_rk3588"
            
            logger.info("🎯 YOLO11n RKNN RK3588 listo para inferencia!")
            return True
            
        except Exception as e:
            logger.error(f"❌ Error crítico inicializando RKNN: {e}")
            if self.rknn_model:
                try:
                    self.rknn_model.release()
                except:
                    pass
                self.rknn_model = None
            return False
    
    def _get_or_create_rknn_model(self) -> str:
        """Obtener o crear modelo RKNN YOLO11n para RK3588"""
        
        # ✅ BUSCAR MODELOS EXISTENTES
        rknn_paths = [
            "/app/models/yolo11n.rknn",
            "/app/models/yolo11n_rk3588.rknn"
        ]
        
        for path in rknn_paths:
            if os.path.exists(path):
                logger.info(f"✅ Usando modelo RKNN existente: {path}")
                return path
        
        # ✅ CREAR MODELO USANDO ULTRALYTICS (MÉTODO OFICIAL)
        logger.info("🔄 Creando modelo RKNN con Ultralytics...")
        try:
            if not ULTRALYTICS_AVAILABLE:
                logger.error("❌ Ultralytics no disponible para crear RKNN")
                return None
            
            # ✅ MÉTODO OFICIAL ULTRALYTICS PARA RK3588
            logger.info("📥 Descargando/cargando YOLO11n...")
            model = YOLO("yolo11n.pt")  # Descarga automáticamente si no existe
            
            logger.info("⚙️ Exportando a RKNN formato RK3588...")
            # Según documentación oficial de Ultralytics + Radxa
            rknn_path = model.export(
                format="rknn",
                name="rk3588",  # ✅ TARGET PLATFORM ESPECÍFICO
                imgsz=640       # ✅ FORZAR 640x640
            )
            
            # ✅ MOVER EL MODELO AL DIRECTORIO CORRECTO
            final_path = "/app/models/yolo11n_rk3588.rknn"
            
            # El export puede crear un directorio o archivo
            if os.path.isdir(rknn_path):
                import glob
                rknn_files = glob.glob(os.path.join(rknn_path, "*.rknn"))
                if rknn_files:
                    import shutil
                    shutil.copy2(rknn_files[0], final_path)
                    logger.info(f"✅ Modelo RKNN creado: {final_path}")
                    return final_path
                else:
                    logger.error("❌ No se encontró archivo .rknn en directorio exportado")
                    return None
            elif os.path.isfile(rknn_path) and rknn_path.endswith('.rknn'):
                import shutil
                shutil.copy2(rknn_path, final_path)
                logger.info(f"✅ Modelo RKNN creado: {final_path}")
                return final_path
            else:
                logger.error(f"❌ Export no generó archivo RKNN válido: {rknn_path}")
                return None
                
        except Exception as e:
            logger.error(f"❌ Error creando modelo RKNN con Ultralytics: {e}")
            logger.info("💡 Sugerencia: Ejecute manualmente: model.export(format='rknn', name='rk3588')")
            return None
    
    def _init_yolo11n_pytorch(self) -> bool:
        """Inicializar YOLO11n con PyTorch (fallback)"""
        try:
            logger.info("🔄 Inicializando YOLO11n PyTorch fallback...")
            
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
        """Detectar vehículos - MÉTODO PRINCIPAL"""
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
        """Detectar con RKNN YOLO11n - COMPLETAMENTE REESCRITO"""
        try:
            start_time = time.time()
            
            # ✅ 1. PREPROCESAR A 640x640 EXACTO
            input_data = self._preprocess_frame_rknn(frame)
            
            # ✅ 2. INFERENCIA RKNN
            outputs = self.rknn_model.inference(inputs=[input_data])
            
            if outputs is None or len(outputs) == 0:
                logger.warning("⚠️ RKNN inference devolvió outputs vacíos")
                return []
            
            # ✅ 3. POSTPROCESAR CON FORMATO YOLO11n RKNN CORRECTO
            detections = self._postprocess_yolo11n_rknn(outputs, frame.shape)
            
            inference_time = (time.time() - start_time) * 1000
            
            if len(detections) > 0:
                logger.debug(f"🔍 RKNN inference: {inference_time:.1f}ms, {len(detections)} detecciones")
            
            return detections
            
        except Exception as e:
            logger.error(f"❌ Error en detección RKNN: {e}")
            return []
    
    def _preprocess_frame_rknn(self, frame):
        """Preprocesar frame para RKNN - OPTIMIZADO PARA 640x640"""
        
        # ✅ 1. REDIMENSIONAR A 640x640 EXACTO
        resized = cv2.resize(frame, (self.input_width, self.input_height))
        
        # ✅ 2. BGR A RGB
        rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
        
        # ✅ 3. NORMALIZAR SEGÚN YOLO11n
        input_data = rgb.astype(np.float32) / 255.0
        
        # ✅ 4. CAMBIAR DIMENSIONES: HWC → CHW → NCHW
        input_data = np.transpose(input_data, (2, 0, 1))  # HWC to CHW
        input_data = np.expand_dims(input_data, axis=0)   # Add batch dimension
        
        # Verificar shape final
        assert input_data.shape == (1, 3, 640, 640), f"Shape incorrecto: {input_data.shape}"
        
        return input_data
    
    def _postprocess_yolo11n_rknn(self, outputs, original_frame_shape):
        """Postprocesar salida YOLO11n RKNN - REESCRITO ESPECÍFICAMENTE"""
        
        detections = []
        
        try:
            # ✅ YOLO11n RKNN TIENE FORMATO ESPECÍFICO
            # Según documentación: Output shape típicamente [1, 84, 8400] o similar
            # Donde 84 = 4 (bbox) + 80 (clases COCO)
            
            if len(outputs) == 0:
                return detections
            
            # Tomar primer output (detecciones principales)
            output = outputs[0]
            logger.debug(f"📊 Output shape: {output.shape}")
            
            # ✅ FORMATO YOLO11n: [batch, features, detections]
            # Features = 4 (x,y,w,h) + num_classes (80 para COCO)
            if len(output.shape) == 3 and output.shape[0] == 1:
                # Formato [1, 84, 8400] → [84, 8400]
                output = output[0]
            
            if len(output.shape) != 2:
                logger.warning(f"⚠️ Output shape inesperado: {output.shape}")
                return detections
            
            num_features, num_detections = output.shape
            logger.debug(f"📊 Features: {num_features}, Detecciones: {num_detections}")
            
            # ✅ EXTRAER COORDENADAS Y CONFIANZAS
            if num_features >= 84:  # 4 bbox + 80 clases
                boxes = output[:4, :]      # Primeras 4 filas: x, y, w, h
                scores = output[4:, :]     # Resto: confianzas de clases
            else:
                logger.warning(f"⚠️ Número de features insuficiente: {num_features}")
                return detections
            
            # ✅ CALCULAR ESCALAS PARA REDIMENSIONAR
            orig_h, orig_w = original_frame_shape[:2]
            scale_x = orig_w / self.input_width
            scale_y = orig_h / self.input_height
            
            # ✅ PROCESAR CADA DETECCIÓN
            for i in range(num_detections):
                # Obtener confianza máxima entre todas las clases
                class_scores = scores[:, i]
                max_score_idx = np.argmax(class_scores)
                max_score = class_scores[max_score_idx]
                
                # Filtrar por confianza
                if max_score < self.confidence_threshold:
                    continue
                
                # ✅ FILTRAR SOLO CLASES DE VEHÍCULOS (COCO)
                vehicle_classes = {2: 'car', 3: 'motorcycle', 5: 'bus', 7: 'truck'}
                if max_score_idx not in vehicle_classes:
                    continue
                
                # ✅ EXTRAER COORDENADAS (formato YOLO: center_x, center_y, width, height)
                center_x = boxes[0, i] * scale_x
                center_y = boxes[1, i] * scale_y
                width = boxes[2, i] * scale_x
                height = boxes[3, i] * scale_y
                
                # ✅ CONVERTIR A FORMATO BBOX [x, y, width, height]
                x = int(center_x - width / 2)
                y = int(center_y - height / 2)
                w = int(width)
                h = int(height)
                
                # ✅ VALIDAR COORDENADAS
                x = max(0, min(x, orig_w - 1))
                y = max(0, min(y, orig_h - 1))
                w = max(1, min(w, orig_w - x))
                h = max(1, min(h, orig_h - y))
                
                # ✅ AGREGAR DETECCIÓN VÁLIDA
                detections.append({
                    'bbox': [x, y, w, h],
                    'confidence': float(max_score),
                    'class_id': int(max_score_idx),
                    'class_name': vehicle_classes[max_score_idx]
                })
            
            # ✅ APLICAR NMS
            detections = self._apply_nms(detections, iou_threshold=0.45)
            
            logger.debug(f"🎯 Detecciones post-NMS: {len(detections)}")
            
            return detections
            
        except Exception as e:
            logger.error(f"❌ Error en postprocesamiento YOLO11n RKNN: {e}")
            logger.error(f"   Output shapes: {[o.shape for o in outputs]}")
            return []
    
    def _apply_nms(self, detections, iou_threshold=0.45):
        """Aplicar Non-Maximum Suppression"""
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
            
            # Aplicar NMS usando OpenCV
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
        """Detectar con PyTorch YOLO11n"""
        try:
            # ✅ FORZAR 640x640 TAMBIÉN EN PYTORCH
            results = self.model(
                frame, 
                conf=self.confidence_threshold,
                iou=0.45,
                verbose=False,
                classes=[2, 3, 5, 7],  # Solo vehículos
                imgsz=640
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
        """Mejorar visión nocturna"""
        try:
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
        """Información del modelo - ACTUALIZADA"""
        return {
            "model_type": self.model_type,
            "use_rknn": self.use_rknn,
            "rknn_available": RKNN_AVAILABLE,
            "ultralytics_available": ULTRALYTICS_AVAILABLE,
            "confidence_threshold": self.confidence_threshold,
            "input_size": self.input_size,
            "target_platform": "rk3588",
            "forced_resolution": "640x640",
            "vehicle_classes": ["car", "motorcycle", "bus", "truck"],
            "expected_performance_rknn": "99.5ms per image (~10 FPS)",
            "expected_performance_pytorch": "200-300ms per image (~3-5 FPS)",
            "rknn_toolkit_version": "rknn-toolkit2-lite",
            "hardware_acceleration": "NPU RK3588" if self.use_rknn else "CPU"
        }
    
    def __del__(self):
        """Limpiar recursos"""
        if self.rknn_model:
            try:
                self.rknn_model.release()
                logger.info("🧹 RKNN model released")
            except:
                pass