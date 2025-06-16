import unittest
import numpy as np
import cv2
from unittest.mock import Mock, patch
import sys
import os

# Agregar el directorio raíz al path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from app.core.detector import VehicleDetector

class TestVehicleDetector(unittest.TestCase):
    """Tests para el detector de vehículos"""
    
    def setUp(self):
        """Configurar test"""
        self.detector = VehicleDetector("test_model.onnx", confidence_threshold=0.5)
    
    @patch('app.core.detector.RKNN_AVAILABLE', False)
    def test_detector_initialization_opencv(self):
        """Test inicialización con OpenCV"""
        detector = VehicleDetector("test_model.onnx")
        self.assertFalse(detector.use_rknn)
        self.assertEqual(detector.confidence_threshold, 0.5)
        self.assertEqual(detector.input_size, (640, 640))
    
    def test_preprocess_frame(self):
        """Test preprocesamiento de frame"""
        # Crear frame de prueba
        frame = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
        
        blob, scale = self.detector.preprocess_frame(frame)
        
        # Verificar dimensiones
        self.assertEqual(blob.shape, (1, 3, 640, 640))
        self.assertTrue(0 < scale <= 1)
        
        # Verificar normalización
        self.assertTrue(np.all(blob >= 0) and np.all(blob <= 1))
    
    def test_enhance_night_vision(self):
        """Test mejora de visión nocturna"""
        # Frame oscuro de prueba
        dark_frame = np.random.randint(0, 50, (480, 640, 3), dtype=np.uint8)
        
        enhanced = self.detector.enhance_night_vision(dark_frame)
        
        # Verificar que la mejora aumenta el brillo promedio
        self.assertGreater(np.mean(enhanced), np.mean(dark_frame))
        self.assertEqual(enhanced.shape, dark_frame.shape)
    
    def test_get_vehicle_classes(self):
        """Test clases de vehículos"""
        classes = self.detector._get_vehicle_classes()
        expected_classes = ['car', 'motorcycle', 'bus', 'truck']
        self.assertEqual(classes, expected_classes)
    
    def test_get_class_name(self):
        """Test obtener nombre de clase"""
        self.assertEqual(self.detector._get_class_name(2), 'car')
        self.assertEqual(self.detector._get_class_name(3), 'motorcycle')
        self.assertEqual(self.detector._get_class_name(5), 'bus')
        self.assertEqual(self.detector._get_class_name(7), 'truck')
        self.assertEqual(self.detector._get_class_name(999), 'vehicle')
