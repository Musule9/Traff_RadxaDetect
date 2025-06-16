import unittest
import numpy as np
from app.core.tracker import BYTETracker, Track

class TestBYTETracker(unittest.TestCase):
    """Tests para el tracker BYTETracker"""
    
    def setUp(self):
        """Configurar test"""
        self.tracker = BYTETracker()
    
    def test_tracker_initialization(self):
        """Test inicialización del tracker"""
        self.assertEqual(self.tracker.high_thresh, 0.6)
        self.assertEqual(self.tracker.low_thresh, 0.1)
        self.assertEqual(self.tracker.max_age, 30)
        self.assertEqual(self.tracker.next_id, 1)
        self.assertEqual(len(self.tracker.tracks), 0)
    
    def test_track_creation(self):
        """Test creación de track"""
        track = Track(1, [100, 100, 50, 50], 0.8, 2)
        
        self.assertEqual(track.track_id, 1)
        self.assertEqual(track.bbox, [100, 100, 50, 50])
        self.assertEqual(track.confidence, 0.8)
        self.assertEqual(track.class_id, 2)
        self.assertEqual(track.hits, 1)
        self.assertEqual(track.time_since_update, 0)
    
    def test_track_update(self):
        """Test actualización de track"""
        track = Track(1, [100, 100, 50, 50], 0.8, 2)
        track.update([110, 105, 50, 50], 0.9)
        
        self.assertEqual(track.bbox, [110, 105, 50, 50])
        self.assertEqual(track.confidence, 0.9)
        self.assertEqual(track.hits, 2)
        self.assertEqual(len(track.history), 2)
    
    def test_calculate_iou(self):
        """Test cálculo de IoU"""
        bbox1 = [0, 0, 50, 50]
        bbox2 = [25, 25, 50, 50]
        
        iou = self.tracker._calculate_iou(bbox1, bbox2)
        
        # IoU esperado para overlapping boxes
        self.assertGreater(iou, 0)
        self.assertLess(iou, 1)
    
    def test_update_with_detections(self):
        """Test actualización con detecciones"""
        detections = [
            {'bbox': [100, 100, 50, 50], 'confidence': 0.8, 'class_id': 2},
            {'bbox': [200, 200, 60, 60], 'confidence': 0.7, 'class_id': 2}
        ]
        
        tracks = self.tracker.update(detections)
        
        # Debe crear 2 nuevos tracks
        self.assertEqual(len(tracks), 2)
        self.assertEqual(self.tracker.next_id, 3)
