import numpy as np
from typing import List, Dict, Optional, Tuple
from collections import defaultdict
import time

class Track:
    """Representación de un track de vehículo"""
    
    def __init__(self, track_id: int, bbox: List[float], confidence: float, class_id: int):
        self.track_id = track_id
        self.bbox = bbox  # [x, y, w, h]
        self.confidence = confidence
        self.class_id = class_id
        self.age = 0
        self.hits = 1
        self.time_since_update = 0
        self.history = [bbox]
        self.velocities = []
        self.created_time = time.time()
        
    def update(self, bbox: List[float], confidence: float):
        """Actualizar track con nueva detección"""
        self.bbox = bbox
        self.confidence = confidence
        self.hits += 1
        self.time_since_update = 0
        self.history.append(bbox)
        
        # Mantener solo últimas 10 posiciones
        if len(self.history) > 10:
            self.history.pop(0)
        
        # Calcular velocidad
        if len(self.history) >= 2:
            prev_center = self._get_center(self.history[-2])
            curr_center = self._get_center(self.history[-1])
            velocity = np.sqrt((curr_center[0] - prev_center[0])**2 + 
                             (curr_center[1] - prev_center[1])**2)
            self.velocities.append(velocity)
            
            # Mantener solo últimas 5 velocidades
            if len(self.velocities) > 5:
                self.velocities.pop(0)
    
    def predict(self):
        """Predecir siguiente posición"""
        self.age += 1
        self.time_since_update += 1
    
    def _get_center(self, bbox: List[float]) -> Tuple[float, float]:
        """Obtener centro de bbox"""
        return (bbox[0] + bbox[2] / 2, bbox[1] + bbox[3] / 2)
    
    @property
    def center(self) -> Tuple[float, float]:
        """Centro actual del track"""
        return self._get_center(self.bbox)
    
    @property
    def average_velocity(self) -> float:
        """Velocidad promedio del vehículo"""
        if not self.velocities:
            return 0.0
        return sum(self.velocities) / len(self.velocities)

class BYTETracker:
    """Implementación simplificada de BYTETrack para tracking de vehículos"""
    
    def __init__(self, 
                 high_thresh: float = 0.6,
                 low_thresh: float = 0.1,
                 max_age: int = 30,
                 min_hits: int = 3):
        
        self.high_thresh = high_thresh
        self.low_thresh = low_thresh
        self.max_age = max_age
        self.min_hits = min_hits
        
        self.tracks = []
        self.next_id = 1
        self.frame_count = 0
    
    def update(self, detections: List[Dict]) -> List[Track]:
        """Actualizar tracker con nuevas detecciones"""
        self.frame_count += 1
        
        # Separar detecciones por confianza
        high_det = [d for d in detections if d['confidence'] >= self.high_thresh]
        low_det = [d for d in detections if self.low_thresh <= d['confidence'] < self.high_thresh]
        
        # Predecir tracks existentes
        for track in self.tracks:
            track.predict()
        
        # Asociar detecciones de alta confianza
        matched_tracks, unmatched_dets, unmatched_tracks = self._associate(
            self.tracks, high_det, 0.7)
        
        # Actualizar tracks asociados
        for track_idx, det_idx in matched_tracks:
            self.tracks[track_idx].update(
                high_det[det_idx]['bbox'],
                high_det[det_idx]['confidence']
            )
        
        # Asociar tracks no asociados con detecciones de baja confianza
        unmatched_tracks_active = [i for i in unmatched_tracks 
                                 if self.tracks[i].time_since_update <= 1]
        
        matched_tracks_low, unmatched_dets_low, _ = self._associate(
            [self.tracks[i] for i in unmatched_tracks_active], low_det, 0.5)
        
        # Actualizar con detecciones de baja confianza
        for track_idx, det_idx in matched_tracks_low:
            actual_track_idx = unmatched_tracks_active[track_idx]
            self.tracks[actual_track_idx].update(
                low_det[det_idx]['bbox'],
                low_det[det_idx]['confidence']
            )
        
        # Crear nuevos tracks con detecciones no asociadas de alta confianza
        for det_idx in unmatched_dets:
            det = high_det[det_idx]
            new_track = Track(
                self.next_id,
                det['bbox'],
                det['confidence'],
                det['class_id']
            )
            self.tracks.append(new_track)
            self.next_id += 1
        
        # Remover tracks antiguos
        self.tracks = [track for track in self.tracks 
                      if track.time_since_update <= self.max_age]
        
        # Retornar tracks válidos
        valid_tracks = [track for track in self.tracks 
                       if track.hits >= self.min_hits or track.time_since_update <= 1]
        
        return valid_tracks
    
    def _associate(self, tracks: List[Track], detections: List[Dict], 
                  iou_threshold: float) -> Tuple[List, List, List]:
        """Asociar tracks con detecciones usando IoU"""
        if not tracks or not detections:
            return [], list(range(len(detections))), list(range(len(tracks)))
        
        # Calcular matriz de IoU
        iou_matrix = np.zeros((len(tracks), len(detections)))
        for i, track in enumerate(tracks):
            for j, det in enumerate(detections):
                iou_matrix[i, j] = self._calculate_iou(track.bbox, det['bbox'])
        
        # Asociación greedy (simple pero efectiva)
        matched_tracks = []
        matched_detections = []
        
        for _ in range(min(len(tracks), len(detections))):
            # Encontrar máximo IoU
            max_iou_idx = np.unravel_index(np.argmax(iou_matrix), iou_matrix.shape)
            max_iou = iou_matrix[max_iou_idx]
            
            if max_iou < iou_threshold:
                break
            
            matched_tracks.append(max_iou_idx)
            matched_detections.extend(max_iou_idx)
            
            # Eliminar fila y columna asociadas
            iou_matrix[max_iou_idx[0], :] = 0
            iou_matrix[:, max_iou_idx[1]] = 0
        
        unmatched_tracks = [i for i in range(len(tracks)) 
                          if i not in [m[0] for m in matched_tracks]]
        unmatched_detections = [i for i in range(len(detections)) 
                              if i not in [m[1] for m in matched_tracks]]
        
        return matched_tracks, unmatched_detections, unmatched_tracks
    
    def _calculate_iou(self, bbox1: List[float], bbox2: List[float]) -> float:
        """Calcular Intersection over Union"""
        x1, y1, w1, h1 = bbox1
        x2, y2, w2, h2 = bbox2
        
        # Coordenadas de intersección
        xi1 = max(x1, x2)
        yi1 = max(y1, y2)
        xi2 = min(x1 + w1, x2 + w2)
        yi2 = min(y1 + h1, y2 + h2)
        
        if xi2 <= xi1 or yi2 <= yi1:
            return 0.0
        
        # Áreas
        inter_area = (xi2 - xi1) * (yi2 - yi1)
        bbox1_area = w1 * h1
        bbox2_area = w2 * h2
        union_area = bbox1_area + bbox2_area - inter_area
        
        return inter_area / union_area if union_area > 0 else 0.0