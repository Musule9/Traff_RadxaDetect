import cv2
import numpy as np
from typing import List, Dict, Tuple, Optional
import time
import math
from dataclasses import dataclass
from enum import Enum

class LineType(Enum):
    COUNTING = "counting"
    SPEED = "speed"

@dataclass
class Line:
    id: str
    name: str
    points: List[Tuple[int, int]]  # [(x1, y1), (x2, y2)]
    lane: str
    line_type: LineType
    distance_to_next: Optional[float] = None  # metros

@dataclass
class Zone:
    id: str
    name: str
    points: List[Tuple[int, int]]  # Polígono
    zone_type: str = "red_light"

class TrafficAnalyzer:
    """Analizador de tráfico para conteo, velocidad y zona roja"""
    
    def __init__(self):
        self.lines: List[Line] = []
        self.zones: List[Zone] = []
        self.vehicle_line_crossings = {}  # {vehicle_id: {line_id: timestamp}}
        self.vehicle_speeds = {}  # {vehicle_id: speed_kmh}
        self.red_light_active = False
        self.vehicles_in_red_zone = set()
        self.red_light_start_time = None
        self.red_light_vehicles_start = 0
        self.analytic_sent_this_cycle = False
        
    def add_line(self, line: Line):
        """Agregar línea de conteo o velocidad"""
        self.lines.append(line)
    
    def add_zone(self, zone: Zone):
        """Agregar zona de análisis"""
        self.zones.append(zone)
    
    def update_red_light_status(self, is_red: bool):
        """Actualizar estado del semáforo"""
        if is_red and not self.red_light_active:
            # Inicio de ciclo rojo
            self.red_light_active = True
            self.red_light_start_time = time.time()
            self.red_light_vehicles_start = len(self.vehicles_in_red_zone)
            self.analytic_sent_this_cycle = False
            
        elif not is_red and self.red_light_active:
            # Fin de ciclo rojo
            self.red_light_active = False
            self.red_light_start_time = None
            self.analytic_sent_this_cycle = False
    
    def analyze_frame(self, tracks: List, frame_shape: Tuple[int, int]) -> Dict:
        """Analizar frame con tracks de vehículos"""
        results = {
            'line_crossings': [],
            'speed_calculations': [],
            'red_zone_count': 0,
            'send_analytic': False,
            'vehicles_in_red_zone': []
        }
        
        current_time = time.time()
        current_vehicles_in_zone = set()
        
        for track in tracks:
            vehicle_id = track.track_id
            center = track.center
            
            # Verificar cruces de líneas
            line_crossings = self._check_line_crossings(vehicle_id, center, current_time)
            results['line_crossings'].extend(line_crossings)
            
            # Calcular velocidades
            speed_calc = self._calculate_speed(vehicle_id, current_time)
            if speed_calc:
                results['speed_calculations'].append(speed_calc)
            
            # Verificar vehículos en zona roja
            if self._point_in_red_zones(center):
                current_vehicles_in_zone.add(vehicle_id)
                results['vehicles_in_red_zone'].append({
                    'vehicle_id': vehicle_id,
                    'bbox': track.bbox,
                    'confidence': track.confidence
                })
        
        # Actualizar vehículos en zona roja
        self.vehicles_in_red_zone = current_vehicles_in_zone
        results['red_zone_count'] = len(self.vehicles_in_red_zone)
        
        # Determinar si enviar analítico
        if (self.red_light_active and 
            len(self.vehicles_in_red_zone) > 0 and 
            not self.analytic_sent_this_cycle):
            results['send_analytic'] = True
            self.analytic_sent_this_cycle = True
        
        return results
    
    def _check_line_crossings(self, vehicle_id: int, center: Tuple[float, float], 
                            current_time: float) -> List[Dict]:
        """Verificar si vehículo cruza líneas"""
        crossings = []
        
        if vehicle_id not in self.vehicle_line_crossings:
            self.vehicle_line_crossings[vehicle_id] = {}
        
        for line in self.lines:
            # Verificar si el punto cruza la línea
            if self._point_crosses_line(center, line.points):
                line_id = line.id
                
                # Verificar si ya cruzó esta línea (evitar duplicados)
                if line_id not in self.vehicle_line_crossings[vehicle_id]:
                    self.vehicle_line_crossings[vehicle_id][line_id] = current_time
                    
                    crossings.append({
                        'vehicle_id': vehicle_id,
                        'line_id': line_id,
                        'line_name': line.name,
                        'lane': line.lane,
                        'timestamp': current_time,
                        'center': center
                    })
        
        return crossings
    
    def _calculate_speed(self, vehicle_id: int, current_time: float) -> Optional[Dict]:
        """Calcular velocidad del vehículo entre dos líneas"""
        if vehicle_id not in self.vehicle_line_crossings:
            return None
        
        crossings = self.vehicle_line_crossings[vehicle_id]
        
        # Buscar pares de líneas para cálculo de velocidad
        for line in self.lines:
            if line.line_type == LineType.COUNTING and line.distance_to_next:
                line1_id = line.id
                # Buscar línea siguiente en el mismo carril
                line2 = next((l for l in self.lines 
                            if l.lane == line.lane and l.line_type == LineType.SPEED), None)
                
                if line2 and line1_id in crossings and line2.id in crossings:
                    time1 = crossings[line1_id]
                    time2 = crossings[line2.id]
                    
                    # Calcular velocidad si no se ha calculado ya
                    if vehicle_id not in self.vehicle_speeds:
                        time_diff = abs(time2 - time1)
                        if time_diff > 0:
                            distance_m = line.distance_to_next
                            speed_ms = distance_m / time_diff
                            speed_kmh = speed_ms * 3.6
                            
                            self.vehicle_speeds[vehicle_id] = speed_kmh
                            
                            return {
                                'vehicle_id': vehicle_id,
                                'speed_kmh': speed_kmh,
                                'distance_m': distance_m,
                                'time_diff': time_diff,
                                'lane': line.lane
                            }
        
        return None
    
    def _point_crosses_line(self, point: Tuple[float, float], 
                          line_points: List[Tuple[int, int]]) -> bool:
        """Verificar si punto cruza línea"""
        x, y = point
        x1, y1 = line_points[0]
        x2, y2 = line_points[1]
        
        # Calcular distancia del punto a la línea
        line_length = math.sqrt((x2 - x1)**2 + (y2 - y1)**2)
        if line_length == 0:
            return False
        
        distance = abs((y2 - y1) * x - (x2 - x1) * y + x2 * y1 - y2 * x1) / line_length
        
        # Considerar cruce si está muy cerca de la línea (menos de 10 píxeles)
        return distance < 10
    
    def _point_in_red_zones(self, point: Tuple[float, float]) -> bool:
        """Verificar si punto está en zona roja"""
        for zone in self.zones:
            if zone.zone_type == "red_light":
                if self._point_in_polygon(point, zone.points):
                    return True
        return False
    
    def _point_in_polygon(self, point: Tuple[float, float], 
                         polygon: List[Tuple[int, int]]) -> bool:
        """Verificar si punto está dentro de polígono"""
        x, y = point
        n = len(polygon)
        inside = False
        
        p1x, p1y = polygon[0]
        for i in range(1, n + 1):
            p2x, p2y = polygon[i % n]
            if y > min(p1y, p2y):
                if y <= max(p1y, p2y):
                    if x <= max(p1x, p2x):
                        if p1y != p2y:
                            xinters = (y - p1y) * (p2x - p1x) / (p2y - p1y) + p1x
                        if p1x == p2x or x <= xinters:
                            inside = not inside
            p1x, p1y = p2x, p2y
        
        return inside
    
    def draw_analysis_overlay(self, frame: np.ndarray, tracks: List) -> np.ndarray:
        """Dibujar overlay de análisis en frame"""
        overlay = frame.copy()
        
        # Dibujar líneas
        for line in self.lines:
            color = (0, 255, 0) if line.line_type == LineType.COUNTING else (0, 255, 255)
            cv2.line(overlay, line.points[0], line.points[1], color, 3)
            
            # Etiqueta de línea
            mid_point = ((line.points[0][0] + line.points[1][0]) // 2,
                        (line.points[0][1] + line.points[1][1]) // 2)
            cv2.putText(overlay, f"{line.name} ({line.lane})", 
                       (mid_point[0], mid_point[1] - 10),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
        
        # Dibujar zonas
        for zone in self.zones:
            if zone.zone_type == "red_light":
                color = (0, 0, 255) if self.red_light_active else (100, 100, 100)
                points = np.array(zone.points, np.int32)
                cv2.fillPoly(overlay, [points], color + (50,))  # Semi-transparente
                cv2.polylines(overlay, [points], True, color, 2)
        
        # Dibujar tracks
        for track in tracks:
            x, y, w, h = [int(v) for v in track.bbox]
            
            # Color según si está en zona roja
            in_red_zone = track.track_id in self.vehicles_in_red_zone
            color = (0, 0, 255) if in_red_zone else (0, 255, 0)
            
            # Bounding box
            cv2.rectangle(overlay, (x, y), (x + w, y + h), color, 2)
            
            # ID y velocidad
            label = f"ID:{track.track_id}"
            if track.track_id in self.vehicle_speeds:
                speed = self.vehicle_speeds[track.track_id]
                label += f" {speed:.1f}km/h"
            
            cv2.putText(overlay, label, (x, y - 10),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
        
        # Información de estado
        status_y = 30
        if self.red_light_active:
            cv2.putText(overlay, "SEMAFORO: ROJO", (10, status_y),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
            status_y += 30
            cv2.putText(overlay, f"Vehiculos en zona: {len(self.vehicles_in_red_zone)}", 
                       (10, status_y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
        else:
            cv2.putText(overlay, "SEMAFORO: VERDE/AMARILLO", (10, status_y),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
        
        return overlay