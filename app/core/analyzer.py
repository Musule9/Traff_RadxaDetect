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
    distance_to_next: Optional[float] = None  # metros (para líneas de conteo)
    speed_line_id: Optional[str] = None  # ID de la línea de velocidad asociada
    counting_line_id: Optional[str] = None  # ID de la línea de conteo asociada (para líneas de velocidad)
    speed_line_distance: Optional[float] = None  # Distancia específica para cálculo de velocidad
    direction: Optional[str] = None  # Dirección del flujo

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
        self.vehicle_lanes = {}  # FALTA
        self.vehicle_last_line = {}  # FALTA
        self.red_light_active = False
        self.vehicles_in_red_zone = set()
        self.red_light_start_time = None
        self.red_light_vehicles_start = 0
        self.analytic_sent_this_cycle = False

    def cleanup_old_vehicles(self, max_age_seconds: int = 100):
        """Limpiar vehículos antiguos para evitar memory leak - VERSIÓN CORREGIDA"""
        current_time = time.time()
        vehicles_to_remove = []
        
        for vehicle_id, crossings in self.vehicle_line_crossings.items():
            if crossings:
                last_crossing = max(crossings.values())
                if current_time - last_crossing > max_age_seconds:
                    vehicles_to_remove.append(vehicle_id)
        
        for vehicle_id in vehicles_to_remove:
            self.vehicle_line_crossings.pop(vehicle_id, None)
            # ✅ VERIFICAR EXISTENCIA ANTES DE ELIMINAR:
            if hasattr(self, 'vehicle_lanes'):
                self.vehicle_lanes.pop(vehicle_id, None)
            if hasattr(self, 'vehicle_speeds'):
                self.vehicle_speeds.pop(vehicle_id, None)
            if hasattr(self, 'vehicle_last_line'):
                self.vehicle_last_line.pop(vehicle_id, None)

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
        
        # Limpiar vehículos antiguos cada 100 análisis
        if hasattr(self, '_frame_count'):
            self._frame_count += 1
        else:
            self._frame_count = 0

        if self._frame_count % 100 == 0:
            self.cleanup_old_vehicles()

        return results
    
    def _update_vehicle_lane(self, vehicle_id: int, line_id: str):
        """Actualizar carril del vehículo basado en línea cruzada"""
        if vehicle_id not in self.vehicle_lanes:
            # Encontrar la línea y asignar su carril
            for line in self.lines:
                if line.id == line_id:
                    self.vehicle_lanes[vehicle_id] = line.lane
                    break

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
                    
                    self._update_vehicle_lane(vehicle_id, line_id)

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
        """Calcular velocidad del vehículo entre líneas del mismo carril"""
        if vehicle_id not in self.vehicle_line_crossings:
            return None
        
        crossings = self.vehicle_line_crossings[vehicle_id]
        if len(crossings) < 2:
            return None
        
        # Obtener el carril actual del vehículo
        current_lane = self.vehicle_lanes.get(vehicle_id)
        if not current_lane:
            return None
        
        # Buscar pares de líneas en el mismo carril
        lane_lines = [l for l in self.lines if l.lane == current_lane]
        
        for i, line1 in enumerate(lane_lines):
            if line1.id not in crossings:
                continue
                
            for line2 in lane_lines[i+1:]:
                if line2.id not in crossings:
                    continue
                
                # Verificar que no hayamos calculado ya esta velocidad
                speed_key = f"{vehicle_id}_{line1.id}_{line2.id}"
                if speed_key in self.vehicle_speeds:
                    continue
                
                time1 = crossings[line1.id]
                time2 = crossings[line2.id]
                time_diff = abs(time2 - time1)
                
                if time_diff < 0.5 or time_diff > 30:  # Validar tiempo razonable
                    continue
                
                # Calcular distancia
                distance_m = None
                if line1.line_type == LineType.SPEED and line1.distance_to_next:
                    distance_m = line1.distance_to_next
                elif line2.line_type == LineType.SPEED and line2.distance_to_next:
                    distance_m = line2.distance_to_next
                else:
                    # Estimar distancia por defecto
                    distance_m = 10.0  # 10 metros por defecto
                
                if distance_m and distance_m > 0:
                    speed_ms = distance_m / time_diff
                    speed_kmh = speed_ms * 3.6
                    
                    # Validar velocidad razonable (5-150 km/h)
                    if 5 <= speed_kmh <= 150:
                        self.vehicle_speeds[speed_key] = speed_kmh
                        
                        return {
                            'vehicle_id': vehicle_id,
                            'speed_kmh': round(speed_kmh, 1),
                            'distance_m': distance_m,
                            'time_diff': round(time_diff, 2),
                            'lane': current_lane,
                            'line1_id': line1.id,
                            'line2_id': line2.id,
                            'line1_name': line1.name,
                            'line2_name': line2.name
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