import unittest
from app.core.analyzer import TrafficAnalyzer, Line, Zone, LineType

class TestTrafficAnalyzer(unittest.TestCase):
    """Tests para el analizador de tráfico"""
    
    def setUp(self):
        """Configurar test"""
        self.analyzer = TrafficAnalyzer()
        
        # Agregar línea de prueba
        test_line = Line(
            id="test_line",
            name="Test Line",
            points=[(100, 200), (300, 200)],
            lane="lane1",
            line_type=LineType.COUNTING,
            distance_to_next=10.0
        )
        self.analyzer.add_line(test_line)
        
        # Agregar zona de prueba
        test_zone = Zone(
            id="test_zone",
            name="Test Zone",
            points=[(50, 150), (350, 150), (350, 250), (50, 250)],
            zone_type="red_light"
        )
        self.analyzer.add_zone(test_zone)
    
    def test_add_line(self):
        """Test agregar línea"""
        self.assertEqual(len(self.analyzer.lines), 1)
        self.assertEqual(self.analyzer.lines[0].id, "test_line")
    
    def test_add_zone(self):
        """Test agregar zona"""
        self.assertEqual(len(self.analyzer.zones), 1)
        self.assertEqual(self.analyzer.zones[0].id, "test_zone")
    
    def test_red_light_status_update(self):
        """Test actualización de estado de semáforo"""
        # Cambiar a rojo
        self.analyzer.update_red_light_status(True)
        self.assertTrue(self.analyzer.red_light_active)
        self.assertIsNotNone(self.analyzer.red_light_start_time)
        
        # Cambiar a verde
        self.analyzer.update_red_light_status(False)
        self.assertFalse(self.analyzer.red_light_active)
    
    def test_point_in_polygon(self):
        """Test punto en polígono"""
        # Punto dentro de la zona
        point_inside = (200, 200)
        self.assertTrue(self.analyzer._point_in_polygon(point_inside, self.analyzer.zones[0].points))
        
        # Punto fuera de la zona
        point_outside = (400, 400)
        self.assertFalse(self.analyzer._point_in_polygon(point_outside, self.analyzer.zones[0].points))
    
    def test_point_crosses_line(self):
        """Test cruce de línea"""
        # Punto cerca de la línea
        point_near = (200, 205)
        self.assertTrue(self.analyzer._point_crosses_line(point_near, self.analyzer.lines[0].points))
        
        # Punto lejos de la línea
        point_far = (200, 300)
        self.assertFalse(self.analyzer._point_crosses_line(point_far, self.analyzer.lines[0].points))
