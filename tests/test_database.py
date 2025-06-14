import unittest
import asyncio
import os
import tempfile
import shutil
from datetime import datetime
from app.core.database import DatabaseManager

class TestDatabaseManager(unittest.TestCase):
    """Tests para el gestor de base de datos"""
    
    def setUp(self):
        """Configurar test"""
        self.temp_dir = tempfile.mkdtemp()
        self.db_manager = DatabaseManager(data_path=self.temp_dir, retention_days=30)
    
    def tearDown(self):
        """Limpiar test"""
        shutil.rmtree(self.temp_dir)
    
    def test_get_db_path(self):
        """Test obtener ruta de base de datos"""
        test_date = datetime(2024, 6, 15)
        db_path = self.db_manager.get_db_path(test_date)
        
        expected_path = os.path.join(self.temp_dir, "2024", "06", "2024_06_15.db")
        self.assertEqual(db_path, expected_path)
    
    async def test_init_daily_database(self):
        """Test inicialización de base de datos diaria"""
        await self.db_manager.init_daily_database()
        
        db_path = self.db_manager.get_db_path()
        self.assertTrue(os.path.exists(db_path))
    
    async def test_insert_vehicle_crossing(self):
        """Test insertar cruce de vehículo"""
        await self.db_manager.init_daily_database()
        
        crossing_data = {
            'vehicle_id': 1,
            'line_id': 'test_line',
            'line_name': 'Test Line',
            'fase': 'fase1',
            'semaforo_estado': 'verde',
            'velocidad': 50.0,
            'direccion': 'norte',
            'No_Controladora': 'CTRL_001',
            'confianza': 0.8,
            'carril': 'carril_1',
            'clase_vehiculo': 2,
            'bbox_x': 100,
            'bbox_y': 100,
            'bbox_w': 50,
            'bbox_h': 50,
            'metadata': {}
        }
        
        await self.db_manager.insert_vehicle_crossing(crossing_data)
        
        # Verificar que se insertó
        data = await self.db_manager.export_vehicle_crossings(datetime.now().strftime("%Y_%m_%d"))
        self.assertEqual(len(data), 1)
        self.assertEqual(data[0]['vehicle_id'], 1)
