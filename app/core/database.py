import aiosqlite
import os
from datetime import datetime, timedelta
from typing import List, Dict, Optional
import json
import asyncio
from loguru import logger

class DatabaseManager:
    """Gestor de base de datos SQLite con organización diaria"""
    
    def __init__(self, data_path: str = "/app/data", retention_days: int = 30):
        self.data_path = data_path
        self.retention_days = retention_days
        os.makedirs(data_path, exist_ok=True)
    
    def get_db_path(self, date: datetime = None) -> str:
        """Obtener ruta de base de datos para fecha específica"""
        if date is None:
            date = datetime.now()
        
        year_month = date.strftime("%Y/%m")
        db_dir = os.path.join(self.data_path, year_month)
        os.makedirs(db_dir, exist_ok=True)
        
        db_file = date.strftime("%Y_%m_%d.db")
        return os.path.join(db_dir, db_file)
    
    async def init_daily_database(self, date: datetime = None):
        """Inicializar base de datos del día"""
        db_path = self.get_db_path(date)
        
        async with aiosqlite.connect(db_path) as db:
            # Configurar WAL mode para mejor rendimiento
            await db.execute("PRAGMA journal_mode=WAL")
            await db.execute("PRAGMA synchronous=NORMAL")
            await db.execute("PRAGMA cache_size=10000")
            
            # Crear tabla de cruces de vehículos
            await db.execute("""
                CREATE TABLE IF NOT EXISTS vehicle_crossings (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    vehicle_id INTEGER NOT NULL,
                    line_id TEXT NOT NULL,
                    line_name TEXT NOT NULL,
                    fase TEXT NOT NULL,
                    semaforo_estado TEXT NOT NULL,
                    timestamp DATETIME DEFAULT (datetime('now','localtime')),
                    velocidad REAL,
                    direccion TEXT,
                    No_Controladora TEXT,
                    confianza REAL,
                    carril TEXT,
                    clase_vehiculo INTEGER,
                    bbox_x INTEGER,
                    bbox_y INTEGER,
                    bbox_w INTEGER,
                    bbox_h INTEGER,
                    metadata TEXT
                )
            """)
            
            # Crear tabla de conteos en zona roja
            await db.execute("""
                CREATE TABLE IF NOT EXISTS red_light_counts (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    fase TEXT NOT NULL,
                    inicio_rojo DATETIME NOT NULL,
                    fin_rojo DATETIME,
                    vehiculos_inicio INTEGER DEFAULT 0,
                    vehiculos_final INTEGER DEFAULT 0,
                    vehiculos_cruzaron INTEGER DEFAULT 0,
                    duracion_segundos INTEGER,
                    direccion TEXT,
                    No_Controladora TEXT,
                    analitico_enviado BOOLEAN DEFAULT 0,
                    analitico_recibido BOOLEAN DEFAULT 0
                )
            """)
            
            # Crear índices para mejor rendimiento
            await db.execute("CREATE INDEX IF NOT EXISTS idx_vehicle_crossings_timestamp ON vehicle_crossings(timestamp)")
            await db.execute("CREATE INDEX IF NOT EXISTS idx_vehicle_crossings_vehicle_id ON vehicle_crossings(vehicle_id)")
            await db.execute("CREATE INDEX IF NOT EXISTS idx_red_light_counts_inicio ON red_light_counts(inicio_rojo)")
            
            await db.commit()
    
    async def insert_vehicle_crossing(self, crossing_data: Dict):
        """Insertar cruce de vehículo"""
        db_path = self.get_db_path()
        
        async with aiosqlite.connect(db_path) as db:
            await db.execute("""
                INSERT INTO vehicle_crossings (
                    vehicle_id, line_id, line_name, fase, semaforo_estado,
                    velocidad, direccion, No_Controladora, confianza, carril,
                    clase_vehiculo, bbox_x, bbox_y, bbox_w, bbox_h, metadata
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                crossing_data.get('vehicle_id'),
                crossing_data.get('line_id'),
                crossing_data.get('line_name'),
                crossing_data.get('fase'),
                crossing_data.get('semaforo_estado'),
                crossing_data.get('velocidad'),
                crossing_data.get('direccion'),
                crossing_data.get('No_Controladora'),
                crossing_data.get('confianza'),
                crossing_data.get('carril'),
                crossing_data.get('clase_vehiculo'),
                crossing_data.get('bbox_x'),
                crossing_data.get('bbox_y'),
                crossing_data.get('bbox_w'),
                crossing_data.get('bbox_h'),
                json.dumps(crossing_data.get('metadata', {}))
            ))
            await db.commit()
    
    async def insert_red_light_cycle(self, cycle_data: Dict):
        """Insertar ciclo de semáforo en rojo"""
        db_path = self.get_db_path()
        
        async with aiosqlite.connect(db_path) as db:
            await db.execute("""
                INSERT INTO red_light_counts (
                    fase, inicio_rojo, fin_rojo, vehiculos_inicio, vehiculos_final,
                    vehiculos_cruzaron, duracion_segundos, direccion, No_Controladora,
                    analitico_enviado, analitico_recibido
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                cycle_data.get('fase'),
                cycle_data.get('inicio_rojo'),
                cycle_data.get('fin_rojo'),
                cycle_data.get('vehiculos_inicio'),
                cycle_data.get('vehiculos_final'),
                cycle_data.get('vehiculos_cruzaron'),
                cycle_data.get('duracion_segundos'),
                cycle_data.get('direccion'),
                cycle_data.get('No_Controladora'),
                cycle_data.get('analitico_enviado', False),
                cycle_data.get('analitico_recibido', False)
            ))
            await db.commit()
    
    async def export_vehicle_crossings(self, date: str, fase: str = None) -> List[Dict]:
        """Exportar cruces de vehículos de una fecha"""
        try:
            export_date = datetime.strptime(date, "%Y_%m_%d")
            db_path = self.get_db_path(export_date)
            
            if not os.path.exists(db_path):
                return []
            
            query = "SELECT * FROM vehicle_crossings"
            params = []
            
            if fase:
                query += " WHERE fase = ?"
                params.append(fase)
            
            query += " ORDER BY timestamp"
            
            async with aiosqlite.connect(db_path) as db:
                db.row_factory = aiosqlite.Row
                async with db.execute(query, params) as cursor:
                    rows = await cursor.fetchall()
                    return [dict(row) for row in rows]
        
        except Exception as e:
            logger.error(f"Error exportando cruces: {e}")
            return []
    
    async def export_red_light_counts(self, date: str, fase: str = None) -> List[Dict]:
        """Exportar conteos de zona roja de una fecha"""
        try:
            export_date = datetime.strptime(date, "%Y_%m_%d")
            db_path = self.get_db_path(export_date)
            
            if not os.path.exists(db_path):
                return []
            
            query = "SELECT * FROM red_light_counts"
            params = []
            
            if fase:
                query += " WHERE fase = ?"
                params.append(fase)
            
            query += " ORDER BY inicio_rojo"
            
            async with aiosqlite.connect(db_path) as db:
                db.row_factory = aiosqlite.Row
                async with db.execute(query, params) as cursor:
                    rows = await cursor.fetchall()
                    return [dict(row) for row in rows]
        
        except Exception as e:
            logger.error(f"Error exportando zona roja: {e}")
            return []
    
    async def cleanup_old_databases(self):
        """Limpiar bases de datos antigas"""
        try:
            cutoff_date = datetime.now() - timedelta(days=self.retention_days)
            
            for root, dirs, files in os.walk(self.data_path):
                for file in files:
                    if file.endswith('.db'):
                        try:
                            # Extraer fecha del nombre del archivo
                            date_str = file.replace('.db', '')
                            file_date = datetime.strptime(date_str, "%Y_%m_%d")
                            
                            if file_date < cutoff_date:
                                file_path = os.path.join(root, file)
                                os.remove(file_path)
                                logger.info(f"Base de datos eliminada: {file_path}")
                        
                        except ValueError:
                            continue  # Nombre de archivo no válido
            
            # Limpiar directorios vacíos
            for root, dirs, files in os.walk(self.data_path, topdown=False):
                if not dirs and not files and root != self.data_path:
                    os.rmdir(root)
        
        except Exception as e:
            logger.error(f"Error limpiando bases de datos: {e}")
