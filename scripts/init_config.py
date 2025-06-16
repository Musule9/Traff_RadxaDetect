#!/usr/bin/env python3
"""
Script de inicialización del sistema de detección vehicular
Crea configuraciones por defecto y verifica el entorno
"""

import os
import json
import sqlite3
from pathlib import Path
from datetime import datetime

def create_directories():
    """Crear estructura de directorios necesarios"""
    dirs = [
        '/app/data',
        '/app/config', 
        '/app/models',
        '/app/logs',
        f'/app/data/{datetime.now().year}',
        f'/app/data/{datetime.now().year}/{datetime.now().month:02d}'
    ]
    
    for dir_path in dirs:
        Path(dir_path).mkdir(parents=True, exist_ok=True)
    
    print("✅ Directorios creados/verificados")

def create_default_configs():
    """Crear configuraciones por defecto"""
    
    # Configuración de análisis
    analysis_config = {
        "lines": {},
        "zones": {}
    }
    
    # Configuración de cámaras
    camera_config = {
        "camera_1": {
            "id": "camera_1",
            "name": "Cámara Principal",
            "rtsp_url": "",
            "fase": "fase1",
            "direccion": "norte",
            "controladora_id": "CTRL_001",
            "controladora_ip": "192.168.1.200",
            "enabled": False
        }
    }
    
    # Configuración de controladoras
    controller_config = {
        "controllers": {
            "CTRL_001": {
                "id": "CTRL_001",
                "name": "Controladora Principal",
                "network": {
                    "ip": "192.168.1.200",
                    "port": 8080
                },
                "endpoints": {
                    "analytic": "/api/analitico",
                    "status": "/api/analiticos"
                }
            }
        }
    }
    
    # Configuración del sistema
    system_config = {
        "system": {
            "name": "Sistema de Detección Vehicular",
            "version": "1.0.0",
            "hardware": "Radxa Rock 5T",
            "max_cameras": 4,
            "data_retention_days": 30,
            "processing": {
                "target_fps": 30,
                "detection_confidence": 0.5,
                "tracking_threshold": 0.3,
                "use_rknn": True
            },
            "storage": {
                "db_path": "/app/data",
                "log_path": "/app/logs",
                "model_path": "/app/models"
            }
        }
    }
    
    configs = [
        ('/app/config/analysis.json', analysis_config),
        ('/app/config/cameras.json', camera_config),
        ('/app/config/controllers.json', controller_config),
        ('/app/config/system.json', system_config)
    ]
    
    for file_path, config in configs:
        if not os.path.exists(file_path):
            with open(file_path, 'w') as f:
                json.dump(config, f, indent=2)
            print(f"✅ Creado: {os.path.basename(file_path)}")
        else:
            print(f"🔄 Existe: {os.path.basename(file_path)}")

def init_database():
    """Inicializar base de datos SQLite con esquema robusto"""
    today = datetime.now()
    db_dir = f"/app/data/{today.year}/{today.month:02d}"
    db_path = f"{db_dir}/{today.year}_{today.month:02d}_{today.day:02d}.db"
    
    try:
        # Crear conexión a la base de datos
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()
        
        # Verificar si ya existen tablas
        cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
        existing_tables = [row[0] for row in cursor.fetchall()]
        
        if existing_tables:
            print(f"✅ Base de datos existente con {len(existing_tables)} tablas")
            conn.close()
            return
        
        # Crear esquema completo y robusto
        
        # Tabla principal de cruces de vehículos
        cursor.execute('''
        CREATE TABLE IF NOT EXISTS vehicle_crossings (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp TEXT NOT NULL,
            date TEXT NOT NULL,
            time TEXT NOT NULL,
            camera_id TEXT NOT NULL,
            vehicle_id TEXT NOT NULL,
            track_id INTEGER,
            vehicle_type TEXT DEFAULT 'unknown',
            vehicle_class TEXT DEFAULT 'vehicle',
            lane TEXT,
            direction TEXT,
            speed REAL DEFAULT 0.0,
            confidence REAL DEFAULT 0.0,
            crossing_point TEXT,
            line_name TEXT,
            bbox_x1 REAL,
            bbox_y1 REAL,
            bbox_x2 REAL,
            bbox_y2 REAL,
            fase TEXT,
            controladora_id TEXT,
            metadata TEXT,
            created_at TEXT DEFAULT CURRENT_TIMESTAMP
        )
        ''')
        
        # Índices para cruces
        cursor.execute('CREATE INDEX IF NOT EXISTS idx_crossings_timestamp ON vehicle_crossings(timestamp);')
        cursor.execute('CREATE INDEX IF NOT EXISTS idx_crossings_date ON vehicle_crossings(date);')
        cursor.execute('CREATE INDEX IF NOT EXISTS idx_crossings_camera ON vehicle_crossings(camera_id);')
        cursor.execute('CREATE INDEX IF NOT EXISTS idx_crossings_vehicle ON vehicle_crossings(vehicle_id);')
        
        # Tabla de detecciones en zona roja
        cursor.execute('''
        CREATE TABLE IF NOT EXISTS red_zone_detections (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp TEXT NOT NULL,
            date TEXT NOT NULL,
            time TEXT NOT NULL,
            camera_id TEXT NOT NULL,
            vehicle_id TEXT NOT NULL,
            track_id INTEGER,
            vehicle_type TEXT DEFAULT 'unknown',
            duration REAL DEFAULT 0.0,
            max_duration REAL DEFAULT 0.0,
            confidence REAL DEFAULT 0.0,
            zone_name TEXT,
            zone_id TEXT,
            bbox_x1 REAL,
            bbox_y1 REAL,
            bbox_x2 REAL,
            bbox_y2 REAL,
            violation_severity TEXT DEFAULT 'medium',
            fase TEXT,
            controladora_id TEXT,
            metadata TEXT,
            created_at TEXT DEFAULT CURRENT_TIMESTAMP
        )
        ''')
        
        # Índices para zonas rojas
        cursor.execute('CREATE INDEX IF NOT EXISTS idx_redzone_timestamp ON red_zone_detections(timestamp);')
        cursor.execute('CREATE INDEX IF NOT EXISTS idx_redzone_date ON red_zone_detections(date);')
        cursor.execute('CREATE INDEX IF NOT EXISTS idx_redzone_camera ON red_zone_detections(camera_id);')
        
        # Tabla de estadísticas diarias
        cursor.execute('''
        CREATE TABLE IF NOT EXISTS daily_stats (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            date TEXT UNIQUE NOT NULL,
            camera_id TEXT NOT NULL,
            total_vehicles INTEGER DEFAULT 0,
            total_cars INTEGER DEFAULT 0,
            total_trucks INTEGER DEFAULT 0,
            total_motorcycles INTEGER DEFAULT 0,
            total_buses INTEGER DEFAULT 0,
            avg_speed REAL DEFAULT 0.0,
            max_speed REAL DEFAULT 0.0,
            min_speed REAL DEFAULT 0.0,
            red_zone_violations INTEGER DEFAULT 0,
            peak_hour TEXT,
            peak_count INTEGER DEFAULT 0,
            total_processing_time REAL DEFAULT 0.0,
            frames_processed INTEGER DEFAULT 0,
            detection_accuracy REAL DEFAULT 0.0,
            fase TEXT,
            controladora_id TEXT,
            metadata TEXT,
            created_at TEXT DEFAULT CURRENT_TIMESTAMP,
            updated_at TEXT DEFAULT CURRENT_TIMESTAMP
        )
        ''')
        
        # Tabla de configuración del sistema
        cursor.execute('''
        CREATE TABLE IF NOT EXISTS system_config (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            key TEXT UNIQUE NOT NULL,
            value TEXT NOT NULL,
            description TEXT,
            category TEXT DEFAULT 'general',
            data_type TEXT DEFAULT 'string',
            created_at TEXT DEFAULT CURRENT_TIMESTAMP,
            updated_at TEXT DEFAULT CURRENT_TIMESTAMP
        )
        ''')
        
        # Tabla de logs del sistema
        cursor.execute('''
        CREATE TABLE IF NOT EXISTS system_logs (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp TEXT NOT NULL,
            level TEXT NOT NULL,
            component TEXT NOT NULL,
            message TEXT NOT NULL,
            details TEXT,
            camera_id TEXT,
            created_at TEXT DEFAULT CURRENT_TIMESTAMP
        )
        ''')
        
        # Índice para logs
        cursor.execute('CREATE INDEX IF NOT EXISTS idx_logs_timestamp ON system_logs(timestamp);')
        cursor.execute('CREATE INDEX IF NOT EXISTS idx_logs_level ON system_logs(level);')
        
        # Insertar configuraciones iniciales
        initial_configs = [
            ('system_version', '1.0.0', 'Versión del sistema', 'system', 'string'),
            ('hardware_platform', 'radxa-rock-5t', 'Plataforma de hardware', 'system', 'string'),
            ('detection_confidence', '0.5', 'Confianza mínima para detecciones', 'detection', 'float'),
            ('tracking_threshold', '0.3', 'Umbral para seguimiento', 'tracking', 'float'),
            ('red_zone_duration_threshold', '3.0', 'Duración mínima en zona roja', 'zones', 'float'),
            ('speed_calculation_enabled', 'true', 'Habilitar cálculo de velocidad', 'processing', 'boolean'),
            ('data_retention_days', '30', 'Días de retención de datos', 'storage', 'integer')
        ]
        
        for key, value, desc, category, data_type in initial_configs:
            cursor.execute('''
                INSERT OR IGNORE INTO system_config (key, value, description, category, data_type)
                VALUES (?, ?, ?, ?, ?)
            ''', (key, value, desc, category, data_type))
        
        # Log inicial del sistema
        cursor.execute('''
            INSERT INTO system_logs (timestamp, level, component, message)
            VALUES (?, 'INFO', 'database', 'Database initialized successfully')
        ''', (datetime.now().isoformat(),))
        
        conn.commit()
        conn.close()
        
        print(f"✅ Base de datos inicializada con esquema completo: {os.path.basename(db_path)}")
        
    except Exception as e:
        print(f"❌ Error inicializando base de datos: {e}")
        # Crear base de datos mínima como fallback
        try:
            conn = sqlite3.connect(db_path)
            cursor = conn.cursor()
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS vehicle_crossings (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp TEXT NOT NULL,
                    camera_id TEXT NOT NULL,
                    vehicle_id TEXT NOT NULL,
                    metadata TEXT
                )
            ''')
            conn.commit()
            conn.close()
            print(f"✅ Base de datos mínima creada como fallback")
        except Exception as fallback_error:
            print(f"❌ Error crítico con base de datos: {fallback_error}")

def verify_environment():
    """Verificar entorno y dependencias"""
    print("🔍 Verificando entorno...")
    
    # Verificar Python
    import sys
    print(f"Python: {sys.version}")
    
    # Verificar módulos críticos
    modules = [
        'fastapi', 'uvicorn', 'opencv-python', 'numpy', 
        'ultralytics', 'torch', 'sqlite3', 'json'
    ]
    
    available_modules = []
    missing_modules = []
    
    for module in modules:
        try:
            __import__(module.replace('-', '_'))
            available_modules.append(module)
        except ImportError:
            missing_modules.append(module)
    
    print(f"✅ Módulos disponibles: {len(available_modules)}")
    if missing_modules:
        print(f"⚠️ Módulos faltantes: {missing_modules}")
    
    # Verificar estructura de la aplicación
    app_files = [
        '/app/main.py',
        '/app/app/__init__.py',
        '/app/app/core/__init__.py',
        '/app/app/services/__init__.py'
    ]
    
    for file_path in app_files:
        if os.path.exists(file_path):
            print(f"✅ {file_path}")
        else:
            print(f"❌ {file_path}")

def main():
    """Función principal de inicialización"""
    print("🚀 Inicializando sistema de detección vehicular...")
    print("=" * 50)
    
    try:
        create_directories()
        create_default_configs()
        init_database()
        verify_environment()
        
        print("=" * 50)
        print("✅ Inicialización completada exitosamente")
        
    except Exception as e:
        print(f"❌ Error durante la inicialización: {e}")
        return 1
    
    return 0

if __name__ == "__main__":
    exit(main())