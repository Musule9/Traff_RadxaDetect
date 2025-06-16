# 🚗 Sistema de Detección Vehicular para Radxa Rock 5T

Sistema avanzado de detección y conteo de vehículos optimizado para **Radxa Rock 5T** con soporte para **RKNN** y controladora de semáforos.

## 🎯 Características Principales

- ✅ **Detección en tiempo real** usando YOLOv8n optimizado con RKNN
- ✅ **Tracking persistente** con BYTETracker
- ✅ **Análisis de tráfico** con conteo de líneas y cálculo de velocidad
- ✅ **Zona de semáforo rojo** para analíticos avanzados
- ✅ **Base de datos diaria** con SQLite y retención configurable
- ✅ **API REST** completa con documentación Swagger
- ✅ **Interfaz web** moderna y responsiva
- ✅ **Comunicación con controladora** de semáforos
- ✅ **Docker** para deployment fácil
- ✅ **Autenticación** y seguridad

## 🛠️ Requisitos del Sistema

### Hardware Recomendado
- **Radxa Rock 5T** (o 5B/5A compatible)
- **4GB RAM** mínimo (8GB recomendado)
- **32GB microSD** o eMMC
- **Cámara IP** con stream RTSP
- **Red Ethernet** estable

### Software
- **Ubuntu 22.04** para Radxa
- **Docker** y **Docker Compose**
- **Python 3.9+**
- **Librerías RKNN** (se instalan automáticamente)

## 🚀 Instalación Rápida

### 1. Clonar el Repositorio
```bash
git clone https://github.com/tu-usuario/vehicle-detection-system.git
cd vehicle-detection-system
```

### 2. Ejecutar Instalador Automático
```bash
sudo chmod +x deploy/install_radxa.sh
sudo ./deploy/install_radxa.sh
```

### 3. Configurar el Sistema
```bash
vehicle-detection-setup
```

### 4. Acceder a la Interfaz Web
```
http://IP_DE_TU_RADXA:8000
Usuario: admin
Contraseña: admin123
```

## 📋 Instalación Manual

### 1. Preparar el Sistema
```bash
# Actualizar sistema
sudo apt update && sudo apt upgrade -y

# Instalar Docker
sudo apt install -y docker.io docker-compose
sudo usermod -aG docker $USER

# Crear directorios
sudo mkdir -p /opt/vehicle-detection
sudo chown $USER:$USER /opt/vehicle-detection
```

### 2. Configurar la Aplicación
```bash
cd /opt/vehicle-detection
git clone https://github.com/tu-usuario/vehicle-detection-system.git .

# Construir imagen Docker
docker-compose build

# Iniciar servicios
docker-compose up -d
```

## ⚙️ Configuración

### 1. Configuración de Cámara
En la interfaz web, vaya a **Configuración** y complete:
- **URL RTSP**: `rtsp://admin:password@192.168.1.100:554/stream1`
- **Fase del semáforo**: `fase1`, `fase2`, `fase3`, o `fase4`
- **Dirección**: `norte`, `sur`, `este`, `oeste`
- **IP de controladora**: `192.168.1.200`

### 2. Configuración de Líneas de Conteo
En **Vista de Cámara**:
1. Haga clic en "Dibujar Línea"
2. Trace líneas perpendiculares al flujo vehicular
3. Configure la distancia entre líneas para cálculo de velocidad
4. Guarde la configuración

### 3. Configuración de Zona Roja
1. Haga clic en "Dibujar Zona"
2. Defina el área donde detectar vehículos durante semáforo en rojo
3. Finalice la zona y guarde

## 🔧 Comandos Útiles

```bash
# Controlar el servicio
vehicle-detection-ctl start     # Iniciar
vehicle-detection-ctl stop      # Detener
vehicle-detection-ctl restart   # Reiniciar
vehicle-detection-ctl status    # Estado
vehicle-detection-ctl logs      # Ver logs

# Mantenimiento
vehicle-detection-ctl backup    # Crear respaldo
vehicle-detection-ctl cleanup   # Limpiar datos antiguos
vehicle-detection-ctl update    # Actualizar sistema
```

## 📊 API REST

### Endpoints Principales

#### Autenticación
```bash
POST /api/auth/login
POST /api/auth/logout
```

#### Cámara
```bash
GET  /api/camera/status
POST /api/camera/config
GET  /api/camera/stream
GET  /api/camera_health
```

#### Análisis
```bash
POST /api/analysis/lines
POST /api/analysis/zones
```

#### Datos
```bash
GET  /api/data/export?date=2024_06_15&type=vehicle
```

#### Controladora
```bash
POST /api/rojo_status
GET  /api/rojo_status
POST /api/analitico_camara
```

### Documentación Swagger
Acceda a la documentación completa en: `http://IP_RADXA:8000/docs`

## 🗄️ Estructura de Base de Datos

### Tabla: vehicle_crossings
```sql
CREATE TABLE vehicle_crossings (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    vehicle_id INTEGER NOT NULL,
    line_id TEXT NOT NULL,
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
);
```

### Tabla: red_light_counts
```sql
CREATE TABLE red_light_counts (
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
);
```

## 🔒 Seguridad

### Autenticación
- **JWT Tokens** con expiración configurable
- **Contraseñas encriptadas** con bcrypt
- **Sesiones seguras** con revocación

### Firewall
```bash
# Configuración automática durante instalación
sudo ufw enable
sudo ufw allow ssh
sudo ufw allow 8000/tcp
```

### Fail2ban
Protección automática contra ataques de fuerza bruta en SSH.

## 📈 Monitoreo y Logs

### Ubicaciones de Logs
- **Aplicación**: `/opt/vehicle-detection/logs/`
- **Sistema**: `journalctl -u vehicle-detection`
- **Docker**: `docker-compose logs`

### Métricas
- **FPS de procesamiento**
- **Estado de cámara**
- **Conteo de vehículos**
- **Estado de controladora**

## 🧪 Testing

### Ejecutar Tests
```bash
# Tests unitarios
cd /opt/vehicle-detection
python -m pytest tests/ -v

# Test manual de componentes
python tests/run_tests.py
```

### Simulador de Controladora
```bash
# Iniciar simulador para testing
docker-compose --profile testing up mock-controller
```

## 🔧 Desarrollo

### Entorno de Desarrollo
```bash
# Iniciar en modo desarrollo
docker-compose --profile development up vehicle-detection-dev

# Acceder a Jupyter (opcional)
# http://IP_RADXA:8888
```

### Estructura del Proyecto
```
vehicle-detection-system/
├── app/                    # Backend Python
│   ├── core/              # Módulos principales
│   ├── services/          # Servicios
│   └── api/               # Rutas API
├── frontend/              # Frontend React
├── config/                # Configuraciones
├── deploy/                # Scripts de deployment
├── tests/                 # Tests unitarios
└── docker-compose.yml     # Orquestación
```

## 📝 Troubleshooting

### Problemas Comunes

#### 1. Cámara no se conecta
```bash
# Verificar URL RTSP
ffmpeg -i "rtsp://admin:password@IP:554/stream1" -t 10 -f null -

# Verificar red
ping IP_DE_CAMARA
```

#### 2. Bajo rendimiento
```bash
# Verificar uso de NPU
dmesg | grep -i rknn

# Monitorear recursos
htop
iotop
```

#### 3. Base de datos corrupta
```bash
# Reparar base de datos
vehicle-detection-ctl stop
sqlite3 /opt/vehicle-detection/data/YYYY/MM/archivo.db "PRAGMA integrity_check;"
vehicle-detection-ctl start
```

### Logs de Depuración
```bash
# Habilitar debug
export LOG_LEVEL=DEBUG
vehicle-detection-ctl restart

# Ver logs en tiempo real
vehicle-detection-ctl logs
```

## 🤝 Contribución

### Cómo Contribuir
1. Fork el repositorio
2. Cree una rama feature (`git checkout -b feature/nueva-funcion`)
3. Commit sus cambios (`git commit -am 'Agregar nueva función'`)
4. Push a la rama (`git push origin feature/nueva-funcion`)
5. Cree un Pull Request

### Estándares de Código
- **PEP 8** para Python
- **ESLint** para JavaScript
- **Docstrings** en todas las funciones
- **Tests unitarios** para nuevas funciones

## 📄 Licencia

Este proyecto está licenciado bajo la Licencia MIT - vea el archivo [LICENSE](LICENSE) para detalles.

## 🙏 Agradecimientos

- **Ultralytics** por YOLOv8
- **Radxa** por el hardware y herramientas
- **Comunidad OpenCV** por las librerías de visión computacional
- **FastAPI** por el framework web

## 📞 Soporte

- **Issues**: [GitHub Issues](https://github.com/tu-usuario/vehicle-detection-system/issues)
- **Documentación**: [Wiki del Proyecto](https://github.com/tu-usuario/vehicle-detection-system/wiki)
- **Email**: soporte@tu-dominio.com

---

<div align="center">

**🚗 Sistema de Detección Vehicular para Radxa Rock 5T**

*Hecho con ❤️ para la comunidad de tráfico inteligente*

[🏠 Inicio](/) • [📖 Docs](https://github.com/tu-usuario/vehicle-detection-system/wiki) • [🐛 Issues](https://github.com/tu-usuario/vehicle-detection-system/issues) • [💬 Discusiones](https://github.com/tu-usuario/vehicle-detection-system/discussions)

</div>
