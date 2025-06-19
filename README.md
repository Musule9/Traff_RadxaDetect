# 🚗 Sistema de Detección Vehicular para Radxa Rock 5T

<div align="center">

![Version](https://img.shields.io/badge/version-1.0.0-blue.svg)
![Python](https://img.shields.io/badge/python-3.9%2B-blue.svg)
![License](https://img.shields.io/badge/license-MIT-green.svg)
![Platform](https://img.shields.io/badge/platform-Radxa%20Rock%205T-orange.svg)

**Sistema avanzado de detección y análisis de tráfico vehicular con IA optimizado para Radxa Rock 5T**

[📚 Documentación](#documentación) • [🚀 Instalación](#instalación) • [⚙️ Configuración](#configuración) • [🔧 API](#api-rest) • [🐛 Solución de Problemas](#troubleshooting)

</div>

---

## 📋 Tabla de Contenidos

- [🎯 Características Principales](#-características-principales)
- [🏗️ Arquitectura del Sistema](#️-arquitectura-del-sistema)
- [🛠️ Requisitos del Sistema](#️-requisitos-del-sistema)
- [🚀 Instalación](#-instalación)
- [⚙️ Configuración](#️-configuración)
- [📊 Uso del Sistema](#-uso-del-sistema)
- [🔧 API REST](#-api-rest)
- [📁 Estructura del Proyecto](#-estructura-del-proyecto)
- [🗄️ Base de Datos](#️-base-de-datos)
- [🔒 Seguridad](#-seguridad)
- [📈 Monitoreo](#-monitoreo)
- [🧪 Testing](#-testing)
- [🚨 Errores Conocidos](#-errores-conocidos)
- [🐛 Troubleshooting](#-troubleshooting)
- [🤝 Contribución](#-contribución)

---

## 🎯 Características Principales

### 🔥 Detección y Análisis Avanzado
- ✅ **Detección en tiempo real** usando YOLOv8n optimizado con RKNN NPU
- ✅ **Tracking persistente** con algoritmo BYTETracker
- ✅ **Análisis de velocidad** entre líneas de conteo configurables
- ✅ **Conteo multi-carril** con líneas personalizables
- ✅ **Zona de semáforo rojo** para detectar infracciones
- ✅ **Clasificación de vehículos** (auto, motocicleta, bus, camión)

### 🎛️ Gestión y Control
- ✅ **Interfaz web moderna** y responsiva
- ✅ **API REST completa** con documentación Swagger
- ✅ **Autenticación JWT** con sesiones seguras
- ✅ **Base de datos diaria** con retención configurable
- ✅ **Comunicación con controladora** de semáforos TICSA
- ✅ **Exportación de datos** en múltiples formatos

### 🚀 Optimización para Hardware
- ✅ **NPU Radxa Rock 5T** con aceleración RKNN
- ✅ **Procesamiento multi-hilo** para máximo rendimiento
- ✅ **Mejora de visión nocturna** con algoritmos adaptativos
- ✅ **Streaming HTTP** optimizado para web
- ✅ **Docker containerizado** para deployment fácil

---

## 🏗️ Arquitectura del Sistema

```mermaid
graph TB
    A[Cámara IP RTSP] --> B[Video Processor]
    B --> C[YOLOv8n + RKNN]
    C --> D[BYTETracker]
    D --> E[Traffic Analyzer]
    E --> F[SQLite Database]
    E --> G[Controller Service]
    
    H[React Frontend] --> I[FastAPI Backend]
    I --> B
    I --> F
    I --> G
    
    G --> J[Controladora de Semáforos]
    
    K[Auth Service] --> I
    L[Analysis Config] --> E
    M[Camera Config] --> B
```

### Componentes Principales

| Componente | Función | Tecnología |
|------------|---------|------------|
| **Video Processor** | Procesamiento de video en tiempo real | OpenCV + Threading |
| **Vehicle Detector** | Detección de vehículos con IA | YOLOv8n + RKNN NPU |
| **BYTETracker** | Seguimiento de vehículos | Algoritmo de tracking |
| **Traffic Analyzer** | Análisis de tráfico y velocidades | Python + NumPy |
| **Database Manager** | Gestión de datos con retención automática | SQLite + AsyncIO |
| **Controller Service** | Comunicación con semáforos | HTTP REST |
| **Auth Service** | Autenticación y autorización | JWT + bcrypt |
| **Frontend** | Interfaz de usuario | React + Tailwind CSS |
| **Backend API** | API REST y streaming | FastAPI + Uvicorn |

---

## 🛠️ Requisitos del Sistema

### 📟 Hardware Recomendado
| Componente | Mínimo | Recomendado |
|------------|--------|-------------|
| **SBC** | Radxa Rock 5A | **Radxa Rock 5T** |
| **RAM** | 4GB | **8GB** |
| **Almacenamiento** | 32GB microSD | **64GB eMMC** |
| **Red** | WiFi 802.11n | **Ethernet Gigabit** |
| **Cámara** | IP con RTSP | **IP PoE con H.264** |

### 💻 Software Base
- **OS**: Ubuntu 22.04 LTS para Radxa
- **Docker**: 20.10+ y Docker Compose 2.0+
- **Python**: 3.9+ (incluido en imagen Docker)
- **Node.js**: 18+ (para desarrollo frontend)

### 📷 Cámaras Compatibles
- Cualquier cámara IP con stream **RTSP**
- Codecs soportados: **H.264**, H.265
- Resoluciones: 720p, **1080p**, 2K
- Marcas probadas: Hikvision, Dahua, Axis, Uniview

---

## 🚀 Instalación

### 📦 Instalación Automática (Recomendada)

```bash
# 1. Clonar repositorio
git clone https://github.com/tu-usuario/vehicle-detection-system.git
cd vehicle-detection-system

# 2. Ejecutar instalador automático
sudo chmod +x deploy/install_radxa.sh
sudo ./deploy/install_radxa.sh

# 3. Configurar sistema
vehicle-detection-setup
```

### 🐳 Instalación con Docker

```bash
# 1. Clonar y preparar
git clone https://github.com/tu-usuario/vehicle-detection-system.git
cd vehicle-detection-system

# 2. Construir y ejecutar
docker-compose build
docker-compose up -d

# 3. Verificar estado
docker-compose ps
curl http://localhost:8000/api/camera_health
```

### 🔧 Instalación Manual

<details>
<summary><b>Click para ver pasos detallados</b></summary>

```bash
# 1. Actualizar sistema
sudo apt update && sudo apt upgrade -y

# 2. Instalar Docker
sudo apt install -y docker.io docker-compose
sudo usermod -aG docker $USER
newgrp docker

# 3. Instalar dependencias del sistema
sudo apt install -y python3-pip python3-dev python3-opencv 
sudo apt install -y python3-numpy python3-setuptools
sudo apt install -y rknpu2-rk3588 python3-rknnlite2  # Para Radxa

# 4. Crear directorios
sudo mkdir -p /opt/vehicle-detection
sudo chown $USER:$USER /opt/vehicle-detection
cd /opt/vehicle-detection

# 5. Instalar aplicación
git clone https://github.com/tu-usuario/vehicle-detection-system.git .
pip3 install -r requirements.txt

# 6. Configurar archivos
mkdir -p data config models logs
cp config/examples/* config/

# 7. Iniciar aplicación
python3 main.py
```

</details>

---

## ⚙️ Configuración

### 🎬 1. Configuración de Cámara

Acceda a la interfaz web en `http://IP_RADXA:8000` y vaya a **Config. Cámara**:

#### Configuración RTSP
```bash
# Formato general
rtsp://usuario:contraseña@IP_CAMARA:puerto/ruta

# Ejemplos por marca
# Hikvision
rtsp://admin:password@192.168.1.100:554/Streaming/Channels/101

# Dahua  
rtsp://admin:password@192.168.1.100:554/cam/realmonitor?channel=1&subtype=0

# Axis
rtsp://admin:password@192.168.1.100:554/axis-media/media.amp

# Genérica
rtsp://admin:password@192.168.1.100:554/stream1
```

#### Configuración de Semáforo
| Campo | Descripción | Ejemplo |
|-------|-------------|---------|
| **Fase** | Fase del semáforo (1-4) | `fase1` |
| **Dirección** | Dirección del tráfico | `norte` |
| **ID Controladora** | Identificador único | `CTRL_001` |
| **IP Controladora** | IP de la controladora | `192.168.1.200` |

### 📏 2. Configuración de Líneas de Análisis

En la **Vista de Cámara**:

1. **Activar stream** de video
2. **Dibujar líneas** haciendo clic en dos puntos
3. **Configurar cada línea**:
   - **Tipo**: Conteo o Velocidad
   - **Carril**: carril_1, carril_2, etc.
   - **Distancia**: Para cálculo de velocidad (metros)
4. **Guardar configuración**

### 🔴 3. Configuración de Zona Roja

Para detectar vehículos durante semáforo en rojo:

1. **Dibujar zona** haciendo clic en múltiples puntos (mínimo 3)
2. **Finalizar zona** cuando esté completa
3. **Guardar configuración**

### ⚙️ 4. Configuración del Sistema

En **Config. Sistema**:

| Parámetro | Valor Recomendado | Descripción |
|-----------|-------------------|-------------|
| **Umbral de Confianza** | 0.5 - 0.7 | Precisión vs. sensibilidad |
| **Visión Nocturna** | Habilitado | Mejora automática en poca luz |
| **Overlay de Análisis** | Habilitado | Mostrar líneas y zonas |
| **FPS Objetivo** | 30 | Rendimiento vs. calidad |
| **Retención de Datos** | 30 días | Limpieza automática |

---

## 📊 Uso del Sistema

### 🏠 Dashboard Principal

El dashboard muestra información en tiempo real:

- **Estado de cámara** y FPS actual
- **Vehículos en zona roja** durante semáforos
- **Conteos del día** por línea y carril  
- **Velocidades promedio** y estadísticas
- **Estado de controladora** y comunicación

### 📹 Vista de Cámara

- **Stream en tiempo real** con análisis visual
- **Overlays configurables** (líneas, zonas, tracks)
- **Herramientas de dibujo** interactivas
- **Control de stream** (play/pause/refresh)
- **Información de tracks** en tiempo real

### 📈 Reportes y Analytics

Genere reportes detallados:

```bash
# Datos de vehículos por fecha
GET /api/data/export?date=2024_06_15&type=vehicle

# Datos de zona roja
GET /api/data/export?date=2024_06_15&type=red_light&fase=fase1

# Reporte completo
GET /api/data/export?date=2024_06_15&type=all
```

### 📋 Comandos de Control

```bash
# Control del servicio
vehicle-detection-ctl start      # Iniciar
vehicle-detection-ctl stop       # Detener  
vehicle-detection-ctl restart    # Reiniciar
vehicle-detection-ctl status     # Estado
vehicle-detection-ctl logs       # Ver logs

# Mantenimiento
vehicle-detection-ctl backup     # Crear respaldo
vehicle-detection-ctl cleanup    # Limpiar datos antiguos
vehicle-detection-ctl update     # Actualizar sistema

# Usando Docker
docker-compose up -d             # Iniciar
docker-compose down              # Detener
docker-compose logs -f           # Ver logs
docker-compose restart          # Reiniciar
```

---

## 🔧 API REST

### 🔐 Autenticación

```bash
# Login
curl -X POST http://localhost:8000/api/auth/login \
  -H "Content-Type: application/json" \
  -d '{"username":"admin","password":"admin123"}'

# Respuesta
{
  "token": "eyJ0eXAiOiJKV1QiLCJhbGciOiJIUzI1NiJ9...",
  "message": "Login exitoso"
}

# Usar token
curl -H "Authorization: Bearer TOKEN" \
  http://localhost:8000/api/camera/status
```

### 📷 Endpoints de Cámara

| Método | Endpoint | Descripción |
|--------|----------|-------------|
| `GET` | `/api/camera/status` | Estado actual de cámara |
| `POST` | `/api/camera/config` | Actualizar configuración |
| `GET` | `/api/camera/stream` | Stream de video HTTP |
| `POST` | `/api/camera/test` | Probar conexión RTSP |
| `POST` | `/api/camera/restart` | Reiniciar procesamiento |
| `GET` | `/api/camera_health` | Health check completo |

### 📊 Endpoints de Análisis

| Método | Endpoint | Descripción |
|--------|----------|-------------|
| `GET` | `/api/analysis/lines` | Obtener líneas configuradas |
| `POST` | `/api/analysis/lines` | Agregar nueva línea |
| `DELETE` | `/api/analysis/lines/{id}` | Eliminar línea |
| `GET` | `/api/analysis/zones` | Obtener zonas configuradas |
| `POST` | `/api/analysis/zones` | Agregar nueva zona |
| `DELETE` | `/api/analysis/zones/{id}` | Eliminar zona |

### 🚦 Endpoints de Controladora

| Método | Endpoint | Descripción |
|--------|----------|-------------|
| `GET` | `/api/rojo_status` | Estado de semáforos |
| `POST` | `/api/rojo_status` | Actualizar estado |
| `POST` | `/api/analitico_camara` | Enviar analítico |

### 📈 Endpoints de Datos

| Método | Endpoint | Descripción |
|--------|----------|-------------|
| `GET` | `/api/data/export` | Exportar datos por fecha |

### 📚 Documentación Completa

- **Swagger UI**: `http://localhost:8000/docs`
- **ReDoc**: `http://localhost:8000/redoc`

---

## 📁 Estructura del Proyecto

```
vehicle-detection-system/
├── 📁 app/                     # Backend Python
│   ├── 📁 core/               # Módulos principales
│   │   ├── analyzer.py        # Análisis de tráfico
│   │   ├── database.py        # Gestión de base de datos
│   │   ├── detector.py        # Detección con YOLOv8n
│   │   ├── tracker.py         # BYTETracker implementation
│   │   └── video_processor.py # Procesamiento de video
│   └── 📁 services/           # Servicios
│       ├── auth_service.py    # Autenticación JWT
│       └── controller_service.py # Comunicación con controladora
├── 📁 frontend/               # Frontend React
│   ├── 📁 src/
│   │   ├── 📁 components/     # Componentes React
│   │   ├── 📁 services/       # Servicios API
│   │   └── 📁 context/        # Context providers
│   ├── package.json           # Dependencias NPM
│   └── tailwind.config.js     # Configuración Tailwind
├── 📁 config/                 # Configuraciones
│   ├── analysis.json          # Líneas y zonas
│   ├── camera_config.json     # Configuración de cámara
│   └── system_config.json     # Configuración del sistema
├── 📁 data/                   # Bases de datos por fecha
│   └── 📁 YYYY/MM/           # Organización temporal
├── 📁 models/                 # Modelos de IA
│   ├── yolov8n.onnx          # Modelo OpenCV
│   └── yolov8n.rknn          # Modelo RKNN optimizado
├── 📁 deploy/                 # Scripts de deployment
├── 📁 tests/                  # Tests unitarios
├── main.py                    # Aplicación principal FastAPI
├── requirements.txt           # Dependencias Python
├── docker-compose.yml         # Orquestación Docker
├── Dockerfile                 # Imagen Docker
└── README.md                  # Este archivo
```

---

## 🗄️ Base de Datos

### 📊 Esquema de Datos

#### Tabla: `vehicle_crossings`
```sql
CREATE TABLE vehicle_crossings (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    vehicle_id INTEGER NOT NULL,           -- ID del vehículo tracked
    line_id TEXT NOT NULL,                 -- ID de línea cruzada
    line_name TEXT NOT NULL,               -- Nombre de línea
    fase TEXT NOT NULL,                    -- Fase del semáforo
    semaforo_estado TEXT NOT NULL,         -- Estado: rojo/verde/amarillo
    timestamp DATETIME DEFAULT (datetime('now','localtime')),
    velocidad REAL,                        -- Velocidad en km/h
    direccion TEXT,                        -- Dirección del tráfico
    No_Controladora TEXT,                  -- ID de controladora
    confianza REAL,                        -- Confianza de detección (0-1)
    carril TEXT,                           -- Carril (carril_1, carril_2, etc.)
    clase_vehiculo INTEGER,                -- Clase COCO (2=car, 3=motorcycle, etc.)
    bbox_x INTEGER,                        -- Bounding box X
    bbox_y INTEGER,                        -- Bounding box Y  
    bbox_w INTEGER,                        -- Bounding box ancho
    bbox_h INTEGER,                        -- Bounding box alto
    metadata TEXT                          -- JSON con datos adicionales
);
```

#### Tabla: `red_light_counts`
```sql
CREATE TABLE red_light_counts (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    fase TEXT NOT NULL,                    -- Fase del semáforo
    inicio_rojo DATETIME NOT NULL,         -- Inicio del ciclo rojo
    fin_rojo DATETIME,                     -- Fin del ciclo rojo
    vehiculos_inicio INTEGER DEFAULT 0,    -- Vehículos al inicio del rojo
    vehiculos_final INTEGER DEFAULT 0,     -- Vehículos al final del rojo
    vehiculos_cruzaron INTEGER DEFAULT 0,  -- Vehículos que cruzaron en rojo
    duracion_segundos INTEGER,             -- Duración del ciclo
    direccion TEXT,                        -- Dirección del tráfico
    No_Controladora TEXT,                  -- ID de controladora
    analitico_enviado BOOLEAN DEFAULT 0,   -- Si se envió analítico
    analitico_recibido BOOLEAN DEFAULT 0   -- Si se recibió confirmación
);
```

### 🔄 Organización Temporal

- **Base de datos diaria**: Una base SQLite por día
- **Estructura de carpetas**: `/data/YYYY/MM/YYYY_MM_DD.db`
- **Retención automática**: Configurable (30 días por defecto)
- **Limpieza automática**: Diaria a las 2:00 AM

### 📈 Consultas Útiles

```sql
-- Conteo de vehículos por hora
SELECT 
    strftime('%H', timestamp) as hora,
    COUNT(*) as total_vehiculos,
    AVG(velocidad) as velocidad_promedio
FROM vehicle_crossings 
WHERE date(timestamp) = '2024-06-15'
GROUP BY hora
ORDER BY hora;

-- Vehículos por carril
SELECT 
    carril,
    COUNT(*) as total,
    AVG(velocidad) as vel_promedio,
    MAX(velocidad) as vel_maxima
FROM vehicle_crossings 
WHERE date(timestamp) = date('now')
GROUP BY carril;

-- Estadísticas de zona roja
SELECT 
    fase,
    COUNT(*) as ciclos_total,
    AVG(vehiculos_cruzaron) as promedio_infracciones,
    AVG(duracion_segundos) as duracion_promedio
FROM red_light_counts 
WHERE date(inicio_rojo) = date('now')
GROUP BY fase;
```

---

## 🔒 Seguridad

### 🔐 Autenticación

- **JWT Tokens** con expiración configurable (1 hora por defecto)
- **Contraseñas hasheadas** con bcrypt y salt
- **Sesiones revocables** con blacklist de tokens
- **Credenciales por defecto**: `admin` / `admin123` (¡CAMBIAR EN PRODUCCIÓN!)

### 🛡️ Configuración de Firewall

```bash
# Configuración básica UFW
sudo ufw enable
sudo ufw default deny incoming
sudo ufw default allow outgoing

# Permitir servicios necesarios
sudo ufw allow ssh
sudo ufw allow 8000/tcp  # API y Frontend
sudo ufw allow from 192.168.1.0/24 to any port 22  # SSH solo desde LAN

# Verificar reglas
sudo ufw status verbose
```

### 🔄 Configuración de Fail2ban

```bash
# Instalar Fail2ban
sudo apt install fail2ban

# Configurar /etc/fail2ban/jail.local
[DEFAULT]
bantime = 3600
findtime = 600
maxretry = 3

[sshd]
enabled = true
port = ssh
filter = sshd
logpath = /var/log/auth.log
```

### 🔑 Configuración HTTPS

Para producción, configure un proxy reverso con SSL:

```nginx
# /etc/nginx/sites-available/vehicle-detection
server {
    listen 443 ssl;
    server_name your-domain.com;

    ssl_certificate /path/to/certificate.crt;
    ssl_certificate_key /path/to/private.key;

    location / {
        proxy_pass http://localhost:8000;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto $scheme;
    }

    location /api/camera/stream {
        proxy_pass http://localhost:8000;
        proxy_buffering off;
        proxy_request_buffering off;
    }
}
```

---

## 📈 Monitoreo

### 📊 Métricas del Sistema

El sistema expone métricas en `/api/camera_health`:

```json
{
  "status": "healthy",
  "timestamp": "2024-06-15T10:30:00",
  "camera_connected": true,
  "camera_fps": 30,
  "camera_configured": true,
  "hardware": "Radxa ROCK 5 Model T",
  "modules_available": true,
  "version": "1.0.0"
}
```

### 📝 Logs del Sistema

```bash
# Logs de aplicación
tail -f /opt/vehicle-detection/logs/app.log

# Logs de contenedor
docker logs -f vehicle-detection-production

# Logs del sistema
journalctl -u vehicle-detection -f

# Logs por nivel
grep "ERROR" /opt/vehicle-detection/logs/app.log
grep "WARNING" /opt/vehicle-detection/logs/app.log
```

### 🎯 Alertas Configurables

Configure alertas para eventos importantes:

```python
# Ejemplo de integración con Telegram
import requests

def send_alert(message):
    bot_token = "YOUR_BOT_TOKEN"
    chat_id = "YOUR_CHAT_ID"
    url = f"https://api.telegram.org/bot{bot_token}/sendMessage"
    
    requests.post(url, json={
        "chat_id": chat_id,
        "text": f"🚨 Vehicle Detection Alert: {message}"
    })

# Usar en el código
if not camera_connected:
    send_alert("Cámara desconectada")
```

### 📊 Dashboard de Monitoreo

Para monitoreo avanzado, integre con:

- **Grafana**: Visualización de métricas
- **Prometheus**: Recolección de métricas
- **InfluxDB**: Base de datos de tiempo
- **Elastic Stack**: Análisis de logs

---

## 🧪 Testing

### 🔬 Ejecutar Tests

```bash
# Tests unitarios
cd /opt/vehicle-detection
python -m pytest tests/ -v

# Tests con coverage
python -m pytest tests/ --cov=app --cov-report=html

# Test específico
python -m pytest tests/test_detector.py::TestVehicleDetector::test_preprocess_frame -v

# Test manual de componentes
python tests/run_tests.py
```

### 🧪 Tests de Integración

```bash
# Test completo de RTSP
python SCRIPT_PRUEBARTSP.py

# Test de API
curl -X POST http://localhost:8000/api/auth/login \
  -H "Content-Type: application/json" \
  -d '{"username":"admin","password":"admin123"}'

# Test de stream
curl -I http://localhost:8000/api/camera/stream

# Test de health
curl http://localhost:8000/api/camera_health | jq .
```

### 🎭 Simulador de Controladora

Para testing sin hardware real:

```bash
# Iniciar simulador
docker-compose --profile testing up mock-controller

# El simulador estará disponible en http://localhost:8080
```

---

## 🚨 Errores Conocidos

### ❌ Error Crítico 1: Variables no inicializadas en `analyzer.py`

**Síntoma**: Error `AttributeError: 'TrafficAnalyzer' object has no attribute 'vehicle_lanes'`

**Solución**:
```python
# En app/core/analyzer.py línea 43-44, agregar:
def __init__(self):
    # ... código existente ...
    self.vehicle_lanes = {}      # AGREGAR ESTA LÍNEA
    self.vehicle_last_line = {}  # AGREGAR ESTA LÍNEA
```

### ❌ Error Crítico 2: Callback None en `video_processor.py`

**Síntoma**: Error `TypeError: 'NoneType' object is not callable`

**Solución**:
```python
# En app/core/video_processor.py línea 357, cambiar:
if results['send_analytic'] and self.callback_func:  # AGREGAR VERIFICACIÓN
    await self.callback_func('send_analytic', analytic_data)
```

### ❌ Error 3: Iconos no encontrados en Frontend

**Síntoma**: Warning sobre iconos no disponibles

**Solución**:
```bash
# Actualizar @heroicons/react
cd frontend
npm install @heroicons/react@latest
```

### ❌ Error 4: Fallo en streaming RTSP

**Síntoma**: Stream no se conecta o frames en negro

**Solución**:
1. Verificar URL RTSP con VLC Media Player
2. Comprobar credenciales y permisos de cámara
3. Verificar conectividad de red
4. Revisar formato de stream (H.264 preferido)

### ❌ Error 5: NPU RKNN no disponible

**Síntoma**: Warning "RKNN no disponible, usando OpenCV como fallback"

**Solución**:
```bash
# En Radxa Rock 5T
sudo apt update
sudo apt install rknpu2-rk3588 python3-rknnlite2

# Verificar instalación
python3 -c "from rknnlite.api import RKNNLite; print('RKNN OK')"
```

---

## 🐛 Troubleshooting

### 🔍 Diagnóstico General

```bash
# 1. Verificar estado del sistema
vehicle-detection-ctl status

# 2. Verificar logs recientes
vehicle-detection-ctl logs | tail -50

# 3. Verificar health del sistema
curl http://localhost:8000/api/camera_health | jq .

# 4. Verificar conectividad de red
ping IP_DE_CAMARA
telnet IP_DE_CAMARA 554
```

### 📹 Problemas con Cámara

#### Cámara no se conecta
```bash
# Verificar URL RTSP
ffmpeg -i "rtsp://admin:password@IP:554/stream1" -t 10 -f null -

# Probar con VLC
vlc rtsp://admin:password@IP:554/stream1

# Verificar códec
ffprobe rtsp://admin:password@IP:554/stream1
```

#### Stream lento o con latencia
```bash
# Reducir buffer en configuración de cámara
# Cambiar resolución a 720p
# Verificar ancho de banda de red
iperf3 -c IP_DE_CAMARA
```

### 🧠 Problemas con IA/RKNN

#### NPU no funciona
```bash
# Verificar dispositivos NPU
ls -la /dev/dri/
ls -la /dev/mali*

# Verificar librerías RKNN
ldd /usr/local/lib/python3.9/dist-packages/rknnlite/api/rknn_api.py

# Reinstalar RKNN
sudo apt remove --purge rknpu2-rk3588
sudo apt install rknpu2-rk3588 python3-rknnlite2
```

#### Detección imprecisa
```bash
# Ajustar umbral de confianza (Config. Análisis)
# Mejorar iluminación de la escena
# Verificar ángulo y posición de cámara
# Calibrar zona de detección
```

### 🗄️ Problemas con Base de Datos

#### Base de datos corrupta
```bash
# Verificar integridad
sqlite3 /app/data/YYYY/MM/YYYY_MM_DD.db "PRAGMA integrity_check;"

# Reparar base de datos
sqlite3 /app/data/YYYY/MM/YYYY_MM_DD.db ".recover" | sqlite3 repaired.db

# Crear respaldo antes de reparar
vehicle-detection-ctl backup
```

#### Espacio en disco lleno
```bash
# Verificar uso de disco
df -h /app/data

# Limpiar datos antiguos manualmente
vehicle-detection-ctl cleanup

# Reducir retención de datos
# Editar /app/config/system_config.json
# Cambiar "data_retention_days" a valor menor
```

### 🌐 Problemas de Red

#### Controladora no responde
```bash
# Verificar conectividad
ping IP_CONTROLADORA
telnet IP_CONTROLADORA 8080

# Verificar configuración
curl http://IP_CONTROLADORA:8080/api/analiticos

# Revisar logs de comunicación
grep "controller" /app/logs/app.log
```

#### Puerto 8000 ocupado
```bash
# Verificar qué proceso usa el puerto
sudo netstat -tulpn | grep :8000

# Cambiar puerto en docker-compose.yml
# Reiniciar servicios
docker-compose down
docker-compose up -d
```

### 🔧 Herramientas de Diagnóstico

```bash
# Script de diagnóstico completo
#!/bin/bash
echo "=== DIAGNÓSTICO DEL SISTEMA ==="

echo "1. Estado de servicios:"
docker ps | grep vehicle

echo "2. Uso de recursos:"
free -h
df -h

echo "3. Estado de red:"
ip addr show

echo "4. Logs recientes:"
docker logs vehicle-detection-production --tail 10

echo "5. Health check:"
curl -s http://localhost:8000/api/camera_health | jq .

echo "=== FIN DEL DIAGNÓSTICO ==="
```

---

## 🤝 Contribución

### 🔄 Workflow de Desarrollo

1. **Fork** el repositorio
2. **Crear rama** feature: `git checkout -b feature/nueva-funcion`
3. **Commit** cambios: `git commit -am 'Agregar nueva función'`
4. **Push** rama: `git push origin feature/nueva-funcion`
5. **Crear Pull Request**

### 📏 Estándares de Código

#### Python
```bash
# Formateo con Black
black app/

# Importaciones con isort
isort app/

# Linting con flake8
flake8 app/

# Type checking con mypy
mypy app/
```

#### JavaScript/React
```bash
# Linting con ESLint
npm run lint

# Formateo con Prettier
npm run format

# Tests con Jest
npm test
```

### 📋 Checklist para Pull Requests

- [ ] Código formateado correctamente
- [ ] Tests unitarios agregados/actualizados
- [ ] Documentación actualizada
- [ ] No hay secrets o credenciales hardcodeadas
- [ ] Compatible con Radxa Rock 5T
- [ ] Performance no degradado
- [ ] API backward compatible

### 🐛 Reportar Bugs

Use el template de issues con:

1. **Descripción** del problema
2. **Pasos** para reproducir
3. **Comportamiento esperado** vs actual
4. **Environment** (hardware, OS, versiones)
5. **Logs** relevantes
6. **Screenshots** si aplica

### 💡 Solicitar Features

1. **Descripción** detallada de la funcionalidad
2. **Justificación** del caso de uso
3. **Mockups** o diseños si aplica
4. **Consideraciones** de implementación

---

## 📜 Licencia

Este proyecto está licenciado bajo la **Licencia MIT**. Ver [LICENSE](LICENSE) para detalles.

```
MIT License

Copyright (c) 2024 Vehicle Detection System

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
```

---

## 🙏 Agradecimientos

- **[Ultralytics](https://ultralytics.com/)** por YOLOv8
- **[Radxa](https://radxa.com/)** por el hardware y soporte RKNN
- **[OpenCV](https://opencv.org/)** por las librerías de visión computacional
- **[FastAPI](https://fastapi.tiangolo.com/)** por el framework web
- **[React](https://reactjs.org/)** por la interfaz de usuario
- **Comunidad Open Source** por las librerías y herramientas

---

## 📞 Soporte

### 🆘 Canales de Ayuda

- **Issues**: [GitHub Issues](https://github.com/tu-usuario/vehicle-detection-system/issues)
- **Documentación**: [Wiki del Proyecto](https://github.com/tu-usuario/vehicle-detection-system/wiki)
- **Discusiones**: [GitHub Discussions](https://github.com/tu-usuario/vehicle-detection-system/discussions)

### 📧 Contacto

- **Email**: soporte@tu-dominio.com
- **Documentación Técnica**: [docs.tu-dominio.com](https://docs.tu-dominio.com)

---

<div align="center">

## 🚗 **Sistema de Detección Vehicular para Radxa Rock 5T**

*Desarrollado con ❤️ para la comunidad de tráfico inteligente*

[![Made with Python](https://img.shields.io/badge/Made%20with-Python-blue.svg)](https://python.org)
[![Made with React](https://img.shields.io/badge/Made%20with-React-61DAFB.svg)](https://reactjs.org)
[![Powered by RKNN](https://img.shields.io/badge/Powered%20by-RKNN-orange.svg)](https://github.com/rockchip-linux/rknn-toolkit2)
[![Docker](https://img.shields.io/badge/Docker-Ready-blue.svg)](https://docker.com)

**[🏠 Inicio](/) • [📖 Documentación](https://github.com/tu-usuario/vehicle-detection-system/wiki) • [🐛 Issues](https://github.com/tu-usuario/vehicle-detection-system/issues) • [💬 Discusiones](https://github.com/tu-usuario/vehicle-detection-system/discussions)**

</div>