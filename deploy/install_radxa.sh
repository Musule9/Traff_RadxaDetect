set -e

echo "🚀 Instalador del Sistema de Detección Vehicular para Radxa Rock 5T"
echo "================================================================="

# Verificar si es root
if [ "$EUID" -ne 0 ]; then
    echo "❌ Este script debe ejecutarse como root"
    echo "   Uso: sudo $0"
    exit 1
fi

# Verificar hardware
if [ -f /proc/device-tree/model ]; then
    MODEL=$(cat /proc/device-tree/model 2>/dev/null)
    echo "📋 Hardware detectado: $MODEL"
    
    if [[ "$MODEL" != *"Radxa"* ]] && [[ "$MODEL" != *"ROCK"* ]]; then
        echo "⚠️  ADVERTENCIA: Este instalador está optimizado para Radxa Rock"
        read -p "¿Continuar de todas formas? (y/N): " -n 1 -r
        echo
        if [[ ! $REPLY =~ ^[Yy]$ ]]; then
            exit 1
        fi
    fi
else
    echo "⚠️  No se pudo detectar el modelo del hardware"
fi

# Crear usuario del sistema
SYSTEM_USER="vehicle-detection"
if ! id "$SYSTEM_USER" &>/dev/null; then
    echo "👤 Creando usuario del sistema: $SYSTEM_USER"
    useradd -r -s /bin/bash -d /opt/vehicle-detection -m $SYSTEM_USER
    usermod -aG docker $SYSTEM_USER 2>/dev/null || echo "Grupo docker no encontrado, se agregará después"
fi

# Crear directorios del sistema
INSTALL_DIR="/opt/vehicle-detection"
echo "📁 Creando directorios del sistema en $INSTALL_DIR"

mkdir -p $INSTALL_DIR/{data,config,models,logs,backups}
mkdir -p $INSTALL_DIR/data/{$(date +%Y),$(date +%Y)/$(date +%m)}

# Instalar dependencias del sistema
echo "📦 Instalando dependencias del sistema..."
apt-get update
apt-get install -y \
    curl \
    wget \
    git \
    docker.io \
    docker-compose \
    python3 \
    python3-pip \
    build-essential \
    cmake \
    pkg-config \
    libopencv-dev \
    v4l-utils \
    htop \
    nano \
    ufw \
    fail2ban \
    logrotate

# Configurar Docker
echo "🐳 Configurando Docker..."
systemctl enable docker
systemctl start docker
usermod -aG docker $SYSTEM_USER

# Instalar dependencias específicas de Radxa
echo "🔧 Instalando dependencias de Radxa Rock..."
if [[ "$MODEL" == *"Radxa"* ]] || [[ "$MODEL" == *"ROCK"* ]]; then
    # Agregar repositorio de Radxa
    if [ ! -f /etc/apt/sources.list.d/radxa.list ]; then
        echo "deb http://apt.radxa.com/focal/ focal main" > /etc/apt/sources.list.d/radxa.list
        wget -O - http://apt.radxa.com/focal/public.key | apt-key add -
        apt-get update
    fi
    
    # Instalar RKNN toolkit
    apt-get install -y \
        python3-rknnlite \
        librknn-runtime \
        rockchip-mpp-dev \
        rockchip-rga-dev || echo "Algunos paquetes de Radxa no están disponibles"
fi

# Descargar código fuente
echo "📥 Descargando código fuente..."
cd /tmp
if [ -d "vehicle-detection-system" ]; then
    rm -rf vehicle-detection-system
fi

# Aquí normalmente sería: git clone https://github.com/tu-repo/vehicle-detection-system.git
# Por ahora copiamos desde directorio actual
if [ -d "$PWD/vehicle-detection-system" ]; then
    cp -r "$PWD/vehicle-detection-system" /tmp/
else
    echo "❌ Código fuente no encontrado en $PWD/vehicle-detection-system"
    exit 1
fi

# Copiar archivos al directorio de instalación
echo "📋 Copiando archivos de la aplicación..."
cp -r /tmp/vehicle-detection-system/* $INSTALL_DIR/
chown -R $SYSTEM_USER:$SYSTEM_USER $INSTALL_DIR

# Construir imagen Docker
echo "🐳 Construyendo imagen Docker..."
cd $INSTALL_DIR
sudo -u $SYSTEM_USER docker-compose build

# Crear servicios systemd
echo "🔧 Configurando servicios del sistema..."

# Servicio principal
cat > /etc/systemd/system/vehicle-detection.service << EOF
[Unit]
Description=Vehicle Detection System
Requires=docker.service
After=docker.service

[Service]
Type=simple
User=$SYSTEM_USER
Group=$SYSTEM_USER
WorkingDirectory=$INSTALL_DIR
ExecStart=/usr/bin/docker-compose up --no-deps vehicle-detection
ExecStop=/usr/bin/docker-compose down
Restart=always
RestartSec=10

[Install]
WantedBy=multi-user.target
EOF

# Servicio de limpieza diaria
cat > /etc/systemd/system/vehicle-detection-cleanup.service << EOF
[Unit]
Description=Vehicle Detection Daily Cleanup
Requires=vehicle-detection.service

[Service]
Type=oneshot
User=$SYSTEM_USER
Group=$SYSTEM_USER
WorkingDirectory=$INSTALL_DIR
ExecStart=/usr/bin/docker-compose exec -T vehicle-detection python3 -c "
import asyncio
from app.core.database import DatabaseManager
async def cleanup():
    db = DatabaseManager()
    await db.cleanup_old_databases()
asyncio.run(cleanup())
"
EOF

# Timer para limpieza diaria
cat > /etc/systemd/system/vehicle-detection-cleanup.timer << EOF
[Unit]
Description=Run vehicle detection cleanup daily
Requires=vehicle-detection.service

[Timer]
OnCalendar=daily
Persistent=true

[Install]
WantedBy=timers.target
EOF

# Configurar firewall
echo "🔥 Configurando firewall..."
ufw --force enable
ufw allow ssh
ufw allow 8000/tcp  # API web
ufw allow from 192.168.0.0/16 to any port 8000  # Solo red local para web

# Configurar fail2ban para SSH
echo "🛡️  Configurando fail2ban..."
systemctl enable fail2ban
systemctl start fail2ban

# Configurar logrotate
echo "📊 Configurando rotación de logs..."
cat > /etc/logrotate.d/vehicle-detection << EOF
$INSTALL_DIR/logs/*.log {
    daily
    missingok
    rotate 30
    compress
    delaycompress
    notifempty
    copytruncate
    su $SYSTEM_USER $SYSTEM_USER
}
EOF

# Habilitar servicios
echo "🚀 Habilitando servicios..."
systemctl daemon-reload
systemctl enable vehicle-detection.service
systemctl enable vehicle-detection-cleanup.timer
systemctl start vehicle-detection-cleanup.timer

# Crear scripts de utilidad
echo "🛠️  Creando scripts de utilidad..."

# Script de inicio/parada
cat > /usr/local/bin/vehicle-detection-ctl << 'EOF'
#!/bin/bash

INSTALL_DIR="/opt/vehicle-detection"
SYSTEM_USER="vehicle-detection"

case "$1" in
    start)
        echo "🚀 Iniciando Vehicle Detection System..."
        systemctl start vehicle-detection
        ;;
    stop)
        echo "🛑 Deteniendo Vehicle Detection System..."
        systemctl stop vehicle-detection
        ;;
    restart)
        echo "🔄 Reiniciando Vehicle Detection System..."
        systemctl restart vehicle-detection
        ;;
    status)
        systemctl status vehicle-detection
        ;;
    logs)
        journalctl -u vehicle-detection -f
        ;;
    update)
        echo "📥 Actualizando sistema..."
        cd $INSTALL_DIR
        sudo -u $SYSTEM_USER docker-compose pull
        sudo -u $SYSTEM_USER docker-compose build
        systemctl restart vehicle-detection
        ;;
    backup)
        echo "💾 Creando respaldo..."
        DATE=$(date +%Y%m%d_%H%M%S)
        tar -czf "$INSTALL_DIR/backups/backup_$DATE.tar.gz" \
            -C $INSTALL_DIR data config
        echo "Respaldo creado: $INSTALL_DIR/backups/backup_$DATE.tar.gz"
        ;;
    cleanup)
        echo "🧹 Ejecutando limpieza manual..."
        systemctl start vehicle-detection-cleanup
        ;;
    *)
        echo "Uso: $0 {start|stop|restart|status|logs|update|backup|cleanup}"
        exit 1
        ;;
esac
EOF

chmod +x /usr/local/bin/vehicle-detection-ctl

# Script de configuración inicial
cat > /usr/local/bin/vehicle-detection-setup << 'EOF'
#!/bin/bash

INSTALL_DIR="/opt/vehicle-detection"
SYSTEM_USER="vehicle-detection"

echo "🔧 Configuración inicial del Sistema de Detección Vehicular"
echo "==========================================================="

# Verificar que el servicio esté corriendo
if ! systemctl is-active --quiet vehicle-detection; then
    echo "❌ El servicio no está corriendo. Iniciando..."
    systemctl start vehicle-detection
    sleep 10
fi

# Mostrar información del sistema
echo
echo "📊 Información del sistema:"
echo "- Directorio de instalación: $INSTALL_DIR"
echo "- Usuario del sistema: $SYSTEM_USER"
echo "- URL de acceso: http://$(hostname -I | awk '{print $1}'):8000"
echo
echo "🔑 Credenciales por defecto:"
echo "- Usuario: admin"
echo "- Contraseña: admin123"
echo
echo "📁 Directorios importantes:"
echo "- Datos: $INSTALL_DIR/data"
echo "- Configuración: $INSTALL_DIR/config"
echo "- Logs: $INSTALL_DIR/logs"
echo
echo "🛠️  Comandos útiles:"
echo "- Controlar servicio: vehicle-detection-ctl {start|stop|restart|status|logs}"
echo "- Ver logs: vehicle-detection-ctl logs"
echo "- Crear respaldo: vehicle-detection-ctl backup"
echo
echo "🌐 Para configurar el sistema:"
echo "1. Abra http://$(hostname -I | awk '{print $1}'):8000 en su navegador"
echo "2. Inicie sesión con las credenciales por defecto"
echo "3. Configure la URL RTSP de su cámara"
echo "4. Configure las líneas de conteo y zonas"
echo "5. Configure la IP de su controladora de semáforos"
echo
echo "✅ Instalación completada exitosamente!"
EOF

chmod +x /usr/local/bin/vehicle-detection-setup

# Descargar modelo por defecto
echo "🤖 Descargando modelo YOLOv8n..."
cd $INSTALL_DIR/models
if [ ! -f "yolov8n.onnx" ]; then
    sudo -u $SYSTEM_USER wget -q -O yolov8n.onnx \
        https://github.com/ultralytics/assets/releases/download/v0.0.0/yolov8n.onnx || \
        echo "⚠️  No se pudo descargar el modelo, se descargará en el primer inicio"
fi

# Configurar permisos finales
chown -R $SYSTEM_USER:$SYSTEM_USER $INSTALL_DIR

# Iniciar servicio
echo "🚀 Iniciando servicio..."
systemctl start vehicle-detection

# Mostrar información final
echo
echo "🎉 ¡Instalación completada exitosamente!"
echo "========================================"
echo
echo "📊 Estado del servicio:"
systemctl status vehicle-detection --no-pager -l
echo
echo "🌐 URL de acceso: http://$(hostname -I | awk '{print $1}'):8000"
echo "🔑 Usuario por defecto: admin / admin123"
echo
echo "ℹ️  Para más información ejecute: vehicle-detection-setup"
echo