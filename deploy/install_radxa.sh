#!/bin/bash
set -e

echo "🚀 Vehicle Detection System - Instalador Masivo para Producción"
echo "==============================================================="

# Verificar root
if [ "$EUID" -ne 0 ]; then
    echo "❌ Ejecutar como root: sudo bash $0"
    exit 1
fi

# Obtener directorio del proyecto
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"

# Verificar estructura del proyecto
if [ ! -f "$PROJECT_DIR/main.py" ] || [ ! -f "$PROJECT_DIR/requirements.txt" ]; then
    echo "❌ Error: Estructura de proyecto incorrecta"
    exit 1
fi

# Detectar hardware
if [ -f /proc/device-tree/model ]; then
    MODEL=$(tr -d '\0' < /proc/device-tree/model 2>/dev/null)
    echo "📋 Hardware: $MODEL"
fi

echo "📦 Instalando dependencias del sistema..."
apt-get update
apt-get install -y curl wget git python3 python3-pip python3-opencv ffmpeg ca-certificates gnupg

# Instalar Docker oficial
if ! command -v docker &> /dev/null; then
    echo "🐳 Instalando Docker..."
    install -m 0755 -d /etc/apt/keyrings
    curl -fsSL https://download.docker.com/linux/debian/gpg | gpg --dearmor -o /etc/apt/keyrings/docker.gpg
    chmod a+r /etc/apt/keyrings/docker.gpg
    echo "deb [arch=$(dpkg --print-architecture) signed-by=/etc/apt/keyrings/docker.gpg] https://download.docker.com/linux/debian $(. /etc/os-release && echo "$VERSION_CODENAME") stable" | tee /etc/apt/sources.list.d/docker.list > /dev/null
    apt-get update
    apt-get install -y docker-ce docker-ce-cli containerd.io docker-compose-plugin
fi

# Instalar docker-compose standalone
if ! command -v docker-compose &> /dev/null; then
    COMPOSE_VERSION=$(curl -s https://api.github.com/repos/docker/compose/releases/latest | grep 'tag_name' | cut -d\" -f4)
    curl -L "https://github.com/docker/compose/releases/download/${COMPOSE_VERSION}/docker-compose-$(uname -s)-$(uname -m)" -o /usr/local/bin/docker-compose
    chmod +x /usr/local/bin/docker-compose
fi

# Configurar Docker
systemctl enable docker
systemctl start docker

# Instalar RKNN
echo "🔧 Instalando RKNN..."
apt-get install -y python3-rknnlite2 || echo "⚠️ RKNN no disponible"

# Crear usuario del sistema
SYSTEM_USER="vehicle-detection"
if ! id "$SYSTEM_USER" &>/dev/null; then
    useradd -r -s /bin/bash -d /opt/vehicle-detection -m $SYSTEM_USER
    usermod -aG docker $SYSTEM_USER
fi

# Crear estructura de directorios
INSTALL_DIR="/opt/vehicle-detection"
mkdir -p $INSTALL_DIR/{data,config,models,logs,backups}
mkdir -p $INSTALL_DIR/data/{$(date +%Y),$(date +%Y)/$(date +%m)}

# Copiar archivos del proyecto
echo "📋 Copiando archivos..."
cp -r "$PROJECT_DIR"/* "$INSTALL_DIR/"
cd "$INSTALL_DIR"

# Crear archivos de configuración
[ ! -f config/system.json ] && [ -f config/system.json.example ] && cp config/system.json.example config/system.json
[ ! -f config/cameras.json ] && [ -f config/cameras.json.example ] && cp config/cameras.json.example config/cameras.json
[ ! -f config/analysis.json ] && [ -f config/analysis.json.example ] && cp config/analysis.json.example config/analysis.json
[ ! -f config/controllers.json ] && [ -f config/controllers.json.example ] && cp config/controllers.json.example config/controllers.json
[ ! -f .env ] && [ -f .env.example ] && cp .env.example .env

# Crear Dockerfile optimizado para producción
echo "🐳 Creando Dockerfile..."
cat > Dockerfile << 'EOF'
FROM python:3.11-slim-bookworm

WORKDIR /app

ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1

RUN apt-get update && apt-get install -y \
    python3-opencv \
    curl \
    ffmpeg \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

COPY . .
RUN mkdir -p /app/{data,config,models,logs}

RUN echo '#!/bin/bash\n\
mkdir -p /app/{data,config,models,logs}\n\
if [ ! -f /app/config/system.json ] && [ -f /app/scripts/init_config.py ]; then\n\
    python3 /app/scripts/init_config.py 2>/dev/null || true\n\
fi\n\
if [ ! -f /app/models/yolov8n.onnx ]; then\n\
    wget -q -O /app/models/yolov8n.onnx https://github.com/ultralytics/assets/releases/download/v0.0.0/yolov8n.onnx 2>/dev/null || true\n\
fi\n\
exec python3 main.py' > /app/start.sh && chmod +x /app/start.sh

EXPOSE 8000
HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 CMD curl -f http://localhost:8000/api/camera_health || exit 1
CMD ["/app/start.sh"]
EOF

# Crear docker-compose.yml LIMPIO (SIN ERRORES)
echo "🐳 Creando docker-compose.yml..."
cat > docker-compose.yml << 'EOF'
version: '3.8'

services:
  vehicle-detection:
    build: .
    container_name: vehicle-detection-prod
    restart: unless-stopped
    ports:
      - "8000:8000"
    volumes:
      - ./data:/app/data
      - ./config:/app/config
      - ./models:/app/models
      - ./logs:/app/logs
    environment:
      - APP_ENV=production
      - USE_RKNN=1
      - LOG_LEVEL=INFO
    devices:
      - /dev/dri:/dev/dri
    privileged: true
    healthcheck:
      test: ["CMD", "curl", "-f", "http://localhost:8000/api/camera_health"]
      interval: 30s
      timeout: 10s
      retries: 3
      start_period: 60s
    logging:
      driver: "json-file"
      options:
        max-size: "10m"
        max-file: "3"
EOF

# Cambiar propietario
chown -R $SYSTEM_USER:$SYSTEM_USER $INSTALL_DIR

# Construir imagen Docker
echo "🐳 Construyendo imagen Docker..."
sudo -u $SYSTEM_USER docker-compose build

# Crear servicio systemd
echo "🔧 Configurando servicio systemd..."
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
ExecStart=/usr/bin/docker-compose up vehicle-detection
ExecStop=/usr/bin/docker-compose down
Restart=always
RestartSec=10

[Install]
WantedBy=multi-user.target
EOF

# Crear script de control
cat > /usr/local/bin/vehicle-detection-ctl << 'EOF'
#!/bin/bash
INSTALL_DIR="/opt/vehicle-detection"
SYSTEM_USER="vehicle-detection"

case "$1" in
    start) systemctl start vehicle-detection ;;
    stop) systemctl stop vehicle-detection ;;
    restart) systemctl restart vehicle-detection ;;
    status) systemctl status vehicle-detection ;;
    logs) journalctl -u vehicle-detection -f ;;
    docker-logs) cd $INSTALL_DIR && sudo -u $SYSTEM_USER docker-compose logs -f ;;
    health) curl -s http://localhost:8000/api/camera_health ;;
    *) echo "Uso: $0 {start|stop|restart|status|logs|docker-logs|health}" ;;
esac
EOF

chmod +x /usr/local/bin/vehicle-detection-ctl

# Configurar firewall
echo "🔥 Configurando firewall..."
ufw --force enable
ufw allow ssh
ufw allow 8000/tcp

# Descargar modelo
echo "🤖 Descargando modelo..."
cd $INSTALL_DIR/models
sudo -u $SYSTEM_USER wget -q -O yolov8n.onnx https://github.com/ultralytics/assets/releases/download/v0.0.0/yolov8n.onnx || true

# Habilitar e iniciar servicio
systemctl daemon-reload
systemctl enable vehicle-detection.service
systemctl start vehicle-detection

# Esperar inicio
sleep 10

# Verificar instalación
echo ""
echo "🎉 INSTALACIÓN COMPLETADA"
echo "========================="
echo "🌐 URL: http://$(hostname -I | awk '{print $1}'):8000"
echo "🔑 Usuario: admin / Contraseña: admin123"
echo ""
echo "📊 Estado:"
systemctl status vehicle-detection --no-pager -l | head -5
echo ""
echo "🔍 Health Check:"
curl -s http://localhost:8000/api/camera_health 2>/dev/null || echo "API iniciando..."
echo ""
echo "✅ LISTO PARA PRODUCCIÓN"
echo ""
echo "Comandos:"
echo "  vehicle-detection-ctl status"
echo "  vehicle-detection-ctl logs"
echo "  vehicle-detection-ctl health"
EOF