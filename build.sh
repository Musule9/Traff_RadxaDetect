#!/bin/bash
# ============================================================================
# SCRIPT DE CONSTRUCCIÓN AUTOMATIZADA
# Sistema de Detección Vehicular - Radxa Rock 5T
# ============================================================================

set -e

echo "🚀 CONSTRUCCIÓN AUTOMATIZADA DEL SISTEMA"
echo "========================================"

# Variables
PROJECT_DIR="/opt/vehicle-detection"
BACKUP_DIR="/opt/vehicle-detection-backup-$(date +%Y%m%d_%H%M%S)"
CONTAINER_NAME="vehicle-detection-prod"
IMAGE_NAME="vehicle-detection:latest"

# Funciones
log_info() {
    echo "ℹ️  $1"
}

log_success() {
    echo "✅ $1"
}

log_error() {
    echo "❌ $1"
}

log_warning() {
    echo "⚠️  $1"
}

# Verificar prerrequisitos específicos para Radxa Rock 5T
check_prerequisites() {
    log_info "Verificando prerrequisitos para Radxa Rock 5T..."
    
    # Verificar arquitectura
    ARCH=$(uname -m)
    if [[ "$ARCH" != "aarch64" && "$ARCH" != "arm64" ]]; then
        log_warning "Arquitectura detectada: $ARCH (esperada: aarch64/arm64)"
        log_warning "Este sistema está optimizado para Radxa Rock 5T"
    else
        log_success "Arquitectura ARM64 detectada: $ARCH"
    fi
    
    # Verificar Docker
    if ! command -v docker &> /dev/null; then
        log_error "Docker no está instalado"
        log_info "Para instalar Docker en Radxa Rock 5T:"
        log_info "curl -fsSL https://get.docker.com | sh"
        exit 1
    fi
    
    # Verificar Docker Compose
    if ! command -v docker-compose &> /dev/null; then
        log_error "Docker Compose no está instalado"
        log_info "Para instalar Docker Compose:"
        log_info "sudo apt update && sudo apt install docker-compose-plugin"
        exit 1
    fi
    
    # Verificar memoria disponible (RK3588 necesita al menos 2GB)
    AVAILABLE_MEM=$(free -m | awk 'NR==2{printf "%.0f", $7}')
    if [ "$AVAILABLE_MEM" -lt 1024 ]; then
        log_warning "Memoria disponible: ${AVAILABLE_MEM}MB (recomendado: >2GB)"
        log_warning "El sistema podría funcionar lento con poca memoria"
    else
        log_success "Memoria disponible: ${AVAILABLE_MEM}MB"
    fi
    
    # Verificar espacio en disco
    AVAILABLE_SPACE=$(df -BG . | awk 'NR==2 {print $4}' | sed 's/G//')
    if [ "$AVAILABLE_SPACE" -lt 5 ]; then
        log_warning "Espacio disponible: ${AVAILABLE_SPACE}GB (recomendado: >10GB)"
    else
        log_success "Espacio disponible: ${AVAILABLE_SPACE}GB"
    fi
    
    # Verificar GPU/NPU (específico para RK3588)
    if [ -d "/dev/dri" ]; then
        log_success "GPU Mali detectada en /dev/dri"
    else
        log_warning "GPU Mali no detectada - funcionalidad de aceleración limitada"
    fi
    
    # Verificar usuario
    if [ "$EUID" -ne 0 ]; then
        log_warning "Ejecutándose como usuario no-root"
        log_info "Si hay problemas de permisos, ejecuta con: sudo $0"
    fi
    
    log_success "Prerrequisitos verificados"
}

# Crear backup del sistema actual
create_backup() {
    log_info "Creando backup del sistema actual..."
    
    if [ -d "$PROJECT_DIR" ]; then
        # Parar contenedor si está ejecutándose
        docker stop $CONTAINER_NAME 2>/dev/null || true
        
        # Crear backup
        cp -r "$PROJECT_DIR" "$BACKUP_DIR"
        log_success "Backup creado en: $BACKUP_DIR"
    else
        log_warning "No hay sistema previo para hacer backup"
    fi
}

# Crear estructura de directorios
create_structure() {
    log_info "Creando estructura de directorios..."
    
    cd "$PROJECT_DIR"
    
    # Crear directorios principales
    mkdir -p {app/{core,services,api,utils},frontend/src,config,data,models,logs,scripts,tests,docs}
    
    # Crear archivos __init__.py
    touch app/__init__.py
    touch app/core/__init__.py
    touch app/services/__init__.py
    touch app/api/__init__.py
    touch app/utils/__init__.py
    
    log_success "Estructura de directorios creada"
}

# Configurar permisos
setup_permissions() {
    log_info "Configurando permisos..."
    
    # Crear usuario del sistema si no existe
    if ! id "vehicle-detection" &>/dev/null; then
        useradd -r -s /bin/false vehicle-detection
        log_success "Usuario 'vehicle-detection' creado"
    fi
    
    # Configurar permisos
    chown -R vehicle-detection:vehicle-detection "$PROJECT_DIR"
    chmod -R 755 "$PROJECT_DIR"
    chmod -R 777 "$PROJECT_DIR"/{data,logs,models}
    
    log_success "Permisos configurados"
}

# Limpiar sistema anterior
cleanup_previous() {
    log_info "Limpiando sistema anterior..."
    
    # Parar y eliminar contenedor
    docker stop $CONTAINER_NAME 2>/dev/null || true
    docker rm $CONTAINER_NAME 2>/dev/null || true
    
    # Eliminar imagen anterior
    docker rmi $IMAGE_NAME 2>/dev/null || true
    
    # Limpiar cache de Docker
    docker system prune -f
    
    log_success "Sistema anterior limpiado"
}

# Construir imagen
build_image() {
    log_info "Construyendo imagen Docker..."
    
    cd "$PROJECT_DIR"
    
    # Construcción con logs detallados
    docker-compose build --no-cache --pull --progress=plain
    
    log_success "Imagen construida exitosamente"
}

# Iniciar sistema
start_system() {
    log_info "Iniciando sistema..."
    
    cd "$PROJECT_DIR"
    
    # Iniciar contenedor
    docker-compose up -d
    
    # Esperar a que esté listo
    log_info "Esperando que el sistema esté listo..."
    sleep 30
    
    # Verificar estado
    if docker ps | grep -q $CONTAINER_NAME; then
        log_success "Sistema iniciado correctamente"
        
        # Mostrar logs iniciales
        echo ""
        log_info "Logs iniciales del sistema:"
        docker logs $CONTAINER_NAME --tail 20
        
    else
        log_error "Error al iniciar el sistema"
        docker logs $CONTAINER_NAME
        exit 1
    fi
}

# Verificar funcionamiento
verify_system() {
    log_info "Verificando funcionamiento del sistema..."
    
    # Esperar un poco más
    sleep 15
    
    # Verificar API
    if curl -f http://localhost:8000/api/camera_health &>/dev/null; then
        log_success "API funcionando correctamente"
    else
        log_warning "API no responde aún, puede estar iniciando..."
    fi
    
    # Verificar estructura interna
    log_info "Verificando estructura interna..."
    docker exec $CONTAINER_NAME /bin/bash -c "
        echo 'Archivos Python en /app:'
        find /app -name '*.py' | head -10
        echo ''
        echo 'Estructura de directorios:'
        ls -la /app/
        echo ''
        echo 'Configuraciones:'
        ls -la /app/config/
    "
}

# Mostrar información final
show_final_info() {
    echo ""
    echo "🎉 CONSTRUCCIÓN COMPLETADA"
    echo "=========================="
    echo ""
    echo "🌐 Sistema disponible en:"
    echo "   - Web Interface: http://localhost:8000"
    echo "   - API Docs: http://localhost:8000/docs"
    echo "   - Health Check: http://localhost:8000/api/camera_health"
    echo ""
    echo "📋 Comandos útiles:"
    echo "   - Ver logs: docker logs $CONTAINER_NAME -f"
    echo "   - Acceder al contenedor: docker exec -it $CONTAINER_NAME /bin/bash"
    echo "   - Parar sistema: docker-compose down"
    echo "   - Reiniciar: docker-compose restart"
    echo ""
    echo "📁 Directorios importantes:"
    echo "   - Datos: $PROJECT_DIR/data"
    echo "   - Logs: $PROJECT_DIR/logs" 
    echo "   - Modelos: $PROJECT_DIR/models"
    echo "   - Backup: $BACKUP_DIR"
    echo ""
    log_success "Sistema listo para usar"
}

# Función principal
main() {
    echo "Iniciando construcción del sistema de detección vehicular..."
    echo "Directorio del proyecto: $PROJECT_DIR"
    echo ""
    
    # Ejecutar pasos
    check_prerequisites
    create_backup
    create_structure
    setup_permissions
    cleanup_previous
    build_image
    start_system
    verify_system
    show_final_info
    
    echo ""
    log_success "🚀 ¡Sistema de detección vehicular construido y funcionando!"
}

# Manejo de errores
trap 'log_error "Error en línea $LINENO. Saliendo..."; exit 1' ERR

# Ejecutar script principal
main "$@"