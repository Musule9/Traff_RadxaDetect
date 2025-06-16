.PHONY: help install build start stop restart status logs clean test lint format backup restore update

# Variables
PROJECT_NAME = vehicle-detection-system
DOCKER_COMPOSE = docker-compose
INSTALL_DIR = /opt/vehicle-detection
BACKUP_DIR = $(INSTALL_DIR)/backups
TIMESTAMP = $(shell date +%Y%m%d_%H%M%S)

# Colores para output
RED = \033[0;31m
GREEN = \033[0;32m
YELLOW = \033[1;33m
NC = \033[0m # No Color

help: ## Mostrar ayuda
	@echo "$(GREEN)Sistema de Detección Vehicular - Comandos Disponibles$(NC)"
	@echo "========================================================="
	@grep -E '^[a-zA-Z_-]+:.*?## .*$' $(MAKEFILE_LIST) | sort | awk 'BEGIN {FS = ":.*?## "}; {printf "$(YELLOW)%-20s$(NC) %s\n", $1, $2}'

install: ## Instalar sistema completo
	@echo "$(GREEN)Instalando Sistema de Detección Vehicular...$(NC)"
	sudo chmod +x deploy/install_radxa.sh
	sudo ./deploy/install_radxa.sh

build: ## Construir imágenes Docker
	@echo "$(GREEN)Construyendo imágenes Docker...$(NC)"
	$(DOCKER_COMPOSE) build --no-cache

build-dev: ## Construir imágenes para desarrollo
	@echo "$(GREEN)Construyendo imágenes de desarrollo...$(NC)"
	$(DOCKER_COMPOSE) --profile development build --no-cache

start: ## Iniciar servicios
	@echo "$(GREEN)Iniciando servicios...$(NC)"
	$(DOCKER_COMPOSE) up -d

start-dev: ## Iniciar en modo desarrollo
	@echo "$(GREEN)Iniciando servicios de desarrollo...$(NC)"
	$(DOCKER_COMPOSE) --profile development up -d

stop: ## Detener servicios
	@echo "$(YELLOW)Deteniendo servicios...$(NC)"
	$(DOCKER_COMPOSE) down

restart: ## Reiniciar servicios
	@echo "$(YELLOW)Reiniciando servicios...$(NC)"
	$(DOCKER_COMPOSE) restart

status: ## Ver estado de servicios
	@echo "$(GREEN)Estado de servicios:$(NC)"
	$(DOCKER_COMPOSE) ps
	@echo "\n$(GREEN)Uso de recursos:$(NC)"
	docker stats --no-stream

logs: ## Ver logs en tiempo real
	@echo "$(GREEN)Logs del sistema:$(NC)"
	$(DOCKER_COMPOSE) logs -f

logs-app: ## Ver logs de la aplicación
	@echo "$(GREEN)Logs de la aplicación:$(NC)"
	$(DOCKER_COMPOSE) logs -f vehicle-detection

shell: ## Acceder a shell del contenedor
	@echo "$(GREEN)Accediendo al contenedor...$(NC)"
	$(DOCKER_COMPOSE) exec vehicle-detection /bin/bash

shell-dev: ## Acceder a shell de desarrollo
	@echo "$(GREEN)Accediendo al contenedor de desarrollo...$(NC)"
	$(DOCKER_COMPOSE) exec vehicle-detection-dev /bin/bash

test: ## Ejecutar tests
	@echo "$(GREEN)Ejecutando tests...$(NC)"
	$(DOCKER_COMPOSE) exec vehicle-detection python -m pytest tests/ -v

test-coverage: ## Ejecutar tests con coverage
	@echo "$(GREEN)Ejecutando tests con coverage...$(NC)"
	$(DOCKER_COMPOSE) exec vehicle-detection python -m pytest tests/ --cov=app --cov-report=html

lint: ## Ejecutar linting
	@echo "$(GREEN)Ejecutando linting...$(NC)"
	$(DOCKER_COMPOSE) exec vehicle-detection flake8 app/
	$(DOCKER_COMPOSE) exec vehicle-detection mypy app/

format: ## Formatear código
	@echo "$(GREEN)Formateando código...$(NC)"
	$(DOCKER_COMPOSE) exec vehicle-detection black app/
	$(DOCKER_COMPOSE) exec vehicle-detection isort app/

backup: ## Crear respaldo
	@echo "$(GREEN)Creando respaldo...$(NC)"
	mkdir -p $(BACKUP_DIR)
	tar -czf $(BACKUP_DIR)/backup_$(TIMESTAMP).tar.gz \
		-C $(INSTALL_DIR) data config --exclude='*.log'
	@echo "$(GREEN)Respaldo creado: $(BACKUP_DIR)/backup_$(TIMESTAMP).tar.gz$(NC)"

backup-db: ## Respaldar solo base de datos
	@echo "$(GREEN)Respaldando base de datos...$(NC)"
	mkdir -p $(BACKUP_DIR)
	tar -czf $(BACKUP_DIR)/db_backup_$(TIMESTAMP).tar.gz \
		-C $(INSTALL_DIR) data
	@echo "$(GREEN)Respaldo de DB creado: $(BACKUP_DIR)/db_backup_$(TIMESTAMP).tar.gz$(NC)"

restore: ## Restaurar desde respaldo (requiere BACKUP_FILE)
	@if [ -z "$(BACKUP_FILE)" ]; then \
		echo "$(RED)Error: Especifique BACKUP_FILE=ruta_del_respaldo$(NC)"; \
		exit 1; \
	fi
	@echo "$(YELLOW)Restaurando desde $(BACKUP_FILE)...$(NC)"
	$(DOCKER_COMPOSE) down
	tar -xzf $(BACKUP_FILE) -C $(INSTALL_DIR)
	$(DOCKER_COMPOSE) up -d
	@echo "$(GREEN)Restauración completada$(NC)"

clean: ## Limpiar sistema
	@echo "$(YELLOW)Limpiando sistema...$(NC)"
	$(DOCKER_COMPOSE) down --volumes --remove-orphans
	docker system prune -f
	docker volume prune -f

clean-logs: ## Limpiar logs antiguos
	@echo "$(YELLOW)Limpiando logs antiguos...$(NC)"
	find $(INSTALL_DIR)/logs -name "*.log" -mtime +7 -delete 2>/dev/null || true
	journalctl --vacuum-time=7d

clean-data: ## Limpiar datos antiguos (CUIDADO!)
	@echo "$(RED)¿Está seguro de eliminar datos antiguos? [y/N]$(NC)"
	@read -r response && if [ "$response" = "y" ]; then \
		$(DOCKER_COMPOSE) exec vehicle-detection python -c "import asyncio; from app.core.database import DatabaseManager; asyncio.run(DatabaseManager().cleanup_old_databases())"; \
		echo "$(GREEN)Limpieza completada$(NC)"; \
	else \
		echo "$(YELLOW)Operación cancelada$(NC)"; \
	fi

update: ## Actualizar sistema
	@echo "$(GREEN)Actualizando sistema...$(NC)"
	git pull origin main
	$(DOCKER_COMPOSE) pull
	$(DOCKER_COMPOSE) build --no-cache
	$(DOCKER_COMPOSE) down
	$(DOCKER_COMPOSE) up -d
	@echo "$(GREEN)Actualización completada$(NC)"

monitor: ## Monitorear sistema
	@echo "$(GREEN)Iniciando monitoreo del sistema...$(NC)"
	@echo "Presione Ctrl+C para salir"
	@while true; do \
		clear; \
		echo "$(GREEN)=== Estado del Sistema $(shell date) ===$(NC)"; \
		$(DOCKER_COMPOSE) ps; \
		echo "\n$(GREEN)=== Uso de Recursos ===$(NC)"; \
		docker stats --no-stream | head -5; \
		echo "\n$(GREEN)=== Últimos Logs ===$(NC)"; \
		$(DOCKER_COMPOSE) logs --tail=5; \
		sleep 10; \
	done

health: ## Verificar salud del sistema
	@echo "$(GREEN)Verificando salud del sistema...$(NC)"
	@curl -s http://localhost:8000/api/camera_health | jq . || echo "$(RED)Error: API no disponible$(NC)"
	@echo "\n$(GREEN)Estado de servicios:$(NC)"
	systemctl is-active vehicle-detection || echo "$(RED)Servicio systemd no activo$(NC)"

install-dev: ## Instalar dependencias de desarrollo
	@echo "$(GREEN)Instalando dependencias de desarrollo...$(NC)"
	$(DOCKER_COMPOSE) exec vehicle-detection pip install -r requirements-dev.txt

jupyter: ## Iniciar Jupyter Lab
	@echo "$(GREEN)Iniciando Jupyter Lab...$(NC)"
	$(DOCKER_COMPOSE) --profile development up -d vehicle-detection-dev
	@echo "$(GREEN)Jupyter disponible en: http://localhost:8888$(NC)"

mock-controller: ## Iniciar controladora simulada
	@echo "$(GREEN)Iniciando controladora simulada...$(NC)"
	$(DOCKER_COMPOSE) --profile testing up -d mock-controller
	@echo "$(GREEN)Controladora simulada en: http://localhost:8080$(NC)"

performance: ## Análisis de rendimiento
	@echo "$(GREEN)Analizando rendimiento...$(NC)"
	$(DOCKER_COMPOSE) exec vehicle-detection python -m cProfile -s cumulative -m app.core.video_processor

security-scan: ## Escaneo de seguridad
	@echo "$(GREEN)Ejecutando escaneo de seguridad...$(NC)"
	docker run --rm -v $(PWD):/app securecodewarrior/bandit bandit -r /app/app/

docs: ## Generar documentación
	@echo "$(GREEN)Generando documentación...$(NC)"
	$(DOCKER_COMPOSE) exec vehicle-detection sphinx-build -b html docs/ docs/_build/html/

# Comandos de conveniencia
up: start ## Alias para start
down: stop ## Alias para stop
ps: status ## Alias para status