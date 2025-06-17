#!/bin/bash

# Control simple del sistema Vehicle Detection

case "$1" in
    start)
        echo "🚀 Iniciando sistema..."
        docker start vehicle-detection-production 2>/dev/null || \
        docker run -d --name vehicle-detection-production \
          --restart unless-stopped --privileged -p 8000:8000 \
          -v $(pwd)/data:/app/data -v $(pwd)/config:/app/config \
          -v $(pwd)/models:/app/models -v $(pwd)/logs:/app/logs \
          -v /dev:/dev -v /sys:/sys:ro \
          -v /usr/lib/aarch64-linux-gnu:/usr/lib/aarch64-linux-gnu:ro \
          --device=/dev/dri --device=/dev/mali0 --device=/dev/dma_heap \
          --device=/dev/rga --device=/dev/mpp_service \
          -e USE_RKNN=1 -e PYTHONPATH=/app \
          vehicle-detection-final
        
        sleep 3
        if curl -s http://localhost:8000/api/camera_health > /dev/null; then
            echo "✅ Sistema iniciado - http://$(hostname -I | awk '{print $1}'):8000"
        else
            echo "⚠️ Sistema iniciando..."
        fi
        ;;
        
    stop)
        echo "⏹️ Parando sistema..."
        docker stop vehicle-detection-production
        echo "✅ Sistema parado"
        ;;
        
    restart)
        echo "🔄 Reiniciando sistema..."
        docker restart vehicle-detection-production
        sleep 3
        echo "✅ Sistema reiniciado"
        ;;
        
    status)
        echo "📊 Estado del sistema:"
        if docker ps | grep -q vehicle-detection-production; then
            echo "✅ RUNNING"
            docker ps | grep vehicle-detection-production
        else
            echo "❌ STOPPED"
        fi
        ;;
        
    logs)
        echo "📋 Logs del sistema:"
        docker logs -f vehicle-detection-production
        ;;
        
    health)
        echo "🏥 Health check:"
        curl -s http://localhost:8000/api/camera_health | python3 -c "import sys, json; print(json.dumps(json.load(sys.stdin), indent=2))" 2>/dev/null || echo "Sistema no responde"
        ;;
        
    url)
        IP=$(hostname -I | awk '{print $1}')
        echo "🌐 URLs del sistema:"
        echo "   Principal: http://$IP:8000"
        echo "   API Docs:  http://$IP:8000/docs"
        echo "   Health:    http://$IP:8000/api/camera_health"
        ;;
        
    *)
        echo "🎮 Control del Sistema Vehicle Detection"
        echo "========================================"
        echo ""
        echo "USO: $0 [COMANDO]"
        echo ""
        echo "COMANDOS:"
        echo "  start    - Iniciar sistema"
        echo "  stop     - Parar sistema"
        echo "  restart  - Reiniciar sistema"
        echo "  status   - Ver estado"
        echo "  logs     - Ver logs"
        echo "  health   - Health check"
        echo "  url      - Mostrar URLs"
        echo ""
        ;;
esac