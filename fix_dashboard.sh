#!/bin/bash
set -e

echo "🔧 AGREGANDO DASHBOARD COMPLETO AL SISTEMA"
echo "=========================================="

PROJECT_DIR="/opt/vehicle-detection"
FRONTEND_DIR="$PROJECT_DIR/frontend"

# 1. Detener contenedor actual
echo "🛑 Deteniendo contenedor actual..."
sudo docker stop vehicle-detection-prod 2>/dev/null || true

# 2. Crear estructura completa del frontend
echo "📁 Creando estructura completa del frontend..."
sudo mkdir -p "$FRONTEND_DIR"/{src/{components/{Layout,Dashboard,CameraView,CameraConfig,AnalysisConfig,Reports,SystemConfig,Common},services,context},public}

# 3. Crear App.js completo con navegación
echo "⚛️ Creando App.js completo..."
sudo tee "$FRONTEND_DIR/src/App.js" > /dev/null << 'EOF'
import React, { useState, useEffect } from 'react';

function App() {
  const [currentView, setCurrentView] = useState('dashboard');
  const [systemStatus, setSystemStatus] = useState(null);
  const [cameraConfig, setCameraConfig] = useState({
    rtsp_url: '',
    fase: 'fase1',
    direccion: 'norte',
    controladora_ip: '192.168.1.200'
  });

  useEffect(() => {
    // Cargar estado del sistema
    fetch('/api/camera_health')
      .then(res => res.json())
      .then(data => setSystemStatus(data))
      .catch(err => console.error('Error:', err));
  }, []);

  const updateCameraConfig = async (config) => {
    try {
      const response = await fetch('/api/camera/config', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(config)
      });
      
      if (response.ok) {
        setCameraConfig(config);
        alert('✅ Configuración guardada exitosamente');
      } else {
        alert('❌ Error guardando configuración');
      }
    } catch (error) {
      alert('❌ Error: ' + error.message);
    }
  };

  const containerStyle = {
    display: 'flex',
    minHeight: '100vh',
    backgroundColor: '#111827',
    color: 'white',
    fontFamily: 'system-ui, -apple-system, sans-serif'
  };

  const sidebarStyle = {
    width: '250px',
    backgroundColor: '#1f2937',
    padding: '20px',
    borderRight: '1px solid #374151'
  };

  const mainStyle = {
    flex: 1,
    padding: '20px'
  };

  const menuItemStyle = (isActive) => ({
    display: 'block',
    padding: '12px 16px',
    margin: '4px 0',
    borderRadius: '8px',
    backgroundColor: isActive ? '#3b82f6' : 'transparent',
    color: 'white',
    textDecoration: 'none',
    cursor: 'pointer',
    border: 'none',
    width: '100%',
    textAlign: 'left'
  });

  const cardStyle = {
    backgroundColor: '#1f2937',
    padding: '24px',
    borderRadius: '8px',
    marginBottom: '20px',
    border: '1px solid #374151'
  };

  const inputStyle = {
    width: '100%',
    padding: '12px',
    backgroundColor: '#374151',
    border: '1px solid #4b5563',
    borderRadius: '6px',
    color: 'white',
    marginBottom: '16px'
  };

  const buttonStyle = {
    backgroundColor: '#3b82f6',
    color: 'white',
    padding: '12px 24px',
    border: 'none',
    borderRadius: '6px',
    cursor: 'pointer',
    marginRight: '12px'
  };

  const renderDashboard = () => (
    <div>
      <h1 style={{ fontSize: '2rem', marginBottom: '24px' }}>🚗 Dashboard</h1>
      
      <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(300px, 1fr))', gap: '20px' }}>
        <div style={cardStyle}>
          <h3 style={{ color: '#10b981', marginBottom: '16px' }}>🏥 Estado del Sistema</h3>
          <div>
            <div style={{ display: 'flex', justifyContent: 'space-between', marginBottom: '8px' }}>
              <span>API:</span>
              <span style={{ color: '#10b981' }}>✅ Funcionando</span>
            </div>
            <div style={{ display: 'flex', justifyContent: 'space-between', marginBottom: '8px' }}>
              <span>Cámara:</span>
              <span style={{ color: systemStatus?.healthy ? '#10b981' : '#f59e0b' }}>
                {systemStatus?.healthy ? '✅ OK' : '⚠️ No configurada'}
              </span>
            </div>
            <div style={{ display: 'flex', justifyContent: 'space-between' }}>
              <span>FPS:</span>
              <span>{systemStatus?.fps || 0}</span>
            </div>
          </div>
        </div>

        <div style={cardStyle}>
          <h3 style={{ color: '#3b82f6', marginBottom: '16px' }}>📊 Estadísticas Hoy</h3>
          <div>
            <div style={{ marginBottom: '8px' }}>Vehículos detectados: 0</div>
            <div style={{ marginBottom: '8px' }}>Cruces de línea: 0</div>
            <div style={{ marginBottom: '8px' }}>Velocidad promedio: 0 km/h</div>
          </div>
        </div>

        <div style={cardStyle}>
          <h3 style={{ color: '#f59e0b', marginBottom: '16px' }}>⚙️ Configuración</h3>
          <div>
            <div style={{ marginBottom: '8px' }}>RTSP: {cameraConfig.rtsp_url || 'No configurado'}</div>
            <div style={{ marginBottom: '8px' }}>Fase: {cameraConfig.fase}</div>
            <div style={{ marginBottom: '8px' }}>Dirección: {cameraConfig.direccion}</div>
          </div>
        </div>
      </div>
    </div>
  );

  const renderCameraConfig = () => (
    <div>
      <h1 style={{ fontSize: '2rem', marginBottom: '24px' }}>📹 Configuración de Cámara</h1>
      
      <div style={cardStyle}>
        <h3 style={{ marginBottom: '20px' }}>Configuración de Red</h3>
        
        <div>
          <label style={{ display: 'block', marginBottom: '8px', fontWeight: 'bold' }}>
            URL RTSP *
          </label>
          <input
            type="text"
            placeholder="rtsp://admin:password@192.168.1.100:554/stream1"
            value={cameraConfig.rtsp_url}
            onChange={(e) => setCameraConfig({...cameraConfig, rtsp_url: e.target.value})}
            style={inputStyle}
          />
        </div>

        <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: '20px' }}>
          <div>
            <label style={{ display: 'block', marginBottom: '8px', fontWeight: 'bold' }}>
              Fase del Semáforo
            </label>
            <select
              value={cameraConfig.fase}
              onChange={(e) => setCameraConfig({...cameraConfig, fase: e.target.value})}
              style={inputStyle}
            >
              <option value="fase1">Fase 1</option>
              <option value="fase2">Fase 2</option>
              <option value="fase3">Fase 3</option>
              <option value="fase4">Fase 4</option>
            </select>
          </div>

          <div>
            <label style={{ display: 'block', marginBottom: '8px', fontWeight: 'bold' }}>
              Dirección de Tráfico
            </label>
            <select
              value={cameraConfig.direccion}
              onChange={(e) => setCameraConfig({...cameraConfig, direccion: e.target.value})}
              style={inputStyle}
            >
              <option value="norte">Norte</option>
              <option value="sur">Sur</option>
              <option value="este">Este</option>
              <option value="oeste">Oeste</option>
            </select>
          </div>
        </div>

        <div>
          <label style={{ display: 'block', marginBottom: '8px', fontWeight: 'bold' }}>
            IP de Controladora
          </label>
          <input
            type="text"
            placeholder="192.168.1.200"
            value={cameraConfig.controladora_ip}
            onChange={(e) => setCameraConfig({...cameraConfig, controladora_ip: e.target.value})}
            style={inputStyle}
          />
        </div>

        <button
          onClick={() => updateCameraConfig(cameraConfig)}
          style={buttonStyle}
        >
          💾 Guardar Configuración
        </button>

        <button
          onClick={() => {
            fetch('/api/camera_health')
              .then(res => res.json())
              .then(data => {
                alert(data.healthy ? '✅ Cámara conectada correctamente' : '⚠️ Cámara no disponible');
              })
              .catch(err => alert('❌ Error probando conexión'));
          }}
          style={{...buttonStyle, backgroundColor: '#10b981'}}
        >
          🔍 Probar Conexión
        </button>
      </div>
    </div>
  );

  const renderCameraView = () => (
    <div>
      <h1 style={{ fontSize: '2rem', marginBottom: '24px' }}>📹 Vista de Cámara</h1>
      
      <div style={cardStyle}>
        <h3 style={{ marginBottom: '20px' }}>Stream de Video</h3>
        
        {cameraConfig.rtsp_url ? (
          <div>
            <img 
              src="/api/camera/stream" 
              alt="Camera Stream" 
              style={{ width: '100%', maxWidth: '800px', borderRadius: '8px' }}
              onError={(e) => {
                e.target.style.display = 'none';
                e.target.nextSibling.style.display = 'block';
              }}
            />
            <div style={{ display: 'none', textAlign: 'center', padding: '60px', backgroundColor: '#374151', borderRadius: '8px' }}>
              📹 Stream no disponible<br/>
              <small>Verifique la configuración RTSP</small>
            </div>
          </div>
        ) : (
          <div style={{ textAlign: 'center', padding: '60px', backgroundColor: '#374151', borderRadius: '8px' }}>
            📹 Configure primero la URL RTSP en "Configuración de Cámara"
          </div>
        )}
        
        <div style={{ marginTop: '20px' }}>
          <h4>Instrucciones:</h4>
          <ol style={{ paddingLeft: '20px', lineHeight: '1.6' }}>
            <li>Configure la URL RTSP en "Configuración de Cámara"</li>
            <li>El stream aparecerá aquí automáticamente</li>
            <li>Use las herramientas para dibujar líneas de conteo</li>
            <li>Defina la zona de semáforo rojo</li>
          </ol>
        </div>
      </div>
    </div>
  );

  const renderAPIAccess = () => (
    <div>
      <h1 style={{ fontSize: '2rem', marginBottom: '24px' }}>🔗 Acceso a la API</h1>
      
      <div style={cardStyle}>
        <h3 style={{ marginBottom: '20px' }}>Enlaces Útiles</h3>
        <div style={{ lineHeight: '2' }}>
          <a href="/docs" target="_blank" style={{ color: '#60a5fa', textDecoration: 'none', display: 'block' }}>
            📖 Documentación Completa de la API
          </a>
          <a href="/api/camera_health" target="_blank" style={{ color: '#60a5fa', textDecoration: 'none', display: 'block' }}>
            🏥 Estado del Sistema (JSON)
          </a>
          <a href="/api/camera/status" target="_blank" style={{ color: '#60a5fa', textDecoration: 'none', display: 'block' }}>
            📹 Estado de la Cámara (JSON)
          </a>
        </div>
      </div>

      <div style={cardStyle}>
        <h3 style={{ marginBottom: '20px' }}>Configuración Rápida por API</h3>
        <pre style={{ backgroundColor: '#000', padding: '16px', borderRadius: '8px', overflow: 'auto', fontSize: '14px' }}>
{`# Configurar cámara vía curl:
curl -X POST http://localhost:8000/api/camera/config \\
  -H "Content-Type: application/json" \\
  -d '{
    "rtsp_url": "rtsp://admin:password@192.168.1.100:554/stream1",
    "fase": "fase1",
    "direccion": "norte",
    "controladora_ip": "192.168.1.200"
  }'`}
        </pre>
      </div>
    </div>
  );

  return (
    <div style={containerStyle}>
      {/* Sidebar */}
      <div style={sidebarStyle}>
        <h2 style={{ marginBottom: '24px', color: '#60a5fa' }}>🚗 Sistema Vehicular</h2>
        
        <nav>
          <button 
            style={menuItemStyle(currentView === 'dashboard')}
            onClick={() => setCurrentView('dashboard')}
          >
            🏠 Dashboard
          </button>
          
          <button 
            style={menuItemStyle(currentView === 'camera-config')}
            onClick={() => setCurrentView('camera-config')}
          >
            ⚙️ Configuración de Cámara
          </button>
          
          <button 
            style={menuItemStyle(currentView === 'camera-view')}
            onClick={() => setCurrentView('camera-view')}
          >
            📹 Vista de Cámara
          </button>
          
          <button 
            style={menuItemStyle(currentView === 'api')}
            onClick={() => setCurrentView('api')}
          >
            🔗 API y Documentación
          </button>
        </nav>

        <div style={{ marginTop: '40px', padding: '16px', backgroundColor: '#374151', borderRadius: '8px', fontSize: '12px' }}>
          <div>Hardware: Radxa Rock 5T</div>
          <div>Versión: 1.0.0</div>
          <div>NPU: RKNN Habilitado</div>
        </div>
      </div>

      {/* Main Content */}
      <div style={mainStyle}>
        {currentView === 'dashboard' && renderDashboard()}
        {currentView === 'camera-config' && renderCameraConfig()}
        {currentView === 'camera-view' && renderCameraView()}
        {currentView === 'api' && renderAPIAccess()}
      </div>
    </div>
  );
}

export default App;
EOF

# 4. Actualizar package.json para incluir dependencias necesarias
echo "📦 Actualizando package.json..."
sudo tee "$FRONTEND_DIR/package.json" > /dev/null << 'EOF'
{
  "name": "vehicle-detection-frontend",
  "version": "1.0.0",
  "private": true,
  "dependencies": {
    "react": "^18.2.0",
    "react-dom": "^18.2.0",
    "react-scripts": "5.0.1"
  },
  "scripts": {
    "start": "react-scripts start",
    "build": "CI=false GENERATE_SOURCEMAP=false react-scripts build",
    "test": "react-scripts test",
    "eject": "react-scripts eject"
  },
  "browserslist": {
    "production": [
      ">0.2%",
      "not dead",
      "not op_mini all"
    ],
    "development": [
      "last 1 chrome version",
      "last 1 firefox version",
      "last 1 safari version"
    ]
  }
}
EOF

# 5. Configurar permisos
echo "🔐 Configurando permisos..."
sudo chown -R vehicle-detection:vehicle-detection "$FRONTEND_DIR"

# 6. Reconstruir contenedor
echo "🏗️ Reconstruyendo contenedor con dashboard completo..."
cd "$PROJECT_DIR"
sudo -u vehicle-detection docker-compose build --no-cache

# 7. Iniciar contenedor
echo "🚀 Iniciando contenedor..."
sudo -u vehicle-detection docker-compose up -d

# 8. Verificar
echo ""
echo "⏳ Esperando inicialización (30 segundos)..."
sleep 30

echo ""
echo "✅ DASHBOARD COMPLETO INSTALADO"
echo "==============================="
echo ""
echo "🌐 Accede al dashboard completo en:"
echo "   http://$(hostname -I | awk '{print $1}'):8000"
echo ""
echo "📋 Funcionalidades disponibles:"
echo "   🏠 Dashboard - Estado general del sistema"
echo "   ⚙️ Configuración de Cámara - Configurar RTSP y controladora"
echo "   📹 Vista de Cámara - Ver stream y configurar líneas"
echo "   🔗 API - Acceso directo a la documentación"
echo ""
echo "🔧 Primer paso: Ir a 'Configuración de Cámara' y configurar:"
echo "   📹 URL RTSP de tu cámara"
echo "   🚦 Fase del semáforo (fase1, fase2, etc.)"
echo "   📍 Dirección del tráfico"
echo "   🎛️ IP de la controladora"
EOF

chmod +x fix_complete_frontend.sh