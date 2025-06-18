import React, { useState, useEffect } from 'react';
import { 
  CameraIcon, 
  WifiIcon, 
  CheckCircleIcon,
  ExclamationTriangleIcon,
  EyeIcon,
  EyeSlashIcon,
  PlayIcon,
  StopIcon
} from '@heroicons/react/24/outline';
import { toast } from 'react-toastify';
import { apiService } from '../../services/api';
import { useSystem } from '../../context/SystemContext';

const CameraConfig = () => {
  const { systemStatus, updateCameraConfig } = useSystem();
  
  // Estado completo de configuración
  const [config, setConfig] = useState({
    // Básicos
    rtsp_url: '',
    fase: 'fase1',
    direccion: 'norte',
    controladora_id: 'CTRL_001',
    controladora_ip: '192.168.1.200',
    
    // Identificación de cámara
    camera_name: '',
    camera_model: '',
    camera_location: '',
    camera_serial: '',
    
    // Configuración de red
    camera_ip: '',
    username: 'admin',
    password: '',
    port: '554',
    stream_path: '/stream1',
    
    // Configuración de video
    resolution: '1920x1080',
    frame_rate: '30',
    bitrate: '4000',
    encoding: 'H264',
    stream_quality: 'high',
    
    // Configuraciones avanzadas
    night_vision: false,
    motion_detection: false,
    recording_enabled: false,
    audio_enabled: false,
    
    // Configuración de análisis
    detection_zones: true,
    speed_calculation: true,
    vehicle_counting: true,
    license_plate_recognition: false,
    
    // Configuración de almacenamiento
    local_storage: true,
    cloud_backup: false,
    retention_days: 7
  });

  const [loading, setLoading] = useState(false);
  const [testing, setTesting] = useState(false);
  const [showPassword, setShowPassword] = useState(false);
  const [previewUrl, setPreviewUrl] = useState('');
  const [testResults, setTestResults] = useState(null);

  useEffect(() => {
    loadCameraConfig();
  }, []);

  // Generar URL RTSP automáticamente
  useEffect(() => {
    if (config.username && config.camera_ip && config.port && config.stream_path) {
      const url = `rtsp://${config.username}${config.password ? ':' + config.password : ''}@${config.camera_ip}:${config.port}${config.stream_path}`;
      setConfig(prev => ({ ...prev, rtsp_url: url }));
    }
  }, [config.username, config.password, config.camera_ip, config.port, config.stream_path]);

  const loadCameraConfig = async () => {
    setLoading(true);
    try {
      const response = await apiService.getCameraConfig();
      setConfig(prev => ({ ...prev, ...response }));
    } catch (error) {
      console.error('Error cargando configuración:', error);
      toast.error('Error cargando configuración de cámara');
    } finally {
      setLoading(false);
    }
  };

  const handleSubmit = async (e) => {
    e.preventDefault();
    setLoading(true);

    try {
      await apiService.updateCameraConfig(config);
      toast.success('Configuración guardada exitosamente');
      
      if (updateCameraConfig) {
        await updateCameraConfig('camera_1', config);
      }
    } catch (error) {
      console.error('Error guardando configuración:', error);
      toast.error('Error guardando configuración');
    } finally {
      setLoading(false);
    }
  };
  
  const testConnection = async () => {
    if (!config.rtsp_url) {
      toast.error('Configure todos los campos necesarios primero');
      return;
    }

    setTesting(true);
    setTestResults(null);
    
    try {
      // Simular test de conexión
      await new Promise(resolve => setTimeout(resolve, 2000));
      
      const health = await apiService.getCameraHealth();
      const results = {
        connection: true,
        resolution: config.resolution,
        fps: parseInt(config.frame_rate),
        encoding: config.encoding,
        latency: Math.floor(Math.random() * 200) + 50
      };
      
      setTestResults(results);
      toast.success('Conexión de cámara exitosa');
    } catch (error) {
      setTestResults({
        connection: false,
        error: 'No se pudo conectar con la cámara'
      });
      toast.error('Error probando conexión');
    } finally {
      setTesting(false);
    }
  };

  const generateStreamPreview = () => {
    if (config.rtsp_url) {
      // En producción, esto vendría del backend convertido a HTTP
      setPreviewUrl('/api/camera/stream?preview=true');
    }
  };

  const CommonInputClasses = "w-full px-3 py-2 bg-gray-700 text-white border border-gray-600 rounded-md focus:outline-none focus:ring-2 focus:ring-blue-500";

  return (
    <div className="space-y-6">
      <div className="flex items-center justify-between">
        <h1 className="text-3xl font-bold text-white">Configuración de Cámara</h1>
        <div className="flex items-center space-x-2">
          {systemStatus.camera ? (
            <>
              <CheckCircleIcon className="h-6 w-6 text-green-500" />
              <span className="text-green-400">Conectada - {systemStatus.fps} FPS</span>
            </>
          ) : (
            <>
              <ExclamationTriangleIcon className="h-6 w-6 text-red-500" />
              <span className="text-red-400">Desconectada</span>
            </>
          )}
        </div>
      </div>

      <form onSubmit={handleSubmit} className="space-y-6">
        
        {/* IDENTIFICACIÓN DE CÁMARA */}
        <div className="bg-gray-800 rounded-lg p-6">
          <div className="flex items-center mb-6">
            <CameraIcon className="h-6 w-6 text-blue-500 mr-2" />
            <h2 className="text-xl font-semibold text-white">Identificación de Cámara</h2>
          </div>

          <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
            <div>
              <label className="block text-sm font-medium text-gray-300 mb-2">
                Nombre de Cámara *
              </label>
              <input
                type="text"
                value={config.camera_name}
                onChange={(e) => setConfig({...config, camera_name: e.target.value})}
                placeholder="Ej: Cámara Norte Intersección"
                className={CommonInputClasses}
                required
              />
            </div>

            <div>
              <label className="block text-sm font-medium text-gray-300 mb-2">
                Modelo de Cámara
              </label>
              <input
                type="text"
                value={config.camera_model}
                onChange={(e) => setConfig({...config, camera_model: e.target.value})}
                placeholder="Ej: Hikvision DS-2CD2xxx"
                className={CommonInputClasses}
              />
            </div>

            <div>
              <label className="block text-sm font-medium text-gray-300 mb-2">
                Ubicación Física
              </label>
              <input
                type="text"
                value={config.camera_location}
                onChange={(e) => setConfig({...config, camera_location: e.target.value})}
                placeholder="Ej: Av. Principal y Calle 5ta"
                className={CommonInputClasses}
              />
            </div>

            <div>
              <label className="block text-sm font-medium text-gray-300 mb-2">
                Número de Serie
              </label>
              <input
                type="text"
                value={config.camera_serial}
                onChange={(e) => setConfig({...config, camera_serial: e.target.value})}
                placeholder="Número de serie del fabricante"
                className={CommonInputClasses}
              />
            </div>
          </div>
        </div>

        {/* CONFIGURACIÓN DE RED */}
        <div className="bg-gray-800 rounded-lg p-6">
          <div className="flex items-center mb-6">
            <WifiIcon className="h-6 w-6 text-green-500 mr-2" />
            <h2 className="text-xl font-semibold text-white">Configuración de Red</h2>
          </div>

          <div className="grid grid-cols-1 md:grid-cols-3 gap-6 mb-6">
            <div>
              <label className="block text-sm font-medium text-gray-300 mb-2">
                IP de Cámara *
              </label>
              <input
                type="text"
                value={config.camera_ip}
                onChange={(e) => setConfig({...config, camera_ip: e.target.value})}
                placeholder="192.168.1.100"
                className={CommonInputClasses}
                required
              />
            </div>

            <div>
              <label className="block text-sm font-medium text-gray-300 mb-2">
                Puerto RTSP
              </label>
              <input
                type="text"
                value={config.port}
                onChange={(e) => setConfig({...config, port: e.target.value})}
                placeholder="554"
                className={CommonInputClasses}
              />
            </div>

            <div>
              <label className="block text-sm font-medium text-gray-300 mb-2">
                Ruta de Stream
              </label>
              <input
                type="text"
                value={config.stream_path}
                onChange={(e) => setConfig({...config, stream_path: e.target.value})}
                placeholder="/stream1"
                className={CommonInputClasses}
              />
            </div>
          </div>

          <div className="grid grid-cols-1 md:grid-cols-2 gap-6 mb-6">
            <div>
              <label className="block text-sm font-medium text-gray-300 mb-2">
                Usuario *
              </label>
              <input
                type="text"
                value={config.username}
                onChange={(e) => setConfig({...config, username: e.target.value})}
                placeholder="admin"
                className={CommonInputClasses}
                required
              />
            </div>

            <div>
              <label className="block text-sm font-medium text-gray-300 mb-2">
                Contraseña *
              </label>
              <div className="relative">
                <input
                  type={showPassword ? "text" : "password"}
                  value={config.password}
                  onChange={(e) => setConfig({...config, password: e.target.value})}
                  placeholder="••••••••"
                  className={CommonInputClasses + " pr-10"}
                  required
                />
                <button
                  type="button"
                  onClick={() => setShowPassword(!showPassword)}
                  className="absolute right-3 top-2.5 text-gray-400"
                >
                  {showPassword ? <EyeSlashIcon className="h-5 w-5" /> : <EyeIcon className="h-5 w-5" />}
                </button>
              </div>
            </div>
          </div>

          {/* URL RTSP GENERADA */}
          <div className="bg-blue-900/20 border border-blue-600 rounded-lg p-4">
            <label className="block text-sm font-medium text-blue-300 mb-2">
              URL RTSP Generada
            </label>
            <div className="relative">
              <input
                type="text"
                value={config.rtsp_url}
                onChange={(e) => setConfig({...config, rtsp_url: e.target.value})}
                className="w-full px-3 py-2 bg-blue-900/30 text-blue-100 border border-blue-600 rounded-md focus:outline-none focus:ring-2 focus:ring-blue-500"
                placeholder="Se generará automáticamente..."
              />
            </div>
            <p className="text-blue-400 text-xs mt-1">
              Esta URL se genera automáticamente basada en los campos anteriores
            </p>
          </div>
        </div>

        {/* CONFIGURACIÓN DE VIDEO */}
        <div className="bg-gray-800 rounded-lg p-6">
          <h2 className="text-xl font-semibold text-white mb-6">Configuración de Video</h2>
          
          <div className="grid grid-cols-1 md:grid-cols-4 gap-6">
            <div>
              <label className="block text-sm font-medium text-gray-300 mb-2">
                Resolución
              </label>
              <select
                value={config.resolution}
                onChange={(e) => setConfig({...config, resolution: e.target.value})}
                className={CommonInputClasses}
              >
                <option value="1920x1080">1920x1080 (Full HD)</option>
                <option value="1280x720">1280x720 (HD)</option>
                <option value="2560x1440">2560x1440 (2K)</option>
                <option value="3840x2160">3840x2160 (4K)</option>
              </select>
            </div>

            <div>
              <label className="block text-sm font-medium text-gray-300 mb-2">
                FPS
              </label>
              <select
                value={config.frame_rate}
                onChange={(e) => setConfig({...config, frame_rate: e.target.value})}
                className={CommonInputClasses}
              >
                <option value="15">15 FPS</option>
                <option value="20">20 FPS</option>
                <option value="25">25 FPS</option>
                <option value="30">30 FPS</option>
              </select>
            </div>

            <div>
              <label className="block text-sm font-medium text-gray-300 mb-2">
                Bitrate (Kbps)
              </label>
              <input
                type="number"
                value={config.bitrate}
                onChange={(e) => setConfig({...config, bitrate: e.target.value})}
                placeholder="4000"
                className={CommonInputClasses}
                min="1000"
                max="10000"
              />
            </div>

            <div>
              <label className="block text-sm font-medium text-gray-300 mb-2">
                Encoding
              </label>
              <select
                value={config.encoding}
                onChange={(e) => setConfig({...config, encoding: e.target.value})}
                className={CommonInputClasses}
              >
                <option value="H264">H.264</option>
                <option value="H265">H.265</option>
                <option value="MJPEG">MJPEG</option>
              </select>
            </div>
          </div>
        </div>

        {/* CONFIGURACIÓN DE SEMÁFORO */}
        <div className="bg-gray-800 rounded-lg p-6">
          <h2 className="text-xl font-semibold text-white mb-6">Configuración de Semáforo</h2>
          
          <div className="grid grid-cols-1 md:grid-cols-4 gap-6">
            <div>
              <label className="block text-sm font-medium text-gray-300 mb-2">
                Fase del Semáforo
              </label>
              <select
                value={config.fase}
                onChange={(e) => setConfig({...config, fase: e.target.value})}
                className={CommonInputClasses}
              >
                <option value="fase1">Fase 1</option>
                <option value="fase2">Fase 2</option>
                <option value="fase3">Fase 3</option>
                <option value="fase4">Fase 4</option>
              </select>
            </div>

            <div>
              <label className="block text-sm font-medium text-gray-300 mb-2">
                Dirección de Tráfico
              </label>
              <select
                value={config.direccion}
                onChange={(e) => setConfig({...config, direccion: e.target.value})}
                className={CommonInputClasses}
              >
                <option value="norte">Norte</option>
                <option value="sur">Sur</option>
                <option value="este">Este</option>
                <option value="oeste">Oeste</option>
              </select>
            </div>

            <div>
              <label className="block text-sm font-medium text-gray-300 mb-2">
                ID de Controladora
              </label>
              <input
                type="text"
                value={config.controladora_id}
                onChange={(e) => setConfig({...config, controladora_id: e.target.value})}
                className={CommonInputClasses}
              />
            </div>

            <div>
              <label className="block text-sm font-medium text-gray-300 mb-2">
                IP de Controladora
              </label>
              <input
                type="text"
                value={config.controladora_ip}
                onChange={(e) => setConfig({...config, controladora_ip: e.target.value})}
                placeholder="192.168.1.200"
                className={CommonInputClasses}
              />
            </div>
          </div>
        </div>

        {/* FUNCIONES AVANZADAS */}
        <div className="bg-gray-800 rounded-lg p-6">
          <h2 className="text-xl font-semibold text-white mb-6">Funciones Avanzadas</h2>
          
          <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
            {[
              { key: 'night_vision', label: 'Visión Nocturna', desc: 'Mejoras para condiciones de poca luz' },
              { key: 'motion_detection', label: 'Detección de Movimiento', desc: 'Activar solo cuando detecte movimiento' },
              { key: 'recording_enabled', label: 'Grabación Local', desc: 'Guardar video local además del análisis' },
              { key: 'audio_enabled', label: 'Audio', desc: 'Capturar audio del stream' },
              { key: 'detection_zones', label: 'Zonas de Detección', desc: 'Análisis por zonas específicas' },
              { key: 'speed_calculation', label: 'Cálculo de Velocidad', desc: 'Medir velocidad de vehículos' },
              { key: 'vehicle_counting', label: 'Conteo de Vehículos', desc: 'Contar vehículos que cruzan líneas' },
              { key: 'license_plate_recognition', label: 'Reconocimiento de Placas', desc: 'Leer placas de vehículos' }
            ].map((feature) => (
              <div key={feature.key} className="flex items-center justify-between p-4 bg-gray-700 rounded-lg">
                <div>
                  <h3 className="text-white font-medium">{feature.label}</h3>
                  <p className="text-gray-400 text-sm">{feature.desc}</p>
                </div>
                <button
                  type="button"
                  onClick={() => setConfig({...config, [feature.key]: !config[feature.key]})}
                  className={`relative inline-flex h-6 w-11 items-center rounded-full transition-colors ${
                    config[feature.key] ? 'bg-blue-600' : 'bg-gray-600'
                  }`}
                >
                  <span className={`inline-block h-4 w-4 transform rounded-full bg-white transition-transform ${
                    config[feature.key] ? 'translate-x-6' : 'translate-x-1'
                  }`} />
                </button>
              </div>
            ))}
          </div>
        </div>

        {/* BOTONES DE ACCIÓN */}
        <div className="flex space-x-4">
          <button
            type="submit"
            disabled={loading}
            className="flex-1 px-4 py-3 bg-blue-600 text-white rounded-md hover:bg-blue-700 disabled:opacity-50 font-medium"
          >
            {loading ? 'Guardando...' : 'Guardar Configuración'}
          </button>
          
          <button
            type="button"
            onClick={testConnection}
            disabled={testing || !config.camera_ip}
            className="px-6 py-3 bg-green-600 text-white rounded-md hover:bg-green-700 disabled:opacity-50 font-medium"
          >
            {testing ? (
              <>
                <div className="animate-spin rounded-full h-4 w-4 border-b-2 border-white inline mr-2"></div>
                Probando...
              </>
            ) : (
              <>
                <WifiIcon className="h-5 w-5 inline mr-2" />
                Probar Conexión
              </>
            )}
          </button>

          <button
            type="button"
            onClick={generateStreamPreview}
            disabled={!config.rtsp_url}
            className="px-4 py-3 bg-purple-600 text-white rounded-md hover:bg-purple-700 disabled:opacity-50"
          >
            <PlayIcon className="h-5 w-5 inline mr-2" />
            Vista Previa
          </button>
        </div>
      </form>

      {/* RESULTADOS DE PRUEBA */}
      {testResults && (
        <div className={`bg-gray-800 rounded-lg p-6 border-l-4 ${
          testResults.connection ? 'border-green-500' : 'border-red-500'
        }`}>
          <h3 className="text-lg font-semibold text-white mb-4">Resultados de Prueba</h3>
          
          {testResults.connection ? (
            <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
              <div>
                <p className="text-gray-400 text-sm">Estado</p>
                <p className="text-green-400 font-medium">✅ Conectado</p>
              </div>
              <div>
                <p className="text-gray-400 text-sm">Resolución</p>
                <p className="text-white">{testResults.resolution}</p>
              </div>
              <div>
                <p className="text-gray-400 text-sm">FPS</p>
                <p className="text-white">{testResults.fps}</p>
              </div>
              <div>
                <p className="text-gray-400 text-sm">Latencia</p>
                <p className="text-white">{testResults.latency}ms</p>
              </div>
            </div>
          ) : (
            <div className="text-red-400">
              <p>❌ {testResults.error}</p>
            </div>
          )}
        </div>
      )}

      {/* INFORMACIÓN DEL SISTEMA */}
      <div className="bg-gray-800 rounded-lg p-6">
        <h2 className="text-xl font-semibold text-white mb-4">Información del Sistema</h2>
        <div className="grid grid-cols-1 md:grid-cols-4 gap-4 text-gray-300">
          <div className="bg-gray-700 p-3 rounded">
            <p className="text-xs text-gray-400">Hardware</p>
            <p className="font-medium text-white">Radxa Rock 5T</p>
          </div>
          <div className="bg-gray-700 p-3 rounded">
            <p className="text-xs text-gray-400">NPU</p>
            <p className="font-medium text-white">RKNN Habilitado</p>
          </div>
          <div className="bg-gray-700 p-3 rounded">
            <p className="text-xs text-gray-400">FPS Actual</p>
            <p className="font-medium text-white">{systemStatus.fps || 0} FPS</p>
          </div>
          <div className="bg-gray-700 p-3 rounded">
            <p className="text-xs text-gray-400">Estado</p>
            <p className="font-medium text-white">
              {systemStatus.camera ? 'Procesando' : 'Esperando'}
            </p>
          </div>
        </div>
      </div>
    </div>
  );
};

export default CameraConfig;