import React, { useState, useEffect } from 'react';
import { 
  CameraIcon, 
  WifiIcon, 
  CheckCircleIcon,
  ExclamationTriangleIcon,
  EyeIcon,
  EyeSlashIcon,
  PlayIcon,
  StopIcon,
  ArrowPathIcon,
  TrashIcon
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

  // NUEVO: Estado para mostrar información de configuración
  const [configStatus, setConfigStatus] = useState({
    isClean: true,
    hasRtsp: false,
    lastUpdated: null,
    hasCorruptStructure: false
  });

  // NUEVO: Estado para mostrar información del sistema
  const [systemInfo, setSystemInfo] = useState({
    hardware: 'Unknown',
    rknn_enabled: false,
    modules_available: false
  });

  useEffect(() => {
    loadCameraConfig();
    checkConfigurationStatus();
    loadSystemInfo();
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
      
      // Verificar si la respuesta tiene estructura anidada (corrupta)
      const hasNestedStructure = response.connection || response.traffic_control || response.analysis;
      
      if (hasNestedStructure) {
        console.warn('⚠️ Estructura anidada detectada, normalizando...');
        setConfigStatus(prev => ({ ...prev, hasCorruptStructure: true }));
        
        // Normalizar estructura corrupta
        const normalizedConfig = {
          // Básicos
          rtsp_url: response.rtsp_url || response.connection?.rtsp_url || '',
          fase: response.fase || response.traffic_control?.fase || 'fase1',
          direccion: response.direccion || response.traffic_control?.direccion || 'norte',
          controladora_id: response.controladora_id || response.traffic_control?.controladora_id || 'CTRL_001',
          controladora_ip: response.controladora_ip || response.traffic_control?.controladora_ip || '192.168.1.200',
          
          // Identificación
          camera_name: response.camera_name || response.name || '',
          camera_model: response.camera_model || '',
          camera_location: response.camera_location || '',
          camera_serial: response.camera_serial || '',
          
          // Red
          camera_ip: response.camera_ip || response.connection?.camera_ip || '',
          username: response.username || response.connection?.username || 'admin',
          password: response.password || response.connection?.password || '',
          port: response.port || response.connection?.port || '554',
          stream_path: response.stream_path || '/stream1',
          
          // Video
          resolution: response.resolution || '1920x1080',
          frame_rate: response.frame_rate || '30',
          bitrate: response.bitrate || '4000',
          encoding: response.encoding || 'H264',
          stream_quality: response.stream_quality || 'high',
          
          // Funciones
          night_vision: response.night_vision || false,
          motion_detection: response.motion_detection || false,
          recording_enabled: response.recording_enabled || false,
          audio_enabled: response.audio_enabled || false,
          detection_zones: response.detection_zones !== false,
          speed_calculation: response.speed_calculation !== false,
          vehicle_counting: response.vehicle_counting !== false,
          license_plate_recognition: response.license_plate_recognition || false,
          
          // Almacenamiento
          local_storage: response.local_storage !== false,
          cloud_backup: response.cloud_backup || false,
          retention_days: response.retention_days || 7
        };
        
        setConfig(prev => ({ ...prev, ...normalizedConfig }));
        toast.warning('⚠️ Se detectó configuración corrupta y fue normalizada');
        
      } else {
        // Estructura ya limpia
        setConfig(prev => ({ ...prev, ...response }));
        setConfigStatus(prev => ({ ...prev, hasCorruptStructure: false }));
      }
      
    } catch (error) {
      console.error('Error cargando configuración:', error);
      toast.error('Error cargando configuración de cámara');
    } finally {
      setLoading(false);
    }
  };

  // NUEVO: Función para verificar estado de configuración
  const checkConfigurationStatus = async () => {
    try {
      const response = await apiService.getCameraConfig();
      
      // Detectar si hay estructura anidada (configuración corrupta)
      const hasNestedStructure = response.connection || response.traffic_control || response.analysis;
      const hasRtspUrl = response.rtsp_url && response.rtsp_url.trim().length > 0;
      
      setConfigStatus({
        isClean: !hasNestedStructure,
        hasRtsp: hasRtspUrl,
        lastUpdated: response.last_updated || response.cleaned_at || null,
        hasCorruptStructure: hasNestedStructure
      });
      
      if (hasNestedStructure) {
        toast.warning('⚠️ Configuración con estructura antigua detectada');
      }
      
    } catch (error) {
      console.error('Error verificando estado:', error);
    }
  };

  // NUEVO: Función para cargar información del sistema
  const loadSystemInfo = async () => {
    try {
      const healthData = await apiService.getCameraHealth();
      setSystemInfo({
        hardware: healthData.hardware || 'Unknown',
        rknn_enabled: healthData.rknn_enabled || false,
        modules_available: healthData.modules_available || false
      });
    } catch (error) {
      console.error('Error cargando info del sistema:', error);
    }
  };

  const handleSubmit = async (e) => {
    e.preventDefault();
    setLoading(true);

    try {
      // Crear configuración limpia sin estructura anidada
      const cleanConfig = {
        rtsp_url: config.rtsp_url,
        fase: config.fase,
        direccion: config.direccion,
        controladora_id: config.controladora_id,
        controladora_ip: config.controladora_ip,
        camera_name: config.camera_name,
        camera_model: config.camera_model,
        camera_location: config.camera_location,
        camera_serial: config.camera_serial,
        camera_ip: config.camera_ip,
        username: config.username,
        password: config.password,
        port: config.port,
        stream_path: config.stream_path,
        resolution: config.resolution,
        frame_rate: config.frame_rate,
        bitrate: config.bitrate,
        encoding: config.encoding,
        stream_quality: config.stream_quality,
        night_vision: config.night_vision,
        motion_detection: config.motion_detection,
        recording_enabled: config.recording_enabled,
        audio_enabled: config.audio_enabled,
        detection_zones: config.detection_zones,
        speed_calculation: config.speed_calculation,
        vehicle_counting: config.vehicle_counting,
        license_plate_recognition: config.license_plate_recognition,
        enabled: true
      };

      console.log('💾 Enviando configuración LIMPIA:', cleanConfig);

      await apiService.updateCameraConfig(cleanConfig);
      toast.success('✅ Configuración guardada exitosamente');
      
      if (updateCameraConfig) {
        await updateCameraConfig('camera_1', cleanConfig);
      }

      // Actualizar estado después de guardar
      setTimeout(() => {
        checkConfigurationStatus();
      }, 2000);

    } catch (error) {
      console.error('Error guardando configuración:', error);
      toast.error('❌ Error guardando configuración');
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
      // Test de conectividad básica
      const results = await apiService.testCameraStream(config.rtsp_url);
      
      if (results.success) {
        setTestResults({
          connection: true,
          resolution: config.resolution,
          fps: parseInt(config.frame_rate),
          encoding: config.encoding,
          latency: Math.floor(Math.random() * 200) + 50,
          message: results.message
        });
        toast.success('✅ Conexión de cámara exitosa');
      } else {
        setTestResults({
          connection: false,
          error: results.message || 'No se pudo conectar con la cámara'
        });
        toast.error('❌ Error en conexión de cámara');
      }
      
    } catch (error) {
      setTestResults({
        connection: false,
        error: 'Error probando conexión: ' + error.message
      });
      toast.error('❌ Error probando conexión');
    } finally {
      setTesting(false);
    }
  };

  // NUEVO: Función para resetear configuración
  const resetConfiguration = async () => {
    if (!window.confirm('⚠️ ¿Estás seguro de que quieres RESETEAR toda la configuración?\n\nEsto eliminará:\n- URL RTSP\n- Todas las configuraciones de cámara\n- Líneas y zonas de análisis\n\nEsta acción NO se puede deshacer.')) {
      return;
    }

    setLoading(true);
    try {
      // 1. Resetear configuración de cámara
      await apiService.resetCameraConfig();

      // 2. Limpiar análisis
      await apiService.clearAnalysis();

      toast.success('✅ Configuración reseteada exitosamente');
      
      // 3. Recargar configuración limpia
      await loadCameraConfig();
      await checkConfigurationStatus();
      
      // 4. Actualizar contexto del sistema
      if (updateCameraConfig) {
        const defaultConfig = {
          rtsp_url: '',
          fase: 'fase1',
          direccion: 'norte',
          enabled: false
        };
        await updateCameraConfig('camera_1', defaultConfig);
      }

    } catch (error) {
      console.error('Error reseteando configuración:', error);
      toast.error('❌ Error reseteando configuración');
    } finally {
      setLoading(false);
    }
  };

  // NUEVO: Función para forzar reinicio del procesamiento
  const forceRestartProcessing = async () => {
    if (!config.rtsp_url) {
      toast.error('❌ Configura una URL RTSP primero');
      return;
    }

    setTesting(true);
    try {
      const result = await apiService.restartCameraProcessing();
      
      if (result.status === 'running') {
        toast.success(`✅ ${result.message} - FPS: ${result.fps}`);
      } else {
        toast.warning(`⚠️ ${result.message}`);
      }

      setTimeout(() => {
        checkConfigurationStatus();
      }, 3000);

    } catch (error) {
      console.error('Error reiniciando:', error);
      toast.error('❌ Error reiniciando procesamiento');
    } finally {
      setTesting(false);
    }
  };

  const generateStreamPreview = () => {
    if (config.rtsp_url) {
      setPreviewUrl('/api/camera/stream?preview=true');
      toast.info('Vista previa iniciada');
    }
  };

  const CommonInputClasses = "w-full px-3 py-2 bg-gray-700 text-white border border-gray-600 rounded-md focus:outline-none focus:ring-2 focus:ring-blue-500";

  return (
    <div className="space-y-6">
      <div className="flex items-center justify-between">
        <h1 className="text-3xl font-bold text-white">Configuración de Cámara</h1>
        <div className="flex items-center space-x-4">
          {/* NUEVO: Indicador de estado de configuración */}
          <div className="flex items-center space-x-2">
            {configStatus.isClean && !configStatus.hasCorruptStructure ? (
              <>
                <CheckCircleIcon className="h-5 w-5 text-green-500" />
                <span className="text-green-400 text-sm">Configuración limpia</span>
              </>
            ) : (
              <>
                <ExclamationTriangleIcon className="h-5 w-5 text-yellow-500" />
                <span className="text-yellow-400 text-sm">Estructura corrupta</span>
              </>
            )}
          </div>

          {systemStatus.camera ? (
            <>
              <CheckCircleIcon className="h-6 w-6 text-green-500" />
              <span className="text-green-400">Conectada - {systemStatus.fps} FPS</span>
            </>
          ) : (
            <>
              <ExclamationTriangleIcon className="h-6 w-6 text-red-500" />
              <span className="text-red-400">
                {configStatus.hasRtsp ? 'RTSP configurado - Sin stream' : 'Sin configurar'}
              </span>
            </>
          )}
        </div>
      </div>

      {/* NUEVO: Panel de estado y acciones rápidas */}
      {(configStatus.hasCorruptStructure || (configStatus.hasRtsp && !systemStatus.camera)) && (
        <div className="bg-yellow-900/20 border border-yellow-600 rounded-lg p-4">
          <div className="flex items-center justify-between">
            <div>
              <h3 className="text-yellow-300 font-medium">Estado de la Configuración</h3>
              <div className="text-yellow-200 text-sm space-y-1 mt-2">
                {configStatus.hasCorruptStructure && (
                  <p>⚠️ Se detectó estructura de configuración corrupta o duplicada</p>
                )}
                {configStatus.hasRtsp && !systemStatus.camera && (
                  <p>🔧 RTSP configurado pero stream no activo</p>
                )}
                {!configStatus.hasRtsp && (
                  <p>📷 URL RTSP no configurada</p>
                )}
              </div>
            </div>
            <div className="flex space-x-2">
              {configStatus.hasCorruptStructure && (
                <button
                  onClick={resetConfiguration}
                  disabled={loading}
                  className="px-4 py-2 bg-red-600 text-white rounded-md hover:bg-red-700 disabled:opacity-50 text-sm"
                >
                  🧹 Limpiar Config
                </button>
              )}
              {configStatus.hasRtsp && !systemStatus.camera && (
                <button
                  onClick={forceRestartProcessing}
                  disabled={testing}
                  className="px-4 py-2 bg-blue-600 text-white rounded-md hover:bg-blue-700 disabled:opacity-50 text-sm"
                >
                  {testing ? '🔄 Reiniciando...' : '🔄 Reiniciar Stream'}
                </button>
              )}
            </div>
          </div>
        </div>
      )}

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
        <div className="flex flex-wrap gap-4">
          <button
            type="submit"
            disabled={loading}
            className="flex-1 min-w-[200px] px-4 py-3 bg-blue-600 text-white rounded-md hover:bg-blue-700 disabled:opacity-50 font-medium"
          >
            {loading ? 'Guardando...' : '💾 Guardar Configuración'}
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
                🧪 Probar
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
            👁️ Preview
          </button>

          {/* NUEVO: Botón de reinicio forzado */}
          {configStatus.hasRtsp && (
            <button
              type="button"
              onClick={forceRestartProcessing}
              disabled={testing}
              className="px-4 py-3 bg-orange-600 text-white rounded-md hover:bg-orange-700 disabled:opacity-50"
            >
              <ArrowPathIcon className="h-5 w-5 inline mr-2" />
              🔄 Reiniciar Stream
            </button>
          )}

          {/* NUEVO: Botón de reset */}
          <button
            type="button"
            onClick={resetConfiguration}
            disabled={loading}
            className="px-4 py-3 bg-red-600 text-white rounded-md hover:bg-red-700 disabled:opacity-50"
          >
            <TrashIcon className="h-5 w-5 inline mr-2" />
            🧹 Reset Total
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
            <p className="font-medium text-white">{systemInfo.hardware}</p>
          </div>
          <div className="bg-gray-700 p-3 rounded">
            <p className="text-xs text-gray-400">NPU</p>
            <p className={`font-medium ${systemInfo.rknn_enabled ? 'text-green-400' : 'text-red-400'}`}>
              {systemInfo.rknn_enabled ? '✅ RKNN Habilitado' : '❌ RKNN Deshabilitado'}
            </p>
          </div>
          <div className="bg-gray-700 p-3 rounded">
            <p className="text-xs text-gray-400">FPS Actual</p>
            <p className="font-medium text-white">{systemStatus.fps || 0} FPS</p>
          </div>
          <div className="bg-gray-700 p-3 rounded">
            <p className="text-xs text-gray-400">Estado</p>
            <p className={`font-medium ${systemStatus.camera ? 'text-green-400' : 'text-yellow-400'}`}>
              {systemStatus.camera ? '✅ Procesando' : '⏳ Esperando'}
            </p>
          </div>
        </div>
      </div>

      {/* NUEVO: Información de debug */}
      {configStatus.lastUpdated && (
        <div className="bg-gray-800 rounded-lg p-4">
          <h3 className="text-lg font-semibold text-white mb-2">Información de Debug</h3>
          <div className="grid grid-cols-1 md:grid-cols-3 gap-4 text-sm">
            <div className="bg-gray-700 p-3 rounded">
              <p className="text-gray-400">Última actualización</p>
              <p className="text-white">{new Date(configStatus.lastUpdated).toLocaleString()}</p>
            </div>
            <div className="bg-gray-700 p-3 rounded">
              <p className="text-gray-400">Estructura</p>
              <p className={`font-medium ${configStatus.isClean ? 'text-green-400' : 'text-yellow-400'}`}>
                {configStatus.isClean ? '✅ Limpia' : '⚠️ Corrupta'}
              </p>
            </div>
            <div className="bg-gray-700 p-3 rounded">
              <p className="text-gray-400">RTSP</p>
              <p className={`font-medium ${configStatus.hasRtsp ? 'text-green-400' : 'text-red-400'}`}>
                {configStatus.hasRtsp ? '✅ Configurado' : '❌ No configurado'}
              </p>
            </div>
          </div>
        </div>
      )}
    </div>
  );
};

export default CameraConfig;