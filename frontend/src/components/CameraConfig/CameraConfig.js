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
  ArrowPathIcon
} from '@heroicons/react/24/outline';
import { toast } from 'react-toastify';
import { apiService } from '../../services/api';
import { useSystem } from '../../context/SystemContext';

const CameraConfig = () => {
  const { systemStatus, updateCameraConfig } = useSystem();
  
  // Estado principal de configuración - SIMPLIFICADO
  const [config, setConfig] = useState({
    rtsp_url: '',
    fase: 'fase1',
    direccion: 'norte',
    controladora_id: 'CTRL_001',
    controladora_ip: '192.168.1.200',
    camera_name: '',
    camera_location: '',
    camera_ip: '',
    username: 'admin',
    password: '',
    port: '554',
    stream_path: '/stream1',
    resolution: '1920x1080',
    frame_rate: '30',
    enabled: false
  });

  const [loading, setLoading] = useState(false);
  const [testing, setTesting] = useState(false);
  const [showPassword, setShowPassword] = useState(false);
  const [testResults, setTestResults] = useState(null);
  const [configLoaded, setConfigLoaded] = useState(false);

  // Cargar configuración al inicializar
  useEffect(() => {
    loadCameraConfig();
  }, []);

  // Generar URL RTSP automáticamente
  useEffect(() => {
    if (config.username && config.camera_ip && config.port) {
      const password_part = config.password ? `:${config.password}` : '';
      const stream_path = config.stream_path || '/stream1';
      const url = `rtsp://${config.username}${password_part}@${config.camera_ip}:${config.port}${stream_path}`;
      
      setConfig(prev => ({ 
        ...prev, 
        rtsp_url: url 
      }));
    }
  }, [config.username, config.password, config.camera_ip, config.port, config.stream_path]);

  const loadCameraConfig = async () => {
    setLoading(true);
    try {
      console.log('🔄 Cargando configuración de cámara...');
      const response = await apiService.getCameraConfig();
      
      console.log('📦 Configuración recibida:', response);
      
      // Actualizar estado con la configuración recibida
      setConfig(prev => ({
        ...prev,
        ...response
      }));
      
      setConfigLoaded(true);
      console.log('✅ Configuración cargada exitosamente');
      
    } catch (error) {
      console.error('❌ Error cargando configuración:', error);
      toast.error('Error cargando configuración de cámara');
    } finally {
      setLoading(false);
    }
  };

  const handleSubmit = async (e) => {
    e.preventDefault();
    
    // Validación básica
    if (!config.rtsp_url || !config.rtsp_url.trim()) {
      toast.error('La URL RTSP es requerida');
      return;
    }

    setLoading(true);

    try {
      console.log('💾 Guardando configuración:', config);

      // Llamar al API
      const response = await apiService.updateCameraConfig(config);
      
      console.log('✅ Respuesta del servidor:', response);
      
      // Actualizar contexto del sistema si está disponible
      if (updateCameraConfig) {
        await updateCameraConfig('camera_1', config);
      }

      toast.success('✅ Configuración guardada exitosamente');
      
      // Esperar un poco para que el video processor se reinicie
      setTimeout(() => {
        toast.info('🔄 Reiniciando procesamiento de video...');
      }, 2000);

    } catch (error) {
      console.error('❌ Error guardando configuración:', error);
      toast.error('❌ Error guardando configuración');
    } finally {
      setLoading(false);
    }
  };

  const testConnection = async () => {
    if (!config.rtsp_url || !config.rtsp_url.trim()) {
      toast.error('Configure la URL RTSP primero');
      return;
    }

    setTesting(true);
    setTestResults(null);
    
    try {
      console.log('🧪 Probando conexión RTSP:', config.rtsp_url);
      
      const response = await apiService.testCameraStream(config.rtsp_url);
      
      console.log('📊 Resultado del test:', response);
      
      if (response.success) {
        setTestResults({
          success: true,
          message: response.message,
          frames_tested: response.frames_tested || 0
        });
        toast.success('✅ Conexión exitosa');
      } else {
        setTestResults({
          success: false,
          message: response.message
        });
        toast.error('❌ Error en conexión');
      }
      
    } catch (error) {
      console.error('❌ Error en test de conexión:', error);
      setTestResults({
        success: false,
        message: `Error: ${error.message}`
      });
      toast.error('❌ Error probando conexión');
    } finally {
      setTesting(false);
    }
  };

  const restartProcessing = async () => {
    if (!config.rtsp_url) {
      toast.error('Configure la URL RTSP primero');
      return;
    }

    setTesting(true);
    try {
      console.log('🔄 Reiniciando procesamiento...');
      
      const response = await apiService.restartCameraProcessing();
      
      console.log('📊 Resultado del reinicio:', response);
      
      if (response.status === 'running') {
        toast.success(`✅ ${response.message} - FPS: ${response.fps}`);
      } else {
        toast.warning(`⚠️ ${response.message}`);
      }

    } catch (error) {
      console.error('❌ Error reiniciando:', error);
      toast.error('❌ Error reiniciando procesamiento');
    } finally {
      setTesting(false);
    }
  };

  const resetConfiguration = async () => {
    if (!window.confirm('⚠️ ¿Está seguro de resetear toda la configuración?\n\nEsto eliminará toda la configuración actual.')) {
      return;
    }

    setLoading(true);
    try {
      console.log('🧹 Reseteando configuración...');
      
      const response = await apiService.resetCameraConfig();
      
      console.log('✅ Configuración reseteada:', response);
      
      // Actualizar el estado local
      setConfig(response.config);
      setTestResults(null);
      
      toast.success('✅ Configuración reseteada exitosamente');
      
    } catch (error) {
      console.error('❌ Error reseteando configuración:', error);
      toast.error('❌ Error reseteando configuración');
    } finally {
      setLoading(false);
    }
  };

  const handleInputChange = (field, value) => {
    setConfig(prev => ({
      ...prev,
      [field]: value
    }));
  };

  const CommonInputClasses = "w-full px-3 py-2 bg-gray-700 text-white border border-gray-600 rounded-md focus:outline-none focus:ring-2 focus:ring-blue-500";

  if (loading && !configLoaded) {
    return (
      <div className="flex items-center justify-center h-64">
        <div className="text-center">
          <div className="animate-spin rounded-full h-12 w-12 border-b-2 border-blue-500 mx-auto mb-4"></div>
          <p className="text-gray-400">Cargando configuración...</p>
        </div>
      </div>
    );
  }

  return (
    <div className="space-y-6">
      <div className="flex items-center justify-between">
        <h1 className="text-3xl font-bold text-white">Configuración de Cámara</h1>
        <div className="flex items-center space-x-4">
          {systemStatus.camera ? (
            <>
              <CheckCircleIcon className="h-6 w-6 text-green-500" />
              <span className="text-green-400">
                Conectada - {systemStatus.fps || 0} FPS
              </span>
            </>
          ) : (
            <>
              <ExclamationTriangleIcon className="h-6 w-6 text-red-500" />
              <span className="text-red-400">
                {config.rtsp_url ? 'RTSP configurado - Sin stream' : 'Sin configurar'}
              </span>
            </>
          )}
        </div>
      </div>

      <form onSubmit={handleSubmit} className="space-y-6">
        
        {/* CONFIGURACIÓN BÁSICA DE RTSP */}
        <div className="bg-gray-800 rounded-lg p-6">
          <div className="flex items-center mb-6">
            <CameraIcon className="h-6 w-6 text-blue-500 mr-2" />
            <h2 className="text-xl font-semibold text-white">Configuración RTSP</h2>
          </div>

          <div className="grid grid-cols-1 md:grid-cols-2 gap-6 mb-6">
            <div>
              <label className="block text-sm font-medium text-gray-300 mb-2">
                Nombre de Cámara
              </label>
              <input
                type="text"
                value={config.camera_name}
                onChange={(e) => handleInputChange('camera_name', e.target.value)}
                placeholder="Ej: Cámara Norte"
                className={CommonInputClasses}
              />
            </div>

            <div>
              <label className="block text-sm font-medium text-gray-300 mb-2">
                Ubicación
              </label>
              <input
                type="text"
                value={config.camera_location}
                onChange={(e) => handleInputChange('camera_location', e.target.value)}
                placeholder="Ej: Intersección Principal"
                className={CommonInputClasses}
              />
            </div>
          </div>

          <div className="grid grid-cols-1 md:grid-cols-3 gap-6 mb-6">
            <div>
              <label className="block text-sm font-medium text-gray-300 mb-2">
                IP de Cámara *
              </label>
              <input
                type="text"
                value={config.camera_ip}
                onChange={(e) => handleInputChange('camera_ip', e.target.value)}
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
                onChange={(e) => handleInputChange('port', e.target.value)}
                placeholder="554"
                className={CommonInputClasses}
              />
            </div>

            <div>
              <label className="block text-sm font-medium text-gray-300 mb-2">
                Usuario *
              </label>
              <input
                type="text"
                value={config.username}
                onChange={(e) => handleInputChange('username', e.target.value)}
                placeholder="admin"
                className={CommonInputClasses}
                required
              />
            </div>

            <div>
              <label className="block text-sm font-medium text-gray-300 mb-2">
                Ruta de Stream
              </label>
              <input
                type="text"
                value={config.stream_path || '/stream1'}
                onChange={(e) => handleInputChange('stream_path', e.target.value)}
                placeholder="/VideoInput/1/h264/1"
                className={CommonInputClasses}
              />
              <p className="text-gray-400 text-xs mt-1">
                Ejemplos: /stream1, /VideoInput/1/h264/1, /cam/realmonitor
              </p>
            </div>
          </div>

          <div className="grid grid-cols-1 md:grid-cols-2 gap-6 mb-6">
            <div>
              <label className="block text-sm font-medium text-gray-300 mb-2">
                Contraseña *
              </label>
              <div className="relative">
                <input
                  type={showPassword ? "text" : "password"}
                  value={config.password}
                  onChange={(e) => handleInputChange('password', e.target.value)}
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

            <div>
              <label className="block text-sm font-medium text-gray-300 mb-2">
                Resolución
              </label>
              <select
                value={config.resolution}
                onChange={(e) => handleInputChange('resolution', e.target.value)}
                className={CommonInputClasses}
              >
                <option value="1920x1080">1920x1080 (Full HD)</option>
                <option value="1280x720">1280x720 (HD)</option>
                <option value="2560x1440">2560x1440 (2K)</option>
              </select>
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
                onChange={(e) => handleInputChange('rtsp_url', e.target.value)}
                className="w-full px-3 py-2 bg-blue-900/30 text-blue-100 border border-blue-600 rounded-md focus:outline-none focus:ring-2 focus:ring-blue-500"
                placeholder="rtsp://admin:password@192.168.1.100:554/stream1"
              />
            </div>
            <p className="text-blue-400 text-xs mt-1">
              Se genera automáticamente o puede editarse manualmente
            </p>
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
                onChange={(e) => handleInputChange('fase', e.target.value)}
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
                onChange={(e) => handleInputChange('direccion', e.target.value)}
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
                onChange={(e) => handleInputChange('controladora_id', e.target.value)}
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
                onChange={(e) => handleInputChange('controladora_ip', e.target.value)}
                placeholder="192.168.1.200"
                className={CommonInputClasses}
              />
            </div>
          </div>
        </div>

        {/* BOTONES DE ACCIÓN */}
        <div className="flex flex-wrap gap-4">
          <button
            type="submit"
            disabled={loading}
            className="flex-1 min-w-[200px] px-4 py-3 bg-blue-600 text-white rounded-md hover:bg-blue-700 disabled:opacity-50 font-medium"
          >
            {loading ? (
              <>
                <div className="animate-spin rounded-full h-4 w-4 border-b-2 border-white inline mr-2"></div>
                Guardando...
              </>
            ) : (
              '💾 Guardar Configuración'
            )}
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
                🧪 Probar Conexión
              </>
            )}
          </button>

          {config.rtsp_url && (
            <button
              type="button"
              onClick={restartProcessing}
              disabled={testing}
              className="px-4 py-3 bg-orange-600 text-white rounded-md hover:bg-orange-700 disabled:opacity-50"
            >
              <ArrowPathIcon className="h-5 w-5 inline mr-2" />
              🔄 Reiniciar Stream
            </button>
          )}

          <button
            type="button"
            onClick={resetConfiguration}
            disabled={loading}
            className="px-4 py-3 bg-red-600 text-white rounded-md hover:bg-red-700 disabled:opacity-50"
          >
            🧹 Reset Total
          </button>
        </div>
      </form>

      {/* RESULTADOS DE PRUEBA */}
      {testResults && (
        <div className={`bg-gray-800 rounded-lg p-6 border-l-4 ${
          testResults.success ? 'border-green-500' : 'border-red-500'
        }`}>
          <h3 className="text-lg font-semibold text-white mb-4">
            Resultados de Prueba de Conexión
          </h3>
          
          {testResults.success ? (
            <div className="space-y-2">
              <div className="flex items-center">
                <CheckCircleIcon className="h-5 w-5 text-green-500 mr-2" />
                <span className="text-green-400 font-medium">Conexión Exitosa</span>
              </div>
              <p className="text-white">{testResults.message}</p>
              {testResults.frames_tested && (
                <p className="text-gray-300 text-sm">
                  Frames probados: {testResults.frames_tested}
                </p>
              )}
            </div>
          ) : (
            <div className="space-y-2">
              <div className="flex items-center">
                <ExclamationTriangleIcon className="h-5 w-5 text-red-500 mr-2" />
                <span className="text-red-400 font-medium">Error de Conexión</span>
              </div>
              <p className="text-white">{testResults.message}</p>
              <div className="mt-4 p-3 bg-yellow-900/20 border border-yellow-600 rounded">
                <p className="text-yellow-300 text-sm font-medium">💡 Posibles soluciones:</p>
                <ul className="text-yellow-200 text-sm mt-2 space-y-1">
                  <li>• Verificar que la IP de la cámara sea correcta</li>
                  <li>• Comprobar usuario y contraseña</li>
                  <li>• Asegurar que el puerto 554 esté abierto</li>
                  <li>• Verificar conectividad de red</li>
                  <li>• Probar la URL RTSP en VLC media player</li>
                </ul>
              </div>
            </div>
          )}
        </div>
      )}

      {/* INFORMACIÓN DEL SISTEMA */}
      <div className="bg-gray-800 rounded-lg p-6">
        <h2 className="text-xl font-semibold text-white mb-4">Estado del Sistema</h2>
        <div className="grid grid-cols-1 md:grid-cols-4 gap-4 text-gray-300">
          <div className="bg-gray-700 p-3 rounded">
            <p className="text-xs text-gray-400">Cámara</p>
            <p className={`font-medium ${systemStatus.camera ? 'text-green-400' : 'text-red-400'}`}>
              {systemStatus.camera ? '✅ Conectada' : '❌ Desconectada'}
            </p>
          </div>
          <div className="bg-gray-700 p-3 rounded">
            <p className="text-xs text-gray-400">FPS Actual</p>
            <p className="font-medium text-white">{systemStatus.fps || 0} FPS</p>
          </div>
          <div className="bg-gray-700 p-3 rounded">
            <p className="text-xs text-gray-400">RTSP Configurado</p>
            <p className={`font-medium ${config.rtsp_url ? 'text-green-400' : 'text-yellow-400'}`}>
              {config.rtsp_url ? '✅ Sí' : '⚠️ No'}
            </p>
          </div>
          <div className="bg-gray-700 p-3 rounded">
            <p className="text-xs text-gray-400">Procesamiento</p>
            <p className={`font-medium ${systemStatus.processing ? 'text-green-400' : 'text-yellow-400'}`}>
              {systemStatus.processing ? '✅ Activo' : '⏳ Inactivo'}
            </p>
          </div>
        </div>
      </div>
    </div>
  );
};

export default CameraConfig;