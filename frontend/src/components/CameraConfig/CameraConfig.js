
import React, { useState, useEffect } from 'react';
import { 
  CameraIcon, 
  WifiIcon, 
  CheckCircleIcon,
  ExclamationTriangleIcon 
} from '@heroicons/react/24/outline';
import { toast } from 'react-toastify';
import { apiService } from '../../services/api';
import { useSystem } from '../../context/SystemContext';

const CameraConfig = () => {
  const { systemStatus, updateCameraConfig } = useSystem();
  const [config, setConfig] = useState({
    rtsp_url: '',
    fase: 'fase1',
    direccion: 'norte',
    controladora_id: 'CTRL_001',
    controladora_ip: '192.168.1.200'
  });
  const [loading, setLoading] = useState(false);
  const [testing, setTesting] = useState(false);

  useEffect(() => {
    loadCameraConfig();
  }, []);

  const loadCameraConfig = async () => {
    try {
      const status = await apiService.getCameraStatus();
      setConfig({
        rtsp_url: status.rtsp_url || '',
        fase: status.fase || 'fase1',
        direccion: status.direccion || 'norte',
        controladora_id: 'CTRL_001',
        controladora_ip: '192.168.1.200'
      });
    } catch (error) {
      console.error('Error cargando configuración:', error);
    }
  };

  const handleSubmit = async (e) => {
    e.preventDefault();
    setLoading(true);

    try {
      await updateCameraConfig('camera_1', config);
      toast.success('Configuración guardada exitosamente');
    } catch (error) {
      toast.error('Error guardando configuración');
    } finally {
      setLoading(false);
    }
  };

  const testConnection = async () => {
    if (!config.rtsp_url) {
      toast.error('Ingrese una URL RTSP válida');
      return;
    }

    setTesting(true);
    try {
      const health = await apiService.getCameraHealth();
      if (health.healthy) {
        toast.success('Conexión de cámara exitosa');
      } else {
        toast.error('No se pudo conectar con la cámara');
      }
    } catch (error) {
      toast.error('Error probando conexión');
    } finally {
      setTesting(false);
    }
  };

  return (
    <div className="space-y-6">
      <div className="flex items-center justify-between">
        <h1 className="text-3xl font-bold text-white">Configuración de Cámara</h1>
        <div className="flex items-center space-x-2">
          {systemStatus.camera ? (
            <>
              <CheckCircleIcon className="h-6 w-6 text-green-500" />
              <span className="text-green-400">Conectada</span>
            </>
          ) : (
            <>
              <ExclamationTriangleIcon className="h-6 w-6 text-red-500" />
              <span className="text-red-400">Desconectada</span>
            </>
          )}
        </div>
      </div>

      <div className="bg-gray-800 rounded-lg p-6">
        <div className="flex items-center mb-6">
          <CameraIcon className="h-6 w-6 text-blue-500 mr-2" />
          <h2 className="text-xl font-semibold text-white">Configuración de Red</h2>
        </div>

        <form onSubmit={handleSubmit} className="space-y-6">
          <div>
            <label className="block text-sm font-medium text-gray-300 mb-2">
              URL RTSP *
            </label>
            <input
              type="text"
              value={config.rtsp_url}
              onChange={(e) => setConfig({...config, rtsp_url: e.target.value})}
              placeholder="rtsp://admin:password@192.168.1.100:554/stream1"
              className="w-full px-3 py-2 bg-gray-700 text-white border border-gray-600 rounded-md focus:outline-none focus:ring-2 focus:ring-blue-500"
              required
            />
            <p className="text-gray-400 text-sm mt-1">
              Formato: rtsp://usuario:contraseña@ip:puerto/ruta
            </p>
          </div>

          <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
            <div>
              <label className="block text-sm font-medium text-gray-300 mb-2">
                Fase del Semáforo
              </label>
              <select
                value={config.fase}
                onChange={(e) => setConfig({...config, fase: e.target.value})}
                className="w-full px-3 py-2 bg-gray-700 text-white border border-gray-600 rounded-md focus:outline-none focus:ring-2 focus:ring-blue-500"
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
                className="w-full px-3 py-2 bg-gray-700 text-white border border-gray-600 rounded-md focus:outline-none focus:ring-2 focus:ring-blue-500"
              >
                <option value="norte">Norte</option>
                <option value="sur">Sur</option>
                <option value="este">Este</option>
                <option value="oeste">Oeste</option>
              </select>
            </div>
          </div>

          <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
            <div>
              <label className="block text-sm font-medium text-gray-300 mb-2">
                ID de Controladora
              </label>
              <input
                type="text"
                value={config.controladora_id}
                onChange={(e) => setConfig({...config, controladora_id: e.target.value})}
                className="w-full px-3 py-2 bg-gray-700 text-white border border-gray-600 rounded-md focus:outline-none focus:ring-2 focus:ring-blue-500"
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
                className="w-full px-3 py-2 bg-gray-700 text-white border border-gray-600 rounded-md focus:outline-none focus:ring-2 focus:ring-blue-500"
              />
            </div>
          </div>

          <div className="flex space-x-4">
            <button
              type="submit"
              disabled={loading}
              className="flex-1 px-4 py-2 bg-blue-600 text-white rounded-md hover:bg-blue-700 disabled:opacity-50"
            >
              {loading ? 'Guardando...' : 'Guardar Configuración'}
            </button>
            <button
              type="button"
              onClick={testConnection}
              disabled={testing || !config.rtsp_url}
              className="px-6 py-2 bg-green-600 text-white rounded-md hover:bg-green-700 disabled:opacity-50"
            >
              <WifiIcon className="h-5 w-5 inline mr-2" />
              {testing ? 'Probando...' : 'Probar Conexión'}
            </button>
          </div>
        </form>
      </div>

      {/* Información adicional */}
      <div className="bg-gray-800 rounded-lg p-6">
        <h2 className="text-xl font-semibold text-white mb-4">Información del Sistema</h2>
        <div className="grid grid-cols-1 md:grid-cols-3 gap-4 text-gray-300">
          <div>
            <p className="font-medium">Hardware:</p>
            <p>Radxa Rock 5T</p>
          </div>
          <div>
            <p className="font-medium">NPU:</p>
            <p>RKNN Habilitado</p>
          </div>
          <div>
            <p className="font-medium">FPS Actual:</p>
            <p>{systemStatus.fps || 0} FPS</p>
          </div>
        </div>
      </div>
    </div>
  );
};

export default CameraConfig;
