import React, { useState, useEffect } from 'react';
import { CameraIcon, Cog6ToothIcon, WifiIcon } from '@heroicons/react/24/outline';
import toast from 'react-hot-toast';
import api from '../services/api';

const Configuration = () => {
  const [cameraConfig, setCameraConfig] = useState({
    rtsp_url: '',
    fase: 'fase1',
    direccion: 'norte',
    controladora_id: 'CTRL_001',
    controladora_ip: '192.168.1.200'
  });

  const [systemConfig, setSystemConfig] = useState({
    confidence_threshold: 0.5,
    night_vision_enhancement: true,
    show_overlay: true,
    data_retention_days: 30
  });

  const [loading, setLoading] = useState(false);

  useEffect(() => {
    fetchConfigurations();
  }, []);

  const fetchConfigurations = async () => {
    try {
      const [systemResponse, cameraResponse] = await Promise.all([
        api.get('/api/config/system'),
        api.get('/api/camera/status')
      ]);

      setSystemConfig(systemResponse.data);
      setCameraConfig({
        rtsp_url: cameraResponse.data.rtsp_url || '',
        fase: cameraResponse.data.fase || 'fase1',
        direccion: cameraResponse.data.direccion || 'norte',
        controladora_id: 'CTRL_001',
        controladora_ip: '192.168.1.200'
      });
    } catch (error) {
      console.error('Error fetching configurations:', error);
    }
  };

  const handleCameraConfigSubmit = async (e) => {
    e.preventDefault();
    setLoading(true);

    try {
      await api.post('/api/camera/config', cameraConfig);
      toast.success('Configuración de cámara actualizada');
    } catch (error) {
      toast.error('Error actualizando configuración de cámara');
    } finally {
      setLoading(false);
    }
  };

  const handleSystemConfigSubmit = async (e) => {
    e.preventDefault();
    setLoading(true);

    try {
      await api.post('/api/config/system', systemConfig);
      toast.success('Configuración del sistema actualizada');
    } catch (error) {
      toast.error('Error actualizando configuración del sistema');
    } finally {
      setLoading(false);
    }
  };

  const testCameraConnection = async () => {
    try {
      const response = await api.get('/api/camera_health');
      if (response.data.healthy) {
        toast.success('Conexión de cámara exitosa');
      } else {
        toast.error('Cámara no disponible');
      }
    } catch (error) {
      toast.error('Error probando conexión de cámara');
    }
  };

  return (
    <div className="space-y-6">
      <h1 className="text-3xl font-bold text-white">Configuración</h1>

      <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
        {/* Configuración de Cámara */}
        <div className="bg-gray-800 rounded-lg p-6">
          <div className="flex items-center mb-6">
            <CameraIcon className="h-6 w-6 text-blue-500 mr-2" />
            <h2 className="text-xl font-semibold text-white">Configuración de Cámara</h2>
          </div>

          <form onSubmit={handleCameraConfigSubmit} className="space-y-4">
            <div>
              <label className="block text-sm font-medium text-gray-300 mb-2">
                URL RTSP
              </label>
              <input
                type="text"
                value={cameraConfig.rtsp_url}
                onChange={(e) => setCameraConfig({...cameraConfig, rtsp_url: e.target.value})}
                placeholder="rtsp://admin:password@192.168.1.100:554/stream1"
                className="w-full px-3 py-2 bg-gray-700 text-white border border-gray-600 rounded-md focus:outline-none focus:ring-2 focus:ring-blue-500"
                required
              />
            </div>

            <div className="grid grid-cols-2 gap-4">
              <div>
                <label className="block text-sm font-medium text-gray-300 mb-2">
                  Fase
                </label>
                <select
                  value={cameraConfig.fase}
                  onChange={(e) => setCameraConfig({...cameraConfig, fase: e.target.value})}
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
                  Dirección
                </label>
                <select
                  value={cameraConfig.direccion}
                  onChange={(e) => setCameraConfig({...cameraConfig, direccion: e.target.value})}
                  className="w-full px-3 py-2 bg-gray-700 text-white border border-gray-600 rounded-md focus:outline-none focus:ring-2 focus:ring-blue-500"
                >
                  <option value="norte">Norte</option>
                  <option value="sur">Sur</option>
                  <option value="este">Este</option>
                  <option value="oeste">Oeste</option>
                </select>
              </div>
            </div>

            <div className="grid grid-cols-2 gap-4">
              <div>
                <label className="block text-sm font-medium text-gray-300 mb-2">
                  ID Controladora
                </label>
                <input
                  type="text"
                  value={cameraConfig.controladora_id}
                  onChange={(e) => setCameraConfig({...cameraConfig, controladora_id: e.target.value})}
                  className="w-full px-3 py-2 bg-gray-700 text-white border border-gray-600 rounded-md focus:outline-none focus:ring-2 focus:ring-blue-500"
                />
              </div>

              <div>
                <label className="block text-sm font-medium text-gray-300 mb-2">
                  IP Controladora
                </label>
                <input
                  type="text"
                  value={cameraConfig.controladora_ip}
                  onChange={(e) => setCameraConfig({...cameraConfig, controladora_ip: e.target.value})}
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
                onClick={testCameraConnection}
                className="px-4 py-2 bg-green-600 text-white rounded-md hover:bg-green-700"
              >
                <WifiIcon className="h-5 w-5 inline mr-2" />
                Probar Conexión
              </button>
            </div>
          </form>
        </div>

        {/* Configuración del Sistema */}
        <div className="bg-gray-800 rounded-lg p-6">
          <div className="flex items-center mb-6">
            <Cog6ToothIcon className="h-6 w-6 text-green-500 mr-2" />
            <h2 className="text-xl font-semibold text-white">Configuración del Sistema</h2>
          </div>

          <form onSubmit={handleSystemConfigSubmit} className="space-y-4">
            <div>
              <label className="block text-sm font-medium text-gray-300 mb-2">
                Umbral de Confianza: {systemConfig.confidence_threshold}
              </label>
              <input
                type="range"
                min="0.1"
                max="1.0"
                step="0.1"
                value={systemConfig.confidence_threshold}
                onChange={(e) => setSystemConfig({...systemConfig, confidence_threshold: parseFloat(e.target.value)})}
                className="w-full"
              />
            </div>

            <div>
              <label className="block text-sm font-medium text-gray-300 mb-2">
                Retención de Datos (días)
              </label>
              <input
                type="number"
                min="7"
                max="365"
                value={systemConfig.data_retention_days}
                onChange={(e) => setSystemConfig({...systemConfig, data_retention_days: parseInt(e.target.value)})}
                className="w-full px-3 py-2 bg-gray-700 text-white border border-gray-600 rounded-md focus:outline-none focus:ring-2 focus:ring-blue-500"
              />
            </div>

            <div className="space-y-3">
              <div className="flex items-center">
                <input
                  type="checkbox"
                  id="night_vision"
                  checked={systemConfig.night_vision_enhancement}
                  onChange={(e) => setSystemConfig({...systemConfig, night_vision_enhancement: e.target.checked})}
                  className="mr-3"
                />
                <label htmlFor="night_vision" className="text-gray-300">
                  Mejora de Visión Nocturna
                </label>
              </div>

              <div className="flex items-center">
                <input
                  type="checkbox"
                  id="show_overlay"
                  checked={systemConfig.show_overlay}
                  onChange={(e) => setSystemConfig({...systemConfig, show_overlay: e.target.checked})}
                  className="mr-3"
                />
                <label htmlFor="show_overlay" className="text-gray-300">
                  Mostrar Overlays de Análisis
                </label>
              </div>
            </div>

            <button
              type="submit"
              disabled={loading}
              className="w-full px-4 py-2 bg-green-600 text-white rounded-md hover:bg-green-700 disabled:opacity-50"
            >
              {loading ? 'Guardando...' : 'Guardar Configuración del Sistema'}
            </button>
          </form>
        </div>
      </div>

      {/* Información del Sistema */}
      <div className="bg-gray-800 rounded-lg p-6">
        <h2 className="text-xl font-semibold text-white mb-4">Información del Sistema</h2>
        <div className="grid grid-cols-1 md:grid-cols-3 gap-4 text-gray-300">
          <div>
            <p className="font-medium">Hardware:</p>
            <p>Radxa Rock 5T</p>
          </div>
          <div>
            <p className="font-medium">Versión:</p>
            <p>1.0.0</p>
          </div>
          <div>
            <p className="font-medium">Estado NPU:</p>
            <p>Habilitado</p>
          </div>
        </div>
      </div>
    </div>
  );
};

export default Configuration;