import React, { useState, useEffect } from 'react';
import {
  CpuChipIcon,
  ClockIcon,
  ServerIcon,
  ShieldCheckIcon
} from '@heroicons/react/24/outline';
import { toast } from 'react-toastify';
import { useSystem } from '../../context/SystemContext';

const SystemConfig = () => {
  const { config, updateSystemConfig } = useSystem();
  const [systemConfig, setSystemConfig] = useState({
    data_retention_days: 30,
    target_fps: 30,
    log_level: 'INFO',
    enable_debug: false,
    auto_cleanup: true,
    backup_enabled: true
  });
  const [loading, setLoading] = useState(false);

  useEffect(() => {
    if (config) {
      setSystemConfig({
        data_retention_days: config.data_retention_days || 30,
        target_fps: config.target_fps || 30,
        log_level: config.log_level || 'INFO',
        enable_debug: config.enable_debug || false,
        auto_cleanup: config.auto_cleanup !== false,
        backup_enabled: config.backup_enabled !== false
      });
    }
  }, [config]);

  const handleSubmit = async (e) => {
    e.preventDefault();
    setLoading(true);

    try {
      await updateSystemConfig(systemConfig);
      toast.success('Configuración del sistema actualizada');
    } catch (error) {
      toast.error('Error actualizando configuración del sistema');
    } finally {
      setLoading(false);
    }
  };

  const handleToggle = (field) => {
    setSystemConfig(prev => ({
      ...prev,
      [field]: !prev[field]
    }));
  };

  return (
    <div className="space-y-6">
      <h1 className="text-3xl font-bold text-white">Configuración del Sistema</h1>

      <form onSubmit={handleSubmit} className="space-y-6">
        {/* Configuración de Almacenamiento */}
        <div className="bg-gray-800 rounded-lg p-6">
          <div className="flex items-center mb-6">
            <ServerIcon className="h-6 w-6 text-blue-500 mr-2" />
            <h2 className="text-xl font-semibold text-white">Gestión de Datos</h2>
          </div>

          <div className="space-y-4">
            <div>
              <label className="block text-sm font-medium text-gray-300 mb-2">
                Retención de Datos (días)
              </label>
              <input
                type="number"
                min="7"
                max="365"
                value={systemConfig.data_retention_days}
                onChange={(e) => setSystemConfig({
                  ...systemConfig, 
                  data_retention_days: parseInt(e.target.value)
                })}
                className="w-full px-3 py-2 bg-gray-700 text-white border border-gray-600 rounded-md focus:outline-none focus:ring-2 focus:ring-blue-500"
              />
              <p className="text-gray-400 text-xs mt-1">
                Los datos más antiguos se eliminarán automáticamente
              </p>
            </div>

            <div className="flex items-center justify-between">
              <div>
                <h3 className="text-white font-medium">Limpieza Automática</h3>
                <p className="text-gray-400 text-sm">
                  Eliminar automáticamente datos antiguos cada día a las 2:00 AM
                </p>
              </div>
              <button
                type="button"
                onClick={() => handleToggle('auto_cleanup')}
                className={`relative inline-flex h-6 w-11 items-center rounded-full transition-colors ${
                  systemConfig.auto_cleanup ? 'bg-blue-600' : 'bg-gray-600'
                }`}
              >
                <span
                  className={`inline-block h-4 w-4 transform rounded-full bg-white transition-transform ${
                    systemConfig.auto_cleanup ? 'translate-x-6' : 'translate-x-1'
                  }`}
                />
              </button>
            </div>

            <div className="flex items-center justify-between">
              <div>
                <h3 className="text-white font-medium">Respaldos Automáticos</h3>
                <p className="text-gray-400 text-sm">
                  Crear respaldos automáticos de configuración y datos
                </p>
              </div>
              <button
                type="button"
                onClick={() => handleToggle('backup_enabled')}
                className={`relative inline-flex h-6 w-11 items-center rounded-full transition-colors ${
                  systemConfig.backup_enabled ? 'bg-blue-600' : 'bg-gray-600'
                }`}
              >
                <span
                  className={`inline-block h-4 w-4 transform rounded-full bg-white transition-transform ${
                    systemConfig.backup_enabled ? 'translate-x-6' : 'translate-x-1'
                  }`}
                />
              </button>
            </div>
          </div>
        </div>

        {/* Configuración de Rendimiento */}
        <div className="bg-gray-800 rounded-lg p-6">
          <div className="flex items-center mb-6">
            <CpuChipIcon className="h-6 w-6 text-green-500 mr-2" />
            <h2 className="text-xl font-semibold text-white">Rendimiento</h2>
          </div>

          <div className="space-y-4">
            <div>
              <label className="block text-sm font-medium text-gray-300 mb-2">
                FPS Objetivo
              </label>
              <select
                value={systemConfig.target_fps}
                onChange={(e) => setSystemConfig({
                  ...systemConfig, 
                  target_fps: parseInt(e.target.value)
                })}
                className="w-full px-3 py-2 bg-gray-700 text-white border border-gray-600 rounded-md focus:outline-none focus:ring-2 focus:ring-blue-500"
              >
                <option value={15}>15 FPS (Bajo consumo)</option>
                <option value={20}>20 FPS (Balanceado)</option>
                <option value={30}>30 FPS (Alto rendimiento)</option>
              </select>
            </div>
          </div>
        </div>

        {/* Configuración de Logging */}
        <div className="bg-gray-800 rounded-lg p-6">
          <div className="flex items-center mb-6">
            <ClockIcon className="h-6 w-6 text-yellow-500 mr-2" />
            <h2 className="text-xl font-semibold text-white">Logging y Debug</h2>
          </div>

          <div className="space-y-4">
            <div>
              <label className="block text-sm font-medium text-gray-300 mb-2">
                Nivel de Log
              </label>
              <select
                value={systemConfig.log_level}
                onChange={(e) => setSystemConfig({
                  ...systemConfig, 
                  log_level: e.target.value
                })}
                className="w-full px-3 py-2 bg-gray-700 text-white border border-gray-600 rounded-md focus:outline-none focus:ring-2 focus:ring-blue-500"
              >
                <option value="ERROR">ERROR (Solo errores)</option>
                <option value="WARNING">WARNING (Errores y advertencias)</option>
                <option value="INFO">INFO (Información general)</option>
                <option value="DEBUG">DEBUG (Información detallada)</option>
              </select>
            </div>

            <div className="flex items-center justify-between">
              <div>
                <h3 className="text-white font-medium">Modo Debug</h3>
                <p className="text-gray-400 text-sm">
                  Activar información de depuración detallada (reduce rendimiento)
                </p>
              </div>
              <button
                type="button"
                onClick={() => handleToggle('enable_debug')}
                className={`relative inline-flex h-6 w-11 items-center rounded-full transition-colors ${
                  systemConfig.enable_debug ? 'bg-yellow-600' : 'bg-gray-600'
                }`}
              >
                <span
                  className={`inline-block h-4 w-4 transform rounded-full bg-white transition-transform ${
                    systemConfig.enable_debug ? 'translate-x-6' : 'translate-x-1'
                  }`}
                />
              </button>
            </div>
          </div>
        </div>

        {/* Información del Sistema */}
        <div className="bg-gray-800 rounded-lg p-6">
          <div className="flex items-center mb-6">
            <ShieldCheckIcon className="h-6 w-6 text-purple-500 mr-2" />
            <h2 className="text-xl font-semibold text-white">Información del Sistema</h2>
          </div>

          <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-4">
            <div className="bg-gray-700 p-3 rounded">
              <p className="text-xs text-gray-400">Hardware</p>
              <p className="font-medium text-white">Radxa Rock 5T</p>
            </div>
            <div className="bg-gray-700 p-3 rounded">
              <p className="text-xs text-gray-400">Versión</p>
              <p className="font-medium text-white">1.0.0</p>
            </div>
            <div className="bg-gray-700 p-3 rounded">
              <p className="text-xs text-gray-400">NPU</p>
              <p className="font-medium text-white">RKNN Habilitado</p>
            </div>
            <div className="bg-gray-700 p-3 rounded">
              <p className="text-xs text-gray-400">Modelo</p>
              <p className="font-medium text-white">YOLOv8n</p>
            </div>
          </div>
        </div>

        {/* Botón de guardar */}
        <div className="flex justify-end">
          <button
            type="submit"
            disabled={loading}
            className="px-6 py-2 bg-blue-600 text-white rounded-md hover:bg-blue-700 disabled:opacity-50"
          >
            {loading ? 'Guardando...' : 'Guardar Configuración'}
          </button>
        </div>
      </form>
    </div>
  );
};

export default SystemConfig;