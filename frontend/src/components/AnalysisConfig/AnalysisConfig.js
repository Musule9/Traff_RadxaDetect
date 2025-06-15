import React, { useState } from 'react';
import {
  Cog6ToothIcon,
  ChartBarIcon,
  AdjustmentsHorizontalIcon
} from '@heroicons/react/24/outline';
import { toast } from 'react-toastify';
import { useSystem } from '../../context/SystemContext';

const AnalysisConfig = () => {
  const { config, updateSystemConfig } = useSystem();
  const [analysisConfig, setAnalysisConfig] = useState({
    confidence_threshold: config.confidence_threshold || 0.5,
    night_vision_enhancement: config.night_vision_enhancement || true,
    show_overlay: config.show_overlay || true,
    speed_calculation_enabled: true,
    min_track_length: 3,
    max_track_age: 30
  });
  const [loading, setLoading] = useState(false);

  const handleSubmit = async (e) => {
    e.preventDefault();
    setLoading(true);

    try {
      await updateSystemConfig(analysisConfig);
      toast.success('Configuración de análisis actualizada');
    } catch (error) {
      toast.error('Error actualizando configuración');
    } finally {
      setLoading(false);
    }
  };

  const handleSliderChange = (field, value) => {
    setAnalysisConfig(prev => ({
      ...prev,
      [field]: parseFloat(value)
    }));
  };

  const handleToggle = (field) => {
    setAnalysisConfig(prev => ({
      ...prev,
      [field]: !prev[field]
    }));
  };

  return (
    <div className="space-y-6">
      <h1 className="text-3xl font-bold text-white">Configuración de Análisis</h1>

      <form onSubmit={handleSubmit} className="space-y-6">
        {/* Configuración de Detección */}
        <div className="bg-gray-800 rounded-lg p-6">
          <div className="flex items-center mb-6">
            <AdjustmentsHorizontalIcon className="h-6 w-6 text-green-500 mr-2" />
            <h2 className="text-xl font-semibold text-white">Parámetros de Detección</h2>
          </div>

          <div className="space-y-6">
            <div>
              <label className="block text-sm font-medium text-gray-300 mb-2">
                Umbral de Confianza: {analysisConfig.confidence_threshold.toFixed(1)}
              </label>
              <input
                type="range"
                min="0.1"
                max="1.0"
                step="0.1"
                value={analysisConfig.confidence_threshold}
                onChange={(e) => handleSliderChange('confidence_threshold', e.target.value)}
                className="w-full h-2 bg-gray-700 rounded-lg appearance-none cursor-pointer slider"
              />
              <div className="flex justify-between text-xs text-gray-400 mt-1">
                <span>0.1 (Menos estricto)</span>
                <span>1.0 (Más estricto)</span>
              </div>
            </div>

            <div>
              <label className="block text-sm font-medium text-gray-300 mb-2">
                Longitud Mínima de Track: {analysisConfig.min_track_length}
              </label>
              <input
                type="range"
                min="1"
                max="10"
                step="1"
                value={analysisConfig.min_track_length}
                onChange={(e) => handleSliderChange('min_track_length', e.target.value)}
                className="w-full h-2 bg-gray-700 rounded-lg appearance-none cursor-pointer slider"
              />
              <p className="text-gray-400 text-xs mt-1">
                Número mínimo de detecciones para considerar un track válido
              </p>
            </div>

            <div>
              <label className="block text-sm font-medium text-gray-300 mb-2">
                Edad Máxima de Track: {analysisConfig.max_track_age}
              </label>
              <input
                type="range"
                min="10"
                max="60"
                step="5"
                value={analysisConfig.max_track_age}
                onChange={(e) => handleSliderChange('max_track_age', e.target.value)}
                className="w-full h-2 bg-gray-700 rounded-lg appearance-none cursor-pointer slider"
              />
              <p className="text-gray-400 text-xs mt-1">
                Frames que un track puede existir sin detecciones
              </p>
            </div>
          </div>
        </div>

        {/* Configuración de Procesamiento */}
        <div className="bg-gray-800 rounded-lg p-6">
          <div className="flex items-center mb-6">
            <Cog6ToothIcon className="h-6 w-6 text-blue-500 mr-2" />
            <h2 className="text-xl font-semibold text-white">Opciones de Procesamiento</h2>
          </div>

          <div className="space-y-4">
            <div className="flex items-center justify-between">
              <div>
                <h3 className="text-white font-medium">Mejora de Visión Nocturna</h3>
                <p className="text-gray-400 text-sm">
                  Aplicar mejoras de contraste y brillo para condiciones de poca luz
                </p>
              </div>
              <button
                type="button"
                onClick={() => handleToggle('night_vision_enhancement')}
                className={`relative inline-flex h-6 w-11 items-center rounded-full transition-colors ${
                  analysisConfig.night_vision_enhancement ? 'bg-blue-600' : 'bg-gray-600'
                }`}
              >
                <span
                  className={`inline-block h-4 w-4 transform rounded-full bg-white transition-transform ${
                    analysisConfig.night_vision_enhancement ? 'translate-x-6' : 'translate-x-1'
                  }`}
                />
              </button>
            </div>

            <div className="flex items-center justify-between">
              <div>
                <h3 className="text-white font-medium">Mostrar Overlays de Análisis</h3>
                <p className="text-gray-400 text-sm">
                  Mostrar líneas, zonas y tracks en el video
                </p>
              </div>
              <button
                type="button"
                onClick={() => handleToggle('show_overlay')}
                className={`relative inline-flex h-6 w-11 items-center rounded-full transition-colors ${
                  analysisConfig.show_overlay ? 'bg-blue-600' : 'bg-gray-600'
                }`}
              >
                <span
                  className={`inline-block h-4 w-4 transform rounded-full bg-white transition-transform ${
                    analysisConfig.show_overlay ? 'translate-x-6' : 'translate-x-1'
                  }`}
                />
              </button>
            </div>

            <div className="flex items-center justify-between">
              <div>
                <h3 className="text-white font-medium">Cálculo de Velocidad</h3>
                <p className="text-gray-400 text-sm">
                  Habilitar cálculo automático de velocidad entre líneas
                </p>
              </div>
              <button
                type="button"
                onClick={() => handleToggle('speed_calculation_enabled')}
                className={`relative inline-flex h-6 w-11 items-center rounded-full transition-colors ${
                  analysisConfig.speed_calculation_enabled ? 'bg-blue-600' : 'bg-gray-600'
                }`}
              >
                <span
                  className={`inline-block h-4 w-4 transform rounded-full bg-white transition-transform ${
                    analysisConfig.speed_calculation_enabled ? 'translate-x-6' : 'translate-x-1'
                  }`}
                />
              </button>
            </div>
          </div>
        </div>

        {/* Botón de guardar */}
        <div className="flex justify-end">
          <button
            type="submit"
            disabled={loading}
            className="px-6 py-2 bg-green-600 text-white rounded-md hover:bg-green-700 disabled:opacity-50"
          >
            {loading ? 'Guardando...' : 'Guardar Configuración'}
          </button>
        </div>
      </form>

      {/* Información de rendimiento */}
      <div className="bg-gray-800 rounded-lg p-6">
        <div className="flex items-center mb-4">
          <ChartBarIcon className="h-6 w-6 text-yellow-500 mr-2" />
          <h2 className="text-xl font-semibold text-white">Información de Rendimiento</h2>
        </div>
        
        <div className="grid grid-cols-1 md:grid-cols-3 gap-4 text-gray-300">
          <div className="bg-gray-700 p-3 rounded">
            <p className="text-xs text-gray-400">Modelo</p>
            <p className="font-medium">YOLOv8n + RKNN</p>
          </div>
          <div className="bg-gray-700 p-3 rounded">
            <p className="text-xs text-gray-400">Tracker</p>
            <p className="font-medium">BYTETracker</p>
          </div>
          <div className="bg-gray-700 p-3 rounded">
            <p className="text-xs text-gray-400">Hardware</p>
            <p className="font-medium">NPU Radxa 5T</p>
          </div>
        </div>
      </div>
    </div>
  );
};

export default AnalysisConfig;
