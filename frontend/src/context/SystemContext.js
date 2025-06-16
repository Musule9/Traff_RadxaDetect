import React, { createContext, useContext, useState, useCallback, useEffect } from 'react';
import { apiService } from '../services/api';
import { toast } from 'react-toastify';

const SystemContext = createContext();

export const useSystem = () => {
  const context = useContext(SystemContext);
  if (!context) {
    throw new Error('useSystem debe ser usado dentro de SystemProvider');
  }
  return context;
};

export const SystemProvider = ({ children }) => {
  const [systemStatus, setSystemStatus] = useState({
    camera: false,
    controller: false,
    processing: false,
    fps: 0
  });

  const [cameras, setCameras] = useState([
    { id: 'camera_1', name: 'Cámara Principal', enabled: true }
  ]);
  const [selectedCamera, setSelectedCamera] = useState('camera_1');
  const [config, setConfig] = useState({
    confidence_threshold: 0.5,
    night_vision_enhancement: true,
    show_overlay: true,
    data_retention_days: 30,
    target_fps: 30,
    log_level: 'INFO'
  });
  const [loading, setLoading] = useState(false);

  const loadSystemData = useCallback(async () => {
    try {
      setLoading(true);

      // Cargar estado del sistema
      const [statusResponse, configResponse] = await Promise.all([
        apiService.getCameraStatus().catch(() => ({ connected: false, fps: 0 })),
        apiService.getSystemConfig().catch(() => ({}))
      ]);

      setSystemStatus({
        camera: statusResponse.connected || false,
        controller: Math.random() > 0.3, // Simulado hasta que esté la controladora
        processing: statusResponse.connected || false,
        fps: statusResponse.fps || 0
      });

      if (Object.keys(configResponse).length > 0) {
        setConfig(prev => ({ ...prev, ...configResponse }));
      }

    } catch (error) {
      console.error('Error cargando datos del sistema:', error);
      // No mostrar toast aquí para evitar spam
    } finally {
      setLoading(false);
    }
  }, []);

  const updateCameraConfig = useCallback(async (cameraId, newConfig) => {
    try {
      await apiService.updateCameraConfig(newConfig);
      await loadSystemData();
      toast.success('Configuración de cámara actualizada');
    } catch (error) {
      console.error('Error actualizando configuración:', error);
      toast.error('Error actualizando configuración');
    }
  }, [loadSystemData]);

  const updateSystemConfig = useCallback(async (newConfig) => {
    try {
      await apiService.updateSystemConfig(newConfig);
      setConfig(prev => ({ ...prev, ...newConfig }));
      toast.success('Configuración del sistema actualizada');
    } catch (error) {
      console.error('Error actualizando configuración del sistema:', error);
      toast.error('Error actualizando configuración del sistema');
    }
  }, []);

  // Cargar datos iniciales
  useEffect(() => {
    loadSystemData();
  }, [loadSystemData]);

  const value = {
    systemStatus,
    cameras,
    selectedCamera,
    setSelectedCamera,
    config,
    loading,
    loadSystemData,
    updateCameraConfig,
    updateSystemConfig
  };

  return (
    <SystemContext.Provider value={value}>
      {children}
    </SystemContext.Provider>
  );
};