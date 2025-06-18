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

      // CORREGIDO: Usar llamadas reales en lugar de simulación
      const [statusResponse, configResponse, healthResponse] = await Promise.all([
        apiService.getCameraStatus().catch((err) => {
          console.warn('Error getting camera status:', err);
          return { connected: false, fps: 0 };
        }),
        apiService.getSystemConfig().catch((err) => {
          console.warn('Error getting system config:', err);
          return {};
        }),
        apiService.getCameraHealth().catch((err) => {
          console.warn('Error getting camera health:', err);
          return { status: 'unhealthy' };
        })
      ]);

      // Actualizar estado del sistema con datos reales
      setSystemStatus({
        camera: statusResponse.connected || false,
        controller: healthResponse.controller_connected || false, // Agregar al endpoint
        processing: statusResponse.connected || false,
        fps: statusResponse.fps || 0
      });

      // Actualizar configuración si hay datos
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
      await loadSystemData(); // Recargar datos después de actualizar
      return true; // Éxito
    } catch (error) {
      console.error('Error actualizando configuración:', error);
      throw error; // Re-lanzar para manejo en el componente
    }
  }, [loadSystemData]);

  const updateSystemConfig = useCallback(async (newConfig) => {
    try {
      await apiService.updateSystemConfig(newConfig);
      setConfig(prev => ({ ...prev, ...newConfig }));
      return true; // Éxito
    } catch (error) {
      console.error('Error actualizando configuración del sistema:', error);
      throw error; // Re-lanzar para manejo en el componente
    }
  }, []);

  // Cargar datos iniciales
  useEffect(() => {
    loadSystemData();
    
    // Actualizar cada 30 segundos
    const interval = setInterval(() => {
      loadSystemData();
    }, 10000);

    return () => clearInterval(interval);
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