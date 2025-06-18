// REEMPLAZAR EN frontend/src/context/SystemContext.js

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

      // CORREGIDO: Verificación de estado más robusta
      const [statusResponse, configResponse, healthResponse] = await Promise.all([
        apiService.getCameraStatus().catch((err) => {
          console.warn('Error getting camera status:', err);
          return { connected: false, fps: 0, enabled: false };
        }),
        apiService.getSystemConfig().catch((err) => {
          console.warn('Error getting system config:', err);
          return {};
        }),
        apiService.getCameraHealth().catch((err) => {
          console.warn('Error getting camera health:', err);
          return { status: 'unhealthy', camera_connected: false };
        })
      ]);

      // CORREGIDO: Actualizar estado con datos reales y verificación de RTSP
      const hasRtspUrl = statusResponse.rtsp_url && statusResponse.rtsp_url.trim().length > 0;
      const isProcessingActive = statusResponse.connected && statusResponse.fps > 0;
      
      setSystemStatus({
        camera: hasRtspUrl && statusResponse.enabled && isProcessingActive,
        controller: healthResponse.controller_connected || false,
        processing: isProcessingActive,
        fps: statusResponse.fps || 0
      });

      // Actualizar configuración si hay datos
      if (Object.keys(configResponse).length > 0) {
        setConfig(prev => ({ ...prev, ...configResponse }));
      }

      // LOGGING para debug
      console.log('🔍 System Status Update:', {
        hasRtspUrl,
        enabled: statusResponse.enabled,
        connected: statusResponse.connected,
        fps: statusResponse.fps,
        finalCameraStatus: hasRtspUrl && statusResponse.enabled && isProcessingActive
      });

    } catch (error) {
      console.error('Error cargando datos del sistema:', error);
      // Mantener estado anterior en caso de error de red
    } finally {
      setLoading(false);
    }
  }, []);

  const updateCameraConfig = useCallback(async (cameraId, newConfig) => {
    try {
      setLoading(true);
      
      // CORREGIDO: Verificar que newConfig tiene la estructura correcta
      const cleanConfig = {
        rtsp_url: newConfig.rtsp_url || '',
        fase: newConfig.fase || 'fase1',
        direccion: newConfig.direccion || 'norte',
        controladora_id: newConfig.controladora_id || 'CTRL_001',
        controladora_ip: newConfig.controladora_ip || '192.168.1.200',
        camera_name: newConfig.camera_name || '',
        camera_model: newConfig.camera_model || '',
        camera_location: newConfig.camera_location || '',
        camera_serial: newConfig.camera_serial || '',
        camera_ip: newConfig.camera_ip || '',
        username: newConfig.username || 'admin',
        password: newConfig.password || '',
        port: newConfig.port || '554',
        stream_path: newConfig.stream_path || '/stream1',
        resolution: newConfig.resolution || '1920x1080',
        frame_rate: newConfig.frame_rate || '30',
        bitrate: newConfig.bitrate || '4000',
        encoding: newConfig.encoding || 'H264',
        stream_quality: newConfig.stream_quality || 'high',
        night_vision: newConfig.night_vision || false,
        motion_detection: newConfig.motion_detection || false,
        recording_enabled: newConfig.recording_enabled || false,
        audio_enabled: newConfig.audio_enabled || false,
        detection_zones: newConfig.detection_zones !== false,
        speed_calculation: newConfig.speed_calculation !== false,
        vehicle_counting: newConfig.vehicle_counting !== false,
        license_plate_recognition: newConfig.license_plate_recognition || false,
        enabled: true
      };

      console.log('🔧 Enviando configuración limpia:', cleanConfig);
      
      const response = await apiService.updateCameraConfig(cleanConfig);
      
      console.log('✅ Respuesta del servidor:', response);
      
      // Recargar datos después de 2 segundos para dar tiempo al video processor
      setTimeout(() => {
        loadSystemData();
      }, 2000);
      
      return true;
    } catch (error) {
      console.error('Error actualizando configuración:', error);
      throw error;
    } finally {
      setLoading(false);
    }
  }, [loadSystemData]);

  const updateSystemConfig = useCallback(async (newConfig) => {
    try {
      await apiService.updateSystemConfig(newConfig);
      setConfig(prev => ({ ...prev, ...newConfig }));
      return true;
    } catch (error) {
      console.error('Error actualizando configuración del sistema:', error);
      throw error;
    }
  }, []);

  // NUEVO: Método para resetear configuración
  const resetCameraConfig = useCallback(async () => {
    try {
      setLoading(true);
      
      const response = await fetch('/api/camera/config/reset', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
          'Authorization': `Bearer ${localStorage.getItem('token')}`
        }
      });
      
      if (!response.ok) {
        throw new Error('Error reseteando configuración');
      }
      
      const result = await response.json();
      console.log('🧹 Configuración reseteada:', result);
      
      // Recargar datos
      await loadSystemData();
      
      return true;
    } catch (error) {
      console.error('Error reseteando configuración:', error);
      throw error;
    } finally {
      setLoading(false);
    }
  }, [loadSystemData]);

  // Cargar datos iniciales y configurar polling más frecuente
  useEffect(() => {
    loadSystemData();
    
    // Polling cada 5 segundos para detección de cambios más rápida
    const interval = setInterval(() => {
      loadSystemData();
    }, 5000);

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
    updateSystemConfig,
    resetCameraConfig  // NUEVO método
  };

  return (
    <SystemContext.Provider value={value}>
      {children}
    </SystemContext.Provider>
  );
};