import axios from 'axios';

const api = axios.create({
  baseURL: process.env.REACT_APP_API_URL || '',
  timeout: 10000,
});

// Interceptor para manejar errores
api.interceptors.response.use(
  (response) => response,
  (error) => {
    if (error.response?.status === 401) {
      localStorage.removeItem('token');
      window.location.href = '/';
    }
    return Promise.reject(error);
  }
);

// Interceptor para agregar token automáticamente
api.interceptors.request.use(
  (config) => {
    const token = localStorage.getItem('token');
    if (token) {
      config.headers.Authorization = `Bearer ${token}`;
    }
    return config;
  },
  (error) => Promise.reject(error)
);

export const apiService = {
  // Autenticación
  async login(username, password) {
    const response = await api.post('/api/auth/login', { username, password });
    return response.data;
  },

  async logout() {
    const response = await api.post('/api/auth/logout');
    return response.data;
  },

  // Cámara - COMPLETO Y CORREGIDO
  async getCameraStatus() {
    const response = await api.get('/api/camera/status');
    return response.data;
  },

  async getCameraConfig() {
    const response = await api.get('/api/camera/config');
    return response.data;
  },

  async updateCameraConfig(config) {
    // CORREGIDO: Asegurar que se envían todos los campos necesarios
    const fullConfig = {
      // Campos básicos requeridos
      rtsp_url: config.rtsp_url || '',
      fase: config.fase || 'fase1',
      direccion: config.direccion || 'norte',
      controladora_id: config.controladora_id || 'CTRL_001',
      controladora_ip: config.controladora_ip || '192.168.1.200',
      
      // Identificación de cámara
      camera_name: config.camera_name || '',
      camera_model: config.camera_model || '',
      camera_location: config.camera_location || '',
      camera_serial: config.camera_serial || '',
      
      // Configuración de red
      camera_ip: config.camera_ip || '',
      username: config.username || 'admin',
      password: config.password || '',
      port: config.port || '554',
      stream_path: config.stream_path || '/stream1',
      
      // Configuración de video
      resolution: config.resolution || '1920x1080',
      frame_rate: config.frame_rate || '30',
      bitrate: config.bitrate || '4000',
      encoding: config.encoding || 'H264',
      stream_quality: config.stream_quality || 'high',
      
      // Configuraciones avanzadas
      night_vision: config.night_vision || false,
      motion_detection: config.motion_detection || false,
      recording_enabled: config.recording_enabled || false,
      audio_enabled: config.audio_enabled || false,
      
      // Configuración de análisis
      detection_zones: config.detection_zones !== false,
      speed_calculation: config.speed_calculation !== false,
      vehicle_counting: config.vehicle_counting !== false,
      license_plate_recognition: config.license_plate_recognition || false,
      
      // Estado
      enabled: config.enabled !== false
    };

    const response = await api.post('/api/camera/config', fullConfig);
    return response.data;
  },

  async getCameraHealth() {
    const response = await api.get('/api/camera_health');
    return response.data;
  },

  getCameraFrameUrl(cameraId) {
    return `${api.defaults.baseURL}/api/camera/stream?camera=${cameraId}&t=${Date.now()}`;
  },

  // Sistema
  async getSystemConfig() {
    const response = await api.get('/api/config/system');
    return response.data;
  },

  async updateSystemConfig(config) {
    const response = await api.post('/api/config/system', config);
    return response.data;
  },

  // Análisis - COMPLETO
  async getLines() {
    const response = await api.get('/api/analysis/lines');
    return response.data;
  },

  async getZones() {
    const response = await api.get('/api/analysis/zones');
    return response.data;
  },

  async addLine(line) {
    const response = await api.post('/api/analysis/lines', line);
    return response.data;
  },

  async addZone(zone) {
    const response = await api.post('/api/analysis/zones', zone);
    return response.data;
  },

  async deleteLine(lineId) {
    const response = await api.delete(`/api/analysis/lines/${lineId}`);
    return response.data;
  },

  async deleteZone(zoneId) {
    const response = await api.delete(`/api/analysis/zones/${zoneId}`);
    return response.data;
  },

  async clearAnalysis() {
    const response = await api.post('/api/analysis/clear');
    return response.data;
  },

  // CORREGIDO: Eliminando método simulado - ahora usa datos reales
  async getCameraTracks(cameraId) {
    try {
      // Intentar obtener datos reales del tracker (cuando esté implementado)
      const response = await api.get(`/api/camera/tracks/${cameraId}`);
      return response.data;
    } catch (error) {
      // Fallback temporal mientras se implementa el endpoint
      console.warn('Endpoint de tracks no disponible, usando fallback temporal');
      return {
        total_tracks: 0,
        tracks: [],
        tracker_stats: {
          confirmed_tracks: 0
        },
        processing_stats: {
          frames_processed: 0,
          detections_count: 0,
          processing_time: 0
        }
      };
    }
  },

  // Datos y exportación
  async exportData(date, type, fase = null) {
    const params = new URLSearchParams({ date, type });
    if (fase) params.append('fase', fase);
    
    const response = await api.get(`/api/data/export?${params}`);
    return response.data;
  },

  // Controladora - CORREGIDO para manejar errores
  async updateTrafficLightStatus(fases) {
    try {
      const response = await api.post('/api/rojo_status', { fases });
      return response.data;
    } catch (error) {
      console.warn('Error updating traffic light status:', error);
      throw error;
    }
  },

  async getTrafficLightStatus() {
    try {
      const response = await api.get('/api/rojo_status');
      return response.data;
    } catch (error) {
      console.warn('Error getting traffic light status:', error);
      // Retornar estado por defecto en lugar de fallar
      return { fases: {} };
    }
  },

  // NUEVO: Método para obtener estadísticas en tiempo real
  async getRealTimeStats() {
    try {
      const response = await api.get('/api/stats/realtime');
      return response.data;
    } catch (error) {
      console.warn('Real-time stats endpoint not available');
      return {
        vehicles_in_zone: 0,
        current_detections: 0,
        processing_fps: 0
      };
    }
  },

  // NUEVO: Método para test de conectividad
  async testCameraConnection(rtspUrl) {
    try {
      const response = await api.post('/api/camera/test', { rtsp_url: rtspUrl });
      return response.data;
    } catch (error) {
      console.warn('Camera test endpoint not available');
      throw error;
    }
  }
};

export default api;