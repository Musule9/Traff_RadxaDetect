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

  // Cámara
  async getCameraStatus() {
    const response = await api.get('/api/camera/status');
    return response.data;
  },

  async updateCameraConfig(config) {
    const response = await api.post('/api/camera/config', config);
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

  // Análisis
  async addLine(line) {
    const response = await api.post('/api/analysis/lines', line);
    return response.data;
  },

  async addZone(zone) {
    const response = await api.post('/api/analysis/zones', zone);
    return response.data;
  },

  async getCameraTracks(cameraId) {
    // Simular datos hasta que esté implementado
    return {
      total_tracks: Math.floor(Math.random() * 5),
      tracks: [],
      tracker_stats: {
        confirmed_tracks: Math.floor(Math.random() * 3)
      },
      processing_stats: {
        frames_processed: Math.floor(Math.random() * 1000),
        detections_count: Math.floor(Math.random() * 100),
        processing_time: Math.random() * 0.1
      }
    };
  },

  // Datos y exportación
  async exportData(date, type, fase = null) {
    const params = new URLSearchParams({ date, type });
    if (fase) params.append('fase', fase);
    
    const response = await api.get(`/api/data/export?${params}`);
    return response.data;
  },

  // Controladora
  async updateTrafficLightStatus(fases) {
    const response = await api.post('/api/rojo_status', { fases });
    return response.data;
  },

  async getTrafficLightStatus() {
    const response = await api.get('/api/rojo_status');
    return response.data;
  }
};

export default api;