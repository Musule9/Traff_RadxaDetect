import axios from 'axios';

const api = axios.create({
  baseURL: process.env.REACT_APP_API_URL || '',
  timeout: 15000, // Aumentado para pruebas de conexión
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
  // ============================================================================
  // AUTENTICACIÓN
  // ============================================================================
  async login(username, password) {
    console.log('🔐 Intentando login:', username);
    const response = await api.post('/api/auth/login', { username, password });
    console.log('✅ Login exitoso:', response.data);
    return response.data;
  },

  async logout() {
    console.log('🚪 Logout');
    const response = await api.post('/api/auth/logout');
    return response.data;
  },

  // ============================================================================
  // CÁMARA - SIMPLIFICADO Y CORREGIDO
  // ============================================================================
  async getCameraConfig() {
    console.log('📥 Obteniendo configuración de cámara...');
    const response = await api.get('/api/camera/config');
    console.log('📦 Configuración recibida:', response.data);
    return response.data;
  },

  async updateCameraConfig(config) {
    console.log('📤 Enviando configuración de cámara:', config);
    
    // Asegurar que tenemos todos los campos necesarios
    const cleanConfig = {
      rtsp_url: config.rtsp_url || '',
      fase: config.fase || 'fase1',
      direccion: config.direccion || 'norte',
      controladora_id: config.controladora_id || 'CTRL_001',
      controladora_ip: config.controladora_ip || '192.168.1.200',
      camera_name: config.camera_name || '',
      camera_location: config.camera_location || '',
      camera_ip: config.camera_ip || '',
      username: config.username || 'admin',
      password: config.password || '',
      port: config.port || '554',
      stream_path: config.stream_path || '/stream1',
      resolution: config.resolution || '1920x1080',
      frame_rate: config.frame_rate || '30',
      enabled: true
    };

    const response = await api.post('/api/camera/config', cleanConfig);
    console.log('✅ Configuración guardada:', response.data);
    return response.data;
  },

  async resetCameraConfig() {
    console.log('🧹 Reseteando configuración de cámara...');
    const response = await api.post('/api/camera/config/reset');
    console.log('✅ Configuración reseteada:', response.data);
    return response.data;
  },

  async getCameraStatus() {
    console.log('📊 Obteniendo estado de cámara...');
    const response = await api.get('/api/camera/status');
    console.log('📊 Estado de cámara:', response.data);
    return response.data;
  },

  async restartCameraProcessing() {
    console.log('🔄 Reiniciando procesamiento de cámara...');
    const response = await api.post('/api/camera/restart');
    console.log('✅ Procesamiento reiniciado:', response.data);
    return response.data;
  },

  async testCameraStream(rtspUrl) {
    console.log('🧪 Probando stream RTSP:', rtspUrl);
    
    try {
      const response = await api.post('/api/camera/test', { 
        rtsp_url: rtspUrl 
      });
      console.log('📊 Resultado del test:', response.data);
      return response.data;
    } catch (error) {
      console.error('❌ Error en test de stream:', error);
      
      // Si el endpoint no existe, hacer una validación básica
      if (error.response?.status === 404) {
        console.log('⚠️ Endpoint de test no disponible, validando URL...');
        
        if (!rtspUrl || !rtspUrl.trim()) {
          return { success: false, message: 'URL RTSP vacía' };
        }
        
        if (!rtspUrl.startsWith('rtsp://')) {
          return { success: false, message: 'URL debe empezar con rtsp://' };
        }
        
        return { 
          success: true, 
          message: 'URL RTSP tiene formato válido (test offline)',
          frames_tested: 0
        };
      }
      
      throw error;
    }
  },

  async getCameraHealth() {
    console.log('🏥 Obteniendo health de cámara...');
    const response = await api.get('/api/camera_health');
    console.log('🏥 Health:', response.data);
    return response.data;
  },

  // ============================================================================
  // ANÁLISIS - LÍNEAS Y ZONAS
  // ============================================================================
  async getLines() {
    console.log('📏 Obteniendo líneas...');
    const response = await api.get('/api/analysis/lines');
    console.log('📏 Líneas:', response.data);
    return response.data;
  },

  async addLine(line) {
    console.log('➕ Agregando línea:', line);
    const response = await api.post('/api/analysis/lines', line);
    console.log('✅ Línea agregada:', response.data);
    return response.data;
  },

  async deleteLine(lineId) {
    console.log('🗑️ Eliminando línea:', lineId);
    const response = await api.delete(`/api/analysis/lines/${lineId}`);
    console.log('✅ Línea eliminada:', response.data);
    return response.data;
  },

  async getZones() {
    console.log('🔳 Obteniendo zonas...');
    const response = await api.get('/api/analysis/zones');
    console.log('🔳 Zonas:', response.data);
    return response.data;
  },

  async addZone(zone) {
    console.log('➕ Agregando zona:', zone);
    const response = await api.post('/api/analysis/zones', zone);
    console.log('✅ Zona agregada:', response.data);
    return response.data;
  },

  async deleteZone(zoneId) {
    console.log('🗑️ Eliminando zona:', zoneId);
    const response = await api.delete(`/api/analysis/zones/${zoneId}`);
    console.log('✅ Zona eliminada:', response.data);
    return response.data;
  },

  async clearAnalysis() {
    console.log('🧹 Limpiando análisis...');
    const response = await api.post('/api/analysis/clear');
    console.log('✅ Análisis limpiado:', response.data);
    return response.data;
  },

  // ============================================================================
  // SISTEMA
  // ============================================================================
  async getSystemConfig() {
    try {
      console.log('⚙️ Obteniendo configuración del sistema...');
      const response = await api.get('/api/config/system');
      console.log('⚙️ Config sistema:', response.data);
      return response.data;
    } catch (error) {
      console.warn('⚠️ Error obteniendo config del sistema:', error);
      // Retornar configuración por defecto
      return {
        confidence_threshold: 0.5,
        night_vision_enhancement: true,
        show_overlay: true,
        data_retention_days: 30,
        target_fps: 30,
        log_level: 'INFO'
      };
    }
  },

  async updateSystemConfig(config) {
    console.log('📤 Actualizando configuración del sistema:', config);
    const response = await api.post('/api/config/system', config);
    console.log('✅ Config sistema actualizada:', response.data);
    return response.data;
  },

  // ============================================================================
  // DATOS Y EXPORTACIÓN
  // ============================================================================
  async exportData(date, type, fase = null) {
    console.log(`📊 Exportando datos: ${date}, ${type}, ${fase}`);
    
    const params = new URLSearchParams({ date, type });
    if (fase) params.append('fase', fase);
    
    const response = await api.get(`/api/data/export?${params}`);
    console.log('📊 Datos exportados:', response.data);
    return response.data;
  },

  // ============================================================================
  // CONTROLADORA - SIMPLIFICADO
  // ============================================================================
  async getTrafficLightStatus() {
    try {
      console.log('🚦 Obteniendo estado de semáforos...');
      const response = await api.get('/api/rojo_status');
      console.log('🚦 Estado semáforos:', response.data);
      return response.data;
    } catch (error) {
      console.warn('⚠️ Error obteniendo estado de semáforos:', error);
      // Retornar estado por defecto
      return { 
        fases: {
          fase1: false,
          fase2: false,
          fase3: false,
          fase4: false
        }
      };
    }
  },

  async updateTrafficLightStatus(fases) {
    try {
      console.log('📤 Actualizando estado de semáforos:', fases);
      const response = await api.post('/api/rojo_status', { fases });
      console.log('✅ Estado actualizado:', response.data);
      return response.data;
    } catch (error) {
      console.warn('⚠️ Error actualizando estado de semáforos:', error);
      throw error;
    }
  },

  // ============================================================================
  // MÉTODOS DE UTILIDAD
  // ============================================================================
  getCameraFrameUrl() {
    return `${api.defaults.baseURL}/api/camera/stream?t=${Date.now()}`;
  },

  // Método para verificar conectividad general
  async ping() {
    try {
      const response = await api.get('/api/camera_health');
      return response.status === 200;
    } catch (error) {
      return false;
    }
  },

  // ============================================================================
  // MÉTODOS PARA COMPATIBILIDAD (TEMPORAL)
  // ============================================================================
  
  // Método temporal para estadísticas en tiempo real
  async getRealTimeStats() {
    try {
      const response = await api.get('/api/stats/realtime');
      return response.data;
    } catch (error) {
      console.warn('⚠️ Real-time stats no disponibles');
      return {
        vehicles_in_zone: 0,
        current_detections: 0,
        processing_fps: 0
      };
    }
  },

  // Método temporal para tracks de cámara
  async getCameraTracks(cameraId) {
    try {
      const response = await api.get(`/api/camera/tracks/${cameraId}`);
      return response.data;
    } catch (error) {
      console.warn('⚠️ Camera tracks no disponibles');
      return {
        total_tracks: 0,
        tracks: [],
        tracker_stats: { confirmed_tracks: 0 },
        processing_stats: {
          frames_processed: 0,
          detections_count: 0,
          processing_time: 0
        }
      };
    }
  }
};

export default api;