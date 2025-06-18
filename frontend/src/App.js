import React, { useState, useEffect } from 'react';
import { BrowserRouter as Router, Routes, Route, Navigate } from 'react-router-dom';
import { ToastContainer, toast } from 'react-toastify';
import 'react-toastify/dist/ReactToastify.css';

// Componentes principales
import Sidebar from './components/Layout/Sidebar';
import Header from './components/Layout/Header';
import CameraView from './components/CameraView/CameraView';
import CameraConfig from './components/CameraConfig/CameraConfig';
import AnalysisConfig from './components/AnalysisConfig/AnalysisConfig';
import Dashboard from './components/Dashboard/Dashboard';
import Reports from './components/Reports/Reports';
import SystemConfig from './components/SystemConfig/SystemConfig';
import Login from './components/Login';

// Servicios
import { apiService } from './services/api';

// Contexto global
import { SystemProvider, useSystem } from './context/SystemContext';

// Estilos
import './App.css';

// Componente de autenticación
function AuthWrapper({ children }) {
  const [isAuthenticated, setIsAuthenticated] = useState(false);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    // Verificar token existente
    const token = localStorage.getItem('token');
    if (token) {
      setIsAuthenticated(true);
    }
    setLoading(false);
  }, []);

  const handleLogin = async (username, password) => {
    try {
      const response = await apiService.login(username, password);
      localStorage.setItem('token', response.token);
      setIsAuthenticated(true);
      toast.success('Login exitoso');
    } catch (error) {
      toast.error('Credenciales incorrectas');
      throw error;
    }
  };

  const handleLogout = async () => {
    try {
      await apiService.logout();
    } catch (error) {
      // Ignorar errores de logout
    } finally {
      localStorage.removeItem('token');
      setIsAuthenticated(false);
      toast.info('Sesión cerrada');
    }
  };

  if (loading) {
    return (
      <div className="min-h-screen bg-gray-900 flex items-center justify-center">
        <div className="text-white">Cargando...</div>
      </div>
    );
  }

  if (!isAuthenticated) {
    return <Login onLogin={handleLogin} />;
  }

  return React.cloneElement(children, { onLogout: handleLogout });
}

function AppContent({ onLogout }) {
  const { systemStatus, cameras, selectedCamera, setSelectedCamera, loadSystemData } = useSystem();
  const [sidebarCollapsed, setSidebarCollapsed] = useState(false);

  useEffect(() => {
    loadSystemData();
  }, [loadSystemData]);

  const handleCameraSelect = (cameraId) => {
    setSelectedCamera(cameraId);
  };

  const handleRefresh = async () => {
    try {
      await loadSystemData();
      toast.success('Datos actualizados');
    } catch (error) {
      toast.error('Error actualizando datos');
    }
  };

  return (
    <div className="min-h-screen bg-gray-900">
      <div className="flex">
        {/* Sidebar */}
        <Sidebar
          collapsed={sidebarCollapsed}
          cameras={cameras}
          selectedCamera={selectedCamera}
          onCameraSelect={handleCameraSelect}
        />

        {/* Contenido principal */}
        <div className={`flex-1 transition-all duration-300 ${
          sidebarCollapsed ? 'ml-16' : 'ml-64'
        }`}>
          {/* Header CORREGIDO con props necesarios */}
          <Header
            systemStatus={systemStatus}
            onToggleSidebar={() => setSidebarCollapsed(!sidebarCollapsed)}
            onRefresh={handleRefresh}
            onLogout={onLogout}
          />

          <main className="p-6">
            <Routes>
              <Route path="/" element={<Navigate to="/dashboard" replace />} />
              <Route path="/dashboard" element={<Dashboard />} />
              <Route path="/camera" element={<CameraView />} />
              <Route path="/camera-config" element={<CameraConfig />} />
              <Route path="/analysis-config" element={<AnalysisConfig />} />
              <Route path="/system-config" element={<SystemConfig />} />
              <Route path="/reports" element={<Reports />} />
            </Routes>
          </main>
        </div>
      </div>

      {/* Toast notifications */}
      <ToastContainer
        position="top-right"
        autoClose={3000}
        hideProgressBar={false}
        newestOnTop={false}
        closeOnClick
        rtl={false}
        pauseOnFocusLoss
        draggable
        pauseOnHover
        theme="dark"
      />
    </div>
  );
}

function App() {
  return (
    <SystemProvider>
      <Router>
        <AuthWrapper>
          <AppContent />
        </AuthWrapper>
      </Router>
    </SystemProvider>
  );
}

export default App