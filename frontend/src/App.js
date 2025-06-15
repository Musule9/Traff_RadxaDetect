import React, { useState, useEffect } from 'react';
import { BrowserRouter as Router, Routes, Route, Navigate } from 'react-router-dom';
import { ToastContainer, toast } from 'react-toastify';
import 'react-toastify/dist/ReactToastify.css';

// Componentes principales - RUTAS CORREGIDAS
import Sidebar from './components/Layout/Sidebar';
import Header from './components/Layout/Header';
import CameraView from './components/CameraView/CameraView';
import CameraConfig from './components/CameraConfig/CameraConfig';
import AnalysisConfig from './components/AnalysisConfig/AnalysisConfig';
import Dashboard from './components/Dashboard/Dashboard';
import Reports from './components/Reports/Reports';
import SystemConfig from './components/SystemConfig/SystemConfig';
import LoadingSpinner from './components/Common/LoadingSpinner';

// Servicios
import { apiService } from './services/api';

// Contexto global
import { SystemProvider, useSystem } from './context/SystemContext';

// Estilos
import './App.css';

function AppContent() {
  const { systemStatus, cameras, selectedCamera, setSelectedCamera, loadSystemData } = useSystem();
  const [sidebarCollapsed, setSidebarCollapsed] = useState(false);
  const [isLoading, setIsLoading] = useState(true);
  const [currentView, setCurrentView] = useState('dashboard');

  useEffect(() => {
    initializeApp();
  }, []);

  const initializeApp = async () => {
    try {
      setIsLoading(true);
      
      // Cargar datos iniciales del sistema
      await loadSystemData();
      
      toast.success('Sistema inicializado correctamente');
      
    } catch (error) {
      console.error('Error inicializando aplicación:', error);
      toast.error('Error inicializando el sistema');
    } finally {
      setIsLoading(false);
    }
  };

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

  const handleViewChange = (view) => {
    setCurrentView(view);
  };

  if (isLoading) {
    return (
      <div className="min-h-screen bg-gray-900 flex items-center justify-center">
        <LoadingSpinner text="Inicializando sistema..." />
      </div>
    );
  }

  return (
    <div className="min-h-screen bg-gray-900">
      <div className="flex">
        {/* Sidebar */}
        <Sidebar
          collapsed={sidebarCollapsed}
          currentView={currentView}
          onViewChange={handleViewChange}
          cameras={cameras}
          selectedCamera={selectedCamera}
          onCameraSelect={handleCameraSelect}
        />

        {/* Contenido principal */}
        <div className={`flex-1 transition-all duration-300 ${
          sidebarCollapsed ? 'ml-16' : 'ml-64'
        }`}>
          <Header
            systemStatus={systemStatus}
            onToggleSidebar={() => setSidebarCollapsed(!sidebarCollapsed)}
            onRefresh={handleRefresh}
          />

          <main className="p-6">
            <Routes>
              <Route path="/" element={<Navigate to="/dashboard" replace />} />
              <Route path="/dashboard" element={<Dashboard />} />
              <Route path="/camera" element={<CameraView />} />
              <Route path="/camera-config" element={<CameraConfig />} />
              <Route path="/analysis-config" element={<AnalysisConfig />} />
              <Route path="/reports" element={<Reports />} />
              <Route path="/system-config" element={<SystemConfig />} />
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
        <AppContent />
      </Router>
    </SystemProvider>
  );
}

export default App;