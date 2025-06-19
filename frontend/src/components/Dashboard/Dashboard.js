// REEMPLAZAR en frontend/src/components/Dashboard/Dashboard.js

import React, { useState, useEffect } from 'react';
import { 
  CameraIcon, 
  ChartBarIcon, 
  ClockIcon, 
  ExclamationTriangleIcon,
  CheckCircleIcon,
  TruckIcon
} from '@heroicons/react/24/outline';
import { useSystem } from '../../context/SystemContext';
import { apiService } from '../../services/api';

const Dashboard = () => {
  const { systemStatus } = useSystem();
  const [stats, setStats] = useState({
    vehiclesInZone: 0,
    totalCrossings: 0,
    avgSpeed: 0,
    trafficLightStatus: 'verde'
  });
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    fetchDashboardData();
    const interval = setInterval(fetchDashboardData, 10000); // Actualizar cada 10 segundos
    return () => clearInterval(interval);
  }, []);

  const fetchDashboardData = async () => {
    try {
      setLoading(true);

      // CORREGIDO: Usar datos reales en lugar de simulación
      const [
        trafficStatus, 
        cameraStatus,
        todayData
      ] = await Promise.all([
        apiService.getTrafficLightStatus().catch(() => ({ fases: {} })),
        apiService.getCameraStatus().catch(() => ({ connected: false })),
        apiService.exportData(
          new Date().toISOString().split('T')[0].replace(/-/g, '_'),
          'vehicle'
        ).catch(() => ({ data: [] }))
      ]);

      // Calcular estadísticas reales
      const vehicleData = todayData.data || [];
      const totalCrossings = vehicleData.length;
      
      const speedValues = vehicleData
        .filter(v => v.velocidad && v.velocidad > 0)
        .map(v => v.velocidad);
      
      const avgSpeed = speedValues.length > 0 
        ? speedValues.reduce((a, b) => a + b, 0) / speedValues.length 
        : 0;

      // Estado del semáforo real
      const cameraConfig = await apiService.getCameraConfig();
      const currentPhase = cameraConfig.fase || 'fase1';
      const isRed = trafficStatus.fases?.[currentPhase] || false;

      setStats({
        vehiclesInZone: 0, // Se actualizará desde el video processor
        totalCrossings: totalCrossings,
        avgSpeed: Math.round(avgSpeed),
        trafficLightStatus: isRed ? 'rojo' : 'verde'
      });

    } catch (error) {
      console.error('Error fetching dashboard data:', error);
      // Mantener datos anteriores en caso de error
    } finally {
      setLoading(false);
    }
  };

  // Resto del componente igual...
  const StatCard = ({ icon: Icon, title, value, color = "blue", subtitle }) => (
    <div className="bg-gray-800 rounded-lg p-6">
      <div className="flex items-center">
        <Icon className={`h-8 w-8 text-${color}-500`} />
        <div className="ml-4">
          <p className="text-sm font-medium text-gray-400">{title}</p>
          <p className="text-2xl font-bold text-white">{value}</p>
          {subtitle && <p className="text-xs text-gray-500">{subtitle}</p>}
        </div>
      </div>
    </div>
  );

  if (loading) {
    return (
      <div className="flex items-center justify-center h-64">
        <div className="animate-spin rounded-full h-12 w-12 border-b-2 border-blue-500"></div>
      </div>
    );
  }

  return (
    <div className="space-y-6">
      <div className="flex justify-between items-center">
        <h1 className="text-3xl font-bold text-white">Dashboard</h1>
        <div className="flex items-center space-x-2">
          <div className={`w-3 h-3 rounded-full ${systemStatus.camera ? 'bg-green-500' : 'bg-red-500'}`}></div>
          <span className="text-gray-300">
            {systemStatus.camera ? 'Sistema Operativo' : 'Sistema Desconectado'}
          </span>
        </div>
      </div>
      
      {/* Cards de estado - USANDO DATOS REALES */}
      <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-6">
        <StatCard
          icon={CameraIcon}
          title="Estado de Cámara"
          value={systemStatus.camera ? 'Conectada' : 'Desconectada'}
          color={systemStatus.camera ? 'green' : 'red'}
          subtitle={systemStatus.camera ? `${systemStatus.fps || 0} FPS` : 'Verificar configuración'}
        />

        <StatCard
          icon={ChartBarIcon}
          title="Vehículos en Zona"
          value={stats.vehiclesInZone}
          color="yellow"
          subtitle="Zona de semáforo"
        />

        <StatCard
          icon={TruckIcon}
          title="Cruces Hoy"
          value={stats.totalCrossings}
          color="blue"
          subtitle="Total de detecciones"
        />

        <StatCard
          icon={ClockIcon}
          title="Velocidad Promedio"
          value={`${stats.avgSpeed} km/h`}
          color="purple"
          subtitle="Promedio del día"
        />
      </div>

      {/* Estado del semáforo - USANDO DATOS REALES */}
      <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
        <div className="bg-gray-800 rounded-lg p-6">
          <h2 className="text-xl font-semibold text-white mb-4">Estado del Semáforo</h2>
          <div className="flex items-center space-x-4">
            <div className={`w-6 h-6 rounded-full ${
              stats.trafficLightStatus === 'rojo' ? 'bg-red-500' : 
              stats.trafficLightStatus === 'amarillo' ? 'bg-yellow-500' : 'bg-green-500'
            }`}></div>
            <div>
              <span className="text-white text-lg capitalize">{stats.trafficLightStatus}</span>
              <p className="text-gray-400 text-sm">Estado actual del semáforo</p>
            </div>
          </div>
        </div>

        <div className="bg-gray-800 rounded-lg p-6">
          <h2 className="text-xl font-semibold text-white mb-4">Estado del Sistema</h2>
          <div className="space-y-3">
            <div className="flex items-center justify-between">
              <span className="text-gray-300">Procesamiento</span>
              <div className="flex items-center">
                {systemStatus.processing ? (
                  <CheckCircleIcon className="h-5 w-5 text-green-500" />
                ) : (
                  <ExclamationTriangleIcon className="h-5 w-5 text-red-500" />
                )}
                <span className="ml-2 text-sm text-gray-300">
                  {systemStatus.processing ? 'Activo' : 'Inactivo'}
                </span>
              </div>
            </div>
            
            <div className="flex items-center justify-between">
              <span className="text-gray-300">Controladora</span>
              <div className="flex items-center">
                {systemStatus.controller ? (
                  <CheckCircleIcon className="h-5 w-5 text-green-500" />
                ) : (
                  <ExclamationTriangleIcon className="h-5 w-5 text-yellow-500" />
                )}
                <span className="ml-2 text-sm text-gray-300">
                  {systemStatus.controller ? 'Conectada' : 'Desconectada'}
                </span>
              </div>
            </div>
          </div>
        </div>
      </div>

      {/* Resto del componente igual... */}
      <div className="bg-gray-800 rounded-lg p-6">
        <h2 className="text-xl font-semibold text-white mb-4">Configuración Inicial</h2>
        <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
          <div>
            <h3 className="text-lg font-medium text-white mb-2">Pasos de Configuración</h3>
            <ol className="space-y-2 text-gray-300 text-sm">
              <li className="flex items-start">
                <span className="bg-blue-600 text-white rounded-full w-5 h-5 flex items-center justify-center text-xs mr-2 mt-0.5">1</span>
                Configure la URL RTSP de su cámara en Configuración de Cámara
              </li>
              <li className="flex items-start">
                <span className="bg-blue-600 text-white rounded-full w-5 h-5 flex items-center justify-center text-xs mr-2 mt-0.5">2</span>
                Defina las líneas de conteo y velocidad en Vista de Cámara
              </li>
              <li className="flex items-start">
                <span className="bg-blue-600 text-white rounded-full w-5 h-5 flex items-center justify-center text-xs mr-2 mt-0.5">3</span>
                Configure la zona de detección de semáforo en rojo
              </li>
              <li className="flex items-start">
                <span className="bg-blue-600 text-white rounded-full w-5 h-5 flex items-center justify-center text-xs mr-2 mt-0.5">4</span>
                Establezca la IP de su controladora de semáforos
              </li>
            </ol>
          </div>
          
          <div>
            <h3 className="text-lg font-medium text-white mb-2">Especificaciones</h3>
            <div className="space-y-2 text-gray-300 text-sm">
              <div className="flex justify-between">
                <span>Hardware:</span>
                <span>Radxa Rock 5T</span>
              </div>              
              <div className="flex justify-between">
                <span>Tracker:</span>
                <span>BYTETracker</span>
              </div>
              <div className="flex justify-between">
                <span>Resolución:</span>
                <span>1080p @ 30 FPS</span>
              </div>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
};

export default Dashboard;