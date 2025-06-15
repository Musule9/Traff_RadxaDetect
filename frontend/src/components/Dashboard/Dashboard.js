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
      // Simular datos para el dashboard
      // En producción, estos vendrían de la API
      setStats({
        vehiclesInZone: Math.floor(Math.random() * 5),
        totalCrossings: Math.floor(Math.random() * 100) + 50,
        avgSpeed: Math.floor(Math.random() * 20) + 30,
        trafficLightStatus: Math.random() > 0.7 ? 'rojo' : 'verde'
      });
    } catch (error) {
      console.error('Error fetching dashboard data:', error);
    } finally {
      setLoading(false);
    }
  };

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
      
      {/* Cards de estado */}
      <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-6">
        <StatCard
          icon={CameraIcon}
          title="Estado de Cámara"
          value={systemStatus.camera ? 'Conectada' : 'Desconectada'}
          color={systemStatus.camera ? 'green' : 'red'}
          subtitle={systemStatus.camera ? `${systemStatus.fps} FPS` : 'Verificar configuración'}
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
          subtitle="Última hora"
        />
      </div>

      {/* Estado del semáforo */}
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
              <p className="text-gray-400 text-sm">Fase actual del semáforo</p>
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

      {/* Instrucciones de configuración */}
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
                <span>Modelo de IA:</span>
                <span>YOLOv8n + RKNN</span>
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