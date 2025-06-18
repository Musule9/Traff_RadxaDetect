import React, { useEffect, useState } from 'react';
import { 
  ArrowRightOnRectangleIcon, 
  UserIcon,
  WifiIcon,
  ExclamationTriangleIcon,
  ArrowPathIcon
} from '@heroicons/react/24/outline';
import { apiService } from '../../services/api';

const Header = ({ systemStatus, onToggleSidebar, onRefresh, onLogout }) => {
  const [user] = useState({ username: 'admin' }); // Simplificado por ahora
  const [lastUpdate, setLastUpdate] = useState(new Date());

  useEffect(() => {
    const interval = setInterval(() => {
      setLastUpdate(new Date());
    }, 30000); // Actualizar cada 30 segundos

    return () => clearInterval(interval);
  }, []);

  const handleRefresh = async () => {
    if (onRefresh) {
      await onRefresh();
      setLastUpdate(new Date());
    }
  };

  return (
    <header className="bg-gray-800 shadow-sm border-b border-gray-700 px-6 py-4">
      <div className="flex justify-between items-center">
        <div className="flex items-center space-x-6">
          <h2 className="text-lg font-semibold text-white">
            Sistema de Detección Vehicular - Radxa Rock 5T
          </h2>
          
          {/* Indicadores de estado REALES */}
          <div className="flex items-center space-x-4 text-sm">
            <div className="flex items-center">
              <div className={`w-2 h-2 rounded-full mr-2 ${
                systemStatus.camera ? 'bg-green-500 animate-pulse' : 'bg-red-500'
              }`}></div>
              <span className="text-gray-300">
                Cámara {systemStatus.camera ? `(${systemStatus.fps || 0} FPS)` : '(Desconectada)'}
              </span>
            </div>
            
            <div className="flex items-center">
              <div className={`w-2 h-2 rounded-full mr-2 ${
                systemStatus.controller ? 'bg-green-500' : 'bg-yellow-500'
              }`}></div>
              <span className="text-gray-300">Controladora</span>
            </div>
            
            <div className="flex items-center">
              <div className={`w-2 h-2 rounded-full mr-2 ${
                systemStatus.processing ? 'bg-green-500' : 'bg-red-500'
              }`}></div>
              <span className="text-gray-300">Procesamiento</span>
            </div>
          </div>
        </div>
        
        <div className="flex items-center space-x-4">
          {/* Botón de actualizar */}
          <button
            onClick={handleRefresh}
            className="flex items-center px-3 py-1.5 text-sm text-gray-300 hover:text-white hover:bg-gray-700 rounded-md transition-colors"
            title="Actualizar estado"
          >
            <ArrowPathIcon className="h-4 w-4 mr-1" />
            Actualizar
          </button>

          {/* Última actualización */}
          <div className="text-xs text-gray-400">
            Actualizado: {lastUpdate.toLocaleTimeString()}
          </div>

          {/* Alertas REALES */}
          {!systemStatus.controller && (
            <div className="flex items-center text-yellow-400">
              <ExclamationTriangleIcon className="h-5 w-5 mr-1" />
              <span className="text-sm">Controladora desconectada</span>
            </div>
          )}

          {!systemStatus.camera && (
            <div className="flex items-center text-red-400">
              <ExclamationTriangleIcon className="h-5 w-5 mr-1" />
              <span className="text-sm">Cámara no configurada</span>
            </div>
          )}
          
          {/* Usuario */}
          <div className="flex items-center space-x-3">
            <div className="flex items-center text-gray-300">
              <UserIcon className="h-5 w-5 mr-2" />
              <span className="text-sm">{user?.username || 'Admin'}</span>
            </div>
            
            <button
              onClick={onLogout}
              className="flex items-center px-3 py-1.5 text-sm text-gray-300 hover:text-white hover:bg-gray-700 rounded-md transition-colors"
            >
              <ArrowRightOnRectangleIcon className="h-4 w-4 mr-1" />
              Salir
            </button>
          </div>
        </div>
      </div>
    </header>
  );
};

export default Header;