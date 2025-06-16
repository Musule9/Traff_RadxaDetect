import React from 'react';
import { 
  ArrowRightOnRectangleIcon, 
  UserIcon,
  WifiIcon,
  ExclamationTriangleIcon
} from '@heroicons/react/24/outline';

const Header = ({ user, onLogout }) => {
  const [systemStatus, setSystemStatus] = React.useState({
    camera: false,
    controller: false,
    processing: false
  });

  React.useEffect(() => {
    // Simular estado del sistema
    setSystemStatus({
      camera: true,
      controller: Math.random() > 0.3,
      processing: true
    });
  }, []);

  return (
    <header className="bg-gray-800 shadow-sm border-b border-gray-700 px-6 py-4">
      <div className="flex justify-between items-center">
        <div className="flex items-center space-x-6">
          <h2 className="text-lg font-semibold text-white">
            Sistema de Detección Vehicular
          </h2>
          
          {/* Indicadores de estado */}
          <div className="flex items-center space-x-4 text-sm">
            <div className="flex items-center">
              <div className={`w-2 h-2 rounded-full mr-2 ${
                systemStatus.camera ? 'bg-green-500' : 'bg-red-500'
              }`}></div>
              <span className="text-gray-300">Cámara</span>
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
          {/* Alertas */}
          {!systemStatus.controller && (
            <div className="flex items-center text-yellow-400">
              <ExclamationTriangleIcon className="h-5 w-5 mr-1" />
              <span className="text-sm">Controladora desconectada</span>
            </div>
          )}
          
          {/* Usuario */}
          <div className="flex items-center space-x-3">
            <div className="flex items-center text-gray-300">
              <UserIcon className="h-5 w-5 mr-2" />
              <span className="text-sm">{user?.username}</span>
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