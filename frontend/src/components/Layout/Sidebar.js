import React from 'react';
import { NavLink } from 'react-router-dom';
import {
  HomeIcon,
  CameraIcon,
  Cog6ToothIcon,
  DocumentChartBarIcon,
  TruckIcon,
  AdjustmentsHorizontalIcon,
  ComputerDesktopIcon
} from '@heroicons/react/24/outline';

const Sidebar = ({ collapsed, cameras, selectedCamera, onCameraSelect }) => {
  const navigation = [
    { 
      name: 'Dashboard', 
      href: '/dashboard', 
      icon: HomeIcon,
      description: 'Vista general del sistema'
    },
    { 
      name: 'Vista de Cámara', 
      href: '/camera', 
      icon: CameraIcon,
      description: 'Stream en vivo y configuración de análisis'
    },
    { 
      name: 'Config. Cámara', 
      href: '/camera-config', 
      icon: Cog6ToothIcon,
      description: 'Configuración RTSP y parámetros de cámara'
    },
    { 
      name: 'Config. Análisis', 
      href: '/analysis-config', 
      icon: AdjustmentsHorizontalIcon,
      description: 'Parámetros de detección y tracking'
    },
    { 
      name: 'Config. Sistema', 
      href: '/system-config', 
      icon: ComputerDesktopIcon,
      description: 'Configuración general del sistema'
    },
    { 
      name: 'Reportes', 
      href: '/reports', 
      icon: DocumentChartBarIcon,
      description: 'Análisis de datos y exportación'
    },
  ];

  return (
    <div className={`bg-gray-800 min-h-screen p-4 transition-all duration-300 ${
      collapsed ? 'w-16' : 'w-64'
    }`}>
      {/* Header del sidebar */}
      <div className="flex items-center mb-8">
        <TruckIcon className="h-8 w-8 text-blue-500 flex-shrink-0" />
        {!collapsed && (
          <div className="ml-3">
            <h1 className="text-xl font-bold text-white">Vehicle Detection</h1>
            <p className="text-xs text-gray-400">Radxa Rock 5T</p>
          </div>
        )}
      </div>
      
      {/* Navegación principal */}
      <nav className="space-y-2">
        {navigation.map((item) => (
          <NavLink
            key={item.name}
            to={item.href}
            className={({ isActive }) =>
              `flex items-center px-3 py-3 text-sm font-medium rounded-md transition-colors group ${
                isActive
                  ? 'bg-blue-600 text-white'
                  : 'text-gray-300 hover:bg-gray-700 hover:text-white'
              }`
            }
            title={collapsed ? item.description : ''}
          >
            <item.icon className={`h-5 w-5 flex-shrink-0 ${
              collapsed ? 'mx-auto' : 'mr-3'
            }`} />
            {!collapsed && (
              <div>
                <div>{item.name}</div>
                <div className="text-xs text-gray-400 group-hover:text-gray-300">
                  {item.description}
                </div>
              </div>
            )}
          </NavLink>
        ))}
      </nav>
      
      {/* Información del sistema */}
      {!collapsed && (
        <div className="mt-8 pt-8 border-t border-gray-700">
          <div className="text-xs text-gray-400 space-y-1">
            <div className="flex justify-between">
              <span>Hardware:</span>
              <span className="text-gray-300">Radxa 5T</span>
            </div>
            <div className="flex justify-between">
              <span>Versión:</span>
              <span className="text-gray-300">1.0.0</span>
            </div>
            <div className="flex justify-between">
              <span>NPU:</span>
              <span className="text-green-400">RKNN</span>
            </div>
          </div>
        </div>
      )}

      {/* Información de cámaras */}
      {!collapsed && cameras && cameras.length > 0 && (
        <div className="mt-6">
          <h3 className="text-xs font-semibold text-gray-400 uppercase tracking-wider mb-2">
            Cámaras
          </h3>
          <div className="space-y-1">
            {cameras.map((camera) => (
              <button
                key={camera.id}
                onClick={() => onCameraSelect && onCameraSelect(camera.id)}
                className={`w-full text-left px-3 py-2 text-sm rounded-md transition-colors ${
                  selectedCamera === camera.id
                    ? 'bg-blue-600 text-white'
                    : 'text-gray-300 hover:bg-gray-700 hover:text-white'
                }`}
              >
                <div className="flex items-center">
                  <div className={`w-2 h-2 rounded-full mr-2 ${
                    camera.enabled ? 'bg-green-500' : 'bg-gray-500'
                  }`}></div>
                  <span className="truncate">{camera.name}</span>
                </div>
              </button>
            ))}
          </div>
        </div>
      )}

      {/* Toggle collapse button */}
      <div className="absolute bottom-4 left-4 right-4">
        <button
          onClick={() => {/* Implementar toggle en el componente padre */}}
          className="w-full p-2 text-gray-400 hover:text-white transition-colors"
          title={collapsed ? 'Expandir sidebar' : 'Contraer sidebar'}
        >
          <svg 
            className={`w-5 h-5 mx-auto transition-transform ${collapsed ? 'rotate-180' : ''}`}
            fill="none" 
            stroke="currentColor" 
            viewBox="0 0 24 24"
          >
            <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M15 19l-7-7 7-7" />
          </svg>
        </button>
      </div>
    </div>
  );
};

export default Sidebar;