import React from 'react';
import { NavLink } from 'react-router-dom';
import {
  HomeIcon,
  CameraIcon,
  Cog6ToothIcon,
  DocumentChartBarIcon,
  TruckIcon
} from '@heroicons/react/24/outline';

const Sidebar = () => {
  const navigation = [
    { name: 'Dashboard', href: '/dashboard', icon: HomeIcon },
    { name: 'Vista de Cámara', href: '/camera', icon: CameraIcon },
    { name: 'Configuración', href: '/config', icon: Cog6ToothIcon },
    { name: 'Reportes', href: '/reports', icon: DocumentChartBarIcon },
  ];

  return (
    <div className="bg-gray-800 w-64 min-h-screen p-4">
      <div className="flex items-center mb-8">
        <TruckIcon className="h-8 w-8 text-blue-500 mr-3" />
        <h1 className="text-xl font-bold text-white">Detección Vehicular</h1>
      </div>
      
      <nav className="space-y-2">
        {navigation.map((item) => (
          <NavLink
            key={item.name}
            to={item.href}
            className={({ isActive }) =>
              `flex items-center px-4 py-3 text-sm font-medium rounded-md transition-colors ${
                isActive
                  ? 'bg-blue-600 text-white'
                  : 'text-gray-300 hover:bg-gray-700 hover:text-white'
              }`
            }
          >
            <item.icon className="h-5 w-5 mr-3" />
            {item.name}
          </NavLink>
        ))}
      </nav>
      
      <div className="mt-8 pt-8 border-t border-gray-700">
        <div className="text-xs text-gray-400 space-y-1">
          <p>Radxa Rock 5T</p>
          <p>Versión 1.0.0</p>
          <p>RKNN Habilitado</p>
        </div>
      </div>
    </div>
  );
};

export default Sidebar;