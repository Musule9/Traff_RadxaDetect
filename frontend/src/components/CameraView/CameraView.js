import React, { useState, useEffect, useRef } from 'react';
import { 
  PlayIcon, 
  PauseIcon, 
  PencilIcon,
  TrashIcon,
  CheckIcon,
  XMarkIcon,
  CameraIcon  // AGREGADO - FALTABA ESTE IMPORT
} from '@heroicons/react/24/outline';
import { toast } from 'react-toastify';
import { apiService } from '../../services/api';
import { useSystem } from '../../context/SystemContext';

const CameraView = () => {
  const { systemStatus } = useSystem();
  const [isStreamActive, setIsStreamActive] = useState(false);
  const [isDrawingLine, setIsDrawingLine] = useState(false);
  const [isDrawingZone, setIsDrawingZone] = useState(false);
  const [lines, setLines] = useState([]);
  const [zones, setZones] = useState([]);
  const [currentLine, setCurrentLine] = useState(null);
  const [currentZone, setCurrentZone] = useState([]);
  const [showOverlay, setShowOverlay] = useState(true);
  const [lineConfig, setLineConfig] = useState({
    name: '',
    lane: '',
    distance: 10.0,
    type: 'counting'
  });

  const imgRef = useRef(null);
  const streamUrl = '/api/camera/stream';

  useEffect(() => {
    if (systemStatus.camera) {
      setIsStreamActive(true);
    }
  }, [systemStatus.camera]);

  const handleMouseClick = (e) => {
    if (!isDrawingLine && !isDrawingZone) return;

    const rect = e.target.getBoundingClientRect();
    const x = Math.round(e.clientX - rect.left);
    const y = Math.round(e.clientY - rect.top);

    if (isDrawingLine) {
      if (!currentLine) {
        setCurrentLine({ start: { x, y }, end: null });
      } else {
        const newLine = {
          id: `line_${Date.now()}`,
          name: lineConfig.name || `Línea ${lines.length + 1}`,
          points: [[currentLine.start.x, currentLine.start.y], [x, y]],
          lane: lineConfig.lane || `carril_${lines.length + 1}`,
          line_type: lineConfig.type,
          distance_to_next: lineConfig.type === 'counting' ? lineConfig.distance : null
        };
        
        setLines([...lines, newLine]);
        setCurrentLine(null);
        setIsDrawingLine(false);
        toast.success('Línea agregada');
        
        // Reset form
        setLineConfig({
          name: '',
          lane: '',
          distance: 10.0,
          type: 'counting'
        });
      }
    } else if (isDrawingZone) {
      setCurrentZone([...currentZone, { x, y }]);
    }
  };

  const handleMouseMove = (e) => {
    if (isDrawingLine && currentLine && !currentLine.end) {
      const rect = e.target.getBoundingClientRect();
      const x = Math.round(e.clientX - rect.left);
      const y = Math.round(e.clientY - rect.top);
      
      setCurrentLine({
        ...currentLine,
        end: { x, y }
      });
    }
  };

  const finishZone = () => {
    if (currentZone.length >= 3) {
      const newZone = {
        id: `zone_${Date.now()}`,
        name: `Zona ${zones.length + 1}`,
        points: currentZone.map(p => [p.x, p.y]),
        zone_type: 'red_light'
      };
      setZones([...zones, newZone]);
      setCurrentZone([]);
      setIsDrawingZone(false);
      toast.success('Zona agregada');
    } else {
      toast.error('La zona debe tener al menos 3 puntos');
    }
  };

  const cancelDrawing = () => {
    setIsDrawingLine(false);
    setIsDrawingZone(false);
    setCurrentLine(null);
    setCurrentZone([]);
  };

  const saveConfiguration = async () => {
    try {
      for (const line of lines) {
        await apiService.addLine(line);
      }
      for (const zone of zones) {
        await apiService.addZone(zone);
      }
      toast.success('Configuración guardada exitosamente');
      setLines([]);
      setZones([]);
    } catch (error) {
      console.error('Error guardando configuración:', error);
      toast.error('Error guardando configuración');
    }
  };

  const clearAll = () => {
    setLines([]);
    setZones([]);
    setCurrentLine(null);
    setCurrentZone([]);
    setIsDrawingLine(false);
    setIsDrawingZone(false);
    toast.info('Configuración limpiada');
  };

  return (
    <div className="space-y-6">
      <div className="flex justify-between items-center">
        <h1 className="text-3xl font-bold text-white">Vista de Cámara</h1>
        <div className="flex space-x-4">
          <button
            onClick={() => setShowOverlay(!showOverlay)}
            className={`px-4 py-2 rounded-md transition-colors ${
              showOverlay ? 'bg-green-600 hover:bg-green-700' : 'bg-gray-600 hover:bg-gray-700'
            } text-white`}
          >
            {showOverlay ? 'Ocultar Overlay' : 'Mostrar Overlay'}
          </button>
          <button
            onClick={() => setIsStreamActive(!isStreamActive)}
            className="flex items-center px-4 py-2 bg-blue-600 text-white rounded-md hover:bg-blue-700"
          >
            {isStreamActive ? <PauseIcon className="h-5 w-5 mr-2" /> : <PlayIcon className="h-5 w-5 mr-2" />}
            {isStreamActive ? 'Pausar' : 'Iniciar'} Stream
          </button>
        </div>
      </div>

      {/* Configuración de línea */}
      {isDrawingLine && (
        <div className="bg-gray-800 rounded-lg p-4">
          <h3 className="text-lg font-semibold text-white mb-4">Configurar Línea</h3>
          <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
            <input
              type="text"
              placeholder="Nombre de línea"
              value={lineConfig.name}
              onChange={(e) => setLineConfig({...lineConfig, name: e.target.value})}
              className="px-3 py-2 bg-gray-700 text-white border border-gray-600 rounded-md"
            />
            <input
              type="text"
              placeholder="Carril"
              value={lineConfig.lane}
              onChange={(e) => setLineConfig({...lineConfig, lane: e.target.value})}
              className="px-3 py-2 bg-gray-700 text-white border border-gray-600 rounded-md"
            />
            <select
              value={lineConfig.type}
              onChange={(e) => setLineConfig({...lineConfig, type: e.target.value})}
              className="px-3 py-2 bg-gray-700 text-white border border-gray-600 rounded-md"
            >
              <option value="counting">Conteo</option>
              <option value="speed">Velocidad</option>
            </select>
            {lineConfig.type === 'counting' && (
              <input
                type="number"
                placeholder="Distancia (m)"
                value={lineConfig.distance}
                onChange={(e) => setLineConfig({...lineConfig, distance: parseFloat(e.target.value)})}
                className="px-3 py-2 bg-gray-700 text-white border border-gray-600 rounded-md"
                min="1"
                step="0.1"
              />
            )}
          </div>
          <p className="text-gray-400 mt-2">
            {!currentLine ? 'Haz clic para establecer el primer punto' : 'Haz clic para establecer el segundo punto'}
          </p>
        </div>
      )}

      {/* Controles de dibujo */}
      <div className="bg-gray-800 rounded-lg p-4">
        <div className="flex flex-wrap gap-4">
          <button
            onClick={() => {
              if (!isDrawingLine) {
                setIsDrawingLine(true);
                setIsDrawingZone(false);
                setCurrentLine(null);
              }
            }}
            disabled={isDrawingLine || isDrawingZone}
            className={`flex items-center px-4 py-2 rounded-md ${
              isDrawingLine ? 'bg-green-600' : 'bg-gray-600 hover:bg-gray-700'
            } text-white disabled:opacity-50`}
          >
            <PencilIcon className="h-4 w-4 mr-2" />
            {isDrawingLine ? 'Dibujando Línea...' : 'Dibujar Línea'}
          </button>

          <button
            onClick={() => {
              if (!isDrawingZone) {
                setIsDrawingZone(true);
                setIsDrawingLine(false);
                setCurrentZone([]);
              }
            }}
            disabled={isDrawingLine || isDrawingZone}
            className={`flex items-center px-4 py-2 rounded-md ${
              isDrawingZone ? 'bg-blue-600' : 'bg-gray-600 hover:bg-gray-700'
            } text-white disabled:opacity-50`}
          >
            <PencilIcon className="h-4 w-4 mr-2" />
            {isDrawingZone ? 'Dibujando Zona...' : 'Dibujar Zona'}
          </button>

          {isDrawingZone && currentZone.length >= 3 && (
            <button
              onClick={finishZone}
              className="flex items-center px-4 py-2 bg-purple-600 text-white rounded-md hover:bg-purple-700"
            >
              <CheckIcon className="h-4 w-4 mr-2" />
              Finalizar Zona
            </button>
          )}

          {(isDrawingLine || isDrawingZone) && (
            <button
              onClick={cancelDrawing}
              className="flex items-center px-4 py-2 bg-red-600 text-white rounded-md hover:bg-red-700"
            >
              <XMarkIcon className="h-4 w-4 mr-2" />
              Cancelar
            </button>
          )}

          <button
            onClick={saveConfiguration}
            disabled={lines.length === 0 && zones.length === 0}
            className="flex items-center px-4 py-2 bg-green-600 text-white rounded-md hover:bg-green-700 disabled:opacity-50"
          >
            <CheckIcon className="h-4 w-4 mr-2" />
            Guardar Configuración
          </button>

          <button
            onClick={clearAll}
            disabled={lines.length === 0 && zones.length === 0}
            className="flex items-center px-4 py-2 bg-red-600 text-white rounded-md hover:bg-red-700 disabled:opacity-50"
          >
            <TrashIcon className="h-4 w-4 mr-2" />
            Limpiar Todo
          </button>
        </div>
      </div>

      {/* Stream de video */}
      <div className="bg-gray-800 rounded-lg p-4">
        <div className="relative">
          {isStreamActive && systemStatus.camera ? (
            <div className="relative">
              <img
                ref={imgRef}
                src={streamUrl}
                alt="Camera Stream"
                className="w-full h-auto rounded-lg cursor-crosshair"
                onClick={handleMouseClick}
                onMouseMove={handleMouseMove}
                style={{ maxHeight: '600px', objectFit: 'contain' }}
              />
              
              {/* Overlay SVG para líneas y zonas */}
              {showOverlay && (
                <svg 
                  className="absolute top-0 left-0 w-full h-full pointer-events-none"
                  viewBox="0 0 1280 720"
                  preserveAspectRatio="xMidYMid meet"
                >
                  {/* Líneas guardadas */}
                  {lines.map((line) => (
                    <g key={line.id}>
                      <line
                        x1={line.points[0][0]}
                        y1={line.points[0][1]}
                        x2={line.points[1][0]}
                        y2={line.points[1][1]}
                        stroke={line.line_type === 'counting' ? '#10B981' : '#F59E0B'}
                        strokeWidth="3"
                      />
                      <text
                        x={(line.points[0][0] + line.points[1][0]) / 2}
                        y={(line.points[0][1] + line.points[1][1]) / 2 - 10}
                        fill="#FFFFFF"
                        fontSize="12"
                        textAnchor="middle"
                        className="pointer-events-none"
                      >
                        {line.name}
                      </text>
                    </g>
                  ))}
                  
                  {/* Línea en progreso */}
                  {currentLine && currentLine.end && (
                    <line
                      x1={currentLine.start.x}
                      y1={currentLine.start.y}
                      x2={currentLine.end.x}
                      y2={currentLine.end.y}
                      stroke="#FBBF24"
                      strokeWidth="3"
                      strokeDasharray="5,5"
                    />
                  )}
                  
                  {/* Zonas guardadas */}
                  {zones.map((zone) => (
                    <g key={zone.id}>
                      <polygon
                        points={zone.points.map(p => `${p[0]},${p[1]}`).join(' ')}
                        fill="rgba(239, 68, 68, 0.3)"
                        stroke="#EF4444"
                        strokeWidth="2"
                      />
                      <text
                        x={zone.points.reduce((sum, p) => sum + p[0], 0) / zone.points.length}
                        y={zone.points.reduce((sum, p) => sum + p[1], 0) / zone.points.length}
                        fill="#FFFFFF"
                        fontSize="12"
                        textAnchor="middle"
                        className="pointer-events-none"
                      >
                        {zone.name}
                      </text>
                    </g>
                  ))}
                  
                  {/* Zona en progreso */}
                  {currentZone.length > 0 && (
                    <>
                      <polygon
                        points={currentZone.map(p => `${p.x},${p.y}`).join(' ')}
                        fill="rgba(59, 130, 246, 0.3)"
                        stroke="#3B82F6"
                        strokeWidth="2"
                        strokeDasharray="5,5"
                      />
                      {currentZone.map((point, index) => (
                        <circle
                          key={index}
                          cx={point.x}
                          cy={point.y}
                          r="4"
                          fill="#3B82F6"
                        />
                      ))}
                    </>
                  )}
                </svg>
              )}
            </div>
          ) : (
            <div className="w-full h-96 bg-gray-700 rounded-lg flex items-center justify-center">
              <div className="text-center">
                <CameraIcon className="h-16 w-16 text-gray-400 mx-auto mb-4" />
                <p className="text-gray-400">
                  {systemStatus.camera ? 'Stream no activo' : 'Cámara desconectada'}
                </p>
                <p className="text-gray-500 text-sm mt-2">
                  Configure la cámara en la sección de configuración
                </p>
              </div>
            </div>
          )}
        </div>
      </div>

      {/* Lista de configuraciones */}
      <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
        <div className="bg-gray-800 rounded-lg p-4">
          <h3 className="text-lg font-semibold text-white mb-4">
            Líneas Configuradas ({lines.length})
          </h3>
          <div className="space-y-2 max-h-64 overflow-y-auto">
            {lines.map((line) => (
              <div key={line.id} className="bg-gray-700 p-3 rounded flex justify-between items-center">
                <div>
                  <p className="text-white font-medium">{line.name}</p>
                  <p className="text-gray-400 text-sm">
                    Carril: {line.lane} | Tipo: {line.line_type}
                  </p>
                  {line.distance_to_next && (
                    <p className="text-gray-400 text-sm">
                      Distancia: {line.distance_to_next}m
                    </p>
                  )}
                </div>
                <button
                  onClick={() => setLines(lines.filter(l => l.id !== line.id))}
                  className="text-red-400 hover:text-red-300"
                >
                  <TrashIcon className="h-4 w-4" />
                </button>
              </div>
            ))}
            {lines.length === 0 && (
              <p className="text-gray-400 text-center py-8">
                No hay líneas configuradas
              </p>
            )}
          </div>
        </div>

        <div className="bg-gray-800 rounded-lg p-4">
          <h3 className="text-lg font-semibold text-white mb-4">
            Zonas Configuradas ({zones.length})
          </h3>
          <div className="space-y-2 max-h-64 overflow-y-auto">
            {zones.map((zone) => (
              <div key={zone.id} className="bg-gray-700 p-3 rounded flex justify-between items-center">
                <div>
                  <p className="text-white font-medium">{zone.name}</p>
                  <p className="text-gray-400 text-sm">
                    Tipo: {zone.zone_type} | Puntos: {zone.points.length}
                  </p>
                </div>
                <button
                  onClick={() => setZones(zones.filter(z => z.id !== zone.id))}
                  className="text-red-400 hover:text-red-300"
                >
                  <TrashIcon className="h-4 w-4" />
                </button>
              </div>
            ))}
            {zones.length === 0 && (
              <p className="text-gray-400 text-center py-8">
                No hay zonas configuradas
              </p>
            )}
          </div>
        </div>
      </div>
    </div>
  );
};

export default CameraView;