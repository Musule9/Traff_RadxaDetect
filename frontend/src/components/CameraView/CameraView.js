import React, { useState, useEffect, useRef } from 'react';
import { PlayIcon, PauseIcon, Cog6ToothIcon } from '@heroicons/react/24/outline';
import toast from 'react-hot-toast';

const CameraView = () => {
  const [isStreamActive, setIsStreamActive] = useState(false);
  const [isDrawingLine, setIsDrawingLine] = useState(false);
  const [isDrawingZone, setIsDrawingZone] = useState(false);
  const [lines, setLines] = useState([]);
  const [zones, setZones] = useState([]);
  const [currentLine, setCurrentLine] = useState(null);
  const [currentZone, setCurrentZone] = useState([]);
  const imgRef = useRef(null);
  const canvasRef = useRef(null);

  const handleMouseDown = (e) => {
    if (!isDrawingLine && !isDrawingZone) return;

    const rect = e.target.getBoundingClientRect();
    const x = e.clientX - rect.left;
    const y = e.clientY - rect.top;

    if (isDrawingLine) {
      if (!currentLine) {
        setCurrentLine({ start: { x, y }, end: null });
      } else {
        const newLine = {
          id: `line_${Date.now()}`,
          name: `Línea ${lines.length + 1}`,
          points: [[currentLine.start.x, currentLine.start.y], [x, y]],
          lane: `carril_${lines.length + 1}`,
          line_type: 'counting',
          distance_to_next: 10.0
        };
        setLines([...lines, newLine]);
        setCurrentLine(null);
        setIsDrawingLine(false);
        toast.success('Línea agregada');
      }
    } else if (isDrawingZone) {
      setCurrentZone([...currentZone, { x, y }]);
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

  const saveConfiguration = async () => {
    try {
      for (const line of lines) {
        await api.post('/api/analysis/lines', line);
      }
      for (const zone of zones) {
        await api.post('/api/analysis/zones', zone);
      }
      toast.success('Configuración guardada');
    } catch (error) {
      toast.error('Error guardando configuración');
    }
  };

  return (
    <div className="space-y-6">
      <div className="flex justify-between items-center">
        <h1 className="text-3xl font-bold text-white">Vista de Cámara</h1>
        <div className="flex space-x-4">
          <button
            onClick={() => setIsStreamActive(!isStreamActive)}
            className="flex items-center px-4 py-2 bg-blue-600 text-white rounded-md hover:bg-blue-700"
          >
            {isStreamActive ? <PauseIcon className="h-5 w-5 mr-2" /> : <PlayIcon className="h-5 w-5 mr-2" />}
            {isStreamActive ? 'Pausar' : 'Iniciar'} Stream
          </button>
        </div>
      </div>

      {/* Controles de dibujo */}
      <div className="bg-gray-800 rounded-lg p-4">
        <div className="flex space-x-4">
          <button
            onClick={() => {
              setIsDrawingLine(true);
              setIsDrawingZone(false);
              setCurrentLine(null);
            }}
            className={`px-4 py-2 rounded-md ${isDrawingLine ? 'bg-green-600' : 'bg-gray-600'} text-white`}
          >
            Dibujar Línea
          </button>
          <button
            onClick={() => {
              setIsDrawingZone(true);
              setIsDrawingLine(false);
              setCurrentZone([]);
            }}
            className={`px-4 py-2 rounded-md ${isDrawingZone ? 'bg-blue-600' : 'bg-gray-600'} text-white`}
          >
            Dibujar Zona
          </button>
          {isDrawingZone && currentZone.length >= 3 && (
            <button
              onClick={finishZone}
              className="px-4 py-2 bg-purple-600 text-white rounded-md"
            >
              Finalizar Zona
            </button>
          )}
          <button
            onClick={saveConfiguration}
            className="px-4 py-2 bg-green-600 text-white rounded-md"
          >
            Guardar Configuración
          </button>
        </div>
      </div>

      {/* Stream de video */}
      <div className="bg-gray-800 rounded-lg p-4">
        <div className="relative">
          {isStreamActive ? (
            <img
              ref={imgRef}
              src="/api/camera/stream"
              alt="Camera Stream"
              className="w-full h-auto rounded-lg cursor-crosshair"
              onMouseDown={handleMouseDown}
            />
          ) : (
            <div className="w-full h-96 bg-gray-700 rounded-lg flex items-center justify-center">
              <p className="text-gray-400">Stream no activo</p>
            </div>
          )}
          
          {/* Overlay para líneas y zonas */}
          <svg className="absolute top-0 left-0 w-full h-full pointer-events-none">
            {/* Líneas existentes */}
            {lines.map((line, index) => (
              <line
                key={line.id}
                x1={line.points[0][0]}
                y1={line.points[0][1]}
                x2={line.points[1][0]}
                y2={line.points[1][1]}
                stroke="lime"
                strokeWidth="3"
              />
            ))}
            
            {/* Línea en progreso */}
            {currentLine && currentLine.end && (
              <line
                x1={currentLine.start.x}
                y1={currentLine.start.y}
                x2={currentLine.end.x}
                y2={currentLine.end.y}
                stroke="yellow"
                strokeWidth="3"
                strokeDasharray="5,5"
              />
            )}
            
            {/* Zonas existentes */}
            {zones.map((zone, index) => (
              <polygon
                key={zone.id}
                points={zone.points.map(p => `${p[0]},${p[1]}`).join(' ')}
                fill="rgba(255,0,0,0.3)"
                stroke="red"
                strokeWidth="2"
              />
            ))}
            
            {/* Zona en progreso */}
            {currentZone.length > 0 && (
              <polygon
                points={currentZone.map(p => `${p.x},${p.y}`).join(' ')}
                fill="rgba(0,0,255,0.3)"
                stroke="blue"
                strokeWidth="2"
                strokeDasharray="5,5"
              />
            )}
          </svg>
        </div>
      </div>

      {/* Lista de configuraciones */}
      <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
        <div className="bg-gray-800 rounded-lg p-4">
          <h3 className="text-lg font-semibold text-white mb-4">Líneas Configuradas</h3>
          <div className="space-y-2">
            {lines.map((line, index) => (
              <div key={line.id} className="bg-gray-700 p-3 rounded">
                <p className="text-white font-medium">{line.name}</p>
                <p className="text-gray-400 text-sm">Carril: {line.lane}</p>
                <p className="text-gray-400 text-sm">Distancia: {line.distance_to_next}m</p>
              </div>
            ))}
            {lines.length === 0 && (
              <p className="text-gray-400">No hay líneas configuradas</p>
            )}
          </div>
        </div>

        <div className="bg-gray-800 rounded-lg p-4">
          <h3 className="text-lg font-semibold text-white mb-4">Zonas Configuradas</h3>
          <div className="space-y-2">
            {zones.map((zone, index) => (
              <div key={zone.id} className="bg-gray-700 p-3 rounded">
                <p className="text-white font-medium">{zone.name}</p>
                <p className="text-gray-400 text-sm">Tipo: {zone.zone_type}</p>
                <p className="text-gray-400 text-sm">Puntos: {zone.points.length}</p>
              </div>
            ))}
            {zones.length === 0 && (
              <p className="text-gray-400">No hay zonas configuradas</p>
            )}
          </div>
        </div>
      </div>
    </div>
  );
};

export default CameraView;