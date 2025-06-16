import React, { useState, useEffect, useRef } from 'react';
import { 
  PlayIcon, 
  PauseIcon, 
  PencilIcon,
  TrashIcon,
  CheckIcon,
  XMarkIcon,
  CameraIcon,
  ArrowPathIcon
} from '@heroicons/react/24/outline';
import { toast } from 'react-toastify';
import { apiService } from '../../services/api';
import { useSystem } from '../../context/SystemContext';

const CameraView = () => {
  const { systemStatus } = useSystem();
  
  // Estados principales
  const [isStreamActive, setIsStreamActive] = useState(false);
  const [isDrawingLine, setIsDrawingLine] = useState(false);
  const [isDrawingZone, setIsDrawingZone] = useState(false);
  const [lines, setLines] = useState([]);
  const [zones, setZones] = useState([]);
  const [currentLine, setCurrentLine] = useState(null);
  const [currentZone, setCurrentZone] = useState([]);
  const [showOverlay, setShowOverlay] = useState(true);
  
  // Estados de carga y configuración
  const [loading, setLoading] = useState(false);
  const [configLoaded, setConfigLoaded] = useState(false);
  const [streamError, setStreamError] = useState(false);
  
  // Configuración de línea
  const [lineConfig, setLineConfig] = useState({
    name: '',
    lane: '',
    distance: 10.0,
    type: 'counting'
  });

  const imgRef = useRef(null);
  const streamUrl = '/api/camera/stream';

  // Cargar configuración existente al inicializar
  useEffect(() => {
    loadAnalysisConfig();
  }, []);

  // Activar stream si hay cámara
  useEffect(() => {
    if (systemStatus.camera) {
      setIsStreamActive(true);
      setStreamError(false);
    }
  }, [systemStatus.camera]);

  const loadAnalysisConfig = async () => {
    if (configLoaded) return; // Evitar carga múltiple
    
    setLoading(true);
    try {
      console.log('🔄 Cargando configuración de análisis...');
      
      // Cargar líneas existentes
      const linesResponse = await apiService.getLines();
      if (linesResponse && linesResponse.lines) {
        const loadedLines = Object.values(linesResponse.lines).map(line => ({
          id: line.id,
          name: line.name,
          points: line.points, // Ya están en formato [[x, y], [x, y]]
          lane: line.lane,
          line_type: line.line_type,
          distance_to_next: line.distance_to_next
        }));
        setLines(loadedLines);
        console.log(`✅ ${loadedLines.length} líneas cargadas`);
      }
      
      // Cargar zonas existentes
      const zonesResponse = await apiService.getZones();
      if (zonesResponse && zonesResponse.zones) {
        const loadedZones = Object.values(zonesResponse.zones).map(zone => ({
          id: zone.id,
          name: zone.name,
          points: zone.points, // Ya están en formato [[x, y], [x, y], ...]
          zone_type: zone.zone_type
        }));
        setZones(loadedZones);
        console.log(`✅ ${loadedZones.length} zonas cargadas`);
      }
      
      setConfigLoaded(true);
      
    } catch (error) {
      console.error('❌ Error cargando configuración de análisis:', error);
      toast.error('Error cargando configuración existente');
    } finally {
      setLoading(false);
    }
  };

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
    if (lines.length === 0 && zones.length === 0) {
      toast.error('No hay líneas o zonas para guardar');
      return;
    }
    
    setLoading(true);
    try {
      let savedLines = 0;
      let savedZones = 0;
      
      // Guardar líneas
      for (const line of lines) {
        const lineData = {
          id: line.id,
          name: line.name,
          points: line.points, // Ya están en formato correcto [[x, y], [x, y]]
          lane: line.lane,
          line_type: line.line_type,
          distance_to_next: line.distance_to_next
        };
        
        await apiService.addLine(lineData);
        savedLines++;
      }
      
      // Guardar zonas
      for (const zone of zones) {
        const zoneData = {
          id: zone.id,
          name: zone.name,
          points: zone.points, // Ya están en formato correcto [[x, y], [x, y], ...]
          zone_type: zone.zone_type
        };
        
        await apiService.addZone(zoneData);
        savedZones++;
      }
      
      toast.success(`Configuración guardada: ${savedLines} líneas, ${savedZones} zonas`);
      
      // Limpiar configuración temporal
      setLines([]);
      setZones([]);
      setCurrentLine(null);
      setCurrentZone([]);
      
      // Recargar configuración guardada
      setTimeout(() => {
        setConfigLoaded(false);
        loadAnalysisConfig();
      }, 1000);
      
    } catch (error) {
      console.error('Error guardando configuración:', error);
      toast.error('Error guardando configuración');
    } finally {
      setLoading(false);
    }
  };

  const deleteLineById = async (lineId) => {
    try {
      await apiService.deleteLine(lineId);
      setLines(lines.filter(l => l.id !== lineId));
      toast.success('Línea eliminada');
      
      // Recargar configuración
      setTimeout(() => {
        setConfigLoaded(false);
        loadAnalysisConfig();
      }, 500);
    } catch (error) {
      console.error('Error eliminando línea:', error);
      toast.error('Error eliminando línea');
    }
  };

  const deleteZoneById = async (zoneId) => {
    try {
      await apiService.deleteZone(zoneId);
      setZones(zones.filter(z => z.id !== zoneId));
      toast.success('Zona eliminada');
      
      // Recargar configuración
      setTimeout(() => {
        setConfigLoaded(false);
        loadAnalysisConfig();
      }, 500);
    } catch (error) {
      console.error('Error eliminando zona:', error);
      toast.error('Error eliminando zona');
    }
  };

  const clearAll = () => {
    if (window.confirm('¿Está seguro de eliminar TODA la configuración temporal?')) {
      setLines([]);
      setZones([]);
      setCurrentLine(null);
      setCurrentZone([]);
      toast.info('Configuración temporal limpiada');
    }
  };

  const clearAllSaved = async () => {
    if (window.confirm('¿Está seguro de eliminar TODA la configuración guardada? Esta acción no se puede deshacer.')) {
      try {
        await apiService.clearAnalysis();
        setLines([]);
        setZones([]);
        toast.success('Toda la configuración eliminada');
        
        // Recargar configuración
        setTimeout(() => {
          setConfigLoaded(false);
          loadAnalysisConfig();
        }, 500);
      } catch (error) {
        console.error('Error limpiando configuración:', error);
        toast.error('Error limpiando configuración');
      }
    }
  };

  const reloadConfiguration = async () => {
    setConfigLoaded(false);
    await loadAnalysisConfig();
    toast.success('Configuración recargada');
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
            onClick={reloadConfiguration}
            disabled={loading}
            className="flex items-center px-4 py-2 bg-blue-600 text-white rounded-md hover:bg-blue-700 disabled:opacity-50"
          >
            <ArrowPathIcon className="h-5 w-5 mr-2" />
            Recargar
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
            disabled={lines.length === 0 && zones.length === 0 || loading}
            className="flex items-center px-4 py-2 bg-green-600 text-white rounded-md hover:bg-green-700 disabled:opacity-50"
          >
            <CheckIcon className="h-4 w-4 mr-2" />
            {loading ? 'Guardando...' : 'Guardar Configuración'}
          </button>

          <button
            onClick={clearAll}
            disabled={lines.length === 0 && zones.length === 0}
            className="flex items-center px-4 py-2 bg-yellow-600 text-white rounded-md hover:bg-yellow-700 disabled:opacity-50"
          >
            <TrashIcon className="h-4 w-4 mr-2" />
            Limpiar Temporal
          </button>

          <button
            onClick={clearAllSaved}
            className="flex items-center px-4 py-2 bg-red-600 text-white rounded-md hover:bg-red-700"
          >
            <TrashIcon className="h-4 w-4 mr-2" />
            Limpiar Todo
          </button>
        </div>
      </div>

      {/* Stream de video */}
      <div className="bg-gray-800 rounded-lg p-4">
        <div className="relative">
          {isStreamActive && systemStatus.camera && !streamError ? (
            <div className="relative">
              <img
                ref={imgRef}
                src={streamUrl}
                alt="Camera Stream"
                className="w-full h-auto rounded-lg cursor-crosshair"
                onClick={handleMouseClick}
                onMouseMove={handleMouseMove}
                onError={() => setStreamError(true)}
                style={{ maxHeight: '600px', objectFit: 'contain' }}
              />
              
              {/* Overlay SVG para líneas y zonas */}
              {showOverlay && (
                <svg 
                  className="absolute top-0 left-0 w-full h-full pointer-events-none"
                  style={{ maxHeight: '600px' }}
                  preserveAspectRatio="none"
                  viewBox="0 0 1280 720"
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
                  {streamError ? 'Error en stream de cámara' :
                   systemStatus.camera ? 'Stream no activo' : 'Cámara desconectada'}
                </p>
                <p className="text-gray-500 text-sm mt-2">
                  {streamError ? 'Verifique la configuración RTSP' :
                   'Configure la cámara en la sección de configuración'}
                </p>
              </div>
            </div>
          )}
        </div>
      </div>

      {/* Lista de configuraciones */}
      <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
        <div className="bg-gray-800 rounded-lg p-4">
          <div className="flex justify-between items-center mb-4">
            <h3 className="text-lg font-semibold text-white">
              Líneas Configuradas ({lines.length})
            </h3>
            {lines.length > 0 && (
              <button
                onClick={clearAllSaved}
                className="text-red-400 hover:text-red-300 text-sm"
                disabled={loading}
              >
                <TrashIcon className="h-4 w-4" />
              </button>
            )}
          </div>
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
                  onClick={() => deleteLineById(line.id)}
                  className="text-red-400 hover:text-red-300"
                  disabled={loading}
                >
                  <TrashIcon className="h-4 w-4" />
                </button>
              </div>
            ))}
            {lines.length === 0 && (
              <p className="text-gray-400 text-center py-8">
                {loading ? 'Cargando líneas...' : 'No hay líneas configuradas'}
              </p>
            )}
          </div>
        </div>

        <div className="bg-gray-800 rounded-lg p-4">
          <div className="flex justify-between items-center mb-4">
            <h3 className="text-lg font-semibold text-white">
              Zonas Configuradas ({zones.length})
            </h3>
          </div>
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
                  onClick={() => deleteZoneById(zone.id)}
                  className="text-red-400 hover:text-red-300"
                  disabled={loading}
                >
                  <TrashIcon className="h-4 w-4" />
                </button>
              </div>
            ))}
            {zones.length === 0 && (
              <p className="text-gray-400 text-center py-8">
                {loading ? 'Cargando zonas...' : 'No hay zonas configuradas'}
              </p>
            )}
          </div>
        </div>
      </div>

      {/* Estado de configuración */}
      <div className="bg-gray-800 rounded-lg p-4">
        <h3 className="text-lg font-semibold text-white mb-4">Estado de la Configuración</h3>
        <div className="grid grid-cols-1 md:grid-cols-3 gap-4 text-gray-300">
          <div className="bg-gray-700 p-3 rounded">
            <p className="text-xs text-gray-400">Líneas Temporales</p>
            <p className="font-medium text-white">{lines.filter(l => !l.saved).length || 0}</p>
          </div>
          <div className="bg-gray-700 p-3 rounded">
            <p className="text-xs text-gray-400">Zonas Temporales</p>
            <p className="font-medium text-white">{zones.filter(z => !z.saved).length || 0}</p>
          </div>
          <div className="bg-gray-700 p-3 rounded">
            <p className="text-xs text-gray-400">Estado</p>
            <p className="font-medium text-white">
              {loading ? 'Cargando...' : configLoaded ? 'Configurado' : 'Sin configurar'}
            </p>
          </div>
        </div>
      </div>
    </div>
  );
};

export default CameraView;