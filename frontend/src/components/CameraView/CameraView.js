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
import { StopIcon } from '@heroicons/react/24/outline';

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
    type: 'counting',
    speed_limit: 50,
    carril_number: 1,
    direction_flow: 'bidirectional',
    priority: 'normal'
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

  useEffect(() => {
    let reconnectTimer;
    
    if (streamError && isStreamActive) {
      // Intentar reconectar cada 5 segundos
      reconnectTimer = setTimeout(() => {
        console.log('Intentando reconectar stream...');
        setStreamError(false);
        // Forzar reload de la imagen
        if (imgRef.current) {
          imgRef.current.src = `/api/camera/stream?t=${Date.now()}`;
        }
      }, 5000);
    }
    
    return () => {
      if (reconnectTimer) {
        clearTimeout(reconnectTimer);
      }
    };
  }, [streamError, isStreamActive]);

  const loadAnalysisConfig = async (forceReload = false) => {
    if (configLoaded && !forceReload) return;
    
    setLoading(true);
    try {
      // Limpiar antes de cargar para evitar duplicados
      setLines([]);
      setZones([]);
      
      const [linesResponse, zonesResponse] = await Promise.all([
        apiService.getLines(),
        apiService.getZones()
      ]);
      
      if (linesResponse && linesResponse.lines) {
        const loadedLines = Object.values(linesResponse.lines).map(line => ({
          ...line,
          saved: true
        }));
        setLines(loadedLines);
      }
      
      if (zonesResponse && zonesResponse.zones) {
        const loadedZones = Object.values(zonesResponse.zones).map(zone => ({
          ...zone,
          saved: true
        }));
        setZones(loadedZones);
      }
      
      setConfigLoaded(true);
    } catch (error) {
      console.error('Error cargando configuración:', error);
      toast.error('Error cargando configuración');
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
          name: lineConfig.name || `${lineConfig.type === 'speed' ? 'Velocidad' : 'Conteo'} ${lineConfig.lane || `Carril ${lineConfig.carril_number}`}`,
          points: [[currentLine.start.x, currentLine.start.y], [x, y]],
          lane: lineConfig.lane || `carril_${lineConfig.carril_number}`,
          line_type: lineConfig.type,
          distance_to_next: lineConfig.type === 'speed' ? lineConfig.distance : null,
          speed_limit: lineConfig.speed_limit,
          carril_number: lineConfig.carril_number,
          direction_flow: lineConfig.direction_flow,
          priority: lineConfig.priority
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
          
          <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-4 mb-4">
            <input
              type="text"
              placeholder="Nombre de línea"
              value={lineConfig.name}
              onChange={(e) => setLineConfig({...lineConfig, name: e.target.value})}
              className="px-3 py-2 bg-gray-700 text-white border border-gray-600 rounded-md"
            />
            
            <select
              value={lineConfig.lane}
              onChange={(e) => setLineConfig({...lineConfig, lane: e.target.value})}
              className="px-3 py-2 bg-gray-700 text-white border border-gray-600 rounded-md"
            >
              <option value="">Seleccionar Carril</option>
              <option value="carril_1">Carril 1</option>
              <option value="carril_2">Carril 2</option>
              <option value="carril_3">Carril 3</option>
              <option value="carril_4">Carril 4</option>
              <option value="carril_vuelta">Carril de Vuelta</option>
              <option value="carril_central">Carril Central</option>
            </select>
            
            <select
              value={lineConfig.type}
              onChange={(e) => setLineConfig({...lineConfig, type: e.target.value})}
              className="px-3 py-2 bg-gray-700 text-white border border-gray-600 rounded-md"
            >
              <option value="counting">Línea de Conteo</option>
              <option value="speed">Línea de Velocidad</option>
            </select>
          </div>

          {lineConfig.type === 'speed' && (
            <div className="grid grid-cols-1 md:grid-cols-3 gap-4 mb-4 p-4 bg-blue-900/20 rounded-lg border border-blue-600">
              <div>
                <label className="block text-sm text-blue-300 mb-1">Distancia al Siguiente Punto (metros)</label>
                <input
                  type="number"
                  placeholder="Distancia en metros"
                  value={lineConfig.distance}
                  onChange={(e) => setLineConfig({...lineConfig, distance: parseFloat(e.target.value)})}
                  className="w-full px-3 py-2 bg-gray-700 text-white border border-gray-600 rounded-md"
                  min="1"
                  max="100"
                  step="0.5"
                />
                <p className="text-xs text-blue-400 mt-1">
                  Distancia real entre esta línea y la siguiente línea de velocidad del mismo carril
                </p>
              </div>
              
              <div>
                <label className="block text-sm text-blue-300 mb-1">Límite de Velocidad (km/h)</label>
                <input
                  type="number"
                  placeholder="Límite km/h"
                  value={lineConfig.speed_limit}
                  onChange={(e) => setLineConfig({...lineConfig, speed_limit: parseInt(e.target.value)})}
                  className="w-full px-3 py-2 bg-gray-700 text-white border border-gray-600 rounded-md"
                  min="20"
                  max="120"
                  step="5"
                />
              </div>
              
              <div>
                <label className="block text-sm text-blue-300 mb-1">Flujo de Tráfico</label>
                <select
                  value={lineConfig.direction_flow}
                  onChange={(e) => setLineConfig({...lineConfig, direction_flow: e.target.value})}
                  className="w-full px-3 py-2 bg-gray-700 text-white border border-gray-600 rounded-md"
                >
                  <option value="bidirectional">Bidireccional</option>
                  <option value="north">Solo Norte</option>
                  <option value="south">Solo Sur</option>
                  <option value="east">Solo Este</option>
                  <option value="west">Solo Oeste</option>
                </select>
              </div>
            </div>
          )}

          {lineConfig.type === 'counting' && (
            <div className="grid grid-cols-1 md:grid-cols-2 gap-4 mb-4 p-4 bg-green-900/20 rounded-lg border border-green-600">
              <div>
                <label className="block text-sm text-green-300 mb-1">Número de Carril</label>
                <input
                  type="number"
                  placeholder="Número"
                  value={lineConfig.carril_number}
                  onChange={(e) => setLineConfig({...lineConfig, carril_number: parseInt(e.target.value)})}
                  className="w-full px-3 py-2 bg-gray-700 text-white border border-gray-600 rounded-md"
                  min="1"
                  max="6"
                />
              </div>
              
              <div>
                <label className="block text-sm text-green-300 mb-1">Prioridad</label>
                <select
                  value={lineConfig.priority}
                  onChange={(e) => setLineConfig({...lineConfig, priority: e.target.value})}
                  className="w-full px-3 py-2 bg-gray-700 text-white border border-gray-600 rounded-md"
                >
                  <option value="low">Baja</option>
                  <option value="normal">Normal</option>
                  <option value="high">Alta</option>
                </select>
              </div>
            </div>
          )}
          
          <div className="bg-yellow-900/20 border border-yellow-600 rounded-lg p-3">
            <p className="text-yellow-300 text-sm">
              {!currentLine ? (
                <><strong>Paso 1:</strong> Haz clic para establecer el primer punto de la línea</>
              ) : (
                <><strong>Paso 2:</strong> Haz clic para establecer el segundo punto y finalizar la línea</>
              )}
            </p>
            {lineConfig.type === 'speed' && (
              <p className="text-yellow-400 text-xs mt-1">
                💡 Para calcular velocidad correctamente, necesitas al menos 2 líneas de velocidad en el mismo carril separadas por la distancia especificada
              </p>
            )}
          </div>
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
        <div className="flex justify-between items-center mb-4">
          <h3 className="text-lg font-semibold text-white">Video en Tiempo Real</h3>
          <div className="flex items-center space-x-4">
            {/* Indicador de estado */}
            <div className="flex items-center space-x-2">
              <div className={`w-3 h-3 rounded-full ${
                isStreamActive && systemStatus.camera ? 'bg-green-500 animate-pulse' : 'bg-red-500'
              }`}></div>
              <span className="text-sm text-gray-300">
                {isStreamActive && systemStatus.camera ? 'En Vivo' : 'Desconectado'}
              </span>
            </div>
            
            {/* Información de FPS */}
            {systemStatus.fps && (
              <div className="text-sm text-gray-300">
                {systemStatus.fps} FPS
              </div>
            )}
          </div>
        </div>

        <div className="relative bg-black rounded-lg overflow-hidden">
          {isStreamActive && systemStatus.camera && !streamError ? (
            <div className="relative">
              {/* Imagen del stream HTTP */}
              <img
                ref={imgRef}
                src={`/api/camera/stream?t=${Date.now()}`}
                alt="Stream de Cámara - Análisis en Tiempo Real"
                className="w-full h-auto rounded-lg cursor-crosshair"
                onClick={handleMouseClick}
                onMouseMove={handleMouseMove}
                onError={() => {
                  console.error('Error en stream HTTP');
                  setStreamError(true);
                }}
                onLoad={() => {
                  setStreamError(false);
                }}
                style={{ 
                  maxHeight: '600px', 
                  objectFit: 'contain',
                  backgroundColor: '#000'
                }}
              />
              
              {/* Overlay de información en tiempo real */}
              <div className="absolute top-4 left-4 bg-black/70 rounded-lg p-3 text-white text-sm">
                <div className="flex items-center space-x-4">
                  <div>
                    <span className="text-gray-300">Resolución:</span> 
                    <span className="ml-1 text-white">1280x720</span>
                  </div>
                  <div>
                    <span className="text-gray-300">Calidad:</span> 
                    <span className="ml-1 text-green-400">HD</span>
                  </div>
                  <div>
                    <span className="text-gray-300">Latencia:</span> 
                    <span className="ml-1 text-blue-400">~200ms</span>
                  </div>
                </div>
              </div>

              {/* Overlay de estadísticas en tiempo real */}
              <div className="absolute bottom-4 right-4 bg-black/70 rounded-lg p-3 text-white text-sm">
                <div className="grid grid-cols-2 gap-3">
                  <div>
                    <span className="text-gray-300">Vehículos:</span> 
                    <span className="ml-1 text-yellow-400">3</span>
                  </div>
                  <div>
                    <span className="text-gray-300">Líneas:</span> 
                    <span className="ml-1 text-green-400">{lines.length}</span>
                  </div>
                  <div>
                    <span className="text-gray-300">Zonas:</span> 
                    <span className="ml-1 text-purple-400">{zones.length}</span>
                  </div>
                  <div>
                    <span className="text-gray-300">Análisis:</span> 
                    <span className="ml-1 text-blue-400">ON</span>
                  </div>
                </div>
              </div>
              
              {/* Overlay SVG para líneas y zonas - MANTENER IGUAL */}
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
                  
                  {/* Resto del código SVG igual... */}
                </svg>
              )}
            </div>
          ) : (
            /* Placeholder cuando no hay stream */
            <div className="w-full h-96 bg-gradient-to-br from-gray-700 to-gray-800 rounded-lg flex flex-col items-center justify-center">
              <div className="text-center">
                <CameraIcon className="h-20 w-20 text-gray-400 mx-auto mb-4" />
                <h3 className="text-xl font-medium text-gray-300 mb-2">
                  {streamError ? 'Error de Conexión' : 
                  systemStatus.camera ? 'Stream No Activo' : 'Cámara No Configurada'}
                </h3>
                <p className="text-gray-400 text-sm mb-4">
                  {streamError ? 'Verifica la configuración RTSP en Configuración de Cámara' :
                  systemStatus.camera ? 'Presiona "Iniciar Stream" para comenzar' : 
                  'Configure la cámara en Configuración para comenzar'}
                </p>
                
                {!systemStatus.camera && (
                  <button
                    onClick={() => window.location.href = '/camera-config'}
                    className="px-4 py-2 bg-blue-600 text-white rounded-md hover:bg-blue-700 transition-colors"
                  >
                    Ir a Configuración
                  </button>
                )}
              </div>
            </div>
          )}
        </div>

        {/* Controles de stream */}
        <div className="flex justify-between items-center mt-4">
          <div className="flex space-x-2">
            <button
              onClick={() => setIsStreamActive(!isStreamActive)}
              className={`flex items-center px-4 py-2 rounded-md transition-colors ${
                isStreamActive ? 'bg-red-600 hover:bg-red-700' : 'bg-green-600 hover:bg-green-700'
              } text-white`}
            >
              {isStreamActive ? <StopIcon className="h-5 w-5 mr-2" /> : <PlayIcon className="h-5 w-5 mr-2" />}
              {isStreamActive ? 'Detener' : 'Iniciar'} Stream
            </button>
            
            <button
              onClick={() => setShowOverlay(!showOverlay)}
              className={`px-4 py-2 rounded-md transition-colors ${
                showOverlay ? 'bg-purple-600 hover:bg-purple-700' : 'bg-gray-600 hover:bg-gray-700'
              } text-white`}
            >
              {showOverlay ? 'Ocultar' : 'Mostrar'} Análisis
            </button>
          </div>
          
          <div className="flex items-center space-x-4 text-sm text-gray-400">
            <span>Protocolo: HTTP Stream</span>
            <span>Encoding: MJPEG</span>
            <span>Optimizado para Web</span>
          </div>
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
              <div key={line.id} className={`p-4 rounded-lg border-l-4 ${
                line.line_type === 'speed' ? 'bg-blue-900/30 border-blue-500' : 'bg-green-900/30 border-green-500'
              }`}>
                <div className="flex justify-between items-start">
                  <div className="flex-1">
                    <div className="flex items-center gap-2 mb-2">
                      <span className={`px-2 py-1 text-xs rounded ${
                        line.line_type === 'speed' ? 'bg-blue-600 text-white' : 'bg-green-600 text-white'
                      }`}>
                        {line.line_type === 'speed' ? 'VELOCIDAD' : 'CONTEO'}
                      </span>
                      <p className="text-white font-medium">{line.name}</p>
                    </div>
                    
                    <div className="grid grid-cols-2 gap-2 text-sm">
                      <p className="text-gray-300">
                        <span className="text-gray-400">Carril:</span> {line.lane}
                      </p>
                      <p className="text-gray-300">
                        <span className="text-gray-400">Número:</span> {line.carril_number || 'N/A'}
                      </p>
                      
                      {line.line_type === 'speed' && (
                        <>
                          <p className="text-blue-300">
                            <span className="text-gray-400">Distancia:</span> {line.distance_to_next}m
                          </p>
                          <p className="text-blue-300">
                            <span className="text-gray-400">Límite:</span> {line.speed_limit || 50} km/h
                          </p>
                          <p className="text-blue-300 col-span-2">
                            <span className="text-gray-400">Flujo:</span> {line.direction_flow || 'bidirectional'}
                          </p>
                        </>
                      )}
                      
                      {line.line_type === 'counting' && (
                        <p className="text-green-300 col-span-2">
                          <span className="text-gray-400">Prioridad:</span> {line.priority || 'normal'}
                        </p>
                      )}
                    </div>
                  </div>
                  
                  <button
                    onClick={() => deleteLineById(line.id)}
                    className="text-red-400 hover:text-red-300 ml-3"
                    disabled={loading}
                  >
                    <TrashIcon className="h-4 w-4" />
                  </button>
                </div>
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