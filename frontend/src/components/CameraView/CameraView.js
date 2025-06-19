import React, { useState, useEffect, useRef } from 'react';
import { 
  PlayIcon, 
  PauseIcon, 
  PencilIcon,
  TrashIcon,
  CheckIcon,
  XMarkIcon,
  CameraIcon,
  ArrowPathIcon,
  ExclamationTriangleIcon
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
  const [streamError, setStreamError] = useState(false);
  const [cameraConfigured, setCameraConfigured] = useState(false);
  
  // Configuración de línea simplificada
  const [lineConfig, setLineConfig] = useState({
    name: '',
    lane: 'carril_1',
    type: 'counting',
    distance: 10.0
  });

  const imgRef = useRef(null);
  const [streamUrl, setStreamUrl] = useState('');

  // Cargar configuración al inicializar
  useEffect(() => {
    checkCameraConfiguration();
    loadAnalysisConfig();
  }, []);

  // Actualizar stream basado en estado de sistema
  useEffect(() => {
    if (systemStatus.camera) {
      setCameraConfigured(true);
      setIsStreamActive(true);
      setStreamError(false);
      updateStreamUrl();
    } else {
      setCameraConfigured(false);
      // No apagar stream automáticamente - dejar que usuario controle
    }
  }, [systemStatus.camera]);

  // Manejar errores de stream con reintento automático
  useEffect(() => {
    let reconnectTimer;
    
    if (streamError && isStreamActive && cameraConfigured) {
      console.log('🔄 Stream error detectado, reintentando en 5 segundos...');
      reconnectTimer = setTimeout(() => {
        console.log('🔄 Reintentando conexión de stream...');
        setStreamError(false);
        updateStreamUrl();
      }, 5000);
    }
    
    return () => {
      if (reconnectTimer) {
        clearTimeout(reconnectTimer);
      }
    };
  }, [streamError, isStreamActive, cameraConfigured]);

  const checkCameraConfiguration = async () => {
    try {
      const config = await apiService.getCameraConfig();
      const hasValidRtsp = config.rtsp_url && config.rtsp_url.trim().length > 0;
      setCameraConfigured(hasValidRtsp);
      
      if (hasValidRtsp) {
        console.log('✅ Cámara configurada, stream disponible');
      } else {
        console.log('⚠️ Cámara no configurada');
      }
    } catch (error) {
      console.error('❌ Error verificando configuración:', error);
      setCameraConfigured(false);
    }
  };

  const updateStreamUrl = () => {
    const newUrl = `/api/camera/stream?t=${Date.now()}`;
    setStreamUrl(newUrl);
    console.log('🔄 Stream URL actualizada:', newUrl);
  };

  const loadAnalysisConfig = async () => {
    setLoading(true);
    try {
      console.log('📥 Cargando configuración de análisis...');
      
      // Limpiar estados anteriores
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
        console.log(`✅ ${loadedLines.length} líneas cargadas`);
      }
      
      if (zonesResponse && zonesResponse.zones) {
        const loadedZones = Object.values(zonesResponse.zones).map(zone => ({
          ...zone,
          saved: true
        }));
        setZones(loadedZones);
        console.log(`✅ ${loadedZones.length} zonas cargadas`);
      }
      
    } catch (error) {
      console.error('❌ Error cargando configuración de análisis:', error);
      toast.error('Error cargando configuración de análisis');
    } finally {
      setLoading(false);
    }
  };

  const handleMouseClick = (e) => {
    if (!isDrawingLine && !isDrawingZone) return;

    const rect = e.target.getBoundingClientRect();
    const imgElement = imgRef.current;
    if (!imgElement) return;
    
    // Calcular coordenadas reales basadas en el tamaño de la imagen
    const scaleX = imgElement.naturalWidth / imgElement.clientWidth;
    const scaleY = imgElement.naturalHeight / imgElement.clientHeight;
    
    const x = Math.round((e.clientX - rect.left) * scaleX);
    const y = Math.round((e.clientY - rect.top) * scaleY);

    console.log(`🎯 Clic en: (${x}, ${y}) - Escala: ${scaleX.toFixed(2)} x ${scaleY.toFixed(2)}`);

    if (isDrawingLine) {
      if (!currentLine) {
        setCurrentLine({ start: { x, y }, end: null });
        console.log('📍 Primer punto de línea establecido');
      } else {
        // Completar línea
        const newLine = {
          id: `line_${Date.now()}`,
          name: lineConfig.name || `${lineConfig.type === 'speed' ? 'Velocidad' : 'Conteo'} ${lineConfig.lane}`,
          points: [[currentLine.start.x, currentLine.start.y], [x, y]],
          lane: lineConfig.lane,
          line_type: lineConfig.type,
          distance_to_next: lineConfig.type === 'speed' ? parseFloat(lineConfig.distance) : null,
          enabled: true,
          saved: false // Marcar como no guardado
        };
        
        setLines(prev => [...prev, newLine]);
        setCurrentLine(null);
        setIsDrawingLine(false);
        toast.success(`✅ Línea "${newLine.name}" creada`);
        
        console.log('✅ Línea creada:', newLine);
        
        // Reset form
        setLineConfig({
          name: '',
          lane: 'carril_1',
          type: 'counting',
          distance: 10.0
        });
      }
    } else if (isDrawingZone) {
      setCurrentZone([...currentZone, { x, y }]);
      console.log(`📍 Punto de zona agregado: (${x}, ${y}) - Total: ${currentZone.length + 1}`);
    }
  };

  const handleMouseMove = (e) => {
    if (isDrawingLine && currentLine && !currentLine.end) {
      const rect = e.target.getBoundingClientRect();
      const imgElement = imgRef.current;
      
      if (imgElement) {
        const scaleX = imgElement.naturalWidth / imgElement.clientWidth;
        const scaleY = imgElement.naturalHeight / imgElement.clientHeight;
        
        const x = Math.round((e.clientX - rect.left) * scaleX);
        const y = Math.round((e.clientY - rect.top) * scaleY);
        
        setCurrentLine({
          ...currentLine,
          end: { x, y }
        });
      }
    }
  };

  const finishZone = () => {
    if (currentZone.length >= 3) {
      const newZone = {
        id: `zone_${Date.now()}`,
        name: `Zona ${zones.length + 1}`,
        points: currentZone.map(p => [p.x, p.y]),
        zone_type: 'red_light',
        saved: false // Marcar como no guardado
      };
      setZones([...zones, newZone]);
      setCurrentZone([]);
      setIsDrawingZone(false);
      toast.success('✅ Zona creada');
    } else {
      toast.error('❌ La zona debe tener al menos 3 puntos');
    }
  };

  const cancelDrawing = () => {
    setIsDrawingLine(false);
    setIsDrawingZone(false);
    setCurrentLine(null);
    setCurrentZone([]);
    toast.info('❌ Dibujo cancelado');
  };

  const saveConfiguration = async () => {
    const unsavedLines = lines.filter(line => !line.saved);
    const unsavedZones = zones.filter(zone => !zone.saved);
    
    if (unsavedLines.length === 0 && unsavedZones.length === 0) {
      toast.info('ℹ️ No hay cambios para guardar');
      return;
    }
    
    setLoading(true);
    try {
      let savedLines = 0;
      let savedZones = 0;
      
      // Guardar líneas nuevas
      for (const line of unsavedLines) {
        const lineData = {
          id: line.id,
          name: line.name,
          points: line.points,
          lane: line.lane,
          line_type: line.line_type,
          distance_to_next: line.distance_to_next,
          enabled: true
        };
        
        console.log('💾 Guardando línea:', lineData);
        await apiService.addLine(lineData);
        savedLines++;
      }
      
      // Guardar zonas nuevas
      for (const zone of unsavedZones) {
        const zoneData = {
          id: zone.id,
          name: zone.name,
          points: zone.points,
          zone_type: zone.zone_type,
          enabled: true
        };
        
        console.log('💾 Guardando zona:', zoneData);
        await apiService.addZone(zoneData);
        savedZones++;
      }
      
      toast.success(`✅ Configuración guardada: ${savedLines} líneas, ${savedZones} zonas`);
      
      // Marcar como guardado
      setLines(prev => prev.map(line => ({ ...line, saved: true })));
      setZones(prev => prev.map(zone => ({ ...zone, saved: true })));
      
      // Recargar configuración para verificar
      setTimeout(() => {
        loadAnalysisConfig();
      }, 1000);
      
    } catch (error) {
      console.error('❌ Error guardando configuración:', error);
      toast.error('❌ Error guardando configuración');
    } finally {
      setLoading(false);
    }
  };

  const deleteLineById = async (lineId) => {
    try {
      console.log('🗑️ Eliminando línea:', lineId);
      await apiService.deleteLine(lineId);
      setLines(lines.filter(l => l.id !== lineId));
      toast.success('✅ Línea eliminada');
      
      // Recargar configuración
      setTimeout(() => {
        loadAnalysisConfig();
      }, 500);
    } catch (error) {
      console.error('❌ Error eliminando línea:', error);
      toast.error('❌ Error eliminando línea');
    }
  };

  const deleteZoneById = async (zoneId) => {
    try {
      console.log('🗑️ Eliminando zona:', zoneId);
      await apiService.deleteZone(zoneId);
      setZones(zones.filter(z => z.id !== zoneId));
      toast.success('✅ Zona eliminada');
      
      // Recargar configuración
      setTimeout(() => {
        loadAnalysisConfig();
      }, 500);
    } catch (error) {
      console.error('❌ Error eliminando zona:', error);
      toast.error('❌ Error eliminando zona');
    }
  };

  const clearAll = () => {
    if (window.confirm('¿Está seguro de eliminar TODA la configuración temporal?')) {
      setLines([]);
      setZones([]);
      setCurrentLine(null);
      setCurrentZone([]);
      toast.success('🧹 Configuración temporal limpiada');
    }
  };

  const clearAllSaved = async () => {
    if (window.confirm('¿Está seguro de eliminar TODA la configuración guardada? Esta acción no se puede deshacer.')) {
      try {
        await apiService.clearAnalysis();
        setLines([]);
        setZones([]);
        toast.success('🧹 Toda la configuración eliminada');
        
        // Recargar configuración
        setTimeout(() => {
          loadAnalysisConfig();
        }, 500);
      } catch (error) {
        console.error('❌ Error limpiando configuración:', error);
        toast.error('❌ Error limpiando configuración');
      }
    }
  };

  const reloadConfiguration = async () => {
    await loadAnalysisConfig();
    toast.success('🔄 Configuración recargada');
  };

  const toggleStream = () => {
    if (!cameraConfigured) {
      toast.error('❌ Configure la cámara primero en "Config. Cámara"');
      return;
    }
    
    setIsStreamActive(!isStreamActive);
    
    if (!isStreamActive) {
      // Activar stream
      updateStreamUrl();
      setStreamError(false);
      toast.info('▶️ Stream activado');
    } else {
      // Desactivar stream
      toast.info('⏸️ Stream pausado');
    }
  };

  const refreshStream = () => {
    if (!cameraConfigured) {
      toast.error('❌ Configure la cámara primero');
      return;
    }
    
    console.log('🔄 Refrescando stream...');
    setStreamError(false);
    updateStreamUrl();
    toast.info('🔄 Stream refrescado');
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
            onClick={toggleStream}
            className={`flex items-center px-4 py-2 rounded-md text-white ${
              isStreamActive ? 'bg-red-600 hover:bg-red-700' : 'bg-green-600 hover:bg-green-700'
            }`}
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
          
          <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-4 mb-4">
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
              <option value="carril_1">Carril 1</option>
              <option value="carril_2">Carril 2</option>
              <option value="carril_3">Carril 3</option>
              <option value="carril_4">Carril 4</option>
            </select>
            
            <select
              value={lineConfig.type}
              onChange={(e) => setLineConfig({...lineConfig, type: e.target.value})}
              className="px-3 py-2 bg-gray-700 text-white border border-gray-600 rounded-md"
            >
              <option value="counting">Línea de Conteo</option>
              <option value="speed">Línea de Velocidad</option>
            </select>

            {lineConfig.type === 'speed' && (
              <input
                type="number"
                placeholder="Distancia (metros)"
                value={lineConfig.distance}
                onChange={(e) => setLineConfig({...lineConfig, distance: parseFloat(e.target.value)})}
                className="px-3 py-2 bg-gray-700 text-white border border-gray-600 rounded-md"
                min="1"
                max="100"
                step="0.5"
              />
            )}
          </div>
          
          <div className="bg-yellow-900/20 border border-yellow-600 rounded-lg p-3">
            <p className="text-yellow-300 text-sm">
              {!currentLine ? (
                <><strong>Paso 1:</strong> Haz clic para establecer el primer punto de la línea</>
              ) : (
                <><strong>Paso 2:</strong> Haz clic para establecer el segundo punto y finalizar la línea</>
              )}
            </p>
          </div>
        </div>
      )}

      {/* Controles de dibujo */}
      <div className="bg-gray-800 rounded-lg p-4">
        <div className="flex flex-wrap gap-4">
          <button
            onClick={() => {
              if (!cameraConfigured) {
                toast.error('❌ Configure la cámara primero');
                return;
              }
              if (!isDrawingLine) {
                setIsDrawingLine(true);
                setIsDrawingZone(false);
                setCurrentLine(null);
              }
            }}
            disabled={isDrawingLine || isDrawingZone || !cameraConfigured}
            className={`flex items-center px-4 py-2 rounded-md ${
              isDrawingLine ? 'bg-green-600' : 'bg-gray-600 hover:bg-gray-700'
            } text-white disabled:opacity-50`}
          >
            <PencilIcon className="h-4 w-4 mr-2" />
            {isDrawingLine ? 'Dibujando Línea...' : 'Dibujar Línea'}
          </button>

          <button
            onClick={() => {
              if (!cameraConfigured) {
                toast.error('❌ Configure la cámara primero');
                return;
              }
              if (!isDrawingZone) {
                setIsDrawingZone(true);
                setIsDrawingLine(false);
                setCurrentZone([]);
              }
            }}
            disabled={isDrawingLine || isDrawingZone || !cameraConfigured}
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
            disabled={loading || (lines.filter(l => !l.saved).length === 0 && zones.filter(z => !z.saved).length === 0)}
            className="flex items-center px-4 py-2 bg-green-600 text-white rounded-md hover:bg-green-700 disabled:opacity-50"
          >
            <CheckIcon className="h-4 w-4 mr-2" />
            {loading ? 'Guardando...' : `Guardar (${lines.filter(l => !l.saved).length + zones.filter(z => !z.saved).length} cambios)`}
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
                isStreamActive && cameraConfigured && !streamError ? 'bg-green-500 animate-pulse' : 'bg-red-500'
              }`}></div>
              <span className="text-sm text-gray-300">
                {isStreamActive && cameraConfigured && !streamError ? 'En Vivo' : 
                 streamError ? 'Error' :
                 !cameraConfigured ? 'Sin Configurar' : 'Pausado'}
              </span>
            </div>
            
            {/* Información de FPS */}
            {systemStatus.fps > 0 && (
              <div className="text-sm text-gray-300">
                {systemStatus.fps} FPS
              </div>
            )}

            {/* Botón de refresh */}
            <button
              onClick={refreshStream}
              disabled={!cameraConfigured}
              className="px-3 py-1 bg-blue-600 text-white rounded text-sm hover:bg-blue-700 disabled:opacity-50"
            >
              <ArrowPathIcon className="h-4 w-4" />
            </button>
          </div>
        </div>

        <div className="relative bg-black rounded-lg overflow-hidden">
          {isStreamActive && cameraConfigured && !streamError ? (
            <div className="relative">
              <img
                ref={imgRef}
                src={streamUrl}
                alt="Stream de Cámara - Análisis en Tiempo Real"
                className="w-full h-auto rounded-lg cursor-crosshair"
                onClick={handleMouseClick}
                onMouseMove={handleMouseMove}
                onError={(e) => {
                  console.error('❌ Error en stream:', e);
                  setStreamError(true);
                }}
                onLoad={() => {
                  console.log('✅ Stream cargado exitosamente');
                  setStreamError(false);
                }}
                style={{ 
                  maxHeight: '600px', 
                  objectFit: 'contain',
                  backgroundColor: '#000',
                  minHeight: '400px'
                }}
              />
              
              {/* Overlay SVG con líneas y zonas */}
              {showOverlay && imgRef.current && (
                <svg 
                  className="absolute top-0 left-0 w-full h-full pointer-events-none"
                  style={{ maxHeight: '600px' }}
                  preserveAspectRatio="xMidYMid meet"
                  viewBox={`0 0 ${imgRef.current?.naturalWidth || 1280} ${imgRef.current?.naturalHeight || 720}`}
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
                        strokeWidth="4"
                        strokeDasharray={line.saved ? "0" : "10,5"}
                      />
                      <text
                        x={(line.points[0][0] + line.points[1][0]) / 2}
                        y={(line.points[0][1] + line.points[1][1]) / 2 - 15}
                        fill="#FFFFFF"
                        fontSize="14"
                        fontWeight="bold"
                        textAnchor="middle"
                        stroke="#000000"
                        strokeWidth="1"
                        className="pointer-events-none"
                      >
                        {line.name}
                      </text>
                    </g>
                  ))}
                  
                  {/* Línea siendo dibujada */}
                  {currentLine && currentLine.end && (
                    <line
                      x1={currentLine.start.x}
                      y1={currentLine.start.y}
                      x2={currentLine.end.x}
                      y2={currentLine.end.y}
                      stroke="#FF0000"
                      strokeWidth="3"
                      strokeDasharray="5,5"
                    />
                  )}
                  
                  {/* Zonas guardadas */}
                  {zones.map((zone) => (
                    <g key={zone.id}>
                      <polygon
                        points={zone.points.map(p => `${p[0]},${p[1]}`).join(' ')}
                        fill="rgba(255, 0, 0, 0.3)"
                        stroke="#FF0000"
                        strokeWidth="3"
                        strokeDasharray={zone.saved ? "0" : "8,4"}
                      />
                      <text
                        x={zone.points.reduce((sum, p) => sum + p[0], 0) / zone.points.length}
                        y={zone.points.reduce((sum, p) => sum + p[1], 0) / zone.points.length}
                        fill="#FFFFFF"
                        fontSize="14"
                        fontWeight="bold"
                        textAnchor="middle"
                        stroke="#000000"
                        strokeWidth="1"
                        className="pointer-events-none"
                      >
                        {zone.name}
                      </text>
                    </g>
                  ))}
                  
                  {/* Zona siendo dibujada */}
                  {currentZone.length > 0 && (
                    <g>
                      <polygon
                        points={currentZone.map(p => `${p.x},${p.y}`).join(' ')}
                        fill="rgba(0, 255, 0, 0.3)"
                        stroke="#00FF00"
                        strokeWidth="3"
                        strokeDasharray="5,5"
                      />
                      {currentZone.map((point, index) => (
                        <circle
                          key={index}
                          cx={point.x}
                          cy={point.y}
                          r="5"
                          fill="#00FF00"
                          stroke="#000000"
                          strokeWidth="2"
                        />
                      ))}
                    </g>
                  )}
                </svg>
              )}
            </div>
          ) : (
            /* Placeholder cuando no hay stream */
            <div className="w-full h-96 bg-gradient-to-br from-gray-700 to-gray-800 rounded-lg flex flex-col items-center justify-center">
              <div className="text-center">
                {streamError ? (
                  <>
                    <ExclamationTriangleIcon className="h-20 w-20 text-red-400 mx-auto mb-4" />
                    <h3 className="text-xl font-medium text-red-300 mb-2">🔴 Error de Conexión</h3>
                    <p className="text-red-400 text-sm mb-4">
                      Problema conectando con el stream de cámara
                    </p>
                    <button
                      onClick={refreshStream}
                      className="px-4 py-2 bg-blue-600 text-white rounded-md hover:bg-blue-700 transition-colors"
                    >
                      🔄 Reintentar
                    </button>
                  </>
                ) : !cameraConfigured ? (
                  <>
                    <CameraIcon className="h-20 w-20 text-gray-400 mx-auto mb-4" />
                    <h3 className="text-xl font-medium text-gray-300 mb-2">📷 Cámara No Configurada</h3>
                    <p className="text-gray-400 text-sm mb-4">
                      Configure la cámara en "Config. Cámara" para comenzar
                    </p>
                    <button
                      onClick={() => window.location.href = '/camera-config'}
                      className="px-4 py-2 bg-blue-600 text-white rounded-md hover:bg-blue-700 transition-colors"
                    >
                      ⚙️ Ir a Configuración
                    </button>
                  </>
                ) : (
                  <>
                    <CameraIcon className="h-20 w-20 text-gray-400 mx-auto mb-4" />
                    <h3 className="text-xl font-medium text-gray-300 mb-2">⏸️ Stream Pausado</h3>
                    <p className="text-gray-400 text-sm mb-4">
                      Presiona "Iniciar Stream" para comenzar la transmisión
                    </p>
                    <button
                      onClick={toggleStream}
                      className="px-4 py-2 bg-green-600 text-white rounded-md hover:bg-green-700 transition-colors"
                    >
                      ▶️ Iniciar Stream
                    </button>
                  </>
                )}
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
                      <span className={`px-2 py-1 text-xs rounded ${
                        line.saved ? 'bg-gray-600 text-white' : 'bg-yellow-600 text-white'
                      }`}>
                        {line.saved ? 'GUARDADO' : 'TEMPORAL'}
                      </span>
                      <p className="text-white font-medium">{line.name}</p>
                    </div>
                    
                    <div className="grid grid-cols-2 gap-2 text-sm">
                      <p className="text-gray-300">
                        <span className="text-gray-400">Carril:</span> {line.lane}
                      </p>
                      {line.line_type === 'speed' && line.distance_to_next && (
                        <p className="text-blue-300">
                          <span className="text-gray-400">Distancia:</span> {line.distance_to_next}m
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
                  <div className="flex items-center gap-2 mb-1">
                    <p className="text-white font-medium">{zone.name}</p>
                    <span className={`px-2 py-1 text-xs rounded ${
                      zone.saved ? 'bg-gray-600 text-white' : 'bg-yellow-600 text-white'
                    }`}>
                      {zone.saved ? 'GUARDADO' : 'TEMPORAL'}
                    </span>
                  </div>
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
        <h3 className="text-lg font-semibold text-white mb-4">Estado del Sistema</h3>
        <div className="grid grid-cols-1 md:grid-cols-4 gap-4 text-gray-300">
          <div className="bg-gray-700 p-3 rounded">
            <p className="text-xs text-gray-400">Cámara</p>
            <p className={`font-medium ${cameraConfigured ? 'text-green-400' : 'text-red-400'}`}>
              {cameraConfigured ? '✅ Configurada' : '❌ Sin configurar'}
            </p>
          </div>
          <div className="bg-gray-700 p-3 rounded">
            <p className="text-xs text-gray-400">Stream</p>
            <p className={`font-medium ${isStreamActive && !streamError ? 'text-green-400' : 'text-red-400'}`}>
              {isStreamActive && !streamError ? '✅ Activo' : '❌ Inactivo'}
            </p>
          </div>
          <div className="bg-gray-700 p-3 rounded">
            <p className="text-xs text-gray-400">Cambios Sin Guardar</p>
            <p className="font-medium text-white">
              {lines.filter(l => !l.saved).length + zones.filter(z => !z.saved).length}
            </p>
          </div>
          <div className="bg-gray-700 p-3 rounded">
            <p className="text-xs text-gray-400">Análisis Total</p>
            <p className="font-medium text-white">
              {lines.length} líneas, {zones.length} zonas
            </p>
          </div>
        </div>
      </div>
    </div>
  );
};

export default CameraView;