import React, { useState, useEffect } from 'react';
import { 
  CalendarIcon, 
  DocumentArrowDownIcon,
  ChartBarIcon,
  ClockIcon,
  TableCellsIcon
} from '@heroicons/react/24/outline';
import { LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer, BarChart, Bar } from 'recharts';
import { toast } from 'react-toastify';
import { apiService } from '../../services/api';

const Reports = () => {
  const [selectedDate, setSelectedDate] = useState(new Date().toISOString().split('T')[0]);
  const [reportType, setReportType] = useState('vehicle');
  const [reportData, setReportData] = useState(null);
  const [loading, setLoading] = useState(false);
  const [chartData, setChartData] = useState([]);
  const [summary, setSummary] = useState({
    totalRecords: 0,
    avgSpeed: 0,
    peakHour: '--:--',
    maxPerHour: 0
  });

  useEffect(() => {
    if (reportData && reportData.length > 0) {
      generateChartData();
      generateSummary();
    }
  }, [reportData]);

  const generateChartData = () => {
    if (!reportData) return;
    
    // Agrupar datos por hora
    const hourlyData = {};
    
    reportData.forEach(item => {
      const hour = new Date(item.timestamp).getHours();
      if (!hourlyData[hour]) {
        hourlyData[hour] = { 
          hour: `${hour.toString().padStart(2, '0')}:00`, 
          count: 0, 
          speeds: [] 
        };
      }
      hourlyData[hour].count++;
      if (item.velocidad && item.velocidad > 0) {
        hourlyData[hour].speeds.push(item.velocidad);
      }
    });

    // Calcular velocidad promedio por hora y completar horas faltantes
    const chartArray = [];
    for (let hour = 0; hour < 24; hour++) {
      const hourStr = `${hour.toString().padStart(2, '0')}:00`;
      const data = hourlyData[hour] || { hour: hourStr, count: 0, speeds: [] };
      
      chartArray.push({
        hour: hourStr,
        count: data.count,
        avgSpeed: data.speeds.length > 0 
          ? Math.round(data.speeds.reduce((a, b) => a + b, 0) / data.speeds.length)
          : 0
      });
    }

    setChartData(chartArray);
  };

  const generateSummary = () => {
    if (!reportData || reportData.length === 0) return;

    const speedValues = reportData
      .filter(r => r.velocidad && r.velocidad > 0)
      .map(r => r.velocidad);

    const avgSpeed = speedValues.length > 0 
      ? speedValues.reduce((a, b) => a + b, 0) / speedValues.length 
      : 0;

    // Encontrar hora pico
    const hourCounts = {};
    reportData.forEach(item => {
      const hour = new Date(item.timestamp).getHours();
      hourCounts[hour] = (hourCounts[hour] || 0) + 1;
    });

    const peakHour = Object.keys(hourCounts).reduce((a, b) => 
      hourCounts[a] > hourCounts[b] ? a : b, '0'
    );

    const maxPerHour = Object.values(hourCounts).length > 0 
      ? Math.max(...Object.values(hourCounts)) 
      : 0;

    setSummary({
      totalRecords: reportData.length,
      avgSpeed: Math.round(avgSpeed),
      peakHour: `${peakHour.padStart(2, '0')}:00`,
      maxPerHour
    });
  };

  const fetchReport = async () => {
    setLoading(true);
    try {
      const dateStr = selectedDate.replace(/-/g, '_');
      const response = await apiService.exportData(dateStr, reportType);
      
      if (reportType === 'all') {
        setReportData(response.data.vehicle_crossings || []);
      } else {
        setReportData(response.data || []);
      }
      
      toast.success('Reporte generado exitosamente');
    } catch (error) {
      toast.error('Error generando reporte');
      console.error('Error:', error);
      setReportData([]);
    } finally {
      setLoading(false);
    }
  };

  const exportReport = async () => {
    try {
      const dateStr = selectedDate.replace(/-/g, '_');
      const response = await apiService.exportData(dateStr, reportType);
      
      // Crear y descargar archivo JSON
      const dataStr = JSON.stringify(response, null, 2);
      const dataUri = 'data:application/json;charset=utf-8,'+ encodeURIComponent(dataStr);
      
      const exportFileDefaultName = `reporte_${reportType}_${dateStr}.json`;
      
      const linkElement = document.createElement('a');
      linkElement.setAttribute('href', dataUri);
      linkElement.setAttribute('download', exportFileDefaultName);
      linkElement.click();
      
      toast.success('Reporte exportado exitosamente');
    } catch (error) {
      toast.error('Error exportando reporte');
    }
  };

  const SummaryCard = ({ icon: Icon, title, value, color = "blue" }) => (
    <div className="bg-gray-800 rounded-lg p-4">
      <div className="flex items-center">
        <Icon className={`h-8 w-8 text-${color}-500`} />
        <div className="ml-3">
          <p className="text-sm font-medium text-gray-400">{title}</p>
          <p className="text-xl font-bold text-white">{value}</p>
        </div>
      </div>
    </div>
  );

  return (
    <div className="space-y-6">
      <h1 className="text-3xl font-bold text-white">Reportes y Analíticas</h1>

      {/* Controles de reporte */}
      <div className="bg-gray-800 rounded-lg p-6">
        <div className="grid grid-cols-1 md:grid-cols-4 gap-4 items-end">
          <div>
            <label className="block text-sm font-medium text-gray-300 mb-2">
              Fecha
            </label>
            <div className="relative">
              <input
                type="date"
                value={selectedDate}
                onChange={(e) => setSelectedDate(e.target.value)}
                className="w-full px-3 py-2 bg-gray-700 text-white border border-gray-600 rounded-md focus:outline-none focus:ring-2 focus:ring-blue-500"
              />
              <CalendarIcon className="absolute right-3 top-2.5 h-5 w-5 text-gray-400 pointer-events-none" />
            </div>
          </div>

          <div>
            <label className="block text-sm font-medium text-gray-300 mb-2">
              Tipo de Reporte
            </label>
            <select
              value={reportType}
              onChange={(e) => setReportType(e.target.value)}
              className="w-full px-3 py-2 bg-gray-700 text-white border border-gray-600 rounded-md focus:outline-none focus:ring-2 focus:ring-blue-500"
            >
              <option value="vehicle">Cruces de Vehículos</option>
              <option value="red_light">Zona de Semáforo Rojo</option>
              <option value="all">Reporte Completo</option>
            </select>
          </div>

          <button
            onClick={fetchReport}
            disabled={loading}
            className="flex items-center justify-center px-4 py-2 bg-blue-600 text-white rounded-md hover:bg-blue-700 disabled:opacity-50"
          >
            <ChartBarIcon className="h-5 w-5 mr-2" />
            {loading ? 'Generando...' : 'Generar Reporte'}
          </button>

          <button
            onClick={exportReport}
            disabled={!reportData || loading}
            className="flex items-center justify-center px-4 py-2 bg-green-600 text-white rounded-md hover:bg-green-700 disabled:opacity-50"
          >
            <DocumentArrowDownIcon className="h-5 w-5 mr-2" />
            Exportar
          </button>
        </div>
      </div>

      {/* Resumen estadístico */}
      {reportData && (
        <div className="grid grid-cols-1 md:grid-cols-4 gap-4">
          <SummaryCard
            icon={TableCellsIcon}
            title="Total Registros"
            value={summary.totalRecords}
            color="blue"
          />
          <SummaryCard
            icon={ClockIcon}
            title="Velocidad Promedio"
            value={`${summary.avgSpeed} km/h`}
            color="green"
          />
          <SummaryCard
            icon={ChartBarIcon}
            title="Hora Pico"
            value={summary.peakHour}
            color="yellow"
          />
          <SummaryCard
            icon={ChartBarIcon}
            title="Máx. por Hora"
            value={summary.maxPerHour}
            color="purple"
          />
        </div>
      )}

      {/* Gráficos */}
      {chartData.length > 0 && (
        <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
          <div className="bg-gray-800 rounded-lg p-6">
            <h3 className="text-lg font-semibold text-white mb-4">Conteo por Hora</h3>
            <ResponsiveContainer width="100%" height={300}>
              <BarChart data={chartData}>
                <CartesianGrid strokeDasharray="3,3" stroke="#374151" />
                <XAxis dataKey="hour" stroke="#9CA3AF" fontSize={12} />
                <YAxis stroke="#9CA3AF" fontSize={12} />
                <Tooltip 
                  contentStyle={{ 
                    backgroundColor: '#1F2937', 
                    border: '1px solid #374151',
                    color: '#F9FAFB',
                    borderRadius: '8px'
                  }} 
                />
                <Bar dataKey="count" fill="#3B82F6" radius={[2, 2, 0, 0]} />
              </BarChart>
            </ResponsiveContainer>
          </div>

          <div className="bg-gray-800 rounded-lg p-6">
            <h3 className="text-lg font-semibold text-white mb-4">Velocidad Promedio por Hora</h3>
            <ResponsiveContainer width="100%" height={300}>
              <LineChart data={chartData}>
                <CartesianGrid strokeDasharray="3,3" stroke="#374151" />
                <XAxis dataKey="hour" stroke="#9CA3AF" fontSize={12} />
                <YAxis stroke="#9CA3AF" fontSize={12} />
                <Tooltip 
                  contentStyle={{ 
                    backgroundColor: '#1F2937', 
                    border: '1px solid #374151',
                    color: '#F9FAFB',
                    borderRadius: '8px'
                  }} 
                />
                <Line 
                  type="monotone" 
                  dataKey="avgSpeed" 
                  stroke="#10B981" 
                  strokeWidth={3}
                  dot={{ fill: '#10B981', strokeWidth: 2, r: 4 }}
                  activeDot={{ r: 6, stroke: '#10B981', strokeWidth: 2 }}
                />
              </LineChart>
            </ResponsiveContainer>
          </div>
        </div>
      )}

      {/* Tabla de datos */}
      {reportData && reportData.length > 0 && (
        <div className="bg-gray-800 rounded-lg p-6">
          <h3 className="text-lg font-semibold text-white mb-4">
            Detalle de Registros ({reportData.length} total)
          </h3>
          <div className="overflow-x-auto">
            <table className="w-full text-sm text-gray-300">
              <thead>
                <tr className="border-b border-gray-700">
                  <th className="text-left py-3 px-4 font-medium">Hora</th>
                  <th className="text-left py-3 px-4 font-medium">Vehículo</th>
                  <th className="text-left py-3 px-4 font-medium">Línea</th>
                  <th className="text-left py-3 px-4 font-medium">Velocidad</th>
                  <th className="text-left py-3 px-4 font-medium">Carril</th>
                  <th className="text-left py-3 px-4 font-medium">Confianza</th>
                </tr>
              </thead>
              <tbody>
                {reportData.slice(0, 100).map((row, index) => (
                  <tr key={index} className="border-b border-gray-700 hover:bg-gray-700/50">
                    <td className="py-2 px-4">
                      {new Date(row.timestamp).toLocaleTimeString()}
                    </td>
                    <td className="py-2 px-4">#{row.vehicle_id}</td>
                    <td className="py-2 px-4">{row.line_name || row.line_id}</td>
                    <td className="py-2 px-4">
                      {row.velocidad ? (
                        <span className={`px-2 py-1 rounded text-xs ${
                          row.velocidad > 60 ? 'bg-red-600' : 
                          row.velocidad > 40 ? 'bg-yellow-600' : 'bg-green-600'
                        }`}>
                          {Math.round(row.velocidad)} km/h
                        </span>
                      ) : (
                        <span className="text-gray-500">-</span>
                      )}
                    </td>
                    <td className="py-2 px-4">{row.carril || '-'}</td>
                    <td className="py-2 px-4">
                      {row.confianza ? (
                        <span className="text-xs">
                          {Math.round(row.confianza * 100)}%
                        </span>
                      ) : (
                        <span className="text-gray-500">-</span>
                      )}
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
            {reportData.length > 100 && (
              <div className="text-center mt-4">
                <p className="text-gray-400 text-sm">
                  Mostrando primeros 100 registros de {reportData.length} total
                </p>
                <button
                  onClick={exportReport}
                  className="mt-2 px-4 py-2 bg-blue-600 text-white rounded text-sm hover:bg-blue-700"
                >
                  Descargar todos los datos
                </button>
              </div>
            )}
          </div>
        </div>
      )}

      {/* Estado sin datos */}
      {!loading && !reportData && (
        <div className="bg-gray-800 rounded-lg p-12 text-center">
          <ChartBarIcon className="h-16 w-16 text-gray-400 mx-auto mb-4" />
          <h3 className="text-lg font-medium text-gray-300 mb-2">No hay datos disponibles</h3>
          <p className="text-gray-400">
            Seleccione una fecha y genere un reporte para ver los datos
          </p>
        </div>
      )}

      {/* Estado de cargando */}
      {loading && (
        <div className="bg-gray-800 rounded-lg p-12 text-center">
          <div className="animate-spin rounded-full h-12 w-12 border-b-2 border-blue-500 mx-auto mb-4"></div>
          <p className="text-gray-400">Generando reporte...</p>
        </div>
      )}
    </div>
  );
};

export default Reports;