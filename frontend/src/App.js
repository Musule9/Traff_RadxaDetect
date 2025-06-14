import React, { useState, useEffect } from 'react';
import { BrowserRouter as Router, Routes, Route, Navigate } from 'react-router-dom';
import toast, { Toaster } from 'react-hot-toast';
import Login from './components/Login';
import Dashboard from './components/Dashboard';
import CameraView from './components/CameraView/CameraView';
import Configuration from './components/Configuration';
import Reports from './components/Reports/Reports';
import Sidebar from './components/Sidebar';
import Header from './components/Header';
import api from './services/api';
import './App.css';

function App() {
    const [isAuthenticated, setIsAuthenticated] = useState(false);
    const [loading, setLoading] = useState(true);
    const [user, setUser] = useState(null);

    useEffect(() => {
        checkAuth();
    }, []);

    const checkAuth = async () => {
        try {
            const token = localStorage.getItem('token');
            if (token) {
                api.defaults.headers.common['Authorization'] = `Bearer ${token}`;
                // Verificar token con una llamada a la API
                await api.get('/api/camera/status');
                setIsAuthenticated(true);
                setUser({ username: 'admin' }); // Simplificado
            }
        } catch (error) {
            localStorage.removeItem('token');
            delete api.defaults.headers.common['Authorization'];
        } finally {
            setLoading(false);
        }
    };

    const handleLogin = async (username, password) => {
        try {
            const response = await api.post('/api/auth/login', { username, password });
            const { token } = response.data;
      
            localStorage.setItem('token', token);
            api.defaults.headers.common['Authorization'] = `Bearer ${token}`;
      
            setIsAuthenticated(true);
            setUser({ username });
            toast.success('Inicio de sesión exitoso');
        } catch (error) {
            toast.error('Credenciales inválidas');
            throw error;
        }
    };

    const handleLogout = async () => {
        try {
            await api.post('/api/auth/logout');
        } catch (error) {
            // Ignorar errores de logout
        } finally {
            localStorage.removeItem('token');
            delete api.defaults.headers.common['Authorization'];
            setIsAuthenticated(false);
            setUser(null);
            toast.success('Sesión cerrada');
        }
    };

    if (loading) {
        return (
            <div className="min-h-screen bg-gray-900 flex items-center justify-center">
                <div className="animate-spin rounded-full h-32 w-32 border-b-2 border-blue-500"></div></div>
        )
    }
}