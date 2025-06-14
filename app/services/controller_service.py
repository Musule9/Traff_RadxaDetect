import aiohttp
import asyncio
import json
from typing import Dict, Optional
from loguru import logger
import time

class ControllerService:
    """Servicio de comunicación con controladora de semáforos"""
    
    def __init__(self):
        self.controller_config = self._load_controller_config()
        self.current_status = {}
        self.last_analytic_sent = {}
        self.session = None
    
    def _load_controller_config(self) -> Dict:
        """Cargar configuración de controladora"""
        try:
            with open("/app/config/controllers.json", "r") as f:
                config = json.load(f)
                return config.get("controllers", {})
        except Exception as e:
            logger.error(f"Error cargando configuración de controladora: {e}")
            return {}
    
    async def _get_session(self):
        """Obtener sesión HTTP"""
        if self.session is None or self.session.closed:
            timeout = aiohttp.ClientTimeout(total=5)
            self.session = aiohttp.ClientSession(timeout=timeout)
        return self.session
    
    async def send_analytic(self, data: Dict) -> bool:
        """Enviar analítico a controladora"""
        try:
            # Obtener configuración de controladora
            controller_id = data.get("controladora_id", "CTRL_001")
            if controller_id not in self.controller_config:
                logger.error(f"Controladora no configurada: {controller_id}")
                return False
            
            controller = self.controller_config[controller_id]
            url = f"http://{controller['network']['ip']}:{controller['network']['port']}{controller['endpoints']['analytic']}"
            
            # Evitar spam de analíticos
            phase = data.get("fase", "fase1")
            current_time = time.time()
            
            if phase in self.last_analytic_sent:
                if current_time - self.last_analytic_sent[phase] < 5:  # Mínimo 5 segundos entre analíticos
                    logger.debug(f"Analítico ignorado por spam protection: {phase}")
                    return True
            
            # Preparar payload
            payload = {
                "fase": phase,
                "puntos": data.get("puntos", 1),
                "vehiculos": True,
                "timestamp": current_time
            }
            
            # Enviar analítico
            session = await self._get_session()
            async with session.post(url, json=payload) as response:
                if response.status == 200:
                    logger.info(f"✅ Analítico enviado exitosamente: {phase}")
                    self.last_analytic_sent[phase] = current_time
                    return True
                else:
                    logger.error(f"Error enviando analítico: {response.status}")
                    return False
                    
        except asyncio.TimeoutError:
            logger.error("Timeout enviando analítico a controladora")
            return False
        except Exception as e:
            logger.error(f"Error enviando analítico: {e}")
            return False
    
    async def get_traffic_light_status(self) -> Optional[Dict]:
        """Obtener estado de semáforos de controladora"""
        try:
            # Para simplificar, usamos la primera controladora configurada
            if not self.controller_config:
                return None
            
            controller_id = list(self.controller_config.keys())[0]
            controller = self.controller_config[controller_id]
            url = f"http://{controller['network']['ip']}:{controller['network']['port']}{controller['endpoints']['status']}"
            
            session = await self._get_session()
            async with session.get(url) as response:
                if response.status == 200:
                    data = await response.json()
                    fases = data.get("fases", {})
                    self.current_status = fases
                    return fases
                else:
                    logger.warning(f"Error obteniendo estado: {response.status}")
                    return None
                    
        except asyncio.TimeoutError:
            logger.warning("Timeout obteniendo estado de controladora")
            return None
        except Exception as e:
            logger.error(f"Error obteniendo estado: {e}")
            return None
    
    def update_traffic_light_status(self, fases: Dict):
        """Actualizar estado local de semáforos"""
        self.current_status.update(fases)
        logger.debug(f"Estado de semáforos actualizado: {self.current_status}")
    
    async def close(self):
        """Cerrar sesión HTTP"""
        if self.session and not self.session.closed:
            await self.session.close()