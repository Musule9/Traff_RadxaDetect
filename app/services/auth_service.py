import jwt
import bcrypt
import json
import time
from datetime import datetime, timedelta
from typing import Optional, Dict
from loguru import logger

class AuthService:
    """Servicio de autenticación JWT"""
    
    def __init__(self, secret_key: str = "vehicle_detection_secret_key_2024"):
        self.secret_key = secret_key
        self.algorithm = "HS256"
        self.token_expiry = 3600  # 1 hora
        self.revoked_tokens = set()
        self.users = self._load_users()
    
    def _load_users(self) -> Dict:
        """Cargar usuarios desde configuración"""
        try:
            with open("/app/config/system.json", "r") as f:
                config = json.load(f)
                auth_config = config.get("authentication", {})
                
                # Usuario por defecto
                default_user = auth_config.get("default_username", "admin")
                default_pass = auth_config.get("default_password", "admin123")
                
                # Hash de la contraseña
                hashed_pass = bcrypt.hashpw(default_pass.encode(), bcrypt.gensalt())
                
                return {
                    default_user: {
                        "password_hash": hashed_pass,
                        "role": "admin"
                    }
                }
        except Exception as e:
            logger.error(f"Error cargando usuarios: {e}")
            # Usuario por defecto de emergencia
            return {
                "admin": {
                    "password_hash": bcrypt.hashpw(b"admin123", bcrypt.gensalt()),
                    "role": "admin"
                }
            }
    
    async def authenticate(self, username: str, password: str) -> Optional[str]:
        """Autenticar usuario y generar token"""
        try:
            if username not in self.users:
                return None
            
            user = self.users[username]
            password_hash = user["password_hash"]
            
            # Verificar contraseña
            if bcrypt.checkpw(password.encode(), password_hash):
                # Generar token
                payload = {
                    "username": username,
                    "role": user["role"],
                    "exp": datetime.utcnow() + timedelta(seconds=self.token_expiry),
                    "iat": datetime.utcnow()
                }
                
                token = jwt.encode(payload, self.secret_key, algorithm=self.algorithm)
                logger.info(f"Usuario autenticado: {username}")
                return token
            
            return None
            
        except Exception as e:
            logger.error(f"Error en autenticación: {e}")
            return None
    
    def verify_token(self, token: str) -> bool:
        """Verificar validez del token"""
        try:
            if token in self.revoked_tokens:
                return False
            
            payload = jwt.decode(token, self.secret_key, algorithms=[self.algorithm])
            return True
            
        except jwt.ExpiredSignatureError:
            return False
        except jwt.InvalidTokenError:
            return False
        except Exception as e:
            logger.error(f"Error verificando token: {e}")
            return False
    
    def revoke_token(self, token: str):
        """Revocar token"""
        self.revoked_tokens.add(token)
    
    def get_user_from_token(self, token: str) -> Optional[Dict]:
        """Obtener información de usuario desde token"""
        try:
            if token in self.revoked_tokens:
                return None
            
            payload = jwt.decode(token, self.secret_key, algorithms=[self.algorithm])
            return {
                "username": payload.get("username"),
                "role": payload.get("role")
            }
        except Exception:
            return None
