import os
import tarfile
import shutil
import argparse
from datetime import datetime, timedelta
from pathlib import Path
from loguru import logger

class BackupManager:
    """Gestor de respaldos del sistema"""
    
    def __init__(self, data_dir="/app/data", config_dir="/app/config", backup_dir="/app/backups"):
        self.data_dir = Path(data_dir)
        self.config_dir = Path(config_dir)
        self.backup_dir = Path(backup_dir)
        self.backup_dir.mkdir(exist_ok=True)
    
    def create_backup(self, backup_type="full"):
        """Crear respaldo"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        if backup_type == "full":
            backup_file = self.backup_dir / f"full_backup_{timestamp}.tar.gz"
            self._create_full_backup(backup_file)
        elif backup_type == "data":
            backup_file = self.backup_dir / f"data_backup_{timestamp}.tar.gz"
            self._create_data_backup(backup_file)
        elif backup_type == "config":
            backup_file = self.backup_dir / f"config_backup_{timestamp}.tar.gz"
            self._create_config_backup(backup_file)
        else:
            raise ValueError(f"Tipo de respaldo no válido: {backup_type}")
        
        logger.info(f"✅ Respaldo creado: {backup_file}")
        return backup_file
    
    def _create_full_backup(self, backup_file):
        """Crear respaldo completo"""
        with tarfile.open(backup_file, "w:gz") as tar:
            if self.data_dir.exists():
                tar.add(self.data_dir, arcname="data", filter=self._exclude_logs)
            if self.config_dir.exists():
                tar.add(self.config_dir, arcname="config")
    
    def _create_data_backup(self, backup_file):
        """Crear respaldo solo de datos"""
        with tarfile.open(backup_file, "w:gz") as tar:
            if self.data_dir.exists():
                tar.add(self.data_dir, arcname="data", filter=self._exclude_logs)
    
    def _create_config_backup(self, backup_file):
        """Crear respaldo solo de configuración"""
        with tarfile.open(backup_file, "w:gz") as tar:
            if self.config_dir.exists():
                tar.add(self.config_dir, arcname="config")
    
    def _exclude_logs(self, tarinfo):
        """Filtro para excluir archivos de log"""
        if tarinfo.name.endswith('.log'):
            return None
        return tarinfo
    
    def restore_backup(self, backup_file, restore_type="full"):
        """Restaurar desde respaldo"""
        backup_path = Path(backup_file)
        if not backup_path.exists():
            raise FileNotFoundError(f"Archivo de respaldo no encontrado: {backup_file}")
        
        logger.info(f"🔄 Restaurando desde: {backup_file}")
        
        with tarfile.open(backup_file, "r:gz") as tar:
            if restore_type == "full":
                tar.extractall("/app")
            elif restore_type == "data":
                tar.extractall("/app", members=[m for m in tar.getmembers() if m.name.startswith("data/")])
            elif restore_type == "config":
                tar.extractall("/app", members=[m for m in tar.getmembers() if m.name.startswith("config/")])
        
        logger.info("✅ Restauración completada")
    
    def cleanup_old_backups(self, retention_days=7):
        """Limpiar respaldos antiguos"""
        cutoff_date = datetime.now() - timedelta(days=retention_days)
        
        for backup_file in self.backup_dir.glob("*.tar.gz"):
            file_date = datetime.fromtimestamp(backup_file.stat().st_mtime)
            if file_date < cutoff_date:
                backup_file.unlink()
                logger.info(f"🗑️  Respaldo eliminado: {backup_file}")
    
    def list_backups(self):
        """Listar respaldos disponibles"""
        backups = []
        for backup_file in sorted(self.backup_dir.glob("*.tar.gz")):
            stat = backup_file.stat()
            backups.append({
                "file": backup_file.name,
                "size": stat.st_size,
                "date": datetime.fromtimestamp(stat.st_mtime)
            })
        return backups

def main():
    parser = argparse.ArgumentParser(description="Gestor de respaldos")
    parser.add_argument("action", choices=["create", "restore", "list", "cleanup"])
    parser.add_argument("--type", default="full", choices=["full", "data", "config"])
    parser.add_argument("--file", help="Archivo de respaldo para restaurar")
    parser.add_argument("--retention", type=int, default=7, help="Días de retención")
    
    args = parser.parse_args()
    
    manager = BackupManager()
    
    if args.action == "create":
        manager.create_backup(args.type)
    elif args.action == "restore":
        if not args.file:
            logger.error("❌ Especifique --file para restaurar")
            return
        manager.restore_backup(args.file, args.type)
    elif args.action == "list":
        backups = manager.list_backups()
        for backup in backups:
            logger.info(f"{backup['file']} - {backup['size']} bytes - {backup['date']}")
    elif args.action == "cleanup":
        manager.cleanup_old_backups(args.retention)

if __name__ == "__main__":
    main()
      </div>
    );
  }

  if (!isAuthenticated) {
    return (
      <div className="min-h-screen bg-gray-900">
        <Login onLogin={handleLogin} />
        <Toaster position="top-right" />
      </div>
    );
  }

  return (
    <Router>
      <div className="min-h-screen bg-gray-900 text-white">
        <div className="flex">
          <Sidebar />
          <div className="flex-1 flex flex-col">
            <Header user={user} onLogout={handleLogout} />
            <main className="flex-1 p-6">
              <Routes>
                <Route path="/" element={<Navigate to="/dashboard" />} />
                <Route path="/dashboard" element={<Dashboard />} />
                <Route path="/camera" element={<CameraView />} />
                <Route path="/config" element={<Configuration />} />
                <Route path="/reports" element={<Reports />} />
              </Routes>
            </main>
          </div>
        </div>
        <Toaster position="top-right" />
      </div>
    </Router>
  );
}

export default App;