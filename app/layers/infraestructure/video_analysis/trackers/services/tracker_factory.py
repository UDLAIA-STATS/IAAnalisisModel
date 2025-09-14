from typing import Dict, List, Type

from ultralytics import YOLO
from app.layers.infraestructure.video_analysis.trackers.interfaces.tracker import Tracker

class TrackerFactoryError(Exception):
    pass

class TrackerFactory:
    def __init__(self, model: YOLO):
        """
        Factory que crea instancias de trackers usando un único modelo YOLO compartido.
        """
        self._registry: Dict[str, Tracker] = {}
        self.model = model
    
    def register(self, key: str, tracker_cls: Type[Tracker]) -> None:
        """
        Registrar una clase tracker con una clave (ej: 'player', 'ball').
        """
        if key in self._registry:
            raise TrackerFactoryError(f"Tracker '{key}' is already registered.")
        self._registry[key] = tracker_cls(self.model)
    
    def create(self, key: str, *args, **kwargs) -> None:
        """
        Crear una instancia de tracker.
        """
        tracker_cls = self._registry.get(key)
        if not tracker_cls:
            raise TrackerFactoryError(f"Tracker '{key}' is not registered.")
    
    def get_trackers(self) -> Dict[str, Tracker]:
        """
        Obtener todos los trackers registrados.
        """
        return self._registry
    
    def create_from_config(self, config: List[dict]) -> None:
        """
        Crear varios trackers a partir de una lista de configuración:
          config = [
            {"key": "player"},
            {"key": "ball"}
          ]
        """
        for item in config:
            key = item.get("key")
            if key is None:
                raise TrackerFactoryError("Cada config debe tener 'key'")
            kwargs = dict(item)
            kwargs.pop("key", None)
            self.create(key, **kwargs)