from typing import Dict, List, Type

from app.entities.interfaces.tracker_base import Tracker
from ultralytics import YOLO

from app.entities.trackers.ball_tracker import BallTracker
from app.entities.trackers.player_tracker import PlayerTracker


class TrackerFactoryError(Exception):
    pass

class TrackerFactory:
    def __init__(self, model: YOLO):
        """
        Factory que crea instancias de trackers
        usando un único modelo YOLO compartido.
        """
        self._registry: Dict[str, Tracker] = {}
        self.create_defaul_trackers()
        self.model = model
        
    def create_defaul_trackers(self) -> None:
        """
        Registrar los trackers por defecto: 'player' y 'ball'
        """
        self.register("player", PlayerTracker)
        self.register("ball", BallTracker)

    def register(self, key: str, tracker_cls: Type[Tracker]) -> None:
        """
        Registrar una clase tracker con una clave (ej: 'player', 'ball').
        """
        if key in self._registry:
            raise TrackerFactoryError(
                f"Tracker '{key}' is already registered.")
        self._registry[key] = tracker_cls(self.model)

    def get_trackers(self) -> Dict[str, Tracker]:
        """
        Obtener todos los trackers registrados.
        """
        return self._registry