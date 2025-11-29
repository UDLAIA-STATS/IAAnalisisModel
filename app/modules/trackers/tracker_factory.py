from typing import Dict, Type
from app.entities.interfaces.tracker_base import Tracker
from ultralytics.models import YOLO

from app.entities.utils.singleton import Singleton

class TrackerFactoryError(Exception):
    pass

class TrackerFactory(metaclass=Singleton):
    def __init__(self, model: YOLO):
        self.model = model
        # registramos clases (no instancias)
        self._registry_classes: Dict[str, Type[Tracker]] = {}
        # cache para instancias creadas (lazy)
        self._instances: Dict[str, Tracker] = {}
        self._create_default_trackers()

    def _create_default_trackers(self) -> None:
        # Importar aquí para evitar circular imports al nivel module
        from app.entities.trackers.player_tracker import PlayerTracker
        from app.entities.trackers.ball_tracker import BallTracker

        self._register_class("player", PlayerTracker)
        self._register_class("ball", BallTracker)

    def _register_class(self, key: str, tracker_cls: Type[Tracker]) -> None:
        if key in self._registry_classes:
            raise TrackerFactoryError(f"Tracker '{key}' is already registered.")
        self._registry_classes[key] = tracker_cls

    def _instantiate(self, key: str) -> Tracker:
        if key in self._instances:
            return self._instances[key]
        cls = self._registry_classes.get(key)
        if cls is None:
            raise TrackerFactoryError(f"No tracker class registered for '{key}'.")
        # Intentar pasar model si el constructor lo acepta, fallback a no args.
        try:
            inst = cls(self.model)
        except TypeError:
            inst = cls(self.model)
        self._instances[key] = inst
        return inst

    def get_trackers(self) -> Dict[str, Tracker]:
        # crea todas las instancias si no existen
        for key in list(self._registry_classes.keys()):
            if key not in self._instances:
                self._instantiate(key)
        return dict(self._instances)

    def get_tracker(self, key: str) -> Tracker:
        if key in self._instances:
            return self._instances[key]
        return self._instantiate(key)

    def reset_all(self) -> None:
        for tracker in self.get_trackers().values():
            try:
                tracker.reset()
            except Exception:
                pass
