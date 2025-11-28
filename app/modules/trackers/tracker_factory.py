from typing import Dict, Type

from app.entities.interfaces.tracker_base import Tracker
from ultralytics import YOLO

from app.entities.trackers.ball_tracker import BallTracker
from app.entities.trackers.player_tracker import PlayerTracker


class TrackerFactoryError(Exception):
    """Errores relacionados con la creación y registro de trackers."""
    pass


class TrackerFactory:
    """
    Crea e inicializa todos los trackers del sistema.
    Cada tracker comparte el modelo YOLO principal, pero NO el tracker (ByteTrack),
    ya que este pertenece a TrackerService.
    """

    def __init__(self, model: YOLO):
        """
        Inicializa el factory con el modelo YOLO compartido y crea
        los trackers por defecto.
        """
        self.model = model  # Primero asignamos el modelo
        self._registry: Dict[str, Tracker] = {}

        # Luego creamos los trackers por defecto
        self._create_default_trackers()

    # ---------------------------------------------------------
    # Registro básico
    # ---------------------------------------------------------

    def _create_default_trackers(self) -> None:
        """
        Registrar los trackers por defecto: 'player' y 'ball'.
        """
        self._register("player", PlayerTracker)
        self._register("ball", BallTracker)

    def _register(self, key: str, tracker_cls: Type[Tracker]) -> None:
        """
        Registrar una clase tracker bajo una clave única.
        """
        if key in self._registry:
            raise TrackerFactoryError(
                f"Tracker '{key}' is already registered."
            )

        # Instancia del tracker con el modelo compartido
        self._registry[key] = tracker_cls()

    # ---------------------------------------------------------
    # Acceso a los trackers
    # ---------------------------------------------------------

    def get_trackers(self) -> Dict[str, Tracker]:
        """
        Obtener todos los trackers registrados.
        """
        return self._registry

    def get_tracker(self, key: str) -> Tracker:
        """
        Obtener un tracker por clave.
        """
        tracker = self._registry.get(key)
        if not tracker:
            raise TrackerFactoryError(f"Tracker '{key}' no está registrado.")
        return tracker

    # ---------------------------------------------------------
    # Métodos adicionales útiles
    # ---------------------------------------------------------

    def reset_all(self) -> None:
        """
        Permite reiniciar todos los trackers.
        Útil para escenarios de streaming donde el flujo se corta.
        """
        for tracker in self._registry.values():
            tracker.reset()
