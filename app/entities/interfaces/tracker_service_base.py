# tracker_service_base.py
import logging
from typing import List, Union, TYPE_CHECKING

import supervision as sv
from cv2.typing import MatLike
import torch
from ultralytics.models import YOLO


from app.entities.utils.singleton import AbstractSingleton
from app.layers.infraestructure.video_analysis.trackers.entities.ball_tracker import BallTracker

if TYPE_CHECKING:
    from app.modules.trackers.tracker_factory import TrackerFactory


class TrackerServiceBase(metaclass=AbstractSingleton):
    """
    Servicio base para detección + tracking.
    - Carga detector (YOLO)
    - Mantiene un ByteTrack interno (self.tracker) para continuidad entre frames
    - Provee métodos streaming: process_frame (1 frame) y get_object_tracks (compatibilidad con listas)
    """

    def __init__(self, model_path: str, use_half_precision: bool = False):
        from app.modules.trackers.tracker_factory import TrackerFactory
        self.model = self.__load_detector(model_path, use_half_precision)
        self.tracker = sv.ByteTrack()
        self.tracker_factory = TrackerFactory(self.model)
        self.tracker_path = "bytetrack.yaml"
        self._device = 'cuda' if torch.cuda.is_available() else 'cpu'
        logging.info(f"TrackerServiceBase initialized on device={self._device}")

    def __load_detector(self, model_path: str, use_half_precision: bool = False) -> YOLO:
        model = YOLO(model=model_path, task='obb', verbose=False)
        if torch.cuda.is_available():
            model.to('cuda')
            if use_half_precision:
                try:
                    model.half()
                except Exception as e:
                    logging.warning(f"Could not enable half precision: {e}")
        return model



    def reset_tracking(self) -> None:
        """
        Reinicia el tracker (p. ej. cuando se corta el stream o hay un gap grande)
        """
        self.tracker = sv.ByteTrack()
        logging.info("ByteTrack reset.")

    def detect_frames(self, frames: Union[List[MatLike], MatLike], conf: float = 0.1) -> list:
        """
        Wrapper sobre el detector. Acepta un frame o lista de frames y devuelve lista de Results.
        Nota: el modelo YOLO puede aceptar listas y devolver lista de Results.
        """
        print("Detectando en frames...")
        return self.model.predict(frames, conf=conf)
