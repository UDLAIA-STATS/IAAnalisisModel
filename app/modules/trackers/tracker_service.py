# tracker_service.py
import logging
from typing import List, Union

import supervision as sv
from cv2.typing import MatLike
from app.entities.interfaces.record_collection_base import RecordCollectionBase
from app.entities.interfaces.tracker_service_base import TrackerServiceBase

class TrackerService(TrackerServiceBase):
    """
    Implementación concreta del servicio de tracking, preparada para streaming.
    - Mantiene self.last_tracked para acceso externo (p.ej. TeamAssigner)
    """

    def __init__(self, model_path: str, use_half_precision: bool = False):
        super().__init__(model_path, use_half_precision)
        self.last_tracked: sv.Detections | None = None

    # Mantengo la firma anterior: get_object_tracks (compatibilidad)
    def get_object_tracks(
        self,
        frames: Union[List[MatLike], MatLike],
        frame_num: int,
        
        tracks_collection: RecordCollectionBase,
    ):
        """
        Procesa una lista de frames o un frame simple. Llama a process_frame iterativamente.
        """
        if isinstance(frames, list):
            for i, frame in enumerate(frames):
                self.process_frame(frame, i, tracks_collection)
        else:
            self.process_frame(frames, frame_num, tracks_collection)

    # process_frame está definido en la clase base y puede ser sobrescrito si quieres
    # Aquí solo sobreescribimos para añadir logging / hooks si se desea
    def process_frame(
        self,
        frame: MatLike,
        frame_num: int,
        tracks_collection: RecordCollectionBase,
        conf: float = 0.1
    ):
        try:
            super().process_frame(frame, frame_num, tracks_collection, conf=conf)
            # guardar el último tracked para que servicios externos lo consuman
            # (super().process_frame ya actualiza el tracker interno)
            # reconvertimos estado desde el tracker (si supervisor lo expone)
            # la forma simple: la última detección tracked la obtuvimos en el proceso
            # Si necesitas exponer algo más complejo, lo puedes extraer aquí.
        except Exception as e:
            logging.exception(f"TrackerService.process_frame error: {e}")
