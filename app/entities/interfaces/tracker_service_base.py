# tracker_service_base.py
import logging
from abc import abstractmethod
from typing import List, Optional, Union

import supervision as sv
from cv2.typing import MatLike
import torch
from ultralytics import YOLO

from app.entities.utils.singleton import AbstractSingleton
from app.modules.services import get_center_of_bbox
from app.entities.tracks.track_detail import TrackDetailBase
from app.entities.interfaces.record_collection_base import RecordCollectionBase
from app.modules.trackers.tracker_factory import TrackerFactory


class TrackerServiceBase(metaclass=AbstractSingleton):
    """
    Servicio base para detección + tracking.
    - Carga detector (YOLO)
    - Mantiene un ByteTrack interno (self.tracker) para continuidad entre frames
    - Provee métodos streaming: process_frame (1 frame) y get_object_tracks (compatibilidad con listas)
    """

    def __init__(self, model_path: str, use_half_precision: bool = False):
        self.model = self.__load_detector(model_path, use_half_precision)
        self.tracker = sv.ByteTrack()
        self.tracker_factory = TrackerFactory(self.model)
        # Por compatibilidad con implementaciones previas
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

    # ---------------------------
    # Métodos que puedes sobreescribir en la subclase
    # ---------------------------

    @abstractmethod
    def get_object_track(
        self,
        frames: MatLike,
        tracks_collection: RecordCollectionBase,
    ):
        """
        Método legacy (no forzado a frame). Lo dejo abstracto para compatibilidad.
        """
        raise NotImplementedError

    # ---------------------------
    # Utilitarios streaming
    # ---------------------------

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
        return self.model.predict(frames, conf=conf)

    # ---------------------------
    # API principal para streaming (1 frame por llamada)
    # ---------------------------
    def process_frame(
        self,
        frame: MatLike,
        frame_num: int,
        tracks_collection: RecordCollectionBase,
        conf: float = 0.1,
    ) -> None:
        """
        Procesa un único frame en modo streaming:
        1) detecta
        2) convierte a supervision
        3) actualiza ByteTrack
        4) distribuye a trackers registrados
        """

        # 1) Detectar (modelo retorna lista de Results aunque pasemos single frame)
        results = self.detect_frames([frame], conf=conf)
        if not results:
            return

        # results[0] es la detección del frame
        result = results[0]

        # 2) map de clases
        cls_names = getattr(result, "names", {})
        cls_names_inv = {v: k for k, v in cls_names.items()}

        # 3) convertir a supervision
        det_sv = sv.Detections.from_ultralytics(result)

        # 4) tracking (ByteTrack) — devuelve detections con track_id
        tracked = self.tracker.update_with_detections(det_sv)

        # 5) distribuir a trackers concretos
        for tracker in self.get_trackers():
            try:
                tracker.get_object_tracks(
                    detection_with_tracks=tracked,
                    cls_names_inv=cls_names_inv,
                    frame_num=frame_num,
                    detection_supervision=det_sv,
                    tracks_collection=tracks_collection
                )
            except Exception as e:
                logging.exception(f"Error executing tracker {tracker}: {e}")

    # ---------------------------
    # Compatibilidad: acepta lista de frames (batch) y llama a process_frame por cada uno
    # ---------------------------
    def get_object_tracks(
        self,
        frames: Union[List[MatLike], MatLike],
        tracks_collection: RecordCollectionBase,
    ):
        """
        Compatibilidad con API anterior que pasaba una lista de frames.
        En streaming, se le puede pasar un único frame.
        """
        if isinstance(frames, list):
            for i, frame in enumerate(frames):
                self.process_frame(frame, i, tracks_collection)
        else:
            # Un solo frame — mantenemos frame_num = 0 si no se especifica
            self.process_frame(frames, 0, tracks_collection)

    # ---------------------------
    # Helpers existentes
    # ---------------------------
    def get_tracker(self, key: str):
        from app.modules.trackers.tracker_factory import TrackerFactoryError

        tracker = self.tracker_factory.get_trackers().get(key)
        if not tracker:
            raise TrackerFactoryError(f"Tracker '{key}' is not registered.")
        return tracker

    def get_trackers(self) -> List:
        return list(self.tracker_factory.get_trackers().values())

    def add_position_to_track(self, track_detail: TrackDetailBase) -> TrackDetailBase:
        bbox = track_detail.bbox
        position = get_center_of_bbox(bbox)
        track_detail.position = position
        return track_detail
