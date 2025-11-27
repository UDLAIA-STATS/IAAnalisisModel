import pickle
from abc import abstractmethod
from pathlib import Path
from typing import List, Type

import supervision as sv
from cv2.typing import MatLike
import torch
from app.entities.collections.track_collection import TrackCollection
from app.entities.tracks.track_detail import TrackDetailBase
from app.entities.utils.singleton import AbstractSingleton
from app.modules.services import (get_center_of_bbox)
from ultralytics import YOLO
from ultralytics.engine.results import Results

from .tracker_base import Tracker


class TrackerServiceBase(metaclass=AbstractSingleton):
    def __init__(self, model_path: str, use_half_precision: bool = False):
        # Import locally to avoid circular import
        from app.modules.trackers.tracker_factory import \
            TrackerFactory

        self.model = self.__load_detector(model_path, use_half_precision)
        self.tracker = sv.ByteTrack()
        self.tracker_factory = TrackerFactory(self.model)
        self.tracker_path = "bytetrack.yaml"
        
    def __load_detector(self, model_path: str, use_half_precision: bool = False) -> YOLO:
        model = YOLO(model=model_path, task='obb', verbose=True)
        
        if torch.cuda.is_available():
            model.to('cuda')
            try:
                if not use_half_precision: return model
                model.half()
            except Exception as e:
                print(f"Warning: Could not convert model to half precision: {e}")
        
        return model

    @abstractmethod
    def get_object_track(
        self,
        frames: MatLike,
        tracks_collection: TrackCollection,
    ):
        raise NotImplementedError

    def get_tracker(self, key: str) -> Tracker:
        from app.modules.trackers.tracker_factory import \
            TrackerFactoryError

        tracker = self.tracker_factory.get_trackers().get(key)
        if not tracker:
            raise TrackerFactoryError(f"Tracker '{key}' is not registered.")
        return tracker

    def get_trackers(self) -> List[Tracker]:
        return list(self.tracker_factory.get_trackers().values())

    def add_position_to_track(self, track_detail: TrackDetailBase) -> TrackDetailBase:
        bbox = track_detail.bbox
        position = get_center_of_bbox(bbox)
        track_detail.position = position
        return track_detail

    def detect_frames(
            self,
            frames: MatLike,
            conf: float = 0.1) -> list[Results]:
        """
        Detects objects in a frame using the YOLO model.

        Args:
        frames (MatLike): The frame to detect objects in.
        conf (float, optional): The confidence threshold for detection. Defaults to 0.1.

        Returns:
        list[Results]: The detected objects in the frame.
        """
        return self.model.predict(frames, conf=conf)
