from abc import ABC, abstractmethod
from typing import Dict

import cv2
import numpy as np
import supervision as sv
from cv2.typing import MatLike
from app.entities.collections.track_collection import TrackCollection
from app.entities.tracks.track_detail import TrackDetailBase, TrackPlayerDetail

from ultralytics import YOLO
from ultralytics.engine.results import Results


class Tracker(ABC):
    def __init__(self, model: YOLO):
        self.model = model
        self.tracker = sv.ByteTrack()
        # self.metric = nn_matching.NearestNeighborDistanceMetric("cosine", 0.2, None)
        # self.tracker = DeepSortTracker()

    @abstractmethod
    def get_object_tracks(
            self,
            detection_with_tracks: sv.Detections,
            cls_names_inv: dict[str, int],
            frame_num: int,
            detection_supervision: sv.Detections,
            tracks_collection: TrackCollection) -> None:
        raise NotImplementedError

    def detect_frames(self, frame: MatLike):
        return self.model.predict(frame, conf=0.1)
        