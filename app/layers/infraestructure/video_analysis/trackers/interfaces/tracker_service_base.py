import pickle
from abc import abstractmethod
from pathlib import Path
from typing import List, Type

import supervision as sv
from cv2.typing import MatLike
from app.layers.domain import tracks
from app.layers.domain.collections.track_collection import TrackCollection
from app.layers.domain.tracks.track_detail import TrackBallDetail, TrackPlayerDetail
from app.layers.domain.utils.singleton import AbstractSingleton
from app.layers.infraestructure.video_analysis.services import (get_center_of_bbox,
                                                            get_foot_position)
from ultralytics import YOLO
from ultralytics.engine.results import Results

from .tracker import Tracker


class TrackerServiceBase(metaclass=AbstractSingleton):
    def __init__(self, model_path: str):
        # Import locally to avoid circular import
        from app.layers.infraestructure.video_analysis.trackers.services import \
            TrackerFactory

        self.model = YOLO(model=model_path, task='obb', verbose=True)
        self.tracker = sv.ByteTrack()
        self.tracker_factory = TrackerFactory(self.model)
        self.tracker_path = "bytetrack.yaml"

    @abstractmethod
    def get_object_tracks(
        self,
        frames: list[MatLike],
        tracks_collection: TrackCollection,
        read_from_stub: bool = False,
        stub_path: str = ""
    ):
        raise NotImplementedError

    def create_tracker(self, key: str, tracker_cls: Type[Tracker]) -> None:
        # Import locally to avoid circular import
        from app.layers.infraestructure.video_analysis.trackers.services import \
            TrackerFactoryError

        try:
            self.tracker_factory.register(key, tracker_cls)
            self.tracker_factory.create(key)
        except TrackerFactoryError as e:
            print(f"Error creating tracker '{key}': {e}")

    def get_tracker(self, key: str) -> Tracker:
        # Import locally to avoid circular import
        from app.layers.infraestructure.video_analysis.trackers.services import \
            TrackerFactoryError

        tracker = self.tracker_factory.get_trackers().get(key)
        if not tracker:
            raise TrackerFactoryError(f"Tracker '{key}' is not registered.")
        return tracker

    def get_trackers(self) -> List[Tracker]:
        return list(self.tracker_factory.get_trackers().values())

    def add_position_to_tracks(self, tracks_collection: TrackCollection):
        for entity_type, frames in tracks_collection.tracks.items():
            for frame_num, tracks_in_frames in frames.items():
                for track_id, track_detail in tracks_in_frames.items():
                    bbox = track_detail.bbox
                    position = get_center_of_bbox(bbox)
                    track_detail.position = position
                    tracks_collection.update_track(
                        entity_type=entity_type,
                        frame_num=frame_num,
                        track_id=track_id,
                        track_detail=track_detail
                    )


    def read_tracks_from_stub(self, stub_path: str) -> dict:
        tracks: dict = {"players": [], "ball": []}
        if stub_path and Path(stub_path).exists():
            with open(stub_path, 'rb') as f:
                tracks = pickle.load(f)
            return tracks
        print("Tracks are: ", tracks)
        return tracks

    def save_tracks_to_stub(self, tracks: dict, stub_path: str):
        if stub_path:
            with open(stub_path, 'wb') as f:
                pickle.dump(tracks, f)

    def detect_frames(
            self,
            frames: List[MatLike],
            batch_size: int = 20,
            conf: float = 0.1) -> list[Results]:
        """Divide los frames en lotes y obtiene detecciones con el modelo YOLO."""
        detections: list[Results] = []
        for i in range(0, len(frames), batch_size):
            batch = frames[i:i + batch_size]
            detections_batch = self.model.predict(
                batch, conf=conf)
            detections.extend(detections_batch)
        return detections
