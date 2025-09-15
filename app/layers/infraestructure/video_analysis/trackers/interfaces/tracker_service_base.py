from abc import ABC, abstractmethod
from pathlib import Path
import pickle
from typing import List, Type
from ultralytics import YOLO
from layers.domain.utils.singleton import AbstractSingleton
import supervision as sv

from cv2.typing import MatLike
from ultralytics.engine.results import Results
from layers.infraestructure.video_analysis.services import get_center_of_bbox, get_foot_position
from .tracker import Tracker

class TrackerServiceBase(metaclass=AbstractSingleton):
    def __init__(self, model_path: str):
        # Import locally to avoid circular import
        from layers.infraestructure.video_analysis.trackers.services import TrackerFactory
        
        self.model = YOLO(model_path)
        self.tracker = sv.ByteTrack()
        self.tracker_factory = TrackerFactory(self.model)
    
    @abstractmethod
    def get_object_tracks(
        self, 
        frames: list[MatLike], 
        tracks: dict | None = None, 
        read_from_stub: bool = False, 
        stub_path: str = ""
    ):
        if tracks is None:
            tracks = {"players": [], "ball": []}
        raise NotImplementedError
    
    def create_tracker(self, key: str, tracker_cls: Type[Tracker]) -> None:
        # Import locally to avoid circular import
        from layers.infraestructure.video_analysis.trackers.services import TrackerFactoryError
        
        try:
            self.tracker_factory.register(key, tracker_cls)
            self.tracker_factory.create(key)
        except TrackerFactoryError as e:
            print(f"Error creating tracker '{key}': {e}")
        
    
    def get_tracker(self, key: str) -> Tracker:
        # Import locally to avoid circular import
        from layers.infraestructure.video_analysis.trackers.services import TrackerFactoryError
        
        tracker = self.tracker_factory.get_trackers().get(key)
        if not tracker:
            raise TrackerFactoryError(f"Tracker '{key}' is not registered.")
        return tracker
    
    def get_trackers(self) -> List[Tracker]:
        return list(self.tracker_factory.get_trackers().values())
    
    def add_position_to_tracks(self, tracks: dict[str, list]):
        for entity, tracked_objects in tracks.items():
            for frame_num, track in enumerate(tracked_objects):
                for track_id, track_info in track.items():
                    bbox = track_info['bbox']
                    if entity == 'ball':
                        position= get_center_of_bbox(bbox)
                    else:
                        position = get_foot_position(bbox)
                    tracks[entity][frame_num][track_id]['position'] = position
                    
    
    def read_tracks_from_stub(self, stub_path: str) -> dict:
        tracks: dict = {"players":[],"referees":[],"ball":[]}
        if stub_path and Path(stub_path).exists():
            with open(stub_path,'rb') as f:
                tracks = pickle.load(f)
            return tracks
        print("Tracks are: ", tracks)
        return tracks
    
    def save_tracks_to_stub(self, tracks: dict, stub_path: str):
        if stub_path:
            with open(stub_path,'wb') as f:
                pickle.dump(tracks, f)
                
    def detect_frames(self, frames: list[MatLike], batch_size: int = 20) -> list[Results]:
        detections: list[Results] = [] 
        for i in range(0, len(frames), batch_size):
            detections_batch = self.model.predict(frames[i:i+batch_size], conf = 0.1)
            detections.extend(detections_batch)
        return detections