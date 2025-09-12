from pathlib import Path
import pickle
from typing import override
from ultralytics import YOLO
import supervision as sv
from cv2.typing import MatLike
from ultralytics.engine.results import Results

from app.layers.domain.utils.singleton import Singleton
from app.layers.infraestructure.video_analysis.services.bbox_processor_service import get_center_of_bbox, get_foot_position
from app.layers.infraestructure.video_analysis.trackers.tracker import Tracker
from app.layers.infraestructure.video_analysis.trackers.tracker_service_base import TrackerServiceBase


class TrackerService(TrackerServiceBase):
    
    def __init__(self, model_path: str):
        super().__init__(model_path)  
        
    @override
    def get_object_tracks(
        self, 
        frames: list[MatLike], 
        tracks: dict | None = None, 
        read_from_stub: bool = False, 
        stub_path: str = ""
    ):
        if read_from_stub and stub_path:
            tracks = self.read_tracks_from_stub(stub_path)
        
        if tracks is None:
            tracks = {"players": [], "referees": [], "ball": []}
        
        detections = self.detect_frames(frames)

        for frame_num, detection in enumerate(detections):
            cls_names = detection.names
            cls_names_inv = {v:k for k,v in cls_names.items()}
            print(cls_names_inv)

            # Covert to supervision Detection format
            detection_supervision = sv.Detections.from_ultralytics(detection)

            print(detection_supervision.data.keys())
                    
            # Track Objects 
            detection_with_tracks = self.tracker.update_with_detections(detection_supervision)

            tracks["players"].append({})
            tracks["ball"].append({})

            for _, val in self.trackers.items():
                track = val.get_object_tracks(
                    detection_with_tracks=detection_with_tracks,
                    cls_names_inv=cls_names_inv,
                    frame_num=frame_num,
                    detection_supervision=detection_supervision,
                    tracks=tracks
                )

        if stub_path is not None:
            with open(stub_path,'wb') as f:
                pickle.dump(tracks, f)

        return tracks