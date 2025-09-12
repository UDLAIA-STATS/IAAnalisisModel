from abc import ABC, abstractmethod
import gc
import pathlib
import pickle
import sys

import cv2
import numpy as np
import pandas as pd
import supervision as sv
from cv2.typing import MatLike
from ultralytics import YOLO
from ultralytics.engine.results import Results

from layers.infraestructure.video_analysis.services.bbox_processor_service import (
    get_bbox_width, get_center_of_bbox, get_foot_position)

class Tracker(ABC):
    def __init__(self, model_path: str):
        self.model = YOLO(model_path)
        self.tracker = sv.ByteTrack()
        #self.metric = nn_matching.NearestNeighborDistanceMetric("cosine", 0.2, None)
        #self.tracker = DeepSortTracker() 
    
    @abstractmethod
    def get_object_tracks(
        self, 
        detection_with_tracks: sv.Detections, 
        cls_names_inv: dict[str, int],
        frame_num: int,
        detection_supervision: sv.Detections,
        tracks: dict = {"players":[],"referees":[],"ball":[]}) -> None:
        pass
    
    def read_tracks_from_stub(self, stub_path: str) -> dict:
        tracks: dict = {"players":[],"referees":[],"ball":[]}
        if stub_path is not None and pathlib.Path(stub_path).exists():
            with open(stub_path,'rb') as f:
                tracks = pickle.load(f)
            return tracks
        print("Tracks are: ", tracks)
        return tracks
    
    def save_tracks_to_stub(self, tracks: dict, stub_path: str):
        if stub_path is not None:
            with open(stub_path,'wb') as f:
                pickle.dump(tracks, f)
    

    def detect_frames(self, frames: list[MatLike]):
        batch_size=20 
        detections: list[Results] = [] 
        for i in range(0, len(frames), batch_size):
            detections_batch = self.model.predict(frames[i:i+batch_size], conf = 0.1)
            detections += detections_batch
        return detections
    
    def draw_ellipse(self, frame, bbox, color, track_id = None):
        y2 = int(bbox[3])
        x_center, _ = get_center_of_bbox(bbox)
        width = get_bbox_width(bbox)

        cv2.ellipse(
            frame,
            center=(x_center,y2),
            axes=(int(width), int(0.35*width)),
            angle=0.0,
            startAngle=-45,
            endAngle=235,
            color = color,
            thickness=2,
            lineType=cv2.LINE_4
        )

        rectangle_width = 40
        rectangle_height=20
        x1_rect = x_center - rectangle_width//2
        x2_rect = x_center + rectangle_width//2
        y1_rect = (y2- rectangle_height//2) +15
        y2_rect = (y2+ rectangle_height//2) +15

        if track_id is not None:
            cv2.rectangle(frame,
                          (int(x1_rect),int(y1_rect) ),
                          (int(x2_rect),int(y2_rect)),
                          color,
                          cv2.FILLED)
            
            x1_text = x1_rect+12
            if track_id > 99:
                x1_text -=10
            
            cv2.putText(
                frame,
                f"{track_id}",
                (int(x1_text),int(y1_rect+15)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (0,0,0),
                2
            )

        return frame

    def draw_triangle(self, frame, bbox, color):
        y= int(bbox[1])
        x,_ = get_center_of_bbox(bbox)

        triangle_points = np.array([
            [x,y],
            [x-10,y-20],
            [x+10,y-20],
        ])
        cv2.drawContours(frame, [triangle_points],0,color, cv2.FILLED)
        cv2.drawContours(frame, [triangle_points],0,(0,0,0), 2)

        return frame
    
    