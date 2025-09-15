import pandas as pd
from layers.infraestructure.video_analysis.trackers.interfaces import Tracker

import supervision as sv

class BallTracker(Tracker):
    
    
    def get_object_tracks(
        self, 
        detection_with_tracks: sv.Detections, 
        cls_names_inv: dict[str, int],
        frame_num: int,
        detection_supervision: sv.Detections,
        tracks: dict | None = None 
        ):       
        if tracks is None:
            tracks = {"players": [], "ball": []}
            
        for frame_detection in detection_supervision:
            bbox = frame_detection[0].tolist()
            cls_id = frame_detection[3]

            if cls_id == cls_names_inv['ball']:
                tracks["ball"][frame_num][1] = {"bbox":bbox}
 
 
    def interpolate_ball_positions(self, ball_positions):
            ball_positions = [pos.get(1, {}).get('bbox', []) for pos in ball_positions]
            
            # Create a DataFrame to handle missing values
            df_ball = pd.DataFrame(
                ball_positions, 
                columns=['x1', 'y1', 'x2', 'y2']
                )
            
            # Interpolate middle gaps, then fill leading/trailing NaNs
            df_ball = df_ball.interpolate(limit_direction='both')

            ball_positions = [ 
                {1: {"bbox": row}} 
                for row in df_ball.to_numpy().tolist()
                ]

            return ball_positions