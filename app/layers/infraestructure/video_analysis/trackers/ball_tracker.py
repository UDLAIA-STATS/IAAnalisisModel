from app.layers.infraestructure.video_analysis.trackers.tracker import Tracker

import supervision as sv

class BallTracker(Tracker):
    
    
    def get_object_tracks(
        self, 
        detection_with_tracks: sv.Detections, 
        cls_names_inv: dict[str, int],
        frame_num: int,
        detection_supervision: sv.Detections,
        tracks: dict = {"players":[],"referees":[],"ball":[]}, 
        ):       
        for frame_detection in detection_supervision:
            bbox = frame_detection[0].tolist()
            cls_id = frame_detection[3]

            if cls_id == cls_names_inv['ball']:
                tracks["ball"][frame_num][1] = {"bbox":bbox}
 