from layers.infraestructure.video_analysis.trackers.interfaces import Tracker
import supervision as sv

class PlayerTracker(Tracker):
    
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
            
        for frame_detection in detection_with_tracks:
                bbox = frame_detection[0].tolist()
                cls_id = frame_detection[3]
                track_id = frame_detection[4]

                if cls_id == cls_names_inv['player']:
                    tracks["players"][frame_num][track_id] = {"bbox":bbox}
            
 