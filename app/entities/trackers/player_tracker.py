from pathlib import Path
import supervision as sv
from app.entities.collections.track_collection import TrackCollection
from app.entities.tracks.track_detail import TrackPlayerDetail
from app.entities.interfaces.tracker_base import Tracker

class PlayerTracker(Tracker):

    def __init__(self, model):
        super().__init__(model)

    def get_object_tracks(
        self,
        detection_with_tracks: sv.Detections,
        cls_names_inv: dict[str, int],
        frame_num: int,
        detection_supervision: sv.Detections,
        tracks_collection: TrackCollection
    ):
        bbox = detection_with_tracks.xyxy
        class_ids = detection_with_tracks.class_id
        track_ids = detection_with_tracks.tracker_id
        player_mask = class_ids == cls_names_inv['player']

        if track_ids is not None and track_ids.any() and player_mask.any() and player_mask.any():
            player_bboxes = bbox[player_mask]
            player_ids = track_ids[player_mask]

            for bbox, track_id in zip(player_bboxes, player_ids):
                print("Track ID is: ", track_id)
                track = TrackPlayerDetail(bbox=bbox.tolist(), track_id=int(track_id))
                tracks_collection.update_track(
                    entity_type="players",
                    frame_num=frame_num,
                    track_id=int(track_id),
                    track_detail=track)

        # for frame_detection in detection_with_tracks:
        #     bbox = frame_detection[0].tolist()
        #     cls_id = frame_detection[3]
        #     track_id = frame_detection[4]

        #     if cls_id == cls_names_inv['player']:
        #         tracks["players"][frame_num][track_id] = {"bbox": bbox}
