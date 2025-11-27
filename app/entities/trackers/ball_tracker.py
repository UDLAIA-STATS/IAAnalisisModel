from typing import Dict, Hashable
import pandas as pd
import supervision as sv
from app.entities.collections.track_collection import TrackCollection
from app.entities.tracks.track_detail import TrackBallDetail, TrackDetailBase
from app.entities.interfaces.tracker_base import Tracker


class BallTracker(Tracker):
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
        ball_mask = class_ids == cls_names_inv['ball']

        if ball_mask is not None and track_ids is not None and track_ids.any() and ball_mask.any():
            ball_bbox = bbox[ball_mask][0].tolist()
            ball_id = 1
            track = TrackBallDetail(bbox=ball_bbox, track_id=int(ball_id))
            tracks_collection.update_track(
                frame_num=frame_num,
                track_id=int(ball_id),
                track_detail=track,
                entity_type="ball")

        # for frame_detection in detection_supervision:
        #     bbox = frame_detection[0].tolist()
        #     cls_id = frame_detection[3]

        #     if cls_id == cls_names_inv['ball']:
        #         tracks["ball"][frame_num][1] = {"bbox": bbox}

    def interpolate_ball_positions(
            self,
            ball_tracks: Dict[int, Dict[int, TrackDetailBase]]
    ) -> Dict[Hashable, Dict[int, Dict[str, list]]]:
        """
    Interpola posiciones del balón entre frames perdidos.

    Args:
        ball_tracks: dict con estructura {frame_num: {track_id: TrackBallDetail}}
        max_gap: máximo número de frames consecutivos sin detección que se interpolan
    """
    #     positions = {
    #     frame: track[1].bbox
    #     for frame, track in ball_tracks.items()
    #     if 1 in track and track[1].bbox is not None
    # }
        
        ball_positions = {}
        frame_indices = []
        for frame_num, tracks_in_frame in ball_tracks.items():
            for track_id, track in tracks_in_frame.items():
                ball_positions[frame_num] = track.bbox
                frame_indices.append(frame_num)

        # DataFrame con índice de frames
        df_ball = pd.DataFrame.from_dict(
            ball_positions, orient="index", columns=["x1", "y1", "x2", "y2"]
        ).sort_index()

        # Interpolación
        df_ball = df_ball.interpolate(limit_direction="both")

        # Reconstruir como dict indexado por frame
        interpolated_tracks = {
            frame: {1: {"bbox": row.tolist()}}
            for frame, row in df_ball.iterrows()
        }
        return interpolated_tracks

        # ball_positions= []
        # for frame_num, tracks_in_frame in ball_tracks.items():
        #     for tracks in tracks_in_frame.values():
        #         print("Bbox is of type: ", type(tracks.bbox))
        #         ball_positions.append(tracks.bbox)

        # # Create a DataFrame to handle missing values
        # df_ball = pd.DataFrame(
        #     ball_positions,
        #     columns=['x1', 'y1', 'x2', 'y2']
        # )

        # # Interpolate middle gaps, then fill leading/trailing NaNs
        # df_ball = df_ball.interpolate(limit_direction='both')

        # ball_positions = [
        #     {1: {"bbox": row}}
        #     for row in df_ball.to_numpy().tolist()
        # ]

        # return ball_positions
