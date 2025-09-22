from typing import Dict
import numpy as np
from app.layers.domain.tracks.track_detail import TrackDetailBase
from app.layers.infraestructure.video_analysis.trackers.entities.ball_tracker import \
    BallTracker

def calculate_bbox_center(bbox):
        return np.array([(bbox[0] + bbox[2]) / 2, (bbox[1] + bbox[3]) / 2])

def calculate_interpolation_error(tracker: BallTracker, original_tracks: Dict[int, Dict[int, TrackDetailBase]]):
    interpolated_tracks = tracker.interpolate_ball_positions(original_tracks)
    errors = []

    for frame_num, track_content in sorted(original_tracks.items()):
        if frame_num not in interpolated_tracks:
            continue

        for track_id, track_detail in track_content.items():
            if track_id == 1:  # solo bola
                original_bbox = track_detail.bbox
                interpolated_bbox = interpolated_tracks[frame_num][1]["bbox"]

                if (
                    original_bbox is None or interpolated_bbox is None
                    or len(original_bbox) != 4 or len(interpolated_bbox) != 4
                ):
                    continue

                error = np.sum((calculate_bbox_center(original_bbox) - calculate_bbox_center(interpolated_bbox)) ** 2)
                errors.append(error)

    return float(np.mean(errors)) if errors else 0.0



    # for orig_frame, interp_frame in zip(original_tracks, interpolated_tracks):
    #     if 1 in orig_frame and 1 in interp_frame:
    #         orig_bbox = orig_frame[1]["bbox"]
    #         interp_bbox = interp_frame[1]["bbox"]
    #         orig_center = np.array([
    #             (orig_bbox[0] + orig_bbox[2]) / 2,
    #             (orig_bbox[1] + orig_bbox[3]) / 2
    #         ])
    #         interp_center = np.array([
    #             (interp_bbox[0] + interp_bbox[2]) / 2,
    #             (interp_bbox[1] + interp_bbox[3]) / 2
    #         ])
    #         error = np.sum((orig_center - interp_center) ** 2)
    #         errors.append(error)

    # return np.mean(errors) if errors else 0.0
