import numpy as np

from app.layers.domain.collections.track_collection import TrackCollection


def check_speed_consistency(tracks_collecion: TrackCollection):
    tracks = tracks_collecion.tracks
    results = {"players": 0}
    track_speeds = {}

    for frame_num, frame_tracks in tracks["players"].items():
        for track_id, track_detail in frame_tracks.items():
            if track_detail.speed_km_per_hour is not None:
                if track_id not in track_speeds:
                    track_speeds[track_id] = []
                track_speeds[track_id].append(track_detail.speed_km_per_hour)

    for track_id, speeds in track_speeds.items():
        if len(speeds) < 2:
            continue

        accelerations = np.abs(np.diff(speeds))
        if np.any(accelerations > 15):
            results["players"] += 1



    return results
