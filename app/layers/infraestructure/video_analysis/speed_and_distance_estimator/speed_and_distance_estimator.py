import gc
from typing import Dict

import cv2
from cv2.typing import MatLike
from app.layers.domain.collections.track_collection import TrackCollection
from app.layers.domain.tracks.track_detail import TrackBallDetail, TrackDetailBase, TrackPlayerDetail
from app.layers.infraestructure.video_analysis.services.bbox_processor_service import (
    get_foot_position, measure_scalar_distance)


class SpeedAndDistanceEstimator():
    def __init__(self):
        self.frame_window = 5
        self.frame_rate = 24

    def add_speed_and_distance_to_tracks(
            self,
            tracks_collection: TrackCollection):
        total_distance = {}
        print("Calculating speed and distance...")
        for entity_type, frames in tracks_collection.tracks.items():
            number_of_frames = len(frames)
            print("Number of frames on entity: ", number_of_frames)
            for frame_num, tracks_in_frames in frames.items():
                last_frame = min(
                    frame_num + self.frame_window,
                    number_of_frames - 1)
                print(f"Processing frame {frame_num} to {last_frame}...")
                for track_id, track_detail in tracks_in_frames.items():
                    if track_id == track_detail.track_id:
                        continue

                    start_position = tracks_collection.tracks[entity_type][frame_num][track_id].position_transformed
                    end_position = tracks_collection.tracks[entity_type][last_frame][track_id].position_transformed
                    print("Start position:", start_position)
                    print("End position:", end_position)

                    if not start_position or not end_position:
                        continue

                    distance_covered = measure_scalar_distance(
                        start_position, end_position)
                    time_elapsed = (last_frame - frame_num) / self.frame_rate
                    if time_elapsed <= 0:
                        continue
                    speed_meters_per_sec = distance_covered / time_elapsed
                    speed_km_per_hour = speed_meters_per_sec * 3.6

                    if entity_type not in total_distance:
                        total_distance[entity_type] = {}

                    if track_id not in total_distance[entity_type]:
                        total_distance[entity_type][track_id] = 0

                    total_distance[entity_type][track_id] += distance_covered

                    print(
                        f"""Track ID: {track_id}, Speed: {
                            speed_km_per_hour:.2f} km/h, Total Distance: {
                            total_distance[entity_type][track_id]:.2f} m""")
                    for frame_num_batch in range(frame_num, last_frame):
                        if track_id not in frames[frame_num_batch]:
                            continue

                        track = tracks_collection.tracks[entity_type][frame_num_batch][track_id]
                        track.update(
                            speed_km_per_hour=speed_km_per_hour,
                            covered_distance=total_distance[entity_type][track_id])
                        # track.speed_km_per_hour = speed_km_per_hour
                        # track.covered_distance = total_distance[entity_type][track_id]
                        tracks_collection.update_track(
                            entity_type=entity_type,
                            frame_num=frame_num_batch,
                            track_id=track_id,
                            track_detail=track
                        )

        # for tracked_object, object_tracks in tracks.items():
        #     number_of_frames = len(object_tracks)
        #     for frame_num in range(0, number_of_frames, self.frame_window):
        #         last_frame = min(
        #             frame_num + self.frame_window,
        #             number_of_frames - 1)

        #         for track_id, _ in object_tracks[frame_num].items():
        #             if track_id not in object_tracks[last_frame]:
        #                 continue

        #             start_position = object_tracks[frame_num][track_id]['position_transformed']
        #             end_position = object_tracks[last_frame][track_id]['position_transformed']
        #             print("Start position:", start_position)
        #             print("End position:", end_position)

        #             if not start_position or not end_position:
        #                 continue

        #             distance_covered = measure_scalar_distance(
        #                 start_position, end_position)
        #             time_elapsed = (last_frame - frame_num) / self.frame_rate
        #             if time_elapsed <= 0:
        #                 continue
        #             speed_meteres_per_second = distance_covered / time_elapsed
        #             speed_km_per_hour = speed_meteres_per_second * 3.6

        #             if tracked_object not in total_distance:
        #                 total_distance[tracked_object] = {}

        #             if track_id not in total_distance[tracked_object]:
        #                 total_distance[tracked_object][track_id] = 0

        #             total_distance[tracked_object][track_id] += distance_covered

        #             for frame_num_batch in range(frame_num, last_frame):
        #                 if track_id not in tracks[tracked_object][frame_num_batch]:
        #                     continue

        #                 if tracked_object == "ball":
        #                     ball_tracker = TrackBallDetail(
        #                         speed_km_per_hour=speed_km_per_hour,
        #                         covered_distance=total_distance[tracked_object][track_id])
        #                     tracks_collection.update_track(
        #                         entity_type="ball",
        #                         frame_num=frame_num_batch,
        #                         track_id=track_id,
        #                         track_detail=ball_tracker
        #                     )
        #                 else:
        #                     player_tracker = TrackPlayerDetail(
        #                         speed_km_per_hour=speed_km_per_hour,
        #                         covered_distance=total_distance[tracked_object][track_id]
        #                     )
        #                     tracks_collection.update_track(
        #                         entity_type="players",
        #                         frame_num=frame_num_batch,
        #                         track_id=track_id,
        #                         track_detail=player_tracker
        #                         )

        #                 tracks[tracked_object][frame_num_batch][track_id]['speed'] = speed_km_per_hour
        #                 tracks[tracked_object][frame_num_batch][track_id]['distance'] = total_distance[tracked_object][track_id]

    def draw_speed_and_distance(
            self,
            frames: list[MatLike],
            tracks: Dict[str, Dict[int, Dict[int, TrackDetailBase]]]):
        output_frames = []
        for frame_num, frame in enumerate(frames):
            for object, object_tracks in tracks.items():
                if object == "ball" or object == "referees":
                    continue
                for _, track_info in object_tracks[frame_num].items():
                    if track_info.speed_km_per_hour is not None and track_info.covered_distance is not None:
                        speed = track_info.speed_km_per_hour
                        distance = track_info.covered_distance
                        if speed is None or distance is None:
                            continue

                        bbox = track_info.bbox
                        position = get_foot_position(bbox)
                        position = list(position)
                        position[1] += 40

                        position = tuple(map(int, position))
                        cv2.putText(
                            frame, f"{
                                speed:.2f} km/h",
                            position,
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.5,
                            (0, 0, 0),
                            2)
                        cv2.putText(frame,
                                    f"{distance:.2f} m",
                                    (position[0],
                                     position[1] + 20),
                                    cv2.FONT_HERSHEY_SIMPLEX,
                                    0.5,
                                    (0,
                                        0,
                                        0),
                                    2)
            output_frames.append(frame)

        return output_frames
