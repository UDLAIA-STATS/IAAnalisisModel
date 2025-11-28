import gc
import logging
import pathlib
import pickle

import cv2
import numpy as np
from cv2.typing import MatLike

from app.layers.domain.collections.track_collection import TrackCollection


class CameraMovementEstimator():
    def __init__(self, frame: MatLike):
        self.minimum_distance = 5

        self.lk_params = dict(winSize=(15, 15), maxLevel=2, criteria=(
            cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03))

        first_frame_grayscale = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        mask_features = np.zeros_like(first_frame_grayscale)
        mask_features[:, 0:20] = 1
        mask_features[:, 900:1050] = 1

        self.features = dict(
            maxCorners=100,
            qualityLevel=0.3,
            minDistance=3,
            blockSize=7,
            mask=mask_features
        )

    def add_adjust_positions_to_tracks(
            self,
            camera_movement_per_frame,
            tracks_collection: TrackCollection):
        for entity_type, frames in tracks_collection.tracks.items():
            for frame_num, tracks_in_frames in frames.items():
                # if frame_num < len(camera_movement_per_frame):
                camera_movement = camera_movement_per_frame[frame_num]
                # else:
                #     # If frame_num is out of range, assume no camera movement
                #     # for this frame
                #     camera_movement = (0, 0)
                dx, dy = camera_movement
                for track_id, track_detail in tracks_in_frames.items():
                    print(f"Actual position of track {track_id}: {track_detail.position}")
                    x, y = track_detail.position or (0, 0)
                    position_adjusted = (x - dx, y - dy)
                    track_detail.update(position_adjusted=position_adjusted)
                    # track_detail.position_adjusted = position_adjusted
                    tracks_collection.update_track(
                        entity_type=entity_type,
                        frame_num=frame_num,
                        track_id=track_id,
                        track_detail=track_detail
                    )

        # for object, object_tracks in tracks.items():
        #     for frame_num, track in enumerate(object_tracks):
        #         if frame_num < len(camera_movement_per_frame):
        #             camera_movement = camera_movement_per_frame[frame_num]
        #         else:
        #             # If frame_num is out of range, assume no camera movement
        #             # for this frame
        #             camera_movement = (0, 0)
        #         dx, dy = camera_movement
        #         for track_id, track_info in track.items():
        #             x, y = track_info['position']
        #             position_adjusted = (x - dx, y - dy)
        #             if object == 'ball':
        #                 ball_track = TrackBallDetail(position_adjusted=position_adjusted)
        #                 tracks_collection.update_track(
        #                     frame_num=frame_num,
        #                     track_id=track_id,
        #                     track_detail=ball_track,
        #                     entity_type="ball"
        #                 )
        #             else:
        #                 player_track = TrackPlayerDetail(position_adjusted=position_adjusted)
        #                 tracks_collection.update_track(
        #                     entity_type="players",
        #                     frame_num=frame_num,
        #                     track_id=track_id,
        #                     track_detail=player_track
        #                 )
        #             tracks[object][frame_num][track_id]['position_adjusted'] = position_adjusted

    def get_camera_movement(
            self,
            frames: list[MatLike],
            read_from_stub: bool = False,
            stub_path: str = ""):
        # Read the stub
        if read_from_stub and stub_path is not None and pathlib.Path(stub_path).exists():
            with pathlib.Path(stub_path).open('rb') as f:
                return pickle.load(f)

        camera_movement = [[0, 0]] * len(frames)

        old_gray = cv2.cvtColor(frames[0], cv2.COLOR_BGR2GRAY)
        old_features = cv2.goodFeaturesToTrack(
            old_gray, **self.features)  # type: ignore

        for frame_num in range(1, len(frames)):
            frame_gray = cv2.cvtColor(frames[frame_num], cv2.COLOR_BGR2GRAY)

            if old_features is None or len(old_features) == 0:
                new_features = cv2.goodFeaturesToTrack(
                    frame_gray, **self.features)  # type: ignore
                if new_features is None:
                    camera_movement[frame_num] = [0, 0] # type: ignore
                    old_gray = frame_gray.copy()
                    continue
            else:
                new_features, _, _ = cv2.calcOpticalFlowPyrLK(
                    old_gray,
                    frame_gray,
                    old_features,
                    None,  # type: ignore
                    **self.lk_params  # type: ignore
                )

            # new_features, _, _ =
            # cv2.calcOpticalFlowFarneback(old_gray, frame_gray,
            # old_features, None, **self.lk_params)
            # cv2.calcOpticalFlowFarneback returna un vector 2D
            max_distance = 0
            camera_movement_x, camera_movement_y = 0, 0

            camera_movement_x, camera_movement_y, max_distance = self.update_camera_distance(
                new_features, old_features) # type: ignore

            if max_distance > self.minimum_distance:
                camera_movement[frame_num] = [camera_movement_x, camera_movement_y]  # type: ignore
                old_features = cv2.goodFeaturesToTrack(
                    frame_gray, **self.features)  # type: ignore

            old_gray = frame_gray.copy()

        if stub_path is not None:
            with pathlib.Path(stub_path).open('wb') as f:
                pickle.dump(camera_movement, f)

        return camera_movement

    def update_camera_distance(
            self, new_features, old_features) -> tuple[float, float, float]:
        """
        Encuentra el punto con mayor desplazamiento entre características antiguas y nuevas,
        y calcula el movimiento de cámara basado en ese punto.

        Esta función es útil para detectar el movimiento dominante de la cámara
        basándose en el punto que más se ha movido entre frames.

        Args:
            new_features: Array de características nuevas, shape (n, 2) o (n, 1, 2)
            old_features: Array de características anteriores, shape (n, 2) o (n, 1, 2)

        Returns:
            Tupla con (movimiento_x, movimiento_y, distancia_maxima)
        """
        if len(new_features) != len(old_features) or len(new_features) == 0:
            return 0.0, 0.0, 0.0
        max_distance = 0.0
        camera_movement_x = 0.0
        camera_movement_y = 0.0

        for new_feat, old_feat in zip(new_features, old_features):
            new_point = new_feat.ravel()
            old_point = old_feat.ravel()

            diff = new_point - old_point
            distance = np.linalg.norm(diff)

            if distance > max_distance:
                max_distance = distance
                camera_movement_x = float(diff[0])
                camera_movement_y = float(diff[1])

        return camera_movement_x, camera_movement_y, float(max_distance)

    def draw_camera_movement(self, frames: list[MatLike], camera_movement_per_frame):
        output_frames = []

        for frame_num, frame in enumerate(frames):
            frame = np.copy(frame)

            overlay = frame.copy()
            cv2.rectangle(overlay, (0, 0), (500, 100), (255, 255, 255), -1)
            alpha = 0.6
            cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0, frame)
            x_movement, y_movement = camera_movement_per_frame[frame_num]
            frame = cv2.putText(frame,
                                f"Camera Movement X: {x_movement:.2f}",
                                (10, 30),
                                cv2.FONT_HERSHEY_SIMPLEX,
                                1,
                                (0, 0, 0),
                                3)
            frame = cv2.putText(frame,
                                f"Camera Movement Y: {y_movement:.2f}",
                                (10, 60),
                                cv2.FONT_HERSHEY_SIMPLEX,
                                1,
                                (0, 0, 0),
                                3)
            output_frames.append(frame)

            if frame_num % 50 == 0:
                gc.collect()

        return output_frames
