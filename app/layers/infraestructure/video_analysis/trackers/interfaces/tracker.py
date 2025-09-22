from abc import ABC, abstractmethod
from typing import Dict

import cv2
import numpy as np
import supervision as sv
from cv2.typing import MatLike
from app.layers.domain.collections.track_collection import TrackCollection
from app.layers.domain.tracks.track_detail import TrackDetailBase, TrackPlayerDetail
from app.layers.infraestructure.video_analysis.services import (get_bbox_width,
                                                            get_center_of_bbox,
                                                            read_stub, save_stub)
from ultralytics import YOLO
from ultralytics.engine.results import Results


class Tracker(ABC):
    def __init__(self, model: YOLO):
        self.model = model
        self.tracker = sv.ByteTrack()
        # self.metric = nn_matching.NearestNeighborDistanceMetric("cosine", 0.2, None)
        # self.tracker = DeepSortTracker()

    @abstractmethod
    def get_object_tracks(
            self,
            detection_with_tracks: sv.Detections,
            cls_names_inv: dict[str, int],
            frame_num: int,
            detection_supervision: sv.Detections,
            tracks_collection: TrackCollection) -> None:
        raise NotImplementedError

    def read_tracks_from_stub(self, stub_path: str) -> dict:
        tracks: dict = {"players": [], "referees": [], "ball": []}
        if stub_path is not None:
            temp = read_stub(stub_path)
            tracks = temp if temp is not None else tracks
        print("Tracks are: ", tracks)
        return tracks

    def save_tracks_to_stub(self, tracks: dict, stub_path: str):
        if stub_path is not None:
            save_stub(tracks, stub_path)

    def detect_frames(self, frames: list[MatLike]):
        batch_size = 20
        detections: list[Results] = []
        for i in range(0, len(frames), batch_size):
            detections_batch = self.model.predict(
                frames[i:i + batch_size], conf=0.1)
            detections += detections_batch
        return detections

    def draw_ellipse(self, frame, bbox, color, track_id=None):
        y2 = int(bbox[3])
        x_center, _ = get_center_of_bbox(bbox)
        width = get_bbox_width(bbox)

        cv2.ellipse(
            frame,
            center=(x_center, y2),
            axes=(int(width), int(0.35 * width)),
            angle=0.0,
            startAngle=-45,
            endAngle=235,
            color=color,
            thickness=2,
            lineType=cv2.LINE_4
        )

        rectangle_width = 40
        rectangle_height = 20
        x1_rect = x_center - rectangle_width // 2
        x2_rect = x_center + rectangle_width // 2
        y1_rect = (y2 - rectangle_height // 2) + 15
        y2_rect = (y2 + rectangle_height // 2) + 15

        if track_id is not None:
            cv2.rectangle(frame,
                          (int(x1_rect), int(y1_rect)),
                          (int(x2_rect), int(y2_rect)),
                          color,
                          cv2.FILLED)

            x1_text = x1_rect + 12
            if track_id > 99:
                x1_text -= 10

            cv2.putText(
                frame,
                f"{track_id}",
                (int(x1_text), int(y1_rect + 15)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (0, 0, 0),
                2
            )

        return frame

    def draw_triangle(self, frame, bbox, color):
        y = int(bbox[1])
        x, _ = get_center_of_bbox(bbox)

        triangle_points = np.array([
            [x, y],
            [x - 10, y - 20],
            [x + 10, y - 20],
        ])
        cv2.drawContours(frame, [triangle_points], 0, color, cv2.FILLED)
        cv2.drawContours(frame, [triangle_points], 0, (0, 0, 0), 2)

        return frame

    def draw_team_ball_control(self, frame, frame_num, team_ball_control):
        # Draw a semi-transparent rectaggle
        overlay = frame.copy()
        cv2.rectangle(overlay, (1350, 850), (1900, 970), (255, 255, 255), -1)
        alpha = 0.4
        cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0, frame)

        team_ball_control_till_frame = team_ball_control[:frame_num + 1]
        # Get the number of time each team had ball control
        team_1_num_frames = team_ball_control_till_frame[team_ball_control_till_frame == 1].shape[0]
        team_2_num_frames = team_ball_control_till_frame[team_ball_control_till_frame == 2].shape[0]
        print("Team 1 ball control, ", team_1_num_frames)
        print("Team 2 ball control, ", team_2_num_frames)
        team_1 = team_1_num_frames / (team_1_num_frames + team_2_num_frames)
        team_2 = team_2_num_frames / (team_1_num_frames + team_2_num_frames)

        cv2.putText(
            frame,
            f"Team 1 Ball Control: {
                team_1 * 100:.2f}%",
            (1400,
             900),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (0,
             0,
             0),
            3)
        cv2.putText(
            frame,
            f"Team 2 Ball Control: {
                team_2 * 100:.2f}%",
            (1400,
             950),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (0,
             0,
             0),
            3)

        return frame

    def draw_annotations(
            self, 
            video_frames: list[MatLike], 
            tracks: Dict[str, Dict[int, Dict[int, TrackDetailBase]]], 
            team_ball_control) -> list[MatLike]:
        """
         Dibuja anotaciones sobre frames de video:
        - Jugadores con elipse (color por equipo).
        - Balón con triángulo verde.
        - Indicador de control de balón por equipo.
        """

        output_video_frames = []

        for frame_num, frame in enumerate(video_frames):
            # Copia defensiva del frame
            frame = np.copy(frame)

            # Obtener tracks del frame (con fallback si no existen)
            player_dict = tracks.get("players", {}).get(frame_num, {})
            ball_dict = tracks.get("ball", {}).get(frame_num, {})

            # --- Dibujar jugadores ---
            for track_id, player in player_dict.items():
                if not isinstance(player, TrackDetailBase):
                    continue
                if player.bbox is None:
                    continue

                # Mejor: usar directamente player (ya es TrackDetailBase o subclase)
                team_color = getattr(player, "team_color", None) or (0, 0, 255)
                frame = self.draw_ellipse(frame, player.bbox, team_color, track_id)

                if getattr(player, "has_ball", False):
                    frame = self.draw_triangle(frame, player.bbox, (0, 0, 255))

            # --- Dibujar balón ---
            for _, ball in ball_dict.items():
                if ball.bbox is None:
                    continue
                frame = self.draw_triangle(frame, ball.bbox, (0, 255, 0))

            # --- Dibujar control de balón ---
            frame = self.draw_team_ball_control(frame, frame_num, team_ball_control)

            output_video_frames.append(frame)

        return output_video_frames
    