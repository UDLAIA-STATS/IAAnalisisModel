import logging
from typing import Dict, List, Optional
from cv2.typing import MatLike
import numpy as np
from sklearn.cluster import KMeans
from app.entities.tracks.track_detail import TrackDetailBase
from app.entities.utils.singleton import Singleton


class TeamAssigner(metaclass=Singleton):
    """
    Optimizado para trabajar *por frame*, con múltiples jugadores.
    - Manejo seguro de bboxes inválidos
    - KMeans solo cuando hay información suficiente
    - Limpieza en logs
    - Código más compacto y eficiente
    """

    def __init__(self):
        self.team_colors: Dict[int, np.ndarray] = {}
        self.player_team_dict: Dict[int, int] = {}
        self.kmeans: Optional[KMeans] = None

    # -------------------------------------------
    # Utils
    # -------------------------------------------
    def get_coords_from_bbox(self, frame: MatLike, bbox: List[int]):
        h, w = frame.shape[:2]

        x1 = max(0, int(bbox[0]))
        y1 = max(0, int(bbox[1]))
        x2 = min(w - 1, int(bbox[2]))
        y2 = min(h - 1, int(bbox[3]))

        return x1, y1, x2, y2

    def validate_bbox_area(self, frame: MatLike, bbox: List[int]) -> bool:
        """
        Validación rápida antes de procesar.
        Evita trabajo innecesario.
        """
        x1, y1, x2, y2 = self.get_coords_from_bbox(frame, bbox)

        if x2 <= x1 or y2 <= y1:
            return False

        # recorte
        crop = frame[y1:y2, x1:x2]
        if crop.size == 0:
            return False

        # tomar solo la mitad superior
        top = crop[: max(1, crop.shape[0] // 2), :]

        return top.size > 0

    # -------------------------------------------
    # Color Extraction
    # -------------------------------------------
    def extract_player_color(self, frame: MatLike, bbox: List[int]) -> Optional[np.ndarray]:
        """
        Devuelve un color dominante (RGB) o None.
        """
        if not self.validate_bbox_area(frame, bbox):
            return None

        x1, y1, x2, y2 = self.get_coords_from_bbox(frame, bbox)
        crop = frame[y1:y2, x1:x2]

        # mitad superior
        top = crop[: crop.shape[0] // 2, :]

        # a 2D
        pixels = top.reshape(-1, 3)

        # si hay pocos pixeles → no vale la pena fit
        if pixels.shape[0] < 20:
            return None

        try:
            kmeans = KMeans(n_clusters=2, init="k-means++", n_init=1)
            kmeans.fit(pixels)
        except Exception as e:
            logging.debug(f"KMeans failed on player crop: {e}")
            return None

        labels = kmeans.labels_.reshape(top.shape[:2])

        # detectar cluster que NO es la camiseta (fondo)
        corners = [
            labels[0, 0], labels[0, -1],
            labels[-1, 0], labels[-1, -1]
        ]
        non_player_cluster = max(set(corners), key=corners.count)
        player_cluster = 1 - non_player_cluster

        return kmeans.cluster_centers_[player_cluster]

    # -------------------------------------------
    # Training clusters per frame
    # -------------------------------------------
    def assign_team_colors(self, frame: MatLike, players: Dict[int, TrackDetailBase]):
        """
        Calcula los 2 clusters de color del frame.
        """
        valid_colors = []

        for _, det in players.items():
            if det.bbox is None:
                continue

            color = self.extract_player_color(frame, det.bbox)

            if color is not None:
                valid_colors.append(color)

        if len(valid_colors) < 2:
            logging.debug("Not enough valid colors to cluster teams.")
            return

        try:
            kmeans = KMeans(n_clusters=2, init="k-means++", n_init=10)
            kmeans.fit(valid_colors)
        except Exception as e:
            logging.debug(f"Team KMeans failed: {e}")
            return

        self.kmeans = kmeans
        self.team_colors = {
            1: kmeans.cluster_centers_[0],
            2: kmeans.cluster_centers_[1]
        }

    # -------------------------------------------
    # Team Assignment
    # -------------------------------------------
    def get_player_team(self, frame: MatLike, bbox: List[int], player_id: int) -> int:
        """
        Devuelve 1, 2 o -1.
        """
        # cache
        if player_id in self.player_team_dict:
            return self.player_team_dict[player_id]

        if self.kmeans is None:
            logging.debug("KMeans model not initialized yet.")
            return -1

        color = self.extract_player_color(frame, bbox)
        if color is None:
            return -1

        try:
            pred = int(self.kmeans.predict(color.reshape(1, -1))[0]) + 1
        except Exception as e:
            logging.debug(f"Prediction error for player {player_id}: {e}")
            return -1

        self.player_team_dict[player_id] = pred
        return pred
