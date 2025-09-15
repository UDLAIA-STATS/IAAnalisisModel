from typing import List, Tuple

import numpy as np
from layers.domain.coordinates.bbox_coords import BoundingBoxCoordinates
from layers.domain.player.player import Player
from layers.domain.player.player_tracker import PlayerTracker


class DeepSortTracker:
    "Servicio a nivel de dominio para el seguimiento de objetos utilizando Deep SORT."

    def __init__(
            self,
            max_age: int = 30,
            min_hits: int = 3,
            iou_threshold: float = 0.3):
        self.max_age = max_age
        self.min_hits = min_hits
        self.iou_threshold = iou_threshold
        self.tracks: List[PlayerTracker] = []
        self.next_id = 1

    def calculate_iou(
            self,
            bbox1: List[float],
            bbox2: List[float]) -> float:
        """Calcula Intersection over Union entre dos bounding boxes"""
        firstCoords = BoundingBoxCoordinates(*bbox1)
        secondCoords = BoundingBoxCoordinates(*bbox2)

        # Coordenadas de intersección
        x1 = max(firstCoords.x1, secondCoords.x1)
        y1 = max(firstCoords.y1, secondCoords.y1)
        x2 = min(firstCoords.x2, secondCoords.x2)
        y2 = min(firstCoords.y2, secondCoords.y2)

        if x2 <= x1 or y2 <= y1:
            return 0.0

        # Área de intersección
        intersection = (x2 - x1) * (y2 - y1)

        # Área de unión
        area1 = (x2 - x1) * (y2 - y1)
        area2 = (secondCoords.x2 - secondCoords.x1) * \
            (secondCoords.y2 - secondCoords.y1)
        union = area1 + area2 - intersection

        return intersection / union if union > 0 else 0.0

    def calculate_distance(
            self,
            pos1: Tuple[float, float],
            pos2: Tuple[float, float]) -> float:
        """Calcula la distancia euclidiana entre dos posiciones"""
        return np.sqrt((pos1[0] - pos2[0]) ** 2 + (pos1[1] - pos2[1]) ** 2)

    def update_tracks(self, players: List[Player]) -> List[Player]:
        updated_players = []

        for player in players:
            best_match_id = None
            best_score = 0.0

            # Buscar mejor coincidencia en tracks existentes
            for track in self.tracks:
                if track.team == player.team:
                    iou = self.calculate_iou(player.bbox, track.bbox)
                    distance = self.calculate_distance(
                        player.position_2d, track.position_2d)

                    # Score combinado (mayor peso a IoU)
                    score = iou * 0.7 + (1.0 / (1.0 + distance * 0.01)) * 0.3

                    if score > best_score and score > 0.3:  # Umbral mínimo
                        best_score = score
                        best_match_id = self.tracks.index(track)

            if best_match_id:
                # Actualizar track existente
                player.id = best_match_id
                self.tracks[best_match_id] = PlayerTracker(**player.__dict__)
                self.tracks[best_match_id].hits += 1
                self.tracks[best_match_id].age = 0
                self.tracks[best_match_id].last_bbox = player.bbox
                self.tracks[best_match_id].last_position = player.position_2d
            else:
                # Crear nuevo track
                player.id = self.next_id
                self.tracks[self.next_id] = PlayerTracker(**player.__dict__)
                self.tracks[self.next_id].hits = 1
                self.tracks[self.next_id].age = 0
                self.tracks[self.next_id].last_bbox = player.bbox
                self.tracks[self.next_id].last_position = player.position_2d
                self.next_id += 1

            updated_players.append(player)

        # Incrementar edad de tracks no actualizados
        tracks_to_remove = []
        for track_id, track in enumerate(self.tracks):
            if not any(p.id == track_id for p in updated_players):
                track.age += 1
                if track.age > self.max_age:
                    tracks_to_remove.append(track_id)

        # Remover tracks viejos
        for track_id in tracks_to_remove:
            del self.tracks[track_id]

        return updated_players
