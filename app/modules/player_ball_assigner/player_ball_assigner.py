
from typing import Dict
from app.entities.tracks.track_detail import TrackDetailBase
from app.modules.services.bbox_processor_service import (
    get_center_of_bbox, measure_scalar_distance)


class PlayerBallAssigner():
    def __init__(self):
        self.max_player_ball_distance = 70

    def assign_ball_to_player(
            self,
            players: Dict[int, TrackDetailBase],
            ball_bbox):
        """
        Assign a ball to a player based on the distance of the player's bounding box from the ball's position.

        Parameters
        ----------
        players : Dict[int, TrackDetailBase]
            Dictionary mapping player IDs to their corresponding TrackDetailBase objects.
        ball_bbox : List[int]
            List of 4 integers representing the bounding box of the ball.

        Returns
        -------
        int
            Track Id of the player to whom the ball is assigned, or -1 if no player is assigned.
        """
        ball_position = get_center_of_bbox(ball_bbox)

        miniumum_distance = 99999
        assigned_player = -1

        for player_id, player in players.items():
            player_bbox = player.bbox

            if player_bbox is None:
                continue

            distance_left = measure_scalar_distance(
                (player_bbox[0], player_bbox[-1]), ball_position)
            distance_right = measure_scalar_distance(
                (player_bbox[2], player_bbox[-1]), ball_position)
            distance = min(distance_left, distance_right)

            if distance < self.max_player_ball_distance:
                if distance < miniumum_distance:
                    miniumum_distance = distance
                    assigned_player = player_id

        return assigned_player
