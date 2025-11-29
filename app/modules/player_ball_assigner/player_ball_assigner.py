
from typing import List
from app.modules.services.bbox_processor_service import (
    get_center_of_bbox, measure_scalar_distance)
from app.entities.models.PlayerState import PlayerStateModel

class PlayerBallAssigner():
    def __init__(self):
        self.max_player_ball_distance = 70

    def assign_ball_to_player(
            self,
            players: List[PlayerStateModel],
            ball_bbox):
        try:
            ball_position = get_center_of_bbox(ball_bbox)

            min_distance = float('inf')
            closest_player_id = -1
            
            for player in players:
                bbox = player.get_bbox()

                if not bbox or len(bbox) < 4:
                    continue

                left_foot  = (bbox[0], bbox[3])
                right_foot = (bbox[2], bbox[3])

                dist_left  = measure_scalar_distance(left_foot, ball_position)
                dist_right = measure_scalar_distance(right_foot, ball_position)
                distance   = min(dist_left, dist_right)

                if distance < self.max_player_ball_distance and distance < min_distance:
                    min_distance = distance
                    closest_player_id = player.to_dict()['id']

            return closest_player_id
        except Exception as e:
            print(f"Error asignando balón a jugador: {e}")
            raise e
