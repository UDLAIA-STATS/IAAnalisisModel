from typing import Dict, List, Literal, Optional, Tuple
import numpy as np
from pydantic import BaseModel, Field


class TrackDetailBase(BaseModel):
    bbox: Optional[List] = None
    position: Optional[Tuple] = None # tuple[int, int] = (x, y)
    position_adjusted: Optional[Tuple] = None
    position_transformed: Optional[List] = None
    covered_distance: Optional[float] = None
    speed_km_per_hour: Optional[float] = None
    track_id: Optional[int] = None # Unique identifier for the track
    class_id: Optional[int] = None # 0: player, 1: ball

    def update(self, **kwargs):
        """
        Actualiza los atributos de la instancia según el tipo de dato recibido.
        No concatena tuplas ni listas salvo que sea necesario (por ejemplo en bbox).
        """
        valid_fields = self.model_dump().keys()
        clean_data = {k: v for k, v in kwargs.items() if k in valid_fields and v is not None}

        for k, v in clean_data.items():
            if isinstance(v, tuple):
                clean_data[k] = v[:2]

        updated = self.model_copy(update=clean_data)
        self.__dict__.update(updated.__dict__)
        return self 


class TrackPlayerDetail(TrackDetailBase):
    has_ball: Optional[bool] = False
    team: Optional[int] = None
    team_color: Optional[np.ndarray] = Field(default_factory=lambda: np.array([0, 0, 255]))
    passing: Optional[bool] = False
    shooting: Optional[bool] = False
    pass_counter: Optional[int] = 0
    shooting_counter: Optional[int] = 0
    passed_to: Optional[int] = -1
    received_from: Optional[int] = -1
    team_id: Optional[int] = -1
    
    class Config:
        arbitrary_types_allowed = True

class TrackBallDetail(TrackDetailBase):
    pass
