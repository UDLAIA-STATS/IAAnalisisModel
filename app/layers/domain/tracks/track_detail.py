from dataclasses import dataclass
from typing import Dict, List, Literal, Optional, Tuple
from pydantic import BaseModel, Field

class TrackDetailBase(BaseModel):
    bbox: Optional[List] = None
    position: Optional[Tuple] = None
    position_adjusted: Optional[Tuple] = None
    position_transformed: Optional[List] = None
    covered_distance: Optional[float] = None
    speed_km_per_hour: Optional[float] = None
    track_id: Optional[int] = None
    class_id: Optional[int] = None

    def update(self, **kwargs):
        for key, value in kwargs.items():
            if hasattr(self, key):
                if value is not None and value != getattr(self, key):
                    setattr(self, key, value)

class TrackPlayerDetail(TrackDetailBase):
    has_ball: Optional[bool] = False
    team: Optional[Literal[1]] = None
    team_color: Optional[Dict] = Field(default_factory=dict)
    passing: Optional[bool] = False
    shooting: Optional[bool] = False
    pass_counter: Optional[int] = 0
    shooting_counter: Optional[int] = 0
    passed_to: Optional[int] = -1
    received_from: Optional[int] = -1
    team_id: Optional[int] = -1

class TrackBallDetail(TrackDetailBase):
    pass