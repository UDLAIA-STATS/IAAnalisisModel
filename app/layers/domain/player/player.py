from dataclasses import dataclass, field
import hashlib
from typing import Any, Dict, List, Optional, Tuple

from layers.domain.player.iplayer import IPlayer


@dataclass
class Player(IPlayer):
    speed: Optional[float]
    distance: Optional[float] 
    team: str
    team_color: Optional[List[float]] # RBG color
    has_ball: bool = False
    timestamp: float = field(default_factory=lambda: 0.0)
    
    def __init__(self, id: int, bbox: List[float], position_2d: Tuple[float, float],
                 position_adjusted_2d: Tuple[float, float], position_transformed_2d: Optional[List[float]] = None,
                 speed: Optional[float] = None, distance: Optional[float] = None, 
                 position_3d: Optional[List[float]] = None, position_adjusted_3d: Optional[List[float]] = None,
                 position_transformed_3d: Optional[List[float]] = None,
                 team: str = '', team_color: Optional[List[float]] = None, has_ball: bool = False,
                 timestamp: float = 0.0):
        self.id = id
        self.bbox = bbox
        self.position_2d = position_2d
        self.position_adjusted_2d = position_adjusted_2d
        self.position_transformed_2d = position_transformed_2d
        self.speed = speed
        self.distance = distance
        self.position_3d = position_3d
        self.position_adjusted_3d = position_adjusted_3d
        self.position_transformed_3d = position_transformed_3d
        self.team = team
        self.team_color = team_color
        self.has_ball = has_ball
        self.timestamp = timestamp
    
    def __post_init__(self):
        """Validación y procesamiento post-inicialización"""
        if self.speed is not None and self.speed < 0:
            raise ValueError("Speed cannot be negative")
        if self.distance is not None and self.distance < 0:
            raise ValueError("Distance cannot be negative")
    
    def get_hash(self) -> str:
        """Genera un hash único para el jugador basado en sus propiedades"""
        data = f"{self.id}_{self.position_2d}_{self.team}_{self.timestamp}"
        return hashlib.md5(data.encode()).hexdigest()
    
    def get_center_position(self) -> Tuple[float, float]:
        """Calcula la posición central del bounding box"""
        x1, y1, x2, y2 = self.bbox
        return ((x1 + x2) / 2, (y1 + y2) / 2)
    
    def get_bbox_area(self) -> float:
        """Calcula el área del bounding box"""
        x1, y1, x2, y2 = self.bbox
        return (x2 - x1) * (y2 - y1)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convierte el jugador a diccionario"""
        return {
            'id': self.id,
            'bbox': self.bbox,
            'position': self.position_3d,
            'position_adjusted': self.position_adjusted_3d,
            'position_transformed': self.position_transformed_3d,
            'speed': self.speed,
            'distance': self.distance,
            'team_id': self.team,
            'team_color': self.team_color,
            'timestamp': self.timestamp
        }