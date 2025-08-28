from abc import ABC
from dataclasses import field
from typing import List, Optional, Tuple


class IPlayer(ABC):

    id: int
    bbox: List[float] = field(default_factory=list)
    position_2d: Tuple[float, float] = field(default_factory=tuple)
    position_adjusted_2d: Tuple[float, float] = field(default_factory=tuple)
    position_transformed_2d: Optional[List[float]] = field(default=None)
    position_3d: Optional[List[float]] = field(default=None)
    position_adjusted_3d: Optional[List[float]] = field(default=None)
    position_transformed_3d: Optional[List[float]] = field(default=None)