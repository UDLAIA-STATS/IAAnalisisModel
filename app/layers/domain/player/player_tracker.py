from dataclasses import dataclass, field
from layers.domain.player.player import Player

@dataclass
class PlayerTracker(Player):
    hits: int = 0
    age: int = 0
    last_position: tuple[float, float] = (0.0, 0.0)
    last_bbox: list[float] = field(default_factory=lambda: [0.0, 0.0, 0.0, 0.0])