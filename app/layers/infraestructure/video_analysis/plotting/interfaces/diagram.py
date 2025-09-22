from abc import ABC, abstractmethod
from typing import Dict

from app.layers.domain.tracks.track_detail import TrackDetailBase


class Diagram(ABC):
    def __init__(self, tracks: Dict[int, Dict[int, TrackDetailBase]], metrics: Dict | None = None):
        self.tracks: Dict[int, Dict[int, TrackDetailBase]] = tracks
        self.metrics: Dict | None = metrics

    @abstractmethod
    def draw_and_save(self) -> None:
        pass
