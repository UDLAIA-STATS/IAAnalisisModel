from abc import ABC, abstractmethod
from typing import Dict


class Diagram(ABC):
    def __init__(self, tracks: Dict):
        self.tracks = tracks
    
    @abstractmethod
    def draw_and_save(self) -> None:
        pass