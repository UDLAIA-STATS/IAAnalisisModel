from dataclasses import field

from app.layers.domain.collection.collectors.frame_collector import FrameCollector

class Collector():
    id: int
    frames: list[FrameCollector] = field(default_factory=list)