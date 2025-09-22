from typing import Dict, Type

from app.layers.domain.tracks.track_detail import TrackDetailBase
from app.layers.infraestructure.video_analysis.plotting.interfaces.diagram import \
    Diagram


class DrawerFactoryError(Exception):
    pass


class DrawerFactory:
    @classmethod
    def run_drawer(cls, drawer_type: Type[Diagram], tracks: Dict[int, Dict[int, TrackDetailBase]], metrics: Dict | None = None) -> None:
        """Runs a specific drawer based on the drawer type."""

        if drawer_type is None or not issubclass(drawer_type, Diagram):
            raise DrawerFactoryError(f"Invalid drawer type: {drawer_type}")

        drawer_instance = drawer_type(tracks=tracks, metrics=metrics)
        drawer_instance.draw_and_save()
