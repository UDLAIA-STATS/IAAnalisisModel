from typing import Dict, Type
from layers.infraestructure.video_analysis.plotting.interfaces.diagram import Diagram


class DrawerFactoryError(Exception):
    pass

class DrawerFactory:
    @classmethod
    def run_drawer(cls, drawer_type: Type[Diagram], tracks: Dict) -> None:
        """Runs a specific drawer based on the drawer type."""
        
        if drawer_type is None or not issubclass(drawer_type, Diagram):
            raise DrawerFactoryError(f"Invalid drawer type: {drawer_type}")
        
        drawer_instance = drawer_type(tracks)
        drawer_instance.draw_and_save()
