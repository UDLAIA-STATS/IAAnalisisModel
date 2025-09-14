from typing import Dict, Type
from app.layers.infraestructure.video_analysis.plotting.interfaces.diagram import Diagram
from layers.infraestructure.video_analysis.plotting.drawers.ball_detection_metrics_drawer import BallDetectionMetricsDrawer
from layers.infraestructure.video_analysis.plotting.drawers.interpolation_error_drawer import InterpolationErrorDrawer
from layers.infraestructure.video_analysis.plotting.drawers.memory_usage_drawer import MemoryUsageDrawer
from layers.infraestructure.video_analysis.plotting.drawers.processing_time_drawer import ProcessingTimeDrawer
from layers.infraestructure.video_analysis.plotting.drawers.velocity_consistency_drawer import VelocityConsistencyDrawer
from layers.infraestructure.video_analysis.plotting.drawers.ball_trajectory_drawer import BallTrajectoryDrawer
from layers.infraestructure.video_analysis.plotting.drawers.heatmap_drawer import HeatmapDrawer
from layers.infraestructure.video_analysis.plotting.drawers.voronoi_diagram_drawer import VoronoiDiagramDrawer

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
