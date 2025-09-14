from typing import Dict, List
from app.layers.infraestructure.video_analysis.plotting.drawers.ball_detection_metrics_drawer import BallDetectionMetricsDrawer
from app.layers.infraestructure.video_analysis.plotting.drawers.ball_trajectory_drawer import BallTrajectoryDrawer
from app.layers.infraestructure.video_analysis.plotting.drawers.heatmap_drawer import HeatmapDrawer
from app.layers.infraestructure.video_analysis.plotting.drawers.interpolation_error_drawer import InterpolationErrorDrawer
from app.layers.infraestructure.video_analysis.plotting.drawers.velocity_consistency_drawer import VelocityConsistencyDrawer
from app.layers.infraestructure.video_analysis.plotting.drawers.voronoi_diagram_drawer import VoronoiDiagramDrawer
from layers.infraestructure.video_analysis.plotting.services.drawer_factory import DrawerFactory


def generate_diagrams(tracks: Dict, metrics: Dict) -> None:
    try:
        DrawerFactory.run_drawer(VoronoiDiagramDrawer, tracks['players'])
        DrawerFactory.run_drawer(HeatmapDrawer, tracks['players'])
        DrawerFactory.run_drawer(BallTrajectoryDrawer, tracks['players'])
        #DrawerFactory.run_drawer('processing_time', metrics)
        #DrawerFactory.run_drawer('memory_usage', metrics)
        DrawerFactory.run_drawer(BallDetectionMetricsDrawer, metrics)
        DrawerFactory.run_drawer(VelocityConsistencyDrawer, metrics)
        DrawerFactory.run_drawer(InterpolationErrorDrawer, metrics)
    except Exception as e:
        print(f"Error drawing diagram: {e}")