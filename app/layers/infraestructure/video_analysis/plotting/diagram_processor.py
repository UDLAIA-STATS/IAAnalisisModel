from typing import Dict
from layers.infraestructure.video_analysis.plotting.drawers import (
    BallDetectionMetricsDrawer, 
    BallTrajectoryDrawer, 
    HeatmapDrawer,
    InterpolationErrorDrawer,
    VelocityConsistencyDrawer,
    VoronoiDiagramDrawer
)
from layers.infraestructure.video_analysis.plotting.services import DrawerFactory


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