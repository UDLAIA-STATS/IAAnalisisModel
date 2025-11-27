from typing import Dict

from app.layers.domain.tracks.track_detail import TrackDetailBase
from app.layers.infraestructure.video_analysis.plotting.drawers import (
    BallDetectionMetricsDrawer, BallTrajectoryDrawer, HeatmapDrawer,
    InterpolationErrorDrawer, VelocityConsistencyDrawer, VoronoiDiagramDrawer)
from app.layers.infraestructure.video_analysis.plotting.services import \
    DrawerFactory


def generate_diagrams(
        tracks: Dict[str, Dict[int, Dict[int, TrackDetailBase]]],
        metrics: Dict) -> None:
    try:
        DrawerFactory.run_drawer(VoronoiDiagramDrawer, tracks['players'])
        DrawerFactory.run_drawer(HeatmapDrawer, tracks['players'])
        DrawerFactory.run_drawer(BallTrajectoryDrawer, tracks['players'])
        # DrawerFactory.run_drawer('processing_time', metrics)
        # DrawerFactory.run_drawer('memory_usage', metrics)
        #DrawerFactory.run_drawer(BallDetectionMetricsDrawer, tracks={}, metrics=metrics)
        #DrawerFactory.run_drawer(VelocityConsistencyDrawer, tracks={}, metrics=metrics)
        #DrawerFactory.run_drawer(InterpolationErrorDrawer, tracks={}, metrics=metrics)
    except Exception as e:
        print(f"Error drawing diagram: {e}")
