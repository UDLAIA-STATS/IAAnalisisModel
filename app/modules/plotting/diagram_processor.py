from typing import Dict

from app.modules.plotting.drawers import (
    BallDetectionMetricsDrawer, BallTrajectoryDrawer, HeatmapDrawer,
    InterpolationErrorDrawer, VelocityConsistencyDrawer, VoronoiDiagramDrawer)
from app.modules.plotting.services import \
    DrawerFactory
from sqlalchemy.orm import Session

def generate_diagrams(db: Session) -> None:
    try:
        DrawerFactory.run_drawer(HeatmapDrawer, db)
        # DrawerFactory.run_drawer('processing_time', metrics)
        # DrawerFactory.run_drawer('memory_usage', metrics)
        #DrawerFactory.run_drawer(BallDetectionMetricsDrawer, tracks={}, metrics=metrics)
        #DrawerFactory.run_drawer(VelocityConsistencyDrawer, tracks={}, metrics=metrics)
        #DrawerFactory.run_drawer(InterpolationErrorDrawer, tracks={}, metrics=metrics)
    except Exception as e:
        print(f"Error drawing diagram: {e}")
