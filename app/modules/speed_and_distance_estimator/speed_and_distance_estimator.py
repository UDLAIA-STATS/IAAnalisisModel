from typing import Dict, List, Optional
import numpy as np
from scipy.signal import savgol_filter
from sqlalchemy.orm import Session

from app.entities.models.PlayerState import PlayerStateModel
from app.entities.utils.singleton import Singleton
from app.modules.services.bbox_processor_service import measure_scalar_distance

# Modelo SQLAlchemy final donde guardarás cada frame:
# from app.db.models import PlayerPerformanceFrame


class SpeedAndDistanceEstimator(metaclass=Singleton):
    """
    Procesamiento en streaming:
    - Mantiene historial corto (N frames) por jugador
    - Suaviza valores
    - Interpola cuando desaparece
    - Calcula velocidades, distancias y aceleración
    - Persiste usando SQLAlchemy
    """

    def __init__(
        self,
        frame_rate: int = 24,
        sprint_threshold_kmh: float = 25.0,
        smoothing_window: int = 7,
        poly_order: int = 2,
        history_size: int = 60,      # N frames de buffer (≈2.5s)
    ) -> None:

        self.frame_rate = frame_rate
        self.sprint_threshold = sprint_threshold_kmh
        self.smoothing_window = smoothing_window
        self.poly_order = poly_order
        self.history_size = history_size

        # Historial
        self.position_history: Dict[int, List[Optional[np.ndarray]]] = {}
        self.speed_history: Dict[int, List[float]] = {}

    # ----------------------------------------------------------------------
    # Suavizado
    # ----------------------------------------------------------------------

    def _smooth_values(self, values: List[float]) -> List[float]:
        if len(values) < self.smoothing_window:
            return values

        try:
            return list(
                savgol_filter(
                    values,
                    self.smoothing_window,
                    self.poly_order,
                    mode='nearest'
                )
            )
        except ValueError:
            return values

    # ----------------------------------------------------------------------
    # Interpolación
    # ----------------------------------------------------------------------

    def _interpolate_last(self, values: List[Optional[np.ndarray]]) -> np.ndarray:
        """Interpola el último valor cuando es None."""
        if len(values) < 3:
            return np.array([0.0, 0.0])
        v1 = values[-2]
        v2 = values[-3]
        if v1 is None or v2 is None:
            return np.array([0.0, 0.0])
        return v1 + (v1 - v2)

    # ----------------------------------------------------------------------
    # Suavizado de posiciones
    # ----------------------------------------------------------------------

    def _smooth_positions_array(self, positions: List[Optional[np.ndarray]]) -> List[np.ndarray]:
        if len(positions) == 0:
            return []

        xs = [p[0] for p in positions if p is not None]
        ys = [p[1] for p in positions if p is not None]

        xs_smooth = self._smooth_values(xs)
        ys_smooth = self._smooth_values(ys)

        return [np.array([x, y]) for x, y in zip(xs_smooth, ys_smooth)]

    # ----------------------------------------------------------------------
    # Procesamiento principal
    # ----------------------------------------------------------------------

    def process_frame(
        self,
        frame_num: int,
        players_frame: Dict[int, PlayerStateModel],   # track_id → TrackDetailPlayer
        db: Session,                         # SQLAlchemy session
        model_class                          # SQLAlchemy model a guardar
    ):
        """
        Procesa un frame completo y persiste todos los jugadores.
        """

        inserts = []

        for track_id, track_detail in players_frame.items():
            track_content = track_detail.to_dict()

            pos = track_content.get('position')  # np.array([x, y]) o None

            # Inicializar buffers
            if track_id not in self.position_history:
                self.position_history[track_id] = []
            if track_id not in self.speed_history:
                self.speed_history[track_id] = []

            # Agregar posición al historial
            self.position_history[track_id].append(pos)

            # Mantener ventana deslizante N
            if len(self.position_history[track_id]) > self.history_size:
                self.position_history[track_id].pop(0)

            # Interpolación si no hay detección
            if self.position_history[track_id][-1] is None:
                interpolated = self._interpolate_last(self.position_history[track_id])
                self.position_history[track_id][-1] = interpolated
                pos = interpolated

            # Suavizado
            smooth_positions = self._smooth_positions_array(self.position_history[track_id])
            smoothed_pos = smooth_positions[-1]

            # Velocidad
            if len(smooth_positions) >= 2:
                dist_m = measure_scalar_distance(
                    smooth_positions[-1], smooth_positions[-2]
                )
                speed_kmh = (dist_m * self.frame_rate) * 3.6
            else:
                speed_kmh = 0.0

            self.speed_history[track_id].append(speed_kmh)
            if len(self.speed_history[track_id]) > self.history_size:
                self.speed_history[track_id].pop(0)

            smooth_speed_kmh = self._smooth_values(self.speed_history[track_id])[-1]

            # Aceleración
            if len(self.speed_history[track_id]) >= 2:
                v1 = self.speed_history[track_id][-1]
                v2 = self.speed_history[track_id][-2]
                acceleration = (v1 - v2) / (1 / self.frame_rate)
            else:
                acceleration = 0.0

            # Distancia incremental
            if len(smooth_positions) >= 2:
                incremental_dist = measure_scalar_distance(
                    smooth_positions[-1], smooth_positions[-2]
                )
            else:
                incremental_dist = 0.0

            # Distancia total histórica
            total_distance = float(sum(
                measure_scalar_distance(p1, p2)
                for p1, p2 in zip(smooth_positions[:-1], smooth_positions[1:])
            ))

            # Sprint
            is_sprint = smooth_speed_kmh >= self.sprint_threshold

            row = model_class(
                track_id=track_id,
                frame=frame_num,
                pos_x=float(smoothed_pos[0]),
                pos_y=float(smoothed_pos[1]),
                speed_kmh=float(smooth_speed_kmh),
                acceleration=float(acceleration),
                incremental_distance=float(incremental_dist),
                total_distance=float(total_distance),
                is_sprint=is_sprint,
            )
            inserts.append(row)

        # Insertar todo junto (mejor rendimiento)
        db.bulk_save_objects(inserts)
        db.commit()
