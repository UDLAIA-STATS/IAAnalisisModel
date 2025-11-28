import logging
import json
from typing import Dict, Hashable, Optional

import numpy as np
import supervision as sv

from app.entities.interfaces.tracker_base import Tracker
from app.entities.interfaces.record_collection_base import RecordCollectionBase


class BallTracker(Tracker):
    """
    Guarda eventos del balón en la colección (BallEventModel).
    - No usa TrackDetailBase.
    - Usa FIXED_BALL_ID sólo internamente si hace falta; el modelo BallEventModel
      está indexado por frame_index.
    """

    def __init__(self):
        super().__init__()

    @staticmethod
    def _bbox_to_center_xy(bbox: list) -> tuple[float, float]:
        x1, y1, x2, y2 = bbox
        cx = float((x1 + x2) / 2.0)
        cy = float((y1 + y2) / 2.0)
        return cx, cy
    
    def reset(self):
        """
        Resetea el estado interno del tracker.
        """
        # BallTracker no mantiene estado interno por ahora
        pass

    def get_object_tracks(
        self,
        detection_with_tracks: sv.Detections,
        cls_names_inv: dict[str, int],
        frame_num: int,
        detection_supervision: sv.Detections,
        tracks_collection: RecordCollectionBase
    ):
        """
        Extrae la primera detección de clase 'ball' (si existe) y persiste un
        registro en BallEventModel via tracks_collection.post/patch.
        """

        if detection_with_tracks is None:
            return

        xyxy = getattr(detection_with_tracks, "xyxy", None)
        class_ids = getattr(detection_with_tracks, "class_id", None)

        if xyxy is None or class_ids is None:
            return

        # normalizar arrays
        try:
            class_ids_arr = np.asarray(class_ids)
            xyxy_arr = np.asarray(xyxy)
        except Exception:
            class_ids_arr = class_ids
            xyxy_arr = xyxy

        ball_class_idx = cls_names_inv.get("ball")
        if ball_class_idx is None:
            logging.debug("[BallTracker] 'ball' class not present in cls_names_inv")
            return

        # mascara y existencia
        try:
            mask = class_ids_arr == ball_class_idx
        except Exception:
            mask = (class_ids_arr == ball_class_idx)

        if not getattr(mask, "any", lambda: False)():
            return

        # tomar la primera bbox detectada de balón
        try:
            ball_bbox = xyxy_arr[mask][0].tolist()
        except Exception:
            logging.debug("[BallTracker] Error extracting ball bbox")
            return

        cx, cy = self._bbox_to_center_xy(ball_bbox)

        payload = {
            "frame_index": int(frame_num),
            "x": float(cx),
            "y": float(cy),
            "z": 0.0,
            "owner_id": None
        }

        # Intentar buscar registro existente usando la API genérica
        existing = None
        try:
            if hasattr(tracks_collection, "get_record_for_frame"):
                existing = tracks_collection.get_record_for_frame(
                    track_id=0,        # para ball, track_id no aplica; método puede ignorarlo
                    frame_index=int(frame_num)
                )
        except Exception:
            existing = None

        # Si get_record_for_frame no es suficientemente específico (ej. no filtra por frame_index),
        # hacer una consulta directa por frame_index
        if existing is None:
            try:
                orm = getattr(tracks_collection, "orm_model", None)
                db = getattr(tracks_collection, "db", None)
                if orm is not None and db is not None and hasattr(orm, "frame_index"):
                    existing = (
                        db.query(orm)
                        .filter(orm.frame_index == int(frame_num))
                        .first()
                    )
            except Exception:
                existing = None

        try:
            if existing:
                tracks_collection.patch(existing.id, payload)
            else:
                tracks_collection.post(payload)
        except Exception as e:
            logging.exception(f"[BallTracker] DB error persisting ball event: {e}")
