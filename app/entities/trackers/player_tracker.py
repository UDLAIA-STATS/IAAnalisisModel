import logging
import json

import numpy as np
import supervision as sv

from app.entities.interfaces.tracker_base import Tracker
from app.entities.interfaces.record_collection_base import RecordCollectionBase


class PlayerTracker(Tracker):
    """
    Extrae detecciones de jugadores y persiste en PlayerStateModel via
    TrackCollectionPlayer (RecordCollectionBase).
    Ya no depende de TrackPlayerDetail.
    """

    def __init__(self):
        super().__init__()

    @staticmethod
    def _bbox_center(bbox: list) -> tuple[float, float]:
        x1, y1, x2, y2 = bbox
        cx = float((x1 + x2) / 2.0)
        cy = float((y1 + y2) / 2.0)
        return cx, cy
    
    def reset(self) -> None:
        """
        Resetea el estado interno del tracker.
        """
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
        Para cada detección de 'player' persiste un registro:
        - player_id (track_id from tracker)
        - frame_index
        - bbox (JSON encoded)
        - x, y centro
        """

        if detection_with_tracks is None:
            return

        xyxy = getattr(detection_with_tracks, "xyxy", None)
        class_ids = getattr(detection_with_tracks, "class_id", None)
        tracker_ids = getattr(detection_with_tracks, "tracker_id", None)

        if xyxy is None or class_ids is None or tracker_ids is None:
            return

        # normalizar a numpy para poder enmascarar
        try:
            xyxy_arr = np.asarray(xyxy)
            class_ids_arr = np.asarray(class_ids)
            tracker_ids_arr = np.asarray(tracker_ids)
        except Exception:
            xyxy_arr = xyxy
            class_ids_arr = class_ids
            tracker_ids_arr = tracker_ids

        player_class_idx = cls_names_inv.get("player")
        if player_class_idx is None:
            logging.debug("[PlayerTracker] 'player' class not present in cls_names_inv")
            return

        try:
            mask = class_ids_arr == player_class_idx
        except Exception:
            mask = (class_ids_arr == player_class_idx)

        if not getattr(mask, "any", lambda: False)():
            return

        player_bboxes = xyxy_arr[mask]
        player_ids = tracker_ids_arr[mask]

        orm = getattr(tracks_collection, "orm_model", None)
        db = getattr(tracks_collection, "db", None)

        for bbox_arr, raw_tid in zip(player_bboxes, player_ids):
            if raw_tid is None:
                continue

            try:
                track_id = int(raw_tid)
            except Exception:
                logging.debug(f"[PlayerTracker] invalid track id: {raw_tid}")
                continue

            try:
                bbox_list = bbox_arr.tolist()
            except Exception:
                bbox_list = list(map(float, bbox_arr))

            cx, cy = self._bbox_center(bbox_list)

            payload = {
                "player_id": track_id,
                "frame_index": int(frame_num),
                "bbox": json.dumps(bbox_list),
                "x": float(cx),
                "y": float(cy),
                "z": 0.0,
                # leave other fields as defaults (speed, distance, etc.)
            }

            # Intentar encontrar registro existente:
            existing = None
            try:
                if hasattr(tracks_collection, "get_record_for_frame"):
                    existing = tracks_collection.get_record_for_frame(
                        track_id=track_id,
                        frame_index=int(frame_num)
                    )
            except Exception:
                existing = None

            # Si el método genérico no buscó por player_id, intentar consulta directa
            if existing is None and orm is not None and db is not None:
                try:
                    if hasattr(orm, "player_id") and hasattr(orm, "frame_index"):
                        existing = (
                            db.query(orm)
                            .filter(orm.player_id == track_id, orm.frame_index == int(frame_num))
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
                logging.exception(f"[PlayerTracker] DB error persisting player {track_id}: {e}")
