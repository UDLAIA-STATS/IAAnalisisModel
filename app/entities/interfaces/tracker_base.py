from abc import ABC, abstractmethod
import supervision as sv
from app.entities.interfaces.record_collection_base import RecordCollectionBase


class Tracker(ABC):
    """
    Interfaz base para trackers específicos (players, ball, referees, etc).
    - NO deben ejecutar el detector.
    - NO deben crear su propio ByteTrack.
    - Deben implementar get_object_tracks que recibe detecciones ya trackeadas.
    """

    def __init__(self, model):
        pass

    @abstractmethod
    def get_object_tracks(
            self,
            detection_with_tracks: sv.Detections,
            cls_names_inv: dict[str, int],
            frame_num: int,
            detection_supervision: sv.Detections,
            tracks_collection: RecordCollectionBase) -> None:
        """
        Procesa detecciones *ya trackeadas* por el servicio:
        - detection_with_tracks: sv.Detections con atributos de tracking (id, etc.)
        - cls_names_inv: mapping id->nombre de clase para interpretar label indices
        - frame_num: numero de frame relativo al batch/frame procesado
        - detection_supervision: detecciones originales en formato supervision (sin tracks)
        - tracks_collection: repo/collection para persistir resultados
        """
        print("Tracker.get_object_tracks called.")
        raise NotImplementedError

    @abstractmethod
    def reset(self) -> None:
        """
        Resetea el estado interno del tracker.
        NO toca la DB.
        """
        raise NotImplementedError
