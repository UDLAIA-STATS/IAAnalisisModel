from typing import Dict, Mapping
from app.layers.domain.tracks.track_detail import TrackDetailBase
from app.layers.domain.utils.singleton import Singleton


class TrackCollection(metaclass=Singleton):
    """
    Clase que representa una colección de tracks (seguimientos) de entidades
    dentro de un video o secuencia de imágenes, diferenciados por tipo:
    jugadores ("players") y balón ("ball").

    Estructura interna de `self.tracks`:
        self.tracks: Dict[str, Dict[int, Dict[int, TrackDetailBase]]]

        - str  : Tipo de entidad ("players" o "ball").
        - int  : Número de frame.
        - int  : ID del track (track_id).
        - value: Objeto `TrackDetailBase` que contiene la información del track.

    Esta clase sigue el patrón Singleton, asegurando que solo exista una
    instancia de `TrackCollection` en toda la aplicación.
    """

    def __init__(self):
        """
        Inicializa la colección de tracks con dos llaves principales:
        - "players": Diccionario para almacenar los tracks de jugadores.
        - "ball"   : Diccionario para almacenar los tracks del balón.
        """
        super().__init__()
        self.tracks: Dict[str, Dict[int, Dict[int, TrackDetailBase]]] = {
            "players": {},
            "ball": {}
        }

    def exists_track_in_collection(
            self,
            collection: Mapping[int, Mapping[int, TrackDetailBase]],
            frame_num: int,
            frame_id: int | None = None
    ) -> bool:
        """
        Verifica si existe un frame dentro de una colección de tracks.

        Args:
            collection (Mapping[int, Sequence[TrackDetailBase]]): Colección indexada por número de frame.
            frame_num (int): Número del frame a comprobar.

        Returns:
            bool: True si el frame existe en la colección, False en caso contrario.
        """
        return frame_num in collection and (frame_id is None or frame_id in collection[frame_num])

    def add_track(
            self,
            entity_type: str,
            frame_num: int,
            track_detail: TrackDetailBase) -> None:
        """
        Agrega un track a la colección para una entidad y frame específico.

        Args:
            entity_type (str): Tipo de entidad ("players" o "ball").
            frame_num (int): Número de frame.
            track_detail (TrackDetailBase): Objeto que contiene la información del track.

        Raises:
            ValueError: Si el tipo de entidad no es válido.
        """
        if entity_type not in self.tracks:
            raise ValueError(f"Tipo de entidad '{entity_type}' no reconocido.")

        # Inserta el track en la jerarquía: entidad → frame → track_id → track_detail

        self.tracks[entity_type] \
            .setdefault(frame_num, {}) \
            .setdefault(track_detail.track_id or -1, track_detail)

    def update_track(
            self,
            entity_type: str,
            frame_num: int,
            track_id: int,
            track_detail: TrackDetailBase) -> None:
        """
        Actualiza los atributos de un track existente en la colección.

        Args:
            entity_type (str): Tipo de entidad ("players" o "ball").
            frame_num (int): Número de frame donde está registrado el track.
            track_id (int): Identificador único del track.
            track_detail (TrackDetailBase): Objeto con los datos a actualizar.

        Raises:
            ValueError: Si el tipo de entidad no es válido.
        """
        if entity_type not in self.tracks:
            raise ValueError(f"Tipo de entidad '{entity_type}' no reconocido.")

        collection = self.tracks[entity_type]
        # Se pasa el __dict__ para propagar dinámicamente todos los atributos del track
        if self.exists_track_in_collection(collection, frame_num, track_id):
            self._update_track_in_collection(collection, frame_num, track_id, track_detail)
        else:
            self.add_track(entity_type, frame_num, track_detail)

    def _update_track_in_collection(
            self,
            collection: Mapping[int, Mapping[int, TrackDetailBase]],
            frame_num: int,
            track_id: int,
            track_detail: TrackDetailBase
    ) -> None:
        """
        Actualiza un track específico dentro de una colección dada.

        Args:
            collection (Mapping[int, Mapping[int, TrackDetailBase]]): Colección organizada por frame y track_id.
            frame_num (int): Número de frame en el que se encuentra el track.
            track_id (int): Identificador del track.
            **kwargs: Atributos clave-valor para actualizar en el track.

        Nota:
            - Si el frame o el track_id no existen, no se hace nada.
            - Se asume que `TrackDetailBase` o sus derivados implementan un método `update`.
        """
        print(
            f"Attempting to update track in collection for frame {frame_num} and track ID {track_id}")
        if frame_num not in collection:
            return

        print(f"Frame {frame_num} found. Checking for track ID {track_id}...")
        frames = collection[frame_num]
        if track_id not in frames:
            return

        # Recupera el track y aplica los cambios
        track = frames[track_id]
        track.update(**track_detail.__dict__)
