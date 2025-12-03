import pathlib
import time
from typing import Generator, List

import cv2
from cv2.typing import MatLike

from app.entities.interfaces.record_collection_base import RecordCollectionBase
from app.entities.models import PlayerStateModel


def read_video(video_path: str, batch_size: int = 16) -> Generator[List[tuple[MatLike, float]]]:
    try:
        print(f"Abriendo video para lectura: {video_path}...")
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise FileNotFoundError(f"No se pudo abrir el video: {video_path}")
        
        batch = []
        last_time = time.time()
        frame_count = 0

        while True:
            frame_count += 1
            print(f"Leyendo frame {frame_count + 1}...")
            ret, frame = cap.read()
            print(f"Frame leído: {'sí' if ret else 'no'}")
            if not ret or not frame.any():
                break
            print("Frame válido obtenido.")
            now = time.time()
            dt = now - last_time
            last_time = now
            print(f"Tiempo desde último frame: {dt:.4f} segundos")
            if not frame.any():
                print("Frame vacío detectado, saltando...")
                continue
            batch.append((frame, dt))
            print(f"Tamaño del batch actual: {len(batch)}")

            if len(batch) >= batch_size:
                print(f"Rindiendo batch de tamaño {batch_size}...")
                yield batch
                batch = []
            print("Continuando con el siguiente frame...")

        if batch:
            print(f"Rindiendo último batch de tamaño {len(batch)}...")
            yield batch
        print("Finalizando lectura de video.")
        cap.release()
    except Exception as e:
        print(f"Error leyendo el video {video_path}: {e}")
        raise e


def extract_player_images(
    frame: MatLike,
    frame_index: int,
    player,
    output_folder: str,
    player_image_counts: dict, 
    last_frame_taken: dict,
    images_per_player: int = 3,
    frame_skip: int = 5,
):
    """
    Extrae imágenes de torso/cara de jugadores PARA EL FRAME ACTUAL.
    Esta función debe llamarse en cada iteración del lector de video.

    :param frame: Frame actual del video
    :param frame_index: Índice del frame actual
    :param records: Lista de records del frame actual (track_id, bbox, etc.)
    :param output_folder: Carpeta de salida
    :param images_per_player: Máximo de imágenes por jugador
    :param frame_skip: Diferencia mínima entre capturas del mismo jugador
    :param player_image_counts: Mapa acumulado {player_id: cantidad}
    :param last_frame_taken: Mapa {player_id: ultimo_frame_guardado}
    """
    try:
        folder = pathlib.Path(output_folder)
        folder.mkdir(parents=True, exist_ok=True)

        if player_image_counts is None:
            player_image_counts = {}

        if last_frame_taken is None:
            last_frame_taken = {}

        h, w = frame.shape[:2]

        # Procesar solo records del frame actual
        record = player.to_dict()
        player_id = record.get("player_id", None)
        bbox = player.get_bbox()

        if bbox is None or len(bbox) != 4:
            return player_image_counts, last_frame_taken

        # Máximo por jugador
        count = player_image_counts.get(player_id, 0)
        if count >= images_per_player:
            return player_image_counts, last_frame_taken

        # Saltar frames cercanos
        last_f = last_frame_taken.get(player_id, -frame_skip - 1)
        if frame_index - last_f < frame_skip:
            return player_image_counts, last_frame_taken
        
        h, w = frame.shape[:2]

        # Validar bounding box
        x1, y1, x2, y2 = map(int, bbox)

        x1 = max(0, min(x1, w - 1))
        x2 = max(0, min(x2, w - 1))
        y1 = max(0, min(y1, h - 1))
        y2 = max(0, min(y2, h - 1))

        if x2 <= x1 or y2 <= y1:
            return player_image_counts, last_frame_taken

        if (x2 - x1) < 10 or (y2 - y1) < 10:
            return player_image_counts, last_frame_taken
        
        torso_y2 = y1 + int((y2 - y1) * 0.6)
        crop = frame[y1:torso_y2, x1:x2]
        
        if crop.size == 0:
            return player_image_counts, last_frame_taken

        # Guardar imagen
        filename = folder / f"player_{player_id}_img_{count+1}_frame_{frame_index}.png"
        cv2.imwrite(str(filename), crop)
        player_image_counts.update({player_id: count + 1})
        last_frame_taken.update({player_id: frame_index})

        return player_image_counts, last_frame_taken
    except Exception as e:
        print(f"Error extrayendo imagen de jugador { player.to_dict().get('player_id', None) } en frame {frame_index}: {e}")
        raise e