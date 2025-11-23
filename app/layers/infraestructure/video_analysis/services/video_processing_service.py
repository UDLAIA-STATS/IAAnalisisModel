import pathlib
from typing import List

import cv2
from cv2.typing import MatLike


def read_video(video_path: str) -> list[MatLike]:
    cap = cv2.VideoCapture(video_path)
    frames = []
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frames.append(frame)
    return frames


def save_video(ouput_video_frames, output_video_path: str):
    folder = pathlib.Path(output_video_path).parent
    if not folder.exists():
        folder.mkdir(parents=True, exist_ok=True)

    fourcc = cv2.VideoWriter.fourcc(*'XVID')
    out = cv2.VideoWriter(
        output_video_path,
        fourcc,
        24,
        (ouput_video_frames[0].shape[1],
         ouput_video_frames[0].shape[0]))
    for frame in ouput_video_frames:
        out.write(frame)
    out.release()
    
def extract_player_images(
    video_frames: List[MatLike],
    tracks_collection,
    output_folder: str
):
    folder = pathlib.Path(output_folder)
    folder.mkdir(parents=True, exist_ok=True)

    saved_ids = set() 

    for frame_num, player_track in tracks_collection.tracks["players"].items():
        for player_id, track in player_track.items():
            if player_id in saved_ids:
                continue

            bbox = track.bbox
            if bbox is None or len(bbox) != 4:
                continue  # Evita errores si el bbox no es válido

            x1, y1, x2, y2 = map(int, bbox)

            # Validación de límites dentro del frame
            frame = video_frames[int(frame_num)]
            h, w = frame.shape[:2]
            x1, y1 = max(0, x1), max(0, y1)
            x2, y2 = min(w, x2), min(h, y2)
            if x2 <= x1 or y2 <= y1:
                continue  # Bounding box inválido

            # Recorte del jugador
            player_image = frame[y1:y2, x1:x2]

            # Guardar imagen
            player_image_path = folder / f"player_{player_id}_frame_{frame_num}.png"
            cv2.imwrite(str(player_image_path), player_image)

            # Marcar este track_id como ya guardado
            saved_ids.add(player_id)
