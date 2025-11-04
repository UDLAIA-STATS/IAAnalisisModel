import pathlib

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
    video_frames: list[MatLike],
    tracks_collection,
    output_folder: str
):
    folder = pathlib.Path(output_folder)
    if not folder.exists():
        folder.mkdir(parents=True, exist_ok=True)

    for frame_num, player_track in tracks_collection.tracks["players"].items():
        for player_id, track in player_track.items():
            bbox = track.bbox
            x1, y1, x2, y2 = map(int, bbox)
            player_image = video_frames[frame_num][y1:y2, x1:x2]
            player_image_path = folder / f"frame_{frame_num}_player_{player_id}.png"
            cv2.imwrite(str(player_image_path), player_image)
