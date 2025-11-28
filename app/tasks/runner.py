import time
import tracemalloc

import numpy as np
from torch import R
from app.entities.interfaces import RecordCollectionBase
from app.entities.collections import TrackCollectionBall, TrackCollectionPlayer, TrackCollectionHeatmapPoint 
from app.entities.models import HeatmapPointModel, BallEventModel, PlayerStateModel
from app.layers.infraestructure.validation import (calculate_interpolation_error,
                                                   check_speed_consistency)
from app.modules.camera_movement_estimator import \
    CameraMovementEstimator
from app.modules.player_ball_assigner import \
    PlayerBallAssigner
from app.modules.plotting import generate_diagrams
from app.modules.services import read_video, extract_player_images
from app.modules.speed_and_distance_estimator import SpeedAndDistanceEstimator
from app.modules.team_assigner import TeamAssigner
from app.entities.trackers import ( 
    BallTracker, PlayerTracker)
from app.modules.trackers import TrackerService
from app.modules.view_transformer import ViewTransformer
from sqlalchemy.orm import Session

def generate_id(self, obj):
        return obj.track_id

def run(db: Session):

    # -----------------------------
    # MÉTRICAS Y CONFIGURACIÓN
    # -----------------------------
    tracemalloc.start()
    start_time = time.time()

    metrics = {
        "processing_time": [],
        "memory_usage": [],
        "ball_detection": {"detected": 0, "interpolated": 0},
        "interpolation_error": 0.0,
        "velocity_inconsistencies": {"players": 0, "referees": 0},
    }

    # -----------------------------
    # LECTURA DEL VIDEO
    # -----------------------------
    video_stream = read_video("./app/res/input_videos/08fd33_4.mp4")
    if not video_stream:
        print("Error: No frames read from video")
        return

    tracker = TrackerService(
        "./app/res/models/best.torchscript",
        use_half_precision=True
    )

    # -----------------------------
    # COLECCIONES
    # -----------------------------
    player_records = TrackCollectionPlayer(db)
    player_records.orm_model = PlayerStateModel

    ball_records = TrackCollectionBall(db)
    ball_records.orm_model = BallEventModel

    heatmap_records = TrackCollectionHeatmapPoint(db)

    view_transformer = ViewTransformer()
    speed_and_distance = SpeedAndDistanceEstimator()
    team_assigner = TeamAssigner()
    player_assigner = PlayerBallAssigner()

    # -----------------------------
    # FRAME INICIAL PARA CAMERA MOVEMENT
    # -----------------------------
    try:
        first_frame, dt = next(video_stream)
    except StopIteration:
        print("Error: Video is empty")
        return

    camera_movement_estimator = CameraMovementEstimator(first_frame)

    # ==========================================================================
    #                               LOOP PRINCIPAL
    # ==========================================================================
    for frame_num, (frame, dt) in enumerate(video_stream):

        # -------------------------------------------------------
        # 1. Estimar movimiento de cámara
        # -------------------------------------------------------
        camera_movement = camera_movement_estimator.update(frame)

        # -------------------------------------------------------
        # 2. TRACKING DE OBJETOS (jugadores + balón)
        # -------------------------------------------------------
        for collection in (player_records, ball_records):
            tracker.get_object_tracks(frame, frame_num, collection)

            last_track = collection.get_last(db)
            if last_track is None:
                continue

            # Calcular centro y bbox inmediatamente
            tracker.add_position_to_track(last_track)

            # Aplicar compensación de movimiento de cámara
            camera_movement_estimator.add_adjust_positions_to_tracks(
                collection=collection,
                camera_movement_per_frame=camera_movement,
                track=last_track
            )

            # Homografía al campo 2D
            view_transformer.add_transformed_positions(collection)

        # -------------------------------------------------------
        # 3. VALIDAR QUE EL TRACKER DE BALÓN ES CORRECTO
        # -------------------------------------------------------
        ball_tracker = tracker.get_tracker("ball")
        if not isinstance(ball_tracker, BallTracker):
            raise TypeError("Retrieved tracker is not an instance of BallTracker")

        # -------------------------------------------------------
        # 4. MÉTRICAS DE DETECCIÓN DEL BALÓN
        # -------------------------------------------------------
        ball_frames = ball_records.get_all()
        detected = sum(1 for _, t in ball_frames if any(obj.bbox for obj in t.values()))
        total = len(ball_frames)

        metrics["ball_detection"] = {
            "detected": detected,
            "interpolated": total - detected,
        }

        # -------------------------------------------------------
        # 5. ESTIMAR VELOCIDAD/DISTANCIA DEL ÚLTIMO JUGADOR
        # -------------------------------------------------------
        last_player = player_records.get_last(db)
        if last_player:
            speed_and_distance.process_track(
                frame_num=frame_num,
                track_id=last_player.track_id,
                track=last_player,
                db=db,
                model_class=PlayerStateModel,
            )

        # -------------------------------------------------------
        # 6. ASIGNACIÓN DEL BALÓN A UN JUGADOR
        # -------------------------------------------------------
        players = player_records.get_all()
        team_ball_control = []

        for (frame_i, ball_track) in ball_frames:

            # FRAME SIN BALÓN
            if not ball_track:
                last = team_ball_control[-1] if team_ball_control else -1
                team_ball_control.append(last)
                continue

            # Solo un balón por frame
            ball_detail = next(iter(ball_track.values()))
            ball_bbox = ball_detail.bbox

            if ball_bbox is None:
                last = team_ball_control[-1] if team_ball_control else -1
                team_ball_control.append(last)
                continue

            # -----------------------------
            # ASIGNACIÓN A JUGADOR
            # -----------------------------
            assigned_player_id = player_assigner.assign_ball_to_player(
                ball_bbox,
                players
            )

            if assigned_player_id == -1:
                last = team_ball_control[-1] if team_ball_control else -1
                team_ball_control.append(last)
                continue

            # Ubicar el jugador asignado en ese frame real
            player = player_records.get_record_for_frame(
                assigned_player_id,
                frame_i
            )

            if not player:
                last = team_ball_control[-1] if team_ball_control else -1
                team_ball_control.append(last)
                continue

            # Marcar posesión
            player_records.patch(
                player.to_dict()["id"],
                {"has_ball": True}
            )

            # Obtener equipo
            team = team_assigner.get_player_team(frame, player)
            team_ball_control.append(team if team is not None else -1)

        # -------------------------------------------------------
        # 7. MÉTRICAS FINALES
        # -------------------------------------------------------
        snapshot = tracemalloc.take_snapshot()
        total_mem = sum(stat.size for stat in snapshot.statistics("lineno")) / (1024 * 1024)

        if not metrics["memory_usage"]:
            metrics["memory_usage"].append(total_mem)

    # ==========================================================================
    #                           FIN DEL VIDEO
    # ==========================================================================
    total_time = time.time() - start_time

    print("\n" + "=" * 50)
    print("        RESUMEN FINAL DEL PROCESAMIENTO")
    print("=" * 50)
    print(f"Tiempo total: {total_time/60:.2f} min")
    print(f"Memoria máxima usada: {max(metrics['memory_usage']):.2f} MB")
    print(f"Frames balón detectado: {metrics['ball_detection']['detected']}")
    print(f"Frames balón interpolado: {metrics['ball_detection']['interpolated']}")

    return video_stream, metrics

def generate_media(video_stream, records_collection: TrackCollectionPlayer, db: Session):
    extract_player_images(video_stream, records_collection, './app/res/output_images/')
    generate_diagrams(db=db)
