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
    # Initialize metrics and performance tracking
    tracemalloc.start()  # Start memory tracking
    start_time = time.time()
    metrics = {
        'processing_time': [],
        'memory_usage': [],
        'ball_detection': {'detected': 0, 'interpolated': 0},
        'interpolation_error': 0.0,
        'velocity_inconsistencies': {'players': 0, 'referees': 0}
    }

    # Lectura y extracción de frames del video
    video_stream = read_video('./app/res/input_videos/08fd33_4.mp4')
    if not video_stream:
        print("Error: No frames read from video")
        return

    # Inicializa los trackers para el reconocimiento de objetos
    tracker = TrackerService("./app/res/models/best.torchscript", use_half_precision=True)
    

    # Entidades necesarios para el almacenamiento y procesamiento de tracks
    player_records = TrackCollectionPlayer(db)
    player_records.orm_model = PlayerStateModel
    ball_records = TrackCollectionBall(db)
    ball_records.orm_model = BallEventModel
    
    heatmap_point_records = TrackCollectionHeatmapPoint(db)
    view_transformer = ViewTransformer()
    speed_and_distance_estimator = SpeedAndDistanceEstimator()
    team_assigner = TeamAssigner()
    player_assigner = PlayerBallAssigner()
    first_frame = None
    try: 
        first_frame, dt = next(video_stream)
    except StopIteration:
        print("Error: Video stream is empty")
        return
    
    camera_movement_estimator = CameraMovementEstimator(first_frame)

    for frame_num, (frame, dt) in enumerate(video_stream):
        camera_movement = camera_movement_estimator.update(frame)
        records = [ player_records, ball_records]
        for collection in records:
            tracker.get_object_tracks(frame, frame_num, collection)
            # Calcular posiciones centrales inmediatamente
            last_track = collection.get_last(db)

            if last_track is None:
                continue

            tracker.add_position_to_track(last_track)
            camera_movement_estimator.add_adjust_positions_to_tracks(
                collection=collection,
                camera_movement_per_frame=camera_movement,
                track=last_track
            )
            view_transformer.add_transformed_positions(collection)

    
        ball_tracker = tracker.get_tracker('ball')

        if not isinstance(ball_tracker, BallTracker):
            raise TypeError("Retrieved tracker is not an instance of BallTracker")

        detected_frames = sum(
            any(track.bbox for track in tracks.values())
            for tracks in ball_records.get_all()
        )

        total_frames = len(ball_records.get_all())


        # Count ball detections (efficiently)
        metrics["ball_detection"] = {
            "detected": detected_frames,
            "interpolated": len(ball_records.get_all()) - detected_frames
        }

        # Speed and distance estimation
        player_frame = player_records.get_last(db) 
        
        if player_frame is None or not isinstance(player_frame, PlayerStateModel):
            return
        speed_and_distance_estimator.process_track(
            frame_num=frame_num,
            track_id=player_frame.to_dict()['id'],
            track=player_frame,
            db=db,
            model_class=PlayerStateModel,
        )

        print("Assigning ball to players...")

        players = player_records.get_all()
        balls = ball_records.get_all()
        team_ball_control = []
        
        for frame_num, ball_track in balls:
            # -----------------------
            # 1. Obtener bbox del balón
            # -----------------------
            if not ball_track:
                team_ball_control.append(team_ball_control[-1] if team_ball_control else -1)
                continue

            # Solo un balón por frame
            ball_detail = next(iter(ball_track.values()))
            ball_bbox = ball_detail.bbox
            if ball_bbox is None:
                team_ball_control.append(team_ball_control[-1] if team_ball_control else -1)
                continue

            # -----------------------
            # 2. Asignar balón a jugador
            # -----------------------
            assigned_player_id = player_assigner.assign_ball_to_player(ball_bbox, players)
            print("Assigned player: ", assigned_player_id)

            if assigned_player_id == -1:
                # Nadie cerca: repetir último equipo
                team_ball_control.append(team_ball_control[-1] if team_ball_control else -1)
                continue

            # -----------------------
            # 3. Obtener player real del frame
            # -----------------------
            player = player_records.get_record_for_frame(assigned_player_id, frame_num)
            print("Actual player track: ", player)

            if not player:
                team_ball_control.append(team_ball_control[-1] if team_ball_control else -1)
                continue

            # Marcar posesión
            player_records.patch(player.to_dict()['id'], {'has_ball': True})

            # -----------------------
            # 4. Determinar equipo del jugador
            # -----------------------
            team = team_assigner.get_player_team(frame, player)
            team_ball_control.append(team if team is not None else -1)


        # Final metrics report
        total_time = time.time() - start_time
        snapshot = tracemalloc.take_snapshot()
        top_stats = snapshot.statistics('lineno')
        total_memory = sum(stat.size for stat in top_stats) / (1024 * 1024)  # MB

        # Ensure we have memory measurements
        if not metrics['memory_usage']:

            # Add final memory measurement if no others exist
            metrics['memory_usage'].append(total_memory)

        print("\n" + "=" * 50)
        print("RESUMEN DE MÉTRICAS DE RENDIMIENTO")
        print("=" * 50)
        print(f"Tiempo total de procesamiento: {total_time/60:.2f} min")
        print(f"Uso máximo de memoria: {max(metrics['memory_usage']):.2f} MB")
        print(f"Inconsistencias de velocidad: Jugadores={metrics['velocity_inconsistencies']['players']}" )
        print(f"Error de interpolación: {metrics['interpolation_error']:.4f}")

def generate_media(video_stream, records_collection: TrackCollectionPlayer, db: Session):
    extract_player_images(video_stream, records_collection, './app/res/output_images/')
    generate_diagrams(db=db)
