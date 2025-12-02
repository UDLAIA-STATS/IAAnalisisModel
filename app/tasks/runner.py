import time
import tracemalloc

from app.entities.collections import TrackCollectionBall, TrackCollectionPlayer 
from app.entities.models import BallEventModel, PlayerStateModel
from app.modules.camera_movement_estimator import \
    CameraMovementEstimator
from app.modules.player_ball_assigner import \
    PlayerBallAssigner
from app.modules.plotting import generate_diagrams
from app.modules.services import read_video, extract_player_images
from app.modules.services.r2_download import R2Downloader
from app.modules.services.verify_model import prepare_model
from app.modules.services.video_processing_service import extract_player_images
from app.modules.speed_and_distance_estimator import SpeedAndDistanceEstimator
from app.modules.team_assigner import TeamAssigner
from app.entities.trackers import ( 
    BallTracker)
from app.modules.trackers import TrackerService
from app.modules.view_transformer import ViewTransformer
from sqlalchemy.orm import Session

from app.tasks.upload_heatmaps import upload_heatmaps_for_extracted_players
from app.utils.routes import INPUT_VIDEOS_DIR, MODELS_DIR, OUTPUT_IMAGES_DIR


async def run_analysis(db: Session, video_name: str, match_id: int) -> None:

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
    
    print("Prepara el modelo si es necesario...")
    model_path = MODELS_DIR / "football_model.torchscript"
    prepare_model(
        model_path=model_path,
        source_path=model_path.parent)
    
    # Descarga video
    downloader = R2Downloader()
    
    video_name = "fb64992c-0a84-4fb5-8c3c-42f4ddbfda1c-1_720p.mkv"

    print(f"Descargando video {video_name}...")
    download_path = downloader.build_destination_path(key=video_name, base_dir=INPUT_VIDEOS_DIR.as_posix())
    downloader.stream_download(key=video_name, destination_path=download_path.as_posix())
    print(f"Video descargado en {INPUT_VIDEOS_DIR.as_posix()}")
    print(f"Video descargado en {download_path.as_posix()}")

    # -----------------------------
    # LECTURA DEL VIDEO
    # -----------------------------
    video_stream = read_video(download_path.as_posix())
    images_per_player = 3
    if not video_stream:
        print("Error: No frames read from video")
        return

    try:
        tracker = TrackerService(
            model_path.as_posix(),
            use_half_precision=True
        )

        # -----------------------------
        # COLECCIONES
        # -----------------------------
        player_records = TrackCollectionPlayer(db)
        player_records.orm_model = PlayerStateModel

        ball_records = TrackCollectionBall(db)
        ball_records.orm_model = BallEventModel

        view_transformer = ViewTransformer()
        speed_and_distance = SpeedAndDistanceEstimator()
        team_assigner = TeamAssigner()
        player_assigner = PlayerBallAssigner()
    except Exception as e:
        print(f"Error initializing services: {e}")
        raise e

    player_image_counts, last_frame_taken = {}, {}

    # -----------------------------
    # FRAME INICIAL PARA CAMERA MOVEMENT
    # -----------------------------
    try:
        first_frame, _ = next(video_stream)
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
        print("Estimando movimiento de cámara...")
        camera_movement = camera_movement_estimator.update(frame)

        # -------------------------------------------------------
        # 2. TRACKING DE OBJETOS (jugadores + balón)
        # -------------------------------------------------------
        print("Procesando frame en el tracker...")
        for collection in (player_records, ball_records):
            print("Obteniendo tracks de objetos...")
            tracker.get_object_tracks(frame, frame_num, db)

            print("Actualizando último track...")
            last_track = collection.get_last(db)
            if last_track is None:
                print("No hay track para actualizar, saltando...")
                continue
            print(f"Último track ID: {last_track.to_dict().get('player_id', None)}")

            # Calcular centro y bbox inmediatamente
            print("Añadiendo posición al track...")
            tracker.add_position_to_track(last_track)
            print(f"Posición añadida: {last_track.position}")

            # Aplicar compensación de movimiento de cámara
            print("Ajustando posiciones según movimiento de cámara...")
            camera_movement_estimator.add_adjust_positions_to_tracks(
                db=db,
                camera_movement_per_frame=camera_movement,
                track=last_track
            )
            print("Posiciones ajustadas.")

            # Homografía al campo 2D
            print("Aplicando transformación de vista...")
            view_transformer.add_transformed_positions(db)
            print("Transformación aplicada.")

        # -------------------------------------------------------
        # 3. VALIDAR QUE EL TRACKER DE BALÓN ES CORRECTO
        # -------------------------------------------------------
        print("Obteniendo tracker de balón...")
        ball_tracker = tracker.get_tracker("ball")
        print("Tracker de balón obtenido.")
        if not isinstance(ball_tracker, BallTracker):
            raise TypeError("Retrieved tracker is not an instance of BallTracker")

        # -------------------------------------------------------
        # 4. MÉTRICAS DE DETECCIÓN DEL BALÓN
        # -------------------------------------------------------
        print("Obteniendo métricas de detección del balón...")
        ball_frames = ball_records.get_all()
        print(f"Total frames con balón: {len(ball_frames)}")
        detected = sum(1 for _, t in ball_frames if any(obj.bbox for obj in t.values()))
        print(f"Frames con balón detectado: {detected}")
        total = len(ball_frames)
        print(f"Total frames: {total}")

        metrics["ball_detection"] = {
            "detected": detected,
            "interpolated": total - detected,
        }

        # -------------------------------------------------------
        # 5. ESTIMAR VELOCIDAD/DISTANCIA DEL ÚLTIMO JUGADOR
        # -------------------------------------------------------
        print("Estimando velocidad y distancia del último jugador...")
        last_player = player_records.get_last(db)
        if last_player:
            print(f"Último jugador: {last_player.player_id}")
            speed_and_distance.process_track(
                frame_num=frame_num,
                track_id=last_player.track_id,
                track=last_player,
                db=db,
                model_class=PlayerStateModel,
            )
            print("Velocidad y distancia estimadas.")

        # -------------------------------------------------------
        # 6. ASIGNACIÓN DEL BALÓN A UN JUGADOR
        # -------------------------------------------------------
        print("Asignando balón a jugador...")
        players = player_records.get_all()
        team_ball_control = []

        print("Asignando balón a jugador...")
        for (frame_i, ball_track) in ball_frames:
            # FRAME SIN BALÓN
            if not ball_track:
                last = team_ball_control[-1] if team_ball_control else -1
                team_ball_control.append(last)
                print("No hay balón en este frame, asignación por defecto.")
                continue

            # Solo un balón por frame
            print("Obteniendo detalle del balón...")
            ball_detail = next(iter(ball_track.values()))
            ball_bbox = ball_detail.bbox

            if ball_bbox is None:
                last = team_ball_control[-1] if team_ball_control else -1
                team_ball_control.append(last)
                print("No hay balón en este frame, asignación por defecto.")
                continue

            # -----------------------------
            # ASIGNACIÓN A JUGADOR
            # -----------------------------
            print("Asignando balón a jugador...")
            assigned_player_id = player_assigner.assign_ball_to_player(
                ball_bbox,
                players
            )

            if assigned_player_id == -1:
                print("No hay jugador asignado, asignación por defecto.")
                last = team_ball_control[-1] if team_ball_control else -1
                team_ball_control.append(last)
                continue

            # Ubicar el jugador asignado en ese frame real
            print(f"Jugador asignado: {assigned_player_id}")
            player = player_records.get_record_for_frame(
                assigned_player_id,
                frame_i
            )
            print(f"Jugador encontrado en frame: {player is not None}")

            if not player:
                print("Jugador no encontrado, asignación por defecto.")
                last = team_ball_control[-1] if team_ball_control else -1
                team_ball_control.append(last)
                continue

            # Marcar posesión
            print("Marcando posesión del jugador...")
            player_records.patch(
                player.to_dict()["id"],
                {"has_ball": True}
            )

            # Obtener equipo
            print("Obteniendo equipo del jugador...")
            team = team_assigner.get_player_team(frame, player)
            print(f"Equipo del jugador: {team}")
            team_ball_control.append(team if team is not None else -1)
            print("Posesión marcada.")

            player_image_counts, last_frame_taken = extract_player_images(
            frame=frame,
            frame_index=frame_num,
            player=last_player,
            images_per_player=images_per_player,
            output_folder=OUTPUT_IMAGES_DIR.as_posix(),
            player_image_counts=player_image_counts,
            last_frame_taken=last_frame_taken,
            )
            print("Imágenes extraídas.")
            
            if all(count >= images_per_player for count in player_image_counts.values()):
                last_frame_taken.clear() 
        
        snapshot = tracemalloc.take_snapshot()
        total_mem = sum(stat.size for stat in snapshot.statistics("lineno")) / (1024 * 1024)

        if not metrics["memory_usage"]:
            metrics["memory_usage"].append(total_mem)

    extracted_player_ids = set(player_image_counts.keys())
    print(f"Jugadores con imágenes extraídas {extracted_player_ids}")

    generate_diagrams(db)
    print("Diagramas generados.")
    await upload_heatmaps_for_extracted_players(db=db, match_id=match_id, extracted_player_ids=extracted_player_ids)
    print("Heatmaps subidos.")

    total_time = time.time() - start_time

    print("\n" + "=" * 50)
    print("        RESUMEN FINAL DEL PROCESAMIENTO")
    print("=" * 50)
    print(f"Tiempo total: {total_time/60:.2f} min")
    print(f"Memoria máxima usada: {max(metrics['memory_usage']):.2f} MB")
    print(f"Frames balón detectado: {metrics['ball_detection']['detected']}")
    print(f"Frames balón interpolado: {metrics['ball_detection']['interpolated']}")
