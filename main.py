import time
import tracemalloc

import numpy as np
from app.layers.domain.collections.track_collection import TrackCollection
from app.layers.domain.tracks.track_detail import TrackDetailBase, TrackPlayerDetail
from app.layers.infraestructure.validation import (calculate_interpolation_error,
                                               check_speed_consistency)
from app.layers.infraestructure.video_analysis.camera_movement_estimator import \
    CameraMovementEstimator
from app.layers.infraestructure.video_analysis.player_ball_assigner import \
    PlayerBallAssigner
from app.layers.infraestructure.video_analysis.plotting import generate_diagrams
from app.layers.infraestructure.video_analysis.services import (read_video,
                                                            save_video)
from app.layers.infraestructure.video_analysis.speed_and_distance_estimator import \
SpeedAndDistanceEstimator
from app.layers.infraestructure.video_analysis.team_assigner import TeamAssigner
from app.layers.infraestructure.video_analysis.trackers.entities import (
    BallTracker, PlayerTracker)
from app.layers.infraestructure.video_analysis.trackers.services import \
    TrackerService
from app.layers.infraestructure.video_analysis.view_transformer import \
    ViewTransformer


def main():
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

    # Read Video
    video_frames = read_video('./app/res/input_videos/08fd33_4.mp4')
    if not video_frames:
        print("Error: No frames read from video")
        return

    # Initialize components
    tracker = TrackerService("./app/res/models/best.torchscript")
    tracker.create_tracker('players', PlayerTracker)
    tracker.create_tracker('ball', BallTracker)

    tracks_collection = TrackCollection()
    view_transformer = ViewTransformer()
    speed_and_distance_estimator = SpeedAndDistanceEstimator()
    team_assigner = TeamAssigner()
    player_assigner = PlayerBallAssigner()
    camera_movement_estimator = CameraMovementEstimator(video_frames[0])

    # Process video in bulk (not per-frame)
    tracker.get_object_tracks(
        video_frames,
        read_from_stub=True,
        stub_path='./app/res/stubs/track_stubs.pkl',
        tracks_collection=tracks_collection
    )

    # Get object positions
    tracker.add_position_to_tracks(tracks_collection=tracks_collection)

    # Estimate camera movement
    camera_movement_per_frame = camera_movement_estimator.get_camera_movement(
        video_frames,
        read_from_stub=True,
        stub_path='./app/res/stubs/camera_movement_stub.pkl'
    )
    camera_movement_estimator.add_adjust_positions_to_tracks(
        camera_movement_per_frame, tracks_collection=tracks_collection)

    # View Transformation
    view_transformer.add_transformed_position_to_tracks(tracks_collection=tracks_collection)

    # Interpolate Ball Positions
    #tracks["ball"].copy()
    ball_tracker = tracker.get_tracker('ball')

    if not isinstance(ball_tracker, BallTracker):
        raise TypeError("Retrieved tracker is not an instance of BallTracker")
    
    detected_frames = sum(
        1
        for frame_tracks in tracks_collection.tracks["ball"].values()
        if 1 in frame_tracks and getattr(frame_tracks[1], "bbox", None) is not None
    )

    total_frames = len(tracks_collection.tracks["ball"])

    ball_tracker.interpolate_ball_positions(tracks_collection.tracks["ball"])

    # Count ball detections (efficiently)
    metrics["ball_detection"] = {
        "detected": detected_frames,
        "interpolated": max(0, total_frames - detected_frames),
    }

    # Speed and distance estimation
    speed_and_distance_estimator.add_speed_and_distance_to_tracks(tracks_collection)

    # Assign Player Teams
    for frame_num, track_content in tracks_collection.tracks["players"].items():
        print("Assigning player teams...", tracks_collection.tracks["players"][frame_num].values())
        team_assigner.assign_team_color(
            video_frames[0], 
            tracks_collection.tracks["players"][frame_num])
        continue

    for frame_num, player_track in tracks_collection.tracks["players"].items():
        for player_id, track in player_track.items():
            team = team_assigner.get_player_team(
                video_frames[frame_num],
                track.bbox,
                player_id
            )
            player_tracker = TrackPlayerDetail(**track.model_dump())
            player_tracker.team = team
            player_tracker.team_color = team_assigner.team_colors[team]
            tracks_collection.update_track(
                entity_type="players",
                frame_num=frame_num,
                track_id=player_id,
                track_detail=player_tracker
            )
            

    # Assign Ball Acquisition
    team_ball_control = []
    print("Assigning ball to players...")
    print("Total frames to assign ball: ", len(tracks_collection.tracks["players"]))
    for frame_num, player_track in tracks_collection.tracks["players"].items():
        ball_bbox = tracks_collection.tracks['ball'][frame_num][1].bbox
        assigned_player = player_assigner.assign_ball_to_player(
            player_track, ball_bbox)
        print("Assigned player: ", assigned_player)

        if assigned_player != -1:
            player_base: TrackDetailBase = player_track[assigned_player]
            print("Actual player track: ", player_base)
            player: TrackPlayerDetail = TrackPlayerDetail(**player_base.model_dump())
            player.has_ball = True
            team_ball_control.append(player.team)
            tracks_collection.update_track(
                entity_type="players",
                frame_num=frame_num,
                track_id=assigned_player,
                track_detail=player
            )
        else:
            # Handle first frame case
            team_ball_control.append(
                team_ball_control[-1] if team_ball_control else -1)

    team_ball_control = np.array(team_ball_control)

    # Calculate metrics
    metrics['interpolation_error'] = calculate_interpolation_error(
        ball_tracker,  # Pass tracker instance
        tracks_collection.tracks['ball']
    )
    metrics['velocity_inconsistencies'] = check_speed_consistency(tracks_collection)

    # Draw output
    print("Team ball control array: ", team_ball_control)
    print("Total players frames: ", tracks_collection.tracks)
    output_video_frames = tracker.get_tracker('players').draw_annotations(
        video_frames, tracks_collection.tracks, team_ball_control)
    output_video_frames = camera_movement_estimator.draw_camera_movement(
        output_video_frames, camera_movement_per_frame)
    speed_and_distance_estimator.draw_speed_and_distance(
        output_video_frames, tracks_collection.tracks)

    # Save video
    save_video(output_video_frames, './app/res/output_videos/output_video.avi')

    # Generate diagrams (will save each metric separately)
    generate_diagrams(tracks=tracks_collection.tracks, metrics=metrics)

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
    print(f"Tiempo total de procesamiento: {total_time:.2f} s")
    print(f"Tiempo promedio por frame: {total_time / len(video_frames):.4f} s")
    print(f"Uso máximo de memoria: {max(metrics['memory_usage']):.2f} MB")
    print(f"Detección de balón: {metrics['ball_detection']['detected']} frames "
          f"({metrics['ball_detection']['detected'] /
          len(tracks_collection.tracks['ball']) * 100:.1f}%)")
    print(f"Inconsistencias de velocidad: Jugadores={
        metrics['velocity_inconsistencies']['players']}",
          f"Árbitros={metrics['velocity_inconsistencies']['referees']}")
    print(f"Error de interpolación: {metrics['interpolation_error']:.4f}")


if __name__ == '__main__':
    main()
