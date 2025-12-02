from datetime import datetime, timezone
import json
import pathlib
from sqlalchemy import create_engine

from app.entities.collections.track_collections import TrackCollectionPlayer
from app.modules.services.database import Base
from sqlalchemy.orm import sessionmaker, Session

from app.tasks.upload import upload

async def process_video_async(video_name: str, match_id: int):
    """
    Ejecuta el análisis en segundo plano con una BD en memoria aislada.
    """
    engine = create_engine("sqlite://", echo=False, connect_args={"check_same_thread": False})
    Base.metadata.create_all(bind=engine)
    SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
    db = SessionLocal()

    try:
        print(f"Ejecutando análisis en background para video: {video_name}")
        await process_run(
            db=db,
            video_name=video_name,
            match_id=match_id,
            db_session_factory=SessionLocal)
        print("Análisis finalizado.")
    except Exception as e:
        print(f"Error en análisis: {str(e)}")
    finally:
        db.close()
        engine.dispose()

async def process_run(db: Session, video_name: str, match_id: int, db_session_factory):
    
    try:
        from app.tasks.runner import run_analysis
        await run_analysis(db=db, video_name=video_name, match_id=match_id)
        await export_data(db, match_id)

    except Exception as e:
        print(f"[ERROR procesando video]: {e}")

    
async def export_data(db: Session, match_id: int):
    exporter = TrackCollectionPlayer(db)
    records = exporter.get_all()

    export_data = [
        {
        "id": r.id,
        "player_id": r.player_id,
        "team": r.team,
        "color": r.color,
        "frame_index": r.frame_index,
        "bbox": r.get_bbox(),

        "x": r.x,
        "y": r.y,
        "z": r.z,

        "ball_x": r.ball_x,
        "ball_y": r.ball_y,
        "ball_z": r.ball_z,
        "has_ball": r.has_ball,
        "ball_possession_time": r.ball_possession_time,
        "ball_owner_id": r.ball_owner_id,

        "distance": r.distance,
        "incremental_distance": r.incremental_distance,
        "speed": r.speed,
        "acceleration": r.acceleration,
        "is_sprint": r.is_sprint,

        "time_visible": r.time_visible,
        "timestamp_ms": r.timestamp_ms,
        }
        for r in records
    ]

    out_dir = pathlib.Path("exports")
    out_dir.mkdir(exist_ok=True)

    output_file = out_dir / f"player_states_match_{match_id}.json"

    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(export_data, f, indent=2, ensure_ascii=False)
        
    file_bytes = output_file.read_bytes()
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%d%H%M%S")
    key = f"match_{match_id}/{timestamp}_{output_file.name}"
    upload(
        key=key,
        file_bytes=file_bytes,
    )