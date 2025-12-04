from datetime import datetime, timezone
import json
import pathlib
from sqlalchemy import create_engine

from app.entities.collections.track_collections import TrackCollectionPlayer
from app.entities.models.PlayerState import PlayerStateModel
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

    
async def export_data(db: Session, match_id: int, max_records: int = 100000):
    records = (db.query(PlayerStateModel)
               .order_by(PlayerStateModel.id)
               .limit(max_records)
               .all())
    
    if not records:
        print("No hay registros de PlayerState para exportar.")
        return

    export_data = []
    
    for i, record in enumerate(records):
        export_data.append(record.to_dict())
        if (i + 1 ) % 1000 == 0:
            print(f"Exportados {i + 1} registros de PlayerState...")


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
        file_type="application/json"
    )