import asyncio
import pathlib
from app.entities.models import PlayerStateModel
from sqlalchemy.orm import Session
from app.tasks.upload import upload_video_file
from app.utils.routes import OUTPUT_VIDEOS_DIR

async def upload_heatmaps_for_extracted_players(db: Session, match_id: int, extracted_player_ids: set):
    """
    Sube todos los heatmaps generados para los jugadores que se extrajeron.
    Filtra por player_id y sube solo los heatmaps que coincidan con los jugadores.
    """
    try:
        base_path = OUTPUT_VIDEOS_DIR
        home_path = base_path / "home_players"
        rival_path = base_path / "rival_players"

        upload_tasks = []

        for player_id in extracted_player_ids:
            # Archivos HOME
            home_file = home_path / f"heatmap_player_home_{player_id}.png"
            if home_file.exists():
                with open(home_file, "rb") as f:
                    file_bytes = f.read()
                upload_tasks.append(
                    upload_video_file(
                        match_id=match_id,
                        player_id=player_id,
                        filename=home_file.name,
                        file_bytes=file_bytes
                    )
                )

            # Archivos RIVAL
            rival_file = rival_path / f"heatmap_player_rival_{player_id}.png"
            if rival_file.exists():
                with open(rival_file, "rb") as f:
                    file_bytes = f.read()
                upload_tasks.append(
                    upload_video_file(
                        match_id=match_id,
                        player_id=player_id,
                        filename=rival_file.name,
                        file_bytes=file_bytes
                    )
                )

        # Ejecutar todas las subidas concurrentemente
        results = []
        if upload_tasks:
            results = await asyncio.gather(*upload_tasks, return_exceptions=True)

        # Guardar URLs en PlayerStateModel
        for result, player_id in zip(results, list(extracted_player_ids)*2):
            if isinstance(result, Exception):
                print(f"Error subiendo heatmap del jugador {player_id}: {result}")
                continue

            stats = db.query(PlayerStateModel).filter_by(
                match_id=match_id,
                player_id=player_id
            ).first()

            if stats:
                stats.heatmap_image_path = result
                db.commit()
                print(f"Heatmap jugador {player_id} subido correctamente: {result}")

        print("Subida de heatmaps completada.")
    except Exception as e:
        print(f"Error al subir heatmaps: {e}")
        raise e