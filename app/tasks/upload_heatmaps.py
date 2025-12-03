import asyncio
import pathlib
from app.entities.models import PlayerStateModel
from sqlalchemy.orm import Session
from app.tasks.upload import upload_file
from app.utils.routes import OUTPUT_VIDEOS_DIR

async def upload_heatmaps_for_extracted_players(db: Session, match_id: int, extracted_player_ids: set):
    """
    Sube todos los heatmaps generados para los jugadores que se extrajeron.
    Filtra por player_id y sube solo los heatmaps que coincidan con los jugadores.
    """
    try:
        base_path = OUTPUT_VIDEOS_DIR
        players_path = base_path / "players"

        upload_tasks = []
        upload_count = 0
        error_count = 0

        if not extracted_player_ids:
            print("No hay jugadores extraídos para subir heatmaps.")
            return

        for player_id in extracted_player_ids:
            try:
                home_file = players_path / f"heatmap_player_{player_id}.png"
                if not home_file.exists():
                    print(f"No se encontró el heatmap para el jugador {player_id} en home_players.")
                    error_count += 1
                    continue
                
                if home_file.stat().st_size == 0:
                    print(f"El heatmap para el jugador {player_id} en home_players está vacío.")
                    error_count += 1
                    continue

                with open(home_file, "rb") as f:
                    file_bytes = f.read()

                upload_tasks.append(
                    upload_file(
                        match_id=match_id,
                        player_id=player_id,
                        filename=home_file.name,
                        file_bytes=file_bytes
                    )
                )
                home_file.unlink(missing_ok=True)
                upload_count += 1
            except Exception as e:
                error_count += 1
                print(f"Error preparando la subida del heatmap del jugador {player_id}: {e}")

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