from pathlib import Path
from typing import Dict, Set, List
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from mplsoccer import Pitch
from sqlalchemy import Row
from sqlalchemy.orm import Session

from app.modules.plotting.interfaces import Diagram
from app.entities.models import PlayerStateModel, HeatmapPointModel
from app.utils.routes import OUTPUT_VIDEOS_DIR


class HeatmapDrawer(Diagram):
    """
    Genera heatmaps por jugador utilizando PlayerStateModel y HeatmapPointModel.
    Usa coordenadas (x, z). No depende de TrackDetailBase.
    """

    def __init__(self, db: Session):
        # Llamada correcta al constructor de la clase base (sin args).
        super().__init__(db)

        base = OUTPUT_VIDEOS_DIR
        self.save_path = base
        self.rival_players_path = base / "rival_players"
        self.home_players_path = base / "home_players"

        for p in [base, self.rival_players_path, self.home_players_path]:
            p.mkdir(parents=True, exist_ok=True)

    # ---------------------------------------------------------
    # HELPERS
    # ---------------------------------------------------------
    def _safe_concat(self, dfs: List[pd.DataFrame]) -> pd.DataFrame:
        if not dfs:
            return pd.DataFrame(columns=["x", "z"])

        df = pd.concat(dfs, ignore_index=True)
        df["x"] = pd.to_numeric(df["x"], errors="coerce")
        df["z"] = pd.to_numeric(df["z"], errors="coerce")
        df.replace([np.inf, -np.inf], np.nan, inplace=True)
        df.dropna(subset=["x", "z"], inplace=True)
        return df

    def _is_valid_for_kde(self, df: pd.DataFrame) -> bool:
        if df.empty or df.shape[0] < 5:
            return False
        if df["x"].min() == df["x"].max():
            return False
        if df["z"].min() == df["z"].max():
            return False
        return True

    def _draw_pitch(self):
        pitch = Pitch(
            pitch_type="statsbomb",
            pitch_color="#1e4251",
            line_color="white",
            axis=True,
            label=True,
        )
        fig, ax = plt.subplots(figsize=(13, 8.5))
        fig.set_facecolor("white")
        ax.set_facecolor("white")
        pitch.draw(ax=ax)
        return fig, ax, pitch

    # ---------------------------------------------------------
    # BD EXTRACTION
    # ---------------------------------------------------------
    def _fetch_players(self) -> Set[int]:
        rows = self.db.query(PlayerStateModel.player_id).distinct().all()

        player_ids: Set[int] = set()

        for r in rows:
            value = None

            # Caso 1: SQLAlchemy Row (2.x)
            if isinstance(r, Row):
                # Acceso seguro usando atributo
                try:
                    value = r.player_id
                except:
                    # fallback: acceso posicional
                    try:
                        value = r[0]
                    except:
                        pass

            # Caso 2: tuple/list
            elif isinstance(r, (tuple, list)):
                if len(r) > 0:
                    value = r[0]

            # Caso 3: valor directo
            else:
                value = r

            # Asegurar que value es int
            if isinstance(value, int):
                player_ids.add(value)
            else:
                # Intentar conversión segura sin estresar Pylance
                try:
                    if value is None:
                        continue
                    player_ids.add(int(value))
                except:
                    pass

        return player_ids

    def _fetch_player_states(self, player_id: int) -> List[PlayerStateModel]:
        return (
            self.db.query(PlayerStateModel)
            .filter(PlayerStateModel.player_id == player_id)
            .order_by(PlayerStateModel.frame_index)
            .all()
        )

    def _save_heatmap_points(self, states: List[PlayerStateModel]):
        """
        Inserta puntos en HeatmapPointModel. No realiza deduplicación avanzada.
        """
        for st in states:
            # Sólo añadir si las coordenadas están presentes
            if st.x is None or st.z is None:
                continue
            hp = HeatmapPointModel(
                player_id=st.player_id,
                frame_number=st.frame_index,
                x=st.x,
                z=st.z,
            )
            self.db.add(hp)
        self.db.commit()

    # ---------------------------------------------------------
    # MAIN
    # ---------------------------------------------------------
    def draw_and_save(self) -> None:
        print("Generando heatmaps desde BD...")
        player_ids = self._fetch_players()
        print(f"{len(player_ids)} jugadores encontrados.")

        for pid in player_ids:
            states = self._fetch_player_states(pid)
            if not states:
                continue

            # Guardar puntos (opcional, puedes comentar si ya tienes puntos)
            try:
                self._save_heatmap_points(states)
            except Exception as e:
                print(f"Advertencia guardando HeatmapPointModel para player {pid}: {e}")

            # Convertir a DataFrame y separar por equipo
            rows = [{"x": st.x, "z": st.z, "team": (st.team or "").lower()} for st in states]
            df = pd.DataFrame(rows)
            if df.empty:
                continue

            home_df = df[df["team"] == "home"][["x", "z"]]
            rival_df = df[df["team"] == "rival"][["x", "z"]]

            self._draw_player_heatmaps(pid, home_df, rival_df)

    # ---------------------------------------------------------
    # DRAW
    # ---------------------------------------------------------
    def _draw_player_heatmaps(self, pid: int, home_df: pd.DataFrame, rival_df: pd.DataFrame):
        fig, ax, pitch = self._draw_pitch()

        # HOME
        if self._is_valid_for_kde(home_df):
            print(f"Dibujando heatmap jugador HOME {pid}")
            levels = min(60, max(10, home_df.shape[0] // 2))
            pitch.kdeplot(
                home_df["x"], home_df["z"],
                ax=ax,
                cmap="viridis",
                fill=True,
                alpha=0.6,
                levels=levels,
                bw_adjust=0.3,
            )
            fig.savefig(self.home_players_path / f"heatmap_player_home_{pid}.png",
                        dpi=300, bbox_inches="tight")

        # RIVAL
        if self._is_valid_for_kde(rival_df):
            print(f"Dibujando heatmap jugador RIVAL {pid}")
            levels = min(60, max(10, rival_df.shape[0] // 2))
            pitch.kdeplot(
                rival_df["x"], rival_df["z"],
                ax=ax,
                cmap="viridis",
                fill=True,
                alpha=0.6,
                levels=levels,
                bw_adjust=0.3,
            )
            fig.savefig(self.rival_players_path / f"heatmap_player_rival_{pid}.png",
                        dpi=300, bbox_inches="tight")

        plt.close(fig)
