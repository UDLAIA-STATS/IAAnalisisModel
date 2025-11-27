from pathlib import Path
from typing import Dict, Set
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from mplsoccer import Pitch

from app.layers.domain.tracks.track_detail import TrackDetailBase
from app.layers.infraestructure.video_analysis.plotting.interfaces import Diagram
from app.layers.infraestructure.video_analysis.plotting.services import DrawerService


class HeatmapDrawer(Diagram):
    def __init__(self, tracks: Dict[int, Dict[int, TrackDetailBase]]):
        super().__init__(tracks)

        base = Path("./app/res/output_videos/")
        self.save_path = base
        self.rival_players_path = base / "rival_players"
        self.home_players_path = base / "home_players"

        for p in [base, self.rival_players_path, self.home_players_path]:
            p.mkdir(parents=True, exist_ok=True)

        self.drawer_service = DrawerService()

    # ---------------------------------------------------------
    # HELPERS
    # ---------------------------------------------------------
    def _safe_concat(self, dfs: list) -> pd.DataFrame:
        """Concatena listas de DataFrames con limpieza total."""
        if not dfs:
            return pd.DataFrame(columns=["x", "y"])

        df = pd.concat(dfs, ignore_index=True)

        df["x"] = pd.to_numeric(df["x"], errors="coerce")
        df["y"] = pd.to_numeric(df["y"], errors="coerce")

        df.replace([np.inf, -np.inf], np.nan, inplace=True)
        df.dropna(subset=["x", "y"], inplace=True)

        return df

    def _is_valid_for_kde(self, df: pd.DataFrame) -> bool:
        """Verifica si los datos son usables por KDE para evitar errores."""
        if df.empty or df.shape[0] < 5:
            return False

        # Debe existir variación
        if df['x'].min() == df['x'].max():
            return False
        if df['y'].min() == df['y'].max():
            return False

        return True

    def _draw_pitch(self):
        pitch = Pitch(
            pitch_type="statsbomb",
            pitch_color="#1e4251",
            line_color="white",
            axis=True,
            label=True,
            tick=True,
        )
        fig, ax = plt.subplots(figsize=(13, 8.5))
        fig.set_facecolor("white")
        ax.set_facecolor("white")
        pitch.draw(ax=ax)
        return fig, ax, pitch

    
    def draw_and_save(self) -> None:
        self._draw_individual_heatmaps()


    def _draw_individual_heatmaps(self) -> None:
        print("Dibujando heatmaps individuales por jugador...")
        player_ids: Set[int] = {
            track.track_id
            for frame in self.tracks.values()
            for track in frame.values()
            if track.track_id is not None
        }
        
        print(f"Se encontraron {len(player_ids)} jugadores únicos para heatmaps.")
        for pid in player_ids:
            home_frames = []
            rival_frames = []

            for frame_content in self.tracks.values():
                if pid not in frame_content:
                    continue

                filtered = {pid: frame_content[pid]}
                home, rival = self.drawer_service.process_frame(filtered)

                if not home.empty:
                    home_frames.append(home)
                if not rival.empty:
                    rival_frames.append(rival)

            home_df = self._safe_concat(home_frames)
            rival_df = self._safe_concat(rival_frames)

            if home_df.empty and rival_df.empty:
                continue

            fig, ax, pitch = self._draw_pitch()

            # HOME PLAYER
            
            if self._is_valid_for_kde(home_df):
                print(f"Dibujando heatmap para jugador local {pid}...")
                levels = min(60, max(10, home_df.shape[0] // 2))
                pitch.kdeplot(
                    home_df["x"], home_df["y"],
                    ax=ax,
                    cmap="viridis",
                    fill=True,
                    alpha=0.6,
                    levels=levels,
                    bw_adjust=0.3
                )
                fig.savefig(self.home_players_path / f"heatmap_player_home_{pid}.png",
                            dpi=300, bbox_inches="tight")

            # RIVAL PLAYER
            if self._is_valid_for_kde(rival_df):
                print(f"Dibujando heatmap para jugador rival {pid}...")
                levels = min(60, max(10, rival_df.shape[0] // 2))
                pitch.kdeplot(
                    rival_df["x"], rival_df["y"],
                    ax=ax,
                    cmap="viridis",
                    fill=True,
                    alpha=0.6,
                    levels=levels,
                    bw_adjust=0.3
                )
                fig.savefig(self.rival_players_path / f"heatmap_player_rival_{pid}.png",
                            dpi=300, bbox_inches="tight")

            plt.close(fig)
