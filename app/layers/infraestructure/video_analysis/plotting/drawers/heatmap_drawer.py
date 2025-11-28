from pathlib import Path
from typing import Dict, Set
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
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

        for p in (self.save_path, self.rival_players_path, self.home_players_path):
            p.mkdir(parents=True, exist_ok=True)

        self.drawer = DrawerService()

    @staticmethod
    def _clean_df(df: pd.DataFrame) -> pd.DataFrame:
        """Limpieza segura y rápida para cualquier DF de posiciones."""
        if df.empty:
            return pd.DataFrame(columns=["x", "y"])

        df = df.copy()

        for col in ("x", "y"):
            df[col] = pd.to_numeric(df[col], errors="coerce")

        df.replace([np.inf, -np.inf], np.nan, inplace=True)
        df.dropna(subset=["x", "y"], inplace=True)

        return df

    @staticmethod
    def _valid_kde(df: pd.DataFrame) -> bool:
        """Validación estricta pero segura para KDE."""
        if df.empty or len(df) < 5:
            return False

        return not (df["x"].min() == df["x"].max() or df["y"].min() == df["y"].max())

    @staticmethod
    def _draw_pitch():
        pitch = Pitch(
            pitch_type="statsbomb",
            pitch_color="#1e4251",
            line_color="white",
            axis=False,
            label=False,
            tick=False,
        )
        fig, ax = plt.subplots(figsize=(13, 8.5))
        pitch.draw(ax=ax)
        return fig, ax, pitch

    def _plot_kde(self, pitch, ax, df: pd.DataFrame, cmap: str):
        """Plot KDE con validación automática."""
        if not self._valid_kde(df):
            return False

        levels = min(60, max(10, len(df) // 2))

        pitch.kdeplot(
            df["x"],
            df["y"],
            ax=ax,
            cmap=cmap,
            fill=True,
            alpha=0.60,
            levels=levels,
            bw_adjust=0.30,
        )
        return True

    def _save(self, fig, path: Path):
        """Guardar figura con protección."""
        try:
            fig.savefig(path, dpi=300, bbox_inches="tight")
        except Exception as e:
            print(f"[ERROR] No se pudo guardar {path}: {e}")

    def draw_and_save(self) -> None:
        self._draw_individual()

    def _draw_individual(self):
        player_ids: Set[int] = {
            t.track_id
            for frame in self.tracks.values()
            for t in frame.values()
            if t.track_id is not None
        }

        for pid in player_ids:
            home_frames, rival_frames = [], []

            for frame in self.tracks.values():
                if pid not in frame:
                    continue

                filtered = {pid: frame[pid]}
                home, rival = self.drawer.process_frame(filtered)
                if not home.empty:
                    home_frames.append(home)
                if not rival.empty:
                    rival_frames.append(rival)

            home_df = self._clean_df(pd.concat(home_frames, ignore_index=True) if home_frames else pd.DataFrame())
            rival_df = self._clean_df(pd.concat(rival_frames, ignore_index=True) if rival_frames else pd.DataFrame())

            if home_df.empty and rival_df.empty:
                continue

            fig, ax, pitch = self._draw_pitch()

            # HOME PLAYER
            if self._plot_kde(pitch, ax, home_df, cmap="viridis"):
                self._save(fig, self.home_players_path / f"heatmap_player_home_{pid}.png")

            # RIVAL PLAYER
            if self._plot_kde(pitch, ax, rival_df, cmap="viridis"):
                self._save(fig, self.rival_players_path / f"heatmap_player_rival_{pid}.png")

            plt.close(fig)
