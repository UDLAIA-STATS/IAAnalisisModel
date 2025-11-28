from pathlib import Path
from typing import Dict, Set
from matplotlib import pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import gaussian_kde
from mplsoccer import Pitch

from app.layers.domain.tracks.track_detail import TrackDetailBase
from app.layers.infraestructure.video_analysis.plotting.interfaces import Diagram
from app.layers.infraestructure.video_analysis.plotting.services import DrawerService


class HeatmapDrawer(Diagram):

    # -------------------------------------------------------------------------
    # INIT
    # -------------------------------------------------------------------------
    def __init__(self, tracks: Dict[int, Dict[int, TrackDetailBase]]):
        super().__init__(tracks)

        base = Path("./app/res/output_videos/")
        self.save_path = base
        self.rival_players_path = base / "rival_players"
        self.home_players_path = base / "home_players"

        for p in (self.save_path, self.rival_players_path, self.home_players_path):
            p.mkdir(parents=True, exist_ok=True)

        self.drawer = DrawerService()

        # Pitch preinstanciado (mejor rendimiento)
        self.pitch = Pitch(
            pitch_type="statsbomb",
            pitch_color="#1e4251",
            line_color="white",
            axis=False,
            label=False,
            tick=False
        )

        # Dimensions exactas del campo
        self.xmin, self.xmax, self.ymin, self.ymax = self._resolve_pitch_dims()
        print(f"DEBUG pitch dims -> xmin:{self.xmin} xmax:{self.xmax} ymin:{self.ymin} ymax:{self.ymax}")

    def _resolve_pitch_dims(self):
        """
        Intenta resolver las dimensiones del pitch de forma robusta:
        1) Busca atributos comunes en self.pitch (dim.x / dim.y, pitch_length / pitch_width, etc.)
        2) Si no encuentra nada, dibuja temporalmente el pitch y lee ax.get_xlim() / get_ylim()
        Devuelve: xmin, xmax, ymin, ymax
        """
        # 1) Intentar atributos comunes
        # buscar pitch.dim.x / pitch.dim.y
        dim = getattr(self.pitch, "dim", None)
        if dim is not None:
            x = getattr(dim, "x", None) or getattr(dim, "length", None) or getattr(dim, "pitch_length", None)
            y = getattr(dim, "y", None) or getattr(dim, "width", None) or getattr(dim, "pitch_width", None)
            if x is not None and y is not None:
                return 0, float(x), 0, float(y)

            # si dim es tupla/lista
            if isinstance(dim, (tuple, list)) and len(dim) >= 2:
                return 0, float(dim[0]), 0, float(dim[1])

        # 2) Intentar atributos directos en pitch
        for name_x, name_y in (
            ("pitch_length", "pitch_width"),
            ("length", "width"),
            ("pitch_length_m", "pitch_width_m"),
        ):
            x = getattr(self.pitch, name_x, None)
            y = getattr(self.pitch, name_y, None)
            if x is not None and y is not None:
                return 0, float(x), 0, float(y)

        # 3) Dibujar temporalmente y leer límites de los ejes (método más compatible)
        try:
            fig_tmp, ax_tmp = plt.subplots(figsize=(6, 4))
            # pitch.draw puede aceptar ax; si falla, se captura
            try:
                self.pitch.draw(ax=ax_tmp)
            except TypeError:
                # Algunas versiones podrían requerir llamada diferente; intentamos draw() sin ax
                try:
                    self.pitch.draw()
                    ax_tmp = plt.gca()
                except Exception:
                    pass

            xlim = ax_tmp.get_xlim()
            ylim = ax_tmp.get_ylim()
            plt.close(fig_tmp)
            # xlim viene como (xmin, xmax)
            return float(xlim[0]), float(xlim[1]), float(ylim[0]), float(ylim[1])
        except Exception:
            # 4) Fallback final: asumir StatsBomb-like (0..120 x, 0..80 y)
            return 0.0, 120.0, 0.0, 80.0


    # -------------------------------------------------------------------------
    # HELPERS
    # -------------------------------------------------------------------------
    @staticmethod
    def _clean_dataframe(dfs: list) -> pd.DataFrame:
        """Concatena y limpia listas de frames del jugador."""
        if not dfs:
            return pd.DataFrame(columns=["x", "y"])

        df = pd.concat(dfs, ignore_index=True)
        df["x"] = pd.to_numeric(df["x"], errors="coerce")
        df["y"] = pd.to_numeric(df["y"], errors="coerce")

        df.replace([np.inf, -np.inf], np.nan, inplace=True)
        df.dropna(subset=["x", "y"], inplace=True)

        return df

    @staticmethod
    def _is_valid_for_kde(df: pd.DataFrame) -> bool:
        """Debe haber datos suficientes y variación real."""
        if df.empty or df.shape[0] < 5:
            return False
        return not (
            df["x"].min() == df["x"].max() or
            df["y"].min() == df["y"].max()
        )

    # -------------------------------------------------------------------------
    # DIBUJAR CAMPO
    # -------------------------------------------------------------------------
    def _draw_pitch(self):
        fig, ax = plt.subplots(figsize=(13, 8.5))
        fig.set_facecolor("white")
        ax.set_facecolor("white")
        self.pitch.draw(ax=ax)
        return fig, ax

    # -------------------------------------------------------------------------
    # KDE SEGURO — sin contornos
    # -------------------------------------------------------------------------
    def _safe_kde_heatmap(self, df: pd.DataFrame, ax):
        """
        Renderiza un heatmap estable usando gaussian_kde + imshow.
        Nunca produce niveles inválidos.
        """
        try:
            # Grid uniforme
            grid_x, grid_y = np.mgrid[
                self.xmin:self.xmax:200j,
                self.ymin:self.ymax:200j
            ]

            points = np.vstack([df["x"], df["y"]])

            kde = gaussian_kde(points, bw_method=0.3)

            density = kde(np.vstack([grid_x.ravel(), grid_y.ravel()]))
            density = density.reshape(grid_x.shape)

            # Normalizar para buen contraste
            density /= density.max()

            # imshow evita contornos → cero errores
            ax.imshow(
                np.rot90(density),
                extent=[self.xmin, self.xmax, self.ymin, self.ymax],
                cmap="magma",      # color más profesional
                alpha=0.55,
                interpolation="bilinear"
            )
            return True

        except Exception as e:
            print(f"⚠ Error en KDE seguro: {e}")
            return False

    # -------------------------------------------------------------------------
    # PUBLIC
    # -------------------------------------------------------------------------
    def draw_and_save(self) -> None:
        self._draw_individual_heatmaps()

    # -------------------------------------------------------------------------
    # HEATMAPS POR JUGADOR
    # -------------------------------------------------------------------------
    def _draw_individual_heatmaps(self) -> None:
        print("Dibujando heatmaps individuales por jugador...\n")

        # Lista de jugadores con track_id
        player_ids: Set[int] = {
            t.track_id
            for frame in self.tracks.values()
            for t in frame.values()
            if t.track_id is not None
        }

        print(f"Encontrados {len(player_ids)} jugadores únicos.\n")

        for pid in sorted(player_ids):

            home_frames = []
            rival_frames = []

            # Recolectar posiciones de todos los frames
            for frame in self.tracks.values():
                if pid not in frame:
                    continue

                filtered = {pid: frame[pid]}
                home_df, rival_df = self.drawer.process_frame(filtered)

                if not home_df.empty:
                    home_frames.append(home_df)
                if not rival_df.empty:
                    rival_frames.append(rival_df)

            home_df = self._clean_dataframe(home_frames)
            rival_df = self._clean_dataframe(rival_frames)

            if home_df.empty and rival_df.empty:
                continue

            # Crear figura por jugador
            fig, ax = self._draw_pitch()

            # -----------------------------------------------------------------
            # HOME PLAYER
            # -----------------------------------------------------------------
            if self._is_valid_for_kde(home_df):
                print(f" → Dibujando heatmap LOCAL: {pid}")

                if self._safe_kde_heatmap(home_df, ax):
                    fig.savefig(
                        self.home_players_path / f"heatmap_player_home_{pid}.png",
                        dpi=300, bbox_inches="tight"
                    )

            # -----------------------------------------------------------------
            # RIVAL PLAYER
            # -----------------------------------------------------------------
            if self._is_valid_for_kde(rival_df):
                print(f" → Dibujando heatmap RIVAL: {pid}")

                if self._safe_kde_heatmap(rival_df, ax):
                    fig.savefig(
                        self.rival_players_path / f"heatmap_player_rival_{pid}.png",
                        dpi=300, bbox_inches="tight"
                    )

            plt.close(fig)
