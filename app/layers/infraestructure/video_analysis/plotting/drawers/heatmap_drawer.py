from pathlib import Path
from typing import Dict

import matplotlib.pyplot as plt
import pandas as pd
from app.layers.domain.tracks.track_detail import TrackDetailBase
from app.layers.infraestructure.video_analysis.plotting.interfaces import Diagram
from app.layers.infraestructure.video_analysis.plotting.services import \
    DrawerService
from mplsoccer import Pitch


class HeatmapDrawer(Diagram):
    def __init__(self, tracks: Dict[int, Dict[int, TrackDetailBase]], metrics: Dict | None = None):
        super().__init__(tracks)
        self.save_path = './app/res/output_videos/'
        path = Path(self.save_path)
        self.rival_players_path = path / 'rival_players'
        self.home_players_path = path / 'home_players'
        path.mkdir(parents=True, exist_ok=True)
        self.rival_players_path.mkdir(parents=True, exist_ok=True)
        self.home_players_path.mkdir(parents=True, exist_ok=True)

        self.drawer_service = DrawerService()

    def draw_and_save(self) -> None:
        self._draw_heatmap()
        self._draw_individual_heatmaps()

    def _draw_heatmap(self) -> None:
        home_df, rival_df = pd.DataFrame(), pd.DataFrame()

        for frame, track_content in self.tracks.items():
            home_players, rival_players = self.drawer_service.process_frame(
                track_content)
            home_df = pd.concat([home_df, home_players], ignore_index=True)
            rival_df = pd.concat([rival_df, rival_players], ignore_index=True)

        if home_df.empty and rival_df.empty:
            return

        fig, ax = plt.subplots(figsize=(13, 8.5))
        fig.set_facecolor('white')
        ax.patch.set_facecolor('white')

        pitch = Pitch(
            pitch_type='statsbomb',
            pitch_color='#1e4251',
            line_color='white',
            axis=True,
            label=True,
            tick=True
        )
        pitch.draw(ax=ax)

        # Dibujar heatmap de cada equipo
        if not home_df.empty:
            pitch.kdeplot(
                home_df.x,
                home_df.y,
                ax=ax,
                cmap='Blues',
                fill=True,
                alpha=0.6,
                levels=100,
                bw_adjust=0.3)
            plt.savefig(self.save_path + 'heatmap_home.png', dpi=300, bbox_inches='tight')
            
        if not rival_df.empty:
            pitch.kdeplot(
                rival_df.x,
                rival_df.y,
                ax=ax,
                cmap='Reds',
                fill=True,
                alpha=0.6,
                levels=100,
                bw_adjust=0.3)
            plt.savefig(self.save_path + 'heatmap_rival.png', dpi=300, bbox_inches='tight')

        plt.savefig(self.save_path + 'heatmap_both_teams.png', dpi=300, bbox_inches='tight')
        plt.close()

    def _draw_individual_heatmaps(self) -> None:
        
        
        for player_id in set(
            track.track_id for frame in self.tracks.values()
            for track in frame.values()
        ):
            home_players = pd.DataFrame()
            rival_players = pd.DataFrame()

            for frame, track_content in self.tracks.items():
                if player_id in track_content:
                    home_players, rival_players = self.drawer_service.process_frame(
                        {player_id: track_content[player_id]})

            if home_players.empty and rival_players.empty:
                continue

            fig, ax = plt.subplots(figsize=(13, 8.5))
            fig.set_facecolor('white')
            ax.patch.set_facecolor('white')

            pitch = Pitch(
                pitch_type='statsbomb',
                pitch_color='#1e4251',
                line_color='white',
                axis=True,
                label=True,
                tick=True
            )
            pitch.draw(ax=ax)

            pitch.kdeplot(
                home_players.x,
                home_players.y,
                ax=ax,
                cmap='viridis',
                fill=True,
                alpha=0.6,
                levels=100,
                bw_adjust=0.3)
            plt.savefig(self.home_players_path / f'heatmap_player_home_{player_id}.png', dpi=300, bbox_inches='tight')
            pitch.kdeplot(
                rival_players.x,
                rival_players.y,
                ax=ax,
                cmap='viridis',
                fill=True,
                alpha=0.6,
                levels=100,
                bw_adjust=0.3)
            plt.savefig(self.rival_players_path / f'heatmap_player_rival_{player_id}.png', dpi=300, bbox_inches='tight')

            plt.close()