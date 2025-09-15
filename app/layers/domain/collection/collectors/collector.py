from dataclasses import field
from typing import Any, Dict, Iterator, List, Optional

import numpy as np
from layers.domain.collection.collectors.frame_collector import FrameCollector
from layers.domain.player.player import Player
from layers.domain.player.team import Team
from layers.domain.utils.deep_sort_tracker import DeepSortTracker


class Collector():
    """
    Clase principal que centraliza
    toda la lógica de comportamiento y almacenamiento.
    Elimina la redundancia de datos y optimiza el uso de memoria.
    """

    id: int = field(default_factory=int)
    frames: list[FrameCollector] = field(default_factory=list)
    deepsort_tracker: DeepSortTracker = field(default_factory=DeepSortTracker)
    max_frames: int = 500

    def __init__(self, id: int, max_frames: int = 500):
        self.id = id
        self.max_frames = max_frames
        self.frames = []
        self.deepsort_tracker = DeepSortTracker()

    def add_frame_data(self,
                       frame_number: int,
                       teams_data: Dict[int,
                                        Dict[str,
                                             Any]],
                       timestamp: float = 0.0) -> FrameCollector:
        """
        Añade datos de un frame con equipos y jugadores
        teams_data: {team_id:
        {'team_name': str,
        'players': [player_data_dict]}
        }
        """
        # Crear frame
        frame = FrameCollector(frame=frame_number, timestamp=timestamp)

        # Procesar cada equipo
        for team_id, team_data in teams_data.items():
            team_name = team_data.get('team_name', f'Team_{team_id}')
            team = Team(team=team_id, team_name=team_name)

            # Crear jugadores
            players = []
            for player_data in team_data.get('players', []):
                player = Player(
                    id=player_data.get('id', 0),
                    bbox=player_data['bbox'],
                    position_2d=tuple(player_data['position']),
                    position_adjusted_2d=tuple(player_data['position_adjusted']),
                    position_transformed_2d=player_data.get('position_transformed'),
                    speed=player_data.get('speed'),
                    distance=player_data.get('distance'),
                    team=str(team.team),
                    team_color=player_data['team_color'],
                    timestamp=timestamp
                )
                players.append(player)

            # Aplicar DeepSort tracking
            tracked_players = self.deepsort_tracker.update_tracks(players)

            # Añadir jugadores al equipo
            for player in tracked_players:
                team.add_player(player)

            # Añadir equipo al frame
            frame.add_team(team)

        # Añadir frame y mantener límite de memoria
        self.frames.append(frame)
        self._maintain_frame_limit()

        return frame

    def _maintain_frame_limit(self) -> None:
        """Mantiene el límite de frames para controlar el uso de memoria"""
        if len(self.frames) > self.max_frames:
            # Mantener solo los frames más recientes
            self.frames = sorted(self.frames,
                                 key=lambda f: f.frame)[-self.max_frames:]

    def get_frame_by_number(
            self,
            frame_number: int) -> Optional[FrameCollector]:
        """Obtiene un frame por su número"""
        for frame in self.frames:
            if frame.frame == frame_number:
                return frame
        return None

    def get_frames_sorted(
            self,
            descending: bool = False) -> List[FrameCollector]:
        """Obtiene frames ordenados por número"""
        return sorted(self.frames, key=lambda f: f.frame, reverse=descending)

    def get_frames_in_range(
            self,
            start_frame: Optional[int] = None,
            end_frame: Optional[int] = None) -> List[FrameCollector]:
        """Obtiene frames en un rango específico"""
        frames = self.frames

        if start_frame is not None:
            frames = [f for f in frames if f.frame >= start_frame]
        if end_frame is not None:
            frames = [f for f in frames if f.frame <= end_frame]

        return sorted(frames, key=lambda f: f.frame)

    def get_all_players(self) -> Iterator[Player]:
        """Generador que obtiene todos los jugadores sin duplicar en memoria"""
        for frame in self.frames:
            for team in frame.teams:
                for player in team.players:
                    yield player

    def get_all_players_list(self) -> List[Player]:
        """Obtiene todos los jugadores como lista (usar solo cuando sea necesario)"""
        return list(self.get_all_players())

    def get_team_by_id(
            self,
            team_id: int,
            frame_number: Optional[int] = None) -> Optional[Team]:
        """
        Obtiene un equipo por su ID. Si se especifica frame_number, busca en ese frame específico.
        Si no, busca en el frame más reciente.
        """
        if frame_number is not None:
            frame = self.get_frame_by_number(frame_number)
            return frame.get_team_by_id(team_id) if frame else None
        else:
            # Buscar en el frame más reciente
            if self.frames:
                latest_frame = max(self.frames, key=lambda f: f.frame)
                return latest_frame.get_team_by_id(team_id)
        return None

    def get_top_players_by_speed(self, limit: int = 10) -> List[Player]:
        """Obtiene los jugadores más rápidos de todos los frames"""
        valid_players = [
            p for p in self.get_all_players() if p.speed is not None]
        return sorted(valid_players,
                      key=lambda p: p.speed if p.speed is not None else 0.0,
                      reverse=True)[:limit]

    def get_top_players_by_distance(self, limit: int = 10) -> List[Player]:
        """Obtiene los jugadores que más distancia han recorrido"""
        valid_players = [
            p for p in self.get_all_players() if p.distance is not None]
        return sorted(
            valid_players,
            key=lambda p: p.distance if p.distance is not None else 0.0,
            reverse=True)[
            :limit]

    def get_team_comparison(self) -> Dict[str, Any]:
        """Compara estadísticas entre equipos del frame más reciente"""
        if not self.frames:
            return {}

        latest_frame = max(self.frames, key=lambda f: f.frame)
        team_stats = [team.get_team_stats() for team in latest_frame.teams]

        return {
            'collector_id': self.id,
            'frame_number': latest_frame.frame,
            'teams': team_stats,
            'total_teams': len(team_stats)
        }

    def get_frame_evolution(self,
                            start_frame: Optional[int] = None,
                            end_frame: Optional[int] = None
                            ) -> List[Dict[str, Any]]:
        """Obtiene la evolución de estadísticas a través de los frames"""
        frames = self.get_frames_in_range(start_frame, end_frame)
        return [frame.get_frame_stats() for frame in frames]

    def get_statistics(self) -> Dict[str, Any]:
        """Obtiene estadísticas generales del collector"""
        all_players = list(self.get_all_players())
        valid_speeds = [p.speed for p in all_players if p.speed is not None]
        valid_distances = [
            p.distance for p in all_players if p.distance is not None]

        return {
            'collector_id': self.id,
            'total_frames': len(self.frames),
            'total_players': len(all_players),
            'avg_speed': np.mean(valid_speeds) if valid_speeds else 0,
            'max_speed': max(valid_speeds) if valid_speeds else 0,
            'avg_distance': np.mean(valid_distances) if valid_distances else 0,
            'max_distance': max(valid_distances) if valid_distances else 0,
            'teams_count': len(set(p.team for p in all_players))
        }

    def reset_tracking(self) -> None:
        """Resetea el sistema de tracking DeepSort"""
        self.deepsort_tracker = DeepSortTracker()

    def set_frame_limit(self, limit: int) -> None:
        """Establece un nuevo límite de frames y
        aplica la limpieza si es necesario"""
        self.max_frames = limit
        self._maintain_frame_limit()

    def export_data(self) -> Dict[str, Any]:
        """Exporta todos los datos del collector"""
        export_data = {
            'collector_id': self.id,
            'total_frames': len(self.frames),
            'frames': []
        }

        for frame in self.get_frames_sorted():
            frame_data = {
                'frame_number': frame.frame,
                'timestamp': frame.timestamp,
                'teams': []
            }

            for team in frame.teams:
                team_data = {
                    'team_id': team.team,
                    'team_name': team.team_name,
                    'players': [player.to_dict() for player in team.players]
                }
                frame_data['teams'].append(team_data)

            export_data['frames'].append(frame_data)

        return export_data

    def clear_old_frames(self, keep_frames: int) -> None:
        """Limpia frames antiguos manteniendo solo los últimos N"""
        if len(self.frames) > keep_frames:
            self.frames = self.get_frames_sorted(descending=True)[:keep_frames]
