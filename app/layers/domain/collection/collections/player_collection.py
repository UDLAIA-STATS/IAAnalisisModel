from typing import Any, Dict, List, Optional

import numpy as np
from layers.domain.collection.collectors.collector import Collector
from layers.domain.utils.singleton import Singleton


class PlayerCollection(metaclass=Singleton):
    """
    Clase ligera para gestionar múltiples Collectors sin duplicar datos.
    Actúa como un registry simple de collectors.
    """
    
    def __init__(self):
        self.collectors: Dict[int, Collector] = {}
    
    def create_collector(self, collector_id: int, max_frames: int = 100) -> Collector:
        """Crea un nuevo collector"""
        if collector_id in self.collectors:
            raise ValueError(f"Collector with id {collector_id} already exists")
        
        collector = Collector(id = collector_id, max_frames = max_frames)
        self.collectors[collector_id] = collector
        return collector
    
    def get_collector(self, collector_id: int) -> Optional[Collector]:
        """Obtiene un collector por ID"""
        return self.collectors.get(collector_id)
    
    def remove_collector(self, collector_id: int) -> bool:
        """Elimina un collector"""
        if collector_id in self.collectors:
            del self.collectors[collector_id]
            return True
        return False
    
    def list_collectors(self) -> List[int]:
        """Lista todos los IDs de collectors"""
        return list(self.collectors.keys())
    
    def get_global_statistics(self) -> Dict[str, Any]:
        """Obtiene estadísticas agregadas de todos los collectors"""
        if not self.collectors:
            return {}
        
        total_frames = sum(len(c.frames) for c in self.collectors.values())
        all_players = []
        
        # Recolectar jugadores de todos los collectors usando generadores
        for collector in self.collectors.values():
            all_players.extend(collector.get_all_players())
        
        valid_speeds = [p.speed for p in all_players if p.speed is not None]
        valid_distances = [p.distance for p in all_players if p.distance is not None]
        
        return {
            'total_collectors': len(self.collectors),
            'total_frames': total_frames,
            'total_players': len(all_players),
            'avg_speed': np.mean(valid_speeds) if valid_speeds else 0,
            'max_speed': max(valid_speeds) if valid_speeds else 0,
            'avg_distance': np.mean(valid_distances) if valid_distances else 0,
            'max_distance': max(valid_distances) if valid_distances else 0,
            'teams_count': len(set(p.team_id for p in all_players))
        }
    
    def cleanup_all_collectors(self, keep_frames: int = 50) -> None:
        """Limpia datos antiguos en todos los collectors"""
        for collector in self.collectors.values():
            collector.clear_old_frames(keep_frames)
