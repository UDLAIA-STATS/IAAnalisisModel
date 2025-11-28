from app.entities.interfaces.record_collection_base import RecordCollectionBase
from app.entities.models import PlayerStateModel, HeatmapPointModel, BallEventModel

class TrackCollectionPlayer(RecordCollectionBase):
    orm_model = PlayerStateModel

    def generate_id(self, obj):
        return obj.track_id

class TrackCollectionBall(RecordCollectionBase):
    orm_model = BallEventModel

    def generate_id(self, obj):
        return obj.frame_index

class TrackCollectionHeatmapPoint(RecordCollectionBase):
    orm_model = HeatmapPointModel

    def generate_id(self, obj):
        return obj.point_id
    
    def get_record_for_frame(self, track_id: int, frame_index: int):
        return (
            self.db.query(self.orm_model)
            .filter(
                self.orm_model.player_id == track_id,
                self.orm_model.frame_index == frame_index
            )
            .first()
        )
