from app.entities.interfaces.record_collection_base import RecordCollectionBase
from app.entities.models import PlayerState, HeatmapPointModel, BallEventModel


class TrackCollectionPlayer(RecordCollectionBase):
    orm_model = PlayerState

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
