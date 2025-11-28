import json
from sqlalchemy import Column, DateTime, Integer, Float, String, func
from app.modules.services.database import Base

class BallEventModel(Base):
    """
    Eventos o detecciones del balón almacenadas (opcional).
    """
    __tablename__ = "ball_event"
    id = Column(Integer, primary_key=True, index=True)
    frame_index = Column(Integer, index=True)
    x = Column(Float)
    y = Column(Float)
    z = Column(Float)
    bbox = Column(String, nullable=True)
    owner_id = Column(Integer)
    
    def set_bbox(self, bbox_list: list[int]):
        """Convierte lista Python → string JSON seguro"""
        self.bbox = json.dumps(bbox_list)

    def get_bbox(self):
        """Convierte JSON almacenado → lista Python"""
        bbox = self.to_dict()
        if bbox["bbox"] is None:
            return None

        try:
            return json.loads(bbox["bbox"])
        except Exception:
            return None
    
    def to_dict(self):
        return {
            "id": self.id,
            "frame_index": self.frame_index,
            "x": self.x,
            "y": self.y,
            "z": self.z,
            "owner_id": self.owner_id,
        }