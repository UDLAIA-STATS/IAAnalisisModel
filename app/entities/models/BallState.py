from sqlalchemy import Column, DateTime, Integer, Float, func
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
    owner_id = Column(Integer)