from sqlalchemy import Boolean, Column, Integer, Float, String
from app.modules.services.database import Base

class PlayerState(Base):
    __tablename__ = "player_state"

    id = Column(Integer, primary_key=True, index=True)
    
    # Identidad del jugador
    player_id = Column(Integer, index=True)
    team = Column(String)
    color = Column(String)

    # Frame
    frame_index = Column(Integer, index=True)

    # Posición jugador
    x = Column(Float)
    y = Column(Float)
    z = Column(Float)

    # Balón
    ball_x = Column(Float, nullable=True)
    ball_y = Column(Float, nullable=True)
    ball_z = Column(Float, nullable=True)

    has_ball = Column(Boolean, default=False)
    ball_possession_time = Column(Float, default=0.0)

    # NEW → explícito para detectar pases
    ball_owner_id = Column(Integer, index=True, nullable=True)

    # Dinámica
    distance = Column(Float, default=0.0)
    speed = Column(Float, default=0.0)

    time_visible = Column(Float, default=0.0)

    timestamp_ms = Column(Float, index=True)
