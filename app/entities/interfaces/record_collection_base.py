# record_collection_base.py
from abc import ABC, abstractmethod
from sqlalchemy.orm import Session
from typing import Type, List, Optional
from app.entities.utils import AbstractSingleton


class RecordCollectionBase(metaclass=AbstractSingleton):
    orm_model: Type

    def __init__(self, db: Session):
        self.db = db

    @abstractmethod
    def generate_id(self, obj) -> int:
        """
        Genera el ID único usado en la base (depende del tipo de entidad).
        """
        pass

    def get_record_for_frame(self, track_id: int, frame_index: int):
        """
        Busca si existe un registro coincidente con track_id + frame_index.
        Las colecciones pueden sobrescribir este método si usan otros campos.
        """

        # Not all ORM models have both fields, but most do (players/ball).
        query = self.db.query(self.orm_model)

        # Solo agregar filtros si existen esos campos en el modelo.
        if hasattr(self.orm_model, "track_id"):
            query = query.filter(self.orm_model.track_id == track_id)

        if hasattr(self.orm_model, "frame_index"):
            query = query.filter(self.orm_model.frame_index == frame_index)

        return query.first()

    def get_by_id(self, obj_id: int):
        return self.db.query(self.orm_model).filter(self.orm_model.id == obj_id).first()

    def get(self, obj_id: int):
        return (
            self.db.query(self.orm_model)
            .filter(self.orm_model.id == obj_id)
            .first()
        )

    def get_all(self) -> List:
        return self.db.query(self.orm_model).all()

    def post(self, obj_data: dict):
        obj = self.orm_model(**obj_data)
        self.db.add(obj)
        self.db.commit()
        self.db.refresh(obj)
        return obj

    def patch(self, obj_id: int, updates: dict):
        obj = self.get(obj_id)
        if not obj:
            return None

        for key, val in updates.items():
            if hasattr(obj, key):
                setattr(obj, key, val)

        self.db.commit()
        self.db.refresh(obj)
        return obj

    def delete(self, obj_id: int) -> bool:
        obj = self.get(obj_id)
        if not obj:
            return False

        self.db.delete(obj)
        self.db.commit()
        return True
