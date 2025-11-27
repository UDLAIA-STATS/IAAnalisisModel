from abc import ABC, abstractmethod
from sqlalchemy.orm import Session
from typing import Type, List, Optional

class RecordCollectionBase(ABC):
    orm_model: Type

    def __init__(self, db: Session):
        self.db = db

    @abstractmethod
    def generate_id(self, obj) -> int:
        """Extraer o generar el ID de la entidad."""
        pass

    def get(self, obj_id: int):
        """
        Return the object with the given id if it exists, otherwise None

        Args:
            obj_id (int): The id of the object to retrieve

        Returns:
            Optional[self.orm_model]: The retrieved object, or None if not found
        """
        return self.db.query(self.orm_model).filter(self.orm_model.id == obj_id).first()

    def get_all(self) -> List:
        
        """
        Retrieve all objects from the database.

        Returns:
            List[self.orm_model]: A list of all objects in the database
        """
        return self.db.query(self.orm_model).all()

    def post(self, obj_data: dict):
        """
        Create a new object in the database based on the given data.

        Args:
            obj_data (dict): A dictionary containing the data to create the object

        Returns:
            self.orm_model: The newly created object
        """
        obj = self.orm_model(**obj_data)
        self.db.add(obj)
        self.db.commit()
        self.db.refresh(obj)
        return obj

    def patch(self, obj_id: int, updates: dict):
        """
        Update an object in the database with the given id, by applying the given updates.

        Args:
            obj_id (int): The id of the object to update
            updates (dict): A dictionary containing the key-value pairs to update

        Returns:
            Optional[self.orm_model]: The updated object, or None if not found
        """
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
        """
        Delete an object from the database with the given id.

        Args:
            obj_id (int): The id of the object to delete

        Returns:
            bool: True if the object was deleted, False if not found
        """
        obj = self.get(obj_id)
        if not obj:
            return False

        self.db.delete(obj)
        self.db.commit()
        return True
