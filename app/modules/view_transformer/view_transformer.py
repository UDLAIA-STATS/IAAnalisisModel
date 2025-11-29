import cv2
import numpy as np

from app.entities.interfaces.record_collection_base import RecordCollectionBase


class ViewTransformer:
    """
    Transforma un punto desde coordenadas de imagen a coordenadas reales del campo.
    Se integra con RecordCollectionBase para persistir puntos transformados.
    """

    def __init__(self):
        # Medidas reales del campo de fútbol (en metros)
        FIELD_LENGTH = 105.0
        FIELD_WIDTH = 68.0

        # Coordenadas del polígono detectado en la imagen (4 puntos en píxeles)
        self.pixel_vertices = np.array([
            [110, 1035],   # Bottom-left
            [265, 275],    # Top-left
            [910, 260],    # Top-right
            [1640, 915]    # Bottom-right
        ], dtype=np.float32)

        # Corregir contorno para pointPolygonTest
        self.pixel_vertices_contour = self.pixel_vertices.reshape((-1, 1, 2))

        # Mapa de proyección destino en metros
        self.target_vertices = np.array([
            [0, FIELD_WIDTH],     # Bottom-left
            [0, 0],               # Top-left
            [FIELD_LENGTH, 0],    # Top-right
            [FIELD_LENGTH, FIELD_WIDTH]  # Bottom-right
        ], dtype=np.float32)

        # Matriz de transformación perspectiva
        self.perspective_transform = cv2.getPerspectiveTransform(
            self.pixel_vertices,
            self.target_vertices
        )

    # ---------------------------------------------------------
    # TRANSFORMACIÓN DE UN PUNTO INDIVIDUAL
    # ---------------------------------------------------------

    def transform_point(self, point_xy: np.ndarray):
        """
        Transforma un punto x,y a coordenadas reales del campo.
        Retorna None si el punto está fuera del polígono del campo.
        """
        x, y = float(point_xy[0]), float(point_xy[1])
        point_int = (int(x), int(y))

        # Validación: fuera del campo
        if cv2.pointPolygonTest(self.pixel_vertices_contour, point_int, False) < 0:
            return None

        # OpenCV requiere shape (1,1,2)
        p = np.array([[[x, y]]], dtype=np.float32)

        warped = cv2.perspectiveTransform(p, self.perspective_transform)

        return warped.reshape(2).tolist()  # [x,y] en metros

    # ---------------------------------------------------------
    # INTEGRACIÓN CON RECORD COLLECTION BASE
    # ---------------------------------------------------------

    def add_transformed_positions(self, records: RecordCollectionBase):
        """
        Itera sobre todos los registros persistidos y calcula position_transformed.
        Compatible con la API real de RecordCollectionBase.
        """

        print("Transformando posiciones en registros...")
        try:
            all_records = records.get_all()
            for record in all_records:
                pos_adj = getattr(record, "position_adjusted", None)
                if not pos_adj:
                    continue

                px, py = float(pos_adj[0]), float(pos_adj[1])
                transformed = self.transform_point(np.array([px, py], dtype=np.float32))

                # Persiste solo el campo transformado
                records.patch(
                    record.id,
                    {"position_transformed": transformed}
                )
        except Exception as e:
            print(f"Error transformando posiciones en records: {e}")
            raise e
