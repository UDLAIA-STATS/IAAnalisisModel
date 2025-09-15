class BoundingBoxCoordinates():
    x1: float
    y1: float
    x2: float
    y2: float

    def __init__(self, x1: float, y1: float, x2: float, y2: float):
        """Inicializa las coordenadas del bounding box"""
        self.x1 = x1
        self.y1 = y1
        self.x2 = x2
        self.y2 = y2
