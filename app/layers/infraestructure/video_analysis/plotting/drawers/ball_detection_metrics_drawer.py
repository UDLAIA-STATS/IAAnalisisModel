from matplotlib import pyplot as plt

from app.layers.infraestructure.video_analysis.plotting.interfaces.diagram import Diagram


class BallDetectionMetricsDrawer(Diagram):
    def __init__(self, tracks: dict):
        super().__init__(tracks)
        self.save_path = '../app/res/output_videos/ball_detection.png'
    
    def draw_and_save(self) -> None:
        detection = self.tracks['ball_detection']
        labels = ['Detectados', 'Interpolados']
        sizes = [detection['detected'], detection['interpolated']]
        
        plt.figure(figsize=(8, 6))
        plt.pie(sizes, labels=labels, autopct='%1.1f%%', 
               colors=['#4CAF50', '#FFC107'], startangle=90)
        plt.title('Detección de Balón')
        plt.axis('equal')
        plt.savefig(self.save_path, dpi=120)
        plt.close()