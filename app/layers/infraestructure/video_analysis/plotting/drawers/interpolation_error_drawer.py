from layers.infraestructure.video_analysis.plotting.interfaces.diagram import \
    Diagram
from matplotlib import pyplot as plt


class InterpolationErrorDrawer(Diagram):
    def __init__(self, tracks: dict):
        super().__init__(tracks)
        self.save_path = '../app/res/output_videos/interpolation_error.png'

    def draw_and_save(self) -> None:
        error = self.tracks['interpolation_error']
        plt.figure(figsize=(6, 6))
        plt.bar(['Error MSE'], [error], color='#FF5722')
        plt.title('Error de Interpolación del Balón', fontweight='bold')
        plt.ylabel('Error Cuadrático Medio')
        plt.grid(axis='y')
        plt.text(0, error + 0.01, f"{error:.4f}",
                 ha='center', fontweight='bold')
        plt.savefig(self.save_path, dpi=120)
        plt.close()
