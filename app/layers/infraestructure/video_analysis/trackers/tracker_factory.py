from app.layers.infraestructure.video_analysis.trackers.tracker import Tracker


class TrackerFactory:
    def create_tracker(self, tracker_type: str) -> Tracker:
        raise NotImplementedError