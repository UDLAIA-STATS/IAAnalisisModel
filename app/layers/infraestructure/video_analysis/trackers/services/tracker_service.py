import pickle
from typing import override

import supervision as sv
from cv2.typing import MatLike
from app.layers.domain.collections.track_collection import TrackCollection
from app.layers.infraestructure.video_analysis.trackers.interfaces import \
    TrackerServiceBase


class TrackerService(TrackerServiceBase):

    def __init__(self, model_path: str):
        super().__init__(model_path)
        self.detection_frame: sv.Detections | None = None

    @override
    def get_object_tracks(
        self,
        frames: list[MatLike],
        tracks_collection: TrackCollection,
        read_from_stub: bool = False,
        stub_path: str = ""
    ):
        if read_from_stub and stub_path:
            tracks = self.read_tracks_from_stub(stub_path)
            print(f"Tracks loaded players from stub: {tracks.pop('players', None)}")
            print(f"Tracks loaded ball from stub: {tracks.pop('ball', None)}")

        results = self.detect_frames(frames)

        for frame_num, detection in enumerate(results):
            cls_names = detection.names
            cls_names_inv = {v: k for k, v in cls_names.items()}
            print(cls_names_inv)

            # Covert to supervision Detection format
            detection_supervision = sv.Detections.from_ultralytics(detection)
            print(detection_supervision.data.keys())

            # Track Objects
            detection_with_tracks =  self.tracker.update_with_detections(detection_supervision)
            if not self.detection_frame: self.detection_frame = detection_with_tracks

            for _, val in enumerate(self.get_trackers()):
                val.get_object_tracks(
                    detection_with_tracks=detection_with_tracks,
                    cls_names_inv=cls_names_inv,
                    frame_num=frame_num,
                    detection_supervision=detection_supervision,
                    tracks_collection = tracks_collection
                )

        # if stub_path is not None:
        #     with open(stub_path, 'wb') as f:
        #         pickle.dump(tracks, f)

