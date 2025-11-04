import logging
from typing import Dict, List
from sklearn.cluster import KMeans
from cv2.typing import MatLike
from app.layers.domain.tracks.track_detail import TrackDetailBase


class TeamAssigner:
    def __init__(self):
        self.team_colors = {}
        self.player_team_dict = {}

    def get_clustering_model(self, image):
        # Reshape the image to 2D array
        print("Reshaping image to 2D array, actual image shape: ", image.shape)
        image_2d = image.reshape(-1, 3)
        print("Reshaped image shape: ", image_2d.shape)

        # Preform K-means with 2 clusters
        kmeans = KMeans(n_clusters=2, init="k-means++", n_init=1)
        print("Fitting KMeans model")
        kmeans.fit(image_2d)
        print("KMeans model fitted")

        return kmeans
    
    def get_coords_from_bbox(self, frame: MatLike, bbox: List):
        frame_h, frame_w = frame.shape[:2]
        
        x1 = max(0, int(bbox[0]))
        y1 = max(0, int(bbox[1]))
        x2 = min(frame_w - 1, int(bbox[2]))
        y2 = min(frame_h - 1, int(bbox[3]))
        return x1, y1, x2, y2
    
    def validate_frame(
        self,
        frame: MatLike,
        bbox: List):
        x1, y1, x2, y2 = self.get_coords_from_bbox(frame, bbox)
        
        if x2 <= x1 or y2 <= y1:
            return False
        
        image = frame[y1:y2, x1:x2]
        if image.size == 0:
            return False
        
        top_half_image = image[:int(image.shape[0] / 2), :]
        if top_half_image.size == 0:
            return False
        return True

    def get_player_color(
            self,
            frame: MatLike,
            bbox: List):
        print("Getting player color")
        validate = self.validate_frame(frame, bbox)
        x1, y1, x2, y2 = self.get_coords_from_bbox(frame, bbox)
        if not validate:
            print("Invalid frame or bbox, returning default color.")
            return None
        
        # image = frame[int(bbox[1]):int(bbox[3]), int(bbox[0]):int(bbox[2])]
        image = frame[y1:y2, x1:x2]

        top_half_image = image[:int(image.shape[0] / 2), :]

        # Get Clustering model
        kmeans = self.get_clustering_model(top_half_image)

        # Get the cluster labels forr each pixel
        print("Getting cluster labels")
        labels = kmeans.labels_

        # Reshape the labels to the image shape
        print("Reshaping labels to image shape")
        clustered_image = labels.reshape(
            top_half_image.shape[0], top_half_image.shape[1])

        # Get the player cluster
        print("Getting player cluster")
        corner_clusters = [clustered_image[0, 0], clustered_image[0, -1],
                           clustered_image[-1, 0], clustered_image[-1, -1]]
        non_player_cluster = max(
            set(corner_clusters),
            key=corner_clusters.count)
        player_cluster = 1 - non_player_cluster

        player_color = kmeans.cluster_centers_[player_cluster]
        print("Player color: ", player_color)

        return player_color

    def assign_team_color(
            self,
            frame: MatLike,
            player_detections: Dict[int, TrackDetailBase]):

        player_colors = []
        for _, player_detection in player_detections.items():
            bbox = player_detection.bbox
            if bbox is None:
                print("BBox is None, skipping player detection.")
                continue
            print("Trying to get player color with bbox: ", bbox)
            player_color = self.get_player_color(frame, bbox)
            print("Player color: ", player_color)
            player_colors.append(player_color)

        kmeans = KMeans(n_clusters=2, init="k-means++", n_init=10)
        print("Player colors: ", player_colors)
        kmeans.fit(player_colors)

        self.kmeans = kmeans

        self.team_colors[1] = kmeans.cluster_centers_[0]
        self.team_colors[2] = kmeans.cluster_centers_[1]

    # def get_player_team(self, frame: MatLike, player_bbox, player_id):
    #     if player_id in self.player_team_dict:
    #         return self.player_team_dict[player_id]

    #     player_color = self.get_player_color(frame, player_bbox)
    #     if player_color is None:
    #         logging.debug(f"Could not determine color for player {player_id}, assigning team -1")
    #         return -1

    #     team_id = self.kmeans.predict(player_color.reshape(1, -1))[0]
    #     team_id += 1

    #     if player_id == 91:
    #         team_id = 1

    #     self.player_team_dict[player_id] = team_id

    #     return team_id

    def get_player_team(self, frame: MatLike, player_bbox, player_id: int):
        # Si ya se asignó el equipo previamente, usar ese valor
        if player_id in self.player_team_dict:
            return self.player_team_dict[player_id]

        # Validar existencia del modelo
        if not hasattr(self, "kmeans"):
            logging.debug("KMeans model not found, cannot assign team.")
            return -1

        # Obtener color dominante del jugador
        player_color = self.get_player_color(frame, player_bbox)
        if player_color is None:
            logging.debug(f"⚠️ Could not get color for player {player_id}, bbox={player_bbox}")
            return -1

        # Predicción del equipo
        try:
            team_id = int(self.kmeans.predict(player_color.reshape(1, -1))[0]) + 1
        except Exception as e:
            logging.debug(f"⚠️ Error predicting team for player {player_id}: {e}")
            return -1

        # Guardar resultado en cache
        self.player_team_dict[player_id] = team_id
        logging.debug(f"✅ Player {player_id} assigned to team {team_id}")

        return team_id
