from sklearn.cluster import KMeans

class TeamAssigner:
    def __init__(self, config):
        """
        Initialize team assigner with configuration.

        Args:
            config: Configuration object from config_loader
        """
        self.config = config
        self.team_colors = {}
        self.player_team_dict = {}

    def get_clustering_model(self, image):
        """
        Create K-Means clustering model for color segmentation.

        Args:
            image: Image to cluster

        Returns:
            Fitted K-Means model
        """
        # Reshape the image to 2D array
        image_2d = image.reshape(-1, 3)

        # Perform K-means with 2 clusters (background vs jersey)
        kmeans = KMeans(
            n_clusters=2,
            init=self.config.team_assignment.kmeans_init,
            n_init=1
        )
        kmeans.fit(image_2d)

        return kmeans

    def get_player_color(self, frame, bbox):
        """
        Extract player jersey color from bounding box.

        Args:
            frame: Video frame
            bbox: Player bounding box [x1, y1, x2, y2]

        Returns:
            RGB color array
        """
        image = frame[int(bbox[1]):int(bbox[3]), int(bbox[0]):int(bbox[2])]

        # Use top half of image to avoid shorts/legs
        if self.config.team_assignment.use_top_half:
            top_half_image = image[0:int(image.shape[0]/2), :]
        else:
            top_half_image = image

        # Get Clustering model
        kmeans = self.get_clustering_model(top_half_image)

        # Get the cluster labels forr each pixel
        labels = kmeans.labels_

        # Reshape the labels to the image shape
        clustered_image = labels.reshape(top_half_image.shape[0],top_half_image.shape[1])

        # Get the player cluster
        corner_clusters = [clustered_image[0,0],clustered_image[0,-1],clustered_image[-1,0],clustered_image[-1,-1]]
        non_player_cluster = max(set(corner_clusters),key=corner_clusters.count)
        player_cluster = 1 - non_player_cluster

        player_color = kmeans.cluster_centers_[player_cluster]

        return player_color


    def assign_team_color(self, frame, player_detections):
        """
        Assign team colors based on player jersey colors.

        Args:
            frame: Video frame
            player_detections: Dictionary of player detections
        """
        player_colors = []
        for _, player_detection in player_detections.items():
            bbox = player_detection["bbox"]
            player_color = self.get_player_color(frame, bbox)
            player_colors.append(player_color)

        # Cluster player colors into teams
        kmeans = KMeans(
            n_clusters=self.config.team_assignment.num_clusters,
            init=self.config.team_assignment.kmeans_init,
            n_init=self.config.team_assignment.kmeans_iterations
        )
        kmeans.fit(player_colors)

        self.kmeans = kmeans

        self.team_colors[1] = kmeans.cluster_centers_[0]
        self.team_colors[2] = kmeans.cluster_centers_[1]


    def get_player_team(self,frame,player_bbox,player_id):
        if player_id in self.player_team_dict:
            return self.player_team_dict[player_id]

        player_color = self.get_player_color(frame,player_bbox)

        team_id = self.kmeans.predict(player_color.reshape(1,-1))[0]
        team_id+=1

        # TODO: Improve color detection robustness instead of hardcoding player assignments
        # Consider: lighting normalization, HSV color space, multi-frame consensus

        self.player_team_dict[player_id] = team_id

        return team_id
