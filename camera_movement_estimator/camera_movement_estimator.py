import pickle
import cv2
import numpy as np
import os
import sys
sys.path.append('../')
from utils import measure_distance, measure_xy_distance

class CameraMovementEstimator():
    def __init__(self, frame, config):
        """
        Initialize camera movement estimator with configuration.

        Args:
            frame: First frame of video
            config: Configuration object from config_loader
        """
        self.config = config
        self.minimum_distance = config.camera_movement.minimum_distance

        # Lucas-Kanade optical flow parameters
        self.lk_params = dict(
            winSize=tuple(config.camera_movement.window_size),
            maxLevel=config.camera_movement.max_pyramid_level,
            criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT,
                     config.camera_movement.criteria_max_count,
                     config.camera_movement.criteria_eps)
        )

        # Feature detection setup
        first_frame_grayscale = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        mask_features = np.zeros_like(first_frame_grayscale)

        # Apply mask areas from config
        for mask_area in config.camera_movement.mask_areas:
            mask_features[:, mask_area[0]:mask_area[1]] = 1

        self.features = dict(
            maxCorners=config.camera_movement.max_corners,
            qualityLevel=config.camera_movement.quality_level,
            minDistance=config.camera_movement.min_distance,
            blockSize=config.camera_movement.block_size,
            mask=mask_features
        )

    def add_adjust_positions_to_tracks(self,tracks, camera_movement_per_frame):
        for object, object_tracks in tracks.items():
            for frame_num, track in enumerate(object_tracks):
                for track_id, track_info in track.items():
                    position = track_info['position']
                    camera_movement = camera_movement_per_frame[frame_num]
                    position_adjusted = (position[0]-camera_movement[0],position[1]-camera_movement[1])
                    tracks[object][frame_num][track_id]['position_adjusted'] = position_adjusted
                    


    def get_camera_movement(self,frames,read_from_stub=False, stub_path=None):
        # Read the stub 
        if read_from_stub and stub_path is not None and os.path.exists(stub_path):
            with open(stub_path,'rb') as f:
                return pickle.load(f)

        camera_movement = [[0,0]]*len(frames)

        old_gray = cv2.cvtColor(frames[0],cv2.COLOR_BGR2GRAY)
        old_features = cv2.goodFeaturesToTrack(old_gray,**self.features)

        for frame_num in range(1,len(frames)):
            frame_gray = cv2.cvtColor(frames[frame_num],cv2.COLOR_BGR2GRAY)
            new_features, _,_ = cv2.calcOpticalFlowPyrLK(old_gray,frame_gray,old_features,None,**self.lk_params)

            max_distance = 0
            camera_movement_x, camera_movement_y = 0,0

            for i, (new,old) in enumerate(zip(new_features,old_features)):
                new_features_point = new.ravel()
                old_features_point = old.ravel()

                distance = measure_distance(new_features_point,old_features_point)
                if distance>max_distance:
                    max_distance = distance
                    camera_movement_x,camera_movement_y = measure_xy_distance(old_features_point, new_features_point ) 
            
            if max_distance > self.minimum_distance:
                camera_movement[frame_num] = [camera_movement_x,camera_movement_y]
                old_features = cv2.goodFeaturesToTrack(frame_gray,**self.features)

            old_gray = frame_gray.copy()
        
        if stub_path is not None:
            with open(stub_path,'wb') as f:
                pickle.dump(camera_movement,f)

        return camera_movement
    
    def draw_camera_movement(self, frames, camera_movement_per_frame):
        """
        Draw camera movement visualization on frames.

        Args:
            frames: List of video frames
            camera_movement_per_frame: List of [x, y] movement per frame

        Returns:
            List of frames with camera movement overlay
        """
        output_frames = []

        # Get position and styling from config
        pos = self.config.visualization.camera_movement_position
        alpha = self.config.visualization.overlay_alpha
        font = getattr(cv2, self.config.visualization.font)
        font_scale = self.config.visualization.font_scale
        font_thickness = self.config.visualization.font_thickness
        font_color = tuple(self.config.visualization.font_color)

        for frame_num, frame in enumerate(frames):
            frame = frame.copy()

            overlay = frame.copy()
            cv2.rectangle(
                overlay,
                (pos.x, pos.y),
                (pos.x + pos.width, pos.y + pos.height),
                (255, 255, 255),
                -1
            )
            cv2.addWeighted(overlay, alpha, frame, 1-alpha, 0, frame)

            x_movement, y_movement = camera_movement_per_frame[frame_num]
            frame = cv2.putText(
                frame,
                f"Camera Movement X: {x_movement:.2f}",
                (pos.x + 10, pos.y + 30),
                font, font_scale, font_color, font_thickness
            )
            frame = cv2.putText(
                frame,
                f"Camera Movement Y: {y_movement:.2f}",
                (pos.x + 10, pos.y + 60),
                font, font_scale, font_color, font_thickness
            )

            output_frames.append(frame)

        return output_frames