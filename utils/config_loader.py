"""
Configuration Loader for Football Analysis System

This module provides a type-safe configuration loader that reads settings
from a YAML file and validates them.

Usage:
    from utils.config_loader import load_config

    config = load_config('config.yaml')
    print(config.video.input_path)
    print(config.model.confidence_threshold)
"""

import yaml
import os
from dataclasses import dataclass, field
from typing import List, Dict, Optional, Any
from pathlib import Path


@dataclass
class VideoConfig:
    """Video input/output configuration"""
    input_path: str = "input_videos/08fd33_4.mp4"
    output_path: str = "output_videos/output_video.avi"
    fps: int = 24
    codec: str = "XVID"
    read_all_frames: bool = True
    chunk_size: int = 100


@dataclass
class ModelConfig:
    """Model and detection configuration"""
    path: str = "models/best.pt"
    confidence_threshold: float = 0.1
    batch_size: int = 20
    class_confidence: Dict[str, float] = field(default_factory=dict)
    use_fp16: bool = False
    device: str = "cuda"


@dataclass
class TrackingConfig:
    """Object tracking configuration"""
    tracker_type: str = "bytetrack"
    min_hits: int = 3
    max_age: int = 30
    interpolate_ball: bool = True
    interpolation_method: str = "linear"


@dataclass
class TeamAssignmentConfig:
    """Team assignment configuration"""
    method: str = "kmeans"
    num_clusters: int = 2
    kmeans_init: str = "k-means++"
    kmeans_iterations: int = 10
    use_top_half: bool = True
    color_space: str = "rgb"
    cache_assignments: bool = True
    use_consensus_voting: bool = False
    consensus_window: int = 30


@dataclass
class BallPossessionConfig:
    """Ball possession configuration"""
    max_distance: int = 70
    use_foot_position: bool = True
    check_both_feet: bool = True


@dataclass
class CameraMovementConfig:
    """Camera movement estimation configuration"""
    minimum_distance: int = 5
    window_size: List[int] = field(default_factory=lambda: [15, 15])
    max_pyramid_level: int = 2
    criteria_eps: float = 0.03
    criteria_max_count: int = 10
    max_corners: int = 100
    quality_level: float = 0.3
    min_distance: int = 3
    block_size: int = 7
    mask_areas: List[List[int]] = field(default_factory=lambda: [[0, 20], [900, 1050]])


@dataclass
class PerspectiveConfig:
    """Perspective transformation configuration"""
    court_width: float = 68.0
    court_length: float = 23.32
    pixel_vertices: List[List[int]] = field(default_factory=lambda: [
        [110, 1035],
        [265, 275],
        [910, 260],
        [1640, 915]
    ])
    auto_calculate_target: bool = True


@dataclass
class SpeedDistanceConfig:
    """Speed and distance calculation configuration"""
    frame_window: int = 5
    frame_rate: int = 24
    speed_unit: str = "km/h"
    distance_unit: str = "m"
    track_players: bool = True
    track_ball: bool = False
    track_referees: bool = False


@dataclass
class BallControlPosition:
    """Ball control panel position"""
    x: int = 1350
    y: int = 850
    width: int = 550
    height: int = 120


@dataclass
class CameraMovementPosition:
    """Camera movement panel position"""
    x: int = 0
    y: int = 0
    width: int = 500
    height: int = 100


@dataclass
class EllipseConfig:
    """Player ellipse annotation configuration"""
    width_ratio: float = 1.0
    height_ratio: float = 0.35
    start_angle: int = -45
    end_angle: int = 235
    thickness: int = 2


@dataclass
class IDBadgeConfig:
    """Player ID badge configuration"""
    width: int = 40
    height: int = 20
    offset_y: int = 15
    font_scale: float = 0.6
    font_thickness: int = 2


@dataclass
class VisualizationConfig:
    """Visualization configuration"""
    show_players: bool = True
    show_referees: bool = True
    show_ball: bool = True
    show_player_ids: bool = True
    show_speed: bool = True
    show_distance: bool = True
    show_ball_control: bool = True
    show_camera_movement: bool = True
    show_trails: bool = False
    colors: Dict[str, List[int]] = field(default_factory=lambda: {
        "referee": [0, 255, 255],
        "ball": [0, 255, 0],
        "ball_possession": [0, 0, 255],
        "default_player": [0, 0, 255]
    })
    ball_control_position: BallControlPosition = field(default_factory=BallControlPosition)
    camera_movement_position: CameraMovementPosition = field(default_factory=CameraMovementPosition)
    overlay_alpha: float = 0.4
    font: str = "FONT_HERSHEY_SIMPLEX"
    font_scale: float = 1.0
    font_thickness: int = 3
    font_color: List[int] = field(default_factory=lambda: [0, 0, 0])
    ellipse: EllipseConfig = field(default_factory=EllipseConfig)
    id_badge: IDBadgeConfig = field(default_factory=IDBadgeConfig)


@dataclass
class CachingConfig:
    """Caching configuration"""
    enabled: bool = True
    cache_dir: str = "stubs"
    tracks_stub: str = "stubs/track_stubs.pkl"
    camera_movement_stub: str = "stubs/camera_movement_stub.pkl"
    read_from_stub: bool = True
    write_to_stub: bool = True


@dataclass
class PerformanceConfig:
    """Performance configuration"""
    use_gpu: bool = True
    gpu_id: int = 0
    num_workers: int = 4
    enable_multiprocessing: bool = False
    max_memory_gb: int = 8
    enable_garbage_collection: bool = True
    gc_frequency: int = 100


@dataclass
class LoggingConfig:
    """Logging configuration"""
    level: str = "INFO"
    log_to_console: bool = True
    log_to_file: bool = True
    log_file: str = "logs/football_analysis.log"
    format: str = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    date_format: str = "%Y-%m-%d %H:%M:%S"
    max_file_size_mb: int = 10
    backup_count: int = 5


@dataclass
class AdvancedConfig:
    """Advanced features configuration"""
    generate_heatmaps: bool = False
    heatmap_resolution: List[int] = field(default_factory=lambda: [100, 100])
    detect_passes: bool = False
    detect_shots: bool = False
    detect_sprints: bool = False
    detect_formation: bool = False
    analyze_pressing: bool = False
    export_json: bool = False
    export_csv: bool = False
    json_output_path: str = "output/analysis.json"
    csv_output_path: str = "output/stats.csv"


@dataclass
class DebugConfig:
    """Debug configuration"""
    enabled: bool = False
    show_bboxes: bool = False
    show_feature_points: bool = False
    show_optical_flow: bool = False
    save_debug_frames: bool = False
    debug_frames_dir: str = "debug_output"
    profile_performance: bool = False
    print_timing_stats: bool = False


@dataclass
class Config:
    """Main configuration class containing all settings"""
    video: VideoConfig = field(default_factory=VideoConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    tracking: TrackingConfig = field(default_factory=TrackingConfig)
    team_assignment: TeamAssignmentConfig = field(default_factory=TeamAssignmentConfig)
    ball_possession: BallPossessionConfig = field(default_factory=BallPossessionConfig)
    camera_movement: CameraMovementConfig = field(default_factory=CameraMovementConfig)
    perspective: PerspectiveConfig = field(default_factory=PerspectiveConfig)
    speed_distance: SpeedDistanceConfig = field(default_factory=SpeedDistanceConfig)
    visualization: VisualizationConfig = field(default_factory=VisualizationConfig)
    caching: CachingConfig = field(default_factory=CachingConfig)
    performance: PerformanceConfig = field(default_factory=PerformanceConfig)
    logging: LoggingConfig = field(default_factory=LoggingConfig)
    advanced: AdvancedConfig = field(default_factory=AdvancedConfig)
    debug: DebugConfig = field(default_factory=DebugConfig)


def _nested_update(base_dict: Dict[str, Any], update_dict: Dict[str, Any]) -> Dict[str, Any]:
    """
    Recursively update nested dictionaries.

    Args:
        base_dict: Base dictionary
        update_dict: Dictionary with updates

    Returns:
        Updated dictionary
    """
    for key, value in update_dict.items():
        if isinstance(value, dict) and key in base_dict and isinstance(base_dict[key], dict):
            base_dict[key] = _nested_update(base_dict[key], value)
        else:
            base_dict[key] = value
    return base_dict


def _dict_to_dataclass(dataclass_type, data: Dict[str, Any]):
    """
    Convert dictionary to dataclass instance recursively.

    Args:
        dataclass_type: The dataclass type to instantiate
        data: Dictionary with data

    Returns:
        Dataclass instance
    """
    if data is None:
        return dataclass_type()

    # If data is already a dataclass instance, return it
    if hasattr(data, '__dataclass_fields__'):
        return data

    # Get dataclass fields
    field_types = {f.name: f.type for f in dataclass_type.__dataclass_fields__.values()}

    kwargs = {}
    for field_name, field_type in field_types.items():
        if field_name not in data:
            continue

        field_value = data[field_name]

        # If field_value is already a dataclass instance, use it directly
        if hasattr(field_value, '__dataclass_fields__'):
            kwargs[field_name] = field_value
        # Check if the field type is a dataclass and value is a dict
        elif hasattr(field_type, '__dataclass_fields__') and isinstance(field_value, dict):
            kwargs[field_name] = _dict_to_dataclass(field_type, field_value)
        else:
            kwargs[field_name] = field_value

    return dataclass_type(**kwargs)


def load_config(config_path: str = "config.yaml") -> Config:
    """
    Load configuration from YAML file.

    Args:
        config_path: Path to configuration file

    Returns:
        Config object with all settings

    Raises:
        FileNotFoundError: If config file doesn't exist
        ValueError: If config file is invalid
    """
    # Check if file exists
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Configuration file not found: {config_path}")

    # Load YAML file
    try:
        with open(config_path, 'r', encoding='utf-8') as f:
            config_dict = yaml.safe_load(f)
    except yaml.YAMLError as e:
        raise ValueError(f"Invalid YAML in configuration file: {e}")

    if config_dict is None:
        config_dict = {}

    # Convert to Config object
    try:
        config = Config()

        # Update each section
        if 'video' in config_dict:
            config.video = _dict_to_dataclass(VideoConfig, config_dict['video'])
        if 'model' in config_dict:
            config.model = _dict_to_dataclass(ModelConfig, config_dict['model'])
        if 'tracking' in config_dict:
            config.tracking = _dict_to_dataclass(TrackingConfig, config_dict['tracking'])
        if 'team_assignment' in config_dict:
            config.team_assignment = _dict_to_dataclass(TeamAssignmentConfig, config_dict['team_assignment'])
        if 'ball_possession' in config_dict:
            config.ball_possession = _dict_to_dataclass(BallPossessionConfig, config_dict['ball_possession'])
        if 'camera_movement' in config_dict:
            config.camera_movement = _dict_to_dataclass(CameraMovementConfig, config_dict['camera_movement'])
        if 'perspective' in config_dict:
            config.perspective = _dict_to_dataclass(PerspectiveConfig, config_dict['perspective'])
        if 'speed_distance' in config_dict:
            config.speed_distance = _dict_to_dataclass(SpeedDistanceConfig, config_dict['speed_distance'])
        if 'visualization' in config_dict:
            # Handle nested dataclasses in visualization
            viz_data = config_dict['visualization'].copy()
            if 'ball_control_position' in viz_data and isinstance(viz_data['ball_control_position'], dict):
                viz_data['ball_control_position'] = _dict_to_dataclass(
                    BallControlPosition, viz_data['ball_control_position']
                )
            if 'camera_movement_position' in viz_data and isinstance(viz_data['camera_movement_position'], dict):
                viz_data['camera_movement_position'] = _dict_to_dataclass(
                    CameraMovementPosition, viz_data['camera_movement_position']
                )
            if 'ellipse' in viz_data and isinstance(viz_data['ellipse'], dict):
                viz_data['ellipse'] = _dict_to_dataclass(EllipseConfig, viz_data['ellipse'])
            if 'id_badge' in viz_data and isinstance(viz_data['id_badge'], dict):
                viz_data['id_badge'] = _dict_to_dataclass(IDBadgeConfig, viz_data['id_badge'])
            config.visualization = _dict_to_dataclass(VisualizationConfig, viz_data)
        if 'caching' in config_dict:
            config.caching = _dict_to_dataclass(CachingConfig, config_dict['caching'])
        if 'performance' in config_dict:
            config.performance = _dict_to_dataclass(PerformanceConfig, config_dict['performance'])
        if 'logging' in config_dict:
            config.logging = _dict_to_dataclass(LoggingConfig, config_dict['logging'])
        if 'advanced' in config_dict:
            config.advanced = _dict_to_dataclass(AdvancedConfig, config_dict['advanced'])
        if 'debug' in config_dict:
            config.debug = _dict_to_dataclass(DebugConfig, config_dict['debug'])

    except Exception as e:
        raise ValueError(f"Error parsing configuration: {e}")

    return config


def save_config(config: Config, output_path: str = "config.yaml"):
    """
    Save configuration to YAML file.

    Args:
        config: Config object to save
        output_path: Path to save configuration file
    """
    # Convert dataclass to dictionary
    from dataclasses import asdict
    config_dict = asdict(config)

    # Save to YAML
    with open(output_path, 'w', encoding='utf-8') as f:
        yaml.dump(config_dict, f, default_flow_style=False, sort_keys=False)


def get_default_config() -> Config:
    """
    Get default configuration without loading from file.

    Returns:
        Config object with default values
    """
    return Config()


# Example usage
if __name__ == "__main__":
    # Load configuration
    config = load_config("config.yaml")

    # Access settings
    print(f"Video input: {config.video.input_path}")
    print(f"Model path: {config.model.path}")
    print(f"Batch size: {config.model.batch_size}")
    print(f"Ball possession distance: {config.ball_possession.max_distance}")
    print(f"Ball control position: ({config.visualization.ball_control_position.x}, {config.visualization.ball_control_position.y})")
