"""
================================================================================
SSPV Irrigation AI Evaluation Pipeline — Package Init
================================================================================
M.Tech Thesis Project (MTP)
Title: AI Based Evaluation of Empirical Subsurface Porous Vessel (SSPV)
       Irrigation System Performance Using Images
Author: Subhranil Bala (Roll No.: M25AI2038)
Supervisor: Dr. Anand Krishnan Plappally
Institute: Indian Institute of Technology Jodhpur (School of AI & Data Science)
================================================================================
"""

from .config import (
    CHANNEL_NAMES, SEG_CLASSES,
    STRESS_CLASSES, MOISTURE_CLASSES, YIELD_CLASSES,
    get_device, HAS_CV2, HAS_RASTERIO,
    GROUND_TRUTH_ROIS, REFERENCE_PLOT_GPS,
    GROUND_TRUTH_GEO_POLYS, DJI_ZENMUSE_P1_SPECS
)
from .preprocessing import (
    UAVPreprocessor,
    extract_exif_gps,
    compute_gps_distance,
    extract_dji_flight_telemetry,
    project_geo_polygon_to_image,
    detect_realtime_plot_boundaries,
    draw_ground_truth_boundaries,
)
from .feature_engineering import VegetationIndexExtractor
from .pseudo_labeling import BiophysicalPseudoLabeler
from .models import (
    UNetResNet18_11Ch, MultiTaskSSPVNet, MultiTaskLoss,
    adapt_resnet_first_conv_to_11ch, adapt_efficientnet_first_conv_to_11ch,
)
from .explainability import MultiTaskGradCAM
from .statistics_engine import StatisticalComparisonEngine
from .recommendation import HybridRecommendationEngine
from .dataset import (
    SSPVDronePatchDataset,
    generate_realistic_7date_field_dataset,
    load_real_dji_folders_or_fallback,
    is_frame_in_study_plot,
)
from .orchestrator import SSPVPipelineOrchestrator
from .dashboard import run_streamlit_dashboard

__all__ = [
    # Config
    "CHANNEL_NAMES", "SEG_CLASSES", "STRESS_CLASSES", "MOISTURE_CLASSES",
    "YIELD_CLASSES", "get_device", "HAS_CV2", "HAS_RASTERIO",
    "GROUND_TRUTH_ROIS", "REFERENCE_PLOT_GPS",
    "GROUND_TRUTH_GEO_POLYS", "DJI_ZENMUSE_P1_SPECS",
    # Preprocessing
    "UAVPreprocessor", "extract_exif_gps", "compute_gps_distance",
    "extract_dji_flight_telemetry", "project_geo_polygon_to_image",
    "detect_realtime_plot_boundaries", "draw_ground_truth_boundaries",
    # Feature Engineering
    "VegetationIndexExtractor",
    # Pseudo-Labeling
    "BiophysicalPseudoLabeler",
    # Models
    "UNetResNet18_11Ch", "MultiTaskSSPVNet", "MultiTaskLoss",
    "adapt_resnet_first_conv_to_11ch", "adapt_efficientnet_first_conv_to_11ch",
    # Explainability
    "MultiTaskGradCAM",
    # Statistics
    "StatisticalComparisonEngine",
    # Recommendation
    "HybridRecommendationEngine",
    # Dataset
    "SSPVDronePatchDataset", "generate_realistic_7date_field_dataset",
    "load_real_dji_folders_or_fallback", "is_frame_in_study_plot",
    # Orchestrator
    "SSPVPipelineOrchestrator",
    # Dashboard
    "run_streamlit_dashboard",
]
