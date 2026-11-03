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
)
from .preprocessing import UAVPreprocessor
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
)
from .orchestrator import SSPVPipelineOrchestrator
from .dashboard import run_streamlit_dashboard

__all__ = [
    # Config
    "CHANNEL_NAMES", "SEG_CLASSES", "STRESS_CLASSES", "MOISTURE_CLASSES",
    "YIELD_CLASSES", "get_device", "HAS_CV2", "HAS_RASTERIO",
    # Preprocessing
    "UAVPreprocessor",
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
    "load_real_dji_folders_or_fallback",
    # Orchestrator
    "SSPVPipelineOrchestrator",
    # Dashboard
    "run_streamlit_dashboard",
]
