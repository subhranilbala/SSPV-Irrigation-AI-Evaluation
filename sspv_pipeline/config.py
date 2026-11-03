"""
Global Configuration, Constants & Device Selection.
"""

import warnings

import torch

# Optional dependency flags (shared across all modules)
try:
    import cv2
    HAS_CV2 = True
except ImportError:
    cv2 = None
    HAS_CV2 = False

try:
    import rasterio
    HAS_RASTERIO = True
except ImportError:
    HAS_RASTERIO = False

warnings.filterwarnings("ignore")

# ==============================================================================
# CHANNEL & CLASS DEFINITIONS
# ==============================================================================

CHANNEL_NAMES = [
    "Red", "Green", "Blue",
    "NGRDI", "ExG", "ExR", "VARI", "MGRVI", "NPCI", "TGI", "ExB"
]

SEG_CLASSES = {
    0: "Bare Soil / Background",
    1: "Healthy Crop Canopy",
    2: "Stressed Crop Zone",
    3: "Salinity / White Salt Patch (SSPV Mound)"
}

STRESS_CLASSES = ["Healthy", "Mild Stress", "Severe Stress"]
MOISTURE_CLASSES = ["Dry", "Optimal", "Wet"]
YIELD_CLASSES = ["Low", "Medium", "High"]


def get_device() -> torch.device:
    """Automatically selects CUDA, Apple Silicon MPS (for Mac), or CPU."""
    if torch.cuda.is_available():
        return torch.device("cuda")
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")
