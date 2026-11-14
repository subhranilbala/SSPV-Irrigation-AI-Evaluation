"""
MODULE 01: Global Configuration, Constants & Device Selection.
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

# ==============================================================================
# GROUND-TRUTH EXPERIMENTAL PLOT GEOMETRY & GPS (DEHRADUN RESEARCH SITE)
# ==============================================================================

# Master reference location derived from DJI Zenmuse P1 RTK EXIF header
REFERENCE_PLOT_GPS = {
    "lat": 30.351714,
    "lon": 77.874181,
    "max_plot_radius_m": 16.0,  # Max distance from plot center to be considered an overhead frame
    "altitude_asl_m": 536.13,
    "alt_m": 536.13
}

# Physical Geographic Ground Coordinates (Lat/Lon) of the Experimental Plot in Dehradun
# Enables real-time photogrammetric projection to any arbitrary drone flight orientation, heading, or altitude
GROUND_TRUTH_GEO_POLYS = {
    "study_area": [
        (30.3517670, 77.8741979),
        (30.3515577, 77.8743901),
        (30.3514503, 77.8742209),
        (30.3516639, 77.8740304),
    ],
    "sspv_zone": [
        (30.3517283, 77.8742321),
        (30.3515591, 77.8743852),
        (30.3514546, 77.8742226),
        (30.3516238, 77.8740695),
    ],
    "control_zone": [
        (30.3517613, 77.8742012),
        (30.3517283, 77.8742321),
        (30.3516238, 77.8740695),
        (30.3516568, 77.8740386),
    ]
}

# Drone Optical Payload Specifications (DJI Zenmuse P1 35mm Survey Camera)
DJI_ZENMUSE_P1_SPECS = {
    "sensor_w_mm": 35.9,
    "sensor_h_mm": 24.0,
    "focal_length_mm": 35.0,
    "native_width_px": 8192,
    "native_height_px": 5460
}

# Resolution-independent normalized bounding boxes [0.0 - 1.0]
# Applicable across both native 8192x5460 and downscaled 1024x682 frames
GROUND_TRUTH_ROIS = {
    "study_area": {
        "name": "Total Experimental Study Area (SSPV + Control)",
        "color_rgb": (255, 0, 0),       # Red
        "color_bgr": (0, 0, 255),       # Red
        "ymin": 0.005,
        "ymax": 0.465,
        "xmin": 0.350,
        "xmax": 0.985,
    },
    "sspv_zone": {
        "name": "SSPV Treatment Zone (Porous Pitcher Mounds)",
        "color_rgb": (30, 144, 255),    # Blue / DodgerBlue
        "color_bgr": (255, 144, 30),    # Blue
        "ymin": 0.020,
        "ymax": 0.365,
        "xmin": 0.355,
        "xmax": 0.980,
    },
    "control_zone": {
        "name": "Control Plot (Partitioned Flood Irrigation Basins)",
        "color_rgb": (0, 230, 0),       # Green
        "color_bgr": (0, 230, 0),       # Green
        "ymin": 0.375,
        "ymax": 0.455,
        "xmin": 0.385,
        "xmax": 0.970,
    }
}

# ==============================================================================
# REGIONAL AGRO-ECOLOGICAL & RUTAG CONFIGURATION
# ==============================================================================

RUTAG_METADATA = {
    "organization": "Rural Technology Action Group (RuTAG), IIT Delhi",
    "mentor_institution": "IIT Jodhpur - School of AI & Data Science",
    "project_scope": "Modernized Subsurface Porous Vessel (SSPV) Pitcher Irrigation",
    "human_ecology_principles": [
        "Indigenous water conservation via matric suction",
        "Subsurface evaporative cooling in the rhizosphere",
        "Saline soil mitigation by perimeter salt crust expulsion",
        "Equitable smallholder water governance"
    ]
}

# Regional soil & agro-climatic presets
REGIONAL_PROFILES = {
    "dehradun_valley": {
        "name": "Dehradun (Doon Valley / Sub-Himalayan Foothill Zone)",
        "soil_type": "Mollisol / Inceptisol (Mountain Loam & Alluvial Sandy Loam)",
        "baseline_soil_rgb": (0.42, 0.38, 0.32),  # Rich organic brown-grey loam
        "evaporation_rate_mm_day": 3.8,
        "groundwater_salinity_ec_dsm": 0.6,
        "capillary_wicking_speed": "moderate",
        "salt_crust_propensity": "low",
        "primary_crops": ["Basmati Rice", "Wheat", "Mustard", "Pulses", "Orchards / Horticulture"]
    },
    "jodhpur_thar": {
        "name": "Western Rajasthan / Jodhpur (Kaylana-Aravalli Arid Zone)",
        "soil_type": "Aridisol / Typic Torripsamment (Calcareous Sand / Sandy Loam)",
        "baseline_soil_rgb": (0.58, 0.48, 0.38),  # Warm yellowish-tan sand
        "evaporation_rate_mm_day": 7.2,
        "groundwater_salinity_ec_dsm": 3.4,
        "capillary_wicking_speed": "high",
        "salt_crust_propensity": "high",
        "primary_crops": ["Pearl Millet (Bajra)", "Cluster Bean (Guar)", "Moth Bean", "Mustard"]
    },
    "nagaur_semiarid": {
        "name": "Nagaur Central Rajasthan Agricultural Research Station",
        "soil_type": "Sandy Clay Loam with Kankar (Caliche) Sub-strata",
        "baseline_soil_rgb": (0.52, 0.44, 0.35),
        "evaporation_rate_mm_day": 6.5,
        "groundwater_salinity_ec_dsm": 2.8,
        "capillary_wicking_speed": "moderate",
        "salt_crust_propensity": "high",
        "primary_crops": ["Fenugreek (Methi)", "Cumin (Jeera)", "Wheat", "Mustard"]
    },
    "south_india_laterite": {
        "name": "Southern India (Kerala / Tamil Nadu Red Laterite Zone)",
        "soil_type": "Oxisol / Ultisol (Ferruginous Red Soil / Sacred Grove Catchments)",
        "baseline_soil_rgb": (0.55, 0.36, 0.30),
        "evaporation_rate_mm_day": 4.8,
        "groundwater_salinity_ec_dsm": 0.9,
        "capillary_wicking_speed": "moderate",
        "salt_crust_propensity": "low",
        "primary_crops": ["Spices", "Coconut", "Vegetables", "Pulses"]
    },
    "indo_gangetic": {
        "name": "Indo-Gangetic Alluvial Plain",
        "soil_type": "Inceptisol / Entisol (Deep Fertile Alluvium)",
        "baseline_soil_rgb": (0.45, 0.40, 0.34),
        "evaporation_rate_mm_day": 5.2,
        "groundwater_salinity_ec_dsm": 1.2,
        "capillary_wicking_speed": "moderate",
        "salt_crust_propensity": "moderate",
        "primary_crops": ["Wheat", "Mustard", "Legumes"]
    }
}

# Active default region (calibrated to Dehradun Doon Valley experimental site)
DEFAULT_REGION_KEY = "dehradun_valley"


def get_device() -> torch.device:
    """Automatically selects CUDA, Apple Silicon MPS (for Mac), or CPU."""
    if torch.cuda.is_available():
        return torch.device("cuda")
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")
