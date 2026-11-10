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
