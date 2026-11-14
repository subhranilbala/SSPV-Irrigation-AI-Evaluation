# Module Documentation: `config.py`

**File Path:** [`sspv_pipeline/config.py`](file:///Users/subhranilbala/Downloads/SSPV-Irrigation-AI-Evaluation/sspv_pipeline/config.py)  
**Component:** Global System Configuration & Hardware Auto-Detection  
**Presentation Mapping:** System-wide Global Setup & Hardware Portability  

---

## 1. Module Overview & Purpose

`config.py` serves as the centralized configuration authority for the entire SSPV AI pipeline. It manages:
1. **Dynamic Hardware Acceleration Selection:** Auto-detects NVIDIA CUDA GPUs, Apple Silicon MPS (Metal Performance Shaders on macOS M1/M2/M3/M4), and CPU fallback.
2. **Channel & Tensor Nomenclature:** Defines standard names and order for the 11-channel spectral feature stack.
3. **Class Lexicons:** Canonical class definitions for U-Net 4-class semantic field masks and Multi-Task CNN classification heads (Crop Stress, Soil Moisture, and Yield Potential).
4. **Third-Party Dependency Flags:** Safely probes for optional compiled C-extensions (`cv2`, `rasterio`) and exposes boolean flags (`HAS_CV2`, `HAS_RASTERIO`) to allow headless fallbacks without crash cascades.

---

## 2. Key Constants & Data Structures

### 2.1 Channel Definition (`CHANNEL_NAMES`)
```python
CHANNEL_NAMES = [
    "Red", "Green", "Blue",
    "NGRDI", "ExG", "ExR", "VARI", "MGRVI", "NPCI", "TGI", "ExB"
]
```
- Channels 0–2 represent normalized physical surface reflectances in the visible spectrum.
- Channels 3–10 represent the eight biophysical RGB vegetation and soil indices.

### 2.2 Semantic Segmentation Mask Classes (`SEG_CLASSES`)
```python
SEG_CLASSES = {
    0: "Bare Soil / Background",
    1: "Healthy Crop Canopy",
    2: "Stressed Crop Zone",
    3: "Salinity / White Salt Patch (SSPV Mound)"
}
```
- **Class 0:** Arid soil and background substrate.
- **Class 1:** Photosynthetically active crop leaves.
- **Class 2:** Chlorotic, water-stressed, or wilted crop canopy.
- **Class 3:** White salt crust halos pushed to the outer perimeter of SSPV mounds by outward capillary percolation (Slide 7).

### 2.3 Multi-Task Classification Classes
```python
STRESS_CLASSES   = ["Healthy", "Mild Stress", "Severe Stress"] # Head 1
MOISTURE_CLASSES = ["Dry", "Optimal", "Wet"]                   # Head 2
YIELD_CLASSES    = ["Low", "Medium", "High"]                   # Head 3
```

### 2.4 Regional Agro-Ecological Profiles (`REGIONAL_PROFILES`)
Configures region-specific soil baseline reflectances and climate parameters (with `dehradun_valley` active as `DEFAULT_REGION_KEY`):
- **`dehradun_valley` (Active Default):** Dehradun Doon Valley / Sub-Himalayan Foothill Zone, Mollisol/Inceptisol mountain & alluvial loam, baseline RGB `(0.42, 0.38, 0.32)`, $\text{ET}_0 = 3.8\text{ mm/day}$, low salinity ($0.6\text{ dS/m}$), primary crops: Basmati Rice, Wheat, Mustard, Horticulture.
- **`jodhpur_thar`:** Western Rajasthan / Jodhpur (Kaylana-Aravalli Arid Zone), Aridisol sandy loam, baseline RGB `(0.58, 0.48, 0.38)`, $\text{ET}_0 = 7.2\text{ mm/day}$.
- **`nagaur_semiarid`:** Nagaur Agricultural Research Station, sandy clay loam with caliche.
- **`south_india_laterite`:** Southern India red ferruginous laterite.
- **`indo_gangetic`:** Deep fertile alluvium.

### 2.5 Ground-Truth Plot Geometries & Drone Specifications
- **`REFERENCE_PLOT_GPS`:** Master WGS84 geographic coordinates (`lat: 30.3516097, lon: 77.8742098`, altitude: `536.13m`) corresponding to the geographic centroid of the experimental study area in Dehradun.
- **`GROUND_TRUTH_GEO_POLYS`:** Physical ground polygon coordinates (WGS84 Lat/Lon vertices) for the overall `study_area`, the `sspv_zone` (upper 55% with 25 pitcher mounds), and the `control_zone` (lower 45% with 8 rectangular check basins), enabling real-time photogrammetric projection.
- **`DJI_ZENMUSE_P1_SPECS`:** Optical payload sensor specifications (35.9mm × 24.0mm full-frame sensor, 35.0mm focal length, 8192×5460 native resolution) used to derive exact Ground Sampling Distance (GSD) per flight altitude.
- **`GROUND_TRUTH_ROIS`:** Resolution-independent normalized bounding boxes $[0.0, 1.0]$ for static fallback boundary clipping.

### 2.6 RuTAG Metadata (`RUTAG_METADATA`)
Formally attributes the engineering modernization to **RuTAG (Rural Technology Action Group), IIT Delhi** and records foundational human-ecology principles (capillary steady-state percolation, evaporative cooling in the rhizosphere, root-zone desalination).

---

## 3. Function Details: `get_device()`

### Signature
```python
def get_device() -> torch.device
```

### Execution Flow
1. Checks `torch.cuda.is_available()`. If true, returns `torch.device("cuda")`.
2. Checks `hasattr(torch.backends, "mps") and torch.backends.mps.is_available()`. If true, returns `torch.device("mps")` for zero-configuration hardware acceleration on macOS Apple Silicon.
3. Defaults to `torch.device("cpu")`.

---

## 4. Example Usage

```python
from sspv_pipeline.config import get_device, CHANNEL_NAMES, SEG_CLASSES

device = get_device()
print(f"Executing on hardware device: {device}")
print(f"Total spectral channels: {len(CHANNEL_NAMES)}")
```
