# Module Documentation: `dataset.py`

**File Path:** [`sspv_pipeline/dataset.py`](file:///Users/subhranilbala/Downloads/SSPV-Irrigation-AI-Evaluation/sspv_pipeline/dataset.py)  
**Component:** Multi-Date Drone Ingestion & Realistic Synthetic Field Simulator  
**Presentation Mapping:** Slide 10 (Data Acquisition), Slide 11 (Spectral Bands), Slide 16 (Synthetic Validation)  

---

## 1. Module Overview & Purpose

`dataset.py` provides the data access layer for the AI pipeline. It is architected with dual functionality:
1. **Real DJI Zenmuse P1 Ingestion:** Traverses nested multi-date flight folders (e.g., `02-12-2024/images`, `12-11-2024/images`, etc.), uniformly strides across redundant 80% flight overlaps, and extracts non-overlapping high-resolution patches.
2. **Physically Realistic 7-Date Field Simulator:** Generates a synthetic multi-date benchmark modeling circular SSPV porous vessels, vegetative halos, outer salt crust halos, and conventional control rows with arid soil profiles matching IIT Jodhpur / Nagaur research sites.

---

## 2. Real Drone Data Ingestion (`load_real_dji_folders_or_fallback`)

### Folder Structure Expected:
```
<Crops Data>/
├── 12-11-2024/
│   └── images/
│       ├── DJI_0001.JPG
│       └── DJI_0002.JPG
├── 22-11-2024/
│   └── images/
│       └── ...
└── 02-12-2024/
```

### Ingestion Logic:
1. Parses date strings matching `%d-%m-%Y` and sorts sequentially to compute elapsed flight days ($t = 1, 11, 21, \dots$).
2. Prioritizes `<folder>/images/` matching DJI flight outputs.
3. Automatically filters out macOS AppleDouble resource fork files (`._DJI_...`).
4. **Uniform Striding:** Drone flights typically take photos with 80% forward/lateral overlap. If `max_images_per_date` is specified (e.g., 20), it takes an evenly spaced linspace of frames across the entire flight path to cover the full field plot without training on redundant duplicates.
5. Extracts Upper-Right patches (designated SSPV mounds) and Lower-Right patches (designated Control rows).

---

## 3. Realistic 7-Date Field Simulator (`generate_realistic_7date_field_dataset`)

When local drone imagery is not provided or runs in automated test environments, the simulator synthesizes realistic spectral signatures:

1. **Dates Modeled:** Exactly matches the user's flight calendar:
   - `12-11-2024` (Day 1)
   - `22-11-2024` (Day 11)
   - `02-12-2024` (Day 21)
   - `12-12-2024` (Day 31)
   - `25-12-2024` (Day 44)
   - `05-01-2025` (Day 55)
   - `15-01-2025` (Day 65)
2. **Arid Sandy Soil Profile:** Normal distribution around $\mu = [0.46, 0.38, 0.27]$ reflecting Western Rajasthan soil reflectance.
3. **SSPV Spatial Geometry:**
   - Circular vessels with radius $r = 0.07 \cdot \text{patch\_size}$.
   - Moisture-darkened wetted zone ($dist < 0.30 \cdot \text{patch\_size}$).
   - Healthy green canopy ring expanding as $f_{\text{growth}}(t) = 0.35 + \frac{t}{70} \times 0.65$.
   - White salt crust outer halo ($dist \in [0.26, 0.31] \cdot \text{patch\_size}$, high luminance $[0.82, 0.81, 0.79]$) representing outward salt percolation (Slide 7).
4. **Control Plot Geometry:** Linear row crops with higher proportion of chlorosis and intermittent moisture stress shocks on dry survey days.

---

## 4. PyTorch Dataset (`SSPVDronePatchDataset`)

Implements standard `torch.utils.data.Dataset`:
- `__len__()`: Number of patches.
- `__getitem__(idx)`:
  - Generates 11-channel tensor via [`VegetationIndexExtractor`](file:///Users/subhranilbala/Downloads/SSPV-Irrigation-AI-Evaluation/sspv_pipeline/feature_engineering.py#L12)
  - Generates 4-class segmentation mask via [`BiophysicalPseudoLabeler`](file:///Users/subhranilbala/Downloads/SSPV-Irrigation-AI-Evaluation/sspv_pipeline/pseudo_labeling.py#L12)
  - Returns dictionary with tensors ready for PyTorch `DataLoader`:
    ```python
    {
        "tensor_11ch": torch.Tensor (11, H, W),
        "seg_mask": torch.Tensor (H, W),
        "stress_label": torch.Tensor (scalar),
        "moisture_label": torch.Tensor (scalar),
        "yield_label": torch.Tensor (scalar),
        "date": str,
        "day_num": int,
        "plot_type": str,
        "rgb": np.ndarray (H, W, 3)
    }
    ```

---

## 5. Example Usage

```python
from torch.utils.data import DataLoader
from sspv_pipeline.preprocessing import UAVPreprocessor
from sspv_pipeline.feature_engineering import VegetationIndexExtractor
from sspv_pipeline.dataset import load_real_dji_folders_or_fallback, SSPVDronePatchDataset

prep = UAVPreprocessor()
vi = VegetationIndexExtractor()

samples = load_real_dji_folders_or_fallback("~/Downloads/Crops Data", prep, max_images_per_date=10)
dataset = SSPVDronePatchDataset(samples, prep, vi)
loader = DataLoader(dataset, batch_size=4, shuffle=True)

batch = next(iter(loader))
print("Batch 11-ch tensor shape:", batch["tensor_11ch"].shape)  # (4, 11, 256, 256)
```
