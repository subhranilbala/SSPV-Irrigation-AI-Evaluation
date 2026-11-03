# Module Documentation: `feature_engineering.py`

**File Path:** [`sspv_pipeline/feature_engineering.py`](file:///Users/subhranilbala/Downloads/SSPV-Irrigation-AI-Evaluation/sspv_pipeline/feature_engineering.py)  
**Component:** 11-Channel Spectral Index Extraction & Tensor Engineering  
**Presentation Mapping:** Slide 11 (Step 5), Slide 12 (Formulas Table), Slide 13 (Input Tensor)  

---

## 1. Module Overview & Purpose

`feature_engineering.py` transforms 3-channel preprocessed RGB reflectance images into high-dimensional 11-channel biophysical tensors:
$$\mathbf{X} \in \mathbb{R}^{11 \times H \times W}$$
By computing eight specialized visible-spectrum vegetation indices pixel-wise, the pipeline exposes chlorophyll absorption, vegetative fractional cover, chlorosis/senescence, and surface soil moisture directly to downstream deep neural networks without requiring expensive multispectral or thermal sensors.

---

## 2. Mathematical Formulations of the 8 Spectral Indices

Given linearized reflectances $R, G, B \in [0, 1]$ and numerical stabilizer $\epsilon = 10^{-6}$:

### 1. Normalized Green-Red Difference Index (`NGRDI`)
$$
\text{NGRDI} = \frac{G - R}{G + R + \epsilon}
$$
- **Range:** $[-1.0, 1.0]$
- **Role:** Primary metric for photosynthetic canopy greenness and vegetative vigor. Replaces satellite NDVI when infrared bands are unavailable.

### 2. Excess Green Index (`ExG`)
$$
\text{ExG} = \frac{2G - R - B}{R + G + B + \epsilon}
$$
- **Range:** $[-2.0, 2.0]$
- **Role:** Maximizes contrast between healthy green crops and background sandy/arid soils.

### 3. Excess Red Index (`ExR`)
$$
\text{ExR} = \frac{1.4R - G}{R + G + B + \epsilon}
$$
- **Range:** $[-1.0, 1.4]$
- **Role:** Sensitive to soil exposed by poor canopy closure and leaf yellowing caused by nitrogen deficiency or drought stress.

### 4. Visible Atmospherically Resistant Index (`VARI`)
$$
\text{VARI} = \frac{G - R}{G + R - B + \delta}
$$
- **Stabilization:** Denominator bounded away from zero ($|\text{denom}| \ge 10^{-3}$) and clamped to $[-2.0, 2.0]$.
- **Role:** Minimizes variations caused by changing sun angles, partial cloud shadows, and aerosol scatter.

### 5. Modified Green-Red Vegetation Index (`MGRVI`)
$$
\text{MGRVI} = \frac{G^2 - R^2}{G^2 + R^2 + \epsilon}
$$
- **Range:** $[-1.0, 1.0]$
- **Role:** Non-linear quadratic enhancement for dense canopy biomass estimation.

### 6. Normalized Pigment Chlorophyll Ratio (`NPCI`)
$$
\text{NPCI} = \frac{B - R}{B + R + \epsilon}
$$
- **Range:** $[-1.0, 1.0]$
- **Role:** Tracks carotenoid accumulation relative to chlorophyll, indicating plant senescence and severe dehydration.

### 7. Triangular Greenness Index (`TGI`)
$$
\text{TGI} = G - 0.39R - 0.61B
$$
- **Role:** Approximates the triangular spectral area formed by the 670 nm chlorophyll absorption trough and 550 nm green reflectance peak. Highly correlated with leaf chlorophyll content.

### 8. Excess Blue Index (`ExB`)
$$
\text{ExB} = 1.4B - G
$$
- **Role:** Detects surface moisture pooling, damp wetted soil surrounding SSPV vessels, and waterlogging.

---

## 3. Class Structure: `VegetationIndexExtractor`

### Core Methods
- `compute_indices(rgb: np.ndarray) -> Dict[str, np.ndarray]`: Takes an $(H, W, 3)$ float32 array in $[0, 1]$ and returns a dictionary of the 8 individual 2D index maps.
- `build_11ch_tensor(rgb: np.ndarray) -> Tuple[np.ndarray, Dict[str, np.ndarray]]`: Stacks the channels into an $(11, H, W)$ NumPy array and returns both the tensor and the dictionary.

---

## 4. Example Usage

```python
import numpy as np
from sspv_pipeline.feature_engineering import VegetationIndexExtractor

extractor = VegetationIndexExtractor()
rgb_sample = np.random.uniform(0.1, 0.8, (256, 256, 3)).astype(np.float32)

tensor_11ch, indices_dict = extractor.build_11ch_tensor(rgb_sample)

print("11-Channel Tensor shape:", tensor_11ch.shape)  # (11, 256, 256)
print("Mean NGRDI:", indices_dict["NGRDI"].mean())
print("Mean ExB (Moisture):", indices_dict["ExB"].mean())
```
