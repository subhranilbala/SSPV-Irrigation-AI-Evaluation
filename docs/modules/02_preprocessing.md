# Module Documentation: `preprocessing.py`

**File Path:** [`sspv_pipeline/preprocessing.py`](file:///Users/subhranilbala/Downloads/SSPV-Irrigation-AI-Evaluation/sspv_pipeline/preprocessing.py)  
**Component:** Photogrammetric & Atmospheric UAV Preprocessing Pipeline  
**Presentation Mapping:** Slide 10 (Tech Stack) & Slide 11 (Preprocessing Steps 1 to 4)  

---

## 1. Module Overview & Purpose

`preprocessing.py` implements the 4-step drone imagery calibration pipeline required to convert raw Digital Numbers (DN) captured by UAV cameras (e.g., DJI Zenmuse P1 at 8192×5460 resolution) into physically meaningful, multi-date aligned, noise-filtered surface reflectance maps.

---

## 2. Theoretical Background & Mathematical Formulations

### Step 1: DN → Reflectance Conversion (`dn_to_reflectance`)
Consumer and industrial UAV RGB cameras save imagery in sRGB color space governed by a non-linear gamma transfer function ($\gamma \approx 2.4$). Calculating vegetation indices directly on non-linear sRGB values leads to severe distortions in greenness estimation.

The function linearizes the input normalized DN:
$$
\rho_{\text{linear}} = 
\begin{cases} 
\frac{\rho}{12.92}, & \rho \le 0.04045 \\
\left(\frac{\rho + 0.055}{1.055}\right)^{2.4}, & \rho > 0.04045 
\end{cases}
$$
where $\rho = \frac{\text{DN}}{255.0}$ (for 8-bit) or $\frac{\text{DN}}{65535.0}$ (for 16-bit GeoTIFFs). If calibration panel measurements are supplied, empirical radiometric gains are multiplied per band.

### Step 2: Dark Object Subtraction (`dark_object_subtraction`)
Low-altitude UAV flights (50–120 m AGL) suffer from Rayleigh and aerosol atmospheric path radiance ($L_p$), causing deep shadows and dark surfaces to appear artificially bright:
$$
\rho_{\text{corrected}}(\lambda) = \frac{\max\left(0, \rho_{\text{linear}}(\lambda) - \text{Percentile}_{1.0}(\rho_{\text{linear}}(\lambda))\right)}{\text{Percentile}_{99.9}(\rho_{\text{linear}}(\lambda)) + \epsilon}
$$
By subtracting the 1st percentile non-zero pixel value per channel and normalizing by the 99.9th percentile, additive atmospheric haze is removed.

### Step 3: Multi-Date SIFT Affine Co-Registration (`geometric_alignment`)
Due to slight GPS drift between drone survey flights across different dates, image boundaries must be co-registered:
1. Grayscale conversion of target and reference orthomosaic patches.
2. SIFT feature detection (`nfeatures=2500`) and descriptor extraction.
3. Brute-Force $L_2$ k-nearest neighbors matching with Lowe's ratio test:
   $$\|d_{\text{best}}\| < 0.75 \|d_{\text{second\_best}}\|$$
4. RANSAC robust partial affine estimation (`cv2.estimateAffinePartial2D`) to filter outliers and compute translation, rotation, and uniform scale.
5. Bilinear warp affine resampling with reflection padding.

### Step 4: ROI Boundary Isolation & Median Filtering (`clip_and_filter`)
Isolates plot boundary bounding boxes $(y_1, y_2, x_1, x_2)$ and applies a $3\times3$ spatial median filter:
$$
I_{\text{clean}}(x, y) = \text{median}\left\{ I(x+i, y+j) \mid -1 \le i, j \le 1 \right\}
$$
This suppresses salt-and-pepper high-frequency sensor noise while preserving sharp boundaries between crop leaves and sandy soil.

---

## 3. Class Structure: `UAVPreprocessor`

### Constructor
```python
UAVPreprocessor(dos_percentile: float = 1.0, median_kernel: int = 3)
```

### Core Methods
- `load_image(image_path: str, target_size: Optional[Tuple[int, int]] = None) -> Tuple[np.ndarray, Dict[str, Any]]`: Handles GeoTIFFs (via `rasterio`) and `.JPG` / `.png` (via `cv2` or `PIL`). If `target_size` is provided (e.g. `(1024, 682)`), hardware-accelerated downscaled decoding (`IMREAD_REDUCED_COLOR_8` / `PIL.draft`) is used for 5–10× faster load times without large RAM allocations.
- `dn_to_reflectance(img_dn, panel_factor=None) -> np.ndarray`: Implements sRGB inverse gamma linearization.
- `dark_object_subtraction(reflectance) -> np.ndarray`: Implements percentile-based DOS.
- `geometric_alignment(target_img, reference_img) -> np.ndarray`: Implements SIFT + RANSAC affine warping.
- `clip_and_filter(img, roi_bbox=None) -> np.ndarray`: Implements bounding box clipping and median filtering.
- `process(img_dn, reference_img=None, roi_bbox=None) -> np.ndarray`: Executes the full 4-step pipeline end-to-end.

---

## 4. Photogrammetry, Telemetry & Real-Time Boundary Detection Functions

In addition to `UAVPreprocessor`, `preprocessing.py` provides photogrammetric transformation functions to project ground GPS coordinates and detect field boundaries:

- `extract_exif_gps(img_path: str) -> Optional[Tuple[float, float, float]]`: Extracts `(latitude, longitude, altitude)` from DJI Zenmuse P1 EXIF headers without decoding full pixel buffers.
- `compute_gps_distance(coord1, coord2) -> float`: Calculates ground distance in meters between two GPS coordinates using spherical earth approximation.
- `extract_dji_flight_telemetry(img_path: str) -> Optional[Dict[str, float]]`: Extracts RTK GNSS and gimbal orientation metadata (`lat`, `lon`, `alt`, `yaw`, `pitch`, `roll`, `rel_alt`).
- `project_geo_polygon_to_image(geo_poly, telem, out_w=1024, out_h=682, specs=None) -> np.ndarray`: Photogrammetric projection mapping ground coordinates (WGS84 Lat/Lon) to pixel coordinates $(x, y)$ on the camera sensor, incorporating flight altitude, camera GSD, heading yaw rotation, and gimbal pitch.
- `compute_polygon_visibility(poly_pts, out_w=1024, out_h=682) -> float`: Calculates the proportion of a projected polygon $[0.0, 1.0]$ visible inside the image frame boundaries.
- `detect_realtime_plot_boundaries(img, img_path=None, telemetry=None) -> Dict[str, Any]`: Dynamically identifies the 3 experimental boundaries (Study Area, SSPV Grid, Control Plot) using a 3-tier hierarchy:
  1. *Mode 1 (RTK Photogrammetric Projection):* Uses drone telemetry and known WGS84 polygon vertices.
  2. *Mode 2 (Computer Vision Clustering):* Uses Difference-of-Gaussians (DoG) + contour circularity to detect earthen pitcher mounds and construct adjacent check basins.
  3. *Mode 3 (Calibrated Static Fallback):* Normalized bounding boxes from `GROUND_TRUTH_ROIS`.
- `draw_ground_truth_boundaries(img, rois=None, img_path=None, alpha=0.18, show_study_area=True, show_sspv=True, show_control=True) -> np.ndarray`: Renders crisp polygon outlines (Red: Study Area, Blue: SSPV, Green: Control) with translucent fills and anti-aliased labels onto the frame.

---

## 5. Example Usage

```python
from sspv_pipeline.preprocessing import UAVPreprocessor

preprocessor = UAVPreprocessor(dos_percentile=1.0, median_kernel=3)
raw_img, meta = preprocessor.load_image("DJI_0042.JPG")
clean_reflectance = preprocessor.process(raw_img)

print("Preprocessed reflectance shape:", clean_reflectance.shape)
print("Value range:", clean_reflectance.min(), clean_reflectance.max())
```
