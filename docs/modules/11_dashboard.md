# Module Documentation: `dashboard.py`

**File Path:** [`sspv_pipeline/dashboard.py`](file:///Users/subhranilbala/Downloads/SSPV-Irrigation-AI-Evaluation/sspv_pipeline/dashboard.py)  
**Component:** Interactive Streamlit Visual Analytics Dashboard  
**Presentation Mapping:** Slide 10 (Decision Support), Slide 14 (Dashboard), Slide 17 (Phase 4 Deployment)  

---

## 1. Module Overview & Purpose

`dashboard.py` provides the web-based interactive graphical interface for the project, enabling agronomists, thesis committee members, and farm operators to inspect spatial predictions, explore multi-date temporal health trends, analyze Grad-CAM visual attention, and download formal thesis artifacts without touching code.

Launched simply via:
```bash
streamlit run main.py
```

---

## 2. Dashboard Layout & Visual Components

### 2.1 Sidebar Controls
- **Data Input Configuration:**
  - Local path to the `Crops Data` directory (auto-detects `~/Downloads/Crops Data` if present).
  - Maximum DJI frames per date slider (default **`5`** for instant 5s startup; 15–30 for deeper flight sampling, 0 = all frames).
  - Single image drag-and-drop file uploader (`.JPG` / `.TIF`).
- **Agro-Ecological Region & Soil Selector:**
  - Dropdown choosing regional presets (`jodhpur_thar` Western Rajasthan Aridisol, `nagaur_semiarid`, `south_india_laterite`, `indo_gangetic`).
- **Model Controls:**
  - Multi-Task CNN Backbone selector (`ResNet18` vs. `EfficientNet-B0`).
  - Training Epochs slider (1–25).
  - ImageNet pretraining toggle.
  - Grad-CAM target head dropdown (`stress_logits`, `moisture_logits`, `yield_logits`).
  - **"⚡ Force Re-train Models"** button for on-demand training without UI freezes.
- **Patch Inspector Slider:** Step through and visually inspect any individual dataset patch.

---

### 2.2 Top KPI Metrics Ribbon
A 4-column summary metric card row displaying:
1. **SSPV Mean NGRDI:** Shows mean value and delta comparison vs. Control.
2. **Welch's t-test p-value:** Formatted in scientific notation with a badge indicating statistical significance ($p < 0.05$).
3. **Cohen's $d$ Effect Size:** Shows numerical effect size alongside its qualitative interpretation (e.g., *Large Effect*).
4. **ExG Temporal Variance:** Directly compares SSPV vs. Control temporal standard deviations to verify the Slide 16 hypothesis.

---

### 2.3 Spatial Diagnostics Visualizer (4-Panel Grid)
Displays four synchronized spatial maps for any selected patch or live uploaded frame:
1. **Preprocessed RGB:** Calibrated visible reflectance image.
2. **NGRDI Canopy Greenness Heatmap:** Color-mapped using `RdYlGn` (red = bare/stressed soil, dark green = dense canopy).
3. **U-Net 4-Class Mask:** Color-mapped using `viridis`:
   - $0$: Bare Soil
   - $1$: Healthy Crop Canopy
   - $2$: Stressed Crop Zone
   - $3$: SSPV Mound Salinity White Salt Crust (Slide 7)
4. **Multi-Task Grad-CAM Attention:** Jet colormap overlaid with $45\%$ transparency on the RGB image to reveal the model's spatial reasoning.

---

### 2.4 Multi-Date Temporal Growth & Statistical Comparison Charts
Rendered across two responsive two-column figures with automatic canvas memory cleanup (`plt.close`):

#### Figure 1:
- **Left:** Temporal NGRDI growth trend line plot (SSPV vs. Control with circular markers and confidence bands across elapsed survey days).
- **Right:** Crop stress pixel proportion boxplot across irrigation systems for ANOVA comparison.

#### Figure 2:
- **Left:** Temporal ExG stability line plot (SSPV vs. Control with square markers) confirming stable demand-driven capillary delivery.
- **Right:** Temporal VARI distribution line plot (SSPV vs. Control with diamond markers) demonstrating resistance to seasonal atmospheric variations.

---

### 2.5 Actionable Recommendations & Artifact Download Buttons
1. **Interactive Table:** Filterable, searchable table displaying date, plot type, spectral indices, predicted stress/moisture/yield states, primary recommendation, confidence score, and top-2 ranked actions.
2. **Download Button 1:** One-click CSV export of the complete 7-date metrics table (`sspv_7date_metrics.csv`).
3. **Download Button 2:** One-click download of the complete formal thesis evaluation chapter report PDF (`MTP_Thesis_Evaluation_Chapter_Report.pdf`) generated with ReportLab.
4. **Download Button 3:** One-click download of the thesis chapter text draft (`MTP_Thesis_Evaluation_Chapter_Report.txt`).

---

### 2.6 RuTAG Human Ecology & Indigenous Hydrology Audit (Section 4)
Dedicated evaluation panel demonstrating:
- **Capillary Stability Ratio:** Quantifies steady unpressurized matric buffer (Gangotri percolation principle).
- **Root-Zone Saline Protection:** Validates that porous pots displace salts outward to perimeter furrows (Parashurama soil reclamation).
- **Estimated Groundwater Saved:** Computes percentage and daily volumetric water savings vs. flood/furrow irrigation (~38.5%).
- **Bilingual Smallholder Field Advisory:** Real-time actionable field checklist in English and Hindi (Devanagari) for the inspected plot.

---

## 3. Example Usage

```python
# Launch from terminal:
# streamlit run main.py

# Or invoke programmatically:
from sspv_pipeline.dashboard import run_streamlit_dashboard

run_streamlit_dashboard()
```
