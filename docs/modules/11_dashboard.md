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
  - Maximum aerial image frames per date slider (default **`5`** for instant 5s startup; 15–30 for deeper flight sampling, 0 = all frames).
  - Single image drag-and-drop file uploader (`.JPG` / `.TIF`).
- **Agro-Ecological Region & Soil Selector:**
  - Dropdown choosing regional presets (`dehradun_valley` Doon Valley Foothill Zone [Default], `jodhpur_thar` Western Rajasthan Aridisol, `nagaur_semiarid`, `south_india_laterite`, `indo_gangetic`).
- **Model Controls:**
  - Multi-Task CNN Backbone selector (`ResNet18` vs. `EfficientNet-B0`).
  - Training Epochs slider (1–25).
  - ImageNet pretraining toggle.
  - Grad-CAM target head dropdown (`stress_logits`, `moisture_logits`, `yield_logits`).
  - **"⚡ Force Re-train Models"** button for on-demand training without UI freezes.
- **Target Inspection Selectors:**
  - **Survey Flight Date:** Filters images by flight acquisition date.
  - **Image Frame & Plot:** Direct dropdown choosing exact aerial frame and plot (`[SSPV]` vs. `[Control]`).
  - **Sequential Patch ID Slider:** Quickly scrub across loaded patches sequentially with real-time target feedback card.
- **Persistent Wide Ribbon:** Styled with `initial_sidebar_state="expanded"` and CSS stretched width ($480\text{px} - 500\text{px}$) for comfortable, wrap-free viewing of file paths, long dropdown titles (e.g. *Select Image Frame & Plot*, *Agro-Climatic Zone*), and metadata.

---

### 2.2 Top KPI Metrics Ribbon
A 5-card responsive summary metric row displaying:
1. **SSPV Mean NGRDI:** Shows mean value and delta comparison vs. Control.
2. **Welch's t-test p-value:** Formatted in scientific notation with a badge indicating statistical significance ($p < 0.05$).
3. **Cohen's $d$ Effect Size:** Shows numerical effect size alongside its qualitative interpretation (e.g., *Large Effect*).
4. **ExG Temporal Variance:** Directly compares SSPV vs. Control temporal standard deviations to verify the Slide 16 hypothesis.
5. **Water Conserved:** Percentage groundwater conservation achieved vs. surface flood baseline (~38.5%).

---

### 2.3 Section 1: 🗺️ Experimental Field Segmentation Ground Truth (Active Flight Frame)
Visualizes dynamic 3-zone ground truth detection on the active selected drone flight frame:
- 🟥 **Red Boundary:** Overall Study Area (SSPV + Control) enclosing the experimental plots (~4,000 m²).
- 🟦 **Blue Boundary:** SSPV Treatment Zone — dynamically bounding the 25 earthen pitcher mounds (5×5 grid).
- 🟩 **Green Boundary:** Control Plot — dynamically outlining the 8 partitioned rectangular border check flood basins.
- **Interactive Boundary Modes:** Switch between composite 3-zone view, SSPV Pitcher Grid Only, Control Flood Basins Only, or Raw Unannotated Drone Photo.

---

### 2.4 Section 2: 🛰️ Spatial Inspection (11-Channel U-Net & Multi-Task Grad-CAM)
Displays four synchronized spatial maps for any selected patch or live uploaded frame:
1. **Preprocessed RGB:** Calibrated visible reflectance image with clean title (`Preprocessed RGB (True Color Reflectance)`), displaying target frame filename, plot type, and flight date directly beneath the image.
2. **NGRDI Canopy Greenness Heatmap:** Color-mapped using `RdYlGn` (red = bare/stressed soil, dark green = dense canopy).
3. **U-Net 4-Class Mask:** Color-mapped using `viridis`:
   - $0$: Bare Soil
   - $1$: Healthy Crop Canopy
   - $2$: Stressed Crop Zone
   - $3$: SSPV Mound Salinity White Salt Crust (Slide 7)
4. **Multi-Task Grad-CAM Attention:** Jet colormap overlaid with $45\%$ transparency on the RGB image to reveal the model's spatial reasoning.

---

### 2.5 Multi-Date Temporal Growth & Statistical Comparison Charts

#### Section 3a: Multi-Date Temporal Growth Trend & SSPV vs. Control Statistical Comparison
- **Left:** Temporal NGRDI growth trend line plot (SSPV vs. Control with circular markers and confidence bands across elapsed survey days).
- **Right:** Crop stress pixel proportion boxplot across irrigation systems for ANOVA comparison.

#### Section 3b: ExG & VARI Temporal Stability — SSPV vs. Control
- **Left:** Temporal ExG stability line plot (SSPV vs. Control with square markers) confirming stable demand-driven capillary delivery.
- **Right:** Temporal VARI distribution line plot (SSPV vs. Control with diamond markers) demonstrating resistance to seasonal atmospheric variations.

---

### 2.6 Section 4: 📋 Actionable Irrigation Scheduling Recommendations
1. **Interactive Table:** Filterable table displaying survey date, aerial image filename, plot type, spectral indices, predicted stress/moisture/yield states, primary recommendation, confidence score, and top-2 ranked actions.
2. **Download Button 1:** One-click CSV export of the complete metrics table (`sspv_7date_metrics.csv`).
3. **Download Button 2:** One-click download of the complete formal thesis evaluation chapter report PDF (`MTP_Thesis_Evaluation_Chapter_Report.pdf`) generated with ReportLab.
4. **Download Button 3:** One-click download of the thesis chapter text draft (`MTP_Thesis_Evaluation_Chapter_Report.txt`).

---

### 2.7 Section 5: 🌿 RuTAG Human Ecology & Indigenous Hydrology Audit
Dedicated evaluation panel demonstrating:
- **Capillary Stability Ratio:** Quantifies steady unpressurized matric buffer.
- **Root-Zone Saline Protection:** Validates that porous pots displace salts outward to perimeter furrows.
- **Estimated Groundwater Saved:** Computes percentage and daily volumetric water savings vs. flood irrigation (~38.5%).

---

### 2.8 Section 6: 📢 Smallholder Field Advisory (Bilingual Voice & Text)
Interactive grassroots advisory module:
- **Bilingual Action Cards:** Real-time actionable field checklist in English (Agronomist) and Hindi Devanagari (Kisaan Mitra).
- **Native Audio Playback:** Text-to-speech audio widgets (`st.audio`) delivering spoken guidance in Indian English (`Rishi`) and Hindi (`Lekha`).
- **Water Savings Telemetry:** Live liters/day conservation counter per individual vessel plot.

---

## 3. Helper Functions

- `init_and_train_pipeline(folder, ep, pre, bbone, max_im, force_retrain=False)`: Caches model initialization, loads weights if available to eliminate UI freezing, and computes multi-date metrics.
- `_get_stretch_kwarg(target_func)`: Version-compatible helper returning modern `width="stretch"` (falling back to `use_container_width=True` on legacy Streamlit versions) to eliminate deprecation warnings.
- `_generate_speech_wav(text, lang="en") -> bytes`: Native macOS text-to-speech synthesizer generating 16-bit PCM WAV audio bytes playable directly in browser HTML5 audio via `st.audio`.

---

## 3. Example Usage

```python
# Launch from terminal:
# streamlit run main.py

# Or invoke programmatically:
from sspv_pipeline.dashboard import run_streamlit_dashboard

run_streamlit_dashboard()
```
