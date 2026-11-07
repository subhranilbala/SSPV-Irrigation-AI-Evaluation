# User Execution & Input/Output Specification Guide

**M.Tech Thesis Project (MTP) — 2026**  
**Candidate:** Subhranil Bala (Roll No.: `M25AI2038`)  
**Supervisor:** Dr. Anand Krishnan Plappally  
**Institute:** Indian Institute of Technology Jodhpur (School of AI & Data Science)  

---

## 📋 Table of Contents
1. [Prerequisites & System Requirements](#1-prerequisites--system-requirements)
2. [Step-by-Step Execution Guide](#2-step-by-step-execution-guide)
   - [Method 1: Interactive Streamlit Dashboard (GUI)](#method-1-interactive-streamlit-dashboard-gui)
   - [Method 2: Batch CLI Pipeline (Headless Execution)](#method-2-batch-cli-pipeline-headless-execution)
   - [Method 3: Programmatic Python API](#method-3-programmatic-python-api)
3. [Input Specifications & Formats](#3-input-specifications--formats)
   - [Input Type A: Multi-Date Drone Folders (DJI Zenmuse P1)](#input-type-a-multi-date-drone-folders-dji-zenmuse-p1)
   - [Input Type B: Single Image Upload](#input-type-b-single-image-upload)
   - [Input Type C: Automated 7-Date Field Simulator (Fallback)](#input-type-c-automated-7-date-field-simulator-fallback)
   - [CLI Arguments & Configuration Flags](#cli-arguments--configuration-flags)
4. [Output Specifications & Explanations](#4-output-specifications--explanations)
   - [Output 1: `sspv_7date_metrics.csv` (Tabular Evaluation Data)](#output-1-sspv_7date_metricscsv-tabular-evaluation-data)
   - [Output 2: `statistical_report.json` (Inferential Statistics)](#output-2-statistical_reportjson-inferential-statistics)
   - [Output 3: Trained PyTorch Model Weights (`.pth`)](#output-3-trained-pytorch-model-weights-pth)
   - [Output 4: Publication Figure `thesis_sspv_vs_control_summary.png`](#output-4-publication-figure-thesis_sspv_vs_control_summarypng)
   - [Output 5: Thesis Chapter Text Report `MTP_Thesis_Evaluation_Chapter_Report.txt`](#output-5-thesis-chapter-text-report-mtp_thesis_evaluation_chapter_reporttxt)
   - [Output 6: Real-Time Streamlit Visual Diagnostics](#output-6-real-time-streamlit-visual-diagnostics)
5. [Troubleshooting & Common Questions](#5-troubleshooting--common-questions)

---

## 1. Prerequisites & System Requirements

### Hardware:
- **macOS (Recommended):** Apple Silicon Mac (M1/M2/M3/M4, Pro, Max, Ultra) running macOS 13+. Automatically accelerated via Apple Silicon Metal Performance Shaders (`mps`).
- **Linux / Windows:** x86_64 CPU or NVIDIA GPU with CUDA 11.8+ installed.
- **RAM:** Minimum 8 GB (16 GB+ recommended when processing raw 8192×5460 DJI orthomosaics).

### Software Environment:
- Python 3.10, 3.11, 3.12, or 3.14.
- Pre-configured virtual environment located in `venv/`.
- Required dependencies installation:
  ```bash
  pip install -r requirements.txt
  # Or manually:
  pip install torch torchvision numpy scipy pandas matplotlib seaborn scikit-learn pillow streamlit rasterio opencv-python reportlab
  ```

---

## 2. Step-by-Step Execution Guide

### Initial Activation:
Open Terminal, navigate to the project directory, and activate the virtual environment:
```bash
cd /Users/subhranilbala/Downloads/SSPV-Irrigation-AI-Evaluation
source venv/bin/activate
```

---

### Method 1: Interactive Streamlit Dashboard (GUI)
The primary interface for demonstration, visual examination, and committee presentations.

#### Launch Command:
```bash
streamlit run main.py
```
*(Alternatively: `streamlit run sspv_pipeline/dashboard.py` or `python -m streamlit run main.py`)*

#### What Happens:
1. Streamlit opens a local browser tab (typically at `http://localhost:8501`).
2. The dashboard auto-detects if you have a `~/Downloads/Crops Data` directory on your machine.
3. If drone imagery is available, it co-registers multi-date flights and extracts non-overlapping patches.
4. If no path is supplied or the folder is restricted, it seamlessly boots on the **7-Date Realistic Field Simulator** modeling circular SSPV vessels and outer salt halos.
5. You can interactively train, select backbones (`ResNet18` or `EfficientNet-B0`), inspect any patch, visualize Grad-CAM attention, and click download buttons to save reports.

---

### Method 2: Batch CLI Pipeline (Headless Execution)
For automated background training, high-throughput batch evaluation, or server deployment.

#### Standard Run (ResNet18 backbone, 4 epochs):
```bash
python main.py --mode run_all --backbone resnet18 --epochs 4 --output_dir ./outputs
```

#### EfficientNet-B0 Run (with ImageNet Pretrained weights):
```bash
python main.py --mode run_all --backbone efficientnet_b0 --pretrained --epochs 5 --output_dir ./outputs_effnet
```

#### Processing Specific Drone Folder:
```bash
python main.py --data_dir "/Users/subhranilbala/Downloads/Crops Data" --max_images_per_date 25 --output_dir ./outputs_field
```

#### Console Progress Output:
```
================================================================================
IIT JODHPUR M.TECH THESIS PROJECT (MTP) - SSPV AI PIPELINE
Compute Device Selected : mps
Multi-Task CNN Backbone : resnet18
Max Frames Per Date     : 20
Input Data Directory    : Synthetic 7-Date Benchmark (Fallback)
================================================================================
[1/5] Loaded 42 multi-date UAV patches.
[2/5] Training 11-Channel U-Net & Multi-Task CNN for 4 epochs...
      Final U-Net Seg Acc: 88.42% | Multi-Task Stress Acc: 92.86%
[3/5] Running Statistical Comparison Engine (t-test, ANOVA, Cohen's d, Regression)...
{
  "t_test": { "t_statistic": 24.43, "p_value": 1.35e-25, "significant_at_005": true }, ...
}
[4/5] Verifying 11-Channel Grad-CAM Explainability...
      Grad-CAM Heatmap Generated Successfully | Shape: (128, 128)
[5/5] Exporting Thesis Artifacts to './outputs'...
      Saved: sspv_7date_metrics.csv, statistical_report.json, model weights, figures, & Thesis Chapter Report.
================================================================================
```

---

### Method 3: Programmatic Python API
Use directly in custom scripts or Jupyter Notebooks:
```python
from sspv_pipeline import SSPVPipelineOrchestrator, MultiTaskGradCAM

# Initialize orchestrator
orch = SSPVPipelineOrchestrator(
    data_dir=None,            # None activates realistic 7-date benchmark
    pretrained=False,
    backbone_name="resnet18"
)

# 1. Train models
history = orch.train_models(epochs=4, batch_size=8)

# 2. Evaluate all dates
df_metrics, stats = orch.evaluate_all_dates()

# 3. Export all thesis files
orch.save_thesis_artifacts("./outputs", df_metrics, stats, history)
```

---

## 3. Input Specifications & Formats

The pipeline accommodates three flexible input modalities:

### Input Type A: Multi-Date Drone Folders (DJI Zenmuse P1)
For real field surveys conducted over the crop growth cycle.

#### Expected Directory Layout:
```
<Crops Data>/
├── 12-11-2024/
│   └── images/
│       ├── DJI_0001.JPG
│       ├── DJI_0002.JPG
│       └── ...
├── 22-11-2024/
│   └── images/
├── 02-12-2024/
│   └── images/
├── 12-12-2024/
├── 25-12-2024/
├── 05-01-2025/
└── 15-01-2025/
```
- **Folder Naming:** Date strings in `%d-%m-%Y` format (`DD-MM-YYYY`). The pipeline automatically sorts them chronologically and calculates elapsed day numbers ($t = 1, 11, 21, \dots$).
- **File Types:** High-resolution RGB JPEGs (`.JPG`, `.jpg`, `.jpeg`) or Georeferenced GeoTIFFs (`.tif`, `.tiff`).
- **Resolution:** Supports native DJI Zenmuse P1 full-frame sensor output ($8192 \times 5460$ pixels).
- **Subfolder Search:** The pipeline automatically looks inside `<date>/images/` first, and recurses into subdirectories if needed. It also automatically ignores hidden macOS resource fork files (`._DJI_...`).
- **Patch Extraction Strategy:** Upper-Right subregions are mapped to SSPV soil mounds; Lower-Right subregions are mapped to conventional control rows.

---

### Input Type B: Single Image Upload
In the Streamlit Dashboard sidebar, you can upload any single `.JPG` or `.TIF` file:
- Allows instantaneous on-the-fly testing of a single drone capture.
- Runs sRGB linearization, DOS correction, 8-index extraction, U-Net crop canopy segmentation, and Grad-CAM spatial heatmaps in under 1 second.

---

### Input Type C: Automated 7-Date Field Simulator (Fallback)
If no local folder path is specified, the pipeline activates an internal physical simulator:
- Synthesizes 42 multi-date agricultural plot patches across 7 dates matching your exact schedule (`12-11-2024` through `15-01-2025`).
- Models sandy Rajasthan soil background ($\mu = [0.46, 0.38, 0.27]$).
- Models circular SSPV porous clay vessels with wetted capillary rings, expanding vegetation vigor over time, and outer white salt crust halos (Slide 7).
- Models conventional row crops with higher chlorosis, spatial patchiness, and irrigation dry-down shocks.

---

### CLI Arguments & Configuration Flags

| Parameter | Type | Default | Choices / Format | Description |
|---|---|---|---|---|
| `--mode` | str | `run_all` | `run_all`, `verify` | Pipeline execution mode |
| `--data_dir` | str | `~/Downloads/Crops Data` | Valid path or `None` | Path to parent folder containing multi-date subfolders |
| `--backbone` | str | `resnet18` | `resnet18`, `efficientnet_b0` | Shared backbone architecture for Multi-Task CNN |
| `--max_images_per_date`| int | `20` | `0` to `200` | Frames per date to sample uniformly across flight. Set to `0` to load ALL images |
| `--epochs` | int | `4` | $1$ to $50$ | Number of epochs for U-Net & CNN training |
| `--batch_size` | int | `8` | $2, 4, 8, 16$ | Mini-batch size for DataLoader |
| `--pretrained` | flag | `False` | Present / Absent | Ingests ImageNet pretrained weights adapted to 11 channels |
| `--output_dir` | str | `./outputs` | Valid directory path | Output directory for all exported weights and reports |

---

## 4. Output Specifications & Explanations

All outputs are saved to the designated `--output_dir` (default: `./outputs`).

```
outputs/
├── sspv_7date_metrics.csv
├── statistical_report.json
├── unet_11ch_weights.pth
├── multitask_cnn_11ch_weights.pth
├── thesis_sspv_vs_control_summary.png
└── MTP_Thesis_Evaluation_Chapter_Report.txt
```

---

### Output 1: `sspv_7date_metrics.csv` (Tabular Evaluation Data)
Contains fine-grained biophysical metrics for every evaluated patch across all dates.

| Column Header | Data Type | Physical Meaning & Agronomic Interpretation |
|---|---|---|
| `sample_id` | Integer | Unique identifier index of the evaluated patch |
| `date` | String | Survey flight date string (`DD-MM-YYYY`) |
| `day_num` | Integer | Elapsed days since the initial flight (`12-11-2024` = Day 1) |
| `plot_type` | Categorical | Irrigation treatment: `"SSPV"` or `"Control"` |
| `mean_ngrdi` | Float | Mean Normalized Green-Red Difference Index across valid crop pixels. Higher positive values indicate higher photosynthetic vigor |
| `mean_exg` | Float | Mean Excess Green Index. Quantifies green canopy biomass separated from bare soil |
| `mean_vari` | Float | Mean Visible Atmospherically Resistant Index. Reflectance greenness normalized against atmospheric haze |
| `mean_tgi` | Float | Mean Triangular Greenness Index. Direct proxy for leaf chlorophyll concentration |
| `mean_exb` | Float | Mean Excess Blue Index. Indicates soil surface wetness and water pooling |
| `moisture_zone_variance`| Float | Spatial variance of moisture index within the plot. SSPV plots exhibit lower variance due to uniform capillary action |
| `stress_pixel_ratio` | Float | Proportion of canopy pixels classified as stressed (Class 2) by U-Net ($[0.0, 1.0]$) |
| `salt_patch_ratio` | Float | Proportion of plot area exhibiting white salt crust halos (Class 3) pushed outward by SSPV percolation |
| `pred_stress` | Categorical | Multi-Task CNN output: `"Healthy"`, `"Mild Stress"`, or `"Severe Stress"` |
| `pred_moisture` | Categorical | Multi-Task CNN output: `"Dry"`, `"Optimal"`, or `"Wet"` |
| `pred_yield` | Categorical | Multi-Task CNN output: `"Low"`, `"Medium"`, or `"High"` |
| `recommendation` | String | Actionable guidance generated by the hybrid decision engine |
| `rec_confidence` | Float | Confidence score of the top-ranked recommendation ($[0.0, 1.0]$) |
| `rec_mode` | String | Mode that fired: `"Rule-Based Logic"` (determinate) or `"Decision Tree Classifier"` (indeterminate) |
| `ranked_top2` | String | Top-2 ranked candidate recommendations with respective confidence values |

---

### Output 2: `statistical_report.json` (Inferential Statistics)
Structured JSON containing all hypothesis testing calculations:

```json
{
  "statistical_comparison": {
    "t_test": {
      "t_statistic": 24.4332,
      "p_value": 1.35e-25,
      "significant_at_005": true,
      "sspv_mean_ngrdi": 0.3611,
      "control_mean_ngrdi": -0.0623
    },
    "anova": {
      "f_statistic": 71.0217,
      "p_value": 2.10e-10,
      "significant_at_005": true,
      "f_statistic_across_dates": 0.4973,
      "p_value_across_dates": 0.8059
    },
    "effect_size": {
      "cohens_d_ngrdi": 7.54,
      "interpretation_ngrdi": "Large Effect (High Practical Significance)",
      "cohens_d_stress_reduction": -2.60,
      "cohens_d_moisture_zone_variance": -4.99,
      "interpretation_moisture_variance": "Large Effect (High Practical Significance)"
    },
    "temporal_regression": {
      "sspv_slope_per_day": 0.00254,
      "sspv_r_value": 0.973,
      "sspv_p_value": 1.47e-13,
      "control_slope_per_day": 0.00195,
      "control_r_value": 0.794,
      "control_p_value": 1.76e-05,
      "inter_plot_slope_difference": 0.00059,
      "inter_plot_slope_z_stat": 1.601,
      "inter_plot_slope_p_value": 0.1094,
      "sspv_exg_temporal_std": 0.0802,
      "control_exg_temporal_std": 0.0712
    }
  },
  "training_history": {
    "unet_loss": [0.852, 0.431, ...],
    "cnn_loss": [1.421, 0.764, ...],
    "seg_acc": [0.65, 0.88, ...],
    "stress_acc": [0.55, 0.93, ...]
  },
  "total_patches_evaluated": 42,
  "dates_evaluated": ["12-11-2024", "22-11-2024", "02-12-2024", ...]
}
```

---

### Output 3: Trained PyTorch Model Weights (`.pth`)
Binary state dictionaries of trained models saved after the specified epochs:
- **`unet_11ch_weights.pth` (~57 MB):** Trained weights for the 11-channel U-Net with ResNet18 encoder. Can be reloaded with:
  ```python
  model = UNetResNet18_11Ch(in_channels=11, num_classes=4)
  model.load_state_dict(torch.load("outputs/unet_11ch_weights.pth"))
  ```
- **`multitask_cnn_11ch_weights.pth` (~45 MB):** Trained weights for the 11-channel Multi-Task CNN. Can be reloaded with:
  ```python
  model = MultiTaskSSPVNet(in_channels=11, backbone_name="resnet18")
  model.load_state_dict(torch.load("outputs/multitask_cnn_11ch_weights.pth"))
  ```

---

### Output 4: Publication Figure `thesis_sspv_vs_control_summary.png`
A high-resolution ($200\text{ DPI}, 14\times10\text{ inches}$) 4-panel graphic designed for inclusion in the M.Tech thesis chapter and conference manuscripts:

```
+---------------------------------------+---------------------------------------+
| Panel (0,0):                          | Panel (0,1):                          |
| Temporal NGRDI Growth Trend           | Crop Stress Pixel Proportion          |
| (SSPV vs. Control OLS lines)          | (One-Way ANOVA boxplot across plots)  |
+---------------------------------------+---------------------------------------+
| Panel (1,0):                          | Panel (1,1):                          |
| Temporal ExG Stability                | Temporal VARI Distribution            |
| (Shows stable SSPV capillary balance) | (Atmospherically invariant greenness) |
+---------------------------------------+---------------------------------------+
```

---

### Output 5a: Formal Thesis Chapter PDF Report `MTP_Thesis_Evaluation_Chapter_Report.pdf`
A publication-grade, multi-page formal academic evaluation chapter compiled with ReportLab:
- **Formal Headers:** Official *Indian Institute of Technology Jodhpur — School of Artificial Intelligence & Data Science* banner.
- **Candidate & Supervisor Attribution:** Subhranil Bala (`M25AI2038`) & Dr. Anand Krishnan Plappally.
- **Hypothesis Testing Tables:** Formal tables detailing Welch's t-test ($t, p$), One-Way ANOVA across plots and survey dates, Cohen's $d$ effect sizes, and temporal regression slopes.
- **RuTAG Human Ecology & Hydrology Audit:** Capillary stability ratios (Gangotri steady percolation), root-zone salt protection status, and estimated groundwater conserved (~38.5%).
- **Embedded Publication Graphic:** Vector-embedded 4-panel multi-date temporal trend graphic.
- **Actionable Decision Distribution:** Complete ranked recommendation percentages.
- **Formal Academic Signatures:** Endorsement lines for candidate and thesis supervisor.

---

### Output 5b: Thesis Chapter Text Report `MTP_Thesis_Evaluation_Chapter_Report.txt`
A pre-formatted, publication-ready academic report in plain text:
- Contains IIT Jodhpur header, roll number, project title, and supervisor attribution.
- Section 1: Executive Summary & Statistical Comparison (t-test stats, ANOVA $F$, Cohen's $d$).
- Section 2: Temporal Growth Dynamics & Slope Z-test.
- Section 3: Recommendation Distribution breakdown (% of patches needing refill, optimal, etc.).
- Section 4: RuTAG Human Ecology & Traditional Hydrology Audit.
- Can be directly copy-pasted into LaTeX, Microsoft Word, or Markdown thesis draft chapters.

---

### Output 6: Real-Time Streamlit Visual Diagnostics
When viewing via the dashboard:
1. **Interactive Top KPI Cards:** Displays SSPV NGRDI, Welch t-test p-value, Cohen's d, and ExG temporal std.
2. **Synchronized 4-Panel Spatial Inspector:** Shows side-by-side RGB, NGRDI greenness heatmap, 4-class U-Net segmentation mask (including mound salt halos), and Grad-CAM attention overlay.
3. **Temporal Trend Figures:** Dynamic graphs showing growth divergence between SSPV and Control.
4. **Actionable Table & CSV Download Buttons:** Instant file download with one click.

---

## 5. Troubleshooting & Common Questions

#### Q1: "Operation not permitted" or permission error accessing `~/Downloads/Crops Data`?
- **Cause:** When running inside certain macOS sandboxed terminals or containers, directories outside the workspace require explicit permissions.
- **Solution:** The pipeline automatically handles this gracefully by printing a notice and switching to the **7-Date Realistic Field Benchmark Simulator**, which replicates the exact field dates and geometry. To grant full access, run the command in your native macOS Terminal (`Terminal.app` or `iTerm2`).

#### Q2: Can I run this without an NVIDIA GPU?
- **Yes!** The pipeline has native auto-detection for Apple Silicon Macs via Metal Performance Shaders (`mps`). On Windows/Linux systems without a GPU, it automatically falls back to multi-threaded CPU execution.

#### Q3: How do I change the number of images loaded per date?
- Use `--max_images_per_date <number>`. For instance, `--max_images_per_date 15` samples 15 uniformly spaced frames along the drone flight path. Setting `--max_images_per_date 0` loads all raw images.
