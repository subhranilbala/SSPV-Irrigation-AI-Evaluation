# Module Documentation: `orchestrator.py`

**File Path:** [`sspv_pipeline/orchestrator.py`](file:///Users/subhranilbala/Downloads/SSPV-Irrigation-AI-Evaluation/sspv_pipeline/orchestrator.py)  
**Component:** End-to-End Pipeline Training, Evaluation & Thesis Artifact Exporter  
**Presentation Mapping:** Slide 10 (System Architecture), Slide 15 (Pipeline Flowchart), Slide 17 (Phase 1–4 Execution)  

---

## 1. Module Overview & Purpose

`orchestrator.py` is the central coordinator of the entire system. The [`SSPVPipelineOrchestrator`](file:///Users/subhranilbala/Downloads/SSPV-Irrigation-AI-Evaluation/sspv_pipeline/orchestrator.py#L29) class wires together the data ingestion layer, the U-Net and Multi-Task CNN deep learning models, the inferential statistical engine, the recommendation system, and the thesis file exporter.

---

## 2. Core Responsibilities & Workflow

### 2.1 Dynamic Inverse-Frequency Class Balancing (`_compute_inverse_class_weights`)
To prevent majority-class domination during training (e.g., more optimal or healthy patches than severe waterlogging):
$$
w_c = \frac{1 / N_c}{\sum_{j=1}^3 1 / N_j} \times 3.0
$$
Computes separate weight vectors for stress, moisture, and yield heads and passes them to [`MultiTaskLoss`](file:///Users/subhranilbala/Downloads/SSPV-Irrigation-AI-Evaluation/sspv_pipeline/models.py#L243).

### 2.2 Dual Model Training Loop (`train_models`)
In each training epoch:
1. Feeds the 11-channel tensor $\mathbf{X}$ to U-Net to compute segmentation cross-entropy loss against pseudo-labels.
2. Derives binary crop gating mask:
   $$\mathbf{M}_{\text{crop}} = (\hat{y}_{\text{unet}} == 1) \lor (\hat{y}_{\text{unet}} == 2)$$
3. Feeds $\mathbf{X}$ and $\mathbf{M}_{\text{crop}}$ into Multi-Task CNN to predict stress, moisture, and yield.
4. Backpropagates both `opt_unet` and `opt_cnn` simultaneously using `AdamW` ($\text{lr} = 10^{-3}, \text{weight\_decay} = 10^{-4}$).
5. Returns training history metrics (`unet_loss`, `cnn_loss`, `seg_acc`, `stress_acc`).

### 2.3 Comprehensive Multi-Date Inference (`evaluate_all_dates`)
Evaluates all patches across all survey dates in `eval` mode without gradients:
- Applies U-Net crop gating to Multi-Task CNN.
- Extracts mean plot indices (`NGRDI`, `ExG`, `VARI`, `TGI`, `ExB`).
- Gathers stress and salt patch area proportions.
- Passes all parameters into [`HybridRecommendationEngine`](file:///Users/subhranilbala/Downloads/SSPV-Irrigation-AI-Evaluation/sspv_pipeline/recommendation.py#L15).
- Compiles the global metrics DataFrame and executes the [`StatisticalComparisonEngine`](file:///Users/subhranilbala/Downloads/SSPV-Irrigation-AI-Evaluation/sspv_pipeline/statistics_engine.py#L18).

### 2.4 Thesis Artifact Exportation (`save_thesis_artifacts`)
Saves all production assets to `--output_dir`:
1. `sspv_7date_metrics.csv`: Full evaluation table.
2. `statistical_report.json`: Formatted JSON of all t-test stats, ANOVA $F$-scores, Cohen's $d$, and regression slope $Z$-test.
3. `unet_11ch_weights.pth`: PyTorch binary state dictionary.
4. `multitask_cnn_11ch_weights.pth`: PyTorch binary state dictionary.
5. `thesis_sspv_vs_control_summary.png`: 4-panel publication graphic (200 DPI):
   - NGRDI temporal growth trend
   - Crop stress pixel proportion boxplot
   - ExG temporal stability line plot
   - VARI temporal distribution consistency plot
6. `MTP_Thesis_Evaluation_Chapter_Report.txt`: Ready-to-copy academic chapter evaluation text report.

---

## 3. Example Usage

```python
from sspv_pipeline.orchestrator import SSPVPipelineOrchestrator

orch = SSPVPipelineOrchestrator(
    data_dir=None,  # Uses 7-date benchmark simulator
    pretrained=False,
    backbone_name="resnet18"
)

# 1. Train models
history = orch.train_models(epochs=4, batch_size=8)

# 2. Evaluate
df_metrics, stats_summary = orch.evaluate_all_dates()

# 3. Export all thesis artifacts
orch.save_thesis_artifacts("./outputs", df_metrics, stats_summary, history)
```
