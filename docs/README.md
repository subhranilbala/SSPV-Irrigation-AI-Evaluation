# SSPV Pipeline Module Documentation Index

This directory provides deep-dive architectural, mathematical, and algorithmic documentation for every module in the `sspv_pipeline` package.

---

## 📑 Module Index

| # | Document | Target Module | Domain / Primary Responsibility |
|---|---|---|---|
| **01** | [`01_config.md`](file:///Users/subhranilbala/Downloads/SSPV-Irrigation-AI-Evaluation/docs/modules/01_config.md) | `config.py` | Global configuration, class dictionaries, and Apple MPS / CUDA hardware auto-detection |
| **02** | [`02_preprocessing.md`](file:///Users/subhranilbala/Downloads/SSPV-Irrigation-AI-Evaluation/docs/modules/02_preprocessing.md) | `preprocessing.py` | sRGB inverse gamma linearization, Dark Object Subtraction (DOS), SIFT affine co-registration, median filtering |
| **03** | [`03_feature_engineering.md`](file:///Users/subhranilbala/Downloads/SSPV-Irrigation-AI-Evaluation/docs/modules/03_feature_engineering.md) | `feature_engineering.py` | Pixel-wise mathematical formulas for 8 RGB vegetation indices & 11-channel tensor assembly |
| **04** | [`04_pseudo_labeling.md`](file:///Users/subhranilbala/Downloads/SSPV-Irrigation-AI-Evaluation/docs/modules/04_pseudo_labeling.md) | `pseudo_labeling.py` | Weakly-supervised biophysical pseudo-labeling, 4-class segmentation masks, mound white salt crust detection |
| **05** | [`05_models.md`](file:///Users/subhranilbala/Downloads/SSPV-Irrigation-AI-Evaluation/docs/modules/05_models.md) | `models.py` | 11-channel first conv weight adapters, ResNet18 U-Net, Multi-Task CNN (ResNet18 / EfficientNet-B0), custom loss |
| **06** | [`06_explainability.md`](file:///Users/subhranilbala/Downloads/SSPV-Irrigation-AI-Evaluation/docs/modules/06_explainability.md) | `explainability.py` | Multi-Task Grad-CAM spatial visual attention heatmaps across 11 input channels |
| **07** | [`07_statistics_engine.md`](file:///Users/subhranilbala/Downloads/SSPV-Irrigation-AI-Evaluation/docs/modules/07_statistics_engine.md) | `statistics_engine.py` | Welch's two-sample t-test, one-way ANOVA (plots & dates), pooled Cohen's d effect sizes, OLS slope Z-test |
| **08** | [`08_recommendation.md`](file:///Users/subhranilbala/Downloads/SSPV-Irrigation-AI-Evaluation/docs/modules/08_recommendation.md) | `recommendation.py` | Two-tier decision support: determinate agronomic rules + 13-dim Decision Tree for ranked action confidence |
| **09** | [`09_dataset.md`](file:///Users/subhranilbala/Downloads/SSPV-Irrigation-AI-Evaluation/docs/modules/09_dataset.md) | `dataset.py` | Real DJI Zenmuse P1 multi-date folder loader with uniform striding & physically realistic 7-date simulator |
| **10** | [`10_orchestrator.md`](file:///Users/subhranilbala/Downloads/SSPV-Irrigation-AI-Evaluation/docs/modules/10_orchestrator.md) | `orchestrator.py` | End-to-end training loop, multi-date inference evaluator, dynamic class weighting, thesis artifact exporter |
| **11** | [`11_dashboard.md`](file:///Users/subhranilbala/Downloads/SSPV-Irrigation-AI-Evaluation/docs/modules/11_dashboard.md) | `dashboard.py` | Interactive Streamlit analytics UI with synchronized 4-panel spatial viewer, temporal trends, RuTAG audit, and bilingual voice advisories |

---

## 🔗 Related Documentation
- [**Overall Project README**](file:///Users/subhranilbala/Downloads/SSPV-Irrigation-AI-Evaluation/README.md)
- [**System Architecture & Scientific Guide**](file:///Users/subhranilbala/Downloads/SSPV-Irrigation-AI-Evaluation/SYSTEM_ARCHITECTURE_GUIDE.md)
