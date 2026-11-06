"""
MODULE 10: Interactive Streamlit Dashboard (Slide 10, 14, 15, 17).

Provides a full interactive web UI for single-image inspection, multi-date
temporal trend visualization, and recommendation table export.
"""

import os
import sys

import numpy as np
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
import seaborn as sns
from PIL import Image

from .config import HAS_CV2, STRESS_CLASSES, MOISTURE_CLASSES, YIELD_CLASSES
from .orchestrator import SSPVPipelineOrchestrator
from .explainability import MultiTaskGradCAM

if HAS_CV2:
    import cv2


def init_and_train_pipeline(folder: str, ep: int, pre: bool, bbone: str, max_im: int):
    orch = SSPVPipelineOrchestrator(
        data_dir=folder.strip() if folder.strip() else None,
        pretrained=pre,
        backbone_name=bbone,
        max_images_per_date=max_im
    )
    hist = orch.train_models(epochs=ep)
    df_res, stat_res = orch.evaluate_all_dates()
    return orch, hist, df_res, stat_res


def run_streamlit_dashboard():
    import streamlit as st

    st.set_page_config(
        page_title="SSPV Irrigation AI Evaluation Dashboard - IIT Jodhpur",
        layout="wide"
    )

    st.title("AI-Based Evaluation of Subsurface Porous Vessel (SSPV) Irrigation Performance")
    st.markdown(
        "**M.Tech Thesis Project (MTP)** | **Subhranil Bala (M25AI2038)** | "
        "**Supervisor:** Dr. Anand Krishnan Plappally | **IIT Jodhpur**"
    )

    # Auto-detect the user's Downloads/Crops Data folder on macOS if it exists
    default_crops_path = os.path.expanduser("~/Downloads/Crops Data")
    if not os.path.exists(default_crops_path):
        default_crops_path = ""

    with st.sidebar:
        st.header("1. Data Input Configuration")
        data_dir = st.text_input(
            "Local Path to 'Crops Data' Folder",
            value=default_crops_path,
            help="Paste the full path to your 'Crops Data' folder containing 02-12-2024/images, 12-11-2024/images, etc."
        )
        max_imgs_ui = st.slider(
            "Max DJI Frames per Date Folder (0 = ALL)",
            min_value=0, max_value=200, value=15, step=5,
            help="Set to 0 to load all 1,924 images (takes ~12 mins of SIFT alignment), or 15-30 for fast uniformly-strided coverage across the drone flight."
        )
        uploaded_file = st.file_uploader(
            "Or Upload a Single DJI Drone Image (.JPG / .TIF)",
            type=["jpg", "jpeg", "tif", "tiff"]
        )
        st.header("2. Model & Inspection Controls")
        backbone_choice = st.selectbox(
            "Multi-Task CNN Backbone (Slide 13)",
            ["resnet18", "efficientnet_b0"]
        )
        epochs = st.slider("Training Epochs", min_value=1, max_value=25, value=4)
        use_pretrained = st.checkbox("Use ImageNet Pretrained Backbone", value=False)
        gradcam_task = st.selectbox(
            "Grad-CAM Target Head",
            ["stress_logits", "moisture_logits", "yield_logits"]
        )

    cached_init = st.cache_resource(init_and_train_pipeline)
    orch, history, df_metrics, stats_summary = cached_init(
        data_dir, epochs, use_pretrained, backbone_choice, max_imgs_ui
    )

    with st.sidebar:
        # Show status badge confirming whether real DJI images or synthetic fallback loaded
        is_real = any("source_file" in s for s in orch.dataset.samples)
        if is_real:
            st.success(f"Loaded {len(orch.dataset)} patches from real DJI images!")
        else:
            st.info("Running on 7-Date Synthetic Benchmark (enter valid 'Crops Data' path above for real DJI images).")

        sample_idx = st.slider(
            "Inspect Dataset Patch ID",
            min_value=0,
            max_value=max(0, len(orch.dataset) - 1),
            value=min(10, max(0, len(orch.dataset) - 1))
        )

    # Top KPI Metrics Row
    col1, col2, col3, col4 = st.columns(4)
    col1.metric(
        "SSPV Mean NGRDI",
        f"{stats_summary['t_test']['sspv_mean_ngrdi']:.4f}",
        delta=f"+{stats_summary['t_test']['sspv_mean_ngrdi'] - stats_summary['t_test']['control_mean_ngrdi']:.4f} vs Control"
    )
    col2.metric(
        "Welch t-test p-value",
        f"{stats_summary['t_test']['p_value']:.4e}",
        delta="Statistically Significant" if stats_summary["t_test"]["significant_at_005"] else "Not Significant"
    )
    col3.metric(
        "Cohen's d (NGRDI)",
        f"{stats_summary['effect_size']['cohens_d_ngrdi']:.2f}",
        delta=stats_summary["effect_size"]["interpretation_ngrdi"]
    )
    col4.metric(
        "ExG Temporal Variance (SSPV vs Ctrl)",
        f"{stats_summary['temporal_regression']['sspv_exg_temporal_std']:.4f}",
        delta=f"Ctrl: {stats_summary['temporal_regression']['control_exg_temporal_std']:.4f}"
    )

    st.divider()

    # Live Single-Image Upload Inspection (if user drags and drops a DJI .JPG)
    if uploaded_file is not None:
        st.subheader(f"Live Uploaded DJI Frame Analysis: `{uploaded_file.name}`")
        if HAS_CV2:
            file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
            bgr = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
            rgb_raw = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
            rgb_resized = cv2.resize(rgb_raw, (256, 256), interpolation=cv2.INTER_AREA)
        else:
            pil_up = Image.open(uploaded_file).convert("RGB").resize((256, 256), Image.BILINEAR)
            rgb_resized = np.asarray(pil_up)
        rgb = orch.preprocessor.process(rgb_resized)
        tensor_11ch, indices = orch.vi_extractor.build_11ch_tensor(rgb)
        x_tensor = torch.from_numpy(tensor_11ch).unsqueeze(0).to(orch.device)
        plot_title = f"Uploaded: {uploaded_file.name}"
    else:
        st.subheader("1. Spatial Inspection: 11-Channel U-Net Segmentation & Multi-Task Grad-CAM")
        sample = orch.dataset[min(sample_idx, len(orch.dataset) - 1)]
        rgb = sample["rgb"]
        x_tensor = sample["tensor_11ch"].unsqueeze(0).to(orch.device)
        _, indices = orch.vi_extractor.build_11ch_tensor(rgb)
        plot_title = f"{sample['plot_type']} ({sample['date']})"

    with torch.no_grad():
        seg_logits = orch.unet(x_tensor)
        seg_mask = torch.argmax(seg_logits, dim=1).squeeze(0).cpu().numpy()

    gradcam = MultiTaskGradCAM(orch.mt_cnn)
    cam_map = gradcam.generate_heatmap(x_tensor, task=gradcam_task)

    fig1, axes = plt.subplots(1, 4, figsize=(16, 4))
    axes[0].imshow(rgb)
    axes[0].set_title(f"Preprocessed RGB: {plot_title}")
    axes[0].axis("off")

    im1 = axes[1].imshow(indices["NGRDI"], cmap="RdYlGn", vmin=-0.2, vmax=0.3)
    axes[1].set_title("NGRDI Canopy Greenness Heatmap")
    axes[1].axis("off")
    plt.colorbar(im1, ax=axes[1], fraction=0.046)

    axes[2].imshow(seg_mask, cmap="viridis", vmin=0, vmax=3)
    axes[2].set_title("U-Net Mask (0:Soil, 1:Crop, 2:Stress, 3:Salt)")
    axes[2].axis("off")

    axes[3].imshow(rgb)
    axes[3].imshow(cam_map, cmap="jet", alpha=0.45)
    axes[3].set_title(f"Grad-CAM Attention ({gradcam_task})")
    axes[3].axis("off")
    st.pyplot(fig1)

    st.divider()

    # Multi-Date Temporal Comparison Charts (Slide 14 & 16)
    st.subheader("2. Multi-Date Temporal Growth Trend & SSPV vs. Control Statistical Comparison")
    fig2, (ax_left, ax_right) = plt.subplots(1, 2, figsize=(14, 4.5))

    sns.lineplot(
        data=df_metrics, x="day_num", y="mean_ngrdi", hue="plot_type",
        marker="o", linewidth=2.5, ax=ax_left
    )
    ax_left.set_title("Temporal NGRDI Growth Trend Across Survey Dates")
    ax_left.set_xlabel("Days Since First Survey (12-11-2024)")
    ax_left.set_ylabel("Mean Plot NGRDI")
    ax_left.grid(True, alpha=0.3)

    sns.boxplot(
        data=df_metrics, x="plot_type", y="stress_pixel_ratio", hue="plot_type", palette="Set2", legend=False, ax=ax_right
    )
    ax_right.set_title("Crop Stress Pixel Proportion (One-Way ANOVA Comparison)")
    ax_right.set_xlabel("Irrigation System")
    ax_right.set_ylabel("Stressed Pixel Ratio")
    ax_right.grid(True, alpha=0.3)
    st.pyplot(fig2)

    st.divider()

    # ExG & VARI Temporal Trend Charts (Slide 14 & 16: Time-series vegetation health graphs)
    st.subheader("2b. ExG & VARI Temporal Stability — SSPV vs. Control")
    fig3, (ax_exg, ax_vari) = plt.subplots(1, 2, figsize=(14, 4.5))

    sns.lineplot(
        data=df_metrics, x="day_num", y="mean_exg", hue="plot_type",
        marker="s", linewidth=2.5, ax=ax_exg
    )
    ax_exg.set_title("Temporal ExG Stability (SSPV vs. Control)")
    ax_exg.set_xlabel("Days Since First Survey (12-11-2024)")
    ax_exg.set_ylabel("Mean Plot ExG")
    ax_exg.grid(True, alpha=0.3)

    sns.lineplot(
        data=df_metrics, x="day_num", y="mean_vari", hue="plot_type",
        marker="D", linewidth=2.5, ax=ax_vari
    )
    ax_vari.set_title("Temporal VARI Distribution (SSPV vs. Control)")
    ax_vari.set_xlabel("Days Since First Survey (12-11-2024)")
    ax_vari.set_ylabel("Mean Plot VARI")
    ax_vari.grid(True, alpha=0.3)

    plt.tight_layout()
    st.pyplot(fig3)

    st.divider()

    # Hybrid Recommendation Table & CSV Download Button
    st.subheader("3. Actionable Irrigation Scheduling Recommendations")

    col_dl1, col_dl2 = st.columns(2)
    with col_dl1:
        st.download_button(
            label="📥 Download Complete 7-Date Metrics Table (CSV)",
            data=df_metrics.to_csv(index=False).encode("utf-8"),
            file_name="sspv_7date_metrics.csv",
            mime="text/csv"
        )
    with col_dl2:
        thesis_pdf_text = orch.generate_thesis_pdf_report_content(df_metrics, stats_summary)
        st.download_button(
            label="📄 Download Complete Thesis PDF Chapter Report (.txt/.pdf)",
            data=thesis_pdf_text.encode("utf-8"),
            file_name="MTP_Thesis_Evaluation_Chapter_Report.txt",
            mime="text/plain"
        )

    st.dataframe(
        df_metrics[[
            "date", "plot_type", "mean_ngrdi", "mean_exg", "mean_vari",
            "salt_patch_ratio", "pred_stress", "pred_moisture", "pred_yield",
            "recommendation", "rec_confidence", "rec_mode", "ranked_top2"
        ]],
        use_container_width=True
    )
