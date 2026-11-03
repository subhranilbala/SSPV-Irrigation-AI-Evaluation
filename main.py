"""
================================================================================
M.Tech Thesis Project (MTP) — CLI & Streamlit Entrypoint
================================================================================
Title: AI Based Evaluation of Empirical Subsurface Porous Vessel (SSPV)
       Irrigation System Performance Using Images
Author: Subhranil Bala (Roll No.: M25AI2038)
Supervisor: Dr. Anand Krishnan Plappally
Institute: Indian Institute of Technology Jodhpur (School of AI & Data Science)
================================================================================

Usage:
    CLI mode:       python main.py --mode run_all --backbone resnet18
    Streamlit mode: streamlit run main.py
"""

import os
import sys
import json
import argparse

from sspv_pipeline.config import get_device
from sspv_pipeline.orchestrator import SSPVPipelineOrchestrator
from sspv_pipeline.explainability import MultiTaskGradCAM


def main():
    default_crops_dir = os.path.expanduser("~/Downloads/Crops Data")
    if not os.path.exists(default_crops_dir):
        default_crops_dir = None

    parser = argparse.ArgumentParser(description="SSPV Irrigation AI Evaluation Pipeline")
    parser.add_argument(
        "--mode", type=str, default="run_all",
        choices=["run_all", "verify"],
        help="Execution mode"
    )
    parser.add_argument(
        "--data_dir", type=str, default=default_crops_dir,
        help="Path to 'Crops Data' folder containing multi-date DJI folders"
    )
    parser.add_argument(
        "--backbone", type=str, default="resnet18",
        choices=["resnet18", "efficientnet_b0"],
        help="Shared CNN backbone architecture (Slide 10 & 13: resnet18 or efficientnet_b0)"
    )
    parser.add_argument(
        "--max_images_per_date", type=int, default=20,
        help="Number of DJI frames to load per date folder (set to 0 to load ALL 1,924 images)"
    )
    parser.add_argument("--epochs", type=int, default=4, help="Number of training epochs")
    parser.add_argument("--batch_size", type=int, default=8, help="Training batch size")
    parser.add_argument(
        "--pretrained", action="store_true",
        help="Enable ImageNet pretrained weights for 11-ch backbones"
    )
    parser.add_argument(
        "--output_dir", type=str, default="./outputs",
        help="Directory to export CSV metrics, JSON stats, plots, and .pth weights"
    )
    args, _ = parser.parse_known_args()

    print("=" * 80)
    print("IIT JODHPUR M.TECH THESIS PROJECT (MTP) - SSPV AI PIPELINE")
    print(f"Compute Device Selected : {get_device()}")
    print(f"Multi-Task CNN Backbone : {args.backbone}")
    print(f"Max Frames Per Date     : {'ALL (100%)' if args.max_images_per_date <= 0 else args.max_images_per_date}")
    print(f"Input Data Directory    : {args.data_dir if args.data_dir else 'Synthetic 7-Date Benchmark (Fallback)'}")
    print("=" * 80)

    orchestrator = SSPVPipelineOrchestrator(
        data_dir=args.data_dir,
        pretrained=args.pretrained,
        backbone_name=args.backbone,
        max_images_per_date=args.max_images_per_date
    )
    print(f"[1/5] Loaded {len(orchestrator.dataset)} multi-date UAV patches.")

    print(f"[2/5] Training 11-Channel U-Net & Multi-Task CNN for {args.epochs} epochs...")
    history = orchestrator.train_models(epochs=args.epochs, batch_size=args.batch_size)
    print(
        f"      Final U-Net Seg Acc: {history['seg_acc'][-1]*100:.2f}% | "
        f"Multi-Task Stress Acc: {history['stress_acc'][-1]*100:.2f}%"
    )

    print("[3/5] Running Statistical Comparison Engine (t-test, ANOVA, Cohen's d, Regression)...")
    df_metrics, stats_summary = orchestrator.evaluate_all_dates()
    print(json.dumps(stats_summary, indent=2))

    print("[4/5] Verifying 11-Channel Grad-CAM Explainability...")
    sample_x = orchestrator.dataset[0]["tensor_11ch"].unsqueeze(0).to(orchestrator.device)
    gradcam = MultiTaskGradCAM(orchestrator.mt_cnn)
    heatmap = gradcam.generate_heatmap(sample_x, task="stress_logits")
    print(f"      Grad-CAM Heatmap Generated Successfully | Shape: {heatmap.shape}")

    print(f"[5/5] Exporting Thesis Artifacts to '{args.output_dir}'...")
    orchestrator.save_thesis_artifacts(args.output_dir, df_metrics, stats_summary, history)

    # Also export the formatted thesis text report
    report_txt = orchestrator.generate_thesis_pdf_report_content(df_metrics, stats_summary)
    report_path = os.path.join(args.output_dir, "MTP_Thesis_Evaluation_Chapter_Report.txt")
    with open(report_path, "w", encoding="utf-8") as f:
        f.write(report_txt)

    print(f"      Saved: sspv_7date_metrics.csv, statistical_report.json, model weights, figures, & Thesis Chapter Report.")
    print("=" * 80)


if __name__ == "__main__":
    # Detect if executed via `streamlit run main.py`
    if "streamlit" in sys.modules:
        from sspv_pipeline.dashboard import run_streamlit_dashboard
        run_streamlit_dashboard()
    else:
        main()
