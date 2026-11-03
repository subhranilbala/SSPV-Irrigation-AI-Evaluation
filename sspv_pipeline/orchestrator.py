"""
MODULE 9: End-to-End Training & Evaluation Orchestrator.

Trains U-Net + Multi-Task CNN, extracts plot metrics, runs statistical tests,
and exports thesis artifacts (CSV, JSON, model weights, publication figures).
"""

import os
import json
from datetime import datetime
from typing import Dict, List, Tuple, Optional, Any

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import seaborn as sns

from .config import (
    STRESS_CLASSES, MOISTURE_CLASSES, YIELD_CLASSES, get_device
)
from .preprocessing import UAVPreprocessor
from .feature_engineering import VegetationIndexExtractor
from .models import UNetResNet18_11Ch, MultiTaskSSPVNet, MultiTaskLoss
from .statistics_engine import StatisticalComparisonEngine
from .recommendation import HybridRecommendationEngine
from .dataset import SSPVDronePatchDataset, load_real_dji_folders_or_fallback


class SSPVPipelineOrchestrator:
    """Trains U-Net + Multi-Task CNN, extracts plot metrics, and runs statistical tests."""

    def __init__(
        self,
        data_dir: Optional[str] = None,
        pretrained: bool = False,
        backbone_name: str = "resnet18",
        max_images_per_date: int = 20
    ):
        self.device = get_device()
        self.preprocessor = UAVPreprocessor()
        self.vi_extractor = VegetationIndexExtractor()
        self.stat_engine = StatisticalComparisonEngine()
        self.rec_engine = HybridRecommendationEngine()

        raw_samples = load_real_dji_folders_or_fallback(
            data_dir,
            self.preprocessor,
            max_images_per_date=max_images_per_date
        )
        self.dataset = SSPVDronePatchDataset(raw_samples, self.preprocessor, self.vi_extractor)

        self.unet = UNetResNet18_11Ch(
            in_channels=11, num_classes=4, pretrained=pretrained
        ).to(self.device)
        self.mt_cnn = MultiTaskSSPVNet(
            in_channels=11, pretrained=pretrained, backbone_name=backbone_name
        ).to(self.device)

        # Compute dynamic inverse-frequency class weights per task head (Slide 10 & Slide 13)
        w_stress, w_moist, w_yield = self._compute_inverse_class_weights()
        self.mt_loss_fn = MultiTaskLoss(
            stress_class_weights=w_stress.to(self.device),
            moisture_class_weights=w_moist.to(self.device),
            yield_class_weights=w_yield.to(self.device),
        )

    def _compute_inverse_class_weights(self) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Computes inverse class frequency weights across the dataset for Weighted Cross-Entropy."""
        s_counts = np.ones(3, dtype=np.float32)
        m_counts = np.ones(3, dtype=np.float32)
        y_counts = np.ones(3, dtype=np.float32)

        for idx in range(len(self.dataset)):
            item = self.dataset[idx]
            s_counts[int(item["stress_label"].item())] += 1.0
            m_counts[int(item["moisture_label"].item())] += 1.0
            y_counts[int(item["yield_label"].item())] += 1.0

        def _inv_norm(counts: np.ndarray) -> torch.Tensor:
            inv = 1.0 / counts
            normed = inv / np.sum(inv) * 3.0
            return torch.tensor(normed, dtype=torch.float32)

        return _inv_norm(s_counts), _inv_norm(m_counts), _inv_norm(y_counts)

    def train_models(self, epochs: int = 5, batch_size: int = 8, lr: float = 1e-3) -> Dict[str, List[float]]:
        loader = DataLoader(self.dataset, batch_size=batch_size, shuffle=True)
        opt_unet = torch.optim.AdamW(self.unet.parameters(), lr=lr, weight_decay=1e-4)
        opt_cnn = torch.optim.AdamW(self.mt_cnn.parameters(), lr=lr, weight_decay=1e-4)
        seg_ce = nn.CrossEntropyLoss()

        history = {"unet_loss": [], "cnn_loss": [], "seg_acc": [], "stress_acc": []}

        self.unet.train()
        self.mt_cnn.train()

        for epoch in range(epochs):
            ep_unet_loss, ep_cnn_loss = 0.0, 0.0
            correct_seg, total_seg = 0, 0
            correct_stress, total_samples = 0, 0

            for batch in loader:
                x = batch["tensor_11ch"].to(self.device)
                seg_y = batch["seg_mask"].to(self.device)
                stress_y = batch["stress_label"].to(self.device)
                moist_y = batch["moisture_label"].to(self.device)
                yield_y = batch["yield_label"].to(self.device)

                # 1. Update U-Net Segmentation
                opt_unet.zero_grad()
                seg_logits = self.unet(x)
                loss_u = seg_ce(seg_logits, seg_y)
                loss_u.backward()
                opt_unet.step()

                # Compute binary valid crop gate from U-Net predictions (Classes 1 & 2)
                with torch.no_grad():
                    seg_preds = torch.argmax(seg_logits, dim=1)
                    valid_crop_mask = ((seg_preds == 1) | (seg_preds == 2)).unsqueeze(1)
                    correct_seg += int((seg_preds == seg_y).sum().item())
                    total_seg += int(seg_y.numel())

                # 2. Update Multi-Task CNN gated by U-Net valid crop mask
                opt_cnn.zero_grad()
                mt_out = self.mt_cnn(x, valid_crop_mask=valid_crop_mask)
                loss_c, _ = self.mt_loss_fn(mt_out, stress_y, moist_y, yield_y)
                loss_c.backward()
                opt_cnn.step()

                ep_unet_loss += float(loss_u.item())
                ep_cnn_loss += float(loss_c.item())

                stress_preds = torch.argmax(mt_out["stress_logits"], dim=1)
                correct_stress += int((stress_preds == stress_y).sum().item())
                total_samples += int(stress_y.size(0))

            history["unet_loss"].append(ep_unet_loss / max(1, len(loader)))
            history["cnn_loss"].append(ep_cnn_loss / max(1, len(loader)))
            history["seg_acc"].append(correct_seg / max(1, total_seg))
            history["stress_acc"].append(correct_stress / max(1, total_samples))

        return history

    def evaluate_all_dates(self) -> Tuple[pd.DataFrame, Dict[str, Any]]:
        """Runs inference across all 7 dates and computes statistical & recommendation outputs."""
        self.unet.eval()
        self.mt_cnn.eval()
        records = []

        with torch.no_grad():
            for idx in range(len(self.dataset)):
                sample = self.dataset[idx]
                x = sample["tensor_11ch"].unsqueeze(0).to(self.device)
                rgb = sample["rgb"]
                _, indices = self.vi_extractor.build_11ch_tensor(rgb)

                seg_logits = self.unet(x)
                seg_pred = torch.argmax(seg_logits, dim=1).squeeze(0).cpu().numpy()
                valid_crop = ((seg_pred == 1) | (seg_pred == 2))

                crop_gate = torch.from_numpy(valid_crop).unsqueeze(0).unsqueeze(0).to(self.device)
                mt_out = self.mt_cnn(x, valid_crop_mask=crop_gate)

                stress_probs = F.softmax(mt_out["stress_logits"], dim=1).squeeze(0).cpu().numpy()
                moist_probs = F.softmax(mt_out["moisture_logits"], dim=1).squeeze(0).cpu().numpy()
                yield_probs = F.softmax(mt_out["yield_logits"], dim=1).squeeze(0).cpu().numpy()

                region = valid_crop if np.any(valid_crop) else np.ones_like(valid_crop, dtype=bool)
                mean_ngrdi = float(np.mean(indices["NGRDI"][region]))
                mean_exg = float(np.mean(indices["ExG"][region]))
                mean_vari = float(np.mean(indices["VARI"][region]))
                mean_tgi = float(np.mean(indices["TGI"][region]))
                mean_exb = float(np.mean(indices["ExB"]))
                moisture_zone_var = float(np.var(indices["ExB"]))

                stress_ratio = float(np.mean(seg_pred == 2))
                salt_ratio = float(np.mean(seg_pred == 3))

                rec = self.rec_engine.recommend(
                    ngrdi=mean_ngrdi,
                    exg=mean_exg,
                    vari=mean_vari,
                    tgi=mean_tgi,
                    exb=mean_exb,
                    salt_ratio=salt_ratio,
                    stress_probs=stress_probs,
                    moisture_probs=moist_probs,
                    yield_probs=yield_probs
                )

                records.append({
                    "sample_id": idx,
                    "date": sample["date"],
                    "day_num": sample["day_num"],
                    "plot_type": sample["plot_type"],
                    "mean_ngrdi": mean_ngrdi,
                    "mean_exg": mean_exg,
                    "mean_vari": mean_vari,
                    "mean_tgi": mean_tgi,
                    "mean_exb": mean_exb,
                    "moisture_zone_variance": moisture_zone_var,
                    "stress_pixel_ratio": stress_ratio,
                    "salt_patch_ratio": salt_ratio,
                    "pred_stress": STRESS_CLASSES[int(np.argmax(stress_probs))],
                    "pred_moisture": MOISTURE_CLASSES[int(np.argmax(moist_probs))],
                    "pred_yield": YIELD_CLASSES[int(np.argmax(yield_probs))],
                    "recommendation": rec["primary_recommendation"],
                    "rec_confidence": rec["confidence"],
                    "rec_mode": rec["engine_mode"],
                    "ranked_top2": f"1. {rec['ranked_recommendations'][0]['action'][:28]}.. ({rec['ranked_recommendations'][0]['confidence']}) | 2. {rec['ranked_recommendations'][1]['action'][:28]}.. ({rec['ranked_recommendations'][1]['confidence']})"
                })

        df = pd.DataFrame(records)
        stats_summary = self.stat_engine.run_full_comparison(df)
        return df, stats_summary

    def save_thesis_artifacts(
        self,
        output_dir: str,
        df_metrics: pd.DataFrame,
        stats_summary: Dict[str, Any],
        history: Dict[str, List[float]]
    ) -> None:
        """
        Exports all trained weights, CSV tables, JSON statistical reports, and
        publication-grade figures into `output_dir` for the M.Tech thesis report.
        """
        os.makedirs(output_dir, exist_ok=True)

        # 1. Save CSV Metrics & JSON Statistical Summary
        csv_path = os.path.join(output_dir, "sspv_7date_metrics.csv")
        json_path = os.path.join(output_dir, "statistical_report.json")
        df_metrics.to_csv(csv_path, index=False)

        full_report = {
            "statistical_comparison": stats_summary,
            "training_history": history,
            "total_patches_evaluated": len(df_metrics),
            "dates_evaluated": sorted(df_metrics["date"].unique().tolist())
        }
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(full_report, f, indent=2)

        # 2. Save Trained PyTorch Model Weights
        torch.save(self.unet.state_dict(), os.path.join(output_dir, "unet_11ch_weights.pth"))
        torch.save(self.mt_cnn.state_dict(), os.path.join(output_dir, "multitask_cnn_11ch_weights.pth"))

        # 3. Save High-Resolution Temporal Trend Figure for Thesis Manuscript
        fig, axes = plt.subplots(2, 2, figsize=(14, 10), dpi=200)
        sns.lineplot(
            data=df_metrics, x="day_num", y="mean_ngrdi", hue="plot_type",
            marker="o", linewidth=2.5, ax=axes[0, 0]
        )
        axes[0, 0].set_title("Temporal NGRDI Growth Trend (SSPV vs. Control)")
        axes[0, 0].set_xlabel("Days Since First Flight (12-11-2024)")
        axes[0, 0].set_ylabel("Mean Plot NGRDI")
        axes[0, 0].grid(True, alpha=0.3)

        sns.boxplot(
            data=df_metrics, x="plot_type", y="stress_pixel_ratio", palette="Set2", ax=axes[0, 1]
        )
        axes[0, 1].set_title("Crop Stress Pixel Proportion (ANOVA Comparison)")
        axes[0, 1].set_xlabel("Irrigation Type")
        axes[0, 1].set_ylabel("Stress Pixel Proportion")
        axes[0, 1].grid(True, alpha=0.3)

        sns.lineplot(
            data=df_metrics, x="day_num", y="mean_exg", hue="plot_type",
            marker="s", linewidth=2.5, ax=axes[1, 0]
        )
        axes[1, 0].set_title("Temporal ExG Stability (SSPV vs. Control)")
        axes[1, 0].set_xlabel("Days Since First Flight (12-11-2024)")
        axes[1, 0].set_ylabel("Mean Plot ExG")
        axes[1, 0].grid(True, alpha=0.3)

        sns.lineplot(
            data=df_metrics, x="day_num", y="mean_vari", hue="plot_type",
            marker="D", linewidth=2.5, ax=axes[1, 1]
        )
        axes[1, 1].set_title("Temporal VARI Distribution (SSPV vs. Control)")
        axes[1, 1].set_xlabel("Days Since First Flight (12-11-2024)")
        axes[1, 1].set_ylabel("Mean Plot VARI")
        axes[1, 1].grid(True, alpha=0.3)

        plt.tight_layout()
        fig.savefig(os.path.join(output_dir, "thesis_sspv_vs_control_summary.png"))
        plt.close(fig)

    def generate_thesis_pdf_report_content(self, df_metrics: pd.DataFrame, stats_summary: Dict[str, Any]) -> str:
        """
        Compiles CSV metrics tables, statistical p-values, Cohen's d effect sizes,
        and temporal regression slopes into a formatted thesis report string suitable
        for direct PDF compilation or Word/LaTeX inclusion.
        """
        t_test = stats_summary["t_test"]
        anova = stats_summary["anova"]
        eff = stats_summary["effect_size"]
        reg = stats_summary["temporal_regression"]

        report_lines = [
            "=" * 85,
            "INDIAN INSTITUTE OF TECHNOLOGY JODHPUR",
            "SCHOOL OF ARTIFICIAL INTELLIGENCE & DATA SCIENCE",
            "M.TECH THESIS PROJECT (MTP) - FORMAL EVALUATION REPORT",
            "=" * 85,
            f"Project Title : AI Based Evaluation of Empirical Subsurface Porous Vessel",
            f"                (SSPV) Irrigation System Performance Using Images",
            f"Candidate     : Subhranil Bala (Roll No.: M25AI2038)",
            f"Supervisor    : Dr. Anand Krishnan Plappally",
            f"Generated On  : {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
            "-" * 85,
            "",
            "1. EXECUTIVE SUMMARY & STATISTICAL COMPARISON (SSPV vs. CONTROL)",
            f"   - Total Evaluated UAV Patches : {len(df_metrics)}",
            f"   - Survey Dates Evaluated      : {', '.join(sorted(df_metrics['date'].unique().tolist()))}",
            f"   - SSPV Mean NGRDI             : {t_test['sspv_mean_ngrdi']:.4f}",
            f"   - Control Mean NGRDI          : {t_test['control_mean_ngrdi']:.4f}",
            f"   - Welch's t-test statistic    : {t_test['t_statistic']:.4f} (p-value = {t_test['p_value']:.4e})",
            f"   - Statistical Significance    : {'YES (p < 0.05)' if t_test['significant_at_005'] else 'NO'}",
            f"   - Cohen's d Effect Size(NGRDI): {eff['cohens_d_ngrdi']:.2f} ({eff['interpretation_ngrdi']})",
            f"   - One-Way ANOVA (Plot Type)   : F = {anova['f_statistic']:.3f} (p = {anova['p_value']:.4e})",
            f"   - One-Way ANOVA (Survey Dates): F = {anova['f_statistic_across_dates']:.3f} (p = {anova['p_value_across_dates']:.4e})",
            "",
            "2. TEMPORAL GROWTH TREND & SLOPE COMPARISON",
            f"   - SSPV NGRDI Slope / Day      : {reg['sspv_slope_per_day']:.5f} (r = {reg['sspv_r_value']:.3f})",
            f"   - Control NGRDI Slope / Day   : {reg['control_slope_per_day']:.5f} (r = {reg['control_r_value']:.3f})",
            f"   - Inter-Plot Slope Difference : {reg['inter_plot_slope_difference']:.5f} (Z-stat = {reg['inter_plot_slope_z_stat']:.3f}, p = {reg['inter_plot_slope_p_value']:.4e})",
            f"   - SSPV ExG Temporal Std (Low) : {reg['sspv_exg_temporal_std']:.4f}",
            f"   - Control ExG Temporal Std    : {reg['control_exg_temporal_std']:.4f}",
            "",
            "3. RECOMMENDATION ENGINE DISTRIBUTION SUMMARY",
        ]

        rec_counts = df_metrics["recommendation"].value_counts()
        for rec_text, count in rec_counts.items():
            pct = (count / len(df_metrics)) * 100
            report_lines.append(f"   - [{pct:5.1f}% ({count} patches)] {rec_text}")

        report_lines.extend([
            "",
            "-" * 85,
            "End of Thesis Evaluation Chapter Report - IIT Jodhpur MTP 2026",
            "=" * 85
        ])
        return "\n".join(report_lines)
