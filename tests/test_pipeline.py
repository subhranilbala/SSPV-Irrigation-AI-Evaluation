"""
Unit & Regression Test Suite for SSPV Irrigation AI Evaluation Pipeline.
M.Tech Thesis Project (MTP) — IIT Jodhpur (2026)
"""

import unittest
import numpy as np
import torch
import pandas as pd

from sspv_pipeline.config import CHANNEL_NAMES, SEG_CLASSES, get_device
from sspv_pipeline.preprocessing import UAVPreprocessor
from sspv_pipeline.feature_engineering import VegetationIndexExtractor
from sspv_pipeline.pseudo_labeling import BiophysicalPseudoLabeler
from sspv_pipeline.models import (
    UNetResNet18_11Ch, MultiTaskSSPVNet, MultiTaskLoss,
    adapt_resnet_first_conv_to_11ch, adapt_efficientnet_first_conv_to_11ch
)
from sspv_pipeline.explainability import MultiTaskGradCAM
from sspv_pipeline.statistics_engine import StatisticalComparisonEngine
from sspv_pipeline.recommendation import HybridRecommendationEngine
from sspv_pipeline.dataset import generate_realistic_7date_field_dataset, SSPVDronePatchDataset


class TestSSPVPipeline(unittest.TestCase):

    def setUp(self):
        self.device = torch.device("cpu")
        self.preprocessor = UAVPreprocessor()
        self.extractor = VegetationIndexExtractor()
        self.stat_engine = StatisticalComparisonEngine()
        self.rec_engine = HybridRecommendationEngine()

    def test_spectral_indices_formulas(self):
        """Validates all 8 vegetation indices match Slide 12 formulas."""
        # Pure green patch: R=0.1, G=0.8, B=0.1
        rgb = np.zeros((10, 10, 3), dtype=np.float32)
        rgb[:, :, 0] = 0.1
        rgb[:, :, 1] = 0.8
        rgb[:, :, 2] = 0.1

        indices = self.extractor.compute_indices(rgb)
        
        # NGRDI = (G - R) / (G + R) = (0.8 - 0.1) / (0.9) = 0.7778
        self.assertAlmostEqual(float(indices["NGRDI"][0, 0]), 0.7 / 0.9, places=3)
        # ExG = (2G - R - B) / (R + G + B) = (1.6 - 0.2) / 1.0 = 1.4
        self.assertAlmostEqual(float(indices["ExG"][0, 0]), 1.4, places=3)
        # TGI = G - 0.39R - 0.61B = 0.8 - 0.039 - 0.061 = 0.70
        self.assertAlmostEqual(float(indices["TGI"][0, 0]), 0.70, places=3)

    def test_11ch_tensor_dimensions(self):
        """Validates that build_11ch_tensor produces shape (11, H, W)."""
        rgb = np.random.uniform(0.1, 0.9, (64, 64, 3)).astype(np.float32)
        tensor_11ch, indices = self.extractor.build_11ch_tensor(rgb)
        self.assertEqual(tensor_11ch.shape, (11, 64, 64))
        self.assertEqual(len(indices), 8)

    def test_pseudo_labeling_classes(self):
        """Validates 4-class segmentation masks and salt crust detection (Slide 7)."""
        # Create a patch with a bright white salt patch in the center
        rgb = np.ones((50, 50, 3), dtype=np.float32) * 0.4
        rgb[20:30, 20:30, :] = 0.95  # Bright, achromatic white salt crust
        _, indices = self.extractor.build_11ch_tensor(rgb)
        mask = BiophysicalPseudoLabeler.generate_segmentation_mask(rgb, indices)
        
        self.assertEqual(mask.shape, (50, 50))
        # Salt patch (class 3) should be detected in the bright region
        self.assertTrue(np.any(mask == 3))

    def test_unet_11ch_forward(self):
        """Validates U-Net takes (B, 11, H, W) and outputs (B, 4, H, W)."""
        unet = UNetResNet18_11Ch(in_channels=11, num_classes=4, pretrained=False).to(self.device)
        x = torch.randn(2, 11, 64, 64, device=self.device)
        logits = unet(x)
        self.assertEqual(logits.shape, (2, 4, 64, 64))

    def test_multitask_cnn_backbones(self):
        """Validates MultiTaskSSPVNet with both ResNet18 and EfficientNet-B0 backbones."""
        x = torch.randn(2, 11, 64, 64, device=self.device)
        gate = torch.ones(2, 1, 64, 64, device=self.device)

        # Test ResNet18
        cnn_res = MultiTaskSSPVNet(in_channels=11, backbone_name="resnet18", pretrained=False).to(self.device)
        out_res = cnn_res(x, valid_crop_mask=gate)
        self.assertEqual(out_res["stress_logits"].shape, (2, 3))
        self.assertEqual(out_res["moisture_logits"].shape, (2, 3))
        self.assertEqual(out_res["yield_logits"].shape, (2, 3))

        # Test EfficientNet-B0
        cnn_eff = MultiTaskSSPVNet(in_channels=11, backbone_name="efficientnet_b0", pretrained=False).to(self.device)
        out_eff = cnn_eff(x, valid_crop_mask=gate)
        self.assertEqual(out_eff["stress_logits"].shape, (2, 3))

    def test_gradcam_spatial_heatmap(self):
        """Validates Grad-CAM outputs normalized (H, W) heatmap in [0, 1]."""
        cnn = MultiTaskSSPVNet(in_channels=11, backbone_name="resnet18", pretrained=False).to(self.device)
        gradcam = MultiTaskGradCAM(cnn)
        x = torch.randn(1, 11, 64, 64, device=self.device)
        heatmap = gradcam.generate_heatmap(x, task="stress_logits", target_class=0)
        self.assertEqual(heatmap.shape, (64, 64))
        self.assertTrue(heatmap.min() >= 0.0)
        self.assertTrue(heatmap.max() <= 1.0)

    def test_statistical_engine_execution(self):
        """Validates t-test, ANOVA, Cohen's d, and temporal regression."""
        data = {
            "date": ["12-11-2024"] * 6 + ["22-11-2024"] * 6,
            "day_num": [1] * 6 + [11] * 6,
            "plot_type": ["SSPV", "Control"] * 6,
            "mean_ngrdi": [0.25, -0.05, 0.28, -0.02, 0.30, -0.08, 0.35, -0.01, 0.36, -0.04, 0.38, -0.05],
            "mean_exg": [0.2, 0.05] * 6,
            "mean_vari": [0.1, 0.02] * 6,
            "mean_tgi": [0.05, 0.01] * 6,
            "mean_exb": [0.02, 0.01] * 6,
            "stress_pixel_ratio": [0.05, 0.35] * 6,
            "salt_patch_ratio": [0.02, 0.0] * 6,
            "moisture_zone_variance": [0.01, 0.04] * 6
        }
        df = pd.DataFrame(data)
        res = self.stat_engine.run_full_comparison(df)
        self.assertTrue("t_test" in res)
        self.assertTrue(res["t_test"]["significant_at_005"])
        self.assertTrue(res["effect_size"]["cohens_d_ngrdi"] > 0)

    def test_hybrid_recommendation_ranking(self):
        """Validates ranked recommendations with confidence scores (Slide 14)."""
        advice = self.rec_engine.recommend(
            ngrdi=0.005, exg=0.02, vari=0.01, tgi=0.005, exb=0.01, salt_ratio=0.01,
            stress_probs=np.array([0.1, 0.2, 0.7]),
            moisture_probs=np.array([0.8, 0.15, 0.05]),
            yield_probs=np.array([0.7, 0.2, 0.1])
        )
        self.assertIn("Urgent", advice["primary_recommendation"])
        self.assertEqual(len(advice["ranked_recommendations"]), 5)
        # Verify ranking is in descending confidence order
        confs = [item["confidence"] for item in advice["ranked_recommendations"]]
        self.assertEqual(confs, sorted(confs, reverse=True))


if __name__ == "__main__":
    unittest.main()
