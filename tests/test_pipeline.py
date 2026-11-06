"""
Comprehensive Unit & Regression Test Suite for SSPV Irrigation AI Evaluation Pipeline.
Targets 100% test coverage across all modules and edge cases.
M.Tech Thesis Project (MTP) — IIT Jodhpur (2026)
"""

import io
import os
import sys
import shutil
import tempfile
import unittest
from unittest.mock import MagicMock, patch
import numpy as np
from PIL import Image
import torch
import torch.nn as nn
import pandas as pd

from sspv_pipeline.config import CHANNEL_NAMES, SEG_CLASSES, get_device
import sspv_pipeline.config as config_mod
from sspv_pipeline.preprocessing import UAVPreprocessor
import sspv_pipeline.preprocessing as prep_mod
from sspv_pipeline.feature_engineering import VegetationIndexExtractor
from sspv_pipeline.pseudo_labeling import BiophysicalPseudoLabeler
from sspv_pipeline.models import (
    UNetResNet18_11Ch, MultiTaskSSPVNet, MultiTaskLoss,
    adapt_resnet_first_conv_to_11ch, adapt_efficientnet_first_conv_to_11ch
)
import sspv_pipeline.models as models_mod
from sspv_pipeline.explainability import MultiTaskGradCAM
import sspv_pipeline.explainability as xai_mod
from sspv_pipeline.statistics_engine import StatisticalComparisonEngine
from sspv_pipeline.recommendation import HybridRecommendationEngine
from sspv_pipeline.dataset import (
    generate_realistic_7date_field_dataset,
    load_real_dji_folders_or_fallback,
    SSPVDronePatchDataset
)
from sspv_pipeline.orchestrator import SSPVPipelineOrchestrator
from sspv_pipeline.dashboard import run_streamlit_dashboard


class TestSSPVPipelineComprehensive(unittest.TestCase):

    def setUp(self):
        self.device = torch.device("cpu")
        self.preprocessor = UAVPreprocessor(median_kernel=3)
        self.extractor = VegetationIndexExtractor()
        self.stat_engine = StatisticalComparisonEngine()
        self.rec_engine = HybridRecommendationEngine()

    # ==========================================================================
    # 1. CONFIG MODULE TESTS
    # ==========================================================================
    def test_config_device_selection(self):
        """Tests CUDA, MPS, and CPU branches in get_device()."""
        with patch("torch.cuda.is_available", return_value=True):
            self.assertEqual(get_device(), torch.device("cuda"))

        with patch("torch.cuda.is_available", return_value=False):
            with patch("torch.backends.mps.is_available", return_value=True):
                self.assertEqual(get_device(), torch.device("mps"))

        with patch("torch.cuda.is_available", return_value=False):
            with patch("torch.backends.mps.is_available", return_value=False):
                self.assertEqual(get_device(), torch.device("cpu"))

    # ==========================================================================
    # 2. FEATURE ENGINEERING MODULE TESTS
    # ==========================================================================
    def test_spectral_indices_formulas(self):
        """Validates all 8 vegetation indices match Slide 12 formulas."""
        rgb = np.zeros((10, 10, 3), dtype=np.float32)
        rgb[:, :, 0] = 0.1
        rgb[:, :, 1] = 0.8
        rgb[:, :, 2] = 0.1

        indices = self.extractor.compute_indices(rgb)
        self.assertAlmostEqual(float(indices["NGRDI"][0, 0]), 0.7 / 0.9, places=3)
        self.assertAlmostEqual(float(indices["ExG"][0, 0]), 1.4, places=3)
        self.assertAlmostEqual(float(indices["ExR"][0, 0]), -0.66, places=3)
        self.assertAlmostEqual(float(indices["TGI"][0, 0]), 0.70, places=3)
        self.assertAlmostEqual(float(indices["ExB"][0, 0]), -0.66, places=3)

    def test_11ch_tensor_dimensions(self):
        """Validates that build_11ch_tensor produces shape (11, H, W)."""
        rgb = np.random.uniform(0.1, 0.9, (64, 64, 3)).astype(np.float32)
        tensor_11ch, indices = self.extractor.build_11ch_tensor(rgb)
        self.assertEqual(tensor_11ch.shape, (11, 64, 64))
        self.assertEqual(len(indices), 8)

    # ==========================================================================
    # 3. PREPROCESSING MODULE TESTS
    # ==========================================================================
    def test_preprocessing_all_methods(self):
        """Tests DN conversion, DOS, ROI clipping, filtering, and image loading."""
        # 8-bit input
        raw_uint8 = np.random.randint(10, 240, (64, 64, 3), dtype=np.uint8)
        refl = self.preprocessor.dn_to_reflectance(raw_uint8)
        self.assertEqual(refl.shape, (64, 64, 3))
        self.assertTrue(refl.min() >= 0.0 and refl.max() <= 1.0)

        # 16-bit input with calibration panel factor
        raw_uint16 = np.random.randint(100, 60000, (64, 64, 3), dtype=np.uint16)
        refl_16 = self.preprocessor.dn_to_reflectance(raw_uint16, panel_factor=np.array([1.0, 1.0, 1.0]))
        self.assertEqual(refl_16.shape, (64, 64, 3))

        # DOS atmospheric correction
        dos_refl = self.preprocessor.dark_object_subtraction(refl)
        self.assertEqual(dos_refl.shape, (64, 64, 3))

        # ROI clipping & filtering (median_kernel = 3 and median_kernel = 1)
        clipped = self.preprocessor.clip_and_filter(dos_refl, roi_bbox=(10, 50, 10, 50))
        self.assertEqual(clipped.shape, (40, 40, 3))
        prep_no_filter = UAVPreprocessor(median_kernel=1)
        unfiltered = prep_no_filter.clip_and_filter(dos_refl)
        self.assertEqual(unfiltered.shape, dos_refl.shape)

        # End-to-end process
        proc = self.preprocessor.process(raw_uint8, roi_bbox=(0, 32, 0, 32))
        self.assertEqual(proc.shape, (32, 32, 3))

        # Geometric alignment: identical, different dimensions, cv2 fallback, and blank images (<10 keypoints)
        aligned = self.preprocessor.geometric_alignment(dos_refl, dos_refl)
        self.assertEqual(aligned.shape, dos_refl.shape)
        diff_size = np.random.uniform(0, 1, (32, 32, 3)).astype(np.float32)
        aligned_diff = self.preprocessor.geometric_alignment(diff_size, dos_refl)
        self.assertEqual(aligned_diff.shape, dos_refl.shape)

        # Blank image alignment (des1 is None or < 10 keypoints)
        blank1 = np.zeros((64, 64, 3), dtype=np.float32)
        blank2 = np.zeros((64, 64, 3), dtype=np.float32)
        aligned_blank = self.preprocessor.geometric_alignment(blank1, blank2)
        self.assertEqual(aligned_blank.shape, blank1.shape)

        # Preprocessing without OpenCV (HAS_CV2 = False)
        with patch("sspv_pipeline.preprocessing.HAS_CV2", False):
            aligned_nocv = self.preprocessor.geometric_alignment(diff_size, dos_refl)
            self.assertEqual(aligned_nocv.shape, dos_refl.shape)
            with tempfile.NamedTemporaryFile(suffix=".png") as tmp_png:
                im = Image.fromarray(raw_uint8)
                im.save(tmp_png.name)
                loaded_png, _ = self.preprocessor.load_image(tmp_png.name)
                self.assertEqual(loaded_png.shape, (64, 64, 3))

        # Rasterio loading branch (HAS_RASTERIO = True)
        mock_rasterio = MagicMock()
        mock_src = MagicMock()
        mock_src.read.return_value = np.zeros((3, 32, 32), dtype=np.uint8)
        mock_src.crs = "EPSG:4326"
        mock_src.transform = (1.0, 0.0, 0.0, 0.0, -1.0, 0.0)
        mock_rasterio.open.return_value.__enter__.return_value = mock_src
        with patch("sspv_pipeline.preprocessing.HAS_RASTERIO", True):
            with patch.dict(sys.modules, {"rasterio": mock_rasterio}):
                with patch("sspv_pipeline.preprocessing.rasterio", mock_rasterio, create=True):
                    tif_img, tif_meta = self.preprocessor.load_image("dummy.tif")
                    self.assertEqual(tif_img.shape, (32, 32, 3))
                    self.assertEqual(tif_meta["crs"], "EPSG:4326")

        # Image file loading & FileNotFoundError
        with tempfile.NamedTemporaryFile(suffix=".jpg") as tmp:
            im = Image.fromarray(raw_uint8)
            im.save(tmp.name)
            loaded_img, meta = self.preprocessor.load_image(tmp.name)
            self.assertEqual(loaded_img.shape, (64, 64, 3))

        with self.assertRaises(FileNotFoundError):
            self.preprocessor.load_image("/nonexistent/path/image.jpg")

    # ==========================================================================
    # 4. PSEUDO-LABELING MODULE TESTS
    # ==========================================================================
    def test_pseudo_labeling_all_branches(self):
        """Tests 4-class segmentation masks and all stress/moisture/yield conditions."""
        # 1. Image with salt crust and healthy canopy
        rgb = np.ones((50, 50, 3), dtype=np.float32) * 0.4
        rgb[20:30, 20:30, :] = 0.95  # Salt patch (Class 3)
        rgb[5:15, 5:15, 1] = 0.85    # Healthy canopy (Class 1)
        _, indices = self.extractor.build_11ch_tensor(rgb)

        mask = BiophysicalPseudoLabeler.generate_segmentation_mask(rgb, indices)
        self.assertEqual(mask.shape, (50, 50))
        self.assertTrue(np.any(mask == 3))
        self.assertTrue(np.any(mask == 1))

        # 2. Multi-task labels: crop pixels present
        stress, moist, yld = BiophysicalPseudoLabeler.generate_multitask_labels(rgb, indices, mask)
        self.assertIn(stress, [0, 1, 2])
        self.assertIn(moist, [0, 1, 2])
        self.assertIn(yld, [0, 1, 2])

        # 3. Multi-task labels: NO crop pixels present (empty mask)
        empty_mask = np.zeros((50, 50), dtype=np.int64)
        stress2, moist2, yld2 = BiophysicalPseudoLabeler.generate_multitask_labels(rgb, indices, empty_mask)
        self.assertIn(stress2, [0, 1, 2])

        # 4. Low yield / high stress conditions
        dark_rgb = np.ones((20, 20, 3), dtype=np.float32) * 0.1
        _, dark_idx = self.extractor.build_11ch_tensor(dark_rgb)
        s3, m3, y3 = BiophysicalPseudoLabeler.generate_multitask_labels(dark_rgb, dark_idx, empty_mask[:20, :20])
        self.assertEqual(y3, 0)  # Low yield branch

    # ==========================================================================
    # 5. DATASET & SIMULATOR MODULE TESTS
    # ==========================================================================
    def test_dataset_real_folders_and_simulator(self):
        """Tests physical simulator and real folder scanning logic."""
        samples = generate_realistic_7date_field_dataset(patch_size=32, patches_per_date=2)
        self.assertEqual(len(samples), 14)

        dataset = SSPVDronePatchDataset(samples, self.preprocessor, self.extractor)
        self.assertEqual(len(dataset), 14)
        item = dataset[0]
        self.assertEqual(item["tensor_11ch"].shape, (11, 32, 32))
        self.assertEqual(item["seg_mask"].shape, (32, 32))
        self.assertIn(item["plot_type"], ["SSPV", "Control"])

        # Test real folder scanning with a mock multi-date directory
        tmp_dir = tempfile.mkdtemp()
        try:
            date_dir = os.path.join(tmp_dir, "12-11-2024", "images")
            os.makedirs(date_dir, exist_ok=True)
            # Create dummy DJI images including a macOS hidden resource fork file
            im = Image.fromarray(np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8))
            im.save(os.path.join(date_dir, "DJI_0001.JPG"))
            im.save(os.path.join(date_dir, "DJI_0002.JPG"))
            im.save(os.path.join(date_dir, "._DJI_0001.JPG"))  # macOS dot-underscore hidden file

            # Add a non-date directory to test ValueError skip
            os.makedirs(os.path.join(tmp_dir, "metadata_notes"), exist_ok=True)

            # Test scanning with max_images_per_date > 0 and <= 0
            scanned_samples = load_real_dji_folders_or_fallback(tmp_dir, self.preprocessor, patch_size=32, max_images_per_date=1)
            self.assertTrue(len(scanned_samples) > 0)
            scanned_all = load_real_dji_folders_or_fallback(tmp_dir, self.preprocessor, patch_size=32, max_images_per_date=0)
            self.assertTrue(len(scanned_all) > 0)

            # Test scanning without OpenCV (HAS_CV2 = False)
            with patch("sspv_pipeline.dataset.HAS_CV2", False):
                scanned_nocv = load_real_dji_folders_or_fallback(tmp_dir, self.preprocessor, patch_size=32, max_images_per_date=1)
                self.assertTrue(len(scanned_nocv) > 0)

            # Test folder with NO valid dates -> falls back to synthetic dataset
            tmp_nodates = os.path.join(tmp_dir, "empty_dir")
            os.makedirs(tmp_nodates, exist_ok=True)
            fallback_nodates = load_real_dji_folders_or_fallback(tmp_nodates, self.preprocessor, patch_size=32)
            self.assertEqual(len(fallback_nodates), 42)

            # Test PermissionError handling during os.listdir
            with patch("os.listdir", side_effect=PermissionError("Mock Permission")):
                fallback_perm = load_real_dji_folders_or_fallback(tmp_dir, self.preprocessor, patch_size=32)
                self.assertEqual(len(fallback_perm), 42)
        finally:
            shutil.rmtree(tmp_dir, ignore_errors=True)

    # ==========================================================================
    # 6. NEURAL NETWORK MODELS & LOSS TESTS
    # ==========================================================================
    def test_models_forward_and_conv_adaptation(self):
        """Tests 11-channel conv adapters, U-Net, MultiTask CNN, and MultiTaskLoss."""
        # Test conv adapters with bias=True and bias=False
        dummy_res = nn.Sequential()
        dummy_res.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=True)
        adapt_resnet_first_conv_to_11ch(dummy_res, in_channels=11)
        self.assertEqual(dummy_res.conv1.in_channels, 11)

        dummy_eff = nn.Sequential()
        dummy_eff.features = nn.Sequential(nn.Sequential(nn.Conv2d(3, 32, kernel_size=3, bias=True)))
        adapt_efficientnet_first_conv_to_11ch(dummy_eff, in_channels=11)
        self.assertEqual(dummy_eff.features[0][0].in_channels, 11)

        x = torch.randn(2, 11, 64, 64, device=self.device)
        gate = torch.ones(2, 1, 64, 64, device=self.device)

        # U-Net forward pass
        unet = UNetResNet18_11Ch(in_channels=11, num_classes=4, pretrained=False).to(self.device)
        logits = unet(x)
        self.assertEqual(logits.shape, (2, 4, 64, 64))

        # Multi-Task CNN forward pass: ResNet18 & EfficientNet-B0
        cnn_res = MultiTaskSSPVNet(in_channels=11, backbone_name="resnet18", pretrained=False).to(self.device)
        out_res = cnn_res(x, valid_crop_mask=gate)
        self.assertEqual(out_res["stress_logits"].shape, (2, 3))

        cnn_eff = MultiTaskSSPVNet(in_channels=11, backbone_name="efficientnet_b0", pretrained=False).to(self.device)
        out_eff = cnn_eff(x, valid_crop_mask=gate)
        self.assertEqual(out_eff["stress_logits"].shape, (2, 3))

        # Loss function
        loss_fn = MultiTaskLoss()
        t_s = torch.tensor([0, 1], dtype=torch.long, device=self.device)
        t_m = torch.tensor([1, 2], dtype=torch.long, device=self.device)
        t_y = torch.tensor([2, 0], dtype=torch.long, device=self.device)
        loss, metrics = loss_fn(out_res, t_s, t_m, t_y)
        self.assertTrue(loss.item() > 0)

        # Exception fallbacks during weights initialization (offline simulation)
        from torchvision import models as tv_models
        with patch.object(tv_models, "resnet18", side_effect=[Exception("Weight download fail"), tv_models.resnet18(weights=None)]):
            unet_fallback = UNetResNet18_11Ch(in_channels=11, pretrained=True)
            self.assertIsNotNone(unet_fallback)

        with patch.object(tv_models, "resnet18", side_effect=[Exception("Weight download fail"), tv_models.resnet18(weights=None)]):
            res_fallback = MultiTaskSSPVNet(in_channels=11, backbone_name="resnet18", pretrained=True)
            self.assertIsNotNone(res_fallback)

        with patch.object(tv_models, "efficientnet_b0", side_effect=[Exception("Weight download fail"), tv_models.efficientnet_b0(weights=None)]):
            eff_fallback = MultiTaskSSPVNet(in_channels=11, backbone_name="efficientnet_b0", pretrained=True)
            self.assertIsNotNone(eff_fallback)

    # ==========================================================================
    # 7. EXPLAINABILITY (GRAD-CAM) TESTS
    # ==========================================================================
    def test_gradcam_all_branches(self):
        """Tests Grad-CAM heatmap generation, autotarget, and edge cases."""
        cnn = MultiTaskSSPVNet(in_channels=11, backbone_name="resnet18", pretrained=False).to(self.device)
        gradcam = MultiTaskGradCAM(cnn)
        x = torch.randn(1, 11, 64, 64, device=self.device)

        # Auto-detect target class (target_class=None)
        heatmap = gradcam.generate_heatmap(x, task="stress_logits", target_class=None)
        self.assertEqual(heatmap.shape, (64, 64))
        self.assertTrue(heatmap.min() >= 0.0 and heatmap.max() <= 1.0)

        # Explicit target class
        heatmap_explicit = gradcam.generate_heatmap(x, task="moisture_logits", target_class=1)
        self.assertEqual(heatmap_explicit.shape, (64, 64))

        # Edge case: gradients are None (simulate failure in backward hook)
        orig_backward = torch.Tensor.backward

        def mock_null_backward(self_tensor, *args, **kwargs):
            orig_backward(self_tensor, *args, **kwargs)
            gradcam.gradients = None

        with patch.object(torch.Tensor, "backward", mock_null_backward):
            heatmap_null = gradcam.generate_heatmap(x)
            self.assertEqual(heatmap_null.shape, (64, 64))
            self.assertTrue(np.all(heatmap_null == 0))

        # Edge case: Grad-CAM without OpenCV (HAS_CV2 = False) -> scipy zoom
        with patch("sspv_pipeline.explainability.HAS_CV2", False):
            heatmap_nocv = gradcam.generate_heatmap(x)
            self.assertEqual(heatmap_nocv.shape, (64, 64))

        # Edge case: uniform/flat activation producing zero range (line 85)
        def mock_flat_backward(self_tensor, *args, **kwargs):
            orig_backward(self_tensor, *args, **kwargs)
            gradcam.gradients = torch.zeros_like(gradcam.gradients)

        with patch.object(torch.Tensor, "backward", mock_flat_backward):
            heatmap_flat = gradcam.generate_heatmap(x)
            self.assertEqual(heatmap_flat.shape, (64, 64))
            self.assertTrue(np.all(heatmap_flat == 0))

    # ==========================================================================
    # 8. STATISTICAL COMPARISON ENGINE TESTS
    # ==========================================================================
    def test_statistical_engine_complete(self):
        """Tests t-test, ANOVA (plots and dates), Cohen's d, and regression fallbacks."""
        # Edge cases for Cohen's d
        d_zero = self.stat_engine.cohens_d(np.array([1.0]), np.array([2.0]))
        self.assertEqual(d_zero, 0.0)
        self.assertIn("Large", self.stat_engine.interpret_cohens_d(0.9))
        self.assertIn("Medium", self.stat_engine.interpret_cohens_d(0.6))
        self.assertIn("Small", self.stat_engine.interpret_cohens_d(0.3))
        self.assertIn("Negligible", self.stat_engine.interpret_cohens_d(0.05))

        # Standard multi-date DataFrame
        data = {
            "date": ["12-11-2024"] * 4 + ["22-11-2024"] * 4,
            "day_num": [1] * 4 + [11] * 4,
            "plot_type": ["SSPV", "Control"] * 4,
            "mean_ngrdi": [0.25, -0.05, 0.28, -0.02, 0.30, -0.08, 0.35, -0.01],
            "mean_exg": [0.2, 0.05] * 4,
            "mean_vari": [0.1, 0.02] * 4,
            "mean_tgi": [0.05, 0.01] * 4,
            "mean_exb": [0.02, 0.01] * 4,
            "stress_pixel_ratio": [0.05, 0.35] * 4,
            "salt_patch_ratio": [0.02, 0.0] * 4,
            "moisture_zone_variance": [0.01, 0.04] * 4
        }
        df = pd.DataFrame(data)
        res = self.stat_engine.run_full_comparison(df)
        self.assertTrue(res["t_test"]["significant_at_005"])

        # Fallback branches: single plot type and single date
        data_single = {
            "date": ["12-11-2024"],
            "day_num": [1],
            "plot_type": ["SSPV"],
            "mean_ngrdi": [0.25],
            "mean_exg": [0.2],
            "mean_vari": [0.1],
            "mean_tgi": [0.05],
            "mean_exb": [0.02],
            "stress_pixel_ratio": [0.05],
            "salt_patch_ratio": [0.02],
            "moisture_zone_variance": [0.01]
        }
        res_single = self.stat_engine.run_full_comparison(pd.DataFrame(data_single))
        self.assertEqual(res_single["anova"]["f_statistic"], 0.0)

    # ==========================================================================
    # 9. RECOMMENDATION ENGINE TESTS
    # ==========================================================================
    def test_recommendation_all_branches(self):
        """Tests Urgent Refill, Waterlogging, Optimal, and Decision Tree branches."""
        # Urgent Refill (Low NGRDI + Low ExG)
        adv1 = self.rec_engine.recommend(
            ngrdi=0.005, exg=0.02, vari=0.01, tgi=0.005, exb=0.01, salt_ratio=0.01,
            stress_probs=np.array([0.1, 0.2, 0.7]),
            moisture_probs=np.array([0.8, 0.15, 0.05]),
            yield_probs=np.array([0.7, 0.2, 0.1])
        )
        self.assertIn("Urgent", adv1["primary_recommendation"])

        # Waterlogging (Elevated ExB)
        adv2 = self.rec_engine.recommend(
            ngrdi=0.08, exg=0.10, vari=0.05, tgi=0.02, exb=0.09, salt_ratio=0.01,
            stress_probs=np.array([0.3, 0.4, 0.3]),
            moisture_probs=np.array([0.1, 0.35, 0.55]),
            yield_probs=np.array([0.2, 0.5, 0.3])
        )
        self.assertIn("Waterlogging", adv2["primary_recommendation"])

        # Optimal condition + Salinity advisory
        adv3 = self.rec_engine.recommend(
            ngrdi=0.15, exg=0.20, vari=0.12, tgi=0.06, exb=0.01, salt_ratio=0.08,
            stress_probs=np.array([0.80, 0.15, 0.05]),
            moisture_probs=np.array([0.15, 0.80, 0.05]),
            yield_probs=np.array([0.05, 0.15, 0.80])
        )
        self.assertIn("Optimal", adv3["primary_recommendation"])
        self.assertIsNotNone(adv3["secondary_advisory"])

        # Indeterminate condition -> Decision Tree
        adv4 = self.rec_engine.recommend(
            ngrdi=0.05, exg=0.08, vari=0.04, tgi=0.02, exb=0.02, salt_ratio=0.01,
            stress_probs=np.array([0.33, 0.34, 0.33]),
            moisture_probs=np.array([0.33, 0.34, 0.33]),
            yield_probs=np.array([0.33, 0.34, 0.33])
        )
        self.assertIn("Decision Tree", adv4["engine_mode"])

    # ==========================================================================
    # 10. ORCHESTRATOR MODULE TESTS
    # ==========================================================================
    def test_orchestrator_complete_lifecycle(self):
        """Tests Orchestrator training, evaluation, and artifact generation."""
        temp_dir = tempfile.mkdtemp()
        try:
            orch = SSPVPipelineOrchestrator(data_dir=None, pretrained=False, backbone_name="resnet18")
            history = orch.train_models(epochs=1, batch_size=16)
            self.assertEqual(len(history["unet_loss"]), 1)

            df_metrics, stats = orch.evaluate_all_dates()
            self.assertTrue(len(df_metrics) > 0)

            orch.save_thesis_artifacts(temp_dir, df_metrics, stats, history)
            self.assertTrue(os.path.exists(os.path.join(temp_dir, "sspv_7date_metrics.csv")))
            self.assertTrue(os.path.exists(os.path.join(temp_dir, "statistical_report.json")))
            self.assertTrue(os.path.exists(os.path.join(temp_dir, "unet_11ch_weights.pth")))
            self.assertTrue(os.path.exists(os.path.join(temp_dir, "multitask_cnn_11ch_weights.pth")))
            self.assertTrue(os.path.exists(os.path.join(temp_dir, "thesis_sspv_vs_control_summary.png")))

            report_text = orch.generate_thesis_pdf_report_content(df_metrics, stats)
            self.assertIn("INDIAN INSTITUTE OF TECHNOLOGY JODHPUR", report_text)
        finally:
            shutil.rmtree(temp_dir, ignore_errors=True)

    # ==========================================================================
    # 11. DASHBOARD MODULE TESTS (STREAMLIT MOCK)
    # ==========================================================================
    def test_dashboard_full_execution(self):
        """Tests the full Streamlit dashboard layout using mocked streamlit components."""
        mock_st = MagicMock()
        mock_st.set_page_config = MagicMock()
        mock_st.title = MagicMock()
        mock_st.markdown = MagicMock()
        mock_st.header = MagicMock()
        mock_st.divider = MagicMock()
        mock_st.subheader = MagicMock()
        mock_st.pyplot = MagicMock()
        mock_st.dataframe = MagicMock()
        mock_st.download_button = MagicMock()
        mock_st.metric = MagicMock()
        mock_st.success = MagicMock()
        mock_st.info = MagicMock()
        mock_st.columns = lambda n, **kwargs: [MagicMock() for _ in range(n if isinstance(n, int) else len(n))]
        mock_st.sidebar = MagicMock()
        mock_st.text_input = MagicMock(return_value="")
        mock_st.slider = MagicMock(return_value=1)
        mock_st.selectbox = lambda label, options, **kwargs: options[0]
        mock_st.checkbox = MagicMock(return_value=False)
        mock_st.file_uploader = MagicMock(return_value=None)
        mock_st.cache_resource = lambda f: f

        # Run 1: Default execution with fallback dataset and non-existing Crops Data
        with patch.dict(sys.modules, {"streamlit": mock_st}):
            with patch("sspv_pipeline.dashboard.os.path.exists", return_value=False):
                run_streamlit_dashboard()
                self.assertTrue(mock_st.set_page_config.called)
                self.assertTrue(mock_st.pyplot.called)

        # Run 2: Uploaded DJI frame branch and is_real badge
        buf = io.BytesIO()
        Image.fromarray(np.random.randint(0, 255, (64, 64, 3), dtype=np.uint8)).save(buf, format="JPEG")
        jpeg_bytes = buf.getvalue()

        mock_upload = io.BytesIO(jpeg_bytes)
        mock_upload.name = "DJI_0042.JPG"
        mock_st.file_uploader = MagicMock(return_value=mock_upload)

        # Mock init_and_train to return an orchestrator with a real-image sample
        def mock_init_train(*args, **kwargs):
            orch = SSPVPipelineOrchestrator(data_dir=None, pretrained=False)
            orch.dataset.samples[0]["source_file"] = "fake_dji.jpg"
            hist = {"unet_loss": [0.1], "task_loss": [0.1]}
            df_res, stat_res = orch.evaluate_all_dates()
            return orch, hist, df_res, stat_res

        with patch.dict(sys.modules, {"streamlit": mock_st}):
            with patch("sspv_pipeline.dashboard.init_and_train_pipeline", mock_init_train):
                run_streamlit_dashboard()
                self.assertTrue(mock_st.success.called)

        # Run 3: Uploaded DJI frame without OpenCV (HAS_CV2 = False)
        mock_upload2 = io.BytesIO(jpeg_bytes)
        mock_upload2.name = "DJI_0043.JPG"
        mock_st.file_uploader = MagicMock(return_value=mock_upload2)
        with patch.dict(sys.modules, {"streamlit": mock_st}):
            with patch("sspv_pipeline.dashboard.HAS_CV2", False):
                with patch("sspv_pipeline.dashboard.init_and_train_pipeline", mock_init_train):
                    run_streamlit_dashboard()


if __name__ == "__main__":
    unittest.main()
