"""
MODULE 06: Grad-CAM Explainability for 11-Channel Multi-Task CNN.

Implements Selvaraju et al. (2017) Grad-CAM on the final convolutional block
of the Multi-Task CNN (supports both ResNet18 and EfficientNet-B0).
"""

from typing import Optional

import numpy as np
import torch
import torch.nn.functional as F
from scipy.ndimage import zoom

from .config import HAS_CV2
from .models import MultiTaskSSPVNet

if HAS_CV2:
    import cv2


class MultiTaskGradCAM:
    """
    Generates spatial attention heatmaps per task head (Slide 8, 10, 17).
    """

    def __init__(self, model: MultiTaskSSPVNet):
        self.model = model
        self.gradients: Optional[torch.Tensor] = None
        self.activations: Optional[torch.Tensor] = None

        self.target_layer = self.model.target_conv_layer
        self.target_layer.register_forward_hook(self._save_activation)
        self.target_layer.register_full_backward_hook(self._save_gradient)

    def _save_activation(self, module, input, output):
        self.activations = output.detach()

    def _save_gradient(self, module, grad_input, grad_output):
        self.gradients = grad_output[0].detach()

    def generate_heatmap(
        self,
        input_tensor: torch.Tensor,
        task: str = "stress_logits",
        target_class: Optional[int] = None
    ) -> np.ndarray:
        """
        Generates a normalized Grad-CAM spatial attention heatmap in [0, 1]
        resized to the input image dimensions (H, W).
        """
        self.model.eval()
        self.model.zero_grad()

        x = input_tensor.clone().detach().requires_grad_(True)
        outputs = self.model(x)
        logits = outputs[task]

        if target_class is None:
            target_class = int(torch.argmax(logits, dim=1).item())

        score = logits[0, target_class]
        score.backward()

        if self.gradients is None or self.activations is None:
            return np.zeros(input_tensor.shape[-2:], dtype=np.float32)

        # Global average pooling of gradients across spatial dimensions
        weights = torch.mean(self.gradients, dim=(2, 3), keepdim=True)  # (1, C, 1, 1)
        cam = torch.sum(weights * self.activations, dim=1).squeeze(0)   # (h, w)
        cam = F.relu(cam)

        cam_np = cam.cpu().numpy()
        target_h, target_w = input_tensor.shape[2], input_tensor.shape[3]
        if HAS_CV2:
            cam_np = cv2.resize(cam_np, (target_w, target_h))
        else:
            zoom_y = target_h / max(1, cam_np.shape[0])
            zoom_x = target_w / max(1, cam_np.shape[1])
            cam_np = zoom(cam_np, (zoom_y, zoom_x), order=1)
        cam_min, cam_max = cam_np.min(), cam_np.max()
        if cam_max - cam_min > 1e-6:
            cam_np = (cam_np - cam_min) / (cam_max - cam_min)
        else:
            cam_np = np.zeros_like(cam_np)
        return cam_np
