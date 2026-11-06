"""
MODULE 4: Deep Learning Architectures (11-Ch U-Net & Multi-Task CNN).

Implements Slide 10 & Slide 13:
- 11-Channel U-Net Segmentation with ResNet18 Encoder & Skip Connections
- 11-Channel Multi-Task CNN supporting BOTH ResNet18 & EfficientNet-B0
- Dynamic Inverse-Frequency Weighted Cross-Entropy Loss
"""

from typing import Dict, Tuple, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models


# ==============================================================================
# BACKBONE ADAPTERS (3-ch -> 11-ch)
# ==============================================================================

def adapt_resnet_first_conv_to_11ch(resnet_model: nn.Module, in_channels: int = 11) -> nn.Module:
    """
    Adapts ResNet18's `conv1` from 3 channels to 11 channels (Slide 13).
    Copies ImageNet weights to the first 3 channels (R, G, B) and initializes
    the 8 vegetation index channels with the mean across RGB weights.
    """
    old_conv = resnet_model.conv1
    new_conv = nn.Conv2d(
        in_channels=in_channels,
        out_channels=old_conv.out_channels,
        kernel_size=old_conv.kernel_size,
        stride=old_conv.stride,
        padding=old_conv.padding,
        bias=old_conv.bias is not None
    )
    with torch.no_grad():
        new_conv.weight[:, :3, :, :] = old_conv.weight.clone()
        rgb_mean_weight = old_conv.weight.mean(dim=1, keepdim=True)
        new_conv.weight[:, 3:, :, :] = rgb_mean_weight.repeat(1, in_channels - 3, 1, 1) * 0.5
        if old_conv.bias is not None:
            new_conv.bias = nn.Parameter(old_conv.bias.clone())

    resnet_model.conv1 = new_conv
    return resnet_model


def adapt_efficientnet_first_conv_to_11ch(effnet_model: nn.Module, in_channels: int = 11) -> nn.Module:
    """
    Adapts EfficientNet-B0's first Conv2d layer (`features[0][0]`) from 3 channels
    to 11 channels (Slide 10 & Slide 13 Shared Backbone alternative).
    """
    old_conv = effnet_model.features[0][0]
    new_conv = nn.Conv2d(
        in_channels=in_channels,
        out_channels=old_conv.out_channels,
        kernel_size=old_conv.kernel_size,
        stride=old_conv.stride,
        padding=old_conv.padding,
        bias=old_conv.bias is not None
    )
    with torch.no_grad():
        new_conv.weight[:, :3, :, :] = old_conv.weight.clone()
        rgb_mean_weight = old_conv.weight.mean(dim=1, keepdim=True)
        new_conv.weight[:, 3:, :, :] = rgb_mean_weight.repeat(1, in_channels - 3, 1, 1) * 0.5
        if old_conv.bias is not None:
            new_conv.bias = nn.Parameter(old_conv.bias.clone())

    effnet_model.features[0][0] = new_conv
    return effnet_model


# ==============================================================================
# U-NET DECODER BUILDING BLOCK
# ==============================================================================

class ConvReluBlock(nn.Module):
    """Standard Double Convolution Block for U-Net Decoder."""
    def __init__(self, in_ch: int, out_ch: int):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


# ==============================================================================
# U-NET SEGMENTATION NETWORK
# ==============================================================================

class UNetResNet18_11Ch(nn.Module):
    """
    Implements Slide 13 U-Net Segmentation Architecture:
    - Input: 11-Channel Tensor (RGB + 8 Vegetation Indices)
    - Encoder: ResNet18 with 1st conv adapted for 11 channels
    - Decoder: Bilinear Upsampling + Skip Connections
    - Output: 4 classes (0: Bare Soil, 1: Healthy Crop, 2: Stressed Crop, 3: Salt Patch)
    """

    def __init__(self, in_channels: int = 11, num_classes: int = 4, pretrained: bool = True):
        super().__init__()
        try:
            weights = models.ResNet18_Weights.DEFAULT if pretrained else None
            base = models.resnet18(weights=weights)
        except Exception:
            base = models.resnet18(pretrained=False)

        base = adapt_resnet_first_conv_to_11ch(base, in_channels=in_channels)

        # Encoder layers
        self.enc0 = nn.Sequential(base.conv1, base.bn1, base.relu)  # 64 ch, H/2
        self.pool0 = base.maxpool                                    # 64 ch, H/4
        self.enc1 = base.layer1                                      # 64 ch, H/4
        self.enc2 = base.layer2                                      # 128 ch, H/8
        self.enc3 = base.layer3                                      # 256 ch, H/16
        self.enc4 = base.layer4                                      # 512 ch, H/32

        # Decoder layers with skip connections
        self.up4 = ConvReluBlock(512 + 256, 256)
        self.up3 = ConvReluBlock(256 + 128, 128)
        self.up2 = ConvReluBlock(128 + 64, 64)
        self.up1 = ConvReluBlock(64 + 64, 64)
        self.final_up = ConvReluBlock(64, 32)
        self.out_head = nn.Conv2d(32, num_classes, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        input_size = x.shape[-2:]

        e0 = self.enc0(x)               # (B, 64, H/2, W/2)
        e1 = self.enc1(self.pool0(e0))  # (B, 64, H/4, W/4)
        e2 = self.enc2(e1)              # (B, 128, H/8, W/8)
        e3 = self.enc3(e2)              # (B, 256, H/16, W/16)
        e4 = self.enc4(e3)              # (B, 512, H/32, W/32)

        d4 = F.interpolate(e4, size=e3.shape[-2:], mode="bilinear", align_corners=False)
        d4 = self.up4(torch.cat([d4, e3], dim=1))

        d3 = F.interpolate(d4, size=e2.shape[-2:], mode="bilinear", align_corners=False)
        d3 = self.up3(torch.cat([d3, e2], dim=1))

        d2 = F.interpolate(d3, size=e1.shape[-2:], mode="bilinear", align_corners=False)
        d2 = self.up2(torch.cat([d2, e1], dim=1))

        d1 = F.interpolate(d2, size=e0.shape[-2:], mode="bilinear", align_corners=False)
        d1 = self.up1(torch.cat([d1, e0], dim=1))

        d0 = F.interpolate(d1, size=input_size, mode="bilinear", align_corners=False)
        d0 = self.final_up(d0)

        logits = self.out_head(d0)
        return logits


# ==============================================================================
# MULTI-TASK CLASSIFICATION NETWORK
# ==============================================================================

class MultiTaskSSPVNet(nn.Module):
    """
    Implements Slide 10 & Slide 13 Multi-Task CNN Architecture:
    - Shared Backbone: Supports BOTH `ResNet18` (512-d) and `EfficientNet-B0` (1280-d)
      adapted for 11-channel input (`RGB + 8 vegetation indices`).
    - Crop-Gated Feature Masking: Gates analysis to valid crop/mound pixels using U-Net mask
    - Task Head 1: Crop Stress Classification (Healthy / Mild / Severe)
    - Task Head 2: Soil Moisture Zone (Dry / Optimal / Wet)
    - Task Head 3: Yield Potential Estimation (Low / Medium / High)
    """

    def __init__(
        self,
        in_channels: int = 11,
        pretrained: bool = True,
        backbone_name: str = "resnet18"
    ):
        super().__init__()
        self.backbone_name = backbone_name.lower()

        if self.backbone_name == "efficientnet_b0":
            try:
                weights = models.EfficientNet_B0_Weights.DEFAULT if pretrained else None
                backbone = models.efficientnet_b0(weights=weights)
            except Exception:
                backbone = models.efficientnet_b0(pretrained=False)
            backbone = adapt_efficientnet_first_conv_to_11ch(backbone, in_channels=in_channels)
            self.features_stem = backbone.features[:-1]
            self.target_conv_layer = backbone.features[-1]  # Grad-CAM target (1280 ch)
            feat_dim = 1280
        else:
            try:
                weights = models.ResNet18_Weights.DEFAULT if pretrained else None
                backbone = models.resnet18(weights=weights)
            except Exception:
                backbone = models.resnet18(pretrained=False)
            backbone = adapt_resnet_first_conv_to_11ch(backbone, in_channels=in_channels)
            self.features_stem = nn.Sequential(
                backbone.conv1, backbone.bn1, backbone.relu, backbone.maxpool,
                backbone.layer1, backbone.layer2, backbone.layer3
            )
            self.target_conv_layer = backbone.layer4  # Grad-CAM target (512 ch)
            feat_dim = 512

        self.pool = nn.AdaptiveAvgPool2d((1, 1))

        # Task-specific classification heads
        self.stress_head = nn.Sequential(
            nn.Linear(feat_dim, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(0.25),
            nn.Linear(128, 3)
        )
        self.moisture_head = nn.Sequential(
            nn.Linear(feat_dim, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(0.25),
            nn.Linear(128, 3)
        )
        self.yield_head = nn.Sequential(
            nn.Linear(feat_dim, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(0.25),
            nn.Linear(128, 3)
        )

    def forward(
        self, x: torch.Tensor, valid_crop_mask: Optional[torch.Tensor] = None
    ) -> Dict[str, torch.Tensor]:
        """
        If `valid_crop_mask` (B, 1, H, W) is provided from U-Net, background pixels are
        soft-gated (weighted down by 0.25) so the CNN focuses on valid vegetation & SSPV zones.
        """
        if valid_crop_mask is not None:
            gate = 0.25 + 0.75 * valid_crop_mask.float()
            x = x * gate

        feat = self.features_stem(x)
        spatial_feat = self.target_conv_layer(feat)
        pooled = torch.flatten(self.pool(spatial_feat), 1)

        return {
            "stress_logits": self.stress_head(pooled),
            "moisture_logits": self.moisture_head(pooled),
            "yield_logits": self.yield_head(pooled),
            "spatial_features": spatial_feat
        }


# ==============================================================================
# COMBINED MULTI-TASK LOSS
# ==============================================================================

class MultiTaskLoss(nn.Module):
    """
    Implements Slide 10 & Slide 13 Combined Loss:
    L = L_stress + L_moisture + L_yield (Weighted Cross-Entropy per task head)
    Supports distinct inverse-frequency class weight tensors per task head.
    """

    def __init__(
        self,
        w_stress: float = 1.0,
        w_moisture: float = 1.0,
        w_yield: float = 1.0,
        stress_class_weights: Optional[torch.Tensor] = None,
        moisture_class_weights: Optional[torch.Tensor] = None,
        yield_class_weights: Optional[torch.Tensor] = None,
    ):
        super().__init__()
        self.w_stress = w_stress
        self.w_moisture = w_moisture
        self.w_yield = w_yield
        self.ce_stress = nn.CrossEntropyLoss(weight=stress_class_weights)
        self.ce_moisture = nn.CrossEntropyLoss(weight=moisture_class_weights)
        self.ce_yield = nn.CrossEntropyLoss(weight=yield_class_weights)

    def forward(
        self,
        outputs: Dict[str, torch.Tensor],
        target_stress: torch.Tensor,
        target_moisture: torch.Tensor,
        target_yield: torch.Tensor
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        l_stress = self.ce_stress(outputs["stress_logits"], target_stress)
        l_moisture = self.ce_moisture(outputs["moisture_logits"], target_moisture)
        l_yield = self.ce_yield(outputs["yield_logits"], target_yield)

        total_loss = (
            self.w_stress * l_stress
            + self.w_moisture * l_moisture
            + self.w_yield * l_yield
        )
        metrics = {
            "loss_total": float(total_loss.item()),
            "loss_stress": float(l_stress.item()),
            "loss_moisture": float(l_moisture.item()),
            "loss_yield": float(l_yield.item()),
        }
        return total_loss, metrics
