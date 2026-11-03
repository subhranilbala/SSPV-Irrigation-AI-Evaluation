# Module Documentation: `models.py`

**File Path:** [`sspv_pipeline/models.py`](file:///Users/subhranilbala/Downloads/SSPV-Irrigation-AI-Evaluation/sspv_pipeline/models.py)  
**Component:** Deep Learning Architectures & Multi-Task Loss  
**Presentation Mapping:** Slide 10 (Tech Stack) & Slide 13 (AI Architecture: U-Net & Multi-Task CNN)  

---

## 1. Module Overview & Purpose

`models.py` implements the core PyTorch neural network architectures powering the pipeline:
1. **11-Channel First-Convolution Weight Adapters:** Enables ImageNet pretrained backbones (`ResNet18` and `EfficientNet-B0`) to ingest 11-channel inputs without losing learned edge/texture representations.
2. **11-Channel U-Net Semantic Segmentation:** Encoder-decoder network with skip connections outputting 4-class masks.
3. **11-Channel Multi-Task CNN (`MultiTaskSSPVNet`):** Shared feature backbone with crop-gated spatial attention driving 3 simultaneous classification heads.
4. **Dynamic Weighted Cross-Entropy Loss (`MultiTaskLoss`):** Joint multi-task objective with inverse-frequency class weight balancing.

---

## 2. Mathematical & Architectural Details

### 2.1 First-Convolution 11-Channel Adaptation

Standard pretrained computer vision models require 3-channel RGB inputs. When given an 11-channel tensor ($R, G, B + 8 \text{ indices}$):
```python
new_conv.weight[:, :3, :, :] = old_conv.weight.clone()
rgb_mean_weight = old_conv.weight.mean(dim=1, keepdim=True)
new_conv.weight[:, 3:, :, :] = rgb_mean_weight.repeat(1, in_channels - 3, 1, 1) * 0.5
```
- The first 3 channels retain exact ImageNet RGB spatial weights.
- The 8 spectral index channels are initialized with the mean across RGB weights scaled by $0.5$, allowing smooth gradient propagation from step 0.

Functions:
- `adapt_resnet_first_conv_to_11ch()`
- `adapt_efficientnet_first_conv_to_11ch()`

---

### 2.2 11-Channel U-Net (`UNetResNet18_11Ch`)

#### Architecture:
- **Encoder:** ResNet18 layers:
  - `enc0`: `conv1` (adapted to 11 channels), `bn1`, `relu` $\to 64$ ch, $H/2$
  - `enc1`: `layer1` $\to 64$ ch, $H/4$
  - `enc2`: `layer2` $\to 128$ ch, $H/8$
  - `enc3`: `layer3` $\to 256$ ch, $H/16$
  - `enc4`: `layer4` $\to 512$ ch, $H/32$
- **Decoder:** Bilinear upsampling + Skip connection concatenation + `ConvReluBlock`:
  - `up4`: $(512 + 256) \to 256$ ch
  - `up3`: $(256 + 128) \to 128$ ch
  - `up2`: $(128 + 64) \to 64$ ch
  - `up1`: $(64 + 64) \to 64$ ch
  - `final_up`: $64 \to 32$ ch
- **Head:** $1\times1$ Conv $\to 4$ output class logits.

---

### 2.3 Multi-Task CNN (`MultiTaskSSPVNet`)

#### Shared Backbone:
Configurable via `backbone_name`:
- `"resnet18"`: Output feature dimension $d = 512$, target layer for Grad-CAM is `layer4`.
- `"efficientnet_b0"`: Output feature dimension $d = 1280$, target layer for Grad-CAM is `features[-1]`.

#### Crop-Gating Mechanism:
Slide 13 specifies: *"Gates all downstream analysis to valid crop pixels only"*.
If a `valid_crop_mask` $\in \{0, 1\}^{B \times 1 \times H \times W}$ is passed from U-Net predictions:
$$
\mathbf{X}_{\text{gated}} = \mathbf{X} \odot (0.25 + 0.75 \cdot \mathbf{M}_{\text{crop}})
$$
This attenuates dry soil background pixels by 75% while maintaining full gradient flow through crop canopies.

#### Task Heads:
All 3 heads follow the structure:
$$\text{Linear}(d \to 128) \to \text{ReLU} \to \text{Dropout}(0.25) \to \text{Linear}(128 \to 3)$$
Outputs:
1. `stress_logits` (3 classes: Healthy, Mild, Severe)
2. `moisture_logits` (3 classes: Dry, Optimal, Wet)
3. `yield_logits` (3 classes: Low, Medium, High)

---

### 2.4 Multi-Task Loss (`MultiTaskLoss`)

Implements Slide 10 & 13:
$$
\mathcal{L} = w_{\text{stress}} \mathcal{L}_{\text{stress}} + w_{\text{moist}} \mathcal{L}_{\text{moist}} + w_{\text{yield}} \mathcal{L}_{\text{yield}}
$$
where each individual loss is class-weighted cross-entropy:
$$\mathcal{L}_k = -\sum_{c=1}^3 w_{k, c} y_c \log \hat{p}_c$$

---

## 3. Example Usage

```python
import torch
from sspv_pipeline.models import UNetResNet18_11Ch, MultiTaskSSPVNet

x = torch.randn(2, 11, 256, 256)

# U-Net Segmentation
unet = UNetResNet18_11Ch(in_channels=11, num_classes=4, pretrained=False)
seg_logits = unet(x)  # (2, 4, 256, 256)

# Multi-Task CNN with valid crop gating
mt_cnn = MultiTaskSSPVNet(in_channels=11, backbone_name="resnet18", pretrained=False)
valid_gate = (torch.argmax(seg_logits, dim=1) == 1).unsqueeze(1)
out = mt_cnn(x, valid_crop_mask=valid_gate)

print("Stress logits shape:", out["stress_logits"].shape)  # (2, 3)
```
