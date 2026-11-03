# Module Documentation: `explainability.py`

**File Path:** [`sspv_pipeline/explainability.py`](file:///Users/subhranilbala/Downloads/SSPV-Irrigation-AI-Evaluation/sspv_pipeline/explainability.py)  
**Component:** 11-Channel Multi-Task Grad-CAM Spatial Explainability  
**Presentation Mapping:** Slide 8 (Selvaraju et al., 2017), Slide 10 (Explainability), Slide 17 (Phase 2 XAI)  

---

## 1. Module Overview & Purpose

Deep convolutional neural networks often operate as uninterpretable black boxes. In precision agriculture, agronomists and farmers must verify *why* an AI model flagged a plot as "severely stressed" or "dry".

`explainability.py` implements Gradient-Weighted Class Activation Mapping (Grad-CAM) tailored for **11-channel input representations** across any of the Multi-Task CNN's 3 task heads (`stress_logits`, `moisture_logits`, `yield_logits`).

---

## 2. Mathematical Formulation of Grad-CAM

Let $A^k \in \mathbb{R}^{u \times v}$ denote the activation feature map of channel $k$ in the final convolutional layer of the shared backbone (`layer4` for ResNet18 or `features[-1]` for EfficientNet-B0).

For a target class $c$ in task head $T$ with logit score $y^c$:

### 1. Neuron Importance Weight ($\alpha_k^c$)
The importance weight of feature map $k$ is computed by global average pooling of the gradients:
$$
\alpha_k^c = \frac{1}{Z} \sum_{i=1}^u \sum_{j=1}^v \frac{\partial y^c}{\partial A_{i, j}^k}
$$
where $Z = u \times v$ is the spatial area of the feature map.

### 2. Class Activation Heatmap ($L_{\text{Grad-CAM}}^c$)
The spatial visual attention map is obtained as the rectified linear combination of feature maps weighted by $\alpha_k^c$:
$$
L_{\text{Grad-CAM}}^c = \text{ReLU}\left( \sum_k \alpha_k^c A^k \right)
$$
The $\text{ReLU}$ non-linearity ensures that only features with a positive influence on the target class are highlighted.

### 3. Upsampling and Normalization
The resulting coarse heatmap is resized to the original patch size $(H, W)$ via bilinear interpolation and normalized to $[0, 1]$:
$$
\text{Heatmap}_{\text{norm}} = \frac{L - \min(L)}{\max(L) - \min(L) + \epsilon}
$$

---

## 3. Class Structure: `MultiTaskGradCAM`

### Hook Registration
In `__init__(model)`:
- `target_layer.register_forward_hook(_save_activation)`
- `target_layer.register_full_backward_hook(_save_gradient)`

### Core Method: `generate_heatmap`
```python
def generate_heatmap(
    self,
    input_tensor: torch.Tensor,
    task: str = "stress_logits",
    target_class: Optional[int] = None
) -> np.ndarray
```
- `input_tensor`: `(1, 11, H, W)` PyTorch float tensor.
- `task`: One of `["stress_logits", "moisture_logits", "yield_logits"]`.
- `target_class`: If `None`, automatically targets the argmax predicted class.
- **Returns:** Normalized 2D NumPy array $(H, W)$ in $[0, 1]$.

---

## 4. Example Usage

```python
import matplotlib.pyplot as plt
from sspv_pipeline.models import MultiTaskSSPVNet
from sspv_pipeline.explainability import MultiTaskGradCAM

model = MultiTaskSSPVNet(in_channels=11, backbone_name="resnet18")
gradcam = MultiTaskGradCAM(model)

# Generate attention heatmap for severe crop stress
heatmap = gradcam.generate_heatmap(x_tensor, task="stress_logits", target_class=2)

plt.imshow(rgb_image)
plt.imshow(heatmap, cmap="jet", alpha=0.5)
plt.title("Grad-CAM: Severe Stress Focus")
plt.show()
```
