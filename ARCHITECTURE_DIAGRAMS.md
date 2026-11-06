# 🏛️ SSPV Irrigation AI Evaluation — Architecture Diagrams Specification

**Project:** AI Based Evaluation of Empirical Subsurface Porous Vessel (SSPV) Irrigation System Performance Using Images  
**Student:** Subhranil Bala (Roll No.: M25AI2038) | **Supervisor:** Dr. Anand Krishnan Plappally  
**Institute:** Indian Institute of Technology Jodhpur (School of AI & Data Science)

---

## 1. End-to-End System Architecture

This diagram illustrates the complete dataflow from raw UAV imagery (or benchmark simulator) to final thesis artifacts and Streamlit dashboard.

> 📐 **High-Resolution Vector Architecture SVG:** [`Architectures/Complete End-to-End System Architecture.svg`](Architectures/Complete%20End-to-End%20System%20Architecture.svg)

```mermaid
flowchart TD
    subgraph Data_Acquisition["1. Data Ingestion & Synthesis"]
        RAW["DJI Zenmuse P1 Imagery\n(8192×5460 GeoTIFF / JPG)"]
        SIM["7-Date Arid Soil Field Simulator\n(Rajasthan Soil Profile + Salt Halos)"]
        STRIDE["Uniform Flight Striding (np.linspace)\nSelects N evenly spaced flight frames"]
        RAW --> STRIDE
    end

    subgraph Preprocessing_Module["2. UAV Preprocessing Pipeline"]
        S1["Step 1: Inverse sRGB Gamma\n(Piecewise Linear / Power 2.4)"]
        S2["Step 2: Dark Object Subtraction (DOS)\n(Atmospheric Correction)"]
        S3["Step 3: SIFT Co-Registration\n(2500 Keypoints + Lowe's Ratio 0.75 + RANSAC Affine)"]
        S4["Step 4: ROI Cropping & Median Filtering\n(Per-channel cv2/scipy median blur)"]
        STRIDE --> S1
        S1 --> S2 --> S3 --> S4
    end

    subgraph Feature_Engineering["3. Feature Engineering"]
        VIs["Compute 8 Normalized Indices:\nNGRDI, ExG, ExR, VARI, MGRVI, NPCI, TGI, ExB"]
        TENSOR["Assemble 11-Channel Tensor\nShape: (B, 11, H, W) float32\n[R, G, B, 8 Spectral Indices]"]
        S4 --> VIs
        SIM --> VIs
        VIs --> TENSOR
    end

    subgraph Pseudo_Labeling["4. Biophysical Pseudo-Labeling"]
        PSEUDO_SEG["4-Class Semantic Segmentation Mask\n0: Soil | 1: Healthy | 2: Stressed | 3: Salt Patch"]
        PSEUDO_CLS["3 Patch-Level Multi-Task Labels\nStress (3) | Moisture (3) | Yield (3)"]
        TENSOR --> PSEUDO_SEG
        TENSOR --> PSEUDO_CLS
    end

    subgraph Deep_Learning["5. Deep Learning Framework"]
        UNET["11-Channel U-Net\n(ResNet18 Encoder + 4 Skip Connections)"]
        GATE["Crop Canopy Soft Feature Gate\nX_gated = X ⊙ (0.25 + 0.75 · CropMask)"]
        CNN["11-Channel Multi-Task CNN\n(ResNet18 or EfficientNet-B0 Backbone)"]
        LOSS["MultiTaskLoss\n(Inverse-Frequency Weighted Cross Entropy)"]

        TENSOR --> UNET
        UNET --> PSEUDO_SEG
        UNET -.->|Canopy Prediction| GATE
        TENSOR --> GATE
        GATE --> CNN
        CNN --> LOSS
        PSEUDO_CLS -.-> LOSS
    end

    subgraph Explainability_Stats["6. Explainability & Statistical Engines"]
        GCAM["Multi-Task Grad-CAM\nPer-Head Attention Heatmaps\n(Target Conv: layer4 / features[-1])"]
        STATS["Statistical Comparison Engine\n• Welch's Two-Sample t-test\n• One-Way ANOVAs (Plot & Date)\n• Cohen's d Effect Sizes\n• Temporal Linear Regression\n• Slope Comparison Z-test"]
        CNN --> GCAM
        CNN --> STATS
    end

    subgraph Decision_Support["7. Decision Support & Outputs"]
        REC["Hybrid Recommendation Engine\nTier 1: Deterministic Agronomic Rules\nTier 2: 13-Dim Decision Tree (Trained on 600 domain rules)\nSecondary: Salinity Advisory (salt_ratio > 0.05)"]
        UI["Interactive Streamlit Dashboard\n• 4 KPI Cards (NGRDI, p-value, Cohen's d, ExG variance)\n• 4-Panel Spatial Inspector (RGB, NGRDI, Seg, Grad-CAM)\n• Multi-Date Temporal Growth Curves"]
        ARTIFACTS["Exported Thesis Deliverables\n• sspv_7date_metrics.csv (18 columns)\n• statistical_report.json\n• PyTorch Weights (.pth)\n• 4-Panel PNG Figures\n• Thesis Chapter Report (.txt)"]

        STATS --> REC
        CNN --> REC
        STATS --> UI
        REC --> UI
        GCAM --> UI
        STATS --> ARTIFACTS
        REC --> ARTIFACTS
    end
```

---

## 2. Module-Wise Architecture Diagrams

### Module 1: Preprocessing Pipeline (`preprocessing.py`)

```mermaid
flowchart LR
    A["Raw Drone Image\n(uint8 or uint16 GeoTIFF/JPG)"] --> B["1. Normalize & sRGB Inverse Gamma\nlinear = ((norm+0.055)/1.055)^2.4"]
    B --> C["2. Dark Object Subtraction (DOS)\nρ_corr = max(0, ρ - P1.0) / (P99.9 + ε)"]
    C --> D["3. SIFT Co-Registration\nSIFT Keypoints → Lowe's 0.75 Ratio → RANSAC Affine"]
    D --> E["4. Median Noise Filter & ROI\nscipy.ndimage.median_filter (kernel=3)"]
    E --> F["Preprocessed Surface Reflectance\nShape: (H, W, 3) in [0.0, 1.0]"]
```

---

### Module 2: Feature Engineering (`feature_engineering.py`)

```mermaid
flowchart TD
    RGB["Surface Reflectance (R, G, B)"] --> NGRDI["NGRDI = (G - R) / (G + R + ε)"]
    RGB --> ExG["ExG = (2G - B - R) / (R + G + B + ε)"]
    RGB --> ExR["ExR = (1.4R - G) / (R + G + B + ε)"]
    RGB --> VARI["VARI = (G - R) / (G + R - B)"]
    RGB --> MGRVI["MGRVI = (G² - R²) / (G² + R² + ε)"]
    RGB --> NPCI["NPCI = (B - R) / (B + R + ε)"]
    RGB --> TGI["TGI = G - 0.39R - 0.61B"]
    RGB --> ExB["ExB = 1.4B - G"]

    RGB --> TENSOR["Concatenate along Channel Dimension"]
    NGRDI --> TENSOR
    ExG --> TENSOR
    ExR --> TENSOR
    VARI --> TENSOR
    MGRVI --> TENSOR
    NPCI --> TENSOR
    TGI --> TENSOR
    ExB --> TENSOR

    TENSOR --> OUT["11-Channel Tensor: (11, H, W)\n[R, G, B, NGRDI, ExG, ExR, VARI, MGRVI, NPCI, TGI, ExB]"]
```

---

### Module 3: Biophysical Weakly-Supervised Pseudo-Labeling (`pseudo_labeling.py`)

```mermaid
flowchart TD
    IN["Inputs: RGB Reflectance + 8 Indices"] --> COND_SALT{"Salt Halo Condition\nLuminance > P92 AND\nRGB_std < 0.06 AND\nExG < 0.05"}
    
    COND_SALT -- Yes --> C3["Class 3: Salt Patch (SSPV Mound)"]
    COND_SALT -- No --> COND_VEG{"Vegetation Candidate\nExG > max(0.06, P65) OR\nNGRDI > 0.04"}
    
    COND_VEG -- No --> C0["Class 0: Bare Arid Soil"]
    COND_VEG -- Yes --> COND_HLT{"Canopy Health\nNGRDI ≥ 0.05 AND\nTGI ≥ 0.01"}
    
    COND_HLT -- Yes --> C1["Class 1: Healthy Canopy"]
    COND_HLT -- No --> C2["Class 2: Stressed Crop Zone"]

    subgraph Patch_Level_Heuristics["Multi-Task Patch Labels"]
        P_STR["Stress Level (0: Healthy, 1: Mild, 2: Severe)\nDriven by Crop NGRDI & ExR"]
        P_MOI["Moisture Zone (0: Dry, 1: Optimal, 2: Wet)\nDriven by Soil Luminance & ExB"]
        P_YLD["Yield Potential (0: Low, 1: Med, 2: High)\nScore = 0.5·CropFrac + 2.5·NGRDI + 2.0·TGI"]
    end

    C1 & C2 --> Patch_Level_Heuristics
```

---

### Module 4: Neural Architectures (`models.py`)

#### A. 11-Channel U-Net Field Segmentation (`UNetResNet18_11Ch`)

> 📐 **High-Resolution Vector Architecture SVG:** [`Architectures/11-Channel U-Net Field Segmentation.svg`](Architectures/11-Channel%20U-Net%20Field%20Segmentation.svg)

```mermaid
flowchart TD
    IN11["Input: (B, 11, H, W)"] --> ADAPT["Adapted Conv1 (11→64, 7×7, s=2)\nRGB weights cloned, indices initialized to 0.5×mean(RGB)"]
    
    subgraph ResNet18_Encoder["Encoder Stages"]
        ADAPT --> ENC0["enc0 (64ch, H/2)"]
        ENC0 --> ENC1["enc1 / layer1 (64ch, H/4)"]
        ENC1 --> ENC2["enc2 / layer2 (128ch, H/8)"]
        ENC2 --> ENC3["enc3 / layer3 (256ch, H/16)"]
        ENC3 --> ENC4["enc4 / layer4 (512ch, H/32)"]
    end

    subgraph Decoder_With_Skips["Decoder Stages"]
        ENC4 --> UP4["up4: Bilinear (512) + Skip enc3 (256) → ConvReluBlock (256ch)"]
        UP4 --> UP3["up3: Bilinear (256) + Skip enc2 (128) → ConvReluBlock (128ch)"]
        UP3 --> UP2["up2: Bilinear (128) + Skip enc1 (64)  → ConvReluBlock (64ch)"]
        UP2 --> UP1["up1: Bilinear (64)  + Skip enc0 (64)  → ConvReluBlock (64ch)"]
        UP1 --> FINAL_UP["final_up: Bilinear → ConvReluBlock (32ch)"]
    end

    ENC3 -.->|Skip 256ch| UP4
    ENC2 -.->|Skip 128ch| UP3
    ENC1 -.->|Skip 64ch| UP2
    ENC0 -.->|Skip 64ch| UP1

    FINAL_UP --> OUT_HEAD["Conv2d(32→4, 1×1) + Interpolate to (H,W)"]
    OUT_HEAD --> MASK["Segmentation Logits: (B, 4, H, W)\n[Soil, Healthy, Stressed, Salt]"]
```

#### B. 11-Channel Multi-Task CNN (`MultiTaskSSPVNet`)

> 📐 **High-Resolution Vector Architecture SVG:** [`Architectures/11-Channel Multi-Task CNN with Crop Gating.svg`](Architectures/11-Channel%20Multi-Task%20CNN%20with%20Crop%20Gating.svg)

```mermaid
flowchart TD
    TENSOR["11-Channel Input Tensor: (B, 11, H, W)"] --> STEM["Shared Backbone Features Stem\nResNet18 or EfficientNet-B0 (Adapted Conv1)"]
    STEM --> TARGET_CONV["target_conv_layer\n(ResNet18 layer4 / EffNet features[-1])"]
    
    CROP_MASK["U-Net Canopy Mask (Classes 1 & 2)"] --> GATE["Soft Spatial Gate:\nGate = 0.25 + 0.75 · CropMask"]
    TARGET_CONV --> APPLY_GATE["X_gated = Target_Conv ⊙ Gate"]
    GATE --> APPLY_GATE
    
    APPLY_GATE --> GAP["AdaptiveAvgPool2d(1, 1) → Flatten (512 or 1280)"]

    subgraph Heads["Task Classification Heads"]
        GAP --> H_STR["Stress Head\nFC(dim→128) → ReLU → Dropout(0.25) → FC(128→3)"]
        GAP --> H_MOI["Moisture Head\nFC(dim→128) → ReLU → Dropout(0.25) → FC(128→3)"]
        GAP --> H_YLD["Yield Head\nFC(dim→128) → ReLU → Dropout(0.25) → FC(128→3)"]
    end

    H_STR --> OUT_S["Stress Logits [Healthy, Mild, Severe]"]
    H_MOI --> OUT_M["Moisture Logits [Dry, Optimal, Wet]"]
    H_YLD --> OUT_Y["Yield Logits [Low, Medium, High]"]
```

---

### Module 5: Explainability Engine (`explainability.py`)

```mermaid
flowchart LR
    INPUT["11-Channel Patch"] --> FWD["Forward Pass"]
    FWD --> LOGITS["Task Head Logits"]
    LOGITS --> BACK["Backpropagate Target Class Score\n(e.g., Severe Stress class)"]
    
    subgraph Hooks["PyTorch Hook Caches"]
        FWD -.->|Forward Hook| ACT["Cached Feature Activations A^k"]
        BACK -.->|Backward Hook| GRAD["Cached Feature Gradients ∂y_c / ∂A^k"]
    end

    ACT & GRAD --> GAP["Global Average Pool Gradients:\nα_k = (1/Z) Σ_i Σ_j (∂y_c / ∂A^k_ij)"]
    GAP --> LIN_COMB["Linear Combination:\nL_CAM = ReLU( Σ_k α_k · A^k )"]
    LIN_COMB --> RESIZE["Upsample to Patch Resolution & Min-Max Normalize"]
    RESIZE --> OVERLAY["Jet Colormap + Alpha Blend (0.45) on RGB\nExplainability Heatmap"]
```

---

### Module 6: Statistical Comparison Engine (`statistics_engine.py`)

```mermaid
flowchart TD
    DF["7-Date Multi-Plot Metrics DataFrame\n(SSPV vs Control across Survey Dates)"] --> TESTS
    
    subgraph TESTS["Parametric, Non-Parametric & Regression Tests"]
        T_TEST["Welch's Two-Sample t-test\nt = (x̄_sspv - x̄_ctrl) / √(s²_1/n_1 + s²_2/n_2)\np-value on mean NGRDI"]
        ANOVA_P["One-Way ANOVA (Plot Type)\nF-statistic on Stress Pixel Ratio"]
        ANOVA_D["One-Way ANOVA (Date Effect)\nF-statistic across 7 Survey Dates"]
        COHEN["Cohen's d Effect Size\nd = (x̄_1 - x̄_2) / s_pooled\n(Negligible / Small / Medium / Large)"]
        REGRESS["Temporal Growth Trends\nLinear Regression: NGRDI = slope · Day + intercept"]
        SLOPE_Z["Inter-Plot Slope Z-Test\nZ = (b_sspv - b_ctrl) / √(SE²_sspv + SE²_ctrl)"]
    end

    TESTS --> OUT_JSON["Statistical Report (JSON / LaTeX Format)"]
```

---

### Module 7: Hybrid Recommendation Engine (`recommendation.py`)

> 📐 **High-Resolution Vector Architecture SVG:** [`Architectures/Hybrid Recommendation Engine Architecture.svg`](Architectures/Hybrid%20Recommendation%20Engine%20Architecture.svg)

```mermaid
flowchart TD
    IN["Inputs: VIs (NGRDI, ExG, VARI, TGI, ExB) + CNN Probs + Salt Ratio"] --> CHECK_T1{"Tier 1: Determinate Agronomic Rules"}
    
    CHECK_T1 -- "NGRDI < 0.01 AND ExG < 0.05" --> R1["Urgent Refill Alert\n(Confidence: 0.94)"]
    CHECK_T1 -- "ExB > 0.08 AND P(Wet) > 0.45" --> R2["Waterlogging Warning\n(Confidence: 0.91)"]
    CHECK_T1 -- "NGRDI > 0.09 AND VARI > 0.07 AND P(Healthy) > 0.50" --> R3["Optimal — No Refill\n(Confidence: 0.95)"]
    
    CHECK_T1 -- "Conditions Indeterminate" --> TIER2["Tier 2: Domain-Calibrated Decision Tree\n(Trained on 600 Agronomist Rule Vectors)\nInput: 13-Dim Vector [3 Stress + 3 Moisture + 3 Yield + 4 VIs]"]
    
    TIER2 --> DT_PRED["Class Probabilities: Optimal / Moderate / Urgent / Waterlog / Salinity"]
    
    R1 & R2 & R3 & DT_PRED --> COMBINE["Rank Recommendations by Priority & Confidence"]
    
    IN -.-> CHECK_SALT{"Salt Halo Ratio > 0.05?"}
    CHECK_SALT -- Yes --> ADVISORY["Secondary Advisory:\nSalinity patch detected around mound. Percolative desalinization active."]
    CHECK_SALT -- No --> FIN["Final Structured Advisory Output"]
    ADVISORY --> FIN
    COMBINE --> FIN
```

---

### Module 8 & 9: Dataset Ingestion & Orchestration Lifecycle (`dataset.py` & `orchestrator.py`)

```mermaid
sequenceDiagram
    autonumber
    actor User as User / CLI / Streamlit
    participant Orch as SSPVPipelineOrchestrator
    participant Dataset as SSPVDronePatchDataset
    participant UNet as UNetResNet18_11Ch
    participant CNN as MultiTaskSSPVNet
    participant Stat as StatisticalComparisonEngine
    participant Rec as HybridRecommendationEngine

    User->>Orch: Initialize(data_dir, backbone, max_images_per_date)
    Orch->>Dataset: load_real_dji_folders_or_fallback()
    Dataset-->>Orch: Returns 42+ UAV Patches (with 11-ch tensors & pseudo-labels)
    User->>Orch: train_models(epochs=4, lr=1e-3)
    loop Each Training Epoch
        Orch->>UNet: Forward 11-ch → CrossEntropyLoss → Backward
        Orch->>UNet: Predict Canopy Gate (Classes 1 & 2)
        Orch->>CNN: Forward 11-ch (gated) → MultiTaskLoss → Backward
    end
    Orch-->>User: Training History (Seg Acc, Stress Acc, Loss)
    User->>Orch: evaluate_all_dates()
    Orch->>UNet: Run Full Inference
    Orch->>CNN: Run Multi-Task Predictions
    Orch->>Stat: run_full_comparison(df_metrics)
    Stat-->>Orch: t-test, ANOVA, Cohen's d, Slopes
    Orch->>Rec: recommend() for each sample
    Rec-->>Orch: Actionable irrigation guidance
    User->>Orch: save_thesis_artifacts(output_dir)
    Orch-->>User: Exports CSV, JSON, PNG plots, .pth weights, & Thesis Report TXT
```
