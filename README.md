# OrCanNet: CNN-ViT Hybrid Model for Oral Disease Classification

A hybrid deep learning framework combining a CNN branch (EfficientNet-B0) and a Vision Transformer (ViT) branch with attention-weighted feature fusion for multi-class oral disease classification.

## Overview

OrCanNet leverages the complementary strengths of two architectures:
- **EfficientNet-B0 (CNN)**: Captures local texture, lesion boundaries, and fine-grained surface details
- **Vision Transformer (ViT)**: Captures global spatial context and long-range dependencies

A learnable **attention-weighted fusion module** combines features from both branches, enabling the model to dynamically balance local and global cues for robust oral disease classification.

## Dataset: MOD (Mouth and Oral Diseases)

**Classes**: 7 oral disease categories
- **CaS** — Canker sores
- **CoS** — Cold sores  
- **Gum** — Gingival disease
- **MC** — Mucocele
- **OC** — Oral cancer
- **OLP** — Oral lichen planus
- **OT** — Other lesions

**Data Split**:
- Training: 2,035 images (original) → 3,150 after augmentation
- Validation: ~420 images
- Test: 1,028 images

**Class Balancing**: Training set resampled to 450 images per class using offline augmentation (RandomResizedCrop, horizontal flip, rotation, color jittering) to eliminate majority-class bias and improve ViT convergence.

## Model Architecture

### 1. CNN Branch (EfficientNet-B0)
```
Input (224×224)
    ↓
EfficientNet-B0 Backbone (pretrained on ImageNet)
    ↓
Multi-scale feature extraction (F1, F2, F3)
    ↓
Global Average Pooling
    ↓
Output: 1,280-dim feature vector
```

### 2. Vision Transformer Branch
```
Input (224×224)
    ↓
ViT-Small-Patch16-224 (vit_small_patch16_224, pretrained)
    ↓
CLS token extraction
    ↓
Output: 384-dim embedding
```

### 3. Feature Fusion Module

**Concatenation & Projection**:
- Concatenate CNN (1,280-dim) + ViT (384-dim) → 1,664-dim
- Project to 512-dim via linear layer + ReLU

**Attention-Weighted Fusion**:
```
Attention Computation:
  attn_hidden = ReLU(Linear(fused_512 → 128))
  attn_weights = Softmax(Linear(attn_hidden → 512))

Weighted Output:
  fused_output = fused_512 * attn_weights
```

**Auxiliary Losses**:
- CNN branch auxiliary classifier (1,280-dim → 7 classes)
- ViT branch auxiliary classifier (384-dim → 7 classes)
- Provides additional supervision and interpretability

### 4. Classifier Head
```
Fused Features (512-dim)
    ↓
Dropout (p=0.3)
    ↓
Linear Layer (512 → 7 classes)
    ↓
Output: 7-class logits
```

## Training Configuration

| Parameter | Value |
|-----------|-------|
| **Batch Size** | 32 |
| **Optimizer** | AdamW |
| **Weight Decay** | 1e-4 |
| **Gradient Clipping** | 1.0 |
| **Label Smoothing** | 0.1 |
| **Early Stopping Patience** | 5 epochs |
| **Max Epochs** | 50 |
| **Actual Training** | 27 epochs (early stopped) |

### Two-Stage Training Strategy

**Phase 1: Frozen Backbone (Epochs 1–5)**
- CNN & ViT backbones frozen; only fusion module and classifier trainable
- Learning rate: 3e-4
- Purpose: Stable initialization of fusion module on balanced data

**Phase 2: Unfrozen Backbone (Epoch 6 onward)**
- All parameters trainable with **differential learning rates**:
  - CNN backbone: 1e-5
  - ViT backbone: 1e-5
  - Fusion module: 3e-5
  - Classifier head: 3e-5
- Purpose: Fine-tune pretrained features with lower learning rates

### Loss Function
```
Total Loss = L_main + λ × (L_cnn_aux + L_vit_aux)

where:
  L_main      = Weighted cross-entropy (main classifier)
  L_cnn_aux   = Cross-entropy (CNN auxiliary classifier)
  L_vit_aux   = Cross-entropy (ViT auxiliary classifier)
  λ           = 0.1 (auxiliary loss weight)
  
Weights are inverse class frequencies (balanced sampling)
```

**Scheduler**: Cosine Annealing (T_max=50)

## Training Results

### Training Dynamics

Training exhibited smooth convergence with two distinct phases:

**Phase 1 (Frozen, Epochs 1–5)**:
- Loss: 2.20 → 1.63
- Val Acc: 31.3% → 51.0%
- Model learned stable fusion module

**Phase 2 (Unfrozen, Epochs 6+)**:
- Loss: 1.49 → 0.65
- Val Acc: 54.2% → **99.42%** (peak at Epoch 22)
- Dramatic improvement after backbone fine-tuning

**Early Stopping**: Triggered at Epoch 27 (no improvement for 5 consecutive epochs)

### Final Test Set Performance

**Overall Metrics** (1,028 test samples):
- **Accuracy**: 99%
- **Macro Precision**: 0.99
- **Macro Recall**: 0.99
- **Macro F1-Score**: 0.99

**Per-Class Performance**:

| Class | Precision | Recall | F1-Score | Support |
|-------|-----------|--------|----------|---------|
| CaS   | 0.98      | 1.00   | 0.99     | 160     |
| CoS   | 1.00      | 1.00   | 1.00     | 149     |
| Gum   | 0.98      | 1.00   | 0.99     | 120     |
| MC    | 0.98      | 0.98   | 0.98     | 180     |
| OC    | 0.99      | 0.98   | 0.99     | 108     |
| OLP   | 1.00      | 0.98   | 0.99     | 180     |
| OT    | 0.99      | 1.00   | 1.00     | 131     |

**Confusion Matrix Summary**:
- **Total Correct**: 1,018 / 1,028 (99.03%)
- **Total Misclassifications**: 10 (0.97%)

Misclassified samples show minimal confusion:
- 1 CaS → MC
- 2 MC → Gum, 1 MC → OC  
- 2 OC → MC
- 2 OLP → CaS, 1 OLP → MC, 1 OLP → OT

### Cross-Validation Metrics
- **ROC-AUC (One-vs-Rest)**: 1.00 across all classes
- **Precision-Recall**: Excellent curves indicating well-separated decision boundaries
- **Confidence Analysis**: Model maintains high confidence (>0.8) on correct predictions

## Usage

### Requirements
```
torch >= 2.0
torchvision >= 0.15
timm >= 0.6.0
numpy >= 1.24
matplotlib >= 3.7
seaborn >= 0.12
scikit-learn >= 1.2
albumentations >= 1.3
opencv-python >= 4.7
```

### Installation
```bash
pip install torch torchvision timm albumentations scikit-learn matplotlib seaborn
```

### Training from Scratch

1. **Prepare Dataset**:
   - Organize MOD dataset into `data_root/Training/`, `data_root/Validation/`, `data_root/Testing/`
   - Each subdirectory contains class folders

2. **Update Paths** in training script:
   ```python
   data_root = "/path/to/MOD/dataset"
   MODEL_SAVE_PATH = "/path/to/save/teeth.pth"
   ```

3. **Run Training**:
   ```bash
   python MODConcat1.py
   ```

   The script will:
   - Load and augment training data (offline augmentation on GPU)
   - Initialize OrCanNet with frozen backbone
   - Train for 5 epochs with frozen backbone
   - Unfreeze backbone and train with differential learning rates
   - Apply early stopping when validation accuracy plateaus
   - Save best checkpoint and generate evaluation plots

### Inference

```python
import torch
from torchvision import transforms

# Load model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = OrCanNet(num_classes=7).to(device)
model.load_state_dict(torch.load("teeth.pth"))
model.eval()

# Prepare input
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

image = transform(PIL_image).unsqueeze(0).to(device)

# Predict
with torch.no_grad():
    logits, _, _ = model(image)
    probs = torch.softmax(logits, dim=1)
    pred_class = torch.argmax(probs, dim=1)

class_names = ["CaS", "CoS", "Gum", "MC", "OC", "OLP", "OT"]
print(f"Prediction: {class_names[pred_class.item()]}")
print(f"Confidence: {probs.max().item():.2%}")
```

## Key Design Insights

1. **Two-Stage Training**: Freezing the backbone initially stabilizes the fusion module on balanced data, then fine-tuning with differential learning rates avoids destructive updates to pretrained features.

2. **Attention-Weighted Fusion**: Rather than naive concatenation, learned attention weights allow the model to dynamically balance CNN (local) and ViT (global) features based on image content.

3. **Auxiliary Losses**: Branch-specific classifiers provide additional supervision during training and enable later interpretability without architectural changes.

4. **Class Balancing**: Augmenting the training set to 450 images per class eliminates majority-class bias—particularly important for ViT convergence on small datasets.

5. **Label Smoothing & Gradient Clipping**: Prevents overconfidence and stabilizes training on medical imaging data with potential labeling noise.

## Explainability

The trained model supports three interpretability approaches:

1. **Grad-CAM** (CNN branch): Highlights salient local regions in EfficientNet feature maps
2. **Attention Maps** (ViT branch): Visualizes CLS-token attention weights to identify global regions of interest
3. **Fusion Heatmap**: Shows where the attention-weighted fusion diverges from raw branch outputs

Together, these reveal the model's decision-making process across local and global spatial scales.

## Reproduction Notes

- All training was performed on Google Colab T4 GPU
- Offline augmentation ensured consistent per-epoch augmentation across runs
- Copying augmented dataset to local Colab storage (/content/) significantly reduced I/O bottlenecks
- Best model was selected at Epoch 22 (Val Acc: 99.42%) but early stopping allowed training to continue until Epoch 27 with minimal overfitting
- No test-time augmentation, ensembling, or external data was used


