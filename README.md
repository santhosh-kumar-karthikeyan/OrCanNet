OrCanNet
========

CNN-ViT Disagreement-Aware Oral Disease Classification

OrCanNet is a hybrid deep learning framework for multi-class oral disease
classification.  It combines a convolutional neural network (CNN) branch and
a Vision Transformer (ViT) branch under a disagreement-aware adaptive fusion
mechanism.  The design addresses a known limitation of pure self-attention
architectures on small, imbalanced medical imaging datasets: ViT models
require large training corpora to generalise, while CNNs saturate on global
context.  The fusion module detects per-sample disagreement between the two
branches and adjusts their relative contribution dynamically at inference
time, without requiring separate models or ensembles.


Table of Contents
-----------------

  1. Dataset
  2. Data Preparation
  3. Model Architecture
  4. Training Configuration
  5. Training Dynamics
  6. Evaluation Results
  7. Explainability
  8. Reproduction
  9. Dependencies


1. Dataset
----------

Dataset: MOD — Mouth and Oral Diseases Dataset
Classes: 7

  CaS   — Canker sores
  CoS   — Cold sores
  Gum   — Gingival disease
  MC    — Mucocele
  OC    — Oral cancer
  OLP   — Oral lichen planus
  OT    — Other lesions

Original training split: approximately 2035 images.  Per-class counts ranged
from roughly 210 to 370, yielding a moderately imbalanced distribution that
presents non-trivial risk for transformer training and hybrid model stability
at this scale.

Validation and test sets were kept intact throughout all preprocessing steps
to preserve evaluation integrity.


2. Data Preparation
-------------------

2.1  Offline Augmentation

The training set was expanded offline using Albumentations.  The following
transforms were applied per image:

  - RandomResizedCrop
  - Horizontal flip
  - Rotation (mild angle range)
  - RandomBrightnessContrast
  - CLAHE (contrast-limited adaptive histogram equalisation, retained for
    medical texture fidelity)

Augmentation was applied independently per class.

2.2  Class Balancing

After augmentation each class was resampled to exactly 450 images:

  450 images/class * 7 classes = 3150 training images total

This eliminates majority-class bias, stabilises ViT gradient updates, and
improves recall on minority classes.

2.3  I/O Optimisation

Training on Google Colab exhibited significant throughput degradation when
reading directly from Google Drive.  The dataset was copied to local Colab
storage (/content/) prior to training, yielding a substantial reduction in
per-epoch wall time.


3. Model Architecture
---------------------

OrCanNet is composed of three sub-systems: a CNN branch, a ViT branch, and a
disagreement-aware fusion module.

3.1  CNN Branch

  Backbone         : EfficientNet-B0 (ImageNet pretrained)
  Feature scales   : F1, F2, F3 (multi-scale extraction)
  Pooling          : Global average pooling per scale
  Auxiliary output : Softmax classifier over 7 classes

The CNN branch captures local texture cues, lesion boundary structure, and
surface irregularities at multiple spatial resolutions.

3.2  Vision Transformer Branch

  Backbone         : ViT-Small-Patch16-224 (timm 1.0.24)
  Representation   : CLS token
  Auxiliary output : Softmax classifier over 7 classes

The ViT branch captures long-range spatial dependencies, global structural
context, and macro lesion distribution patterns.

3.3  Disagreement-Aware Fusion

Given auxiliary predictions P_cnn and P_vit from each branch:

  Step 1 — Compute scalar disagreement:

      D = sum( |P_cnn - P_vit| )

  Step 2 — Derive adaptive weight via a learned linear gate:

      alpha = sigmoid( w * D + b )

  Step 3 — Blend fused and raw representations:

      F = alpha * Phi + (1 - alpha) * Raw

When branch predictions agree, the gate suppresses correction.  When they
diverge, it increases the contribution of the attention-weighted fusion
representation.  This is end-to-end differentiable and adds no inference
overhead beyond a single dot product and sigmoid.

3.4  Multi-Scale Aggregation

CNN feature maps F1, F2, F3 are spatially upsampled to a common resolution,
concatenated along the channel axis, and reduced via 1x1 convolution followed
by global average pooling.  The resulting vector is concatenated with the
fusion output F before the final linear classifier.

3.5  Loss Function

Total loss combines the main classification cross-entropy with two auxiliary
cross-entropy terms:

    L = L_main + lambda * (L_cnn_aux + L_vit_aux)

    lambda = 0.2


4. Training Configuration
--------------------------

  Framework      : PyTorch
  Hardware       : Google Colab T4 GPU
  Epochs         : 20
  Batch size     : 8
  Optimizer      : Adam
  Learning rate  : 1e-4
  Aux loss weight: 0.2
  Input size     : 224 x 224
  Training set   : 3150 images


5. Training Dynamics
--------------------

5.1  Training Loss

  Epoch 1   1.2967
  Epoch 2   0.2964
  Epoch 4   0.0781
  Epoch 7   0.0303
  Epoch 14  0.0119
  Epoch 20  0.0454

Loss decreases sharply from epoch 1 to epoch 7, stabilises through epoch 14,
and exhibits a marginal uptick at epoch 20 consistent with end-of-schedule
noise.  No divergence was observed.

5.2  Validation Accuracy

Range: 56% – 69%

Notable peaks:

  Epoch 2   69.09%
  Epoch 10  69.09%

The validation set is small, so single-percentage oscillations are within
expected statistical variance.  No catastrophic overfitting was detected.


6. Evaluation Results
---------------------

Test set size: 54 images (6–10 samples per class)

Overall metrics:

  Accuracy          0.78
  Macro F1-score    0.77
  Weighted F1-score 0.77

Per-class F1:

  Class   F1
  ------  ----
  CoS     0.94
  Gum     0.88
  CaS     0.82
  OLP     0.75
  MC      0.71
  OC      0.67
  OT      0.62

CoS, Gum, and CaS exhibit strong discriminability.  OC and OT show lower F1,
attributable primarily to the small per-class test support (6–8 samples) and
intra-class visual heterogeneity.  No ensembling, test-time augmentation, or
external data was used.


7. Explainability
-----------------

The trained model exposes three interpretability pathways without requiring
retraining or architectural modification:

  CNN Grad-CAM
      Gradient-weighted class activation maps computed on the EfficientNet
      feature maps.  Highlights salient local regions that drove the CNN
      prediction.

  ViT Attention Map
      CLS-token attention weights extracted from the final transformer block
      via a registered forward hook.  Adapted for the timm 1.0.24 API.
      Visualises global regions of interest captured by self-attention.

  Disagreement Fusion Heatmap
      Overlay derived from the per-sample disagreement scalar D and the
      fusion weight alpha.  Shows where and how much the model relied on the
      attention-corrected representation versus the raw branch outputs.

These three visualisations together demonstrate local vs. global focus
separation and the dynamic weighting behaviour of the fusion gate.


8. Reproduction
---------------

The full pipeline is implemented in a single Jupyter notebook intended for
Google Colab.  To reproduce:

  1. Mount Google Drive and place the MOD dataset under the expected path.
  2. Run the augmentation cells to expand and balance the training split to
     3150 images.
  3. Copy the dataset to local Colab storage for I/O performance.
  4. Run the model definition, training, and evaluation cells in order.
  5. The notebook saves a checkpoint that is reloaded for the explainability
     section; no re-training is required for visualisation.

The saved checkpoint is compatible with the explainability pipeline provided
in the same notebook.


9. Dependencies
---------------

  torch          >= 2.0
  torchvision    >= 0.15
  timm           == 1.0.24
  albumentations >= 1.3
  opencv-python  >= 4.7
  numpy          >= 1.24
  matplotlib     >= 3.7
  scikit-learn   >= 1.2
  Pillow         >= 9.5
  grad-cam       (pytorch-grad-cam)

timm version is pinned to 1.0.24 because the attention hook extraction logic
depends on the internal ViT block API introduced in that release.  Using a
different timm version may require adapting the attention extraction code.
