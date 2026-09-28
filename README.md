# CGSF-Net: Confidence-Guided Semantic Fusion Network for Shadow Removal Localization

## Overview

This repository is the official implementation of **CGSF-Net: Confidence-Guided Semantic Fusion Network for Shadow Removal Localization**.

In this paper, we propose a novel shadow removal localization framework called **CGSF-Net**, which combines manipulation-aware forensic representations with high-level semantic priors to improve the localization of shadow-removed regions.

CGSF-Net adopts a dual-source representation strategy. A SparseViT backbone is employed to extract fine-grained manipulation-aware features, while a frozen DINOv3 model provides high-level semantic priors and global scene context. Since these two feature sources differ in semantic granularity and spatial response, we introduce an **Offset-Aligned Heterogeneous Semantic Fusion Module (HSFM)** to align semantic features with forensic representations before cross-attention-based interaction.

To further account for spatial variations in semantic reliability, a **Fusion Confidence Estimation Module (FCEM)** estimates the reliability of semantic fusion at each spatial location. The resulting confidence information is then used by a **Confidence-Guided Spatially Dynamic Decoder (CGDD)** to adaptively regulate multi-scale feature aggregation and balance local forensic evidence with semantic context.

Finally, a **Boundary-Aware Residual Refinement** branch predicts residual corrections to the coarse localization result, improving the delineation of weak and gradual shadow-removal boundaries.

---

## Contributions

• We propose **CGSF-Net**, a confidence-guided semantic fusion network for shadow removal localization. The proposed framework augments manipulation-aware features extracted by SparseViT with high-level semantic priors from a frozen DINOv3 model, improving the discrimination between shadow-removed regions and visually similar unmanipulated regions.

• We design an **Offset-Aligned Heterogeneous Semantic Fusion Module (HSFM)** to effectively integrate heterogeneous forensic and semantic representations. A learnable offset field is predicted to spatially align DINOv3 semantic features with intermediate SparseViT features before cross-attention, reducing feature misalignment and enabling more reliable semantic enhancement.

• We introduce a **Fusion Confidence Estimation Module (FCEM)** to estimate the reliability of semantic fusion at each spatial location. The estimated confidence map and confidence feature explicitly characterize spatial variations in semantic usefulness and provide guidance for subsequent feature aggregation.

• We develop a **Confidence-Guided Spatially Dynamic Decoder (CGDD)** that dynamically assigns location-dependent weights to multi-scale features and adaptively balances local forensic cues with semantic context. In addition, a boundary-aware residual refinement branch further corrects uncertain regions and improves boundary localization.

---

## Pretrained Weights

Pretrained model weights will be provided here:

```text
Coming soon.
```

---

## Testing

The repository provides the implementation of CGSF-Net and the corresponding testing code.

After downloading the pretrained weights, the model can be evaluated on the shadow removal localization benchmark using the provided testing scripts.

