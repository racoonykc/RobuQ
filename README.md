# RobuQ: Pushing DiTs to W1.58A2 via Robust Activation Quantization
[Kaicheng Yang](https://github.com/KaichengYang),[Xun Zhang](xxx.com),[Haotong Qin](xxx.com),[Yucheng Lin](xxx.com), [Kaisen Yang](xxx.com),[Xianglong Yan](xxx.com) and [Yulun Zhang](https://github.com/YulunZhang).

[[arXiv](xxx.com)] [[supplementary material](xxx.com)] 

#### 🔥🔥🔥 News

- **2025-09-26:** Repository initial release.

---

> **Abstract:** Diffusion Transformers (DiTs) have recently emerged as a powerful backbone for image generation, demonstrating superior scalability and performance over U-Net architectures. However, their practical deployment is hindered by substantial computational and memory costs. While Quantization-Aware Training (QAT) has shown promise for U-Nets, its application to DiTs faces unique challenges, primarily due to the sensitivity and distributional complexity of activations. In this work, we identify activation quantization as the primary bottleneck for pushing DiTs to extremely low-bit settings. To address this, we propose a systematic QAT framework for DiTs, named **RobuQ**. We start by establishing a strong ternary weight (W1.58A4) DiT baseline. Building upon this, we propose **RobustQuantizer** to achieve robust activation quantization. Our theoretical analyses show that the Hadamard transform can convert unknown per-token distributions into per-token normal distributions, providing a strong foundation for this method. Furthermore, we propose **AMPN**, the first **A**ctivation-only **M**ixed-**P**recision **N**etwork pipeline for DiTs. This method applies ternary weights across the entire network while allocating different activation precisions to each layer to eliminate information bottlenecks. Through extensive experiments on unconditional and conditional image generation, our RobuQ framework achieves state-of-the-art performance for DiT quantization in sub-4-bit quantization configuration. To the best of our knowledge, RobuQ is the first achieving stable and competitive image generation on large datasets like **ImageNet-1K** with activations quantized to average 2 bits.
---

### Visualization



![RobuQ-W1.58A3 256×256 images from quantized DiT-XL/2 on ImageNet-1K](figs/A3_visualization.svg)

*Fig1:RobuQ-W1.58A3 256×256 images from quantized DiT-XL/2 on ImageNet-1K*

![RobuQ-W1.58A2 256×256 images from quantized DiT-XL/2 on ImageNet-1K](figs/A2_visualization.svg)

*Fig1:RobuQ-W1.58A2 256×256 images from quantized DiT-XL/2 on ImageNet-1K*
---

### Comparison


![RobuQ consistently outperforms all previous methods](figs/compare.svg)

*Fig2:RobuQ consistently outperforms all previous methods*

## 🔖 TODO


- [ ] Release ckpt,training and inference code
- [ ] Release inference engine
- [ ] Release more quantized DiTs

---

## 💡 Acknowledgements

This code is built on [Diffusion Transformer](https://github.com/facebookresearch/DiT).