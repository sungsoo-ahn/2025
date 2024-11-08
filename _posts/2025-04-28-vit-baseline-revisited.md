---
layout: distill
title: ViT Baseline Revisited # — Debugging ViT, RandAugment, and Inception Crop implementations
description: ViT (vision transformer) has emerged as a major model family for computer vision
  with the same architecture as that of dominant LLMs and performance matching or exceeding that of
  the CNN-based ResNet-like models. Shortly after the ICLR publication, a note was published to
  follow up with better performance of the smaller ViT-S/16 variant on the ImageNet-1k dataset. In
  our effort to reproduce that, we find inconsistencies among major implementations of ViT,
  RandAugment, and Inception crop that impact model performance. We achieve better performance with
  90 / 150 epoch training budget and call for better awareness of implementation discrepancies.
date: 2025-04-28
future: true
htmlwidgets: true
hidden: false

# Anonymize when submitting
authors:
  - name: Anonymous

# must be the exact same name as your blogpost
bibliography: 2025-04-28-vit-baseline-revisited.bib

# Add a table of contents to your post.
#   - make sure that TOC names match the actual section names
#     for hyperlinks within the post to work correctly. 
#   - please use this format rather than manually creating a markdown table of contents.
toc:
  - name: Introduction
  - name: Implementation discrepancies
    subsections:
    - name: ViT initialization
    - name: RandAugment
    - name: Inception crop
    - name: Big Vision miscellaneous
  - name: Corrected reproduction
  - name: Implication
  - name: Conclusion

_styles: >
  .CenteringContainer div {
      margin: 10px auto;
      width: 80%;
  }
---

*Adapted and expanded from [EIFY/mup-vit](https://github.com/EIFY/mup-vit).*

## Introduction

The dawn of the deep learning era is marked by the "Imagenet moment", when Alexnet won the
ILSVRC-2012 competition for classifying the ImageNet-1k dataset <d-cite key="10.1145/3065386"></d-cite>.
Since then, the field of CV (computer vision) had been dominated by deep CNNs (convolutional neural networks),
especially ResNet-like ones featuring residual connections <d-cite key="7780459"></d-cite>. In 2020,
however, Dosovitskiy et al. proposed vision transformer (ViT) <d-cite key="dosovitskiy2020image"></d-cite>,
which uses transformer architecture that is already dominating the field of language modeling for CV.
With competitive performance, ViT is now widely adapted not only for CV tasks, but also as a component
for vision-language models <d-cite key="radford2021learning"></d-cite>.

<img src="/2025/assets/img/2025-04-28-vit-baseline-revisited/model_scheme.png" class="img-fluid" width="auto" height="auto">
<div class="caption">
    ViT architecture. Images are divided into square patches, which are then linearly projected into
    tokens. After adding position embedding, the tokens are fed to the pre-norm transformer encoder.
    In the original architecture, an extra classification token is added to the input sequence and
    its output is used for classification ("token pooling"). Figure from <d-cite key="dosovitskiy2020image"></d-cite>.
</div>

Since transformer encoder is permutation-invariant, ViT is considered to have weaker inductive bias
and relies more on model regularization or data augmentation, especially at smaller scale <d-cite key="steiner2022how"></d-cite>.
It is therefore particularly notable that Beyer et al. published a short note claiming that ViT-S/16
can achieve better metrics on the ImageNet-1k dataset than ResNet-50 <d-cite key="beyer2022better"></d-cite>.
With manageable compute requirement and open-sourced Big Vision repo <d-cite key="big_vision"></d-cite>
for first-party reproduction, we decide to reproduce it in the PyTorch <d-cite key="Ansel_PyTorch_2_Faster_2024"></d-cite>
ecosystem as the entry point to ViT research.

| Model         | Layers        | Width | MLP        | Heads           | Params  |
|:-------------:|:-------------:|:-----:|:-------------:|:-------------:|:-----:|
| ViT-Ti<d-cite key="pmlr-v139-touvron21a"></d-cite>| 12 | 192 | 768 | 3 | 5.8M |
| ViT-S<d-cite key="pmlr-v139-touvron21a"></d-cite>| 12 | 384 | 1536 | 6 | 22.2M |
| ViT-B<d-cite key="dosovitskiy2020image"></d-cite>| 12 | 768 | 3072 | 12 | 86M |
| ViT-L<d-cite key="dosovitskiy2020image"></d-cite>| 24 | 1024 | 4096 | 16 | 307M |

<div class="caption">
    A few common ViT sizes. <d-cite key="beyer2022better"></d-cite> and this blogpost exclusively
    focuses on ViT-S/16, a variant of ViT-S with patch size $16 \times 16$. Table from <d-cite key="steiner2022how"></d-cite>.
</div>

## Implementation discrepancies

The variant of ViT-S/16 used in <d-cite key="beyer2022better"></d-cite> differs from the original ViT:

1. Instead of token pooling, average of all the output tokens ("global average-pooling", GAP) <d-cite key="9711302"></d-cite> is fed to the MLP head for classification.
2. Fixed 2D sin-cos position embeddings <d-cite key="9711302"></d-cite> is used instead of learned position embeddings.

The model is then trained with Inception crop <d-cite key="7298594"></d-cite>, random horizontal flips,
RandAugment <d-cite key="NEURIPS2020_d85b63ef"></d-cite>, and Mixup <d-cite key="zhang2018mixup"></d-cite>.
Furthermore, there is a variant that replaces the MLP head with a single linear layer that we initially
focus on since it is even simpler and makes "no significant difference" <d-cite key="beyer2022better"></d-cite>.
Sometimes called "simple ViT" <d-cite key="vit-pytorch"></d-cite>, it turns out that there is no up-to-date implementation in PyTorch readily
available<d-footnote>simple_vit.py from vit-pytorch does not support modern attention kernels e.g.
FlashAttention <d-cite key="dao2022flashattention"></d-cite>. The simple_flash_attn_vit.py variant
does but is heavily defensive and resorts to the functional F.scaled_dot_product_attention() for
backward compatibility. </d-footnote>, so we decide to implement our own.

### ViT initialization

It turns out that even building simple ViT from built-in modules from PyTorch requires reinitializing
most of the parameters to match that of the first-party implementation from Big Vision, including:

1. `torch.nn.MultiheadAttention`

    * `in_proj_weight`: In the most common use case when the value dimension is equal to query and key dimension, their projection matrices are combined into `in_proj_weight` whose
    initial values are [set with `xavier_uniform_()`](https://github.com/pytorch/pytorch/blob/aafb3deaf1460764432472a749d625f03570a53d/torch/nn/modules/activation.py#L1112).
    Likely unintentionally, this means that the values are sampled from uniform distribution $\mathcal{U}(-a, a)$ where $$a = \sqrt{\frac{3}{2 \text{ hidden_dim}}}$$ instead of $$\sqrt{\frac{3}{\text{hidden_dim}}}$$<d-footnote>This is not the only case where the combined QKV projection matrix misleads shape-aware optimizer and parameter initialization (<a href="https://x.com/kellerjordan0/status/1844820920780947800">1</a>, <a href="https://github.com/graphcore-research/unit-scaling/blob/1edf20543439e042cc314667b348d9b4c4480e23/unit_scaling/_modules.py#L479">2</a>). It may be worth reexamining whether this truly leads to better performance and whether it is worth it.</d-footnote>.

    * `out_proj.weight`:  Furthermore, the output projection is [initialized as `NonDynamicallyQuantizableLinear`](https://github.com/pytorch/pytorch/blob/e62073d7997c9e63896cb5289ffd0874a8cc1838/torch/nn/modules/activation.py#L1097). Just like [`torch.nn.Linear`](https://pytorch.org/docs/stable/generated/torch.nn.Linear.html), its initial weights are sampled from uniform distribution $\mathcal{U}(-a, a)$ where $$a = \sqrt{\frac{1}{\text{hidden_dim}}}$$ instead of $$\sqrt{\frac{3}{\text{hidden_dim}}}$$ <d-footnote>See <a href="https://github.com/pytorch/pytorch/issues/57109#issuecomment-828847575">pytorch/pytorch#57109 (comment)</a> for the origin of this discrepancy.</d-footnote>.

    To conform with [`jax.nn.initializers.xavier_uniform()` used by the first-party implementation from Big Vision](https://github.com/google-research/big_vision/blob/ec86e4da0f4e9e02574acdead5bd27e282013ff1/big_vision/models/vit.py#L93), both are re-initialized with samples from uniform distribution $\mathcal{U}(-a, a)$ where $$a = \sqrt{\frac{3}{\text{hidden_dim}}}$$ <d-footnote>Note that the standard deviation of uniform distribution $\mathcal{U}(-a, a)$ is $$\frac{a}{\sqrt{3}}$$ So this is also the correct initialization for preserving the input scale assuming that the input features are uncorrelated.</d-footnote>.

2. `torch.nn.Conv2d`: Linear projection of flattened patches can be done with 2D convolution, namely
   [`torch.nn.Conv2d` initialized with `in_channels = 3`, `out_channels = hidden_dim`, and both `kernel_size`
   and `stride` set to `patch_size` in PyTorch](https://pytorch.org/docs/stable/generated/torch.nn.Conv2d.html).
   `torch.nn.Conv2d`, however, defaults to weight and bias initialization with uniform distribution
   $\mathcal{U}(-a, a)$ where $$a = \sqrt{\frac{1}{\text{fan_in}}}$$ instead of Big Vision's Lecun normal
   (truncated normal) initialization <d-cite key="klambauer2017self"></d-cite> for weight and zero-init for bias.
   Furthermore, [PyTorch's own `nn.init.trunc_normal_()`](https://pytorch.org/docs/stable/nn.init.html#torch.nn.init.trunc_normal_)
   doesn't take the effect of truncation on standard deviation into account unlike [the JAX implementation](https://github.com/google/jax/blob/1949691daabe815f4b098253609dc4912b3d61d8/jax/_src/nn/initializers.py#L334), so we have to implement the correction ourselves:

   <div class="CenteringContainer">
     <div>
       <img src="/2025/assets/img/2025-04-28-vit-baseline-revisited/truncated_normal.png" class="img-fluid" width="auto" height="auto">
     </div>
   </div>

   <div class="caption">
     Unit normal distribution has standard deviation $\sigma = 1$, but after truncating at $\pm 2$
     the standard deviation is reduced to $\sigma = 0.880$. To restore unit standard deviation, one
     needs to sample $\mathcal{N}(0, \sigma = \frac{1}{.880})$ instead and truncate at $\pm 2 \sigma$.
   </div>

3. `torch.nn.Linear` for the classification head: Specifically for the classification head, Big Vision
   usually zero-init both the weight and bias for the linear layer, including the ViT-S/16 in question.
   Notably, neither [`simple_vit.py`](https://github.com/lucidrains/vit-pytorch/blob/141239ca86afc6e1fe6f4e50b60d173e21ca38ec/vit_pytorch/simple_vit.py#L108) nor [`simple_flash_attn_vit.py`](https://github.com/lucidrains/vit-pytorch/blob/141239ca86afc6e1fe6f4e50b60d173e21ca38ec/vit_pytorch/simple_flash_attn_vit.py#L162) from vit-pytorch does this.

After fixing 1-3, we verify that all of the per-layer summary statistics including minimum, maximum,
mean, and standard deviation of the 21,974,632 model parameters at initialization match that of the
Big Vision first-party implementation.

### RandAugment

Following <d-cite key="beyer2022better"></d-cite>, we train the model with batch size 1024, training
budget 90 epochs = round(1281167 / 1024 * 90) = 112603 steps, 10000 warm-up steps, cosine learning
rate decay, 1e-3 maximum learning rate, 1e-4 decoupled weight decay <d-cite key="adamw-decoupling-blog"></d-cite>
(0.1 in PyTorch's parameterization before multiplying by the learning rate), AdamW optimizer, Inception
crop with 5%-100% of the area of the original image ("scale"), random horizontal flips,
RandAugment <d-cite key="NEURIPS2020_d85b63ef"></d-cite> with [`num_ops = 2` and `magnitude = 10`](https://pytorch.org/vision/main/generated/torchvision.transforms.v2.RandAugment.html),
and Mixup <d-cite key="zhang2018mixup"></d-cite> with [`alpha = 0.2`](https://pytorch.org/vision/main/generated/torchvision.transforms.v2.MixUp.html).
We also notice that the Big Vision implementation "normalizes" the image with mean = std = (0.5, 0.5, 0.5)
([`value_range(-1, 1)`](https://github.com/google-research/big_vision/blob/46b2456f54b9d4f829d1925b78943372b376153d/big_vision/configs/vit_s16_i1k.py#L52))
and uses [the same neutral RGB value as the fill value for RandAugment](https://github.com/google-research/big_vision/blob/46b2456f54b9d4f829d1925b78943372b376153d/big_vision/pp/archive/autoaugment.py#L676)
and make sure that our reproduction conforms to this training setup.

### Inception crop

### Big Vision miscellaneous

## Corrected reproduction

## Implication

## Conclusion
