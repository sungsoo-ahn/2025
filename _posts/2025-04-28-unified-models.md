---
layout: distill
title: Towards Unified Multimodal Models:A Review and Future Vision
description: Recent advancements in unified models for multimodal understanding and generation, such as Transfusion of Meta, Show-O of NUS, and EMU3 of BAAI, highlight a growing trend toward single models capable of handling diverse tasks. These models explore pure autoregressive methods, diffusion-based approaches, or their combinations. Inspired by these works, this blog provides an overview of unified multimodal models, reviews current developments, and offers insights into potential future directions. We also discuss the principles of autoregressive and diffusion models and explore whether the future of unified models lies in one of these paradigms or a hybrid approach.
date: 2025-04-28
future: true
htmlwidgets: true
hidden: false

# Anonymize when submitting
authors:
  - name: Anonymous

# authors:
#   - name: Albert Einstein
#     url: "https://en.wikipedia.org/wiki/Albert_Einstein"
#     affiliations:
#       name: IAS, Princeton
#   - name: Boris Podolsky
#     url: "https://en.wikipedia.org/wiki/Boris_Podolsky"
#     affiliations:
#       name: IAS, Princeton
#   - name: Nathan Rosen
#     url: "https://en.wikipedia.org/wiki/Nathan_Rosen"
#     affiliations:
#       name: IAS, Princeton

# must be the exact same name as your blogpost
bibliography: 2025-04-28-unified-models.bib

# Add a table of contents to your post.
#   - make sure that TOC names match the actual section names
#     for hyperlinks within the post to work correctly. 
#   - please use this format rather than manually creating a markdown table of contents.
toc:
  - name: Introduction
  - name: Background:Autoregressive and Diffusion
    subsections:
    - name: Autoregressive Model
    - name: Diffusion Model
  - name: Unified Multimodal Models
  - name: Challenges and Future Work
  - name: to be removed
    subsections:
    - name: Images and Figures
    - name: Interactive Figures
    - name: Citations
    - name: Footnotes
    - name: Code Blocks
    - name: Diagrams
    - name: Tweets
    - name: Layouts
    - name: Other Typography?

# Below is an example of injecting additional post-specific styles.
# This is used in the 'Layouts' section of this post.
# If you use this post as a template, delete this _styles block.
_styles: >
  .fake-img {
    background: #bbb;
    border: 1px solid rgba(0, 0, 0, 0.1);
    box-shadow: 0 0px 4px rgba(0, 0, 0, 0.1);
    margin-bottom: 12px;
  }
  .fake-img p {
    font-family: monospace;
    color: white;
    text-align: left;
    margin: 12px 0;
    text-align: center;
    font-size: 16px;
  }
  .box-note, .box-warning, .box-error, .box-important {
    padding: 15px 15px 15px 10px;
    margin: 20px 20px 20px 5px;
    border: 1px solid #eee;
    border-left-width: 5px;
    border-radius: 5px 3px 3px 5px;
  }
  d-article .box-note {
    background-color: #eee;
    border-left-color: #2980b9;
  }
  d-article .box-warning {
    background-color: #fdf5d4;
    border-left-color: #f1c40f;
  }
  d-article .box-error {
    background-color: #f4dddb;
    border-left-color: #c0392b;
  }
  d-article .box-important {
    background-color: #d4f4dd;
    border-left-color: #2bc039;
  }
  html[data-theme='dark'] d-article .box-note {
    background-color: #333333;
    border-left-color: #2980b9;
  }
  html[data-theme='dark'] d-article .box-warning {
    background-color: #3f3f00;
    border-left-color: #f1c40f;
  }
  html[data-theme='dark'] d-article .box-error {
    background-color: #300000;
    border-left-color: #c0392b;
  }
  html[data-theme='dark'] d-article .box-important {
    background-color: #003300;
    border-left-color: #2bc039;
  }
  html[data-theme='dark'] d-article blockquote {
    color: var(--global-text-color) !important;
  }
  html[data-theme='dark'] d-article summary {
    color: var(--global-text-color) !important;
  }
  d-article aside * {
    color: var(--global-text-color) !important;
  }
  d-article p {
    text-align: justify;
    text-justify: inter-word;
    -ms-hyphens: auto;
    -moz-hyphens: auto;
    -webkit-hyphens: auto;
    hyphens: auto;
  }
  d-article aside {
    border: 1px solid #aaa;
    border-radius: 4px;
    padding: .5em .5em 0;
    font-size: 90%;
  }
  d-article aside p:first-child {
      margin-top: 0;
  }
  d-article details {
    border: 1px solid #aaa;
    border-radius: 4px;
    padding: .5em .5em 0;
  }
  d-article summary {
    font-weight: bold;
    margin: -.5em -.5em 0;
    padding: .5em;
    display: list-item;
  }
  d-article details[open] {
    padding: .5em;
  }
  d-article details[open] summary {
    border-bottom: 1px solid #aaa;
    margin-bottom: .5em;
  }
categories:
- Data Processing Inequality
- Information Theory
- Data Processing Inequality
- Information Theory
- Function-Space Variational Inference
- Parameter Equivalence Classes
- Entropy Regularization
- Label Entropy Regularization
---



## Introduction

<!-- <aside class="l-body box-note" markdown="1"> -->
<!-- </aside> -->

{% include figure.html path="assets/img/2025-04-28-unified-models/21.07.44.png" class="img-fluid" %}

In recent years, the field of multimodal understanding and generation has seen significant advancements, particularly with the rise of unified models capable of addressing a wide range of tasks. Notable examples include Meta’s Transfusion<d-cite key="zhou2024transfusion"></d-cite>, NUS’s Show-O<d-cite key="xie2024showo"></d-cite>, and BAAI’s EMU3<d-cite key="wang2024emu3"></d-cite>, which have set the stage for a growing trend: the development of single models that can process and generate information across multiple modalities, such as text, images, and more. These unified models utilize a variety of techniques, including pure autoregressive methods, diffusion-based approaches, or even hybrid combinations of both.


<blockquote>
“Unity is strength... when there is teamwork and collaboration, wonderful things can be achieved.” — Mattie Stepanek
</blockquote>

Unity concept resonates deeply in the context of unified models for multimodal understanding and generation. This blog aims to explore the recent developments in unified multimodal models, reviewing their current state and outlining the future directions for research and application. We will delve into the principles behind autoregressive and diffusion models, shedding light on their unique characteristics and how they can be effectively combined. Ultimately, we will discuss whether the future of unified models lies in one of these paradigms, or if a hybrid approach holds the key to even greater advancements.


## Background: Autoregressive and Diffusion

### Autoregressive Model
Autoregressive (AR) sequence generation is a method where a sequence is generated token by token, with each token predicted based on the preceding ones. 
This approach models dependencies within the sequence by conditioning each output on prior outputs, effectively capturing the structure and patterns of sequential data.

**Definition**. For a data sequence $$(x_1, x_2, ..., x_{T})$$, an autoregressive model represents the joint probability as:

$$
P(x_1, x_2, ..., x_T) = \prod_{t=1}^{T} P(x_t | x_1, x_2, ..., x_{t-1})
$$

This formula captures the essence of autoregressive generation: predicting each token based on the prior sequence of tokens.

**Why Choose Autoregressive Models for Unified Multimodal Model?**

* **Handling Sequential Data**: Autoregressive models excel at processing and generating sequential data, which is central to many tasks like text, images, and video.

* **Unified Framework for Multimodal Tasks**: They can integrate different modalities (*e.g.,* text, images, audio) by converting them into a unified sequence of tokens, simplifying model design.

* **Easier to Scale Up**: Autoregressive models align well with scaling laws, making them easier to scale up in terms of model size, data, and computational resources.


**Autoregressive for Vision**. The process for generating an image can be broken down pixel-by-pixel or patch-by-patch, where each pixel/patch is conditioned on previously generated content.
Autoregressive models are simple, interpretable, and effective, but their sequential nature can limit efficiency, particularly for long sequences. These characteristics are central to understanding their role in unified multimodal generation tasks.



### Diffusion Model
Diffusion models are a class of probabilistic generative models used to synthesize data by modeling its distribution through a gradual process of noise addition and removal. These models have shown impressive results in image, audio, and video generation tasks.

**Forward Process**. The forward process is a Markov chain where Gaussian noise is iteratively added to the data $\mathbf{x}_0$, resulting in a series of noisy representations $\mathbf{x}_1,\mathbf{x}_2,...,\mathbf{x}_T$:

$$
q(\mathbf{x}_{t} | \mathbf{x}_{t-1}) = \mathcal{N}(\mathbf{x}_t; \sqrt{1-\beta_t} \mathbf{x}_{t-1}, \beta_t \mathbf{I}),
$$

where $β_t$ is the noise schedule, $\mathcal{N}$ refers to Gaussian distribution. 


**Reverse Process**. The reverse process learns to denoise $x_T$ back to $x_0$ through a neural network $p_θ$, parameterized as:

$$
p_θ(x_{t-1} | x_t) = \mathcal{N}(x_{t-1}; μ_θ(x_t, t), Σ_θ(x_t, t)),
$$

During sampling, we start from pure Gaussian noise $x_T ~ \mathcal{N}(0, I)$ and iteratively sample $x_{t-1}$ from $p_θ$ until $x_0$ is reconstructed.

**Loss Function**. The neural network $ε_θ$ is trained to predict the noise added at each step $t$, using the following loss function:

$$
L(θ) = \mathbb{E}_{x_0, ε, t}[ ||ε - ε_θ(x_t, t)||^2 ],
$$

where $ε ~ \mathcal{N}(0, I)$ and $x_t$ refer to actual noise and noisy data at step $t$, $ε_θ(x_t, t)$ is the model's noise prediction.


**Why Involve Diffusion Models for Unified Multimodal Model?**

* **Modeling Spatial Distributions**: Diffusion models are well-suited for modeling complex spatial data distributions, such as images and videos, by gradually refining noise into structured outputs

*  **Faster Inference Compared to Autoregressive Models**: Diffusion models often have faster inference times because they generate images or videos in parallel, unlike autoregressive models that generate step-by-step.



## Unified Multimodal Model

### Preliminary
Unified multimodal models represent a pivotal advancement in artificial intelligence, aiming to integrate and process multiple data modalities (*e.g.,* text, images, audio, and video) within a single model framework. These models are designed for understanding and generating across modalities, offering flexibility and efficiency that surpass traditional task-specific or modality-specific approaches. 


**Key Concepts in Unified Multimodal Models:**

* **Multimodal Understanding**: 
Unified models are trained to understand relationships and interactions across multiple modalities (*e.g.,* text, image, audio). For example, given a pair $(x_{\text{image}}, x_{\text{text}})$, the model learns a joint representation $z$ such that:

$$
z = f(x_{\text{image}}, x_{\text{text}})
$$

* **Multimodal Generation**: 
These models synthesize cross-modal outputs. For instance, generating text $x_{\text{text}}$ from image $x_{\text{image}}$ can be formulated as:

$$
x_{\text{text}} = g(x_{\text{image}})
$$

* **Cross-Modal Alignment**:
A key challenge is aligning feature spaces of diverse modalities into a unified latent space. Formally, given $x_i$ and $y_j$ from different modalities, the goal is to minimize their alignment loss $\mathcal{L}_{\text{align}}$:

$$
\mathcal{L}_{\text{align}} = \| f(x_i) - f(x_j) \|_2^2
$$

* **Model Architectures**:

*1. Autoregressive Models*.  Predict the next token or step $x_{t+1}$ based on past inputs $x_{\leq t}$, examples include EMU3.


*2. Mixed Architectures*. Combine autoregressive and diffusion methods, leveraging their respective strengths, as seen in models like Show-O.


<aside class="l-body box-note" markdown="1">
Based on model architectures, unified multimodal models can be categorized into two primary divisions: single models and multi-expert models. 
Further, single models can be classified into pure autoregressive architectures and mixed architectures:

{% include figure.html path="assets/img/2025-04-28-unified-models/flowchart.png" class="img-fluid" %}
</aside>




For the single-model paradigm, existing methods can primarily be categorized into two aspects: model architecture and feature representation, as shown in the table below:

  Architecture\Feature     | **Discrete** | **Continuous** 
--- | --- | ---
**Autoregressive** | *Chameleon*<d-cite key="chameleon2024"></d-cite>;*EMU3*<d-cite key="wang2024emu3"></d-cite> | *MMAR*<d-cite key="yang2024mmar"></d-cite>
**AR+Diffusion** | *Show-O*<d-cite key="xie2024showo"></d-cite> | *Transfusion*<d-cite key="zhou2024transfusion"></d-cite>;*MonoFormer* 



### Autoregressive Models v.s. Mixed Architectures (AR+Diffusion)
<!-- https://poloclub.github.io/transformer-explainer/ -->

The fundamental difference between Autoregressive Models (*e.g.,* EMU3, Chameleon) and AR+Diffusion Models (*e.g.,* Show-O, Transfusion) lies in their approach: Autoregressive Models still sequentially predict the next token across all modalities, while AR+Diffusion models combine autoregressive generation for discrete tokens (*e.g., text*) with diffusion processes for continuous data (*e.g.,* images and videos). The modeling differences between these two approaches are as follows:


| **Aspect**             | **Autoregressive Models**                                | **AR+Diffusion Models**                              |
|-------------------------|---------------------------------------------------------|-----------------------------------------------------|
| **Generative Strategy**  | Iteratively predicts the next token.                    | Predicting all tokens at once by iterative denoising. |
| **Generative Fidelity** | Effective for token-based sequence. (*e.g.,* text).  | Superior for high-fidelity image and video generation.    |
| **Unified Framework**   | Simplifies multimodal unification via tokenization.     | Balances autoregressive and diffusion processes.        |
| **Training Complexity** | Relatively efficient.                                   | More computationally demanding due to diffusion.    |
| **Scalability**         | Scales well to large multimodal datasets.               | Requires careful balancing of AR and diffusion.     |



### Discrete v.s. Continuous

In a unified model, **discrete values** refer to categorical data (*e.g.,* tokens or words) predicted sequentially, while **continuous values** involve real-valued data (*e.g.,* pixels or audio signals) that are refined through a denoising process to generate high-quality outputs.

| **Aspect**                  | **Discrete (e.g., Text)**                          | **Continuous (e.g., Images, Audio)**                 |
|-----------------------------|----------------------------------------------------|------------------------------------------------------|
| **Data Type**                | Categorical (text tokens).                        | Real-valued (pixels, audio signals, etc.).           |
| **Primary Focus**            | Token-level generation (*e.g.,* text prediction).   | Continuous signal refinement (*e.g.,* image generation). |



<aside class="l-body box-note" markdown="1">

In autoregressive models, encoding an image with discrete values represents pixels or features as categorical indices (*e.g.,* tokens from a **codebook**), while encoding with continuous values directly processes real-valued pixels or features:

{% include figure.html path="assets/img/2025-04-28-unified-models/xfxc6-p9yvb.png" class="img-fluid" %}
</aside>






<!-- https://poloclub.github.io/transformer-explainer/ -->


## Challenges 


## Conclusion and Future Vision

