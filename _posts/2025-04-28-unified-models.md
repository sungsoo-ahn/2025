---
layout: distill
title: Towards Unified Multimodal Models:Trends and Insights
description: Recent advancements in unified models for multimodal understanding and generation, including works like Transfusion, ImageBind, and EMU3, highlight the trend towards frameworks capable of diverse tasks. These models typically fall into two categories:single models, which use a unified architecture to process multiple modalities, and multi-expert models, where specialized sub-models are used for each modality. Multi-expert models include different alignment strategies such as Image-Centric Alignment (e.g., ImageBind), Text-Centric Alignment (e.g., TextBind), and others, each focusing on aligning specific modalities for more flexible integration. The single models explore techniques like pure autoregressive methods, diffusion-based approaches, or a combination of both. This blog provides a comprehensive overview of unified multimodal models, reviewing current developments and discussing key design principles, including the use of autoregressive and diffusion mechanisms.
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
  - name: Background
    subsections:
    - name: Autoregressive Model
    - name: Diffusion Model
    - name: Preliminary of Unified Models
    - name: Taxonomy of Unified Models
  - name: Single Model for Unified Multimodal Models
    subsections:
    - name: Autoregressive Models v.s. Mixed Architectures (AR+Diffusion)
    - name: Discrete v.s. Continuous
  - name: Multi-Experts for Unified Multimodal Models
    subsections:
    - name: Image-Centric Alignment
    - name: Text-Centric Alignment
    - name: Generalized Alignment
  - name: Challenges in Unified Multimodal Models
  - name: Conclusion

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


<blockquote>
“Unity is strength... when there is teamwork and collaboration, wonderful things can be achieved.” — Mattie Stepanek
</blockquote>

{% include figure.html path="assets/img/2025-04-28-unified-models/21.07.44.png" class="img-fluid" %}

In recent years, the field of multimodal understanding and generation has seen significant advancements, particularly with the rise of unified models capable of addressing a wide range of tasks. Notable examples include Meta’s Transfusion<d-cite key="zhou2024transfusion"></d-cite>, NUS’s Show-o<d-cite key="xie2024showo"></d-cite>, and BAAI’s EMU3<d-cite key="wang2024emu3"></d-cite>, which have set the stage for a growing trend: the development of single models that can process and generate information across multiple modalities, such as text, images, and more. These unified models utilize a variety of techniques, including pure autoregressive methods, diffusion-based approaches, or even hybrid combinations of both.



Unity concept resonates deeply in the context of unified models for multimodal understanding and generation. This blog aims to explore the recent developments in unified multimodal models, reviewing their current state and outlining the future directions for research and application. We will delve into the principles behind autoregressive and diffusion models, shedding light on their unique characteristics and how they can be effectively combined. Ultimately, we will discuss whether the future of unified models lies in one of these paradigms, or if a hybrid approach holds the key to even greater advancements.


 

## Background

### Autoregressive Model
Autoregressive (AR) sequence generation is a method where a sequence is generated token by token, with each token predicted based on the preceding ones. 
This approach models dependencies within the sequence by conditioning each output on prior outputs, effectively capturing the structure and patterns of sequential data.

**Definition**. For a data sequence $$(x_1, x_2, ..., x_{T})$$, an autoregressive model represents the joint probability as:

$$
P(x_1, x_2, ..., x_T) = \prod_{t=1}^{T} P(x_t | x_1, x_2, ..., x_{t-1})
$$

This formula captures the essence of autoregressive generation: predicting each token based on the prior sequence of tokens.

**Why Choose Autoregressive Models for Unified Multimodal Model?**

* **Unified data representation.**: They can integrate different modalities (*e.g.,* text, images, audio) by converting them into a unified sequence of tokens, simplifying model design.


* **Easier to Scale Up**: Autoregressive models align well with scaling laws, making them easier to scale up in terms of model size, data, and computational resources.

* **Scalable architecture supported by strong infra**. Standing on the shoulders of giants (LLMs). Both academia and industry have relatively sufficient experiences on training and scaling AR models, in terms of model size, data, and computational resources.



**Autoregressive for Vision**. The process for generating an image can be broken down pixel-by-pixel or patch-by-patch in a raster-scan order, where each pixel/patch is conditioned on previously generated content. Autoregressive models are proven to be simple, interpretable, and effective in processing language. Images, however, are not sequential. Besides, treading image as a flat sequence means that the auto-regressive sequence length (and the computation) grows quadratically<d-cite key="chang2022maskgit"></d-cite>. This misalignment pose challenges of effectiveness of efficiency on AR model, which are central to understanding their role in unified multimodal generation tasks.





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

*  **Faster Inference Compared to Autoregressive Models**: Diffusion models often have faster inference times because they generate images or videos in parallel with only few steps, unlike autoregressive models that generate token-by-token.





### Preliminary of Unified Models
Unified multimodal models represent a pivotal advancement in artificial intelligence, aiming to integrate and process multiple data modalities (*e.g.,* text, images, audio, and video) within a single model framework. These models are designed for understanding and generating across modalities, offering flexibility and efficiency that surpass traditional task-specific or modality-specific approaches. 


**Key Concepts in Unified Multimodal Models:**

* **Multimodal Understanding**: 
Unified models are trained to understand relationships and interactions across multiple modalities (*e.g.,* text, image, audio). For example, given a pair $(x_{\text{image}}, x_{\text{text}})$, the model learns a joint representation $z$ such that:

$$
z = f_{\text{Unified Model}}(x_{\text{image}}, x_{\text{text}})
$$

* **Multimodal Generation**: 
These models synthesize cross-modal outputs. For instance, generating text $x_{\text{video}}$ from text $x_{\text{text}}$ and image $x_{\text{image}}$ can be formulated as:

$$
x_{\text{video}} = f_{\text{Unified Model}}(x_{\text{text}},x_{\text{image}})
$$

* **Cross-Modal Alignment or Generative Alignment**:

A key design choice in unified models is the method of aligning modalities:

1) **Cross-Modal Alignment**. Aligning feature spaces of diverse modalities into a unified latent space. Formally, given $x_i$ and $y_j$ from different modalities, the goal is to minimize their alignment loss $\mathcal{L}_{\text{align}}$:

$$
\mathcal{L}_{\text{align}} = \| f_{\text{Unified Model}}(x_i) - f_{\text{Unified Model}}(x_j) \|_2^2
$$

2) **Generative Alignment**. Models such as Show-O and Transfusion bypass explicit alignment by directly using generative loss to learn the relationships between modalities. This approach models the inter-modality relationships through tasks like sequence prediction or output reconstruction, focusing on generation rather than explicit feature alignment. For a pair of modalities $x_{\text{src}}$ and $x_{\text{tgt}}$, the loss function is a weighted sum of generative losses for each task:

$$
\mathcal{L}_{\text{unified}} = \sum_{k=1}^K \lambda_k \mathcal{L}_{\text{gen}}^k
$$

Where $K$ is the number of modality pairs, $\mathcal{L}_{\text{gen}}^k$ is the generative loss for the $k$-th modality pair, $\lambda_k$ is a weight balancing the importance of each task.



* **Model Architectures**:

From the perspective of model architectures, unified multimodal systems can generally be categorized into two main types: Autoregressive Models and Mixed Architectures (AR+Diffusion):

*1. Autoregressive Models*.  Predict the next token or step $x_{t+1}$ based on past inputs $x_{\leq t}$, examples include EMU3.


*2. Mixed Architectures*. Combine autoregressive and diffusion methods, leveraging their respective strengths, as seen in models like Show-o.



### Taxonomy of Unified Models

<aside class="l-body box-note" markdown="1">
Based on model architectures, unified multimodal models can be categorized into two primary divisions: single models and multi-expert models. 
Further, single models can be classified into pure autoregressive architectures and mixed architectures:

{% include figure.html path="assets/img/2025-04-28-unified-models/flowchart.png" class="img-fluid" %}
</aside>



| Feature                  | Single Models                          | Multi-Expert Models                    |
|--------------------------|----------------------------------------|----------------------------------------|
| **Architecture**         | Unified model with shared parameters  | Specialized sub-models for each modality |
| **Scalability**          | Less flexible for adding new modalities | Easily extendable by adding new experts |
| **Performance**          | Balanced across modalities             | Optimized for each modality individually |
| **Complexity**           | Simpler pipeline                      | Higher due to fusion and coordination   |
| **Use Cases**            | General-purpose multimodal tasks       | Tasks requiring high modality-specific performance |

The key difference between **Single Models** and Multi-Expert Models lies in their approach to handling modalities. Single models use a unified architecture with shared parameters to process all modalities in a common space, offering simplicity and generalization but limited flexibility for adding new modalities. A great example of this is Chameleon, which processes all modalities using a single unified autoregressive architecture in the same feature space.

In contrast, **Multi-Expert Models** rely on specialized sub-models or experts tailored for each modality, providing better modality-specific performance and scalability. For instance, ImageBind employs separate pre-trained encoders for different modalities (*e.g.,* images, text, audio) to extract features independently and aligns them into a shared embedding space for optimization. This approach allows it to leverage the strengths of pre-trained models while maintaining flexibility for each modality. However, the reliance on multiple encoders adds complexity due to the need for fusion and alignment mechanisms.






## Single Model for Unified Multimodal Model



For the single-model paradigm, existing methods can primarily be categorized into two aspects: model architecture and feature representation, as shown in the table below:

  Architecture\Feature     | **Discrete** | **Continuous** 
--- | --- | ---
**Autoregressive** | *LWM*<d-cite key="liu2024world"></d-cite>; *Chameleon*<d-cite key="chameleon2024"></d-cite>; *VILA-U*<d-cite key="wu2024vila"></d-cite>; *EMU3*<d-cite key="wang2024emu3"></d-cite> | *MMAR*<d-cite key="yang2024mmar"></d-cite>
**AR+Diffusion** | *Show-o*<d-cite key="xie2024showo"></d-cite> | *Transfusion*<d-cite key="zhou2024transfusion"></d-cite>;*MonoFormer*<d-cite key="zhao2024monoformer"></d-cite>



### Autoregressive Models v.s. Mixed Architectures (AR+Diffusion)
<!-- https://poloclub.github.io/transformer-explainer/ -->



The fundamental difference between Autoregressive Models (*e.g.,* EMU3, Chameleon) and AR+Diffusion Models (*e.g.,* Show-o, Transfusion) lies in their approach.Autoregressive Models still sequentially predict the next token across all modalities, while AR+Diffusion models combine autoregressive generation for discrete tokens (*e.g., text*) with diffusion processes for continuous data (*e.g.,* images and videos). 

<aside class="l-body box-note" markdown="1">
Autoregressive (AR) models and mixed architectures (AR + diffusion) differ in how they handle data, particularly for high-dimensional modalities like images and videos. AR models treat all modalities uniformly by sequentially predicting the next token, which works well for temporal tasks but struggles with capturing spatial dependencies. In contrast, mixed architectures combine AR for global structure and diffusion for spatial modeling, allowing all tokens to interact during generation. This results in more coherent outputs for image and video tasks, as diffusion handles spatial distributions through parallel denoising. While AR models are simple and efficient, mixed architectures offer better performance for spatial data at the cost of increased complexity and computational demand.

{% include figure.html path="assets/img/2025-04-28-unified-models/11581732176566_.pic.jpg" class="img-fluid" %}


</aside>

Autoregressive models and mixed architectures (such as AR+Diffusion) differ primarily in the type of attention mechanisms they employ, which significantly impacts their performance and the way they handle modality alignment: 

* **AR-Causal Attention:** Using causal masks where tokens only attend to their previous tokens. This creates a strict unidirectional flow of information.

* **Diffusion-Full Attention:**  Using full attention masks (bi-directional), allowing tokens to attend to both past and future tokens. This enables a richer, more flexible way of modeling the relationships between different parts of the input. 



The modeling differences and the respective strengths and weaknesses of these two approaches are outlined below:

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
| **Training Complexity**      | High due to long sequences.                         | Moderate but requires sophisticated alignment.         |
| **Encoding Approach**        | Requires codebook for quantization, potential errors.      | No codebook, direct continuous embeddings, avoids quantization errors. |


<aside class="l-body box-note" markdown="1">

In autoregressive models, encoding an image with discrete values represents pixels or features as categorical indices (*e.g.,* tokens from a **codebook**), while encoding with continuous values directly processes real-valued pixels or features:

{% include figure.html path="assets/img/2025-04-28-unified-models/xfxc6-p9yvb.png" class="img-fluid" %}
</aside>

### Disscusion for Single-Model Paradigm

**Autoregressive-based Models with Discrete Valued Tokenizer**.
Autoregressive-based models with discrete-valued tokenizers, including Chameleon, EMU3, leverage a unified tokenization framework to process diverse modalities such as text, images, and video. These models transform multimodal inputs into discrete token sequences, enabling a shared representation for both generation and understanding tasks.

*Advantage - Unified Tokenization*. All input data, regardless of modality, is converted into discrete tokens using techniques like VQ-VAE<d-cite key="van2017neural"></d-cite>. This allows seamless integration of text, image, and video data into a single autoregressive sequence.

*Limitations - Inference Speed*. Token-by-token generation leads to slower inference times, particularly for long sequences like video.

**Autoregressive-based Models with Continuous Valued Tokenizer**.
Autoregressive models with continuous-valued tokenizers, such as MMAR, introduce an alternative approach to unified multimodal modeling by representing data as continuous latent representations instead of discrete tokens. This method offers unique advantages in modeling high-dimensional data like images and videos while maintaining the flexibility of autoregressive frameworks.

*Advantage - Continuous Representations:*. Unlike discrete tokenization, these models use continuous embeddings to represent inputs, providing a richer and more compact encoding of complex modalities like video and high-resolution images.

*Limitations - Task Flexibility*: While excellent for understanding tasks, these models may require additional tuning to handle diverse generative tasks effectively.

**Mixed Architectures with Discrete Valued Tokenizer.**
Mixed architectures that combine autoregressive (AR) and diffusion models, such as Show-o, represent an innovative approach to unified multimodal modeling. These architectures leverage the strengths of both AR and diffusion frameworks while operating on discrete-valued tokenized inputs, aiming to address the limitations of each individual method.



*Advantage - Unified Tokenization for Text and Image/Video*. Both AR and diffusion processes operate on tokenized representations, enabling seamless integration of diverse modalities such as text, images, and video within the same framework. Additionally, diffusion models excel at modeling spatial distributions, making them particularly effective for image and video generation tasks. Furthermore, inference with diffusion models tends to be faster because they process data in parallel, unlike autoregressive models that predict tokens sequentially.

*Limitations - Computational Overhead and Model Complexity*. Combining AR and diffusion significantly increases training and inference costs due to the sequential nature of AR and the iterative steps of diffusion. The hybrid approach introduces architectural complexity, making optimization and implementation more challenging.

**Mixed Architectures with Continuous Tokenizer**.
Mixed architectures utilizing continuous tokenizers, such as Transfusion and MonoFormer, represent a cutting-edge approach to unified multimodal modeling. By combining autoregressive (AR) and diffusion frameworks with continuous latent representations, these models aim to balance efficiency, scalability, and high-quality performance across generation and understanding tasks.

*Advantage - Continuous Tokenization and Diffusion Model*. Inputs are represented as compact continuous-valued embeddings rather than discrete tokens, offering a more flexible and efficient approach to handling high-dimensional multimodal inputs. Additionally, diffusion models, with their iterative denoising process, are particularly well-suited for generation tasks in image and video modalities.

*Advantage - Training Complexity*. The hybrid nature of these architectures requires careful optimization, as interactions between AR and diffusion components in continuous space can be challenging to balance.




## Multi-Experts for Unified Multimodal Models

Unlike single, monolithic models trained to handle all modalities and tasks simultaneously, **multi-expert architectures** offer an alternative approach: leveraging specialized expert modules that align, process, and fuse information across diverse modalities. These architectures not only enable task-specific optimization but also facilitate the integration of pre-trained expert models, such as incorporating external capabilities into frameworks like ImageBind. Multi-experts are typically categorized based on their alignment focus: Image-Centric Alignment, Text-Centric Alignment, and Generalized Alignment methods.

| **Aspect**                 | **Image-Centric (e.g., ImageBind<d-cite key="girdhar2023imagebind"></d-cite>)**         | **Text-Centric (e.g., TextBind<d-cite key="li2023textbind"></d-cite>; *SEED-X*<d-cite key="ge2024seed"></d-cite>; *LaVIT*<d-cite key="jin2024unified"></d-cite> )**            | **Generalized (e.g., UniBind<d-cite key="lyu2024unibind"></d-cite>)**                |
|----------------------------|--------------------------------------------|---------------------------------------------|-----------------------------------------------|
| **Alignment Focus**         | Visual-first                              | Text-first                                  | Balanced across all modalities                |
| **Integration Capability**  | Fuses pre-trained visual-centric models   | Leverages pre-trained language models       | Incorporates multi-expert pre-trained modules |
| **Strengths**               | Robust spatial and visual correlations    | Strong text-based reasoning and generation  | Versatile, supports diverse tasks             |
| **Limitations**             | Limited in text/audio-heavy tasks         | Struggles with purely visual or auditory tasks | Increased computational complexity            |


### Image-Centric Alignment

<!-- Image-centric alignment models prioritize visual data as the foundational modality for cross-modal interactions, aligning other modalities such as text, audio, or motion with visual embeddings. -->

Image-centric alignment model refers to the use of images as the core pivot to connect various data modalities. Images are highly structured and versatile representations that can be naturally associated with other modalities like text, audio, or sensor data. 


<aside class="l-body box-note" markdown="1">

ImageBind<d-cite key="girdhar2023imagebind"></d-cite> is a multimodal learning model introduced by Meta that aligns different modalities (*e.g.,* images, text, audio) into a shared image embedding space. Its key innovation lies in binding various modalities with images as the central anchor, enabling seamless interaction across domains even without paired training data for every modality pair. As shown in the image below, the illustration is sourced from ImageBind:

{% include figure.html path="assets/img/2025-04-28-unified-models/23.44.15.png" class="img-fluid" %}
</aside>

Here’s how ImageBind achieves this:

* **Cross-Modal Alignment**: Each modality (text, audio, sensor data, etc.) is independently encoded into the same shared embedding space. The model does not require direct pairwise data between all modalities. Instead, it binds each modality to images, which indirectly links the modalities.

* **InfoNCE Loss**: InfoNCE loss is used to align embeddings of different modalities. For instance, it maximizes the similarity between embeddings of an image and its corresponding text while minimizing the similarity with unrelated texts.

The InfoNCE loss is defined as:

$$
\mathcal{L}_{\text{InfoNCE}} = - \log \frac{\exp(\text{sim}(q, k^+)/\tau)}{\sum_{i=1}^N \exp(\text{sim}(q, k_i)/\tau)}
$$

Where:
- $\( \text{sim}(q, k) \)$: Similarity function, such as cosine similarity.
- $\( \tau \)$: Temperature parameter that controls the sharpness of the distribution.
- $\( N \)$: Total number of candidates (including both positive and negative samples).




### Text-Centric Alignment
Text-Centric Alignment is an approach where text embeddings act as the anchor or central hub for aligning different modalities like images, audio, and video. It relies on pre-trained large language models (LLMs) to generate text representations that allow other modalities to be mapped into a common embedding space. The alignment between modalities is achieved by projecting them into the text space and optimizing for similarity.

<aside class="l-body box-note" markdown="1">


TextBind<d-cite key="li2023textbind"></d-cite> is a multimodal alignment framework that uses text embeddings as the central anchor, allowing other modalities like images and audio to be aligned in a shared semantic space. As shown in the image below:


{% include figure.html path="assets/img/2025-04-28-unified-models/1732205074481.jpg" class="img-fluid" %}
</aside>

Text-Centric Alignment Formulation:

* **1. Text Embedding Space:** Let $T$ represent the text modality, which is encoded by a pre-trained language model to produce an embedding vector $t$. The text embedding $t$ is:

$$
\mathbf{t} = \text{LLM}(T)
$$

* **2. Modality Embedding:** Other modalities, such as images $I$, audio $A$, or video $V$, are encoded into their respective embedding spaces using dedicated encoders. For instance, the image $I$ is encoded as $\mathbf{i}$ using a vision model:

$$
\mathbf{i} = \text{VisionEncoder}(I)
$$

Similarly, audio $A$ is encoded as $\mathbf{a}$ using an audio encoder:

$$
\mathbf{a} = \text{AudioEncoder}(A)
$$

* **2. Aligning Modalities with Text:** Once we have embeddings for both text and the other modalities, the goal is to align them in a shared embedding space. This alignment can be done using a **contrastive loss** or other similarity-based loss functions. The alignment objective is:

$$
\mathcal{L}_{\text{align}} = - \log \frac{\exp(\text{sim}(\mathbf{t}, \mathbf{m}) / \tau)}{\sum_{i=1}^N \exp(\text{sim}(\mathbf{t}, \mathbf{m}_i) / \tau)}
$$

Where $m$ represents the embedding of a modality (image, audio, etc.). $\text{sim}()$ is the similarity function (*e.g.,* cosine similarity). $\tau$ is the temperature hyperparameter that controls the sharpness of the distribution. *N* is the number of possible modality samples.

### Generalized Alignment
Generalized Alignment is an approach to multimodal learning that does not center any single modality, like text or image, but instead creates a unified space for all modalities based on a knowledge base or feature-centered approach. This method aims to align different modalities by utilizing shared characteristics derived from a broader knowledge base, rather than anchoring on one modality's features.

In generalized alignment, the learning model is trained to map various modalities (e.g., text, images, audio) into a shared feature space without giving preference to one modality over others. The alignment is based on common semantic features or concepts that exist across modalities, allowing the model to learn more flexible, cross-modal relationships.


<aside class="l-body box-note" markdown="1">

UniBind<d-cite key="lyu2024unibind"></d-cite> is an example of a generalized alignment framework. Unlike traditional approaches, UniBind does not use text or image as the central anchor for aligning modalities. Instead, it builds alignment using a knowledge base or feature-centered representation. As shown in the image below:


{% include figure.html path="assets/img/2025-04-28-unified-models/1732206695874.jpg" class="img-fluid" %}
</aside>

Formulation for Generalized Alignment:

* **1. Feature Representations:** Let $M_i$ represent the embeddings for each modality $i$, where $M_i$ could be the image, text, or audio features. The model aims to align all modality embeddings $M_i$ into a common shared space $F$ based on their knowledge base features:

$$
M_i = \text{FeatureExtractor}_i(M_i)  
$$

* **2. Knowledge Base Representation:** The knowledge base $K$ represents a shared space that captures the common semantic features of all modalities:

$$
K = \text{KnowledgeBase}(F)  
$$

* **3. Alignment Objective:** The learning objective is to map each modality embedding to the knowledge base space $K$ while maintaining their relationships:

$$
L_{\text{align}} = - log \frac{\text{exp}(\text{sim}(M_i, K_i) / τ)}{\sum_{j=1}^{N} \text{exp}(\text{sim}(M_i, K_j) / τ)}
$$

Where $M_i$ is the embedding of modality $i$ (image, text, audio). $K_i$ is the corresponding knowledge base embedding for the modality. $\text{sim}(., .)$ is the similarity function (*e.g.,* cosine similarity). $τ$ is the temperature hyperparameter that controls the sharpness of the distribution. $N$ is the number of modality samples.


<!-- https://poloclub.github.io/transformer-explainer/ -->


## Challenges in Unified Multimodal Models 

### 1. Technical Challenges

* **Architecture Design Uncertainty**: Autoregressive or Autoregressive + Diffusion Hybrid or Alternative Architectures.

* **Cross-modal Representation Alignment**: Aligning representations across diverse modalities (e.g., text, image, video, and audio) is non-trivial due to their fundamentally different structures.


### 2. Data Challenges

* **Multimodal Dataset Diversity**: Collecting and curating datasets that cover all relevant modalities, tasks, and domains is resource-intensive and often biased toward specific modalities (e.g., text-heavy datasets).

* **Data Quality and Noise**: Large multimodal datasets often contain noisy or misaligned data, leading to suboptimal performance or spurious correlations.


* **Modality Imbalance**: Some modalities, such as text, have abundant training data, while others, like video or audio, are relatively underrepresented, making balanced training challenging.

### 3. Task-Specific Challenges 

* **Balancing Task Generality and Specificity**: In certain specialized downstream tasks, the performance of current unified models falls significantly short compared to large models specifically designed and optimized for those tasks. For example, models like EMU3 and Chameleon currently appear to be far less capable than SORA and other specialized video generation models.

* **Output Modality Compatibility**: Generating compatible outputs across multiple modalities (*e.g.,* synchronizing video frames with audio and text) can be difficult.


## Conclusion

Unified multimodal models hold immense potential to revolutionize AI by integrating diverse modalities and tasks into a single framework, enabling unprecedented versatility and generalization.  However, realizing this vision comes with significant challenges, including unresolved trade-offs in architecture design, the complexity of aligning diverse data, and balancing efficiency with scalability.  While current models demonstrate promising progress, their performance on specialized tasks often lags behind domain-specific models, highlighting the need for further innovation.  Future advancements will likely depend on the development of novel architectures that harmonize flexibility, task-specific optimization, and computational efficiency, paving the way for more robust and capable unified models.