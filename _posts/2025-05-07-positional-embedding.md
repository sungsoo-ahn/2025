---
layout: distill
title: 'Positional Embeddings in Transformer Models: Evolution from Text to Vision Domains'
description: Positional encoding has emerged as a pivotal component in transformer architectures, addressing their inherent permutation invariance and enabling them to capture sequential relationships in data. This blog post explores positional encoding mechanisms in-depth, highlighting their crucial role in standard transformers and their application to 2D data in Vision Transformers (ViT). We analyze two modern approaches—ALiBi (Attention with Linear Biases) and RoPE (Rotary Position Embedding)—delving into their distinct methodologies for addressing sequence length extrapolation during inference, a critical challenge in transformers. We also analyze these approaches' underlying similarities and differences, examining how they affect transformer performance across various domains.
date: 2025-05-07
future: true
htmlwidgets: true
hidden: false

# Anonymize when submitting
# authors:
#   - name: Anonymous

authors:
  - name: anonymous

# must be the exact same name as your blogpost
bibliography: 2025-05-07-positional-embedding.bib  

# Add a table of contents to your post.
#   - make sure that TOC names match the actual section names
#     for hyperlinks within the post to work correctly. 
#   - please use this format rather than manually creating a markdown table of contents.
toc:
  - name: Background
  - name: Preliminaries
    subsections:
    - name: Self-Attention Mechanism
    - name: Multi-Head Attention
  - name: Absolute Positional Embeddings
  - name: Relative Positional Embeddings
  - name: Positional encoding in Vision Transformers
  - name: Rotary Positional Embeddings
    subsections:
    - name: The mechanism of RoPE
    - name: The two-dimensional case
    - name: The general D-dimensional case
  - name: RoPE in Vision Trasformers
    subsections:
    - name: Axial 2D Rope
    - name: Mixed RoPE
  - name: Attention with Linear Biases (Alibi)
    subsections:
    - name: Mechanism and Architecture
    - name: What is m?
    - name: Theoretical Foundation and Advantages
    - name: Performance and Extrapolation
  - name: Alibi in Vision Transformers
    subsections:
    - name: Two-Dimensional Adaptation
  





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
  .caption { 
    font-size: 80%;
    line-height: 1.2;
    text-align: left;
  }
---

## Background

"Attention Is All You Need," introduced by Vaswani et al. in 2017, revolutionized the field of natural language processing and computer vision by proposing a purely attention-based model, the Transformer, eliminating the need for recurrent or convolutional networks.

Later, Dosovitskiy et al. (2021) applied this concept to computer vision, introducing Vision Transformers (ViTs).On its own, the Transformer architecture is position-invariant, i.e., it processes its input as an unordered set. Unlike RNNs, designed to handle ordered sequences, or CNNs, which leverage translation equivariance and locality, transformers lack the inherent ability to capture sequential patterns. This is because the self-attention mechanism is independent of the token index.

Vaswani et al. (2017) introduced positional encoding to address this lack of inductive bias. Since then, researchers have explored various methods to encode positional information directly into tokens or within the self-attention interaction step. The following sections will discuss the broad advancements of positional encoding methods in 1D and their extension for 2d tasks in Vision Transformers.

## Preliminaries

### Self-Attention Mechanism:

The self-attention mechanism processes sequences by representing each element as a vector of embedding dimension $$d$$. For a sequence of length $$L$$ and embedding dimension $$d$$, the input is represented as a matrix $$X \in \mathbb{R}^{d \times L}$$.

The mechanism employs three learnable weight matrices: $$W_Q,W_K,$$ and $$W_V$$ $$\in \mathbb{R}^{d \times d_k}$$, where $$d_k$$ represents the dimensionality of the projected spaces. These matrices transform the input into queries, keys, and values respectively:

$$
Q = W_Q^T \, X  \;\;\;\;\;\;\;\; K = W_K^T \, X  \;\;\;\;\;\;\;\; V = W_V^T \, X
$$

The attention matrix is computed using the following formula:

$$
\text{Attention}(Q, K) = \text{softmax}(QK^T/√d_k)
$$

$$
\mathbf{Z} = \text{Attention}(Q, K) \cdot V
$$

With respect to each token, let $$x_i$$ be the context vector for $$i^{\text{th}}$$ token, the corresponding query, key and value vectors, $$\mathbf{q_i}, \mathbf{k_i}$$ and $$\mathbf{v_i}$$ are defined as follows:

$$
\mathbf{q_i} = W_Q^T \:x_i \; \; \; \; \; \; \mathbf{k_i} = W_K^T \:x_i \; \; \; \; \; \; \mathbf{v_i} = W_V^T \:x_i
$$

and $$N$$ being total number of tokens the self attention is formulated as:

$$
a_{m,n} = \frac{\exp\left( \frac{\mathbf{q}_m^\top \mathbf{k}_n}{\sqrt{d}} \right)}{\sum_{j=1}^{N} \exp\left( \frac{\mathbf{q}_m^\top \mathbf{k}_j}{\sqrt{d}} \right)}
$$

$$
\mathbf{z}_m = \sum_{n=1}^{N} a_{m,n} \mathbf{v}_n
$$

### Multi-Head Attention

In multi-head attention, the computation is extended across h parallel attention heads. Each $$head_i$$ operates as:

$$
\text{head}_i = \text{Attention}(Q\,W_{Q_i}, K\,W_{K_i}, V\,W_{V_i})
$$

The outputs from all heads are concatenated and projected back to the original dimensionality:

$$
\text{MultiHead}(Q, K, V) = \text{Concat}(\text{head}_1,\ldots,\text{head}_h)\:W_O

$$

where $$W_O \in \mathbb{R}^{hd_k \times d}$$ is a learned matrix that maps the concatenated attention outputs back into the original space.

## Absolute Positional Embeddings

Absolute Positional embedding was introduced by (Vaswani et al. 2017) in their famous 2017 paper Attention is All You Need. It involves the direct addition of positional embeddings into the embedding vector. These encodings are injected only once in the embedding vector before passing them into the transformer block. Mathematically, it’s equivalent to 

$$
x'_i := x_i + p_i
$$

Where $$p_i$$ is the positional embedding vector and $$x_i$$ is the context vector corresponding to the $$i_{th}$$ token. Note that $$x '_i,x_i ,p_i \in \mathbb{R}^d$$. Vaswani el at. 2017, proposed following formula for $$p_i$$

$$
\left\{
\begin{array}{ll}
   p_{i,2t} =  \sin(i/10000^{2t/d}) &  \\
   p_{i,2t+1} = \cos(i/10000^{2t/d}) \\
\end{array} 
\right.
$$

where $$p_{i,t}$$ is the  $$t^{th}$$ element in the $$p_i$$ vector.

<div class="l-page">
  <iframe src="{{ 'assets/html/2025-05-07-positional-embedding/sinusoidal.html' | relative_url }}" frameborder='0' scrolling='no' height="600px" width="100%"></iframe>
</div>
<div class="caption">
    Figure 1: Plot of sinusoidal positional embeddings, $i$ (position in sequence) vs $t$ (index in embedding vector). Each vertical stripe can be seen as an absolute position embedding vector to be added.
</div>
One might question why a function of sines and cosines was chosen to represent positional encodings in a sequence—why not any other arbitrary function? The choice of sinusoidal functions was actually a strategic one, offering several advantages that benefit transformer models:

- Firstly the values are normalized between $$[-1,1]$$ allowing the model to learn parameters easily.
- The distance between neighboring positions is symmetrical and decays naturally with position.
- The formulation shows that every two consecutive elements in the embedding $$p_i$$ share the same frequency denominator. For a pair with frequency $$w_k$$, we can demonstrate that

$$
\begin{array}{c}p_i \cdot p_{i+\phi} = \begin{bmatrix}
\sin(iw_k) \\
\cos(iw_k)
\end{bmatrix} \cdot \begin{bmatrix}
\sin((i+\phi)w_k) \\
\cos((i+\phi)w_k)
\end{bmatrix} \\[12pt] = \cos((i+\phi)w_k)\cos(iw_k) + \sin((i+\phi)w_k)\sin(iw_k) 
= \cos(\phi \: w_k)\end{array}
$$

Hence the dot product $$p_i \cdot p_{i+\phi}$$ is independent of position $$i$$, and relative position $$\phi$$ is retained. This consistency allows the model to capture regular intervals and patterns, making it especially effective for tasks requiring sequential dependencies.

The authors hypothesized that sinusoidal positional encoding would enable the model to learn relative positions naturally, thanks to the properties outlined above, and might also allow it to extrapolate to sequence lengths longer than those seen during training. This rationale led to sinusoidal positional embeddings being one of the first methods adopted in transformer architectures.

However, subsequent research challenged these assumptions. Studies found that absolute sinusoidal positional encoding was not well-suited for capturing relative positional information effectively (Shaw et al., 2018). Moreover, it struggled with extrapolation, leading to poor performance on sequence lengths longer than those encountered in training (Press et al., 2021).

In addition to sinusoidal positional embedding, Vaswani et al. also explored learned positional encoding. They explored using a set of trainable vectors $$p \in \mathbb{R}^{d \times L}$$ $$\{ p_t \} _{t=1}^L$$ , $$L$$ represents the maximum sequence length as positional embeddings. However, this approach didn't yield significant performance improvements over sinusoidal positional embedding. Moreover, the upper bound $$L$$ limited the method's ability to extrapolate to sequences longer than $$L$$ without resorting to interpolation techniques.

## Relative Positional Embeddings

Unlike absolute positional encodings, which create embeddings for each position independently, relative positional embeddings focus on capturing the pairwise relationships between tokens in a sequence. Also rather than directly adding the embeddings to the context vectors, the relative positional information is added to keys and values during the attention calculation. Hence - 

$$
\mathbf{q_m} = W_Q \:x_m \; \; \; \; \; \; \mathbf{k_n'} = W_K \:(x_n+ \tilde{p}_r^k)\; \; \; \; \; \; \mathbf{v_n'} = W_V \:(x_n+ \tilde{p}_r^v)
$$

where $$\tilde{\mathbf{p}}_r^k, \tilde{\mathbf{p}}_r^v \in \mathbb{R}^d$$ are trainable relative positional embeddings and $$r$$ represents the relative distance between the two tokens at positions $$m$$ and $$n$$.

## Positional encoding in Vision Transformers

Motivated by the success of transformers in processing one-dimensional textual data, Dosovitskiy et al. (2021) extended this approach to two-dimensional image data by introducing the Vision Transformer (ViT). In this seminal work, they proposed a scalable transformer architecture with minimal adjustments to handle image data. The authors explored several approaches to positional embeddings for 2D images, including standard 1D learnable positional encoding based on the raster order of patches, spatially-aware 2D learned positional encoding and relative positional encoding. Their experiments revealed a significant performance gap between models without positional embeddings and those using any form of positional encoding. Interestingly, they observed little to no difference in performance between the different methods of encoding positional information. They speculated that because the transformer encoder operates on patch-level inputs rather than individual pixels, how spatial information is encoded becomes less critical; however, as will be seen subsequently in our blog, more advanced positional methods proposed for 1-dimensional data also lead to better performance in ViTs.

## Rotary Positional Embeddings

Traditional positional encoding methods have primarily involved adding positional information directly to context embeddings. While this approach introduces a positional bias, it can limit the depth of interaction with attention weights, potentially underutilizing valuable relative position information. Additionally, since this information is added after the query-key multiplication, it does not contribute to the critical query-key similarity calculations that influence attention mechanisms.

Enter RoFormer, which introduced a novel approach with *Rotary Positional Embeddings (RoPE)*. Unlike conventional methods that add positional encodings to word embeddings, RoPE applies a *rotary transformation* to encode relative position information. This approach encodes relative position information without hindering the interaction of query-key pairs.

### The mechanism of RoPE

Given the limitations of directly adding positional information, we need a formulation that encodes relative position information within the attention mechanism by applying a transformation solely to the query and key vectors.

Let token at position $$m$$ has the context embeddings $$x_m$$ , we define function $$f(x_m,m)$$ which takes the context vector $$x_m$$ and its position in the sequence $$m$$ and returns a transformed embedding having the positional information. We want the function $$f$$ to be such that the dot product of two embeddings depends on their relative position to each other i.e.

$$
f(x_m,m)\cdot f(x_n,n) = g(x_m,x_n,m-n)
$$

### The two-dimensional case

Consider the basic case where context vector $$x_m = \begin{pmatrix} x_m^{(1)} &x_m^{(2)}\end{pmatrix}^T$$ is 2 dimensional and in complex space it can be represented as $$x_m^{(1)} + ix_m^{(2)}$$ . For the exponential function, we know that $$e^m \cdot e^n = e^{m+n}$$ which is very similar to the properties we want from the function $$f$$. Thus for defining $$f$$, we can multiply $$f$$ $$x_m$$ with $$e^m$$ , but there is one problem that is explosion of magnitude as the sequence grows. Hence $$e^{im\theta}$$ is an ideal choice for multiplication, as it doesn’t cause any magnitude changes and rotates the vector $$x_m$$ by an angle of $$m\theta$$.

$$
f(x_m,m) = x_m e^{im \theta}
$$

Formally, in order to encode the positional information, the key $$k_m$$ at position $$m$$ and query $$q_n$$ at position $$n$$ are multiplied with $$e^{im\theta}$$ and $$e^{in\theta}$$ respectively as a transformation. Consequently, the $$(m,n)^{th}$$ element of the attention matrix becomes - 

$$
q_n' = q_ne^{in \theta} \; \; k_m' = k_me^{in\theta} 
$$

$$
A'_{(n,m)} = \text{Re}[q^{'}_nk^{'*}_m] = \text{Re}[q_nk^*_me^{i(n−m)θ}] \;\;\;\;\;\;\;\;\;\;\;\;  \scriptstyle \text{where Re[z] is the real part of z}
$$

Hence the $$e^{i(n−m)θ}$$ in this formulation of attention is responsible injecting relative position of the tokens which are $$(n-m)$$ distance apart from each other. The formal derivation that it is a solution to equation formulated is detailed in RoFormer [2023]. Intuitively, we're rotating 2D query and key vectors in the embedding space based on their sequence position. For example, the vector corresponding to first token is rotated by θ, the second by 2θ, and so on. The approach of rotating according to the position provides several benefits as discussed below - 

{% include figure.html path="assets/img/2025-05-07-positional-embedding/RotaryPE1.png" class="img-fluid" %}
<div class="caption">
    Figure 4: Visualization of ALiBi's attention computation mechanism. The method introduces a head-specific linear bias matrix (right) that gets added element-wise to the standard query-key dot product attention scores (left). This bias injection occurs before the softmax operation, while preserving the remaining transformer architecture. The bias strength is controlled by a  $m$, which is predetermined for each attention head.
</div>

- **Stability of Vectors**: Adding tokens at the end of a sentence doesn’t affect the vectors for words at the beginning, facilitating efficient caching.
- **Preservation of Relative Positions**: If two words, say “pig” and “dog,” maintain the same relative distance in different contexts, their vectors are rotated by the same amount. This ensures that the angle, and consequently the dot product between these vectors, remains constant

### The general D-dimensional case

The idea of extending the 2D result to the d dimensional case is to break the embedding vector in various 2 dimensional chunks and rotate each one of them by a specific angle $$m\theta_i$$ where m depends on the position of token in the sequence and $$\theta_i$$ varies according to the chunk. This can be intuitively thought of as a corkscrew rotation in higher dimensions.

{% include figure.html path="assets/img/2025-05-07-positional-embedding/RotaryPE2.png" class="img-fluid" %}
<div class="caption">
    Figure 4: Visual representation of chunking of tokens and applying the rotation of different frequency $\theta_t$ on each chunk.
</div>


More formally, we divide the d dimensional space into smaller 2 dimensional subspaces, apply the rotation in those smaller subspaces and combine them again, this can be achieved through

$$
q_n = R^{d}_{\Theta,n}\:W_q\:x_n, \; \; \; \;k_m = R^{d}_{\Theta,m}\:W_k\:x_m
$$

where the transformation matrix $$R^{d}_{\Theta,m}$$ is defined as follows - 

$$
R^{d}_{\Theta,m} = 
\begin{pmatrix}
\cos m\theta_1 & -\sin m\theta_1 & 0 & 0 & \cdots & 0 & 0 \\
\sin m\theta_1 & \cos m\theta_1 & 0 & 0 & \cdots & 0 & 0 \\
0 & 0 & \cos m\theta_2 & -\sin m\theta_2 & \cdots & 0 & 0 \\
0 & 0 & \sin m\theta_2 & \cos m\theta_2 & \cdots & 0 & 0 \\
\vdots & \vdots & \vdots & \vdots & \ddots & \vdots & \vdots \\
0 & 0 & 0 & 0 & \cdots & \cos m\theta_{d/2} & -\sin m\theta_{d/2} \\
0 & 0 & 0 & 0 & \cdots & \sin m\theta_{d/2} & \cos m\theta_{d/2}
\end{pmatrix}

$$

where the frequencies $$\theta_t$$ are defined as follows -

$$
\theta_t = 10000^{-t/(d/2)} \; \; \; \; \; t \in \{0,1, \cdots,d/2\}
$$

This sums up the mathematical generalization for the d dimensional case of rotary positional embeddings. In order to make the notations simpler and easier to digest, we take an equivalent representation in complex space. Given the query and key vectors $$q_n, k_m \in \mathbb{R}^d$$, we define complex vectors $$\bar{q_n}, \bar{k_m} \in \mathbb{C}^{d/2}$$ by considering the $$2t^{th}$$ dimension as real part and $$2t+1^{th}$$ dimension as the complex part of the vector.

That is  if $$q_n = (q^1_n,q^2_n,q^3_n,q^4_n, \cdots,q^{d-1}_n,q^d_n)$$ then $$\bar{q_n} = (q^1_n+iq^2_n,q^3_n+iq^4_n, \cdots,q^{d-1}_n+iq^d_n) \in \mathbb{C}^{d/2}$$ is its complex space counterpart, similar for key vectors. Now it can be proved that,

$$
q_nk_m^T = \operatorname{Re}[\bar{q_n} \bar{k_m}^T]
$$

Using this idea of complex vectors, we can define a new transformation matrix $$\textbf{R} \in \mathbb{C}^{N \times (d/2)}$$ as 

$$
\textbf{R}(n,t) = e^{i\theta_tn} \;\;\;\;\;\;\;\; \text{or} \;\;\;\;\;\;\;\;\textbf{R}(n) = \begin{pmatrix}
e^{i\theta_1m} \\
e^{i\theta_2m} \\
\vdots \\
e^{i\theta_{d/2-1}m} \\
e^{i\theta_{d/2}m}
\end{pmatrix}
$$

and the transformation of RoPE can be reformed with Hadamard Product $$\circ$$ as - 

$$
\bar{q}' = \bar{q} \circ \textbf{R}, \; \; \; \; \bar{k}' = \bar{q} \circ \textbf{R}, \; \; \; \; \textbf{A}' = \operatorname{Re}(\bar{q_n} \bar{k_m}^T)

$$

we will be using this summarized notation to further define axial and mixed RoPE Positional Embeddings for ViT  in later sections.

 

Roformer has also shown that RoPE has *long-term decay*, meaning tokens with larger relative positional distances naturally exhibit weaker connections. This aligns intuitively with the idea that distant tokens should carry less influence on each other, improving context relevance. Additionally, RoPE enables *linear self-attention* models to incorporate relative positional embeddings, making it a versatile choice for efficient attention calculations. RoPE also enhances sequence length flexibility, allowing models to work with varying sequence lengths. However, RoPE’s extrapolation performance on sequence lengths much longer than those seen during training is somewhat limited.

## RoPE in Vision Trasformers

Byeongho el at. described various method to extend the idea of rotary positional embeddings to 2 dimensional application in Vision Transformers. Two approaches namely “Axial 2D Rope” and “RoPE Mixed” are discussed in the paper.

### Axial 2D Rope

The core idea is to take and divide the embedding dimension into two and apply positional embedding for x-axis and y-axis separately, which is just repeating the 1d embedding twice. For indexing let patch positions in 2d image be $$\textbf{p}_n = (p_n^x, p_n^y)$$ then the rotation matrix $$\textbf{R}$$ from earlier section can be defined as -

$$
\textbf{R}(n,2t) = e^{i\theta_tp_n^x}, \; \; \; \; \textbf{R}(n,2t+1) = e^{i\theta_tp_n^y}
$$

Hence we are interwining the rope along the two axes $$x$$ and $$y$$. As the range of indexes $$(p_n^x, p_n^y)$$ are reduced by square root, the RoPE frequencies $$\theta_t$$ are also reduced by square root, i.e.

$$
\theta_t = 100^{-t/(d/4)}, \;\;\;\;\;\text{where} \; t \in \{0,1,\cdots, d/4\}
$$

Note that frequencies $$\theta_t$$ for vision tasks are much larger than those used for language tasks. Also number of frequencies is halved so that $$d/2$$ is covered by the contribution of both axes $$x,y$$.

### Mixed RoPE

The axial frequencies are not able to handle diagonal directions since frequencies only depend on either x or y axes. As the relative positions in RoPE are embedded in the form of $$e^{i\theta_t(n-m)}$$ the relative positions are always interpretted as either $$e^{i\theta_t(p_n^x-p_m^x)}$$ or $$e^{i\theta_t(p_n^x-p_m^x)}$$, so the axial frequencies are not mixed in the axial direction.

In order to cater for mixed frequencies, the new rotation matrix $$\textbf{R}$$ is proposed as -

$$
\textbf{R}(n,t) = e^{i(\theta_t^xp_n^x+\theta_t^yp_n^y)}
$$

Unlike 1D RoPE and axial frequency RoPE, Mixed RoPE doesn't define frequencies explicitly. Instead, it allows the network to learn frequencies $$(\theta_t^x,\theta_t^y)$$ for $$t \in \{0, 1, \cdots, d/2\}$$ , as these are treated as learnable parameters. Separate sets of frequencies are used per head per self attention layer.

RoPE ViT introduces 2D Fourier analysis to interpret the difference between RoPE-Axial and RoPE-Mixed. Applying a 2D Fast Fourier Transform (FFT) to all frequencies, followed by an inverse FFT (iFFT), allows for the reconstruction of the input image. However, due to the limited number of frequencies used, the reconstruction isn't perfect. The quality of this reconstruction reflects the representation capabilities of the frequencies and how effectively they can inject 2D positional information.

---

While RoPE tries to tackle the problems of Absolute and Learned positional encoding, another method that addresses this fundamental question in the realm of transformer models, i.e., “how to achieve extrapolation at inference time for sequences that are longer than those encountered during training,” is Alibi(Attention with linear bias) introduced by Press et al. [1] in their ICLR 2022 paper.

## **Attention with Linear Biases (**Alibi)

Unlike RoPE's rotary approach, ALiBi introduces a novel mechanism for incorporating positional information into transformer models. The key innovation lies in its integration of positional encoding directly within the attention computation rather than at the token embedding level. This approach significantly differs from the previous one, focusing on relative positional relationships through a bias-based attention mechanism. One of ALiBi's standout features is its complete removal of traditional positional embeddings at the input layer. Instead of adding positional encodings at the bottom of the network, ALiBi integrates them as biases within the attention layers themselves. 

### Mechanism and Architecture

ALiBi operates by injecting positional information through a bias matrix added to the self-attention scores before the softmax operation. The magnitude of this bias is proportional to the distance between the query and key tokens, establishing an intrinsic relationship between position and attention strength. Mathematically, this transforms the standard attention computation by introducing a position-aware bias term.

The architecture proposed by Press et al. specifically addresses autoregressive decoder-only scenarios, implementing causal attention through a lower-triangular matrix structure. This ensures that each token can only attend to its predecessors. The resulting attention pattern can be represented as:

{% include figure.html path="assets/img/2025-05-07-positional-embedding/ALiBi1.png" class="img-fluid" %}
<div class="caption">
    Figure 4: Visualization of ALiBi's attention computation mechanism. The method introduces a head-specific linear bias matrix (right) that gets added element-wise to the standard query-key dot product attention scores (left). This bias injection occurs before the softmax operation, while preserving the remaining transformer architecture. The bias strength is controlled by a  $m$, which is predetermined for each attention head.
</div>

### What is $$\textbf{m}$$ ?

$$\textbf{m}$$ is a non-learnable, head-specific slope parameter whose values follow a geometric sequence defined that starts at $$2^{-\frac{8}{n}}$$ and uses that same value as its ratio, where $$n$$ represents the number of attention heads. 

This choice of slope values offers several benefits:

1. **Diverse Temporal Scales**: Each attention head is able to focus on different temporal scales, allowing the model to capture a range of positional relationships.
2. **Ensemble Effect**: The varied attention patterns create an ensemble-like effect, enriching the model’s representational power.
3. **Contextual Flexibility**: With slopes tailored for different scales, the model can apply the most effective attention mechanisms for different contexts, adapting to both short and long-range dependencies.

### Theoretical Foundation and Advantages

The core principle behind ALiBi lies in *temporal relevance*—a systematic adjustment of attention values based on the positional distance between queries and keys. This mechanism introduces an inductive *recency bias*, which brings several advantages:

- **Proportional Penalties**: ALiBi applies penalties to attention scores based on the distance between tokens, with closer tokens receiving lower penalties and more attention.
- **Enhanced Focus on Proximity**: The method imposes stronger penalization on distant token relationships, naturally biasing the model toward more relevant, nearby tokens.
- **Adaptive Penalization Across Heads**: ALiBi allows each attention head to apply different penalty rates, enabling the model to capture both short-range and longer-range dependencies effectively.

### Performance and Extrapolation

A standout feature of ALiBi is its remarkable ability to *extrapolate* to sequence lengths far beyond those seen during training. Unlike traditional positional encoding methods, ALiBi sustains robust performance even as sequence lengths grow significantly. This capability overcomes a fundamental limitation in standard transformer architectures, representing a major advancement in the scalability and adaptability of attention-based models.

## **Alibi in Vision Transformers**

Building on ALiBi’s success with sequential data, researchers extended its application to two-dimensional image data in Vision Transformers (ViT). Translating ALiBi from 1D to 2D domains introduced technical challenges, notably adapting its causal attention mechanism for general self-attention and designing spatially aware 2D penalty metrics. These adjustments were crucial in enabling ALiBi to effectively capture spatial dependencies in images, pushing the boundaries of positional encoding within vision models.

### Two-Dimensional Adaptation

The adaptation to 2D data, proposed by Fuller et al., introduced several key modifications to the original ALiBi framework:

1. **Symmetric Attention Structure**: Unlike the lower triangular matrix in sequential processing, the 2D implementation employs a symmetric attention matrix, reflecting the bidirectional nature of spatial relationships in images.
2. **Spatial Distance Metrics**: The linear distance penalty is replaced by a Euclidean metric that captures 2D spatial relationships between image patches: $$\text{distance_penalty} = \sqrt{(x_2 - x_1)^2 + (y_2 - y_1)^2} \cdot m$$ 
where $$m$$ retains its role as a non-learnable head-specific parameter.

## Comparative Analysis: RoPE and ALiBi

These two approaches, while sharing certain fundamental principles, represent distinct philosophical approaches to positional encoding:

### Shared Characteristics

- Both implement relative positional encoding
- Both incorporate distance-dependent attention penalties

### Key Distinctions

1. **Mathematical Foundation**
    - RoPE employs trigonometric transformations with theoretical guarantees
    - ALiBi utilizes straightforward linear distance penalties
2. **Embedding Philosophy**
    - RoPE maintains separate vector spaces for positional and contextual information
    - ALiBi eliminates explicit positional encodings in favor of attention-level biases

  3.  Extrapolation Strength

             Alibi shows a strong extrapolation capability, while RoPE struggles to extrapolate well

             enough above a dozen tokens. Both demonstrate strong extrapolation capabilities

## Extrapolation via Interpolation

While ALiBi remains the only positional encoding method to demonstrate successful extrapolation to sequence lengths far beyond those encountered during training, and has been adopted in models like MPT-7B, MPT-30B, BLOOM, and BloombergGPT, it hasn’t gained widespread traction. Many popular large language models (LLMs), like LLaMA and OPT, still use Rotary Positional Embeddings (RoPE) or learned positional embeddings instead. A primary reason is that ALiBi requires models to be trained from scratch, making it computationally prohibitive for large pre-trained models already utilizing RoPE or learned embeddings.

To address this challenge, Chen et al. (2023) proposed a novel *position interpolation* approach to extend RoPE’s context window. They observed that directly extrapolating RoPE often led to exploding attention scores, which contradicts the expected recency bias inherent to RoPE. This issue arises because the theoretical upper bound proposed in RoFormer is loose, as Chen et al. (2023) elaborates. To tackle this exploding attention score issue, they proposed interpolating sequences longer than the longest sequence encountered during training. Mathematically for a pre-trained LLM with a context window of size $$L$$, we have the following formula for sequence of length $$L'$$  with $$L'> L$$ we have,

$$
\mathbf{f}'(\mathbf{x}, m) = \mathbf{f} \left( \mathbf{x}, \frac{mL}{L'} \right) .
$$

Thus, we reduce their position indices from $$[0, L') \to [0, L)$$  to match the original range of indices before computing RoPE. Consequently, as inputs to RoPE, the maximum relative distance between any two tokens is reduced from $$L' \to L$$. This alignment of ranges of position indices and relative distances before and after extension helps mitigate the effect on attention score computation due to context window extensions, which make the model easier to adapt.

![Screenshot 2024-10-28 at 9.28.51 AM.png](https://prod-files-secure.s3.us-west-2.amazonaws.com/224cf403-b393-4eb4-92c1-725bb6b7cd54/5ceaa7f7-06b7-4d0f-8da4-addaada9ffd2/Screenshot_2024-10-28_at_9.28.51_AM.png)

Recent research on extending the context windows of pre-trained Large Language Models (LLMs) has predominantly focused on position interpolation using RoPE (Rotary et al.) due to widespread use in many pre-trained LLMs. At the same time, the Alibi positioning method has received comparatively less attention. A recent work by Faisal Al-Khateeb et al. proposed an innovative approach to position interpolation for Alibi that significantly differs from Chen et al.'s (2023) methodology. While Chen et al.'s scaling operation was motivated by the problem of exploding attention score magnitudes during extrapolation, Alibi presents a different challenge: it generates lower magnitude attention scores for tokens in the extrapolation regime than the interpolation regime. To address this, Al-Khateeb et al. introduced a dynamic slope scaling mechanism that adjusts the Alibi slope to amplify attention scores, effectively preventing the decrease in magnitudes beyond the training context length. This approach involves scaling the slopes by a factor of $$L/L'$$, where $$L$$ represents the maximum sequence length during training and $$L'$$ denotes the extended input sequence length during inference, expressed as $$m_j' = m_j(L/L')$$. Importantly, this slope scaling is only applied when $$L'$$ exceeds $$L$$, ensuring the model maintains its original performance for sequences within the training length.

In the domain of computer vision, Dosovitskiy et al., in their seminal work on Vision Transformers, introduced a unique interpolation method to handle position embeddings during fine-tuning for higher-resolution images. Specifically, when adapting to higher resolutions, they maintained the original patch size, which increased the sequence length and caused pre-trained positional embeddings to lose spatial relevance. To address this, they applied 2D interpolation to the pre-trained position embeddings, preserving spatial relationships based on their original locations in the image. This approach differs from Chen et al., who performed interpolation at the level of position indices rather than on the embeddings themselves. It also aligns more closely with Al-Khateeb et al.'s rescaling technique, which similarly adjusts positional embeddings rather than simply rescaling indices for longer sequences.

Notably, Al-Khateeb et al.'s method requires no additional training or fine-tuning, whereas the approaches by Chen et al. and Dosovitskiy et al. involve fine-tuning the model on longer sequences for effective extrapolation.

## Experimental Evaluation

We conducted comprehensive experiments comparing four architectural variants:

1. Standard Vision Transformer
2. Vision Transformer without positional encoding
3. Vision Transformer with 2D mixed-frequency RoPE
4. Vision Transformer with 2D ALiBi

### Methodology

- Training duration: 90 epochs (approximately 45K steps)
- Optimizer: AdamW with 5K step linear warmup
- Learning rate: 1e-4 (cosine decay schedule)
- Batch size: 256
- Weight decay: 0.05
- Data augmentation: Mixup, Cutmix, RandAugment, Random Erasing
- Dataset: ImageNet-100 (subset with 100 classes, 130K images)

### Key Findings

1. Both ALiBi and RoPE demonstrate superior extrapolation capabilities compared to the baseline ViT.
2. ALiBi exhibits better extrapolation performance over RoPE.
3. ALiBi demonstrates computational efficiency advantages in training time.

## Conclusions

Our analysis of positional encoding mechanisms in Transformer architectures highlights several key insights:

1. The evolution of positional encoding and its critical role across diverse domains within Transformer architectures.
2. The challenge of sequence length extrapolation and the effectiveness of RoPE and ALiBi in handling variable sequence lengths, especially in extended contexts.
3. The adaptation of positional encoding from 1D to 2D tasks showcases ALiBi's practical advantages over RoPE in terms of computational efficiency and implementation simplicity.