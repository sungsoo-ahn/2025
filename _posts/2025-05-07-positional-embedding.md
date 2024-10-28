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
  - name: Equations
  - name: Images and Figures
    subsections:
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
---

Note: please use the table of contents as defined in the front matter rather than the traditional markdown styling.

## Background
"Attention Is All You Need," introduced by Vaswani et al. in 2017, revolutionized the field of natural language processing and computer vision by proposing a purely attention-based model, the Transformer, eliminating the need for recurrent or convolutional networks. 

Later, Dosovitskiy et al. (2021) applied this concept to computer vision, introducing Vision Transformers (ViTs).On its own, the Transformer architecture is position-invariant, i.e., it processes its input as an unordered set. Unlike RNNs, designed to handle ordered sequences, or CNNs, which leverage translation equivariance and locality, transformers lack the inherent ability to capture sequential patterns. This is because the self-attention mechanism is independent of the token index.

Vaswani et al. (2017) introduced positional encoding to address this lack of inductive bias. Since then, researchers have explored various methods to encode positional information directly into tokens or within the self-attention interaction step. The following sections will discuss the broad advancements of positional encoding methods in 1D and their extension for 2d tasks in Vision Transformers.

## Preliminaries
### Self-Attention Mechanism
The self-attention mechanism processes sequences by representing each element as a vector of embedding dimension $$d$$. For a sequence of length $$L$$ and embedding dimension $$d$$, the input is represented as a matrix $$X \in \mathbb{R}^{d \times L}$$.

The mechanism employs three learnable weight matrices: $$W_Q,W_K,$$ and $$W_V \in \mathbb{R}^{d \times d_k}$$, where $$d_k$$ represents the dimensionality of the projected spaces. These matrices transform the input into queries, keys, and values respectively:

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

[image goes here]

One might argue that how a function of sines and cosines can be a good representation of positions in a sequence, why not authors chose any other arbitrary function for mapping each position to a value? The choice of sinusoidal functions was actually clever as it allows some properties to be exihibited that provides an advantage to the transformer

- Firstly the values are normalized between $$[-1,1]$$ allowing the model to learn parameters easily.
- The distance between neighbouring positions is symmetrical and decays naturally with position.
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

Hence the dot product $$p_i \cdot p_{i+\phi}$$ is independent of position $$i$$, and relative position $$\phi$$ is retained.

The authors hypothesised that using sinusoidal positional encoding in this form would allow the model to learn to attend relative positions easily due to properties shown above and may allow the model to extrapolate to sequence lengths longer than the ones encountered during training.

However, subsequent research demonstrated that both authors' hypotheses were incorrect. Later experiments found that absolute sinusoidal positional encoding is not well-suited for capturing relative positional information (Shaw et al., 2018). Additionally, it failed to extrapolate effectively, resulting in poor performance on sequence lengths longer than those encountered during training (Press et al., 2021).

In addition to sinusoidal positional embedding, Vaswani et al. also explored learned positional encoding. They explored using a set of trainable vectors $$p \in \mathbb{R}^{d \times L}$$ $$\{p_t\}_{t=1}^L$$ $$L$$ represents the maximum sequence length as positional embeddings. However, this approach didn't yield significant performance improvements over sinusoidal positional embedding. Moreover, the upper bound $$L$$ limited the method's ability to extrapolate to sequences longer than $$L$$ without resorting to interpolation techniques.


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

Hence the $$e^{i(n−m)θ}$$ in this formulation of attention is responsible injecting relative position of the tokens which are $$(n-m)$$ distance apart from each other. The formal derivation that it is a solution to equation formulated is detailed in RoFormer [2023]. Intuitively, we're rotating 2D query and key vectors in the embedding space based on their sequence position. For example, the vector corresponding to first token is rotated by θ, the second by 2θ, and so on. The approach of rotating according to the position provides several benefits as discussed below

- **Stability of Vectors**: Adding tokens at the end of a sentence doesn’t affect the vectors for words at the beginning, facilitating efficient caching.
- **Preservation of Relative Positions**: If two words, say “pig” and “dog,” maintain the same relative distance in different contexts, their vectors are rotated by the same amount. This ensures that the angle, and consequently the dot product between these vectors, remains constant

### The general D-dimensional case

The idea of extending the 2D result to the d dimensional case is to break the embedding vector in various 2 dimensional chunks and rotate each one of them by a specific angle $$m\theta_i$$ where m depends on the position of token in the sequence and $$\theta_i$$ varies according to the chunk. This can be intutively thought of as a corkscrew rotation in higher dimensions.

![Frame 3 (1).png](https://prod-files-secure.s3.us-west-2.amazonaws.com/224cf403-b393-4eb4-92c1-725bb6b7cd54/3f943f2c-d7ba-4d64-8d40-b6f929ea3700/Frame_3_(1).png)


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



## Equations

This theme supports rendering beautiful math in inline and display modes using [MathJax 3](https://www.mathjax.org/) engine.
You just need to surround your math expression with `$$`, like `$$ E = mc^2 $$`.
If you leave it inside a paragraph, it will produce an inline expression, just like $$ E = mc^2 $$.

To use display mode, again surround your expression with `$$` and place it as a separate paragraph.
Here is an example:

$$
\left( \sum_{k=1}^n a_k b_k \right)^2 \leq \left( \sum_{k=1}^n a_k^2 \right) \left( \sum_{k=1}^n b_k^2 \right)
$$

Note that MathJax 3 is [a major re-write of MathJax](https://docs.mathjax.org/en/latest/upgrading/whats-new-3.0.html) 
that brought a significant improvement to the loading and rendering speed, which is now 
[on par with KaTeX](http://www.intmath.com/cg5/katex-mathjax-comparison.php).


## Images and Figures

Its generally a better idea to avoid linking to images hosted elsewhere - links can break and you
might face losing important information in your blog post.
To include images in your submission in this way, you must do something like the following:

```markdown
{% raw %}{% include figure.html path="assets/img/2025-05-07-distill-example/iclr.png" class="img-fluid" %}{% endraw %}
```

which results in the following image:

{% include figure.html path="assets/img/2025-05-07-distill-example/iclr.png" class="img-fluid" %}

To ensure that there are no namespace conflicts, you must save your asset to your unique directory
`/assets/img/2025-05-07-[SUBMISSION NAME]` within your submission.

Please avoid using the direct markdown method of embedding images; they may not be properly resized.
Some more complex ways to load images (note the different styles of the shapes/shadows):

<div class="row mt-3">
    <div class="col-sm mt-3 mt-md-0">
        {% include figure.html path="assets/img/2025-05-07-distill-example/9.jpg" class="img-fluid rounded z-depth-1" %}
    </div>
    <div class="col-sm mt-3 mt-md-0">
        {% include figure.html path="assets/img/2025-05-07-distill-example/7.jpg" class="img-fluid rounded z-depth-1" %}
    </div>
</div>
<div class="caption">
    A simple, elegant caption looks good between image rows, after each row, or doesn't have to be there at all.
</div>

<div class="row mt-3">
    <div class="col-sm mt-3 mt-md-0">
        {% include figure.html path="assets/img/2025-05-07-distill-example/8.jpg" class="img-fluid z-depth-2" %}
    </div>
    <div class="col-sm mt-3 mt-md-0">
        {% include figure.html path="assets/img/2025-05-07-distill-example/10.jpg" class="img-fluid z-depth-2" %}
    </div>
</div>

<div class="row mt-3">
    <div class="col-sm mt-3 mt-md-0">
        {% include figure.html path="assets/img/2025-05-07-distill-example/11.jpg" class="img-fluid"  %}
    </div>
    <div class="col-sm mt-3 mt-md-0">
        {% include figure.html path="assets/img/2025-05-07-distill-example/12.jpg" class="img-fluid" %}
    </div>
    <div class="col-sm mt-3 mt-md-0">
        {% include figure.html path="assets/img/2025-05-07-distill-example/7.jpg" class="img-fluid" %}
    </div>
</div>

### Interactive Figures

Here's how you could embed interactive figures that have been exported as HTML files.
Note that we will be using plotly for this demo, but anything built off of HTML should work
(**no extra javascript is allowed!**).
All that's required is for you to export your figure into HTML format, and make sure that the file
exists in the `assets/html/[SUBMISSION NAME]/` directory in this repository's root directory.
To embed it into any page, simply insert the following code anywhere into your page.

```markdown
{% raw %}{% include [FIGURE_NAME].html %}{% endraw %} 
```

For example, the following code can be used to generate the figure underneath it.

```python
import pandas as pd
import plotly.express as px

df = pd.read_csv('https://raw.githubusercontent.com/plotly/datasets/master/earthquakes-23k.csv')

fig = px.density_mapbox(
    df, lat='Latitude', lon='Longitude', z='Magnitude', radius=10,
    center=dict(lat=0, lon=180), zoom=0, mapbox_style="stamen-terrain")
fig.show()

fig.write_html('./assets/html/2025-05-07-distill-example/plotly_demo_1.html')
```

And then include it with the following:

```html
{% raw %}<div class="l-page">
  <iframe src="{{ 'assets/html/2025-05-07-distill-example/plotly_demo_1.html' | relative_url }}" frameborder='0' scrolling='no' height="600px" width="100%"></iframe>
</div>{% endraw %}
```

Voila!

<div class="l-page">
  <iframe src="{{ 'assets/html/2025-05-07-distill-example/plotly_demo_1.html' | relative_url }}" frameborder='0' scrolling='no' height="600px" width="100%"></iframe>
</div>

## Citations

Citations are then used in the article body with the `<d-cite>` tag.
The key attribute is a reference to the id provided in the bibliography.
The key attribute can take multiple ids, separated by commas.

The citation is presented inline like this: <d-cite key="gregor2015draw"></d-cite> (a number that displays more information on hover).
If you have an appendix, a bibliography is automatically created and populated in it.

Distill chose a numerical inline citation style to improve readability of citation dense articles and because many of the benefits of longer citations are obviated by displaying more information on hover.
However, we consider it good style to mention author last names if you discuss something at length and it fits into the flow well — the authors are human and it’s nice for them to have the community associate them with their work.

***

## Footnotes

Just wrap the text you would like to show up in a footnote in a `<d-footnote>` tag.
The number of the footnote will be automatically generated.<d-footnote>This will become a hoverable footnote.</d-footnote>

***

## Code Blocks

This theme implements a built-in Jekyll feature, the use of Rouge, for syntax highlighting.
It supports more than 100 languages.
This example is in C++.
All you have to do is wrap your code in a liquid tag:

{% raw  %}
{% highlight c++ linenos %}  <br/> code code code <br/> {% endhighlight %}
{% endraw %}

The keyword `linenos` triggers display of line numbers. You can try toggling it on or off yourself below:

{% highlight c++ %}

int main(int argc, char const \*argv[])
{
string myString;

    cout << "input a string: ";
    getline(cin, myString);
    int length = myString.length();

    char charArray = new char * [length];

    charArray = myString;
    for(int i = 0; i < length; ++i){
        cout << charArray[i] << " ";
    }

    return 0;
}

{% endhighlight %}

***

## Diagrams

This theme supports generating various diagrams from a text description using [jekyll-diagrams](https://github.com/zhustec/jekyll-diagrams){:target="\_blank"} plugin.
Below, we generate a few examples of such diagrams using languages such as [mermaid](https://mermaid-js.github.io/mermaid/){:target="\_blank"}, [plantuml](https://plantuml.com/){:target="\_blank"}, [vega-lite](https://vega.github.io/vega-lite/){:target="\_blank"}, etc.

**Note:** different diagram-generation packages require external dependencies to be installed on your machine.
Also, be mindful of that because of diagram generation the first time you build your Jekyll website after adding new diagrams will be SLOW.
For any other details, please refer to [jekyll-diagrams](https://github.com/zhustec/jekyll-diagrams){:target="\_blank"} README.

**Note:** This is not supported for local rendering! 

The diagram below was generated by the following code:

{% raw %}
```
{% mermaid %}
sequenceDiagram
    participant John
    participant Alice
    Alice->>John: Hello John, how are you?
    John-->>Alice: Great!
{% endmermaid %}
```
{% endraw %}

{% mermaid %}
sequenceDiagram
participant John
participant Alice
Alice->>John: Hello John, how are you?
John-->>Alice: Great!
{% endmermaid %}

***

## Tweets

An example of displaying a tweet:
{% twitter https://twitter.com/rubygems/status/518821243320287232 %}

An example of pulling from a timeline:
{% twitter https://twitter.com/jekyllrb maxwidth=500 limit=3 %}

For more details on using the plugin visit: [jekyll-twitter-plugin](https://github.com/rob-murray/jekyll-twitter-plugin)

***

## Blockquotes

<blockquote>
    We do not grow absolutely, chronologically. We grow sometimes in one dimension, and not in another, unevenly. We grow partially. We are relative. We are mature in one realm, childish in another.
    —Anais Nin
</blockquote>

***


## Layouts

The main text column is referred to as the body.
It is the assumed layout of any direct descendants of the `d-article` element.

<div class="fake-img l-body">
  <p>.l-body</p>
</div>

For images you want to display a little larger, try `.l-page`:

<div class="fake-img l-page">
  <p>.l-page</p>
</div>

All of these have an outset variant if you want to poke out from the body text a little bit.
For instance:

<div class="fake-img l-body-outset">
  <p>.l-body-outset</p>
</div>

<div class="fake-img l-page-outset">
  <p>.l-page-outset</p>
</div>

Occasionally you’ll want to use the full browser width.
For this, use `.l-screen`.
You can also inset the element a little from the edge of the browser by using the inset variant.

<div class="fake-img l-screen">
  <p>.l-screen</p>
</div>
<div class="fake-img l-screen-inset">
  <p>.l-screen-inset</p>
</div>

The final layout is for marginalia, asides, and footnotes.
It does not interrupt the normal flow of `.l-body`-sized text except on mobile screen sizes.

<div class="fake-img l-gutter">
  <p>.l-gutter</p>
</div>

***

## Other Typography?

Emphasis, aka italics, with *asterisks* (`*asterisks*`) or _underscores_ (`_underscores_`).

Strong emphasis, aka bold, with **asterisks** or __underscores__.

Combined emphasis with **asterisks and _underscores_**.

Strikethrough uses two tildes. ~~Scratch this.~~

1. First ordered list item
2. Another item
⋅⋅* Unordered sub-list. 
1. Actual numbers don't matter, just that it's a number
⋅⋅1. Ordered sub-list
4. And another item.

⋅⋅⋅You can have properly indented paragraphs within list items. Notice the blank line above, and the leading spaces (at least one, but we'll use three here to also align the raw Markdown).

⋅⋅⋅To have a line break without a paragraph, you will need to use two trailing spaces.⋅⋅
⋅⋅⋅Note that this line is separate, but within the same paragraph.⋅⋅
⋅⋅⋅(This is contrary to the typical GFM line break behavior, where trailing spaces are not required.)

* Unordered lists can use asterisks
- Or minuses
+ Or pluses

[I'm an inline-style link](https://www.google.com)

[I'm an inline-style link with title](https://www.google.com "Google's Homepage")

[I'm a reference-style link][Arbitrary case-insensitive reference text]

[I'm a relative reference to a repository file](../blob/master/LICENSE)

[You can use numbers for reference-style link definitions][1]

Or leave it empty and use the [link text itself].

URLs and URLs in angle brackets will automatically get turned into links. 
http://www.example.com or <http://www.example.com> and sometimes 
example.com (but not on Github, for example).

Some text to show that the reference links can follow later.

[arbitrary case-insensitive reference text]: https://www.mozilla.org
[1]: http://slashdot.org
[link text itself]: http://www.reddit.com

Here's our logo (hover to see the title text):

Inline-style: 
![alt text](https://github.com/adam-p/markdown-here/raw/master/src/common/images/icon48.png "Logo Title Text 1")

Reference-style: 
![alt text][logo]

[logo]: https://github.com/adam-p/markdown-here/raw/master/src/common/images/icon48.png "Logo Title Text 2"

Inline `code` has `back-ticks around` it.

```javascript
var s = "JavaScript syntax highlighting";
alert(s);
```
 
```python
s = "Python syntax highlighting"
print(s)
```
 
```
No language indicated, so no syntax highlighting. 
But let's throw in a <b>tag</b>.
```

Colons can be used to align columns.

| Tables        | Are           | Cool  |
| ------------- |:-------------:| -----:|
| col 3 is      | right-aligned | $1600 |
| col 2 is      | centered      |   $12 |
| zebra stripes | are neat      |    $1 |

There must be at least 3 dashes separating each header cell.
The outer pipes (|) are optional, and you don't need to make the 
raw Markdown line up prettily. You can also use inline Markdown.

Markdown | Less | Pretty
--- | --- | ---
*Still* | `renders` | **nicely**
1 | 2 | 3

> Blockquotes are very handy in email to emulate reply text.
> This line is part of the same quote.

Quote break.

> This is a very long line that will still be quoted properly when it wraps. Oh boy let's keep writing to make sure this is long enough to actually wrap for everyone. Oh, you can *put* **Markdown** into a blockquote. 


Here's a line for us to start with.

This line is separated from the one above by two newlines, so it will be a *separate paragraph*.

This line is also a separate paragraph, but...
This line is only separated by a single newline, so it's a separate line in the *same paragraph*.
