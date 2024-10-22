---
layout: distill
title: Sigmoid and Softmax
description: Existing explanations of sigmoid and softmax rely on probabilistic interpretation of the output(s) but do not provide the source of probabilistic interpretation. In contrast, both sigmoid and softmax have a simple definition with their roots in basic information theory and calculus. This blog takes an isolated look at sigmoid and softmax and explains how and why they materialize. The definition and derivation show that they are not related.
date: 2025-05-07
future: true
htmlwidgets: true
hidden: false

# Anonymize when submitting
authors:
   - name: Anonymous

# must be the exact same name as your blogpost
# bibliography: sigmoidandsoftmax.bib  

# Add a table of contents to your post.
#   - make sure that TOC names match the actual section names
#     for hyperlinks within the post to work correctly. 
#   - please use this format rather than manually creating a markdown table of contents.
toc:
  - name: Introduction
  - name: Sigmoid
  - name: Softmax
  - name: Discussion and Conclusion

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
    font-size: 16px;\
  }
---

<h2 id="Introduction">Introduction</h2>

<p>Let $$\mathbf{x} \in \mathbf{R^d}$$ and $$\mathbf{S} = \left\{\left(\mathbf{w}_0, \mathbf{w}_1, \ldots, \mathbf{w}_d\right) \mid \sum_{i=1}^d \mathbf{w}_i = 1\right\} \subset \left[0, 1\right]^d$$. Softmax function $$\mathbf{f} : \mathbf{R^d} \rightarrow  \mathbf{S} $$ is given by
$$
\begin{align}
    \mathbf{f}\left(\mathbf{x}\right)_i &= \frac{\exp{\left(-\mathbf{x}_i\right)}}{\sum_{j=1}^d \exp{\left(-\mathbf{x}_j\right)}}.
\end{align}
$$
Softmax does not approximate the $$\arg\max$$ function but approximates the one-hot encoding of the $$\arg\max$$ function [1]. Let $$\mathbf{p} \in \left\{0, 1\right\}^d$$, $$\mathbf{\hat{p}} \in \left[0, 1\right]^d$$ and $$\mathbf{p}, \mathbf{\hat{p}} \in \mathbf{S}$$.
$$
\begin{align}
\left(\mathbf{p}_0, \mathbf{p}_1, \ldots, \mathbf{p}_i, \ldots, \mathbf{p}_d\right) &\approx \left(\mathbf{\hat{p}}_0, \mathbf{\hat{p}}_1, \ldots, \mathbf{\hat{p}}_i, \ldots, \mathbf{\hat{p}}_d\right) \\
 &= \mathbf{f\left(\mathbf{x}\right)}.
\end{align}
$$
The approximation error is quantified by cross entropy
$$
\begin{align}
    \mathcal{L} &= \sum_{i=1}^d -\mathbf{p}_i\log\left(\mathbf{\hat{p}}_i\right).
\end{align}
$$
This should not confuse the reader whether softmax does or does not has its roots in $$\arg\max$$. The next section shows that indeed it does not. But it is still not known where softmax arises from. An analysis on relatively recent online time-series algorithm sheds some light on generalized softmax function [2] and derivation of softmax can be established.
</p>

<h2 id="Sigmoid">Sigmoid</h2>
<p>
   
</p>

<h2 id="Softmax">Softmax</h2>
<p>
   Consider the transformation
$$
\begin{align}
    \mathbf{p} &= \arg\min_{ \mathbf{w} \in \mathbf{S} } \langle \mathbf{w}, \mathbf{x} \rangle
\end{align}
$$
where $$\mathbf{x} \in \mathbf{R^d}, \mathbf{w} \in \mathbf{S} \subset \left[0, 1\right]^d$$ such that $$\mathbf{S} = \left\{\mathbf{w} \mid \sum_{i}^d \mathbf{w}_i = 1\right\}$$, $$\mathbf{p} \in \left[0, 1\right]^d$$ and $$\sum_{i=1}^d \mathbf{p}_i = 1 $$.


$$\mathbf{p}$$ with negative entropy regularizer
$$
\begin{align}
    \mathbf{\hat{p}} &= \arg\min_{ \mathbf{w} \in \mathbf{S} } \langle \mathbf{w}, \mathbf{x} \rangle + \sum_{i=1}^d \mathbf{w}_i\log\left(\mathbf{w}_i\right) 
\end{align}
$$
    Since  $$\mathbf{w} \in \mathbf{S}$$, add a Lagrange multiplier $$\lambda\left(\langle \mathbf{w}, \mathbf{1} \rangle - 1\right)$$ to the objective function. 
    $$
    \begin{align}
        \mathbf{\hat{p}} &= \arg\min_{ \mathbf{w} \in \mathbf{S} } \langle \mathbf{w}, \mathbf{x} \rangle + \sum_{i=1}^d \mathbf{w}_i\log\left(\mathbf{w}_i\right) + \lambda\left(\langle \mathbf{w}, \mathbf{1} \rangle - 1\right) \\
        &= \arg\min_{ \mathbf{w} \in \mathbf{S} } \sum_{i=1}^d \mathbf{w}_i \mathbf{x}_i  + \sum_{i=1}^d \mathbf{w}_i\log\left(\mathbf{w}_i\right) + \lambda\left(\sum_{i=1} ^d \mathbf{w}_i - 1\right) 
    \end{align}
    $$
    Differentiate the objective function with respect to $$\mathbf{w}_i$$ and equate to 0
    $$
    \begin{align}
    \mathbf{x}_i + 1 + \log\left(\mathbf{w}_i\right) + \lambda &= 0 \\
    \mathbf{w}_i^\star &= \exp\left(-\mathbf{x}_i\right)\exp\left(-1 - \lambda\right) \\
    &= \frac{\exp\left(-\mathbf{x}_i\right)}{\exp\left(1 + \lambda\right)}
    \end{align}
    $$
Set $$ \lambda$$ such that $$\sum_{i}^d \mathbf{w}_i^\star = 1$$
$$
\begin{align}
     \mathbf{w}_i^\star &= \frac{\exp\left(-\mathbf{x}_i\right)}{\sum_{j=1}^d\exp\left(-\mathbf{x}_j\right)} \\
     \mathbf{\hat{p}} &= \mathbf{w}^\star
\end{align}
$$
</p>

<h2 id="Conclusion">Conclusion</h2>
<p>
   Softmax is the function that when applied on a vector gives a probability vector that has minimum negative entropy and dot product with the input vector. Sigmoid
</p>
