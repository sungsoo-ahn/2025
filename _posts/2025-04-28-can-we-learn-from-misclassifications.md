---
layout: distill
title: Can We Learn From Misclassifications
description: Model Evaluation in supervised learning based on common methods, such as the final activation map or the final represantations, is not sparse. We introduce a novel evaluation based on the gradients of the model. We compare the gradient weights to the latent representations and the final prediction scores using PCA. We further cluster the results compared to the original classes, using KMeans and GMM, respectively.  
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
bibliography: 2025-04-28-can-we-learn-from-misclassifications.bib  

# Add a table of contents to your post.
#   - make sure that TOC names match the actual section names
#     for hyperlinks within the post to work correctly. 
#   - please use this format rather than manually creating a markdown table of contents.
toc:
  - name: Equation
  - name: Equation
  - name: Figures
  - name: Expirmantal Results
    subsection:
    - name: Clustering using Kmeans
    - name: Clustering using GMM
  - name: Hessian Matrix-Self Supervised Trainng 
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
---
## Equation
## Figures
## 1. Cosine Similarity Computation

To quantify the similarity between images using gradients and feature representations, we utilize the **cosine similarity** metric. The cosine similarity measures the cosine of the angle between two non-zero vectors in an inner product space, which provides a measure of orientation similarity regardless of their magnitude.

Given two vectors, \(\mathbf{u} \in \mathbb{R}^n\) and \(\mathbf{v} \in \mathbb{R}^n\), representing either gradients or feature embeddings of two images, the cosine similarity is defined as:
$$
\[
\text{cos\_sim}(\mathbf{u}, \mathbf{v}) = \frac{\mathbf{u}^\top \mathbf{v}}{\|\mathbf{u}\|_2 \|\mathbf{v}\|_2}
\]
$$

where:

- \(\mathbf{u}^\top \mathbf{v} = \sum_{i=1}^n u_i v_i\) is the dot product of \(\mathbf{u}\) and \(\mathbf{v}\).
- \(\|\mathbf{u}\|_2 = \sqrt{\sum_{i=1}^n u_i^2}\) is the Euclidean (L2) norm of \(\mathbf{u}\).
- \(\|\mathbf{v}\|_2 = \sqrt{\sum_{i=1}^n v_i^2}\) is the Euclidean norm of \(\mathbf{v}\).

The cosine similarity ranges from \(-1\) to \(1\):

- \(1\) indicates that the vectors are identical in direction.
- \(0\) indicates that the vectors are orthogonal.
- \(-1\) indicates that the vectors are diametrically opposed.

### Application to Gradients and Features

For each pair of images \( x_i \) and \( x_j \) belonging to the same class \( c \), we compute:

1. **Cosine Similarity between Gradients:**

   \[
   \text{cos\_sim}(\nabla_{x_i} L, \nabla_{x_j} L) = \frac{ (\nabla_{x_i} L)^\top (\nabla_{x_j} L) }{ \|\nabla_{x_i} L\|_2 \|\nabla_{x_j} L\|_2 }
   \]

2. **Cosine Similarity between Feature Representations:**

   \[
   \text{cos\_sim}(f(x_i), f(x_j)) = \frac{ f(x_i)^\top f(x_j) }{ \|f(x_i)\|_2 \|f(x_j)\|_2 }
   \]

where:

- \( \nabla_{x_i} L = \frac{\partial L(f(x_i), y_i)}{\partial x_i} \) is the gradient of the loss function \( L \) with respect to the input image \( x_i \).
- \( f(x_i) \in \mathbb{R}^d \) is the feature representation of \( x_i \) extracted by a pretrained model (e.g., ResNet-50).
- \( y_i \) is the label of image \( x_i \).

### Properties

- **Normalization:** Cosine similarity normalizes the vectors to unit length, focusing solely on direction.
- **Symmetry:** \( \text{cos\_sim}(\mathbf{u}, \mathbf{v}) = \text{cos\_sim}(\mathbf{v}, \mathbf{u}) \).
- **Computational Efficiency:** Requires simple operations (dot product and norms).

### Interpretation in Neural Networks

High cosine similarity between gradients of images from the same class suggests that the model updates parameters in similar directions for these images, which can enhance class cohesion in the feature space.

---

## 2. Gradient-Based Loss Formulation

To enforce similarity between images of the same class at the gradient level, we propose a **gradient-based loss function** \( L_{\text{grad}} \). This loss encourages the model to produce similar gradients for images within the same class, promoting intra-class consistency.

### Formulation Using Euclidean Distance

\[
L_{\text{grad}} = \sum_{c=1}^{C} \sum_{(i, j) \in S_c} \left\| \nabla_{x_i} L - \nabla_{x_j} L \right\|_2^2
\]

where:

- \( C \) is the total number of classes.
- \( S_c \) is the set of all pairs \( (i, j) \) such that \( x_i, x_j \) belong to class \( c \).
- \( \nabla_{x_i} L \) is the gradient with respect to \( x_i \).

### Expansion of the Loss Term

Expanding the squared Euclidean norm:

\[
\left\| \nabla_{x_i} L - \nabla_{x_j} L \right\|_2^2 = \|\nabla_{x_i} L\|_2^2 + \|\nabla_{x_j} L\|_2^2 - 2 (\nabla_{x_i} L)^\top \nabla_{x_j} L
\]

### Simplification

Since \( \|\nabla_{x_i} L\|_2^2 \) and \( \|\nabla_{x_j} L\|_2^2 \) are constants with respect to each other, we can focus on maximizing the inner product \( (\nabla_{x_i} L)^\top \nabla_{x_j} L \) to minimize \( L_{\text{grad}} \).

### Formulation Using Cosine Similarity

To directly maximize the alignment of gradients, we define:

\[
L_{\text{grad}} = \sum_{c=1}^{C} \sum_{(i, j) \in S_c} \left( 1 - \frac{ (\nabla_{x_i} L)^\top \nabla_{x_j} L }{ \|\nabla_{x_i} L\|_2 \|\nabla_{x_j} L\|_2 } \right)
\]

### Total Loss Function

Combining \( L_{\text{grad}} \) with the standard classification loss \( L_{\text{cls}} \):

\[
L_{\text{total}} = L_{\text{cls}} + \lambda L_{\text{grad}}
\]

where \( \lambda \) is a hyperparameter balancing the two loss terms.

### Gradient Computation for Backpropagation

To compute gradients with respect to model parameters \( \theta \), we apply the chain rule:

\[
\frac{\partial L_{\text{grad}}}{\partial \theta} = \sum_{c=1}^{C} \sum_{(i, j) \in S_c} \left( \frac{\partial L_{\text{grad}, ij}}{\partial \nabla_{x_i} L} \frac{\partial \nabla_{x_i} L}{\partial \theta} + \frac{\partial L_{\text{grad}, ij}}{\partial \nabla_{x_j} L} \frac{\partial \nabla_{x_j} L}{\partial \theta} \right)
\]

**Note:** Computing \( \frac{\partial \nabla_{x_i} L}{\partial \theta} \) involves second-order derivatives, which can be approximated or computed using automatic differentiation tools.

---

## 3. Hessian-Vector Product for Second-Order Derivatives

Second-order derivatives provide curvature information, which can improve optimization and generalization.

### Hessian Matrix

The Hessian \( H \in \mathbb{R}^{n \times n} \) of a scalar function \( L(x) \) is:

\[
H = \nabla^2_x L = \left[ \frac{\partial^2 L}{\partial x_i \partial x_j} \right]_{i,j=1}^n
\]

### Hessian-Vector Product

Given a vector \( \mathbf{v} \in \mathbb{R}^n \), the Hessian-vector product \( H \mathbf{v} \) is defined as:

\[
H \mathbf{v} = \nabla_x \left( \nabla_x L \cdot \mathbf{v} \right)
\]

### Efficient Computation

Computing \( H \) explicitly is computationally expensive (\( O(n^2) \)). However, \( H \mathbf{v} \) can be computed efficiently without forming \( H \), using techniques like:

- **Pearlmutter's Method:** Utilize automatic differentiation to compute \( H \mathbf{v} \) with \( O(n) \) complexity.

### Implementation Steps

1. **Forward Pass:**

   Compute \( L(x) \) and store intermediate activations.

2. **First Backward Pass:**

   Compute \( \nabla_x L \).

3. **Second Backward Pass:**

   Compute \( H \mathbf{v} \) by backpropagating \( \mathbf{v} \) through the computational graph of \( \nabla_x L \).

### Applications

- **Optimization Algorithms:** Newton's method, conjugate gradient, and quasi-Newton methods.
- **Regularization:** Incorporate curvature into regularization terms for better generalization.

---

## Mutual Information and Its Projection to the Gradient Level

---

### 1. Mutual Information Concept

#### Definition

Mutual Information (MI) measures the amount of information one random variable contains about another. For discrete variables \( X \) and \( Y \):

\[
I(X; Y) = \sum_{x \in \mathcal{X}} \sum_{y \in \mathcal{Y}} p(x, y) \log \left( \frac{p(x, y)}{p(x) p(y)} \right)
\]

For continuous variables:

\[
I(X; Y) = \int_{\mathcal{X}} \int_{\mathcal{Y}} p(x, y) \log \left( \frac{p(x, y)}{p(x) p(y)} \right) dx dy
\]

where:

- \( p(x, y) \) is the joint probability distribution.
- \( p(x) \) and \( p(y) \) are marginal distributions.

#### Properties

- \( I(X; Y) \geq 0 \).
- \( I(X; Y) = 0 \) if \( X \) and \( Y \) are independent.
- Symmetric: \( I(X; Y) = I(Y; X) \).

#### Relevance in Deep Learning

MI is used to quantify dependencies between variables, which can help in feature selection, representation learning, and understanding network internals.

### 2. Projection to the Gradient Level

#### Objective

Leverage mutual information to measure and encourage dependencies between gradients of samples within the same class.

#### Formulating Mutual Information Between Gradients

Consider the gradients \( \nabla_{x_i} L \) and \( \nabla_{x_j} L \) as random variables influenced by input data and model parameters.

#### Mutual Information Between Gradients

\[
I(\nabla_{x_i} L; \nabla_{x_j} L) = H(\nabla_{x_i} L) - H(\nabla_{x_i} L | \nabla_{x_j} L)
\]

where:

- \( H(\nabla_{x_i} L) \) is the entropy of \( \nabla_{x_i} L \).
- \( H(\nabla_{x_i} L | \nabla_{x_j} L) \) is the conditional entropy.

#### Estimation Challenges

- High dimensionality of gradients.
- Computational complexity of estimating joint distributions.

#### Approximation Methods

1. **Parametric Models:**

   Assume gradients follow a multivariate Gaussian distribution. Compute MI analytically:

   - **Entropy of a Multivariate Gaussian:**

     \[
     H(\mathbf{X}) = \frac{1}{2} \ln \left( (2\pi e)^n |\Sigma| \right)
     \]

     where \( \Sigma \) is the covariance matrix.

   - **Mutual Information:**

     \[
     I(\mathbf{X}; \mathbf{Y}) = \frac{1}{2} \ln \left( \frac{ |\Sigma_X| |\Sigma_Y| }{ |\Sigma_{XY}| } \right)
     \]

2. **Kullback-Leibler Divergence:**

   Approximate MI using the KL divergence between joint and marginal distributions:

   \[
   I(X; Y) = D_{\text{KL}}(p(x, y) || p(x)p(y))
   \]

3. **Variational Mutual Information Estimation:**

   Use a neural network \( T_\phi \) to approximate the density ratio:

   \[
   I(X; Y) \geq \mathbb{E}_{p(x, y)}[T_\phi(x, y)] - \log \left( \mathbb{E}_{p(x)p(y)}[e^{T_\phi(x, y)}] \right)
   \]

   This lower bound is maximized with respect to \( \phi \).

#### Incorporating into Loss Function

Define a mutual information loss term:

\[
L_{\text{MI}} = -I(\nabla_{x_i} L; \nabla_{x_j} L)
\]

### Total Loss Function

\[
L_{\text{total}} = L_{\text{cls}} + \lambda_1 L_{\text{grad}} + \lambda_2 L_{\text{MI}}
\]

### Gradient of Mutual Information Loss

Compute \( \frac{\partial L_{\text{MI}}}{\partial \theta} \) using the chosen estimation method.

<div class="l-page">

  {% include figure.html path="assets/img/2025-04-28-can-we-learn-from-misclassifications/Fig-repre-clustering.png" class="img-fluid" %}

</div>
<div class="caption">
    Plotting the gradient weights in comparison to the latent representations and the final prediction scores using PCA. The second and third rows represent the clustering results compared to the original classes, using KMeans and GMM, respectively.
</div>

<div class="l-page">

  {% include figure.html path="assets/img/2025-04-28-can-we-learn-from-misclassifications/generic_classes_grid-1.png" class="img-fluid" %}

</div>
<div class="caption">
    Generated animal classes using stable diffusion.
</div>

<div class="l-page">

  {% include figure.html path="assets/img/2025-04-28-can-we-learn-from-misclassifications/snail_grid.png" class="img-fluid" %}

</div>
<div class="caption">
    Illustration of overlaid GradCAM map on the generated snail classes using Resnet50.
</div>

## Expirmantal Results 



| Model\Matrices | 'Cos_Sim_GW'  | 'Cos_Sim_P'   | 'Cos_Sim_FM'  | 'Grad_Cam_CC' | 'Grad_Cam_KLD'|'Grad_Cam_SIM' | 'Grad_Cam_AUC'|
| -------------- |:-------------:|:-------------:|:-------------:|:-------------:|:-------------:|:-------------:|:-------------:|
| Resnet18       | 0.832475      | 0.750405      | 0.809194      | 0.514107      | 0.546371      | 0.640893      | 0.473252      |
| Resnet34       | 0.891979      | 0.760314      | 0.821133      | 0.547336      | 0.303619      | 0.717424      | 0.482986      |
| Resnet50       | 0.970702      | 0.781508      | 0.842356      | 0.387207      | 0.845985      | 0.534813      | 0.441505      |
| Resnet101      | 0.898816      | 0.777873      | 0.841248      | 0.451866      | 0.607102      | 0.587721      | 0.471974      |
| Resnet152      | 0.918961      | 0.784820      | 0.834101      | 0.513276      | 0.560512      | 0.600739      | 0.498934      |


Where the columns of the table represent: 

- **Cos_Sim_G**: Cosin similarity based on the gradiant.
- **Cos_Sim_R**: Cosin similarity based on the final represantations.
- **Cos_Sim_act**: Cosin similarity based on the activation map.
- **Grad_Cam_CC**: Grad_Cam accuracy.
- **Grad_Cam_KLD**: Grad_Cam Kullback–Leibler divergence.
- **Grad_Cam_SIM**: Grad_Cam Saliency similarity metric .
- **Grad_Cam_AUC**: Grad_Cam area under the curve.

### Clustering using Kmeans



|                    | 'Resnet18'                    | 'Resnet34'                    | 'Resnet50'                    | 'Resnet101'                   | 'Resnet152'                   |
| ----------------   |:-----------------------------:|:-----------------------------:|:-----------------------------:|:-----------------------------:|:-----------------------------:|
| Clustring by\Metric| ARI  | NMI  | Silhouette      | ARI  | NMI  | Silhouette      | ARI  | NMI  | Silhouette      | ARI  | NMI  | Silhouette      | ARI  | NMI  | Silhouette      |
| Gradient Weights   | 0.95 | 0.98 | 0.79            | 0.63 | 0.85 | 0.62            | 1.00 | 1.00 | 0.89            | 0.95 | 0.98 | 0.83            | 0.77 | 0.91 | 0.81            |
| Final Predictions  | 0.30 | 0.64 | 0.42            | 0.50 | 0.78 | 0.42            | 0.41 | 0.69 | 0.33            | 0.34 | 0.66 | 0.39            | 0.53 | 0.77 | 0.43            |
| Feature Maps       | 0.41 | 0.72 | 0.42            | 0.57 | 0.80 | 0.51            | 0.41 | 0.73 | 0.49            | 0.23 | 0.57 | 0.41            | 0.39 | 0.69 | 0.35            |



### Clustering using GMM


|                    | 'Resnet18'                    | 'Resnet34'                    | 'Resnet50'                    | 'Resnet101'                   | 'Resnet152'                   |
| ----------------   |:-----------------------------:|:-----------------------------:|:-----------------------------:|:-----------------------------:|:-----------------------------:|
| Clustring by\Metric| ARI  | NMI  | Silhouette      | ARI  | NMI  | Silhouette      | ARI  | NMI  | Silhouette      | ARI  | NMI  | Silhouette      | ARI  | NMI  | Silhouette      |
| Gradient Weights   | 0.95 | 0.98 | 0.79            | 0.61 | 0.85 | 0.62            | 1.00 | 1.00 | 0.89            | 0.95 | 0.98 | 0.83            | 0.77 | 0.91 | 0.81            |
| Final Predictions  | 0.30 | 0.64 | 0.38            | 0.48 | 0.78 | 0.40            | 0.34 | 0.68 | 0.30            | 0.37 | 0.69 | 0.31            | 0.57 | 0.80 | 0.43            |
| Feature Maps       | 0.46 | 0.74 | 0.42            | 0.52 | 0.77 | 0.48            | 0.41 | 0.73 | 0.49            | 0.24 | 0.58 | 0.39            | 0.40 | 0.71 | 0.30            |



Where the columns of the table represent: 

- **ARI**: Adjusted Rand Index
- **NMI**: Normalized Mutual Information 
- **Silhouette**: Silhouette score

