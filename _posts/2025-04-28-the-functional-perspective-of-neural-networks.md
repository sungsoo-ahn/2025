---
layout: distill
title: The Functional Perspective of Neural Networks
description: Common wisdom suggests that neural networks trained on the same dataset reaching the same accuracy and loss can be considered equivalent. However, when considering neural networks as functional representations of their input space, it becomes clear that they represent distinct functions that enable predictive capacity. In this blog post, we review functional perspectives used to understand the success of neural network ensembles on more modern architectures. We concurrently define weak and strong functional similarity analysis that assesses the functional diversity of neural networks with increasing fidelity; we identify the data primacy effect while elucidating the pitfalls of traditional approaches when considering neural networks from a functional perspective.  
date: 2025-04-28
future: true
htmlwidgets: true
hidden: false
authors:
  - name: Anonymous


# must be the exact same name as your blogpost
bibliography: 2025-04-28-the-functional-perspective-of-neural-networks.bib  

toc:
  - name: Why the Functional Perspective
    subsections:
    - name: How Can Functions Differ
  - name: Experimental Setup
  - name: Traditional Network Analysis
    subsections:
      - name: Test Error Analysis
      - name: Loss Analysis
      - name: Summary of Traditional Analysis
  
  - name: Weak Functional Analysis
    subsections:
      - name: Functional Similarity Visualisations
      - name: Prediction Analysis
      - name: Summary of Weak Functional Analysis
  - name: Strong Functional Analysis
    subsections:
      - name: Activation Distance
      - name: Cosine Similairty
      - name: JS divergence 
      - name: Summary of Strong Functional Analysis
  - name: Impact of Functional Network Analysis

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

<!-- Note: please use the table of contents as defined in the front matter rather than the traditional markdown styling. -->

##  Why the Functional Perspective

The genesis of regarding neural networks as function machines is owed to the original analysis of ensembling overfit neural networks in the 1990s as a way to reduce the generalisation gap for two-hidden layer networks performing classification tasks<d-cite key="hansen1990neural"></d-cite>. Ensembling, inspired by fault-tolerant computing, is the act of combining many "noisy" **shallow** network outputs to form one output that is less falible than any single network. Fort et al., popularised this understanding for **deep** neural networks where they explored the properties of ensembled neural networks via prediction comparisons over just traditional loss and accuracy analysis<d-cite key="fort2019deep"></d-cite>; the work echos the understanding of the "noisy" function combination and showed the same effect in deep neural networks that models form different functions when trained on the same data.

<a name="func_perspective" id="func_perspective">**The Functional Perspective**</a> : We define the Functional Perspective as any research endeavour that attempts to exhaustively characterise the divergence and nuance of all network outputs on a layer of interest in a comparative fashion using a combination of qualitative and/or quantitative lines of enquiry. 

Understanding that neural networks form different functions over their input space is a critical idea with numerous safety implications. In this blog post, we reproduce existing functional analysis conducted by Fort et al. on contemporary vision transformers, providing accuracy, loss, prediction disagreement and visual analysis of networks trained on the dataset. We then further the work by introducing the results for the best attempts of contemporary literature to capture the functional divergence of neural networks<d-cite key="klabunde2023similarity"></d-cite>. We provide a commentary on the pitfalls of traditional analysis of accuracy, loss and visual representations of network functions and outline the importance of culminating such analysis with more quantitative methods, closing with a perspective on the wider role functional analysis can have on neural network interpretability and safety. 

###  <a name="why_func" id="why_func">How Neural Network Functions Differ</a> 

Trained neural netorks can be considered functional representations of their input space. As a result, models that train on the same data can vary considerably on inputs, which leads to different overall behaviour. For example, in the figure below, we present two hypothetical models that are trained on the ten-class image classification task of CIFAR10 <d-cite key="krizhevsky2009learning"></d-cite>. 

<div class="row mt-3">
    <div class="col-sm mt-3 mt-md-0">
        {% include figure.html path="assets/img/2025-04-28-the-functional-perspective-of-neural-networks/Model_One.jpeg" %}
    </div>
    <div class="col-sm mt-3 mt-md-0">
        {% include figure.html path="assets/img/2025-04-28-the-functional-perspective-of-neural-networks/Model_Two.jpeg" %}
    </div>
</div>
<div class="caption">
    An example of two hypothetical models trained on CIFAR10 that have equivalent prediction agreement, accuracy and loss on the input space while concurrently having nonequivalent functions.
</div>

It is evident that these models will correctly classify the input image as a cat - additionally, it can be noted that both models have the same loss of **1.139**. Considering these two metrics (accuracy and loss) alone could lead to the misconception that these models are functionally equivalent, given the absolute similarity of their loss and accuracy. However, when considering the output probabilities representing each model's function, it is evident that the functions are different. 

For model one (left image), the four highest prediction probabilities other than the predicted class of cat (0.32) are that of the automobile (0.20), airplane (0.12), ship (0.10) and dog (0.10) - as a result from this output perspective it could be argued that the model's function puts this example closer to various vehicles over other animals. On the other hand, for model two (right image), the four highest prediction probabilities other than the predicted class of cat (0.32) are that of the dog (0.20), deer (0.12), bird (0.10) and frog (0.10). This model's function puts the input image of a cat closer to other species of animals. 



When considering the functions of the two models in deployment - despite the accuracy and the loss being equivalent, it would be reasonable to use model two for these types of inputs as it has a function that better captures the distinction between animals and vehicles. This function is an essential property of model two as it could suggest the model would be more robust. While this is a contrived example, it is not infeasible that such functions could arise in practice, which is why model evaluation should be expanded from loss and accuracy to include strong functional analysis perspectives.

# Experimental Setup

Typically, neural network similarity is considered from an accuracy and loss perspective. In the previous section, we have provided contrived examples of when this approach could be flawed. In this section, we look at using accuracy and loss to compare functional relations of neural networks trained under different conditions on CIFAR10<d-cite key="krizhevsky2009learning"></d-cite>. To make the experimental results relevant, we use the contemporary architecture of the Vision Transformer<d-cite key="dosovitskiy2020image"></d-cite>. All models have the same training hyperparameters with only two variables (initailisation and training data order) altered. 

The three conditions are as follows:

- **Base**: a base model which is trained on a dedicated seed
- **SIDDO**: a model which has the *same initialisation as the base model but is trained on a different data order*.
- **DISDO**: a model which has a *different initialisation but is trained with the same data order as the base model*.  



## Traditional Functional Analysis

The following subsections show how traditional accuracy and loss analysis may be misleading to compare function similairty through these three conditions. 

### Test Error Analysis

To evaluate the neural networks in the three different conditions based on the accuracy, we employ a landscape visualisation tool <d-cite key="li2018visualizing"></d-cite>, to present both 2D and 3D representations of the test error landscapes.

For the 2D and 3D test error plots below, at the minima, where X and Y coordinates are **(0,0)**, it can be observed that the **SIDDO**, **Base** and **DISDO** models have very similar test errors of **26.870**, **27.020** and **27.240**, respectively. Additionally, from both perspectives, it is hard to tell if the models are different, given the similarity between their 2D and 3D visualisations. 

Interact with the figures below and try to gain an understanding of the plots to get an intuitive gauge of the error spaces. 

<div class="l-page">
  <iframe src="{{ 'assets/html/2025-04-28-the-functional-perspective-of-neural-networks/test_error_landscapes.html' | relative_url }}" frameborder='0' scrolling='no' height="600px" width="100%"></iframe>
</div>

<div class="caption">
    <b>Two dimensional</b> test error landscape plots around the local minima by perturbing the parameters along two random approximately orthogonal filter-wise normalised directions in weight space X and Y <d-cite key="li2018visualizing"></d-cite>. 
    <br>
    <b>Left</b> represents a model with the same initailsation trained on a different data order (<b>SIDDO</b>) , <b>middle</b> is the base model (<b>Base</b>) and <b>right</b> is the model trained with a different initialisation but same data order as the base model (<b>DIDDO</b>). 
</div>


<div class="l-page">
  <iframe src="{{ 'assets/html/2025-04-28-the-functional-perspective-of-neural-networks/3d_test_error_landscapes.html' | relative_url }}" frameborder='0' scrolling='no' height="600px" width="100%"></iframe>
</div>
<div class="caption">
    <b>Three dimensional</b> test error landscape plots around the local minima by perturbing the parameters along two random approximately orthogonal filter-wise normalised directions in weight space X and Y<d-cite key="li2018visualizing"></d-cite>.
    <br>
    <b>Left</b> represents a model with the same initailsation trained on a different data order (<b>SIDDO</b>) , <b>middle</b> is the base model (<b>Base</b>) and <b>right</b> is the model trained with a different initialisation but same data order as the base model (<b>DIDDO</b>). 
</div>

Given the similarity of the test error values and visualisations, one could assume that the models have the same or similar functions, with some minor differences, given the subtle misalignments on the 3D plots. 

Comparing the Euclidean distance between **SIDDO** and the **Base** models, we observe a difference in test error landscapes of **159.675**. Comparing the Euclidean distance between **DISDO** and the **Base** models, we observe a difference in test error landscapes of **189.459**. These results suggest from the accuracy perspective that the **SIDDO** models have more similar function to the **Base** model than the **DISDO** models. While this suggestion seems reasonable, we explain further in the blog post why this illusion of functional similarity derived from test error landscapes does not hold, using strong functional similarity measures. 

### Loss Analysis

When considering the loss landscape visualisation analysis<d-cite key="li2018visualizing"></d-cite> in the 2D and 3D figures below, we are confronted with a simlair perspective. For the 2D and 3D test loss plots below, at the minima, where X and Y coordinates are **(0,0)**, for **SIDDO**, **Base** and **DISDO** the losses are **1.993**, **1.948** and **1.932**, respectively. Again, these values are not too dissimilar, and the 2D plots, in particular, suggest the same trend for their loss regions. 

Once again, we invite the reader to play with the 2D and 3D visualisations of the loss landscapes to get an intuitive feel for what the figures are conveying. 

<div class="l-page">
  <iframe src="{{ 'assets/html/2025-04-28-the-functional-perspective-of-neural-networks/test_loss_landscapes.html' | relative_url }}" frameborder='0' scrolling='no' height="600px" width="100%"></iframe>
</div>
<div class="caption">
    <b>Two dimensional</b> loss landscape plots around the local minima by perturbing the parameters along two random approximately orthogonal filter-wise normalised directions in weight space X and Y<d-cite key="li2018visualizing"></d-cite>.
    <br>
    <b>Left</b> represents a model with the same initailsation trained on a different data order (<b>SIDDO</b>) , <b>middle</b> is the base model (<b>Base</b>) and <b>right</b> is the model trained with a different initialisation but same data order as the base model (<b>DIDDO</b>). 
</div>
<div class="l-page">
  <iframe src="{{ 'assets/html/2025-04-28-the-functional-perspective-of-neural-networks/3d_test_loss_landscapes.html' | relative_url }}" frameborder='0' scrolling='no' height="600px" width="100%"></iframe>
</div>
<div class="caption">
    <b>Three dimensional</b> loss landscape plots around the local minima by perturbing the parameters along two random approximately orthogonal filter-wise normalised directions in weight space X and Y<d-cite key="li2018visualizing"></d-cite>.
    <br>
    <b>Left</b> represents a model with the same initailsation trained on a different data order (<b>SIDDO</b>) , <b>middle</b> is the base model (<b>Base</b>) and <b>right</b> is the model trained with a different initialisation but same data order as the base model (<b>DIDDO</b>). 
</div>

On this occasion, when we consider the loss landscape 3D visualisation, statements could be made about the similarity of the loss landscapes and, thus, the similarity of functions for the **SIDDO**, **Base** and **DISDO** models. 

Comparing the Euclidean distance between **SIDDO** and the **Base** models, we observe a difference in loss landscapes of **33.622**. Comparing the Euclidean distance between **DISDO** and the **Base** models, we observe a difference in loss landscapes of **52.074**. These results suggest from the loss perspective that the **SIDDO** models have more similar function to the **Base** model than the **DISDO** models. While this suggestion seems reasonable, we explain further in the blog post why this illusion of functional similarity derived from loss landscapes does not hold, using strong functional similarity measures. 

### Summary of Traditional Analysis

The neural networks trained in these conditions are similar from the test accuracy and loss perspective. Their test accuracy landscape does not deviate massively with perturbation, and despite the loss landscapes appearing differently on the 3D visualisation, the 2D visualisations resemble one another. Throughout the rest of this blog post, we argue why this perspective alone is not enough to gauge the functional similarity of the models in these conditions and discuss and present alternate avenues for analysis that yield improved insight. 

## Weak Functional Analysis

In this section, we employ the analysis conducted by Fort et al.,<d-cite key="fort2019deep"></d-cite> using TSNE<d-cite key="van2008visualizing"></d-cite> that shows functional dissimilarity of neural networks. We further this by inclduing other unexplored visualisation methods of PCA<d-cite key="pearson1901liii"></d-cite>, MDS<d-cite key="mead1992review"></d-cite> and Spectral Embedding<d-cite key="10.5555/2980539.2980616"></d-cite> to support the analysis further. We also explore functional similarity divergence that can be captured at a low fidelity by the Prediction Dissimilarity metric used by Fort et al.,. For both visualisation and prediction dissimilarity, we discuss how these avenues for comparing functions may be misleading and incapable of describing the intricacies of functional divergence but do provide a high-level trends which are important to recognise.

### Functional Similarity Visualisations

For the functional analysis of neural networks Fort et al., employed a TSNE embedding visualisation to visualise qualitatively how functions diverge. Their findings showed that although neural networks have similar loss and accuracy, they are represented in different functional spaces. The visualisation served as compelling evidence for neural networks forming different noisy functions <d-cite key="hansen1990neural"></d-cite>. In the recreation of this plot we have used TSNE visualisations and other embeddings (PCA, Spectral Embedding and MDS) to show the same functional divergence. In the figure below, we plot the function during training, which shows that neural networks become increasingly functionally dissimilar over training. The agreement of this general trend across different embedding strategies shows that this finding is robust. 

<div class="row mt-3">
    <div class="col-sm mt-3 mt-md-0">
        {% include figure.html path="assets/img/2025-04-28-the-functional-perspective-of-neural-networks/projections_3d.png" %}
    </div>
</div>
<div class="caption">
    Qualitative visualisations of the <b>SIDDO</b>, <b>Base</b> and <b>DISDO</b> functions over training. <b>Top-Left</b> TSNE, <b>Top-Right</b> Pinciple Component Analysis , <b>Bottom-Left</b> Spectral Embedding and <b>Bottom-Right</b> Multiple Dimensional Scaling. 
</div>

However, it is important to note that while there is an agreement in the general trend of functional divergence, each of the visualisation strategies shows different training paths - the nuances in each method's functional pathways between **SIDDO**, **Base**, and **DISDO** could lead to different conclusions which may be an artefact of the specific method. As a result, to have the most informative view when using such qualitative methods, it is essential to use a range of embeddings to confirm overall trends without reading too much into the functional illusions that stem from a particular method. Moreover, it is challenging to make statements on the specific functional proximities of different training conditions on functional similarity via functional visualisation as each of the final functional locations is hard to compare qualitatively and can only be commeteted on subjectively. 



### Predcition Disagreement Analysis

Prediction disagreement quantifies how frequently two or more neural networks have the same classification on the same input. It provides a proxy for understanding which inputs neural networks diverge on and allows one to reason how these networks represent different functions. While it can be considered a more quanatative metric, we regard it as a weak functional similairty measure as it only conisders the **argmax** of model outputs instead of measuring the function space of predictions. 

The figure in the section <a href="#why_func">**How Neural Network Functions Differ**</a> illustrates how this metric may provide functional similarity illusions as models can agree on the final prediction but have a divergent prediction space that can indicate apparent modelling properties absent in this analysis. We include it within the weak functional similarity analysis section as it only captures a general trend, which aids the understanding of functional divergence but does not provide a means for evaluating the functions of each model and how they are different as it focuses solely on the **argmax prediction**. 

One could imagine a scenario in which prediction disagreement could lead to false conjecture on functional similarity. For example, two models disagree on the final classification of an input item, such as one model predicting a cat and the other predicting a dog. Yet, if there is only a small difference between each class, these models could be very similar in their prediction output space and, thus, function. However, they are dissimilar from a prediction disagreement perspective, which could be incorrect when considering the overall function space.   

The figure below depicts how the prediction disagreement changes between the **Base** model and the models in the **SIDDO** and **DISDO** conditions during training. The figure provides an intuitive understanding that each model has different functions, which results in prediction discrepancy, which gets stronger through training. However, despite the pitfalls of this functional analysis method, it does reaffirm the notion that accuracy and loss provide a myopic perspective of similarity that must be explored beyond to understand the properties of individual models.

<div class="l-page">
  <iframe src="{{ 'assets/html/2025-04-28-the-functional-perspective-of-neural-networks/predictions.html' | relative_url }}" frameborder='0' scrolling='no' height="600px" width="100%"></iframe>
</div>
<div class="caption">
 Prediction Disimilairty of <b>SIDDO</b> compared to the <b>Base</b> (<b>Left</b>) <b>Base</b> compared to the <b>Base</b> (<b>Middle</b>) and <b>DISDO</b> compared to the <b>Base</b> (<b>Right</b>) during training - a higher prediction disimailirty indicates less agreement on prediction.
</div>

The Prediction Disimilairty of **SIDDO** compared to **Base** shows that early on in training, the models very quickly diverge from predicting the same outputs, with the mdoels starting with a prediction disagreement of **0**. After one epoch (coordinate (1,1)) **SIDDO** and the **Base** have a prediction disagreement of **0.355**, although this does reduce by the end of training (coordinate (200,200)) to **0.263**. 

The Prediction Disimilairty of the **Base** compared to **Base** shows that early in training, there is a lot of change in the prediction outputs from epoch to epoch; however, this decreases through training, with coordinate (0,1),  coordinate (99,100) and coordinate (199,200) resulting in a prediction disagreement of **0.944**, **0.142** and **0.019**, respectively. The diagonal throughout is **0.0** due to being the same model. 

The Prediction Disimilairty of **DISDO** compared to **Base**, starts with a high prediction disagreement of **0.949**, coordinate (0,0); however, after one epoch of training, this reduces to **0.228** coordinate (1,1), this value devates through training however, results in **0.251**, coordinate (200,200).

Interestingly, by the end of training, the **DISDO** model is more similar to the **Base** model than **SIDDO** even though **SIDDO** started with the Prediction Disimilairty of 0 and **DISDO** started with **0.949**. This result suggests that the data order is more important than initialisation being the same for models to be functionally similar. 


### Summary of Weak Functional Analysis

The weak functional analysis indicates that neural networks have different functions, even when using the same initiation **SIDDO** or data order **DISDO**. However, this analysis does not fully convey the functional differences in high fidelity and thus lacks deeper insights and can fall foul of functional illusions. 

## Strong Functional Analysis

In this section, we extend the work of Fort et al., to show how strong functional similarity can provide improved insights into the functional similarity of neural networks and how often they tell a disjointed story from that presented by traditional lines of analysis. The metrics selected represent a portion of the available documented functional analysis metrics<d-cite key="klabunde2023similarity"></d-cite>.

Akin to the previous section, we use the same architecture, datasets and experimental setups to explore the functional similarity. The only modification is that the **SIDDO** and **DISDO** conditions are averaged across three models, which is more feasible because no visualisations are required. In the plots, **Model 1** always refers to the **Base** model. The descision to average the results was made to provide more robustness to the overall analysis and resulting conclusions made in this section. 

For consistency, our calculations of the respective metrics in the figures in this section below are done by comparing each model's output function against every other model and then averaging the metrics per epoch and plotting the values across training.

### Activation Distance

$$\|\mathbf{m^{1}(input)} - \mathbf{m^{2}(input)}\|_2 = \sqrt{(m_{1}^{1} - m_{1}^{2})^2 + (m_{2}^{1} - m_{2}^{2})^2 +...+ (m_{n}^{1} - m_{n}^{2})}$$

Activation distance <d-cite key="chundawat2023can"></d-cite>, also reported as the norm of prediction difference<d-cite key="klabunde2023similarity"></d-cite>,  represents the *l2* distance of neural network outputs - from this distance, the predictive similarity between neural networks can be better understood quantitatively. A **lower activation distance** closer to 0 indicates less functional deviation between model outputs, while a **higher activation distance** suggests functional divergence. It can be calculated by taking the outputs of two models and averaging the *l2* distance of outputs across input batches, **as shown in the Python code below:**

{% highlight python %}

import torch
import torch.nn as nn
def activation_dist_fn(model1,model2,softmaxed=False):
    softmax = nn.Softmax(dim=1)
    if not softmaxed:
        model1 = softmax(model1)    
        model2 = softmax(model2)
    distances = torch.sqrt(torch.sum(torch.square(model1 - model2), axis = 1))
    return distances.mean().item()

{% endhighlight %}

When we compare the activation distance of neural networks trained in different conditions, it can be observed that neural networks, regardless of **SIDDO** or **DISDO** conditions, are dissimilar not only to the **Base** model but to one another. If the activation distance over training remains at or close to 0, one could argue that the models have the same function. However, as we see this activation distance deviate over training, it can be understood that the models move in different functional directions during training. 

<div class="row mt-3">
    <div class="col-sm mt-3 mt-md-0">
        {% include figure.html path="assets/img/2025-04-28-the-functional-perspective-of-neural-networks/act_both.png"%}
    </div>
</div>
<div class="caption">
    Average activation distance of model outputs on the test set through training. <b>SIDDO</b> is presented on the <b>left</b> and <b>DISDO</b> is presented on the <b>right</b>.  <b>Model 1</b> represents the <b>Base</b> model. Higher values indicate increased functional divergence and lower values indicate functional similairty. 
</div>

Moreover, from the above figure, simple factors can impact the functional similarity of the networks. In this instance, models trained in the **SIDDO** condition are less functionally similar than models trained in the **DISDO** condition; this suggests that for models to be more functionally similar, the data order is more important than initialisation being the same. We name this phenomenon <a href="#data_primacy (need to define)">**The Data Primacy Effect**</a>.

<a name="data_primacy" id="data_primacy">**The Data Primacy Effect**</a> : Is a phenomena that acknowledges the importantace of data order for functional similairty, wherein models that take similar updates during training end up in local minima that have a closer functional representation. 

As a result, it can be understood that even though these models reach similar overall loss and accuracy, the functions they create are fundamentally determined by the data and order on which they are trained. From the test 3D loss landscapes produced earlier the models with the **SIDDO** condition would be assumed to be more functionally similar as the visualisations suggest similarity, however, it is the case that **DISDO** can have very different loss landscapes but can resemble similar functions when considering activation distance of predictions. 

### Cosine Similairty

$$\cos \theta = \frac{\mathbf{m^{1}(input)} \cdot \mathbf{m^{2}(input)}}{\|\mathbf{m^{1}(input)}\| \|\mathbf{m^{2}(input)}\|}$$

Cosine Similarity is a metric to measure the cosine angle between two vectors. As a result, model outputs can be vectorised and compared to distinguish how similar their outputs are. For the cosine similarity metric, values that tend towards 1 suggest a more similar functional representation, while values close to zero suggest orthogonal outputs and values of -1 represent polar outputs. This provides a computationally inexpensive mechanism for calculating the functional similarity between model predictions. **The Python code below shows how it can be implemented:**

{% highlight python %}
import torch
import torch.nn as nn

def cosine_sim_fn(model1, model2,softmaxed=False):
    cs = torch.nn.CosineSimilarity(dim=1)
    softmax = nn.Softmax(dim=1):
    if not softmaxed:
        model1 = softmax(model1)    
        model2 = softmax(model2)
    return cs(model1, model1).mean()
{% endhighlight %}


The figure below for both the **SIDDO** and **DISDO** conditions largely reflects that of the results observed for the activation distance plots. At the start of training, the cosine similarity of the models is high, with a sharp drop off in the initial epochs, followed by a steady decline of cosine similarity during the middle of training, which finishes with a slight increase towards the end of training. The overall trend here is that each of the models have different functions as soon as training begins, and they remain different (albeit with varying values) throughout training. 

<div class="row mt-3">
    <div class="col-sm mt-3 mt-md-0">
        {% include figure.html path="assets/img/2025-04-28-the-functional-perspective-of-neural-networks/cs_both.png"%}
    </div>
</div>
<div class="caption">
    Average Cosine similairty of model outputs on the test set of models through training. <b>SIDDO</b> is presented on the <b>left</b> and <b>DISDO</b> is presented on the <b>right</b>.  <b>Model 1</b> represents the <b>Base</b> model. Lower values indicate increased functional divergence and higher values indicate functional similairty.  
</div>

Furthermore, there is an agreement between activation distance and cosine similarity, which states that models within the **DISDO** are more functionally similar than models in the **SIDDO** condition. For **DISDO**, the final cosine similarity value is higher than that of **SIDDO**; additionally, for **SIDDO**, the cosine similarity drops lower **(circa 0.75)** than any value for **DISDO**. The agreement across metrics further suggests the  <a href="#data_primacy">**The Data Primacy Effect**</a>.

### Jensen Shanon Divergence

$$\mathrm{JS}(m^{1} \| m^{2}) = \frac{1}{2} \mathrm{KL}(m^{1}(input) \| M) + \frac{1}{2} \mathrm{KL}(m^{2}(input) \| M)$$

$$M = \frac{1}{2}(m^{1}(input) + m^{2}(input)$$

$$\mathrm{KL}(m^{1}(input) \| M) = \sum_{i} m^{1}(input)(i) \log \frac{m^{1}(input)(i)}{M(i)}$$

Jenson-Shanon (JS) Divergence represents a weighted average of KL divergence that can be employed to evaluate two non-continuous distributions  <d-cite key="lin1991divergence"></d-cite> and is leveraged to understand the functional divergence between model outputs. Models with **high functional similarity have values that tend towards 0**, and models that are **less functionally similar have relatively higher values**. However, the distinction is less clear than with other metrics. **The code below details how JS Divergence can be implemented in Python:**
{% highlight python %}
# Code adapted from https://stackoverflow.com/questions/15880133/jensen-shannon-divergence
import numpy as np
import torch.nn as nn
from numpy.linalg import norm
from scipy.stats import entropy

def JSD(model1, model2, softmaxed=False):
    softmax = nn.Softmax(dim=1):
    if not softmaxed:
        model1 = softmax(model1)    
        model2 = softmax(model2)
    _model1 = model1 / norm(model1, ord=1)
    _model2 = model2 / norm(model2, ord=1)
    _M = 0.5 * (_model1 + _model2)
    return (0.5 * (entropy(_model1, _M) + entropy(_model2, _M))).mean()

{% endhighlight %}

The figure below for the **SIDDO** and **DISDO** conditions largely reflects the results observed for both the activation distance and cosine similarity plots. At the start of training, the JS divergence of the models is essentially zero, with a sharp increase in the initial epochs, followed by a steady increase in the middle of training and a slight decrease towards the end of training. The noticeable trend is that each of the models has different functions as soon as training begins, and they remain different throughout training; again, there is consistency between the functional distance of all the models in the respective conditions, which strengthens the notion that different functions form through training for different models.

<div class="row mt-3">
    <div class="col-sm mt-3 mt-md-0">
        {% include figure.html path="assets/img/2025-04-28-the-functional-perspective-of-neural-networks/js_both.png"%}
    </div>
</div>
<div class="caption">
    Average Cosine similairty of model outputs on the test set of models through training. <b>SIDDO</b> is presented on the <b>left</b> and <b>DISDO</b> is presented on the <b>right</b>.  <b>Model 1</b> represents the <b>Base</b> model. Lower values indicate increased functional divergence and higher values indicate functional similairty.  
</div>

The plots conclude that there is a total agreement between all of the respective strong functional similarity measures that models within the **DISDO** are more functionally similar than models in the **SIDDO** condition. Models in **SIDDO** always have the most functional divergence from one another compared to models in the **DISDO** condition. As a result, the results strongly suggest the impacts of  <a href="#data_primacy">**The Data Primacy Effect**</a>.

### Summary of Strong Functional Analysis

A noticeable trend within the strong functional analysis of models is that they clearly depict the functional diversity of neural networks trained on the same dataset. The metrics provide a more detailed insight into the functional distance between models, which is more grounded than weak functional similarity analysis, which is open to more subjective interpretation. Additionally, while these metrics measure different qualities of functional similarity, they largely agree with general trends of weak functional analysis, which shows that they provide a more robust perspective of neural network functional diversity. Moreover, a more transparent understanding of functional diversity can be obtained when combined with visualisations. A point of interest that has arisen from the strong functional analysis results is that models in the **DISDO** condition are more functionally similar than models within **SIDDO**, which shines a light on the functional variation derived from different data orders and the impact of  <a href="#data_primacy">**The Data Primacy Effect**</a>.  

# Impact of Functional Network Analysis

When considering model safety with respect to the <a href="#func_perspective">**Functional Perspective**</a> we argue that models should be analysed and tested independantly given that functional divergence occurs for networks with the same architetcure trained on the same dataset. As a result, there can be more robust stress testing efforts of neural networks which can lead to more precise operational bound identification. 

Moreover, employing the  <a href="#func_perspective">**Functional Perspective**</a>  will aid endeavours in interpretability and help avoid common misconceptions of the relation between networks, their accuracy and loss as a way to comapre functional simailrity.
