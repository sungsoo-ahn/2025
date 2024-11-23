---
layout: distill
title: Demystifying Vectorfield Modeling in Generative Models
description: One of the most popular technics in recent years when it comes to generative models, have been methods that rely on modeling vectorfields of data transformations in various steps. One of the most recent and widely adpopted methods of this kind, is flow matching. This blog post focuses on the concept of vectorfields, vectorfield modeling via neural network, and its usecase in the flow matching framework.
future: true
htmlwidgets: true

authors:
  - name: Anonymous

# must be the exact same name as your blogpost
bibliography: 2025-04-28-vecfield-modeling.bib  

# Add a table of contents to your post.
#   - make sure that TOC names match the actual section names
#     for hyperlinks within the post to work correctly. 
#   - please use this format rather than manually creating a markdown table of contents.
toc:
  - name: Introduction
  - name: What is a Vectorfield
  - name: What is an ODE
  - name: What is an ODE Solver
  - name: Generative Modeling via Vectorfields
  - name: How to estimate the vectorfield function for a generative model from data?
  - name: Sampling in Vectorfield-Based Generative models
  - name: Choice of Prior Distribution
  - name: Reducing sampling steps

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
  .center-content {
  text-align: center;
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
---


## Introduction
In the recent years, generative models have been relying on a new class of models that learn to map a noise distribution to a target distribution such as images, given a set of data samples thrtough a multi-step denoising process.
Some of the well-known methods are diffusion models<d-cite key="ho2020denoising"></d-cite>, score-based models<d-cite key="song2019generative">dif</d-cite>, and flow matching<d-cite key="lipman2023flow">dif</d-cite>.
Such methods are however closely related, and their relation has been discussed in numerous work in the literature<d-cite key="song2020score,tomczak_score_matching,dieleman_perspectives_2023,luo_diffusion_tutorial_2022,weng_diffusion_models_2021"></d-cite>.
In this blog post, we dig deeper into some of the technical challenges and details of these models such as the concept of vectorfield, the use of ordinary differencial equations (ODS) for sampling, leveraging vectorfield modeling via neural networks for generative models such as flow matching, and more.

<div class="row mt-3" style="justify-content: center;">
{% include figure.html path="assets/img/2025-04-28-vecfield-modeling/dino.gif" class="img-fluid" %}
</div>


## What is a Vectorfield
A vector field is a mathematical construct that assigns a vector to every point in a space. It is often used to represent the distribution of a vector quantity, such as velocity, force, or magnetic field, across a region.

<aside class="l-body box-important" markdown="1">
We can define a vector field $\mathbf{F}$ in a space (e.g., $\mathbb{R}^2$ or $\mathbb{R}^3$) as a function that maps each point $\mathbf{x}$ to a vector $\mathbf{F}(\mathbf{x})$.
</aside>
Now let's take a closer look at some of these examples.

### Velocity Field
The following is an example of a velocity field, representing a rotational flow, often called a vortex:

$$\mathbf{v}(x, y) = \langle -y, x \rangle$$

The x-component of velocity ($$\mathbf{u}$$ is $$-y$$ and y-component of velocity ($$\mathbf{v}$$ is $$x$$. This configuration causes the flow to rotate counterclockwise around the origin.

We can visualize this vectorfield as follows:
<div class="row mt-3">
{% include figure.html path="assets/img/2025-04-28-vecfield-modeling/velocity_field.png" class="img-fluid" %}
</div>
<div class="caption">
    Velocity field of a vortex.
</div>

### Gravitational Field
Another example of a vectorfield is the gravitational field created due to a point mass. This can be furmulated as:

$$\mathbf{g}(x, y) = \left\langle -\frac{G \cdot m \cdot x}{r^2}, -\frac{G \cdot m \cdot y}{r^2} \right\rangle$$

where $$G$$ is the gravitational constant, and 
$$m$$ is the mass.
$${r^2}=x^2+y^2$$ is the square of the distance from the mass.The field points towards the mass, with magnitude inversely proportional to the square of the distance.


<div class="row mt-3">
{% include figure.html path="assets/img/2025-04-28-vecfield-modeling/gravitational_field.png" class="img-fluid" %}
</div>
<div class="caption">
    Gravitational field around a mass at origin.
</div>

### Magnetic Field

In this example, we use the magnetic field around a long, straight current-carrying wire (at the center), which can be written as:

$$\mathbf{B}(x, y) = \left\langle -\frac{\mu_0 \cdot I \cdot y}{r^2}, \frac{\mu_0 \cdot I \cdot x}{r^2} \right\rangle$$

$$\mu_0$$ is the magnetic constant, and $$I$$ is the current.
$${r^2}=x^2+y^2$$ is the square of the distance from the wire that we assume is at the center of the plot. The field circles around the wire, with magnitude inversely proportional to the distance.


<div class="row mt-3">
{% include figure.html path="assets/img/2025-04-28-vecfield-modeling/magnetic_field.png" class="img-fluid" %}
</div>
<div class="caption">
    Magnetic field around a long, straight current-carrying wire.
</div>

## What is an ODE?
Let's start by a simple example:
assume that the direction and rate of change of function $$y$$ w.r.t $$t$$ is described as follows:

$$\frac{dy}{dt} = -2y$$

In other words, this equation describes how the quantity $y$ changes over time (with $t$).

<aside class="l-body box-important" markdown="1">
As this equation uses the derivatives of $y$ w.r.t, it is called a *differential equation*.
Since this differential equation depends only on *one* variable $t$, it is called an *Ordinary Differential Equation*, or ODE.
</aside>

## What is an ODE Solver?

The above ODE defines the relationship of the variable $y$ with time, and more specifically, how $y$ changes through time.
Though it does not tell us the exact form of $y(t)$.
In order to find $y(t)$ from the given ODE, we have to *integrate* the rate of change for $y$ in order to find the original function $y(t)$ that descibes the behaviour of $y$ over time.

<aside class="l-body box-important" markdown="1" id="init-cond-example">
Given an initial condition, and over an specified interval, we can actually numerically approximate a solution to an ODE.
For instance, the *Euler method* provides an algorithm to achieve this.
The Euler method is a numerical technique used to approximate solutions to ODEs, and it is particularly useful for [initial value problems](https://en.wikipedia.org/wiki/Initial_value_problem). The method uses the derivative $\frac{dy}{dt}$ to estimate the value of $y$ at discrete time steps, rather than directly integrating 
$\frac{dy}{y}$:

$$y_{n+1} = y_n + h \cdot f(t_n, y_n)$$

where $\frac{dy}{dt}=f(t,n)$.
</aside>

If we start by the initial conditions of $y_0 = 1$ and $t_0 = 0$, and integrate for the range of 0 to 5 with step size of 0.1, we can follow the formula above and calculate every estimate for $y$ at the timesteps between 0 and 5.

<div class="row mt-3">
{% include figure.html path="assets/img/2025-04-28-vecfield-modeling/euler_solution.png" class="img-fluid" %}
</div>
<div class="caption">
   Euler method solving for $\frac{dy}{dt}=-2y$ and its analytical solution $e^{-2t}$.
</div>

## Generative Modeling by leveraging Vectorfields
Now that we have a better understanding of vectorfields, what they are, and how they are represented and used, let's focus on the main topic of this post, which is about generative modeling.

Vectorfields can be used for modeling generative models. In fact, there are many differet kinds of generative models that leverage vectorfields in their modeling<d-cite key="rezende2015variational, chen2018neural, song2019generative, rubanova2019latent,lipman2023flow"></d-cite>.
But here, we want to focus on one of the recent classes of generative models that leveragte vectorfields as their main modeling paradigm, namely flow matching<d-cite key="lipman2023flow"> </d-cite>.

Recall from the previous sections that if a quantity is represented by an ODE (e.g., [$\frac{dy}{dt}=f(t,n)$](#init-cond-example)) that describes its changes through the range of a given variable (e.g., $t$), we can integrate through that ODE to estimate the function of that quantity.

What we have at hand for this approximation, is the changes of $y$ w.r.t $t$. Let's now assume that, $t$ represents the level of noise (and not time) in $y$, which we can represent with the vectorfiled function $f$:

$$\frac{dy}{dt}=f$$

By integrating through all possible values of $t$ (e.g, having all levels of noise $t$ for $y$), we can find $y = \int f(t) \, dt + C$ where $C$ is a constant.
Therefore, using an ODE solver, and given the vectorfield function $f(t)$, we can have an approximation of $y$.
If the function $y$ here would be the data generation function we are interested in, and by having a good approximation for $f$, we now have a receipt for a generative model.

## How to estimate the vectorfield function for a generative model from data?


Let $x$ represent a sample from a data distribution $X$.
Let also $t$ be a variable representing the amount of noise in $x$: if $t=0$ there is only noise, and if $t=1$ there is no noise.
Given a noisy sample $x_n$ which has the noise level $t_n\in\[0, 1\]$, we would like to consider the following formulation:

$$x_{n+1} = x_n + h \cdot f(t_n, x_n)$$

where $f(t_n, x_n)$ is a vectorfield function representing the rate and direction of change for $x_n$ into $x_{n+1}$.
Given an initial condition of $t_0=0$ and $x_0 = z \sim \mathcal{N}(0, 1)$, we can use an ODE Solver such as Euler to find a denoised sample $x\simX$.


In other words, if we can manage to estimate this vectorfield, we will then have our generative model of $X$!
We can actually parameterize $f$ using a neural network such that given any $t$ it can estimate the direction of change to arrive at the denoised sample $x$:

$$f_{\theta}(x_n, t_n) \approx x_1 - x_0, n=[0,\cdots, 1]$$

We can actually train this parameterized model in a supervised fashion by creating a training set of $(x_\text{input},t, u_\text{output})$ where:


$$x_\text{input} = z + t * (x - z)$$

$$u_\text{output}=x - z$$


To create a diverse input, we can choose a point on a straight line between a real sample $x$ and a noise sample $z$, where $t$ defines the closeness of our sample to the real sample. After the training is finished, the model should be able to estimate the aforementioned vectorfield.

The vectorfield model is a neural network such as the following example, that can been trained supervised, with the $x_\text{input},t$ as input and $u_\text{output}$ as oputput (label), as denoted above.


```python
import torch
import torch.nn as nn

class VFNet(nn.Module):
    def __init__(self, nsteps=1):
        super().__init__()
        self.nsteps = nsteps
        self.vf_model = nn.Sequential(
            nn.Linear(3, 512),
            nn.SiLU(),
            nn.Linear(512, 512),
            nn.SiLU(),
            nn.Linear(512, 512),
            nn.SiLU(),
            nn.Linear(512, 512),
            nn.SiLU(),
            nn.Linear(512, 2),
        )
        self.mse_loss = nn.MSELoss()

    def vf_func(self, t, x):
        shape = x.shape
        x = x.reshape(-1, 2)
        t = t.reshape(-1, 1).expand(x.shape[0], 1)
        tx = torch.cat([t, x], dim=-1)
        return self.vf_model(tx)

    def loss(self, batch):
        bsz = batch.shape[0]
        x0 = torch.randn(bsz, 2).to(batch.device)
        t = torch.rand(bsz, 1).to(batch.device)

        # a sample on the optimal transport path (straight line) between the noise (x0) and data (batch)
        x_t = x0 + t * (batch - x0)

        # ground-truth vectorfield
        u_t = batch - x0

        # predicted vectorfield 
        v_t = self.vf_func(t, x_t)
        loss = self.mse_loss(v_t, u_t)
        return loss

```

## Sampling in Vectorfield-Based Generative models
After training $f_{\theta}$, we can initiate the sampling process via an ODE Solver such as Euler with the initial conditions of $t_1=0$ and $x_1=z \sim\mathcal{N}(0,1)$.
Using this vectorfield, we can generate new samples, which we visualize in the following plot:

<div class="row mt-3">
{% include figure.html path="assets/img/2025-04-28-vecfield-modeling/vec_density_samp.gif" class="img-fluid" %}
</div>
<div class="caption">
   Training and Sampling in a continous setting where $t=[0,1]$, and an adaptive ODE solver<d-cite key="hairer1993solving"></d-cite> was used for sampling.
</div>

## Choice of Prior Disribution
Often, the source distribution used in genrative models such as flow matching are chosen from simple distributions such as Gaussian.
However, this is not an stricktly limiting factor and we can train the vectorfield network to learn from more complex distributions as well.
To demonstrate this, we use a set of toy datasets, including some from the Datasaurous collection<d-cite key="cairo2016datasaurus"></d-cite>, and choose both source and target distributions from them, transforming the source to target, then use target as the next source.
You can see the results below:
<div class="row mt-3">
{% include figure.html path="assets/img/2025-04-28-vecfield-modeling/dopri.gif" class="img-fluid" %}
</div>
<div class="caption">
  [topleft] source. [topmiddle] generated data. [top right] target. 
  [bottomleft] density of generated data. [bottommiddle] evolution of generated samples through $t$. [right] predicted vectorfields.
</div>
You can see that, as long as the vectorfield model can properly learn to predict the vectorfield correctly, the generated samples are fairly good. Though this is only a small example and might not hold in extreme cases.

## Reducing sampling steps

Usually, a flow matching model is trained by continousely sampling $t\in[0,1]$, and consequently using an ODE solver that relies on a continous $t$.
But what if we fixed the solver steps before hand? E.g, Euler with $t\in T=\lbrace0,0.25,...,1 \rbrace$, and stepsize $h=\frac{1}{|T|}$?

Although this may not generalize as well as an adaptive solver, as a proof of concept we have trained a model by only drawing $t$ from $T$, and used a fixed-step Euler $h=\frac{1}{|T|}$ for sampling. Here is the result:



<div class="row mt-3">
{% include figure.html path="assets/img/2025-04-28-vecfield-modeling/5step.gif" class="img-fluid" %}
</div>
<div class="caption">
   Training and Sampling in a 5-step setting where $t={0, 0.25, ..., 1}$.
</div>

It appears that the vectorfield function is still learning meaningfully and can transfer the source samples towards the target.
Reducing sampling costs is an active area of research and similar ideas to the one above have been also explored in the literature<d-cite key="shaul2023bespoke"></d-cite>. 


