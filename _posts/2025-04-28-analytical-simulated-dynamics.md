---
layout: distill
title: A primer on the analytical learning dynamics of nonlinear neural networks
description:
    The learning dynamics of neural networks—in particular, how parameters change over time during training—describe how data, architecture, and algorithm interact in time to produce a trained neural network model. Characterizing these dynamics in general remains an open problem in machine learning, but, handily, restricting the setting allows careful empirical studies and even analytical results. In this blog post, we review approaches to analyzing the learning dynamics of nonlinear neural networks, focusing on a particular setting that permits an explicit analytical expression for the generalization error of a nonlinear neural network trained with online gradient descent. We provide an accessible mathematical formulation of this analysis and a `JAX` codebase to implement simulation of the analytical system of ordinary differential equations alongside neural network training in this setting. We conclude with a discussion of how this analytical paradigm has been used to investigate generalization in neural networks and beyond.
date: 2025-04-28
future: true
htmlwidgets: true
hidden: false

# TODO: Anonymize when submitting!
authors:
  - name: Rodrigo Carrasco-Davis
    url: "https://scholar.google.cl/citations?user=PLBqVGoAAAAJ"
    affiliations:
      name: Gatsby Unit, UCL
  - name: Erin Grant
    url: "https://eringrant.github.io/"
    affiliations:
      name: "Gatsby Unit & SWC, UCL"

# must be the exact same name as your blogpost
bibliography: 2025-04-28-analytical-simulated-dynamics.bib

# Add a table of contents to your post.
#   - make sure that TOC names match the actual section names
#     for hyperlinks within the post to work correctly. 
#   - please use this format rather than manually creating a markdown table of contents.
toc:
- name: "Background"
  subsections:
    - name: Theoretical analyses of neural network dynamics 
    - name: Statistical physics for neural network dynamics 
      subsections:
        - name: A long history 
        - name: "Our focus: Narrow neural networks"
          subsections:
            - name: "The classical analysis"
            - name: "Modern extensions of the analysis"
- name: "Methods"
  subsections: 
    - name: Simulating the ordinary differential equations
    - name: Simulating neural network training
- name: "Replications"
  subsections:
    - name: Theory-experiment overlap in the soft committee machine
    - name: The specialization transition tracks feature learning
    - name: Large initial weights produce individual differences
    - name: Theory-experiment overlap in two-layer neural networks
    - name: Limits of the analytical teacher-student setting
- name: Impact & extensions
  subsections:
    - name: Applications of the teacher-student setting
    - name: The analytical frontier
---

## Background

The dynamics of learning in artificial neural networks capture how the parameters of a network change over time during training as a function of the data, architecture, and algorithm. 
These dynamics describe how the parameters, and thus the internal representations and external behaviors of the network, evolve during training.
In neural networks with at least one hidden layer, these dynamics are *nonlinear* even when the activation function is linear, making them challenging to characterize.

[### Empirical studies of neural network dynamics]: # 

### Theoretical analyses of neural network dynamics 

Theoretical approaches to understanding neural network learning dynamics have revealed two distinct training regimes.
The first regime, known as *lazy training*,<d-cite key="chizat2019lazy"></d-cite> occurs when network parameters stay close to their initialization throughout training, which can be induced by initializing with small weights.
In this regime, the network behaves similarly to a kernel method, with dynamics characterized by the *neural tangent kernel* (NTK).<d-cite key="jacot2020neural"></d-cite><d-cite key="arora2019finegrained"></d-cite><d-cite key="allen-zhu2020learning"></d-cite>

The second regime, termed *feature learning*<d-cite key="yang2022feature"></d-cite>, represents a more complex dynamics in which networks substantially modify their internal representations during training. 
This regime better captures the behavior of modern deep learning systems, where networks learn nontrivial features of a task. 
The interplay between these regimes depends on factors like network width, learning rate, and initialization scale.
Recent work has shown that the transition between lazy and feature learning regimes can be controlled through careful scaling of these parameters.

Even seemingly simple architectures like deep linear networks can exhibit rich feature learning dynamics<d-cite key="saxe2014exact"></d-cite>, including distinct learning phases where different hierarchical features emerge at different timescales.
This theoretical framework has proven valuable for analyzing phenomena like implicit regularization, optimization landscapes, and the role of overparameterization in deep learning.

### Statistical physics for neural network dynamics 

Statistical physics offers powerful tools for characterizing macroscopic behavior emerging from collections of microscopic particles, an approach with deep roots in physics. 
Early pioneering work by Gardner applied these techniques to neural networks by viewing neurons as microscopic particles in a complex system <d-cite key="gardner1989three"></d-cite>.

The primary goal of statistical physics approaches is to analytically characterize the time-varying generalization error through reduction to macroscopic variables to be defined.
These analyses can be performed under various neural network parameter regimes, often considering asymptotic limits (such as infinite width or input dimension) where the system concentrates—exhibiting fewer fluctuations in a precise sense—leading to simpler dynamical descriptions; see Cui (2024)<d-cite key="cui2024highdimensional"></d-cite> for a review of these regimes.

#### Our focus: Narrow neural networks in the teacher-student setting

The teacher-student framework, introduced by Gardner <d-cite key="gardner1989three"></d-cite>, provides a controlled setting for studying neural network learning with statistical physics techniques.
In this paradigm, a student network learns to mimic a fixed teacher network that generates the training data.
Classical analytical results were achieved by Saad and Solla <d-cite key="saad1995online"></d-cite> and Riegler and Biehl <d-cite key="riegler1995online"></d-cite> in the 1990s, who derived exact equations for the generalization dynamics this teacher-student setup with a particular scaling.

These analytical results provide exact, time-resolved predictions of learning dynamics, captures nonlinear feature learning phenomena, and allows precise study of generalization behavior.
These analyses requires specific assumptions about the data distribution—in particular, that it is Gaussian—and the results require the input dimension to be large to enable solvability.
Despite these constraints, this framework allow granular study of various learning regimes, including feature learning.

Much recent work builds on these classical analyses to expand the frontier of solvable training regimes, exploring parameter settings beyond those considered by Saad and Solla and Riegler and Biehl. <d-cite key="goldt2020dynamics"></d-cite>
These techniques have also found applications beyond generalization error dynamics, which we detail in the final section of this blog post. 
Next, we focus on the foundational results from Saad and Solla that form the basis of these analyses.

## Methods

In this section, we describe the teacher-student neural network setup. Next, we present the mathematical 
framework 
to characterize the gradient descent learning dynamics of the student network in terms of its order param
eters. To 
complement the simulations  and theoretical framework, we include code snippets for computing each compon
ent.

Each element of the teacher-student setup is implemented  as an object, with the underlying computations 
efficiently 
coded in JAX for clarity and speed. The code adheres closely to the notation used in the equations,  leve
raging 
broadcasting to accelerate computations and enabling JIT compilation for further optimization

### The teacher-student setup

In the teacher-student setting, the data generation process used to train the student network is 
described by a teacher network. We focus on the online learning setting, where new samples are drawn from the data generation process. In this setting, as new samples are drawn, the batch size has no effect on the dynamics, 
except to reduce the noise in the gradient estimate. We sample multiple $(x_{s}, y_{s})^{u}$ pairs to fill a batch 
$s = 1, \ldots, B$ and update the student network using gradient descent iterations, indexed by $u$.

A teacher can be defined in a general way by 

$$\begin{equation} 
y_{s}^{u} = f^{*}(x_{s}^{u}, W^{*}) + \sigma \xi^{u}, 
\end{equation} $$ 

where $f^{*}( \cdot , W^{*})$ is the mapping defined by the teacher network with parameters $W^{*}$, $\xi^{
u} \sim \mathcal{N}(0, 1)$, and 
$\sigma$ scales the output noise. The student network is generally defined as 

$$ \begin{equation}
\hat{y}_{s}^{u} = f(x_{s}^{u}, W), 
\end{equation} $$ 

where $f( \cdot , W)$ is the mapping generated by the student, and $W$ are the parameters of the student network. 
In general, $f$ and $f^{*}$ are the same (e.g., a neural network), where the parameters are the weights of the network.

To train the student network, we use gradient descent to minimize the mean squared error function at iteration $u$:

$$ \begin{equation}
\mathcal{L}^{u} = \frac{1}{2B} \sum_{s=1}^{B} \left( \hat{y}_{s}^{u} - y_{s}^{u} \right)^{2}, 
\end{equation} $$

where samples to fill a batch consist of $(x_{s}, y_{s})^{u}$ pairs with $i = 1, \ldots, B$, 
$x \sim \mathcal{N}(0, I)$ of dimension $N$, and the target $y_{s}^{u}$ is generated by feeding $x_{s}^{u}$ to the 
teacher network. 
We then update the student network's weights to minimize the loss function $\mathcal{L}^{u}$, effectively learning 
the same mapping as the teacher. The weights of the student network are updated using gradient descent as follows:

$$ \begin{equation}
W^{u+1} = W^{u} - \eta_{W} \frac{\partial \mathcal{L}^{u}}{\partial W}, 
\end{equation} $$

where $\eta_{W}$ is the learning rate. 

**How is the teacher-student setting useful here?** One way to analyze learning 
dynamics in neural networks is to pose the learning network as a dynamical system, where the gradient descent updates 
effectively evolve through continuous time as the parameters of a dynamical system. This transformation is 
commonly known as the *gradient flow limit*, where the discrete gradient descent updates become continuous when 
the learning rate is small, giving

$$ \begin{equation}
\frac{dW}{dt} = - \left\langle \frac{\partial \mathcal{L}^{u}}{\partial W} \right\rangle_{x,y}, 
\end{equation} $$

and a generalization error at each step as

$$ \begin{equation}
\mathcal{E}(W) = \frac{1}{2} \left\langle \left( \hat{y} - y \right)^{2} \right\rangle_{x,y}, 
\end{equation} $$

where the expectation is taken over the data distribution. One way to think about this limit is by considering that 
as the learning rate gets smaller, the amount of data observed by the network increases, becoming virtually infinite 
when the learning rate is zero. This converts the average over data in the loss function to an expectation over the 
data. 

It is possible to solve the above equation for linear networks (CITE ANDREWS, CLEM'S, AND LUKAS' WORK, 
\url{https://arxiv.org/abs/2405.17580}, \url{https://pubmed.ncbi.nlm.nih.gov/38288081/}). For 
non-linear networks, however, other techniques must be used, such as mean-field theory approaches (CITE THIS). 
In particular, using a teacher-student setup allows for the derivation of a closed-form expectation of the 
learning dynamics, even for non-linear transfer functions. To achieve this, the above differential equation can be 
written in terms of the **order parameters**, which fully define the state of the system at each time step. 

In the next sections, we will derive these equations for two specific cases: (SAAD AND SOLLA 1995), where the teacher 
and student are a soft-committee machine, and (GOLDT 2020), which extends these results to allow for non-linear 
two-layer neural networks.

### Theory-experiment overlap in the soft committee machine

To solve the above differential equation, we need to assume a specific form for the teacher and student networks and 
convert the dynamical equation that describes gradient descent to the corresponding order parameters equations. 
In Saad and Solla's work, both the teacher and student networks were modeled as a soft-committee machine, which is an average of non-linear perceptrons. We define the teacher and student networks as follows 
(with a slightly modified version):

$$ \begin{equation}
y_{s}^{u} = \sum_{m=1}^{M} g\left( \frac{W^{*}_{m} x_{s}^{u}}{\sqrt{N}} \right) + \sigma \xi^{u} \hspace{
0.3cm} \text{ and } \hspace
{0.3cm} 
\hat{y}_{s}^{u} = \sum_{k=1}^{K} g\left( \frac{W_{k} x_{s}^{u}}{\sqrt{N}} \right),
\end{equation} $$

where $g( \cdot )$ is the error function (there are some closed form solution for $g( \cdot )$ as a ReLU non-linearity), 
$k$ indexes the perceptrons (rows of $W \in \mathbb{R}^{1 \times N}$), and $M$ 
and $K$ are the number of neurons in the teacher and student networks, respectively. From here on, neuron indexing will
be $i, j, k$ for the student, and $m, n, p$ for the teacher. 

To train the student, we minimize the mean squared error between the teacher and student outputs. 
We then perform gradient descent to update the student's weights.

$$ \begin{align} 
\mathcal{L^{u}} = & \frac{1}{2B} \sum_{i=1}^{B} \left( y_{s}^{u} - \hat{y}_{s}^{u} \right)^{2} \\
 = & \frac{1}{2B} \sum_{s=1}^{B} \left[ \sum_{m=1}^
{M} g\left( \frac{W^{*}_{m}x_{s}^{u}}{\sqrt{N}} \right) + \sigma \xi^{u} -\sum_{k=1}^{K} g\left( \frac{W_
{k}x_{s}^{u}}{\sqrt{N}} 
\right) \right]^{2} \\
\frac{\partial \mathcal{L}}{\partial W_{i}} = & \frac{1}{B}\sum_{s=1}^{B} \left[\sum_{m=1}^
{M} g\left( \frac{W^{*}_{m}x_{s}^{u}}{\sqrt{N}} \right) + \sigma \xi ^{u} -\sum_{k=1}^{K} g\left( \frac{W
_{k}x_{s}^{u}}{\sqrt{N}} 
\right) \right]  \cdot  \left( -g'\left( \frac{W_{i}x_{s}^{u}}{\sqrt{N}} \right)  \cdot  \frac{x_{s}^{u}}{\sqrt{N}} \right) \\
= & - \frac{1}{B}\sum_{s=1}^{B} \Delta_{s}^{u}  \cdot  g'\left( \frac{W_{i}x_{s}^{u}}{\sqrt{N}} \right) \cdot \frac{x_{s}^{u}}{\sqrt{N}}
\end{align} $$

with $\Delta_{s}^{u} = \sum_{m=1}^{M} g\left( \frac{W^{*}_{m}x_{s}^{u}}{\sqrt{N}} \right) + \sigma \xi^{u} - 
\sum_{k=1}^{K} g\left( \frac{W_{k}x_{s}^{u}}{\sqrt{N}} \right)$. 
Hence, he gradient descent update equations for the student network is

$$ W_{i}^{u+1} = W_{i}^{u} + \frac{\eta_{W}}{B}\sum_{s=1}^{B} \Delta_{s}^{u}  \cdot  g'\left( \frac{W_{i}x_
{s}^{u}}{\sqrt{N}} \right)  \cdot \frac{x_{s}^{u}}{\sqrt{N}}. $$

From this expression, we could take the gradient flow limit, however, the expectation induced in the right hand side
does not have a closed form solution. However, we can write the update equation in terms of the order parameters, for 
which this expectation has a solution. The order parameters fully define the state of the system, and is commonly 
understood in physics as a macroscopic variable that describe the time evolution of a complex system, while allowing 
for further mathematical analysis. These order parameters are the overlap between student and teacher neurons 
$R$, the overlap of students neurons with itself $Q$, and the overlap of teacher neurons with itself $T$ 
(which do not change throughout training as the teacher is fixed) which are defined as

$$ \begin{equation}
R = \frac{W^{*}W^{T}}{N}, \hspace
{0.2cm} 
Q = \frac{W W^{T}}{N} \hspace{0.2cm} \text{and} \hspace{0.2cm} T = \frac{W^{*}(W^{*})^{T}}{N}.
\end{equation} $$

Instead of describing the learning using the gradient descent updates for the weights, we can describe it in terms of
the order parameters. To do this, we simply multiply the gradient updates equation by $(W^{*}_{n})^{T}/N$ to obtain $R$ 
updates and by $(W_{j}^{u+1})^{T}/N$ to obtain $Q$ updates. Starting with the $R$ updates, we have

$$ \begin{align}
\frac{W_{i}^{u+1}(W_{n}^{*})^{T}}{N} & = \frac{W_{i}^{u}(W_{n}^{*})^{T}}{N} + \frac{\eta_{W}}{NB}\sum_{s=
1}^{B} \Delta_{s}^ {u}  \cdot  g'\left( \frac{W_{i}x_{s}^{u}}{\sqrt{N}} \right)  \cdot  
\frac{x_{s}^{u} (W_{n}^{*})^{T}}{\sqrt{N
}}, \\
R_{in}^{u+1} & = R_{in}^{u} + \frac{\eta_{W} dt}{B}\sum_{s=1}^{B} 
\Delta_{s}^
{u}  \cdot  g'\left( \frac{W_{i}x_{s}^{u}}{\sqrt{N}} \right)  \cdot  \frac{x_{s}^{u} (W_{n}^{*})^{T}}{\sqrt{N}}. 
\end{align} $$

From this equation, we defined $dt=1/N$, and by moving $R_{in}^{u}$ to the left hand side, dividing by $d
t$, and taking the *thermodynamic limit* $N \rightarrow \infty$, we obtain the time derivative of $R_{in}$ as

$$ \begin{equation}
\frac{d R_{in}}{d t} = \eta_{W} \left< \Delta_{s}^{u} g'(\lambda_{i}^{u}) \rho_{n}^{u} \right>
\end{equation} $$

where we define the *local fields* 

$$ \begin{equation}
\lambda_{i}^{u} = \frac{W_{i}x_{s}^{u}}{\sqrt{N}} \hspace{0.3cm} \text{and} \hspace{0.3cm} \rho_{n}^{u}
 = \frac{(W_
{n}^{*})^{T}x_{s}^{u}}{\sqrt{N}}.
\end{equation} $$

The equation for $\frac{dR_{in}}{dt}$ is now in a convenient form, where the local fields are simply a gaussian 
scalar as $x \sim \mathcal{N}(0, I)$, and the expectation because an integral over gaussian distribution 
with covariances defined by the order parameters. Before solving this expectation, let's derive the same equation for the 
order paramaters $Q$ (slightly trickier). We go back to the gradient descent update equation for the weights, and 
multiply by $(W_{j}^{u+1})^{T}/N$ and $(W_{j}^{u})^{T}/N$ giving

$$ \begin{align} 
\frac{W_{i}^{u+1}(W_{j}^{u+1})^{T}}{N} & = \frac{W_{i}^{u}(W_{j}^{u+1})^{T}}{N} + \frac{\eta_{W}}{NB}\sum
_{s=1}^{B} 
\Delta_{s}^{u}  \cdot  g'\left( \lambda_{i}^
{u} \right)  \cdot  \frac{x_{s}^{u}}{\sqrt{N}}(W_{j}^{u+1})^{T}, \\
Q^{u+1}_{ij} & = \frac{W_{i}^{u}}{N}\left( W_{j}^{u} + \frac{\eta_{W}}{B}\sum_{s=1}^{B} \Delta_{s}^{u} \cdot g'\left
( \frac{W_{j}x_{s}^{u}}{\sqrt{N}} \right)  \cdot  \frac{x_{s}^{u}}{\sqrt{N}} \right)^{T} \\
& + \frac{\eta_{W}}{NB}\sum_{s=1}^{B} \Delta_{s}^{u}  \cdot  g'\left( \lambda_{i}^ {u} \right)  \cdot  \frac{
x_{s}^{u}}
{\sqrt{N}} \left( W_{j}^{u} + \frac{\eta_{W}}{B}\sum_{s=1}^{B} \Delta_{s}^{u}  \cdot  g'\left
( \frac{W_{j}x_{s}^{u}}{\sqrt{N}} \right)  \cdot  \frac{x_{s}^{u}}{\sqrt{N}} \right)^{T}, \\
Q^{u+1}_{ij} & = Q^{u}_{ij} + \frac{\eta_{W}dt}{B}\sum_{s=1}^{B} \Delta_{s}^{u}  \cdot  g'\left( \lambda_{j
}^ {u} 
\right) \lambda_{i}^{u} + \frac{\eta_{W}dt}{B}\sum_{s=1}^{B} \Delta_{s}^{u}  \cdot  g'\left( \lambda_{i}^{u
} 
\right) \lambda_{j}^{u} \\
& + \frac{\eta_{W}^{2}dt}{B^{2}}\sum_{s=1}^{B}\sum_{s'=1}^{B} \Delta_{s}^{u} \Delta_{s'}^{u}g'\left( \lambda_{i}^{u}
\right)g'\left( \lambda_{j}^{u} \right) \frac{x_{s}^{u}(x_{s}^{u})^{T}}{N}.
\end{align} $$

Now dividing by $dt$ and taking the limit $N \rightarrow \infty$, (hence $dt \rightarrow 0$), $\frac{x_{s
}^{u}(x_{s}
^{y})^{T}}{N} \rightarrow 1 $ by the central limit theorem, and expectations over $s$ and $s'$ are $0$ as
 they are 
independent samples, we obtain the time 
derivative of $Q_{ij}$ as

$$ \begin{equation}
\frac{dQ_{ij}}{dt} = \eta_{W} \left< \Delta_{s}^{u} g'(\lambda_{j}^{u}) \lambda_{i}^{u} \right> + \eta_{W
} \left<\Delta_{s}^{u} g'(\lambda_{i}^{u}) \lambda_{j}^{u} \right> + \eta_{W}^{2}
\left< (\Delta_{s}^{u})^{2} g'(\lambda_{i}^{u}) g'(\lambda_{j}^{u}) \right>.
\end{equation} $$

Finally, having the order parameters, we can write the generalization error (the expected mean squared error between
teacher and student) as

$$ \begin{align}
\mathcal{E} = & \frac{1}{2} \left< \left( \hat{y} - y \right)^{2} \right> \\
=  & \frac{1}{2} \sum_
{i=1}^{K} \sum_{j=1}^{K} \left< g(\lambda_{i}^{u}) g(\lambda_{j}^{u}) \right> - \sum_{m=1}^{M} \sum_{i=1}^{K} \left< 
g(\rho_{m}^{u}) g(\lambda_{i}^{u}) \right> + \frac{1}{2} \sum_{m=1}^{M} \sum_{n=1}^{M} \left< g(\rho_{m}^{u}) g(\rho_
{n}^{u}) \right>  + \frac{\sigma^{2}}{2}
\end{align} $$

Here, we encounter the first expectation that can be integrated in closed form for $g$ defined as the 
error function. Now, we introduce the useful expectations that will appear in the computation of expectations in the
generalization error and order parameters:

$$ \begin{align}
I_{2}(a, b) & = \left< g(\nu_{a}) g(\phi_{b}) \right>, \\ 
I_{3}(a, b, c) & = \left< g'(\nu_{a}) \phi_{b} g(\psi_{c}) \right>, \\ 
I_{4}(a, b, c, d) & = \left< g'(\nu_{a}) g'(\phi_{b}) g(\psi_{c}) g(\gamma_{d}) \right> \\
J_{2}(a, b) & = \left< g'(\nu_{a}) g'(\phi_{b}) \right>.
\end{align} $$

These expectations can be solved in closed form as a function of the covariance between each of the variables $\nu_
{a}, \phi_{b}, \psi_{c}$ and $\gamma_{d}$. Let's start with an example from the terms in the generalization error. 
First, closed form expression for $I_{2}$, with $g$ as the error function, is

$$ \begin{align}
I_{2}(a, b) = & \frac{2}{\pi} \text{arcsin}\left( \frac{C_{ab}}{\sqrt{1 + C_{aa}} \sqrt{1+C_{bb}}} \right) 
\end{align} $$

where $C_{ab}$ is the covariance between $\nu_{a}$ and $\phi_{b}$, and $C_{aa}$ and $C_{bb}$ are the variances of
$\nu_{a}$ and $\phi_{b}$, respectively. The way to select the correct covariance structure, is to look at the 
arguments of the expectation. Recall the index notation, $i, j, k$ for the student, and $m, n, p$ for the teacher, 
then $a$ and $b$ can be any of these indices depending on the corresponding local field. For instance, if $a=k$, the 
notation implies that $\nu = \lambda$, if $b=m$, then $\phi = \rho$. From here, we can write the generalization 
error in terms of this integral

$$ \begin{align}
\mathcal{E} = & \frac{1}{2} \sum_{i=1}^{K} \sum_{j=1}^{K} I_{2}(i, j) - \sum_{n=1}^{M} \sum_{i=1}^{K} I_{2}(i, n)+ \frac{1}{2} \sum_{m=1}^{M} \sum_{n=1}^{M} I_{2}(n, m) + \frac{\sigma^{2}}{2}
\end{align}. $$

The covariances between the local fields are defined by the order parameters. For instance, the covariance for $I_{2}
(i, n)$ (student-teacher indexes) is $C_{12} = R_{i, n}$, $C_{11}=\text{diag}(Q)_{i}$ and $C_{22}=\text{diag}(T)_{n}
$, or the covariance for $I_{2}(i, j)$ is $C_{12}=Q_{ij}$, $C_{11}=\text{diag}(Q)_{i}$ and $C_{22}=\text{diag}(Q)_{j}$.
Hence, we can take advantage of broadcasting to compute all elements of $I_{2}$ matrix as a function of the
corresponding order parameters matrices:
```python
def get_I2(c12, c11, c22):
    e_c11 = jnp.expand_dims(jnp.diag(c11), axis=1)
    e_c22 = jnp.expand_dims(jnp.diag(c22), axis=0)
    denom = jnp.sqrt((1 + e_c11) * (1 + e_c22))
    return jnp.arcsin(c12 / denom) * (2 / jnp.pi)
```
and the generalization error being
```python
def order_parameter_loss(Q, R, T, sigma):
    # Student overlaps
    I2_1 = get_I2(Q, Q, Q)
    first_term = jnp.sum(I2_1)

    # Student teacher overlaps
    I2_2 = get_I2(R, Q, T)
    second_term = jnp.sum(I2_2)

    # Teacher overlaps
    I2_3 = get_I2(T, T, T)
    third_term = jnp.sum(I2_3)
    
    return first_term/2 - second_term + third_term/2 + sigma**2/2
```
$I_{3}$, $I_{4}$ and $J_{2}$ follow a similar structure, with the covariance structure defined by the order parameters 
between the arguments of the expectation. These integrals appear in the order parameters dynamical equation by 
expanding the error signal $\Delta_{s}^{u}$, giving

$$ \begin{align}
\frac{dR_{in}}{dt} & = \eta_{W} \left[ \sum_{m=1}^{M} I_{3}(i,n,m) - \sum_{j=1}^{K} I_{3}(i, n, j) \right] \\
\frac{dQ_{ik}}{dt} & = \eta_{W} \left[ \sum_{m=1}^{M} I_{3}(i,k,m) - \sum_{j=1}^{K} I_{3}(i, k, j) \right] \\
& + \eta_{W} \left[ \sum_{m=1}^{M} I_{3}(k, i, m) - \sum_{j=1}^{K} I_{3}(k, i, j) \right] \\
& + \eta_{W}^{2} \left[ \sum_{m, n}^{M} I_{4}(i, k, n, m) - 2 \sum_{j, n} I_{4}(i, k, j, n) + \sum_{j, l} I_{4}(i, k,
j, l) + \sigma^{2} J_{2}(i, j) 
\right]. 
\end{align} $$
 














Fig1: Saad and Solla results.

### The specialization transition tracks feature learning

Fig2: Widget (Saad and Solla from Fig 1) Visualizing order parameters. According to the phase transition in the loss there is a corresponding change in the order parameters, show this by adding a slide that encodes time.

### Large initial weights produce individual differences

Fig3: Large weights, varying simulations and average out to show correspondence with ODEs. Show for M=4 and K=2, 4 and 6 (Sebastian Goldt's)

### Theory-experiment overlap in two-layer neural networks

Fig4: (decide later, perhaps small weights).
Goldt setting?
- <d-cite key="goldt2020dynamics"></d-cite>

### Limits of the analytical teacher-student setting

Check the bounds for learning rate and noise such that the ODEs follow the simulations.

## Impact & extensions

The analytical techniques pioneered by Saad & Solla and others have inspired two broad directions of research: extending the theoretical framework to handle more complex scenarios, and applying these tools to analyze specific phenomena in machine learning. 

### The analytical frontier

The statistical physics approach to neural network dynamics has expanded significantly beyond the early results of Gardner and others.

Early extensions explored different activation functions, with Freeman and Saad analyzing radial basis function networks <d-cite key="freeman1997online"></d-cite>
Richert et al. studied the qualitative convergence for these dynamical systems <d-cite key="richert2022soft"></d-cite>.
Deep networks were analyzed by Tian et al., who first provided empirical evidence for specialization in deep teacher-student networks <d-cite key="tian2019luck"></d-cite>, then developed theoretical characterization of these dynamics <d-cite key="tian2020student"></d-cite>. 
This work illuminated how hierarchical feature learning emerges in deeper architectures.

The Gaussian equivalence property <d-cite key="goldt2020modelling"></d-cite> <d-cite key="goldt2021gaussian"></d-cite>, showed that many results derived for Gaussian inputs extend to other data distributions, broadening the applicability of these analytical techniques.

Recent work has tackled increasingly complex learning scenarios. Loureiro et al. <d-cite key="loureiro2021learning"></d-cite> and Arnaboldi et al. <d-cite key="arnaboldi2021highdimensional"></d-cite> extended the framework to new learning settings, while Bardone et al. analyzed systems with correlated latent variables <d-cite key="bardone2024sliding"></d-cite>. 
Questions of learnability have been addressed by Troiani et al. <d-cite key="troiani2024fundamental"></d-cite>, who established theoretical limits on what neural networks can learn in various settings. 

These advances advance the statistical physics approach from a specialized tool for simple networks into a more comprehensive framework for analyzing modern machine learning systems. 

### Applications of the teacher-student setting

Beyond charactierizing generalization error, the teacher-student framework has been applied to a wide range of problems, often to model interesting phenomena in machine learning. 
In optimization, Yang analyzed natural gradient descent <d-cite key="yang1998complexity"></d-cite>.
Simsek and Martinelli used the framework to reduce overparameterized deep networks to a minimal size by exploiting student neurons with overlapping tuning to teacher neurons <d-cite key="simsek2021geometry"></d-cite> <d-cite key="martinelli2023expand"></d-cite>.
Further, Arnaboldi et al. developed quantitative measures of task difficulty <d-cite key="arnaboldi2024online"></d-cite>.

The framework has been used extensively to study continual learning.
Several analyses examine catastrophic forgetting <d-cite key="straat2018statistical"></d-cite> <d-cite key="lee2021continual"></d-cite> <d-cite key="asanuma2021statistical"></d-cite>, transfer learning limitations <d-cite key="lee2022maslow"></d-cite>, meta-learning <d-cite key="wang2024dynamics"></d-cite>, and factors affecting continual learning <d-cite key="hiratani2024disentangling"></d-cite>.

Applications extend to learning algorithms including multi-pass SGD <d-cite key="arnaboldi2024repetita"></d-cite>, feedback alignment <d-cite key="refinetti2022align"></d-cite>, and reinforcement learning <d-cite key="patel2023rl"></d-cite>.
Current work under review at the ICLR conference <d-cite key="anonymous2024analyzing"></d-cite> <d-cite key="anonymous2024optimal"></d-cite> <d-cite key="anonymous2024theory"></d-cite> applies this framework to additional machine learning problems.




---

[TODO: Anonymize when submitting!]: # 

**Acknowledgements.** 
We thank 
[Stefano Sarao Mannelli](https://stefsmlab.github.io/) for historical perspectives,
[Jin Hwa Lee](https://jinhl9.github.io/) for starter code for the analytical ODEs in Numpy,
and [Nishil Patel](https://jinhl9.github.io/) for invaluable help in narrowing down details of the implementation.
