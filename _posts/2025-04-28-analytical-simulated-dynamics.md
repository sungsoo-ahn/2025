---
layout: distill
title: Analytical learning dynamics of simple neural networks
description:
    The learning dynamics of neural networks—in particular, how parameters change over time during 
    training—describe how data, architecture and algorithm interact in time to produce 
    a trained neural network model.
    Characterizing these dynamics in general remains an open problem in machine learning,
    but, handily, restricting the setting allows careful empirical studies and even analytical results.
    In this blog post, we introduce approaches to analyzing the learning dynamics of 
    neural networks,
    with a focus on a particular setting that permits an explicit analytical 
    expression for the generalization error of a neural network trained with online gradient descent.
    We provide an accessible mathematical formulation of this analysis
     alongside a `JAX` codebase to implement both simulation of the analytical
    system of ordinary differential equations and neural network training in this setting.
    We conclude with discussion of how this analytical paradigm has been used to investigate generalization in neural networks and beyond.
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
    - name: Empirical studies of neural network dynamics 
    - name: Theoretical analyses of neural network dynamics 
    - name: Statistical physics for neural network dynamics 
      subsections:
        - name: A long history 
        - name: "Our focus: Narrow neural networks"
          subsections:
            - name: "The classical analysis"
            - name: "Modern extensions of the analysis"
- name: "Notation"
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
- name: Discussion
  subsections:
    - name: Applications of the teacher-student setting
    - name: The analytical frontier
---




- place the framework in the broader context of DL theory

## Background

### Empirical studies of neural network dynamics 

### Theoretical analyses of neural network dynamics 

### Statistical physics for neural network dynamics 

The common shared assumption is distributional the data.

#### A long history 

In 2024, X and Y won the Nobel Prize in Physics for neural networks.
news that was met with varying emotions: celebration, confusion and skepticism across the scientific communities with an interest in artificial intelligence.

- <d-cite key="mei2018mean"></d-cite>

#### Our focus: Narrow neural networks in the teacher-student setting

- teacher-student intro <d-cite key="gardner1989three"></d-cite>
- overview <d-cite key="cui2024highdimensional"></d-cite>
- analytical results <d-cite key="saad1995online"></d-cite> <d-cite key="riegler1995online"></d-cite>

temporally describe the history: S&S, Goldt, etc. including HMM
Highlight Solla and Goldt paper, and a lot form Lenka's lab, (Hugo Cui), Cengiz lab (Blake's work). Teacher student is a framework that people use to derive analytical results for neural network dynamics, we are focusing in one of them (Solla and Goldt's work).

- describe the (dis)advantages of this approach to other DL theory approaches:
  - assumptions on data distribution
  - asymptotic in the input data dimension

#### Modern applications of teacher-student

Andrew, Lenka, Cenghiz, that meta-learning paper
Features are fate?
Check Blake's work

## Notation



## Methods

In this section, we describe the teacher-student neural network setup. Next, we present the mathematical framework 
to characterize the gradient descent learning dynamics of the student network in terms of its order parameters. To 
complement the simulations  and theoretical framework, we include code snippets for computing each component.

Each element of the teacher-student setup is implemented  as an object, with the underlying computations efficiently 
coded in JAX for clarity and speed. The code adheres closely to the notation used in the equations,  leveraging 
broadcasting to accelerate computations and enabling JIT compilation for further optimization

### The teacher-student setup

In the teacher-student setting, the data generation process of $(x, y)$ pairs used to train the student network is 
described by a teacher network. We focus on the online learning setting, where new samples are drawn from the 
data generation process. In this setting, as new samples are drawn, the batch size has no effect on the dynamics, 
except to reduce the noise in the gradient estimate. We sample multiple $(x, y)$ pairs to fill a batch 
and update the student network using gradient descent iterations, indexed by $u$.

A teacher can be defined in a general way by 
$$\begin{equation} 
y_{i}^{u} = f^{*}(x_{i}^{u}, W^{*}) + \sigma \xi^{u}, 
\end{equation} $$ 
where $f^{*}(\cdot, W^{*})$ is the mapping defined by the teacher network with parameters $W^{*}$, $\xi^{u} \sim \mathcal{N}(0, 1)$, and 
$\sigma$ scales the output noise. The student network is generally defined as 
$$ \hat{y}_{i}^{u} = f(x_{i}^{u}, W), $$ 
where $f(\cdot, W)$ is the mapping generated by the student, and $W$ are the parameters of the student network. 
In general, $f$ and $f^{*}$ are the same (e.g., a neural network), where the parameters are the weights of the network.

To train the student network, we use gradient descent to minimize the mean squared error function at iteration $u$:
$$ \mathcal{L}^{u} = \frac{1}{2B} \sum_{i=1}^{B} \left( y_{i}^{u} - \hat{y}_{i}^{u} \right)^{2}, $$
where samples to fill a batch consist of $(x_{i}, y_{i})^{u}$ pairs with $i = 1, \ldots, B$, 
$x \sim \mathcal{N}(0, I)$ of dimension $N$, and the target $y_{i}^{u}$ is generated by feeding $x_{i}^{u}$ to the 
teacher network. 
We then update the student network's weights to minimize the loss function $\mathcal{L}^{u}$, effectively learning 
the same mapping as the teacher. The weights of the student network are updated using gradient descent as follows:
$$ W^{u+1} = W^{u} - \alpha \frac{\partial \mathcal{L}^{u}}{\partial W}, $$
where $\alpha$ is the learning rate. 

**How is the teacher-student setting useful here?** One way to analyze learning 
dynamics in neural networks is to pose the learning network as a dynamical system, where the gradient descent updates 
effectively evolve through continuous time as the parameters of a dynamical system. This transformation is 
commonly known as the *gradient flow limit*, where the discrete gradient descent updates become continuous when 
the learning rate is small, giving
$$ \frac{dW}{dt} = - \left\langle \frac{\partial \mathcal{L}^{u}}{\partial W} \right\rangle_{x,y}, $$
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
In Saad and Solla's work, both the teacher and student networks were modeled as a soft-committee machine, 
which is an average of non-linear perceptrons. We define the teacher and student networks as follows 
(with a slightly modified version):

$$ 
y_{i}^{u} = \sum_{k=1}^{M} g\left( \frac{W^{*}_{k} x_{i}^{u}}{\sqrt{N}} \right) \hspace{0.3cm} \text{ and } \hspace{0.3cm} 
\hat{y}_{i}^{u} = \sum_{k=1}^{K} g\left( \frac{W_{k} x_{i}^{u}}{\sqrt{N}} \right),
$$

where $g(\cdot)$ is the error function, $k$ indexes the perceptrons (rows of $W$), and $M$ and $K$ are the number of 
neurons in the teacher and student networks, respectively. 

To train the student, we minimize the mean squared error between the teacher and student outputs. 
We then perform gradient descent to update the student's weights.

$$ \begin{align} 
\mathcal{L^{u}} = & \frac{1}{2B} \sum_{i=1}^{B} \left(y_{i}^{u} - \hat{y}_{i}^{u} \right)^{2} \\
 = & \frac{1}{2B} \sum_{i=1}^{B} \left( \sum_{k=1}^{M} g\left( \frac{W^{*}_{k}x_{i}^{u}}{\sqrt{N}} \right) - \sum_{k=1}^{K} g\left( \frac{W_{k}x_{i}^{u}}{\sqrt{N}} \right) \right)^{2} \\
= & sadasd \\
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

## Discussion

### The analytical frontier

Other activation functions:
- RBF <d-cite key="freeman1997online"></d-cite>

Convergence:
- <d-cite key="richert2022soft"></d-cite>

Deep:
- deep teacher-student, empirics <d-cite key="tian2019luck"></d-cite>
- deep teacher-student, theory <d-cite key="tian2020student"></d-cite>

Universality. Gaussian equivalence property.
- <d-cite key="goldt2020modelling"></d-cite>
- <d-cite key="goldt2021gaussian"></d-cite>

Other settings:
- <d-cite key="loureiro2021learning"></d-cite>
- <d-cite key="arnaboldi2023highdimensional"></d-cite>
- correlated latents <d-cite key="bardone2024sliding"></d-cite>

Learnability:
- <d-cite key="troiani2024fundamental"></d-cite>

### Applications of the teacher-student setting

TODO:
-

**Optimization.**
- natural gradient <d-cite key="yang1998complexity"></d-cite>

**Parameterized task difficulty**.
- task difficulty <d-cite key="arnaboldi2024online"></d-cite>

**Parameterized task similarity for transfer and continual learning.**  
- continual learning <d-cite key="straat2018statistical"></d-cite>
- continual learning <d-cite key="lee2021continual"></d-cite>
- catastrophic forgetting <d-cite key="asanuma2021statistical"></d-cite>
- Maslow's hammer <d-cite key="lee2022maslow"></d-cite>
- meta-learning <d-cite key="wang2024dynamics"></d-cite>
- continual learning <d-cite key="hiratani2024disentangling"></d-cite>

**Reinforcement learning.**
- reinforcement learning <d-cite key="patel2023rl"></d-cite>

**Learning algorithms.**
- multi-pass SGD <d-cite key="arnaboldi2024repetita"></d-cite>
- feedback alignment <d-cite key="refinetti2022align"></d-cite>

Even under review at ICLR 2025!
- <d-cite key="anonymous2024analyzing"></d-cite>
- <d-cite key="anonymous2024optimal"></d-cite>
- <d-cite key="anonymous2024theory"></d-cite>



---

[TODO: Anonymize when submitting!]: # 

**Acknowledgements.** 
We thank 
[Stefano Sarao Mannelli](https://stefsmlab.github.io/)
for historical perspectives
and 
[Jin Hwa Lee](https://jinhl9.github.io/)
for invaluable help in narrowing down details of the implementation.
