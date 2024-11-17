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

We provide a JAX codebase to implement both simulation of the system of ordinary differential equations and neural network training in the teacher-student setting.

### Simulating the ordinary differential equations

### Simulating neural network training

## Replications

### Theory-experiment overlap in the soft committee machine

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
