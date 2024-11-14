---
layout: distill
title: Analytical and simulated learning dynamics of simple neural networks
description:
  We survey approaches to analyzing the learning dynamics of simple neural networks,
  and provide parallel implementations of these methods in JAX.
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
        - name: "Modern applications"
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
- name: Discussion
  subsections:
    - name: Limitations of the analytical teacher-student setting
    - name: Further applications of the teacher-student setting
---

The learning dynamics of neural networks---how parameters change during 
training---describe how data, architecture and algorithm interact to produce 
a trained model.
Characterizing these dynamics in general is an open problem in machine learning,
but recent work has made progress in understanding the dynamics of certain settings 
both neural networks.


- place the framework in the broader context of DL theory
- describe the (dis)advantages of this approach to other DL theory approaches:
  - assumptions on data distribution
  - asymptotic in the input data dimension

## Background

### Empirical studies of neural network dynamics 

### Theoretical analyses of neural network dynamics 

### Statistical physics for neural network dynamics 

The common shared assumption is distributional the data.

#### A long history 

In 2024, X and Y won the Nobel Prize in Physics for neural networks.
news that was met with varying emotions: celebration, confusion and skepticism across the scientific communities with an on artificial intelligence.


#### Our focus: Narrow neural networks

temporally describe the history: S&S, Goldt, etc. including HMM
Highlight Solla and Goldt paper, and a lot form Lenka's lab, (Hugo Cui), Cengiz lab (Blake's work). Teacher student is a framework that people use to derive analytical results for neural network dynamics, we are focusing in one of them (Solla and Goldt's work).

#### Modern applications

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

## Discussion

### Limitations of the analytical teacher-student setting

Check the bounds for learning rate and noise such that the ODEs follow the simulations.

### Further applications of the teacher-student setting

Task similarity.

---

[TODO: Anonymize when submitting!]: # 

**Acknowledgements.** 
We thank 
[Stefano Sarao Mannelli](https://stefsmlab.github.io/)
for historical perspectives
and 
[Jin Hwa Lee](https://jinhl9.github.io/)
for invaluable help in narrowing down details of the implementation.
