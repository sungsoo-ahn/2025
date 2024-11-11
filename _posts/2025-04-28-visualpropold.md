---
layout: distill
title: Sample Blog Post
description: Your blog post's abstract.
  Please add your abstract or summary here and not in the main body of your text. 
  Do not include math/latex or hyperlinks.
date: 2025-04-28
future: true
htmlwidgets: true
hidden: false

# Anonymize when submitting
 authors:
   - name: Anonymous

#authors:
#  - name: Albert Einstein
#    url: "https://en.wikipedia.org/wiki/Albert_Einstein"
#    affiliations:
#      name: IAS, Princeton
#  - name: Boris Podolsky
#    url: "https://en.wikipedia.org/wiki/Boris_Podolsky"
#    affiliations:
#      name: IAS, Princeton
#  - name: Nathan Rosen
#    url: "https://en.wikipedia.org/wiki/Nathan_Rosen"
#    affiliations:
#      name: IAS, Princeton

# must be the exact same name as your blogpost
bibliography: 2025-04-28-models_vision_toddlers.bib  

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

** Introduction

Look at toddlers around: they play with/observe almost always the same objects, as they remain almost always stuck in the same caregivers and playground/home/daycare environments. They likely have not seen 10\% of the different dogs of ImageNet and likely not a single of the 1000 sea snakes of ImageNet-1k. Thus, toddlers experience a diversity of objects which is several orders of magnitude lower than ImageNet-1k. Unlike current machine learning (ML) vision models, they also do not access a massive amount category labels (or aligned language utterances), adversarial samples and do not massively color-transform or mask their visual inputs. Yet, they develop strong semantic representations that are robust to image distortions, viewpoints, machine-adversarial samples and different styles (draw, silhouettes...) \cite{wichmann2023deep,huber2023developmental}.

How do toddlers develop such a robust visual system ? The developmental period between 1 and 3 years old seem particularly critical and interesting, as psychophysics experiments demonstrated that toddlers progressively reach fundamental visual milestones and develop visual biases that shape their object recognition abilities over the rest of their life. Specifically, during their second year toddler manage the category of maximally simplified shapes and learn to attend to the shapes of novel objects when judging their similarity based on their ``kind". Simultaneously, they also increasingly focus on object's views that highlight their main axis of elongation, and later, learn to associate objects based the the configural relation of their parts. All these developments likely shape how their strong ability to recognize objects, raising the question of whether ML models also exhibit these properties.%this would clarify what aspect of visual learning is missing in ML models, what aspect of ML models reasonably well model humans' perceptual similarity \cite{wichmann2023deep}, 

In this blogpost, we investigate the presence of the four above-mentioned visual properties in a set of  diverse pre-trained models. Our objective is to 1) clarify the interplay between different visual properties in ML; 2) reproduce and modify previous experimental protocols comparing ML to humans to (in)validate prior claims; 3) provide a global picture of the (dis)similarities between the visual properties of ML models and toddlers, thereby giving perspectives of research for improving ML models. We will also publicly release our github code upon acceptance as an easy-to-use toolbox. 

#{% include figure.html path="assets/img/2025-04-28-models_vision_toddlers/overview.png" class="img-fluid rounded z-depth-1" %}
