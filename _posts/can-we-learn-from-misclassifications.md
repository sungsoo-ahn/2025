---
layout: distill
title: Can We Learn From Misclassifications
description: Evaluation of the performace in supervised learning based on common methods, such as the final activation map or the final represantations, is not sparse. We introduce and a novel evaluation based on the gradients of the model. We establish a robust connection
date: 2025-04-28
future: true
htmlwidgets: true
hidden: false

# Anonymize when submitting
# authors:
#   - name: Anonymous

authors:
  - name: Albert Einstein
    url: "https://en.wikipedia.org/wiki/Albert_Einstein"
    affiliations:
      name: IAS, Princeton
  - name: Boris Podolsky
    url: "https://en.wikipedia.org/wiki/Boris_Podolsky"
    affiliations:
      name: IAS, Princeton
  - name: Nathan Rosen
    url: "https://en.wikipedia.org/wiki/Nathan_Rosen"
    affiliations:
      name: IAS, Princeton

# must be the exact same name as your blogpost
bibliography: can-we-learn-from-misclassifications.bib  

# Add a table of contents to your post.
#   - make sure that TOC names match the actual section names
#     for hyperlinks within the post to work correctly. 
#   - please use this format rather than manually creating a markdown table of contents.
toc:
  - name: Expirmantal Results
  - name: Figures
   

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



## Expirmantal Results 

| Model\Matrices | 'Cos_Sim_G'   | 'Cos_Sim_R'   | 'Cos_Sim_act' | 'Grad_Cam_CC' | 'Grad_Cam_KLD'|'Grad_Cam_SSIM'| 'Grad_Cam_AUC'|
| -------------- |:-------------:|:-------------:|:-------------:|:-------------:|:-------------:|:-------------:|:-------------:|
| Resnet18       | 0.832475      | 0.750405      | 0.809194      | 0.514107      | 0.546371      | 0.640893      | 0.473252      |
| Resnet34       | 0.891979      | 0.760314      | 0.821133      | 0.547336      | 0.303619      | 0.717424      | 0.482986      |
| Resnet50       | 0.970702      | 0.781508      | 0.842356      | 0.387207      | 0.845985      | 0.534813      | 0.441505      |
| Resnet101      | 0.898816      | 0.777873      | 0.841248      | 0.451866      | 0.607102      | 0.587721      | 0.471974      |
| Resnet152      | 0.918961      | 0.784820      | 0.834101      | 0.513276      | 0.560512      | 0.600739      | 0.498934      |

Where the columns of the table represent: 

-**Cos_Sim_G**: Cosin similarity based on the gradiant
-**Cos_Sim_R**: Cosin similarity based on the final represantations
-**Cos_Sim_act**: Cosin similarity based on the activation map
-**Grad_Cam_CC**: Grad_Cam accuracy
-**Grad_Cam_KLD**: Grad_Cam Kullback–Leibler divergence
-**Grad_Cam_SSIM**: Grad_Cam Structural similarity index measure
-**Grad_Cam_AUC**: Grad_Cam area under the curve

## Figures

