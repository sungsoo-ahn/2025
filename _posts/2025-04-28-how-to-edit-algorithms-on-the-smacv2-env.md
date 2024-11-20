---
layout: distill
title: How to edit algorithms for the SMACv2 environment
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
bibliography: 2025-04-28-distill-example.bib  

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
---

Hi everyone,
It took us a while to understand how the [pymarl2 repo](https://github.com/benellis3/pymarl2) works and how to make any changes to the algorithms. We thought we’d write this blog to help y’all understand it better and perhaps save you some time.

MARL is complex enough as it is. Firstly, the algorithms are complex to understand and then matching the algorithm to existing codebase is a big challenge of its own! Don’t we wish that everyone would write better README files in their repositories! We’re writing this blog to reduce the entering barrier to implementing your algorithm to complex environments like smacv2! The repository explained in this blog is the repository that has implemented some MARL algorithms to the smacv2 environment.

The aim of this blog is to help you find where specific parts of an  algorithm are located so that you can edit/omit those parts for your new algorithm.

Note: If there are any errors in this explanation or if you think this blog would benefit from adding more details, please feel free to reach out to us! Moreover, as the repo is updated, some of the information in this blog may not be valid.

## Main components of any algorithm:
1. Agent network
- These are found in src -> modules -> agents
- First refer to src -> modules -> agents -> \__init__.py. Here, all the different types of agent networks are 'registered'.
2. Mixer network
- There are found in src -> modules -> mixers
3. Hyperparameters
- There are found in src -> config -> algs
- For any algorithm, you must first refer to default.yaml. Then you must refer to the [algorithm].yaml file (e.g., qplex.yaml) where some parameters/hyperparameters would be added/overwritten.
4. The algorithm
- There are found in src -> learners
- Refer to src -> learners -> \__init__.py to see how all the algorithm learners are 'registered'. The names of the files here may not be intuitive, so you can refer to the yaml file of that specific algorithm (found in src->config->algs) to see which learner file it is using.



## QPLEX Algorithm

Now let's take an example algorithm and see where the parts of it are located.

