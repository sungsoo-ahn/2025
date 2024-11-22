---
layout: distill
title: Perpetual Text Generation Using Dynamic Temperature Adjustment
description: Large Language Models have demonstrated proficiency across all fields, and has seen huge advancements in the last couple years. However, they often falter when generating extended outputs. This limitation restricts their effectiveness in application requiring sustained text production. We introduce the concept of perpetuity, defining a model's ability to continue generating new tokens indefinitely. To overcome the challenge of early termination, we propose a novel method called Perpetual Text Generation using Dynamic Temperature Adjustment.
date: 2025-11-22
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
bibliography: 2025-11-22-perpetual-text.bib  

# Add a table of contents to your post.
#   - make sure that TOC names match the actual section names
#     for hyperlinks within the post to work correctly. 
#   - please use this format rather than manually creating a markdown table of contents.
toc:
  - name: Introduction
  - name: When do Models stop Generating New Tokens?
  - name: Motivating Experiments
    subsections:
    - name: Block-wise Analysis
    - name: Token-wise Analysis
  - name: Proposed Methods
    subsections:
    - name: Suppressing the EOS 
    - name: Modified Sampling Method Post-EOS
    - name: Regnerating Prior to EOS
    - name: Regenerating and Resampling Prior to EOS with Dynamic Temperature Adjustment
  - name: Conclusion and Looking Forward

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

# Introduction

The AI revolution taking place in the last half decade has been headlined by Large Language Models (LLMs). These transformer models have demonstrated proficiency in a variety of tasks across all fields, and many are adopting LLMs into their regular workflow. Such examples include extended literature reviews, research papers and reports, and even the recent GPT-o1 model, which requires long output for its chain-of-thought reasoning.

The unique ability of these models to quickly follow and execute instructions makes it an important and effective tool. However, with rising demand comes rising expectations of model abilities.

In the last couple of years alone, there have been drastic improvements in LLMs including context capacity, speed, and logical reasoning. One of the most notable is the ability to input long contexts to models beyond the pretraining sequence length.

But what about the *ability to produce long outputs*? We’ve improved language models with the freedom to listen and understand, but we have yet to *provide them the ability to respond with outputs with equal and/or longer lengths*.

We define a large language model’s ability to respond to a prompt as **perpetuity**. In technical terms, perpetuity is the ability of a model to continue generating new tokens. I The current token generation process is sometimes misleading by the nonsensical nature of “\n \n \n” data. We found this led to the generation of a halt <EOS> token, which stops output generation. 

To combat decreasing entropy leading to the generation of the EOS token, we propose a new method: **Perpetual Text Generation using Dynamic Temperature Adjustment**. By dynamically increasing temperature, we provide the model with an opportunity to produce alternative continuations – therefore prolonging output. 

# When do Models Stop Generating New Tokens?

In transformer-based language models, particularly decoder-only architectures used for text generation, the process of generating tokens continues sequentially until a stopping criterion is met. 

There are a couple of ways that a model can do this:
### Setting a max_token parameter
By setting a maximum number of tokens, you can override the model's default tendencies and encourage it to produce longer outputs. However, simply increasing the token limit doesn't guarantee that the model will generate more extended content, as it may still generate the EOS token prematurely due to its learned patterns.

### Hitting a stopping criterion <EOS> token
- If no max_token parameter is set, then the model continues generating tokens until it outputs the <EOS> token, indicating that it should stop.
- Training with <EOS> Tokens: During training, models are exposed to sequences that conclude with a special end-of-sequence token, often denoted as <EOS> (ref: llama 2 paper) or <|eot_id|> (ref: llama 3 paper). This token signifies the completion of a coherent piece of text.
- Learned Termination Behavior: Through exposure to these tokens, the model learns to predict an <EOS> token when it determines that a logical conclusion has been reached in the context of the generated text.

### Probability Distribution and Sampling Methods:
- Next Token Prediction: At each generation step, the model computes a probability distribution over the vocabulary for the next token, conditioned on all previously generated tokens.
- Likelihood of <EOS> Token: The probability assigned to the <EOS> token depends on the context. If the model assesses that the generated content forms a complete thought or sentence, it increases the probability of selecting <EOS>.
- Sampling Strategies: The choice of sampling method (e.g., greedy search, beam search, top-k sampling, nucleus sampling) can affect when the <EOS> token is selected. Some methods may favor high-probability tokens, potentially leading to earlier termination.

When no explicit maximum token limit is set, the stopping point relies heavily on the model's ability to predict when a piece of text should logically end. As previous work (Xiong et al., 2024; Fu et al., 2024) has shown models often “cap” their output lengths based on the upper limits present in their SFT datasets, even if they were exposed to longer sequences during pre-training. The model's training data plays a crucial role here; if it includes diverse examples of text lengths and proper use of <EOS> tokens in the retraining process, the model is better equipped to determine appropriate stopping points.

# Motivating Experiments
To further explore how these factors influence the model's stopping behavior, we conducted a series of experiments to identify trends that may be causing the convergence to the <EOS> token. 

Our first experiment examines the variance and uncertainty in the “blocks” leading up to the EOS token generation. Since the probability of a token can exhibit significant variance, a block-wise trend is analyzed instead of a token-wise trend. A block is defined as a group of consecutive tokens within a sequence. The average probability of a specific token within each block is then calculated to represent its probability in that block. 

Our second experiment examines the tokens leading up to the <EOS> token. 

### Block-Wise Analysis
Observation: All four metrics show an increasing probability of EOS token, decreasing information content, decreasing entropy, and decreasing varentropy. 

[image goes here]

### Token-Wise Analysis

[image goes here]

