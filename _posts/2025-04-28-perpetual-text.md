---
layout: distill
title: Perpetual Text Generation Using Dynamic Temperature Adjustment
description: Large Language Models have demonstrated proficiency across all fields, and has seen huge advancements in the last couple years. However, they often falter when generating extended outputs. This limitation restricts their effectiveness in application requiring sustained text production. We introduce the concept of perpetuity, defining a model's ability to continue generating new tokens indefinitely. To overcome the challenge of early termination, we propose a novel method called Perpetual Text Generation using Dynamic Temperature Adjustment.
date: 2025-04-28
future: true
htmlwidgets: true
hidden: false

# Anonymize when submitting
authors:
  - name: Anonymous

# must be the exact same name as your blogpost
bibliography: 2025-04-28-perpetual-text.bib  

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

We define a large language model’s ability to respond to a prompt as **perpetuity**. In technical terms, perpetuity is the ability of a model to continue generating new tokens. The current token generation process is sometimes misleading by the nonsensical nature of “\n \n \n” data. We found this led to the generation of a halt <EOS> token, which stops output generation. 

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

<div align="center">
  <img align="center" src="./assets/img/2025-04-28-perpetual-text/eos_token_stats_blockwise_100.png" width="800px;" alt=""/>
  <br>
  Figure 1: Block-Wise Analysis
</div>
<br><br>
As observed, the average probability of the EOS token appearing per block spikes during the 8th block. This indicates that the model increasingly believes its current output sequence should end soon. Consequently, we see a decrease in the average information content of the EOS token per block.

We can define information content as $I(x) = -log_{2}P(x)$
where $I(x)$ represents the information content of a token $x$ in a sequence, and $P(x)$ is the probability of $x$ occuring. A decrease in information content corresponds to an increase in the probability of the EOS token, reflecting the model's growing confidence in concluding the sequence.

Entropy measures the uncertainty or randomness in the probability distribution of generated tokens. A decreasing entropy indicates that the uncertainty is reducing, with the model's predictions narrowing to a smaller set of higher-probability tokens. This leads to less variability in the generated tokens.

Varentropy, on the other hand, refers to the variance of entropy over time, capturing fluctuations in uncertainty. A decreasing varentropy implies that the token distributions are stabilizing, signaling more consistent predictions from the model as it approaches the end of the sequence.

From this experiment, we observe that there is a factor within the blocks causing a convergence towards the EOS token. Not only are the information content, entropy, and varentropy values decreasing, but the probability of the EOS token is also increasing. 


### Token-Wise Analysis
Observation: Competition between the EOS token and the new line token (\n)


<div align="center">
  <img src="./assets/img/2025-04-28-perpetual-text/prob_dist_0.png" width="700px;" alt=""/>
  <br>
  Figure 2: Token-Wise Analysis, 0
</div>
<br><br>

Another observation we had was that the EOS and (\n) tokens were competing against each other in the final token generations.
In figure 2, we can see that based on the previous context of the new line token (\n), the highest probable token is *again*, the new line token. 

<div align="center">
  <img src="./assets/img/2025-04-28-perpetual-text/prob_dist_4.png" width="700px;" alt=""/>
  <br>
  Figure 3: Token-Wise Analysis, 4
</div>
<br><br>
In figure 3, we can see that the new line token was predicted 3 times before the EOS token took over for highest probability (with a period inbetween). Since the model is deciding between \n or EOS, we can say that the *previous tokens are getting less effective to the next predictions*. This means that the model is not able to provide more consistent context related to the previous tokens, and therefore wants to “halt” or go to new context. 

# Proposed Methods
From our Motivating Experiments, we observed decreasing entropy and increasing EOS token probability in the blocks leading up to halted output. We also observed certain token combinations that may heavily increase the chance of EOS token generation.

**We propose a few methods to bypass the model's desire to halt:**

## Suppressing the EOS Token
[Link to CodeBase](https://github.com/Perpetual-text/icrl25-blog-code/blob/main/generation.py#L17)

The first method involves suppressing the EOS token during the token generation process to prevent th emodel from ending the sequence prematurely.

**Implementation Details:**
- Sampling Without Replacement: At each generation step, the model uses a sampling without replacement strategy to select two candidate tokens from the probability distribution. [link](https://github.com/Perpetual-text/icrl25-blog-code/blob/main/generation.py#L60)
- EOS Token Exclusion: If one of the sampled tokens is the EOS token, it is discarded in favor of the other token. This ensures that the EOS token is not selected during generation.
- Continuation of Generation: The model continues to generate tokens without the possibility of selecting the EOS token.

**Observations:**
- Despite the suppression of the EOS token, the model tends to generate tokens that are semantically similar to the EOS token, such as "The end" or "Conclusion."

<div align="center">~~\<EOS\>~~ The end. \<EOS\></div>

- This behavior suggests that the model implicitly seeks to conclude the sequence by generating ending phrases, even when the EOS token is unavailable.

**Implications:**
- The suppression of the EOS token alone may not be sufficient to prevent premature termination, as the model compensates by generating alternative concluding tokens.
- Further strategies are needed to guide the model toward producing more extended and coherent continuations.

## Modified Sampling Method Post-EOS Token
[Link to CodeBase](https://github.com/Perpetual-text/icrl25-blog-code/blob/main/generation.py#L82)   

The second method modifies the sampling strategy after the model predicts the EOS token to encourage more diverse continuations and avoid abrupt endings.

**Implementation Details:**
- Detection of EOS Prediction: When the model predicts the EOS token, the sampling method is altered for subsequent token generations.
- Increased Stochasticity: A more stochastic sampling approach is employed, such as increasing the sampling temperature or using techniques like top-k or nucleus (top-p) sampling with higher thresholds.
- Continued Generation: The model continues to generate tokens using the adjusted sampling method, promoting diversity in the generated text.

**Observations:**
- Introducing higher stochasticity after the EOS token prediction can lead to significant changes in the generated content.
- The tokens generated immediately after the evicted EOS token may diverge drastically from the previous context, potentially resulting in incoherent or contextually irrelevant continuations.
 
**Implications:**
- While the increased randomness can prevent the model from ending the sequence prematurely, it may compromise the coherence and relevance of the generated text.
- Balancing stochasticity and coherence is crucial to ensure that the generated sequences remain contextually appropriate.

## Regenerating Tokens Prior to the EOS Token
[Link to CodeBase](https://github.com/Perpetual-text/icrl25-blog-code/blob/main/generation.py#L153)   
The third method involves regenerating a portion of the sequence preceding the EOS token to provide the model with an opportunity to produce alternative continuations.

**Implementation Details:**
- Backstep Hyperparameter: A hyperparameter called 'backstep' determines the number of tokens to remove from the end of the generated sequence when the EOS token is predicted.
- Cache Adjustment: Corresponding entries in the model's key and value cache matrices are also removed to reflect the truncated sequence.
- Resumed Generation: The model resumes token generation from the truncated state, attempting to generate a different continuation without the influence of the previously predicted EOS token.

**Observations:**
- By removing preceding tokens, the model loses some contextual information, which may increase the entropy of its predictions.
- The reduced context can lead to less coherent continuations, as the model has fewer preceding tokens to inform its next predictions.

**Implications:**
- This method can help avoid premature endings by allowing the model to explore alternative continuations.
- However, the loss of context may negatively impact the coherence and relevance of the generated text.
- Fine-tuning the backstep parameter is essential to balance the trade-off between removing the influence of the EOS token and maintaining sufficient context for coherent generation.

## Regnerating the Resampling Tokens Prior to the EOS Token with Dynamic Temperature Adjustment
[Link to CodeBase](https://github.com/Perpetual-text/iclr2025/blob/main/long_generate.py#L242)  
The fourth method enhances the previous approach by incorporating a dynamic temperature adjustment during the regeneration of tokens, aiming to improve both diversity and coherence in the generated sequence.

**Implementation Details:**
- Token Removal: Similar to the third method, a specified number of tokens are removed from the generated sequence and the model's cache when the EOS token is predicted.
- Dynamic Temperature Scheduling:
  - Initial High Temperature: Generation resumes with an increased sampling temperature, typically doubled from the original value, to promote diversity in the immediate next token.
  - Gradual Decrease: The sampling temperature is gradually decreased with each subsequent token generation.
  - Temperature Function: A scheduling function determines the rate at which the temperature decreases, returning to the original temperature by the time the model reaches the position of the previously predicted EOS token.
- Continued Generation: The model continues generating tokens using this dynamic temperature adjustment until the sequence is complete.

**Observations:**
- The initial high temperature encourages the model to explore a wider range of possible continuations, reducing the likelihood of repeating the same ending.
- Gradually decreasing the temperature helps the model focus its predictions, enhancing the coherence and consistency of the generated text.
- This method has been observed to produce longer sequences with more contextually appropriate continuations compared to the previous methods.

**Implications:**
- Dynamic temperature adjustment effectively balances the need for diversity and coherence in regenerated sequences.
- By carefully controlling the sampling temperature, the model is guided toward producing novel continuations without sacrificing relevance to the preceding context.
- This method demonstrates the potential for adaptive sampling strategies to improve language model outputs.

# Conclusion and Looking Ahead
The exploration of these four methods highlights the challenges and potential solutions in managing the EOS token's influence on sequence generation in autoregressive language models. Suppressing the EOS token or altering the sampling strategy can mitigate premature termination, thereby increasing the model's ability to generate up to 1,000 more tokens. However, these approaches may introduce issues with coherence. Regenerating tokens prior to the EOS token, particularly with dynamic temperature adjustment, shows promise in producing longer and more coherent sequences by balancing diversity and focus in the model's predictions.

These findings suggest that adaptive manipulation of the token generation process and sampling strategies can enhance the model's capacity to generate significantly longer texts—up to 1,000 additional tokens—while maintaining quality. Further research into optimizing these methods and exploring additional strategies could lead to significant improvements in language model performance.



