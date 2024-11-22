---
layout: distill
title: Challenges of Sample Inefficiency (CSI) - Practical Limitations of Direct Preference Optimization Algorithm
description: In this blog, we compare Direct Preference Optimization (DPO) and Proximal Policy Optimization (PPO) from a reinforcement learning perspective. The absence of a critic model, the lack of GAE estimation, and the use of off-policy sampling in DPO result in high variance but unbiased token-wise rewards estimates. This leads to a significant drawback of DPO - sample inefficiency. Due to limited training samples and a reliance on off-policy data, DPO faces the state distribution shift problem. Additionally, as a Bradley-Terry model with limited samples, DPO struggles to distinguish response pairs with substantial token overlap while still attempting to maximize the difference between them. This interplay between the state distribution shift problem and the limitations of the Bradley-Terry model can result in reduced likelihoods for both positive and negative samples. DPO poses challenges in sample efficiency when compared to PPO, making it less practical in data-limited tasks.
date: 2025-04-28
future: true
htmlwidgets: true
hidden: false

# Anonymize when submitting
authors:
  - name: Anonymous

# must be the exact same name as your blogpost
bibliography: 2025-04-28-dpo-practice.bib

# Add a table of contents to your post.
toc:
  - name: Introduction
  - name: Contrasting DPO and PPO from an RL Perspective
    subsections:
      - name: Preliminary
      - name: Multi-Arm Bandit vs. Markov Decision Process
      - name: Monte Carlo Method vs. Temporal Difference Learning
      - name: Bradley-Terry Model vs. Weighted Logistic Model
      - name: REINFORCE vs. Actor-Critic Method
      - name: Off-Policy Method vs. On-Policy Method
  - name: Limitations of DPO
    subsections:
      - name: Sample Efficiency
      - name: Both the Probability of the Chosen Response and the Rejected Response Drop
---

## Introduction

> Reinforcement Learning From Human Feedback (RLHF) has been critical to the success of the latest generation of generative AI models. In response to the complex nature of the classical RLHF pipeline, direct alignment algorithms such as Direct Preference Optimization (DPO) have emerged as an alternative approach. Although DPO solves the same objective as the standard RLHF setup, there is a mismatch between the two approaches <d-cite key="rafailov2024from"></d-cite>.

In this blog, we explore the distinctions between Direct Preference Optimization (DPO) <d-cite key="rafailov2024direct"></d-cite> and Reinforcement Learning from Human Feedback (RLHF), focusing on Proximal Policy Optimization (PPO) <d-cite key="schulman2017proximal"></d-cite> from a reinforcement learning (RL) perspective.  We highlight the following contrasts:

- DPO conceptualizes response generation as a multi-arm bandit problem <d-cite key="bouneffouf2019survey"></d-cite>, while PPO views it as a Markov Decision Process (MDP) <d-cite key="knox2012reinforcement"></d-cite>.
- DPO uses a Monte Carlo method <d-cite key="jiang2021reinforcement"></d-cite> for pairwise comparisons, while PPO uses Generalized Advantage Estimation (GAE), a refined n-step Temporal Difference (TD) learning <d-cite key="pong2018temporal"></d-cite>, for token-wise reward estimation.
- DPO utilizes the Bradley-Terry model <d-cite key="han2020asymptotic"></d-cite> for learning preferences, while PPO uses a weighted logistic model <d-cite key="wilson2015weighted"></d-cite> for point-wise token rankings.
- DPO is a variant of the REINFORCE algorithm <d-cite key="sewak2019policy"></d-cite>, while PPO is a variant of the Actor-Critic algorithm <d-cite key="konda1999actor"></d-cite>.
- DPO is an off-policy method <d-cite key="degris2012off"></d-cite>, learning from an offline dataset, while PPO is an on-policy algorithm <d-cite key="deisenroth2013survey"></d-cite>, relying on data from the current policy.

DPO's lack of GAE, a critic model, and its reliance on off-policy sampling cause high-variance reward estimates, leading to **sample inefficiency**.  We detail these differences and present experiments demonstrating DPO's limitations:

- State distribution shift due to limited samples and off-policy data.
- Difficulty distinguishing responses with significant token overlap.
- Potential decrease in likelihoods of both positive and negative samples.


## Contrasting DPO and PPO from an RL Perspective

This section introduces DPO and PPO, demonstrating their mathematical equivalence and exploring their differences from a traditional RL perspective.

### Preliminary

In traditional RLHF <d-cite key="ouyang2022training"></d-cite><d-cite key="achiam2023gpt"></d-cite>, the Bradley-Terry model is utilized to represent the preference function as a sigmoid function of the difference between rewards:

$$
p(y\succ y^{'}|x)=\sigma(r(x,y)-r(x,y^{'}))\ \ \ (1)
$$

where $\sigma$ is the sigmoid function. Given a pairwise human preference dataset $D = (x_i, y \succ y^{'})$, the reward function is learned by minimizing the logistic regression loss:

$$
L(r) = -E_{(x,y,y^{'})\sim D}[\log(p(y \succ y^{'}|x))] \ \ \  (2)
$$

$$
L(r) = -E_{(x,y,y^{'})\sim D}[\log(\sigma(r(x, y) - r(x, y^{'})))] \ \ \  (3)
$$

With the learned reward function $r(x, y)$, RLHF optimizes the policy $\pi$ to maximize expected reward while minimizing the KL divergence from a reference policy $\pi_{ref}$:

$$
J(\pi) = E_{\pi}[r(x,y)] - \beta D_{\text{KL}}(\pi | \pi_{\text{ref}}) \ \ \ (4)
$$

$$
D_{\text{KL}}(\pi | \pi_{\text{ref}}) = E_{\pi}[\log (\frac{\pi}{\pi_{\text{ref}}})] \ \ \ (5)
$$

PPO optimizes this objective. Integrating a reward model with PPO has been successful in practice <d-cite key="ouyang2022training"></d-cite><d-cite key="achiam2023gpt"></d-cite>.

Instead of learning a reward function, DPO <d-cite key="rafailov2024direct"></d-cite> directly optimizes Equation 3 by relating the policy $\pi$ and $r(x, y)$.  It frames learning from preferences as an offline contextual bandit problem, deriving a closed-form expression for the optimized policy $\pi$:

$$
\pi_r(y|x) = \frac{1}{Z(x)}\pi_{\text{ref}}(y|x)\exp(\frac{1}{\beta}r(x,y)) \ \ \ (6)
$$

where  
$$ Z(x) = \sum_{y}\pi_{ref}(y | x) exp(\frac{1}{\beta}r(x, y)) $$ 
 is the *partition function*. See <d-cite key="rafailov2024direct"></d-cite> for a complete derivation.  Taking the logarithm and rearranging Equation 6 yields:

$$
r(x,y) = \beta \log\frac{\pi_r(y|x)}{\pi_{\text{ref}}(y|x)} + \beta \log Z(x) \ \ \ (7)
$$

Applying this to the ground-truth reward $ r^* $ and optimal model $\pi^*$ in Equation 3, DPO minimizes the following loss:

$$
 \min_\pi E_{(x, y,y^{'})\sim D}[-\log \sigma(\beta \log \frac{\pi(y|x)}{\pi_{\text{ref}}(y|x)} - \beta \log \frac{\pi(y^{'}|x)}{\pi_{\text{ref}}(y^{'}|x)})]\ \ \ \ (8)
$$

Both DPO and PPO can achieve the optimal policy $\pi^*$, but their methodologies differ.


### Multi-Arm Bandit vs. Markov Decision Process

DPO conceptualizes LLM response generation as a multi-arm bandit problem, treating the entire response as a single arm.  The user instruction is the state $s$ (distributed according to $\rho$), the response is action $a$, the LLM is the parameterized policy $\pi_{\theta}$, and user preference is $r(s,a)$.  DPO's objective function is:

$$
O_d = \max_{\pi_{\theta}}E_{s \sim \rho, a \sim \pi_\theta(s)}(r(s,a))\ \ \ (9)
$$

PPO views response generation as an MDP. The state $s$ includes user instructions and generated tokens, the action $a$ is the next predicted token, and $\tau$ represents the combined instruction and response. The LLM is the policy $\pi_{\theta}$, user preference is $r(\tau)$, $s_0$ is the initial instruction, and $s_i$ is the instruction with the first $i$ tokens. The probability of $\pi_{\theta}$ generating $\tau$ is:

$$
p_{\pi_{\theta}}(\tau) = \rho(s_0)\prod_{i} \pi_{\theta}(a_i|s_i) C(s_{i+1}|s_i, a_{i}) \ \ \ (10)
$$

where 
$$ C(s_{i+1} | s_i, a_{i}) $$ 
is the state transition matrix, which, given the nature of LLMs, is:

$$
C(s_{i+1} | s_i, a_{i}) = \mathbf{1}(a_i == s_{i+1}) \ \ \ (11)
$$

where $\mathbf{1}$ is an indicator function. PPO's objective function is:

$$
 O_p = \max_{\pi_{\theta}}E_{\tau \sim p_{\pi_\theta}}(r(\tau))\ \ \ (12)
$$

Here, $\tau$ corresponds to $(s, a)$ in Equation 9. Thus, PPO's objective ($O_p$) is identical to DPO's objective ($O_d$).

>💡Although DPO treats response generation as a multi-arm bandit problem and PPO as an MDP, their objectives are identical.



### Monte Carlo Method vs. Temporal Difference Learning

In DPO, rewards for each action (the entire response) are sampled and averaged, similar to a Monte Carlo method.  The probability of generating a response (Equation 6) can be considered a reshaped reward.  DPO treats the entire response as an action, potentially overlooking important tokens and introducing high variance in reward estimation.  Accurate reward estimation requires at least $O(m^n)$ samples (vocabulary size $m$, maximum response length $n$), making DPO sample-inefficient.

PPO estimates token-wise rewards, identifying important tokens but potentially introducing bias. It uses GAE, a form of n-step TD learning. TD learning learns the value function based on the difference between estimated values of consecutive states.  The TD error is:

$$
\delta_t = r(s_t, a_t) + \gamma V(s_{t+1}) - V(s_t) \ \ \ (13)
$$

GAE combines TD learning and Monte Carlo methods to estimate the advantage function (token-wise reshaped rewards):

$$
G_t^{(\gamma, \lambda)} = \sum_{k=t}^T(\gamma \lambda)^{k-t}\delta_k \ \ \ (14)
$$

where $\gamma$ and $\lambda$ are discount factors.

>💡With sufficient (exponential) samples, both DPO and PPO can accurately estimate reward, achieving the optimal policy $\pi^*$.


>💡With limited samples, PPO reduces variance in reward estimation, providing a more stable estimate than DPO. While DPO can theoretically be framed as a token-level MDP <d-cite key="rafailov2024from"></d-cite>, it requires more samples than PPO for accurate token-level credit assignment.



## Bradley-Terry Model vs. Weighted Logistic Model

DPO uses the Bradley-Terry (BT) model:

$$
L_{\text{BT}} = -\log\delta(y_w - y_l) \ \ \ (15)
$$


The BT model in RLHF may suffer from limitations <d-cite key="alignmentguidebook"></d-cite><d-cite key="huggingfacearmor"></d-cite>:

- Reduced generation diversity.
- Neglect of minority preferences.


PPO uses a weighted logistic model <d-cite key="wilson2015weighted"></d-cite> for point-wise token rankings:

$$
 L_{\text{WL}} = - \sum_{i} w_i * \text{CrossEntrophy}(\text{logit}_{i}, y_i) \ \ \ (16)
$$

where $w_i$ is the token-wise reshaped reward for token $y_i$ in context $y_{0:i-1}$.  Both PPO and DPO might suffer from reduced diversity and neglect minority preferences if the reward model uses the BT framework.  Ensembles and other tools are used in practice to mitigate these issues.

>💡DPO is prone to reduced diversity and overlooking minority preferences, hindering LLM improvement. PPO, with a robust reward model, can continue to improve LLMs.


### REINFORCE vs. Actor-Critic Method

DPO is a variant of REINFORCE <d-cite key="sewak2019policy"></d-cite>, equivalent to:

1. Learning a reward model using the BT framework.
2. Optimizing the LLM policy with REINFORCE, sampling queries from the human preference dataset.

REINFORCE has high variance and slow convergence.

PPO, a variant of the Actor-Critic algorithm <d-cite key="konda1999actor"></d-cite>, uses a value function to estimate token-wise rewards, reducing variance and improving learning stability and speed compared to DPO.

>💡 PPO's use of a value function for reward estimation leads to more stable and faster learning than DPO.


### Off-Policy Method vs. On-Policy Method

DPO is off-policy, learning from data not necessarily generated by the current policy.  This can lead to a state distribution shift problem.  Even if the offline dataset is sampled from the SFT model, the distribution of generated responses can shift during training, exacerbating this problem.

PPO is on-policy, sampling responses from the current policy, evaluating them with a reward model, and using the results to train the policy.  While the reward model can encounter out-of-distribution issues, the impact is less severe than in DPO.

>💡DPO's off-policy nature offers flexibility but can cause state distribution shift. PPO's on-policy updates are slower but more stable.


## Limitations of DPO

### Sample Efficiency

From a reinforcement learning perspective, DPO is a variant of the Monte Carlo Policy Gradient Algorithm. The absence of a critic model, the lack of GAE estimation, and the use of off-policy sampling result in high variance but unbiased token-wise reward estimates in DPO, leading to its principal shortcoming: sample inefficiency.

In the following tables, we compare the DPO algorithm and PPO algorithm through our experiments utilizing leetcode datasets.

**Experimental Settings:** We initially gathered 1,200 Python prompts from the LeetCode website, spanning from September 2022 to September 2023, for our training set. Additionally, we collected another 474 prompts from September 2023 to June 2024 for our test set. Subsequently, for both the training and test sets, we collected five responses per prompt from various models, including GPT-4, GPT-4o, DeepSeek, among others. To establish the ground truth for each response, we submitted these responses to the LeetCode website. This process yielded 30,000 training preference pairs and 12,000 test preference pairs. Finally, we randomly selected 20%, 40%, 60%, 80%, and 100% of the data to train the DPO and PPO policies based on the SFT model.

**Experimental Results:** As shown in the following tables, the DPO algorithm finds it difficult to scale to large preference datasets, whereas the PPO algorithm scales easily to larger datasets. Furthermore, across all preference datasets, PPO policies outperform DPO policies. This also demonstrates that the PPO algorithm is more sample-efficient than the DPO algorithm.

|  | SFT | SFT + DPO | SFT + PPO |
| --- | --- | --- | --- |
| 20%  | 21.2% | 21.4% | 22.6% |
| 40% | 21.2% | 22.8% | 23.2% |
| 60% | 21.2% | 22.6% | 24.2% |
| 80% | 21.2% | 21.5% | 25.1% |
| 100% | 21.2% | 19.8% | 25.8% |


### Both the Probability of the Chosen Response and the Rejected Response Drop

We investigate why DPO sometimes causes both the probability of the chosen and rejected responses to drop under a prompt.

As shown in the Figure 1, we find that when trained with a math preference dataset, the DPO algorithm results in both the probability of the chosen response and the probability of the rejected response dropping <d-cite key="pal2024smaug"></d-cite>.

{% include figure.html path="assets/img/2025-04-28-dpo-practice/math-qwen2-1.5b-all.png" class="img-fluid" %}
<div class="caption">
    Figure 1: both of the logits of the chosen response and the logits of the rejected response dropping. (Notes: the rewards is the sum of the logits.)
</div>

This problem is more pronounced when training with only one preference pair with a small edit distance (See Figure 2):

```json
{
  "prompt": "A rectangular prism has a length of 20 cm, a width of 5 cm, and a height of 6 cm. What is its volume?",
  "chosen": "1. It is known that the length of the rectangular prism is 20 cm, the width is 5 cm, and the height is 6 cm.\n2. According to the volume formula of a rectangular prism: volume = length × width × height\n3. Substituting the specific values into the formula: volume = 20 × 5 × 6 = 600 (cubic centimeters)\n4. Therefore, the volume of this rectangular prism is 600 cubic centimeters.",
  "rejected": "1. It is known that the length of the rectangular prism is 20 cm, the width is 5 cm, and the height is 6 cm.\n2. According to the volume formula of a rectangular prism: volume = length × width × height\n3. Substituting the specific values into the formula: volume = 20 × 5 × 6 = 500 (cubic centimeters)\n4. Therefore, the volume of this rectangular prism is 500 cubic centimeters.",
  "target": "600",
  "edit_dis": 3
}
```

{% include figure.html path="assets/img/2025-04-28-dpo-practice/math-qwen2-1.5-top1-pair.png" class="img-fluid" %}
<div class="caption">
    Figure 2: both of the logits of the chosen response and the logits of the rejected response dropping with small edit distance. (Notes: the reward is the sum of the logits.)
</div>

This might be due to:

- **State Distribution Shift:** With limited off-policy samples, the LLM learns specific tokens in positive responses while neglecting others, leading to generation of responses with easily generated tokens instead of the full positive response.
- **Bradley-Terry Limitations:** DPO maximizes the difference between positive and negative samples but doesn't guarantee an increase in the likelihood of positive samples.  With substantial token overlap, DPO struggles to distinguish important tokens, potentially decreasing the likelihood of both samples.
- **Overfitting:** The BT model can overfit simpler pairwise samples and neglect more challenging ones with limited data, leading to model collapse and reduced probability of generating other chosen responses.

The interplay between state distribution shift and the limitations of the Bradley-Terry model can result in reduced likelihoods for both positive and negative samples <d-cite key="rafailov2024from"></d-cite><d-cite key="pal2024smaug"></d-cite>.
