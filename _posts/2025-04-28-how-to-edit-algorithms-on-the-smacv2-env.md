---
layout: distill
title: How to edit algorithms for the SMACv2 environment
description: Your blog post's abstract.
  The SMACv2 environment provides a robust platform for testing and developing Multi-Agent Reinforcement Learning (MARL) algorithms, but navigating its codebase can be challenging. This blog post serves as a comprehensive guide to understanding the structure of MARL algorithms within the SMACv2 framework. By breaking down the key components—such as agent networks, mixer networks, and transformation networks—we demystify how these algorithms are implemented and highlight the critical locations in the codebase for editing or extending functionality. We also provide a step-by-step walkthrough for adding new algorithms to the framework, lowering the barrier to entry for researchers and developers. Whether you're seeking to adapt existing algorithms or contribute novel approaches, this guide equips you with the necessary knowledge to navigate and enhance the SMACv2 ecosystem.
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
  - name: Main components of any algorithm
  - name: QPLEX Algorithm
    subsections:
    - name: Agent network
    - name: Transformation network
    - name: Mixing network
    - name: Training loop
  # - name: Citations
  # - name: Footnotes
  # - name: Code Blocks
  # - name: Diagrams
  # - name: Tweets
  # - name: Layouts
  # - name: Other Typography?


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
- These are found in src -> modules -> mixers
3. Hyperparameters
- These are found in src -> config -> algs
- For any algorithm, you must first refer to default.yaml. Then you must refer to the [algorithm].yaml file (e.g., qplex.yaml) where some parameters/hyperparameters would be added/overwritten.
4. The algorithm
- These are found in src -> learners
- Refer to src -> learners -> \__init__.py to see how all the algorithm learners are 'registered'. The names of the files here may not be intuitive, so you can refer to the yaml file of that specific algorithm (found in src->config->algs) to see which learner file it is using.

<!-- 5. <span style="color: red;">The training loop</span>
- parallel_runner.py -->

## QPLEX Algorithm

Now let's take an example algorithm and see where the parts of it are located. We chose to start with the QPLEX algorithm. The architecture for the QPLEX algorithm <d-cite key="QPLEX"></d-cite> is shown below.
<div style="max-width: 600px; margin: 0 auto;">
{% include figure.html path="assets/img/2025-04-28-how-to-edit-algorithms-on-the-smacv2-env/QPLEX_architecture.png" class="img-fluid" %}
</div>

### 1. Agent Network
  Let's start bottom up from the agent network, i.e., the bottom-green part of Figure (c) in the architecture.
  <div style="max-width: 140px; margin: 0 auto;">
  {% include figure.html path="assets/img/2025-04-28-how-to-edit-algorithms-on-the-smacv2-env/QPLEX_agent_network.png" class="img-fluid" %}
  </div>
This agent network is found in src -> modules -> agents -> rnn_agent.py. There are two outputs to the `forward` function: 
- q : The q is $Q_i(\tau_i, a_i)$ in the diagram above. 
<!-- <span style="color: red;">Maybe mention its shape.</span> -->

- h : The h represents $$h_i^t$$, i.e., the next hidden state of the GRU in the diagram above.




### 2. Transformation Network

  <div style="max-width: 160px; margin: 0 auto;">
  {% include figure.html path="assets/img/2025-04-28-how-to-edit-algorithms-on-the-smacv2-env/QPLEX_transformation_network.png" class="img-fluid" %}
  </div>


Found in src -> modules -> mixers -> dmaq_general.py. There are two main operations happening here: calculating the output of the MLP and calculating the Weighted Sum. Let's see what is happening in both of these operations.


**(a) Calculating the output of the MLP**

There are 2 outputs of the MLP. One is the w and the other is b. In the `forward` function of dmaq_general.py, 

```
w_final = self.hyper_w_final(states)
w_final = th.abs(w_final)
```

The w\_final in the code is the w in the diagram above. The hyper\_w\_final is the MLP in the diagram above.


To calculate the b, once again, in the `forward` function, we have:

```
v = self.V(states)
```

The v in the code represents the b in the diagram above.

Note that in the figure, we just see 1 MLP. However, in the code there are 2 MLPs. One is hyper\_w\_final and another is V. This is because the following equation in the paper (the first part of equation 7 from the QPLEX paper <d-cite key="QPLEX"></d-cite>):

$$V_i(\mathbf{\tau}) = w_i(\mathbf{\tau})V_i(\tau_i) + b_i(\mathbf{\tau}) $$

Therefore, hyper\_w\_final represents the $$w_i(\mathbf{\tau})$$ and V represents the $$b_i(\mathbf{\tau})$$ in the above equation, respectively.



**(b) Calculating the Weighted Sum**

In the code, the advantage values are not calculated directly. The Q values and V values are calculated. Note that advantage can be calculated using the Q and V values, however, the authors find it easier to just use the Q and V values. Therefore, in the code, the output to the Transformation Network are $$V_i(\mathbf{\tau})$$ and $$Q_i(\mathbf{\tau}, a_i)$$ rather than $$V_i(\mathbf{\tau})$$ and $$A_i(\mathbf{\tau}, a_i)$$.


These Q and V values are calculated in the `forward` function as follows:

```
agent_qs = w_final * agent_qs + v
```

Here agent_qs are the Q values.

```
max_q_i = w_final * max_q_i + v
```

Here max\_q\_i are the V values because $$V_i(\tau) = max_{a_i} Q(\tau, a_i)$$


### 3. Mixing Network
  <div style="max-width: 180px; margin: 0 auto;">
  {% include figure.html path="assets/img/2025-04-28-how-to-edit-algorithms-on-the-smacv2-env/QPLEX_mixing_network.png" class="img-fluid" %}
  </div>


The mixing network for the QPLEX algorithm is in the same file as the transformation network. Found in src -> modules -> mixers -> dmaq_general.py. There are 
four main operations happening here: calculating the output of the MLP, calculating the Dot Product, summing over the $$V_i$$ values and summing the $$V_{tot}$$ and the $$A_{tot}$$. Below we explain these four operations:

**(a) Calculating the output of the MLP**

In the `calc_adv` function of dmaq_general.py, 
```
adv_w_final = self.si_weight(states, actions)
```
The `adv_w_final` are the $$\lambda$$ weights. The function `si_weight` implements the multi-head attention mechanism and is implemented in src -> modules -> mixers -> dmaq_si_weight.py.

**(b) Calculating the Dot Product**

In the `forward` function of dmaq_general.py, 
```
y = self.calc(agent_qs, states, actions=actions, max_q_i=max_q_i, is_v=is_v)
```
The function `calc` is used to calculate either $$V_{tot}$$ or $$A_{tot}$$. If the parameter `is_v` is set to True, it returns $$V_{tot}$$ and if it is set to False, the `calc` function returns $$A_{tot}$$. Here, it is set to False. The dot product is implemented in the `calc_adv` function in the same file as follows:

```
adv_tot = th.sum(adv_q * adv_w_final, dim=1)
```
Here `adv_q` are the $$A_i(\mathbf{\tau}, a_i)$$s and `adv_w_final` are the $$\lambda$$ weights. 

NOTE: In src -> modules -> mixers -> dmaq_general.py, in the `calc_adv` function when calculating the dot product, there is the following line of code:
```
if self.args.is_minus_one:
    adv_tot = th.sum(adv_q * (adv_w_final - 1.), dim=1)
```
This resembles the rightmost term in equation 11 in the QPLEX paper<d-cite key="QPLEX"></d-cite>. Therefore, in the code, they are implementing $$Q_{tot}(\mathbf{\tau}, \mathbf{a}) = \sum_{i=1}^n Q_i(\mathbf{\tau}, a_i) + \sum_{i=1}^n (\lambda_i (\mathbf{\tau},\mathbf{a}) - 1) A_i(\mathbf{\tau}, a_i)$$, In the code, the `calc_v` function actually returns $$\sum_{i=1}^n Q_i(\mathbf{\tau}, a_i)$$ and the `calc_adv` function returns $$\sum_{i=1}^n (\lambda_i (\mathbf{\tau},\mathbf{a}) - 1) A_i(\mathbf{\tau}, a_i)$$. However, this is fine because ultimately the sum $$Q_{tot}$$ turns out to be correct, but this is an important detail if you wish to edit the agorithm.



**(c) Summing over the $$V_i$$ Values**

<!-- ```
y = self.calc(agent_qs, states, actions=actions, max_q_i=max_q_i, is_v=is_v)
```

For this case, the parameter `is_v` is set to True. Therefore y returns $$V_{tot}$$. As mentioned in the previous point,  -->

In the `calc_v` function of dmaq_general.py, 
```
v_tot = th.sum(agent_qs, dim=-1)
```

As mentioned in the previous point, this `v_tot` is actually $$\sum_{i=1}^n Q_i(\mathbf{\tau}, a_i)$$.


**(d) Summing the $$V_{tot}$$ and the $$A_{tot}$$**

In src -> modules -> learners -> dmaq_qatten_learner.py
```
chosen_action_qvals = ans_chosen + ans_adv
```

Here chosen_action_qvals represents the $$Q_{tot}$$, ans_chosen represents the $$V_{tot}$$ (but as mentioned in the previous point, mathematically it is $$\sum_{i=1}^n Q_i(\mathbf{\tau}, a_i)$$ )and ans_adv represents $$A_{tot}$$ (but as mentioned in the previous point, mathematically it is $$\sum_{i=1}^n (\lambda_i (\mathbf{\tau},\mathbf{a}) - 1) A_i(\mathbf{\tau}, a_i)$$).


<!-- ### 4. The Training Loop

Found in src -> learners -> dmaq_qatten_learner.py




<span style="color: red;">Talk about basic_controller.</span>


<span style="color: red;">Replay buffer.</span>
1. Initialised as zeros
2. Masking
3. Shape
  Batch_Size = B , Timestep = T, Agents = A
  1. state = (B, T, state_dim)
  2. obs = (B, T, A, obs_dim)
  3. actions = (B, T, A, 1) #chosen actions
  4. avail_actions = (B, T, A, n_actions) # n_actions is the total number of actions that an agents can take.
  5. probs = (B, T, A, 1) #probabilities of the chosen action
  6. reward = (B, T, 1)
  7. terminated = (B, T, 1)
  8. actions_onehot = (B, T, A, n_actions)


action selection is done using EpsilonGreedyActionSelector found in src -> components -> action_selectors.py

lamdba return -->





## How to add a new algorithm

Steps:
1. Registry - add to all init files
2. Create your controller
3. agent 
4. learner



## Some additional information:

1. Procedurally generated, so seed does not work (link github issue here).
2. We never really use the joint trajectory $$\mathbf{\tau}$$. It is assumed that adding the information of the state to $$\tau_i$$ gives us a good representation of $$\mathbf{\tau}$$.
3. batch_size and batch_size_run refer to two different concepts, though their names are used interchangeably in the codebase. batch_size represents the training batch size used to sample episodes from the buffer, while batch_size_run denotes the number of parallel environments configured for algorithm execution.  



## Future Work

If the community finds this blog useful, we would like to add explanations of other algorithms like VDN, Qatten, etc.