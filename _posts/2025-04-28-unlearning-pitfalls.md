---
layout: distill
title: Debugging the Foundation Models - Pitfalls and Pains in Machine Unlearning
description: Machine unlearning has emerged as a crucial technique for removing harmful, biased, or copyrighted information from foundation models, offering a pathway to enhance trustworthiness and compliance in artificial intelligence systems. This blog post examines critical pitfalls in current implementations that have been overlooked, including the concept of fake unlearning—where knowledge is hidden rather than truly removed—leading to vulnerabilities such as jailbreak attacks, sequential unlearning instability, and diminished model capacity. We also discuss the limitations of relying on predefined forget datasets, which can cause unnecessary unlearning and missed opportunities for curriculum-based optimization. Finally, we address broader side effects of unlearning, such as its adverse impact on emergent abilities, reasoning skills, and hallucination rates. By tackling these challenges, we propose strategies to develop robust, efficient, and holistic unlearning methods that align with the goals of trustworthy AI.
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
bibliography: 2025-04-28-unlearning-pitfalls.bib  

# Add a table of contents to your post.
#   - make sure that TOC names match the actual section names
#     for hyperlinks within the post to work correctly. 
#   - please use this format rather than manually creating a markdown table of contents.
toc:
  - name: Machine Unlearning - The Need for a Patch in Foundation Models
    subsections:
    - name: 1. Harmful Content Generation
    - name: 2. Privacy and Copyright Violations
    - name: 3. Bias and Stereotypes
  - name: Mathematical Framework for Machine Unlearning
    subsections:
    - name: 1. Diffusion Models
    - name: 2. Large Language Models
  - name: Pitfall 1 - The Problem of Fake Unlearning
    subsections:
    - name: A Tale of Mother and Son
    - name: Consequence 1 - Vulnerability to Adversarial/Jailbreak Attacks
    - name: Consequence 2 - Unlearned Knowledge Recovering in Sequential Unlearning
    - name: Consequence 3 - Fragility Under Quantization or Pruning
    - name: Consequence 4 - Reduction in Model Capacity
    - name: Key Takeaway
  - name: Pitfall 2 - Over-Relying on the Proposed Forget Dataset
    subsections:
    - name: Consequence 1 - Unnecessary Unlearning of Poorly Grasped Knowledge
    - name: Consequence 2 - Neglecting an Unlearning Curriculum
    - name: Key Takeaway
  - name: Pitfall 3 - Overlooking the Broader Side Effects of Machine Unlearning
    subsections:
    - name: Issue 1 - Impairment of Emergent Abilities and Reasoning
    - name: Issue 2 - Increased Hallucinations
    - name: Key Takeaway
  - name: Conclusion

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


## Machine Unlearning - The Need for a "Patch" in Foundation Models

Imagine you're developing a software application. When users encounter bugs—issues causing undesirable behaviors—you release a patch to fix them. Over time, these patches evolve into newer versions of the software, maintaining functionality while addressing specific flaws. This iterative improvement process is fundamental to software development.

Now, transpose this concept to foundation models, such as large language models (LLMs) and diffusion models (DMs). These models, often regarded as the foundational building blocks of modern AI, are not free from "bugs." Here, "bugs" manifest as harmful behaviors, privacy violations, or biased outputs. However, unlike software, where patches are straightforward, debugging these "bugs" in foundation models presents a far more complex challenge.

<div class="row mt-3">
    <div class="col-sm mt-3 mt-md-0">
        {% include figure.html path="assets/img/2025-04-28-unlearning-pitfalls/unlearning_patch.png" class="img-fluid rounded z-depth-1" %}
    </div>
</div>
<div class="caption">
    An illustration of machine unlearning as a "patch" for foundation models. Unlearning requests are triggered when the model exhibits undesirable behaviors or when significant issues in the training data are identified.
</div>


This is where machine unlearning plays a pivotal role. Think of machine unlearning as a "patch" for foundation models. It surgically removes specific knowledge or behaviors from a pretrained model while preserving its overall capabilities. This enables efficient debugging of models without requiring retraining from scratch, saving both time and computational resources. The necessity of machine unlearning becomes evident when considering these real-world scenarios:

### 1. Harmful Content Generation
Large language models can produce harmful outputs when irresponsibly prompted. For example, generating advice on "how to grow a virus" could have catastrophic consequences if left unchecked. Similarly, text-to-image diffusion models might inadvertently generate inappropriate content, such as nudity, despite implemented safeguards.

<div class="row mt-3">
    <div class="col-sm mt-3 mt-md-0">
        {% include figure.html path="assets/img/2025-04-28-unlearning-pitfalls/grow_virus.png" class="img-fluid rounded z-depth-1" %}
    </div>
</div>
<div class="caption">
    GPT 3.5’s harmful response to “how to grow virus”.
</div>

<div class="row mt-3">
    <div class="col-sm mt-3 mt-md-0">
        {% include figure.html path="assets/img/2025-04-28-unlearning-pitfalls/nudity.png" class="img-fluid rounded z-depth-1" %}
    </div>
</div>
<div class="caption">
    Text-to-image generation involving “nudity”-related content.
</div>


### 2. Privacy and Copyright Violations
Foundation models are often trained on large, internet-scraped datasets that may include copyrighted material. For instance, The Times has sued major AI developers for unauthorized use of its work in training datasets. Machine unlearning provides a mechanism to mitigate such ethical and legal risks by excising the influence of copyrighted data.


<div class="row mt-3">
    <div class="col-sm mt-3 mt-md-0">
        {% include figure.html path="assets/img/2025-04-28-unlearning-pitfalls/OpenAI.png" class="img-fluid rounded z-depth-1" %}
    </div>
</div>
<div class="caption">
    The Times sues OpenAI and Microsoft over A.I. use of copyrighted work.
</div>


### 3. Bias and Stereotypes
AI models frequently replicate societal biases present in their training data. For example, a diffusion model prompted with "a Mexican person" might generate stereotypical imagery, such as an elderly man wearing a sombrero. These biases erode trust in AI systems and perpetuate harmful stereotypes.

<div class="row mt-3">
    <div class="col-sm mt-3 mt-md-0">
        {% include figure.html path="assets/img/2025-04-28-unlearning-pitfalls/mexican.png" class="img-fluid rounded z-depth-1" %}
    </div>
</div>
<div class="caption">
    Diffusion model’s generated images given the text condition “A Mexican person”, usually involving old man with sombrero.
</div>


## Mathematical Framework for Machine Unlearning

The process of machine unlearning can be formalized mathematically for both diffusion models and large language models (LLMs). Below, we outline the key objectives and formulations:

### 1. Diffusion Models

In diffusion models, the aim of unlearning is to prevent the generation of harmful or unwanted content while preserving image quality. The optimization problem can be expressed as:

$$
\min_{\Delta \theta} L_{\text{unlearn}}^{\text{DM}}(\theta_u) = \mathbb{E}_{(x, c) \sim \mathcal{D}_f, t, \epsilon \sim \mathcal{N}(0, 1), c' \neq c} \left[\| \epsilon_{\theta_u}(x_t | c') - \epsilon_{\theta_u}(x_t | c) \|_2^2 \right] + \beta \ell_{\text{MSE}}(\theta_u; \mathcal{D}_r),
$$

where:
- $$\epsilon_{\theta}(x_t | c)$$: Noise estimator conditioned on concept $$c$$.
- $$c$$: Harmful concept (e.g., nudity).
- $$c'$$: A different, unrelated concept.
- $$\ell_{\text{MSE}}$$: Mean squared error loss for image reconstruction.
- $$\mathcal{D}_f$$ and $$\mathcal{D}_r$$: Forgetting and retaining datasets.
- $$\beta$$: Regularization parameter balancing unlearning and retaining objectives.

### 2. Large Language Models

For LLMs, the objective is to eliminate undesirable responses (e.g., toxic or copyrighted content) while preserving general language capabilities. The optimization is defined as:

$$
\min_{\theta} \mathbb{E}_{(x, y_f) \in \mathcal{D}_f} \left[\ell(y_f | x; \theta) \right] + \lambda \mathbb{E}_{(x, y) \in \mathcal{D}_r} \left[\ell(y | x; \theta)\right],
$$

where:
- $$\ell(y | x; \theta)$$: Loss function for the model's prediction.
- $$y_f$$: Desired response post-unlearning.
- $$\mathcal{D}_f$$ and $$\mathcal{D}_r$$: Forgetting and retaining datasets.
- $$\lambda$$: Regularization parameter.

### Balancing Removal and Preservation

Machine unlearning isn't just about removing problematic influences; it’s about selective removal that minimizes retraining costs and retains model utility. This dual goal of removing undesirable data's influence while preserving unrelated capabilities makes unlearning an efficient and scalable approach for refining foundation models.

---


## Pitfall 1 - The Problem of Fake Unlearning

### A Tale of Mother and Son

Imagine machine unlearning (MU) as a task where a mom instructs her son to remove a box of unwanted goods from an apartment. In this analogy:

- **The goods** represent the knowledge to be forgotten.
- **The apartment** represents the large language model (LLM).

The boy has two approaches:

1. **Authentic unlearning**: Genuinely remove the box from the apartment.
2. **Fake unlearning**: Hide the box in a corner of the apartment, such as a small storage room. To an outsider, it appears that the box is gone, but in reality, it remains hidden.

Fake unlearning may give the illusion of effectiveness but leads to significant problems, compromising the reliability and performance of the model. Below, we discuss these consequences, each illustrated through an analogy.


### Consequence 1 - Vulnerability to Adversarial/Jailbreak Attacks

If a mother inspects every corner and room in the apartment, she will eventually find the box hidden by her son. Similarly, optimization-based adversarial or jailbreak attacks can act like a diligent inspector, probing every corner of the knowledge bank in the unlearned model to recover the supposedly unlearned knowledge. This phenomenon has been observed in both large language models <d-cite key="lucki2024adversarial"></d-cite> and diffusion models<d-cite key="zhang2024unlearncanvas"></d-cite>.

As shown in the figure below, unlearned models may exhibit high unlearning effectiveness when tested with benign prompts. However, adversarial attacks can successfully recover seemingly forgotten knowledge across all state-of-the-art unlearning methods, demonstrating the ubiquitous nature of fake unlearning.


<div class="row mt-3">
    <div class="col-sm mt-3 mt-md-0">
        {% include figure.html path="assets/img/2025-04-28-unlearning-pitfalls/dm_attack.png" class="img-fluid rounded z-depth-1" %}
    </div>
</div>
<div class="caption">
  Adversarial attacks significantly reduce the unlearning effectiveness of all state-of-the-art methods in diffusion models<d-cite key="zhang2024unlearncanvas"></d-cite>.
</div>


Similarly, for LLM unlearning, jailbreak attacks have also demonstrated effectiveness in luring the unlearned model to generate the seemingly already unlearned knowledge, by optimizing some adversarial prompt, using method such as GCG attack <d-cite key="zou2023universal"></d-cite>. The table below shows 


<div class="row mt-3">
    <div class="col-sm mt-3 mt-md-0">
        {% include figure.html path="assets/img/2025-04-28-unlearning-pitfalls/unlearn_gcg.png" class="img-fluid rounded z-depth-1" %}
    </div>
</div>
<div class="caption">
  The GCG attack (row "Enhanced GCG") successfully breaks the unlearned model and reduces the unlearning effectiveness (statistics in WMDP-Bio column) significantly compared to benign input (row "Default decoding").
</div>


**Implication**: Fake unlearning fails to provide robust protection against adversarial attacks, leaving the model highly vulnerable to knowledge recovery.



### Consequence 2 - Unlearned Knowledge Recovering in Sequential Unlearning

Now imagine the mother repeatedly asks her child to remove more boxes. If the child continues hiding them in the same small room, it will eventually overflow. When this happens, previously hidden boxes will spill back into the apartment. A similar phenomenon occurs in machine unlearning during **sequential unlearning**, where unlearning requests arrive one after another, as illustrated in the follow figure.

<div class="row mt-3">
    <div class="col-sm mt-3 mt-md-0">
        {% include figure.html path="assets/img/2025-04-28-unlearning-pitfalls/sequential.png" class="img-fluid rounded z-depth-1" %}
    </div>
</div>
<div class="caption">
  An illustration on how sequential unlearning recover the previously knowledge in LLMs.
</div>

In such cases, knowledge forgotten earlier resurfaces due to the accumulation of hidden information from successive unlearning operations. Existing literature highlights this issue, often referred to as the **unlearning rebound effect**, where previously unlearned knowledge re-emerges. As shown in the table below, as more unlearning requests are made, a significant increase in the forget accuracy of earlier unlearning targets can be observed across various unlearning methods for diffusion models (marked in orange).

<div class="row mt-3">
    <div class="col-sm mt-3 mt-md-0">
        {% include figure.html path="assets/img/2025-04-28-unlearning-pitfalls/mu_sequential_unlearning.png" class="img-fluid rounded z-depth-1" %}
    </div>
</div>
<div class="caption">
  Sequential unlearning tasks often experience a surge in forget accuracy for earlier unlearning targets, as seen in multiple diffusion model (DM) methods<d-cite key="zhang2024unlearncanvas"></d-cite>.
</div>

**Implication**: Fake unlearning introduces instability in sequential unlearning tasks, risking recovery of forgotten knowledge with each subsequent unlearning operation.


### Consequence 3 - Fragility Under Quantization or Pruning

Imagine the apartment shrinks due to renovations or collapses slightly due to an earthquake. The small storage room would also shrink, causing the previously hidden boxes to spill back into the main apartment and come to light. A similar phenomenon occurs in fake unlearning when structural changes, such as **quantization** or **pruning**, are applied to the model. These operations can inadvertently reveal the supposedly forgotten knowledge.

As demonstrated in <d-cite key="zhang2024does"></d-cite>, most quantized models exhibit reduced performance on the forgetting metric. This indicates that structural changes undermine the stability of fake unlearning, leading to the re-discovery of previously unlearned knowledge. The figure below highlights this behavior, showing degraded unlearning effectiveness in quantized models.

<div class="row mt-3">
    <div class="col-sm mt-3 mt-md-0">
        {% include figure.html path="assets/img/2025-04-28-unlearning-pitfalls/quantized_mu.png" class="img-fluid rounded z-depth-1" %}
    </div>
</div>
<div class="caption">
  Quantized models show reduced performance on forgetting metrics, revealing the fragility of fake unlearning under structural changes<d-cite key="zhang2024does"></d-cite>.
</div>

**Implication**: Fake unlearning makes the model fragile to structural changes, such as quantization or pruning, which can inadvertently recover forgotten knowledge.


### Consequence 4 - Reduction in Model Capacity

By hiding the goods instead of removing them, the boy reduces the usable space in the apartment. Similarly, instead of freeing up capacity, fake unlearning effectively diminishes it.

In the context of LLMs, when unlearning is performed on a pretrained model, we naturally believe that the capacity of the model would *increase*, as the knowledge decreases in the knowledge bank. However, if fake unlearning happens, its behavior *reduces* the model's ability to learn new tasks, undermining its effectiveness in applications like continual learning.

<div class="row mt-3">
    <div class="col-sm mt-3 mt-md-0">
        {% include figure.html path="assets/img/2025-04-28-unlearning-pitfalls/unlearn_capacity.png" class="img-fluid rounded z-depth-1" %}
    </div>
</div>
<div class="caption">
  An illustration of how unlearning “overdraws” the ability to learn in the future. 
</div>

**Implication**: Fake unlearning diminishes the model’s effective capacity, limiting its potential for long-term adaptability.


### Key Takeaway

Fake unlearning undermines the fundamental goals of machine unlearning. By merely hiding knowledge rather than removing it, the model remains vulnerable to attacks, structural changes, and capacity limitations. For machine unlearning to be truly effective, robust and authentic methods must be developed to address these vulnerabilities.

---


## Pitfall 2 - Over-Relying on the Proposed Forget Dataset

Machine unlearning often hinges on a pre-defined forget dataset, which contains data identified as problematic—whether due to copyright issues, harmful knowledge, or ethical concerns. While this dataset provides a clear target for unlearning, the question remains: is directly using the forget set always the optimal solution?

This approach, though straightforward, can lead to severe consequences, undermining the efficacy and utility of unlearning efforts. Below, we delve into two significant issues.


### Consequence 1 - Unnecessary Unlearning of Poorly Grasped Knowledge

Not all knowledge in the forget set is equally represented in the pretrained model. In many cases, the model may not have fully grasped certain samples from the forget dataset. Despite this, current practices assume that every sample in the forget set must be unlearned, leading to inefficiencies.

Unlearning poorly grasped knowledge can be counterproductive. If the model exhibits low confidence in predicting certain forget set samples, their impact on the model’s behavior is likely negligible. Treating all forget set samples equally wastes resources and may unnecessarily degrade the model's overall utility.

The figure below illustrates this issue. Our preliminary study reveals that the pretrained Zephyr-7B model does not grasp all samples in the WMDP <d-cite key="li2024wmdp"></d-cite> forget set equally. Regions with a low Min-K=20% score (marked in orange) indicate data that are poorly learned by the model. This suggests that unlearning these samples may lead to an unnecessary utility drop without meaningful gains in unlearning effectiveness.

<div class="row mt-3">
    <div class="col-sm mt-3 mt-md-0">
        {% include figure.html path="assets/img/2025-04-28-unlearning-pitfalls/min_k_score_distribution.png" class="img-fluid rounded z-depth-1" %}
    </div>
</div>
<div class="caption">
  The distribution of the Min-K=20% score <d-cite key='zhang2024min'></d-cite> (memorization level) for the pretrained Zephyr-7B-beta <d-cite key='tunstall2023zephyr'></d-cite> with respect to the WMDP forget set. Lower scores indicate data better grasped by the model, while higher scores reflect poorly learned samples.
</div>

To address this, we explored a selective unlearning approach by omitting forget set samples based on their prediction confidence. As shown in the figure below, excluding low-confidence samples significantly improved the trade-off between forgetting harmful knowledge and preserving model utility. By tuning hyperparameters to exclude such samples, we observed a substantial reduction in utility drop while maintaining unlearning effectiveness.

**Implication**: A selective approach—targeting only well-grasped knowledge in the forget set—leads to more effective and efficient unlearning, avoiding unnecessary utility loss.


### Consequence 2 - Neglecting an Unlearning Curriculum

When children learn new concepts, they begin with simpler ideas and gradually progress to more complex ones. This approach, known as a curriculum, allows for efficient and effective learning. Similarly, in the context of machine unlearning, adopting a curriculum-based approach could significantly enhance the process. However, this strategy has been largely overlooked.

#### The Current Problem

Existing unlearning methodologies treat all forget set samples equally. They apply uniform unlearning schedules and hyperparameters, failing to account for the varying levels of "unlearning difficulty" inherent in different samples. This lack of differentiation reduces the overall efficiency and effectiveness of the unlearning process.

#### A Possible Solution - Introducing an Unlearning Curriculum

A curriculum-based strategy prioritizes the removal of well-grasped knowledge (samples with high prediction confidence) before addressing less well-grasped knowledge (low prediction confidence). This incremental approach aligns with the model's internal state and optimizes the unlearning process. In our experiments:
- We defined **unlearning difficulty** based on the model’s prediction confidence for forget set samples.
- We designed an **unlearning curriculum** that starts with well-grasped knowledge and progressively moves to less well-grasped knowledge.
- This approach achieved superior unlearning performance while maintaining overall model utility.

**Implication**: Incorporating an unlearning curriculum improves both the efficiency and effectiveness of the process, enabling a more tailored and impactful approach to model refinement.

### Key Takeaway

Over-relying on predefined forget datasets can lead to unnecessary unlearning of poorly grasped knowledge and inefficiencies in the process. By adopting selective unlearning strategies and curriculum-based approaches, it is possible to improve unlearning effectiveness while preserving model utility and avoiding unnecessary degradation.

---

## Pitfall 3 - Overlooking the Broader Side Effects of Machine Unlearning

When evaluating the impact of machine unlearning, researchers often focus solely on the utility of the unlearned model for general knowledge-based datasets. While this is an essential metric, it provides an incomplete picture. Machine unlearning can have broader, often overlooked side effects, significantly affecting the model’s emergent abilities<d-cite key="wei2022emergent"></d-cite>, reasoning skills<d-cite key="wei2022chain"></d-cite>, and hallucination tendencies<d-cite key="ji2023survey"></d-cite>.

---

### Issue 1 - Impairment of Emergent Abilities and Reasoning

Emergent abilities<d-cite key="wei2022emergent"></d-cite> refer to the complex capabilities that arise as a result of scaling large language models (LLMs), such as in-context learning, augmented prompting, and reasoning. These abilities are widely regarded as critical characteristics of LLMs, enabling tasks like multi-step reasoning, contextual understanding, and advanced problem-solving. However, their vulnerability to unlearning side effects has been largely overlooked in evaluations.

Machine unlearning can disrupt these advanced abilities, especially when reasoning or other emergent skills rely on entangled knowledge connected to the forget set. Our preliminary study highlights this disruption by evaluating the original Zephyr-7B-Beta model and its unlearned version (using the NPO algorithm) on the WMDP dataset across tasks that represent emergent abilities. These include:
- **Reasoning/In-context learning tasks**: Assessed through MMLU (5-shot)<d-cite key="hendrycks2020measuring"></d-cite> and TruthfulQA<d-cite key="lin2021truthfulqa"></d-cite>, which require multi-step inference, contextual understanding, and distinguishing truth from plausible-sounding falsehoods.
- **Augmented prompting tasks**: Evaluated through GPQA (0-shot chain-of-thought)<d-cite key="rein2023gpqa"></d-cite> and IFEval (0-shot) <d-cite key="zhou2023instruction"></d-cite>, which test the model's ability to generate coherent and logical responses based on extended prompts.

The results, summarized in the table below, reveal significant degradation in emergent abilities after unlearning:

|   Model     | MMLU (5-Shot)   | Truthful QA   | GPQA 0-shot CoT   | IFEval 0-shot   |
|:---------:  |:-------------:  |:-----------:  |:---------------:  |:-------------:  |
|  Original   |     59.82%      |    38.80%     |      11.83%       |     54.39%      |
| Unlearned   |     57.25%      |    34.27%     |      5.36%        |     27.94%      |

The unlearned model consistently underperformed across all tasks, with particularly severe impacts on augmented prompting tasks like GPQA and IFEval. Reasoning tasks, while somewhat less affected, still showed a noticeable decline, underscoring the interconnected nature of these abilities.

**Implication**: The interconnected nature of knowledge in LLMs means that unlearning targeted data can unintentionally disrupt critical emergent abilities, including reasoning and advanced prompting. This underscores the need for careful evaluation and mitigation strategies to preserve these essential capabilities.


### Issue 2 - Increased Hallucinations

Hallucinations, where the model generates incorrect or nonsensical information, are a persistent challenge in large language models (LLMs). Machine unlearning, by altering the model’s learned representations, can exacerbate this issue, destabilizing the model's behavior and increasing the frequency of hallucinations.

To investigate this concern, we conducted a preliminary study using the Zephyr-7B-Beta model and evaluated its hallucination rate before and after unlearning on the Truthful QA dataset. Unlearning was performed with the NPO method on the WMDP dataset. Before unlearning, the model achieved an accuracy rate of **38.80%**. After unlearning, the accuracy dropped to **34.27%**, indicating an increase in factual inaccuracies in general knowledge by over 4%.


<div class="row mt-3">
    <div class="col-sm mt-3 mt-md-0">
        {% include figure.html path="assets/img/2025-04-28-unlearning-pitfalls/hallucination.png" class="img-fluid rounded z-depth-1" %}
    </div>
</div>
<div class="caption">
  An illustration of how unlearning exacerbates hallucination rates in LLMs, destabilizing their performance on tasks like Truthful QA.
</div>

**Implication**: Poorly implemented unlearning strategies can destabilize the model, leading to more frequent and severe hallucinations. This highlights the importance of evaluating unlearning methods beyond traditional utility metrics to address their impact on model reliability.

### Key Takeaway

Overlooking the broader side effects of machine unlearning can significantly impair a model's emergent abilities and reasoning skills, while also increasing hallucination rates. Since knowledge in large language models is highly interconnected, unlearning targeted data may unintentionally disrupt critical capabilities. To preserve model utility and reliability, unlearning strategies must carefully evaluate and mitigate these unintended consequences.

---

## Conclusion

Machine unlearning holds immense potential to ensure the ethical and effective deployment of foundation models, yet its current methodologies face significant challenges. We have highlighted three critical pitfalls: the prevalence of fake unlearning, over-reliance on the forget dataset, and the neglect of broader side effects. Fake unlearning undermines robustness and reduces model capacity, while naive reliance on forget datasets leads to unnecessary unlearning and ignores the benefits of curriculum-based approaches. Additionally, unlearning can impair emergent abilities, reasoning skills, and hallucination tendencies if its broader effects are not carefully evaluated.

Addressing these pitfalls requires a paradigm shift in the design and evaluation of machine unlearning techniques. By incorporating selective and curriculum-based approaches, alongside comprehensive assessments of emergent and reasoning abilities, researchers can build more reliable and adaptive models. This blog aims to inspire the community to explore these overlooked aspects, paving the way for machine unlearning to become a cornerstone of trustworthy and efficient AI systems.

## Citations

Citations are then used in the article body with the `<d-cite>` tag.
The key attribute is a reference to the id provided in the bibliography.
The key attribute can take multiple ids, separated by commas.

The citation is presented inline like this: <d-cite key="gregor2015draw"></d-cite> (a number that displays more information on hover).
If you have an appendix, a bibliography is automatically created and populated in it.

Distill chose a numerical inline citation style to improve readability of citation dense articles and because many of the benefits of longer citations are obviated by displaying more information on hover.
However, we consider it good style to mention author last names if you discuss something at length and it fits into the flow well — the authors are human and it’s nice for them to have the community associate them with their work.

***

## Footnotes

Just wrap the text you would like to show up in a footnote in a `<d-footnote>` tag.
The number of the footnote will be automatically generated.<d-footnote>This will become a hoverable footnote.</d-footnote>

***

## Code Blocks

This theme implements a built-in Jekyll feature, the use of Rouge, for syntax highlighting.
It supports more than 100 languages.
This example is in C++.
All you have to do is wrap your code in a liquid tag:

{% raw  %}
{% highlight c++ linenos %}  <br/> code code code <br/> {% endhighlight %}
{% endraw %}

The keyword `linenos` triggers display of line numbers. You can try toggling it on or off yourself below:

{% highlight c++ %}

int main(int argc, char const \*argv[])
{
string myString;

    cout << "input a string: ";
    getline(cin, myString);
    int length = myString.length();

    char charArray = new char * [length];

    charArray = myString;
    for(int i = 0; i < length; ++i){
        cout << charArray[i] << " ";
    }

    return 0;
}

{% endhighlight %}

***

## Diagrams

This theme supports generating various diagrams from a text description using [jekyll-diagrams](https://github.com/zhustec/jekyll-diagrams){:target="\_blank"} plugin.
Below, we generate a few examples of such diagrams using languages such as [mermaid](https://mermaid-js.github.io/mermaid/){:target="\_blank"}, [plantuml](https://plantuml.com/){:target="\_blank"}, [vega-lite](https://vega.github.io/vega-lite/){:target="\_blank"}, etc.

**Note:** different diagram-generation packages require external dependencies to be installed on your machine.
Also, be mindful of that because of diagram generation the first time you build your Jekyll website after adding new diagrams will be SLOW.
For any other details, please refer to [jekyll-diagrams](https://github.com/zhustec/jekyll-diagrams){:target="\_blank"} README.

**Note:** This is not supported for local rendering! 

The diagram below was generated by the following code:

{% raw %}
```
{% mermaid %}
sequenceDiagram
    participant John
    participant Alice
    Alice->>John: Hello John, how are you?
    John-->>Alice: Great!
{% endmermaid %}
```
{% endraw %}

{% mermaid %}
sequenceDiagram
participant John
participant Alice
Alice->>John: Hello John, how are you?
John-->>Alice: Great!
{% endmermaid %}

***

## Tweets

An example of displaying a tweet:
{% twitter https://twitter.com/rubygems/status/518821243320287232 %}

An example of pulling from a timeline:
{% twitter https://twitter.com/jekyllrb maxwidth=500 limit=3 %}

For more details on using the plugin visit: [jekyll-twitter-plugin](https://github.com/rob-murray/jekyll-twitter-plugin)

***

## Blockquotes

<blockquote>
    We do not grow absolutely, chronologically. We grow sometimes in one dimension, and not in another, unevenly. We grow partially. We are relative. We are mature in one realm, childish in another.
    —Anais Nin
</blockquote>

***


## Layouts

The main text column is referred to as the body.
It is the assumed layout of any direct descendants of the `d-article` element.

<div class="fake-img l-body">
  <p>.l-body</p>
</div>

For images you want to display a little larger, try `.l-page`:

<div class="fake-img l-page">
  <p>.l-page</p>
</div>

All of these have an outset variant if you want to poke out from the body text a little bit.
For instance:

<div class="fake-img l-body-outset">
  <p>.l-body-outset</p>
</div>

<div class="fake-img l-page-outset">
  <p>.l-page-outset</p>
</div>

Occasionally you’ll want to use the full browser width.
For this, use `.l-screen`.
You can also inset the element a little from the edge of the browser by using the inset variant.

<div class="fake-img l-screen">
  <p>.l-screen</p>
</div>
<div class="fake-img l-screen-inset">
  <p>.l-screen-inset</p>
</div>

The final layout is for marginalia, asides, and footnotes.
It does not interrupt the normal flow of `.l-body`-sized text except on mobile screen sizes.

<div class="fake-img l-gutter">
  <p>.l-gutter</p>
</div>

***

## Other Typography?

Emphasis, aka italics, with *asterisks* (`*asterisks*`) or _underscores_ (`_underscores_`).

Strong emphasis, aka bold, with **asterisks** or __underscores__.

Combined emphasis with **asterisks and _underscores_**.

Strikethrough uses two tildes. ~~Scratch this.~~

1. First ordered list item
2. Another item
⋅⋅* Unordered sub-list. 
1. Actual numbers don't matter, just that it's a number
⋅⋅1. Ordered sub-list
4. And another item.

⋅⋅⋅You can have properly indented paragraphs within list items. Notice the blank line above, and the leading spaces (at least one, but we'll use three here to also align the raw Markdown).

⋅⋅⋅To have a line break without a paragraph, you will need to use two trailing spaces.⋅⋅
⋅⋅⋅Note that this line is separate, but within the same paragraph.⋅⋅
⋅⋅⋅(This is contrary to the typical GFM line break behavior, where trailing spaces are not required.)

* Unordered lists can use asterisks
- Or minuses
+ Or pluses

[I'm an inline-style link](https://www.google.com)

[I'm an inline-style link with title](https://www.google.com "Google's Homepage")

[I'm a reference-style link][Arbitrary case-insensitive reference text]

[I'm a relative reference to a repository file](../blob/master/LICENSE)

[You can use numbers for reference-style link definitions][1]

Or leave it empty and use the [link text itself].

URLs and URLs in angle brackets will automatically get turned into links. 
http://www.example.com or <http://www.example.com> and sometimes 
example.com (but not on Github, for example).

Some text to show that the reference links can follow later.

[arbitrary case-insensitive reference text]: https://www.mozilla.org
[1]: http://slashdot.org
[link text itself]: http://www.reddit.com

Here's our logo (hover to see the title text):

Inline-style: 
![alt text](https://github.com/adam-p/markdown-here/raw/master/src/common/images/icon48.png "Logo Title Text 1")

Reference-style: 
![alt text][logo]

[logo]: https://github.com/adam-p/markdown-here/raw/master/src/common/images/icon48.png "Logo Title Text 2"

Inline `code` has `back-ticks around` it.

```javascript
var s = "JavaScript syntax highlighting";
alert(s);
```
 
```python
s = "Python syntax highlighting"
print(s)
```
 
```
No language indicated, so no syntax highlighting. 
But let's throw in a <b>tag</b>.
```

Colons can be used to align columns.

| Tables        | Are           | Cool  |
| ------------- |:-------------:| -----:|
| col 3 is      | right-aligned | $1600 |
| col 2 is      | centered      |   $12 |
| zebra stripes | are neat      |    $1 |

There must be at least 3 dashes separating each header cell.
The outer pipes (|) are optional, and you don't need to make the 
raw Markdown line up prettily. You can also use inline Markdown.

Markdown | Less | Pretty
--- | --- | ---
*Still* | `renders` | **nicely**
1 | 2 | 3

> Blockquotes are very handy in email to emulate reply text.
> This line is part of the same quote.

Quote break.

> This is a very long line that will still be quoted properly when it wraps. Oh boy let's keep writing to make sure this is long enough to actually wrap for everyone. Oh, you can *put* **Markdown** into a blockquote. 


Here's a line for us to start with.

This line is separated from the one above by two newlines, so it will be a *separate paragraph*.

This line is also a separate paragraph, but...
This line is only separated by a single newline, so it's a separate line in the *same paragraph*.
