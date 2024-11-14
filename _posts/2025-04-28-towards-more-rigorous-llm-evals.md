---
layout: distill
title: "Towards more rigorous evaluations of language models"
description: "As language models become increasingly sophisticated and existing benchmarks approach saturation, the need for rigorous evaluation methods grows more pressing. Many evaluations lack the statistical rigour needed to draw meaningful conclusions, leading to a potential over-confidence in results that might not hold up under scrutiny or replication. This post advocates for bringing fundamental statistical principles to language model evaluation, demonstrating how basic statistical analysis can provide more reliable insights into model capabilities and limitations."

authors:
- anonymous

# must be the exact same name as your blogpost
bibliography: 2025-04-28-towards-more-rigorous-llm-evals.bib  

toc:
  - name: 1. Introduction
  - name: 2. Elements of Rigorous Empirical Evaluation
  - name: 3. Summary of Mirzadeh et al. (2024)
    subsections:
    - name: 3.1 What is the new benchmark and how is it constructed?
    - name: 3.2 What are the key findings and conclusions from the empirical evaluation?
  - name: 4. Critical Analysis and Re-evaluation
    subsections:
    - name: "4.1 Performance variability: Why is variability not (that) interesting?"
      subsections:
      - name: 4.1.1 When is variability not expected?      
      - name: 4.1.2 When is variability expected?
    - name: 4.2 Performance decline on GSM-Symbolic    
      subsections:
      - name: "4.2.1 Alternative explanation: Distribution mismatch"
      - name: "4.2.2 Considering each model independently: Is the decline in performance statistically significant?"
      - name: "4.2.3 Considering all models together: Is the decline in performance statistically significant?"
    - name: 4.3 Performance decrease and variance increase with question complexity
    - name: 4.4 M1, P1, P2 and No-Op results
  - name: 5. Conclusion
---


# 1. Introduction

A staggering volume of research papers on (large) language models (LMs) is published daily. 
On the day of writing this (5th Nov 2024), 239 papers containing "LLM" or "Language Model" in their titles were 
added to the Computer Science section of arXiv alone.
This flood of publications makes it difficult to separate genuine insights from noise, especially in areas lacking precise definitions, such as "reasoning". 
Given the empirical nature of LLM research, much is open to interpretation, making rigorous 
analysis crucial to reduce bias and improve the reliability of findings.


In this post, we argue that LM researchers---especially those working in areas where core concepts lack established definitions---must adopt statistical methods to improve the rigor of their empirical evaluations.
Leveraging techniques from classical statistics, such as confidence intervals and hypothesis tests, will help move the field beyond anecdotal observations and philosophical arguments toward a more scientific understanding of model behavior.

To illustrate this in practice, we outline three key elements of rigorous empirical evaluation and apply them to [Mirzadeh et al. (2024)](https://arxiv.org/pdf/2410.05229) <d-cite key="mirzadeh2024gsm"></d-cite>---a recent paper that examines whether LMs perform "formal reasoning" or rely on "sophisticated pattern matching".

# 2. Elements of Rigorous Empirical Evaluation

1. Clear articulation of assumptions and consideration of alternative explanations.
2. Quantification of uncertainty in results through appropriate statistical measures.
3. Careful consideration of train-test overlap, with reasonable attempts to evaluate on "out-of-sample" datasets when possible.

> If one does not give thought to what the data would be like under the assumption that one’s theory is false, one is likely reinforcing confirmation bias rather than establishing the validity of the theory. *"Reproducibility, p-values, and type III errors: response to Mayo", Philip B. Stark (2022) <d-cite key="stark2022reproducibility"></d-cite>*

The first step in rigorous evaluation is to clarify assumptions and explore alternative explanations.
Experiments can often be designed in ways that unintentionally favour the hypothesis being tested.
By being upfront about assumptions, questioning their validity, and investigating alternative explanations, researchers will improve the reliability and robustness of their conclusions. 

> In some sense it [the p-value] offers a first line of defense against being fooled by randomness, separating signal from noise [...]. *"It’s Not the P-Values’ Fault", Yoav Benjamini (2016) <d-cite key="benjamini2016not"></d-cite>*

A second essential step is quantifying uncertainty, using tools like error bars and confidence intervals, which help us gauge the reliability of performance metrics by providing a range of plausible values.
Additionally, although often criticized, hypothesis tests and p-values can serve as an initial, coarse filter to distinguish signal from noise, laying the groundwork for deeper exploration into the practical significance of the reported results.


Finally, evaluating models on entirely unseen ("out-of-sample") data is essential for accurately assessing their true 
capabilities. 
Unfortunately, identifying the extent of train-test overlap in LMs is both difficult and often neglected. Many language models are not transparent about this issue, and benchmark datasets frequently get *leaked* (or *contaminated*) during training, leading to biased evaluations and inflated performance metrics (<d-cite key="zhang2024language"></d-cite>, Zhang et al. 2024). 
As Zhang et al. (2024) highlight, reporting train-test overlap statistics is crucial for ensuring the validity of model 
evaluations.


To illustrate these three principles, we use [Mirzadeh et al. (2024)](https://arxiv.org/pdf/2410.05229) as a case study---a recent paper that received substantial attention from the LLM research community (e.g. see [this](https://machinelearning.apple.com/research/gsm-symbolic), [this](https://www.reddit.com/r/singularity/comments/1g1zphu/apple_ai_researchers_question_openais_claims/), or [this](https://x.com/MFarajtabar/status/1844456880971858028)).
We review their methods, identify gaps in their analysis, and offer a more rigorous statistical assessment of their claims.


# 3. Summary of [Mirzadeh et al. (2024)](https://arxiv.org/pdf/2410.05229)

Mirzadeh et al. (2024) <d-cite key="mirzadeh2024gsm"></d-cite> make two technical contributions: (1) a new benchmark, called GSM-Symbolic, for evaluating mathematical reasoning of LMs, and (2) an empirical evaluation of 25 LMs on this new benchmark to assess their reasoning capabilities.

## 3.1 What is the new benchmark and how is it constructed?

The authors propose GSM-Symbolic, a variant of the well-established GSM8K benchmark for grade school math word problems <d-cite key="cobbe2021training"></d-cite>. Since GSM8K has likely leaked into many LMs' training sets due to its popularity, GSM-Symbolic aims to match its distribution whilst eliminating (or reducing) train-test overlap.

The authors also construct four variants of GSM-Symbolic by modifying question difficulty:
- GSM-M1: An easier version that removes one clause from each question.
- GSM-P1 and GSM-P2: Progressively harder versions that add one or two extra clauses respectively.
- GSM-NoOp: A version that adds "seemingly relevant but ultimately irrelevant information" that should have no operational significance (No-Op).

To generate GSM-Symbolic and the variants, the authors create "templates" from GSM8K questions by identifying modifiable variables whilst preserving the underlying logic. 
Each variable has a specified domain and constraints to ensure valid questions and answers. Here's an example template:

{% include figure.html 
  path="assets/img/2025-04-28-towards-more-rigorous-llm-evals/template_gsm.png" 
  class="img-fluid" 
  caption="Figure 1 from Mirzadeh et al. (2024) <d-cite key='mirzadeh2024gsm'></d-cite>"
%}

For their analysis, the authors select 100 questions from GSM8K and create such a template for each of them.
Whilst the paper does not specify their selection method, we suppose that these questions were sampled randomly from GSM8K's test set.
By sampling values for the variables, 50 new questions are generated from each template. 
This means that GSM-Symbolic and each of its 4 variants (GSM-M1, GSM-P1, GSM-P2, and GSM-NoOp) contain **50 datasets of 100 samples each**.

Throughout this post, when we refer to "GSM8K", we specifically mean those 100 original GSM8K questions that were used to create the templates, not the full GSM8K dataset.

## 3.2 What are the key findings and conclusions from the empirical evaluation?

The evaluated model families include Gemma, Phi, Mistral and Llama (open weights), and GPT and o1 (proprietary, OpenAI). Of the open weights models, one can be considered "medium" size (Gemma2 27B params), and all the rest can be considered "small" (9B params or less).
The metrics reported are average accuracy across the 50 versions of each dataset and the standard deviation of these accuracies.

The key findings are:

- Performance variability: LMs exhibit some variability in performance across different instantiations of the same question.

- Performance decline: Compared to GSM8K, performance on GSM-Symbolic drops, suggesting potential data contamination.

- Sensitivity to question complexity: As complexity of the questions increases, “the performance [of models] decreases and the variance increases”, which is said to suggest that “models are not performing formal reasoning”, and that the increase in variance is “in line with the pattern-matching hypothesis”.

- Impact of irrelevant information: Introducing a clause of no operational significance (No-Op), leads to large performance degradation across models, suggesting “deeper issues in their reasoning processes”.

The paper concludes that LMs “are not performing formal reasoning”.

# 4. Critical analysis and re-evaluation

## 4.1 Performance variability: Why is variability not (that) interesting?

> As demonstrated, all models show a **significant variance** across different sets. […] It is **noteworthy that this variation even occurs** […]. <d-cite key="mirzadeh2024gsm"></d-cite>, emphasis added.

The authors emphasise the "non-negligible variance" in model performance across different GSM-Symbolic datasets, framing it as surprising. But is it?

### 4.1.1 When is variability *not expected*?

Variability would indeed be unexpected if each resampled question was effectively the same as the original. 
The implicit assumption here is that if an LM solves (or fails to solve) a given question once, it should always solve (or fail to solve) it when presented with the same problem but with different numbers. 
In other words, this assumes that LMs never make arithmetic mistakes---a very strong assumption that is not examined or addressed in the paper. 

**Is this assumption valid?** 
For humans, it clearly is not. 
Even when solving the same problem with different numbers or subjects, humans can make (non-reasoning related) errors, such as arithmetic mistakes or copying a number incorrectly.
The same applies to LMs. 

We demonstrate this empirically by looking at the performance of LMs on a basic addition task with varying **digit lengths** (e.g. "What is 147 + 562?"). 
Consistent with prior literature that examines the impact of digit length on arithmetic performance [TODO CITE], we find that as the numbers get larger, accuracy declines, showing that simple arithmetic is not perfectly reliable.
We go beyond digit length and also consider how the number of **carry operations** affects performance on this addition task. 
The figure below illustrates how the probability of answering a question correctly is affected by the number of digits $d$ of the numbers being added and the number of carry operations involved in the sum.

{% include figure.html 
  path="assets/img/2025-04-28-towards-more-rigorous-llm-evals/addition_accuracy.png" 
  class="img-fluid" 
  title="Accuracy of Llama3-8b and Phi-3.5-mini-instruct on a simple addition task" 
  caption="Accuracy of Llama3-8b and Phi-3.5-mini-instruct on a simple addition task of adding two $d$-digit numbers. 
  The plot illustrates how the probability of answering a question correctly ($y$-axis) is affected by the total number of digits involved ($2d$, $x$-axis), and the total number of carry operations involved in that sum (colour of the points).  
  Point size reflects the total number of tokens (for Phi, total digits equal total tokens; for Llama, numbers up to 3 digits are 1 token and up to 6 digits are 2 tokens). 
  For this illustration, we group questions by number of digits and carry operations, and plot the average accuracy over 512 samples. 
  Groups containing fewer than 10 questions are excluded. 
  Detailed results of the logistic regressions are available in the Appendix." 
%}

The findings indicate that LM performance is negatively affected by both the number of digits and the number of carry operations in the sum. 
I.e. as the numbers get larger and the number of carry operations increases, accuracy declines.
This suggests that LMs make errors similar to those made by humans, such as forgetting carry operations. 
Detailed regression results can be found in the Appendix.

<!-- Model | 1 digit | 2 digits | 3 digits | 4 digits | 5 digits | 6 digits
--- | --- | --- | --- | --- | --- | ---
Phi-3.5-mini-instruct | 100% | 93.0% | 90.8% | 84.0% | 79.7% | TODO
Llama-3-8B-Instruct | 100% | 100% | 100% | 95.3% | 91.2% | 86.3%

<div class="caption">
Accuracy (zero-shot) on a simple addition task. The larger model (Llama-3-8B-Instruct) is more accurate. For Llama, numbers upto 3 digits are a single token, and 4-digit numbers are 2 tokens. For Phi, a $d$-digit number takes $d$ tokens. 
</div> -->

**Is performing arithmetic part of reasoning?** 
Solving a word math problem consists of two steps: (1) translating the text to a sequence of operations, and (2) performing the operations correctly.
Whilst the first step clearly requires reasoning ability, we argue that the second is more mechanical in nature and should not be considered as part of reasoning.
To isolate "pure reasoning" capabilities, models could be provided with a calculator, which would help reduce (though not completely eliminate) the confounding effect of arithmetic errors.

### 4.1.2 When is variability *expected*?

If we (rightfully) reject the assumption that LMs are infallible in arithmetic, performance variability across datasets becomes entirely expected.
Even if a model applies consistent reasoning, it may still make arithmetic errors, leading to natural performance variation.

**How much variation is expected?**  
By now, it should be clear that the answer to this question depends on the assumptions we make about the data.
One reasonable assumption we could make is that each model $$m=1, \dots, 25$$, answers each question $$n=1,\dots,100$$ correctly with some probability $p_{m,n}$.
Unfortuantely, since the paper does not provide question-level performance data, for the purposes of this analysis, we must further assume that this probability is constant across questions, that is $p_{m,n}=p_m$ for all $n$. 

Thus, an LM answering questions is modelled as an independent and identically distributed Bernoulli trial with a model-specific success probability $p_m$.
Under this assumption, the total number of correct answers on a dataset of size $N=100$ is

$$\text{Binomial}(N, p_m).$$

The variance of this distribution is **fully determined by the success probability $p_m$** and equals 

$$N \cdot p_m \cdot (1-p_m).$$

This variance is maximised when $p_m=1/2$ and goes to $0$ as $p_m$ approaches $0$ or $1$.

To understand what "expected" variation means under our assumption, we can construct confidence intervals (CIs) for the point estimates of $p_m$ on the GSM8K dataset. 
We highlight that the paper does not report such point estimates, but we can calculate the maximum likelihood estimates of $p_m$ by dividing the number of correct answers by the total number of questions.
The number of correct answers (out of 100 questions) on the GSM8K questions are reported in the second column of Table 1 in the Appendix of the paper. 
We denote this estimate as $p_{m,8K}$ to indicate that it is computed from the GSM8K dataset.
There are different ways to construct CI for the [Binomial proportion](https://en.wikipedia.org/wiki/Binomial_proportion_confidence_interval). The next figure shows [Wilson score](https://en.wikipedia.org/wiki/Binomial_proportion_confidence_interval#Wilson_score_interval) intervals, with more results included in the Appendix. 
To put this variability into perspective, we also include the average of the 50 point estimates for the model performance on GSM-Symbolic, which we denote as $p^{Symb}_{m}$. <d-footnote>Similarly to $p^{8K}_{m}$, we obtain maximum likelihood estimates of $p^{Symb}_{m}$ from the average accuracy on GSM-Symbolic, reported in Table 1 of the paper.</d-footnote>

{% include figure.html 
  path="assets/img/2025-04-28-towards-more-rigorous-llm-evals/wilson_0.95.png" 
  class="img-fluid" 
  title="95% Wilson score confidence intervals" 
  caption="95% Wilson score confidence intervals for the point estimates of $p^{8K}_{m}$ (red dots), along with the average (over 50 datasets) point estimate of $p^{Symb}_{m}$ (blue triangles)." 
%}
<!-- <div class="caption">
  "95% Wilson score intervals for the point estimates of $p_m$.
</div> -->


As expected, models with success probabilities closer to $1/2$ (e.g. Gemma-7b, Phi-2, Mistral-7b-v0.1) exhibit wider confidence intervals, reflecting higher variability. 
Conversely, models with success probabilities closer to 0 or 1 (Gemma2b, GPT-4o, o1-preview) have substantially narrower intervals.

**Assuming** that GSM8K and GSM-Symbolic come from the same distributions (more on that in Section 4.2.1), let’s look at Figure 2 of the paper.

{% include figure.html 
  path="assets/img/2025-04-28-towards-more-rigorous-llm-evals/fig2_gsm.png" 
  class="img-fluid" 
  caption="Figure 2 from Mirzadeh et al. (2024) <d-cite key='mirzadeh2024gsm'></d-cite>. Note that the $x$-axis scales are different for different models."
%}

For the models shown in this figure, the GSM8K accuracy ($p^{8K}_{m}$, represented by the dashed line) varies from 74% for the weakest model, Llama3-8B-instruct, to 95% for the strongest model, GPT-4o. 
The range of accuracies achieved on the 50 GSM-Symbolic datasets is relatively *wide* for Llama3-8B-instruct (approximately between 69% and 81%) and relatively *narrow* for GPT-4o (approximately between 91% and 98%).
Importantly, for both models, **the variation in GSM-Symbolic performance falls well within the Wilson score CIs of GSM8K performance that we calculated earlier!** 
We visualise this in the next figure, showing the overlap between the 95% Wilson score CIs for $p^{8K}_{m}$ and the accuracy ranges on GSM-Symbolic for the models that had results reported in the paper (note that this does not include all 25 models).

<!-- 

| Model                          | 95% Wilson score CI | Reported ranges (approximate)  |
|--------------------------------|---------------------|-----------------------|
| Gemma2-2b-it                   | (37%, 56%)          | (34%, 48%)            |
| Gemma2-9b-it                   | (79%, 92%)          | (71%, 85%)            |
| Phi-2                          | (43%, 62%)          | (35%, 50%)            |
| Phi-3-mini-128k-instruct       | (77%, 91%)          | (75%, 90%)            |
| Phi-3-medium-128k-instruct     | (81%, 94%)          | (75%, 84%)            |
| Mistral-7b-instruct-v0.1       | (33%, 52%)          | (23%, 38%)            |
| Mathstral-7b-v0.1              | (71%, 87%)          | (71%, 81%)            |
| Llama3-8b-instruct             | (65%, 82%)          | (69%, 81%)            |
| GPT-4o                         | (89%, 98%)          | (91%, 98%)            |
| o1-mini                        | (86%, 97%)          | (90%, 97%)            |
| o1-preview                     | (90%, 98%)          | (88%, 96%)            |

<div class="caption">
95% Wilson score intervals for the point estimates of $p^{8K}_{m}$ and approximate reported ranges of point estimates of $p^{Symb}_{m}$, derived from Figure 1 in Mirzadeh et al. (2024) <d-cite key='mirzadeh2024gsm'></d-cite>, as well as Figures 10 and 12 from the Appendix of the paper.
</div> -->


{% include figure.html 
  path="assets/img/2025-04-28-towards-more-rigorous-llm-evals/ci_vs_reported.png" 
  class="img-fluid" 
  caption="95% Wilson score confidence intervals for the point estimates of GSM8K accuracy, $p^{8K}_{m}$ (red), and range of accuracies achieved on the 50 GSM-Symbolic datasets, $p^{Symb}_{m}$ (blue). 
  The latter ranges are not explicitly reported; we approximate them from the histograms in Figure 1 of Mirzadeh et al. (2024) <d-cite key='mirzadeh2024gsm'></d-cite>, as well as Figures 10 and 12 from the Appendix of the paper. 
  We only show the subset of the models for which such histograms are available in the paper." 
%}

Note that our confidence intervals tend to be wider than the implied ranges in the figures in the paper, i.e. under the i.i.d. Bernoulli assumption, the expected variation is actually **larger** than what is observed.
This discrepancy is likely to be explained by the unmodelled correlations between questions---as initially suggested, a more reasonable assumption would be to model the probability of success on a question level, $p_{m,n}$, rather than assuming each question is equally likely to be answered correctly. 
The analysis can be repeated once (if) the detailed question-level data becomes available.

**Verdict:** The observed variability in GSM-Symbolic performance is not inherently surprising, and is in fact expected.
<!-- ili3p wrote: An idea I had here, how is the variability in the "adding two numbers" task? Are the results also within the 95% CI of assuming Binomial distribution?  TODO: @DRI CHECK-->

## 4.2 Performance decline on GSM-Symbolic

The paper claims that LMs perform worse on GSM-Symbolic compared to GSM8K. 
Let's examine the evidence presented in Section 4.1, which we quote directly:

> Another noteworthy observation is that the performance (represented by the dashed line in Fig. 2) on the original questions from the 100 examples of GSM8K used as templates is **often more than one standard deviation away from the center** of the GSM-Symbolic performance distribution, frequently on the right side of the distribution (this holds for 21 out of 25 models). **One explanation** for this could be data contamination […]. <d-cite key='mirzadeh2024gsm'></d-cite>, emphasis added.

There are two issues with the above quote. 
First, the authors suggest data contamination as one possible explanation for the performance decline, but do not explore other plausible explanations.
Second, they rely on a hand-wavy "one standard deviation" criterion to suggest that the decline in performance is significant, without proper statistical analysis. 
We address both of these next. 

### 4.2.1 Alternative explanation: Distribution mismatch

In addition to data contamination, another plausible explanation for the alleged performance discrepancy is a distribution mismatch between GSM8K and GSM-Symbolic. 
We note that the two explanations are not mutually exclusive---both can be true at the same time and should be evaluated appropriately.

The paper does not examine or discuss the potential distribution mismatch, making it impossible to draw definitive conclusions without access to the templates used.
However, there is evidence indicating that the distribution of GSM-Symbolic questions differs from that of GSM8K, and that GSM-Symbolic is indeed more difficult.
Looking at the example template (Figure 1 from the paper, reproduced above), we see that the sampling ranges for some variables **exclude** the original GSM8K values:
- The variable `total` is sampled from $(100, 500)$, whilst in the original question we have `total=62`.
- The variable `ans` is sampled from $(85, 200)$, whilst in the original question we have `ans=14`.

In other words, the original GSM8K question cannot be generated from the proposed symbolic template.
We suggest more appropriate ranges for the variables and quantify impact of that choice on the accuracy of the arithmetic operations.
A suitable sampling range for `total` might be $(10, 100)$: It includes the original (62) and aligns with the context that `total` represents the number of toys (500 toys seems rather large). 
Assuming `y` (stuffed animals) and `z` (stacking rings) are on the same scale and in the range $(1, 20)$;
and that `x` (building blocks) and `ans` (bouncy balls) are also on the same scale and are so that the total is in the specified range, i.e. $(4, 40)$;
These ranges actually include the original values (9 rings, 8 stuffed animals, 31 building blocks, 14 bouncy balls, total of 62):


We propose more suitable ranges for all variables, ensuring the original template can be reproduced and assess how these choices affect the accuracy of arithmetic operations, assuming the reasoning is correct.

| Variable              | Symbolic range | Proposed range | Original value |
|-----------------------|----------------|----------------|----------------|
| `total` (toys)        | $(100, 500)$   | $(10, 100)$    | 62             |
| `x` (building blocks) | $(5, 100)$     | $(4, 40)$      | 31             |
| `y` (stuffed animals) | $(5, 100)$     | $(1, 20)$      | 8              |
| `z` (stacking rings)  | $(5, 100)$     | $(1, 20)$      | 9              |
| `ans` (bouncy balls)  | $(85, 200)$    | $(4, 40)$      | 14             |



The question in Figure 1 involves three arithmetic operations (two additions and one subtraction).
Assuming subtraction is as hard as addition, the probability of the three operations being answered correctly is the product of the individual probabilities. 

As we saw in Section 4.1.1, the accuracy of both models decreases as the number of digits increases.
We use the logistic regression model from that section try quantify the difference in accuracy that might arise from using the ranges in the paper vs those we propose here (the "reasoning" that gets us to the correct mathematical expression is correct.)<d-footnote>we compute sums; TODO</d-footnote>

Symbolic:
| Model | p1       | p2       | p3       | p        |
|-------|----------|----------|----------|----------|
| Phi   | 94.9 | 94.1 | 92.0 | 82.1 |
| Llama | 99.6 | 99.5 | 99.1 | 98.2 |


Proposal
| Model | p1       | p2       | p3       | p        |
|-------|----------|----------|----------|----------|
| Phi   | 95.5 | 95.4 | 95.0 | 86.6 |
| Llama | 99.7 | 99.7 | 99.6 | 99.0 |



If the number ranges in GSM-Symbolic are systematically chosen to be larger than those in GSM8K (and don't even include the original question values), then it cannot claimed that the datasets come from the same distribution.
Tokenisation is one mechansim that explains why this matters; larger number ranges in GSM-Symbolic may inherently disadvantage certain (and eventually all) models, potentially explaining some of the observed performance differences between models and datasets.

> We have also observed the performance of LLMs deteriorating as question complexity increases.

Plausible that the reasoning is harder when more clauses are introduced.
worth noting that adding more clauses involves more arithmetic operations; 
more thorough analysis is needed to control for that variability.

The same analysis is applicable to more complex questions; e.g. adding one extra clause, even if it takes only one operation --> similalry for 2 clauses;

Repeat reasoning: translating the text to a sequence of operations; what the paper tests is whether models can do that *and* perofrm the operations correctly. The rest of this post will not deal with reasoning; 

**Verdict:** Some indication that distributions might differ. Contamination is not mutually exclusive. TODO


### 4.2.2 Considering each model independently: Is the decline in performance statistically significant?

For the purpose of this analysis, let's **assume** that GSM8K and GSM-Symbolic come from the same distribution.

For many models in Figure 2, the dashed line is in the right tail of the distribution. Additionally, Figure 3 of the paper, reproduced below, reports substantial performance decrease for many other models. So is the performance decline statistically significant, or could it be attributed to normal variation?

<!-- {{< figure library="true" src="assets/img/2025-04-28-towards-more-rigorous-llm-evals/fig3_gsm.png" title="Figure from Mirzadeh et al. (2024) https://arxiv.org/pdf/2410.05229." numbered="false">}} -->
{% include figure.html 
  path="assets/img/2025-04-28-towards-more-rigorous-llm-evals/fig3_gsm.png" 
  class="img-fluid" 
  caption="Figure 3 from Mirzadeh et al. (2024) <d-cite key='mirzadeh2024gsm'></d-cite>"
%}

The right tool to determine whether these differences are statistically significant is hypothesis testing.
For each model $m$, we want to test whether its success probability on GSM8K, denoted $p^{8K}_{m}$, equals its success probability on GSM-Symbolic, denoted $p^{Symb}_{m}$. 
This equality forms our *null hypothesis*. 
Our *alternative hypothesis* can take two forms:

- Two-sided: The success probabilities are different
$$
  H_0: p^{8K}_{m} = p^{Symb}_{m} \quad\quad\quad H^\text{two-sided}_A: p^{8K}_{m} \neq p^{Symb}_{m}.
$$
- One-sided: The success probability on GSM8K is greater than that on GSM-Symbolic
$$
  H_0: p^{8K}_{m} = p^{Symb}_{m} \quad\quad\quad H^\text{one-sided}_A: p^{8K}_{m} > p^{Symb}_{m}.
$$

We use Fisher exact test for the binomial proportion for all models:

<!-- {{< figure library="true" src="assets/img/2025-04-28-towards-more-rigorous-llm-evals/fisher_pvalues.png" title="Fisher exact test: p-values for two-sided and one-sided tests across models. We use (*) to indicate models for which the null can be rejected at the 5% significance level in favour of the two-sided alternative." numbered="false">}} -->
{% include figure.html 
  path="assets/img/2025-04-28-towards-more-rigorous-llm-evals/fisher_pvalues.png"
  class="img-fluid" 
  title="Fisher exact test" 
  caption="Fisher exact test: p-values for two-sided and one-sided tests across models. We use (*) to indicate models for which the null can be rejected at the 5% significance level in favour of the two-sided alternative." 
%}

At the $5\%$ significance level, we see that there are 4 models for which we are able to reject the null: Gemma-7b, Mistral-7b-instruct-v0.1, Phi-2 and Llama3-8b. 
Note that the performance of Llama3-8b on GSM-Symbolic appears to be statistically better than on GSM8K.

Analysing models independently, 21 out of 25 models show statistically equivalent performance on GSM8K and GSM-Symbolic.


### 4.2.3 Considering all models together: Is the decline in performance statistically significant?

There is a trend that that many models perform worse on GSM-Symbolic than on GSM8K. To assess the statistical significance of this systematic trend, we can conduct what is known as a *paired* difference test. The Wilcoxon signed-rank test would be an appropriate one to apply in our case with two important caveats.  

**Caveat 1**: Non-independent data. It will be incorrect to perform the test on all 25 models as these are not independent. There are several types of dependence to consider. Most obviously, the base models and their instruct-tuned version are clearly related (e.g. Gemma2-9b and Gemma2-9b-it). I’d also argue that different sizes within the same model family cannot be considered independent (e.g. mini-small-medium for Phi or 2b-9b-27b for Gemma); minor version updates (e.g. Mistral v0.1 vs 0.3, Phi 3 vs 3.5) will also likely be correlated. So although we have a sample of 25 models, the “effective” sample size is much, much smaller. 

Here’s our attempt to come up with independent set of models. In each model family, we take the latest, largest instruct-tuned version. We repeat this by also taking the smallest version. This gives us two sets of 7 models (differences between them are in italics):

- Largest subset of models: Gemma2-27b-it, Phi-3.5-mini-instruct, Mistral-7b-instruct-v0.3, Mathstral-7b-v0.1, Llama3-8b-instruct, GPT-4o, o1-preview.

- Smallest subset of models: Gemma2-2b-it, Phi-3.5-mini-instruct, Mistral-7b-instruct-v0.3, Mathstral-7b-v0.1, Llama3-8b-instruct, GPT-4o-mini, o1-mini.

**Caveat 2**: The results of GSM-Symbolic are averages across 50 datasets, whilst GSM8K is based on a single sample. We'd want to compare accuracies on GSM8K vs accuracies on each of the 50 datasets (at the time of writing these are not publicly available). We're still trying to figure out what is the correct way to handle the unbalanced number of samples in this case is (more on this to come). 

With those two caveats, here is the hypothesis and results:

- Two-sided: The success probabilities are different
$$
H_0: p_{8k} = p_{symb} \quad\quad\quad H_A^\text{two-sided}: p_{8k} \neq p_{symb}.
$$
- One-sided: The success probability on GSM8K is greater than that on GSM-Symbolic
$$
H_0: p_{8k} = p_{symb} \quad\quad\quad H_A^\text{one-sided}: p_{8k} > p_{symb}.
$$

where  $p_{8k}=[p_{1,8k}, \dots, p_{7,8k}]$ and $p_{symb}= [p_{1,symb}, \dots, p_{7,symb}]$.

- Largest subset of models: 

  - p-value of the two-sided test is 0.046875. This means that we can reject the null at the 5%, but not at the 1%, that there are performance differences between GSM-Symbolic and GSM8K. 

  - p-value of the one-sided test is 0.0234375.

- Smallest subset of models: 

  - p-value of the two-sided test is 0.078125, i.e. we cannot reject the null at the 5% or the 1% level. 

  - p-value of the one-sided test is 0.0390625.

**To summarise: analysing the results of models jointly, there appears to be some evidence of differences in performance.**


## 4.3 Performance decrease and variance increase with question complexity

The paper highlights this on multiple occasions, most notably in Section 4.3. Some examples include:

> [page 3] We show that performance degradation and variance increase as the number of clauses increases, indicating that LLMs’ reasoning capabilities struggle […]

> [page 9] the increase in variance suggests that searching and pattern-matching become significantly harder for models as the difficulty increases.

> [page 9, Figure 6 reproduced below] the distribution of performance shifts to the left (i.e., accuracy decreases), and the variance increases.

{% include figure.html 
  path="assets/img/2025-04-28-towards-more-rigorous-llm-evals/fig6_gsm.png" 
  class="img-fluid" 
  caption="Figure 6 from Mirzadeh et al. (2024) <d-cite key='mirzadeh2024gsm'></d-cite>"
%}

Here we have 4 different datasets: the baseline GSM-Symbolic, an easier version of it (M1) and two harder versions of it (P1 and P2). It is not unreasonable to expect that a model might perform differently on different datasets, for example better on the easier ones and worse on harder ones. We can make a similar assumption as before, namely that the answers a model m gives to questions of difficulty $d=-1, 0, 1, 2$ are i.i.d. Bernoulli trials with success probability $p_{m,d}$. The distribution of the total number of correct answers is then $\text{Bin}(N=100, p_{m,d})$, which has variance $N \cdot p_{m,d} \cdot (1 - p_{m,d})$.

If we happen to have decreasing probabilities of success, i.e. $p_{m,-1} > p_{m,0} > p_{m,1} > p_{m,2} > 0.5$ as is the case for some of the models in Figure 6, then the corresponding variances must increase. Whether or not this decrease in probability of success has any relationship to the “reasoning abilities” of a model is beyond the scope of this blog :)

**To summarise: the emphasis on “non-negligible variance” and “increase in variance” throughout the paper appears to be an over-interpretation of normal statistical variation.**

## 4.4 M1, P1, P2 and No-Op results
TODO

## 5. Conclusion

There’s huge value in developing new benchmarks and I think the proposed GSM-Symbolic is quite neat and useful! The accompanying analysis, in my opinion, can be substantially improved with the help of basic statistics. Without those we risk over-interpreting results and drawing misleading conclusions.

Findings:
- Depending on assumptions, variability is expected and quantifiable; and the observed increase in variance with question complexity is expected.
- The observed performance degradation on GSM-Symbolic is not statistically significant.
- Mismatch between GSM8K and GSM-Symbolic distributions may explain some of the observed performance differences between models and datasets (in addition to contamination and "lack of reasoning").




<!-- ### Acknowledgement 

I’d like to Alex Coca, Adam Goliński, Roumen Popov, and ... for their feedback on this post. -->

# Appendix


Paragraph about p-values should go somehwere. Things to say: 

- p-values get misinterpreted; Some "Don'ts" : don't base your conclusions solely on whether an association or effect was found to be “statistically significant”; Don’t believe that an association or effect exists just because it was statistically significant. Don’t believe that an association or effect is absent just because it was not statistically significant. Don’t conclude anything about scientific or practical importance based on statistical significance (or lack thereof).


## Clopper-Pearson confidence intervals

For robustness purposes, here are the [Clopper-Pearson](https://en.wikipedia.org/wiki/Binomial_proportion_confidence_interval#Clopper–Pearson_interval) confidence intervals as well:
<!-- {{< figure library="true" src="assets/img/2025-04-28-towards-more-rigorous-llm-evals/clopper_0.95.png" title="95% Clopper-Pearson intervals for the point estimates of $p_m$." numbered="false">}} -->
{% include figure.html 
  path="assets/img/2025-04-28-towards-more-rigorous-llm-evals/clopper_0.95.png" 
  class="img-fluid" 
  title="95% Clopper-Pearson intervals for the point estimates of $p_m$." 
  caption="95% Clopper-Pearson intervals for the point estimates of $p_m$." 
%}

<!-- <iframe title="95% confidence intervals for Binomial proportion" aria-label="Dot Plot" id="datawrapper-chart-LkJtn" src="https://datawrapper.dwcdn.net/LkJtn/9/" scrolling="no" frameborder="0" style="width: 0; min-width: 100% !important; border: none;" height="607" data-external="1"></iframe><script type="text/javascript">!function(){"use strict";window.addEventListener("message",(function(a){if(void 0!==a.data["datawrapper-height"]){var e=document.querySelectorAll("iframe");for(var t in a.data["datawrapper-height"])for(var r=0;r<e.length;r++)if(e[r].contentWindow===a.source){var i=a.data["datawrapper-height"][t]+"px";e[r].style.height=i}}}))}();</script> -->

The Clopper-Pearson CIs are slightly wider than those obtained using the Wilson score. Using Clopper-Pearson would not have changed any of the results presented in this post. 

## Two-sample binomial proportion test


$$
H_0: p^{8K}_{m} - p^{Symb}_{m} =0 \quad\quad\quad H_A: p^{8K}_{m} - p^{Symb}_{m} \neq 0
$$

Under the null $p^{8K}_{m} = p^{Symb}_{m}$ and so we estimate both using a pooled estimate:
$$
p_{pool} = \frac{(100 p^{8K}_{m} + 5000p^{Symb}_{m})}{100+5000} \quad \text{SE}(p_{pool}) = \sqrt{p_{pool}*(1-p_{pool}) (1/100+1/5000)}.
$$
The test statistic (pm,8k - pm,symb) / SE(ppool)  is then approximately normal and is used co compute p-values, which I’ve done in [this spreadsheet](https://docs.google.com/spreadsheets/d/1Ul6ZgFXf_II5EFUCgnJ9hSIQYwHxogxYBmwDn_bA4sA/edit?usp=sharing). The results in this case are exactly the same as before:  we are able to reject the null for Gemma-7b, Mistral-7b-instruct-v0.1 and Phi-2 (performing worse), and Llama3-8b (performing better). 

## 99% Confidence intervals

<!-- {{< figure library="true" src="assets/img/2025-04-28-towards-more-rigorous-llm-evals/wilson_0.99.png" title="99% Wilson score intervals for the point estimates of $p_m$." numbered="false">}} -->
{% include figure.html 
  path="assets/img/2025-04-28-towards-more-rigorous-llm-evals/wilson_0.99.png" 
  class="img-fluid" 
  title="99% Wilson score confidence intervals for the point estimates of $p^{8K}_{m}$ (red dots), along with point estimates of $p^{Symb}_{m}$ (blue triangles)." 
  caption="99% Wilson score confidence intervals for the point estimates of $p^{8K}_{m}$ (red dots), along with point estimates of $p^{Symb}_{m}$ (blue triangles)." 
%}

{% include figure.html 
  path="assets/img/2025-04-28-towards-more-rigorous-llm-evals/clopper_0.99.png" 
  class="img-fluid" 
  title="99% Clopper-Pearson confidence intervals for the point estimates of $p^{8K}_{m}$ (red dots), along with point estimates of $p^{Symb}_{m}$ (blue triangles)." 
  caption="99% Clopper-Pearson confidence intervals for the point estimates of $p^{8K}_{m}$ (red dots), along with point estimates of $p^{Symb}_{m}$ (blue triangles)." 
%}

## Logistic regression results


Llama results:
```
Call:
glm(formula = correct ~ total_digits + carry, family = "binomial", 
    data = .)

Coefficients:
             Estimate Std. Error z value Pr(>|z|)    
(Intercept)   7.35210    0.47409  15.508  < 2e-16 ***
total_digits -0.41672    0.04859  -8.576  < 2e-16 ***
carry        -0.21252    0.07575  -2.806  0.00502 ** 
---
Signif. codes:  0 ‘***’ 0.001 ‘**’ 0.01 ‘*’ 0.05 ‘.’ 0.1 ‘ ’ 1

(Dispersion parameter for binomial family taken to be 1)

    Null deviance: 1125.63  on 3066  degrees of freedom
Residual deviance:  922.02  on 3064  degrees of freedom
AIC: 928.02

Number of Fisher Scoring iterations: 7
```

Phi results:
```
Call:
glm(formula = correct ~ total_digits + carry, family = "binomial", 
    data = .)

Coefficients:
             Estimate Std. Error z value Pr(>|z|)    
(Intercept)   3.95149    0.17593  22.460   <2e-16 ***
total_digits -0.25747    0.02249 -11.447   <2e-16 ***
carry        -0.02325    0.04907  -0.474    0.636    
---
Signif. codes:  0 ‘***’ 0.001 ‘**’ 0.01 ‘*’ 0.05 ‘.’ 0.1 ‘ ’ 1

(Dispersion parameter for binomial family taken to be 1)

    Null deviance: 2460.8  on 3066  degrees of freedom
Residual deviance: 2217.8  on 3064  degrees of freedom
AIC: 2223.8

Number of Fisher Scoring iterations: 5
```


<!-- 
## Suggested workflow for evaluating language models

1. **Specify the objective and any relevant assumptions:** Define what is being investigated and explicitly outline any relevant assumptions that could impact the evaluation.

2. **Design and run experiments:** Collect performance metrics of interest (e.g., accuracy) along with relevant metadata (e.g., runtime) that could constitute potential confounding factors.

3. **Formulate hypotheses and select an appropriate statistical test:** Null hypothesis significance testing (NHST) is the most common framework for assessing whether the results observed in 2. are substantially (or *statistically significantly*) different from a baseline. 
The null hypothesis ($H_0$) typically posits that there is no difference, whilst the alternative hypothesis ($H_1$) suggests a *meaningful difference* exists. 
The choice of which statistical test to use is determined by characteristics of the data and experimental design. For instance, an independent t-test might be appropriate for comparing performance metrics that can be assumed to follow a normal distribution. For paired data that violate normality assumptions, a Wilcoxon signed-rank test might will be more appropriate. 

4. **Compute relevant statistics:** Calculate confidence intervals and p-values for the observed performance metrics.

5. **Interpret the results:** Confidence intervals provide a range of plausible values for the observed metric, helping establish the reliability and potential variability in model performance. The p-value indicates how likely the observed data would be if the null hypothesis were true; a low p-value suggests evidence against it. Consider these statistical indicators within the broader context of model evaluation. (See section ...).

6. **Accept Uncertainty:** A small p-value serves as an initial, coarse filter for distinguishing signal from noise, but statistical methods cannot transform randomness into certainty. Rather than seeking binary declarations of effects, embrace variation, and treat all statistical conclusions as provisional.
Regardless of the NHST outcomes, critically examine the results, initial assumptions, and alternative explanations. 

**Remark:** Although NHST is the most commonly used framework, it is not the only approach for statistical analysis. 
Bayesian analysis offers an alternative with some practical similarities, such as the use of highest density intervals (HDI). 
However, the two approaches differ deeply in their epistemological foundations: NHST relies on a frequentist interpretation, viewing the results (e.g. accuracy) as repeated sampling from a fixed parameter. In contrast, Bayesian analysis treats parameters themselves as random variables and uses probability to express a degree of belief, that gets updated with new evidence. -->

# DGP:

1. Sample a template $T \sim P_T$.
2. Sample filler values 

$$V \sim P_{V\vert T}(·\vert T)$$

. The pair 

$$T, V$$

 produce a question and an answer
 
$$\left(T(V)_Q, T(V)_A\right)$$


3. We take a language model $M$ and we are interested in estimating 

$$\mathbb{P}(M(T(V)_Q) = T(V)_A) =: p_M$$

- We fix 100 samples 

$$t_1, t_2, ..., t_{100}$$

 of templates, sampled from 
 
 $$P_T$$.
- This gives us a collection of 100 conditional distributions 

$$P_{V|t_i}$$

 of filler values
- We might, in fact, postulate that we have two sets of conditional distributions 

$$\{P^{8k}_{V|t_i}: i=1...100\}$$ 

and

$$\{P^{Symb}_{V|t_i}: i=1...100\}$$

- Under this model, the accuracies we're interested in are captured by the following random variables:

  * for 
  
  $$V_i^{8k} \sim P^{8k}_{V|t_i}, 1≤i≤100$$
  
  , we have binary random variables
  
   $$X^{8k}_{M,t_i} = \mathbb{I}(M(t_i(V_i^{8k})_Q) = t_i(V_i^{8k})_A)$$
   
    and an accuracy estimator 
    
    $$\hat{P}^{8k}_M = \frac{1}{100}\sum_{i=1}^{100} X^{8k}_{M,t_i}$$

  * for 
  
  $$V_i^{Symb} \sim P^{Symb}_{V|t_i}, 1≤i≤100$$
  
  , we have binary random variables
  
   $$X^{Symb}_{M,t_i} = \mathbb{I}(M(t_i(V_i^{Symb})_Q) = t_i(V_i^{Symb})_A)$$
   
    and an accuracy estimator 
    
    $$\hat{P}^{Symb}_M = \frac{1}{100}\sum_{i=1}^{100} X^{Symb}_{M,t_i}$$

We have (†):
- one observation 

$$\hat{p}_M^{8k}$ of $\hat{P}^{8k}_M$$


- 50 observations 

$$\{\hat{p}_{M, j}^{Symb}: 1≤j≤50\}$$ 

of

 $$\hat{P}^{Symb}_M$$

We may assume that 

$$\{X^{8k}_{M,t_i}: 1≤i≤100\}$$

 are i.i.d Bernoulli with probability of success 
 
 $$p^{8k}_M$$

$$\Rightarrow \hat{P}^{8k}_M \sim \frac{1}{100}Bin(100,p^{8k}_M)$$

Same for 

$$X^{Symb}_{t_i} \sim_{i.i.d} Bernoulli(p^{Symb}_M)$$

 and 
 
 $$\hat{P}^{Symb}_M \sim \frac{1}{100}Bin(100,p^{Symb}_M)$$


Q: Given our observations (†), what evidence is there to believe that 

$$p^{8k}_M \neq p^{Symb}_M$$

? Similarly, what evidence is there to believe that 

$$p^{8k}_M > p^{Symb}_M$$

?

Note that it _does_ make sense to also assume that 

$$X^{8k}_{t_i} \perp\!\!\!\perp X^{Symb}_{t_i}$$.