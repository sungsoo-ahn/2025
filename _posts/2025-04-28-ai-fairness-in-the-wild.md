---
layout: distill
title: AI Fairness In The Wild
description: Overview of recent political documents and technical papers that are shaping how fairness in AI is applied to real world data.
date: 2025-04-28
future: true
htmlwidgets: true
hidden: false

# Anonymize when submitting
# authors:
#   - name: Anonymous
# must be the exact same name as your blogpost
bibliography: 2025-04-28-AI_Fairness_In_The_Wild.bib  

# Add a table of contents to your post.
#   - make sure that TOC names match the actual section names
#     for hyperlinks within the post to work correctly. 
#   - please use this format rather than manually creating a markdown table of contents.
toc:
  - name: Introduction
  - name: A Techno-Bureaucratic Complex?
  - name: Quantifying Fairness
  - name: From Theory To Practice
  - name: Fairness In The Wild?

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


## Introduction


Recently, the field of AI fairness has exploded (see graph below), with many conferences adding special tracks and requiring societal impact statements from researchers. Here we review some philosophical, technical, and practical papers that tackle fairness in machine learning applications.

<div class="row justify-content-center">
    <div class="col-md-8 align-self-center">
{% include figure.html path="assets/img/2025-04-28-AI_Fairness_In_The_Wild/Fairness_Pubs.png" class="img-fluid" %}
</div>
<p class="text-sm-center" style="margin:0em 8em 1em; font-size:0.8em">Number of publications focusing on fairness in AI by year, From <a href="https://www.dimensions.ai/">dimensions.ai</a>.</p>
</div>


### A Techno-Bureaucratic Complex?
Fashionably late to the party, governments finally began firing their regulatory engines in earnest mid 2022. A fact that is not somewhat worrying considering algorithms like [COMPAS](https://en.wikipedia.org/wiki/COMPAS_(software)) - a software used predict recidivism rates of criminal defendants - have been in use since the mid 2010s, and have remained widely in use despite evidence of discriminatory bias reported in early 2016 (analysis demonstrated that blacks are twice as likely as whites to be labeled a higher risk but not actually re-offend, see [Angwin's excellent ProPoblica article](https://www.propublica.org/article/machine-bias-risk-assessments-in-criminal-sentencing)).

One major new piece of regulation is Biden's [The Whitehouse Blueprint for an AI Bill of Rights](https://www.whitehouse.gov/ostp/ai-bill-of-rights/). While not an enforceable law, this document could (theoretically) heavily shape future legislation. The framework outlined in the document integrates inputs form many ML researchers and stakeholders, and the includes a "From Principles to Practice" section with examples of successful applications of the proposed framework.

The first few pages are important for contextualizing the framework. First, the authors emphasise that the principles outlined in the document *"are not intended to, and do not, prohibit or limit any lawful activity of a government agency, including law enforcement, national security, or intelligence activities"* <d-footnote>This is especially worrying as if, for instance, a facial recognition system places you in the crosshairs of a police investigation your lawyer can not ask for a suppression hearing, and there is no opportunity for cross examination.</d-footnote>.  Second, they note that *Systemic, fair, and just treatment must take into account the status of individuals who belong to underserved communities*. These communities are specifically defined by what U.S. law considers "protected characteristics" (disability, age, race, gender, religion, sexual orientation, maternity, partnership status ...). Considering the document begins with these qualifiers, it is safe to abandon any hope of attempts to rein in the use of algorithms that, for instance, punish poor people for [shopping at Walmart rather than WholeFoods](https://www.nytimes.com/2009/01/31/your-money/credit-and-debit-cards/31money.html) (For concrete exampls, I recommend the excellent Cathy O'Neil book *Weapons of Math Destruction*).

The rest of the document focuses on five points: First that system should be safe and effective, with clear organization levels plans for assessment and  testing. Second, there should be built in algorithmic discrimination protection. Designers, developers, and deployers of automated systems should take proactive and continuous measures to protect individuals and communities from algorithmic discrimination and to use and design systems in an equitable way. <u>Proactive equity assessments</u> should be part of the system design, and protections against <u> proxies for demographic features </u> should be implemented. Third, the document discusses data privacy, specifying that *"only data strictly necessary for the specific context is collected"* and that *"in sensitive domains, your data and related inferences should only be used for necessary functions"*. Clearly, this vague language leaves a wide open door for data misuse, as any data that might help increase in profits could be argued to be necessary <d-footnote>The document does specify however that traditional "terms of service" are inadequate and that users are entitled to transparent and easy mechanisms to control how their data is used.</d-footnote>. Fourth, notice and explanation should be given when your data is used in an automated system. Finally, the document emphasises the importance of human alternatives and fallbacks, allowing access to a person instead of only the automated system.
 
This document might leave the reader with mixed feelings. While the emphasis on measures for protecting communities, and not only individuals, is encouraging. The vague language employed seems corporate friendly by design. This is especially an concerning since, as we will soon see, the devil is in the details.

## Quantifying Fairness

[Fairness Definitions Explained](https://fairware.cs.umass.edu/papers/Verma.pdf)<d-cite key="def_exp"></d-cite> and [Algorithmic Fairness: Choices, Assumptions, and Definitions](https://www.annualreviews.org/doi/abs/10.1146/annurev-statistics-042720-125902)<d-cite key="alg_fair_choices"></d-cite> are surveys of the most popular quantifications used to measure fairness.  Before proceeding, one should consider a few points regarding these definitions.

{% include figure.html path="assets/img/2025-04-28-AI_Fairness_In_The_Wild/AlgorithmicFairness.png" class="img-fluid" %}
<p class="text-sm-center" style="margin:0em 8em 1em; font-size:0.8em">From <a href="https://www.annualreviews.org/doi/abs/10.1146/annurev-statistics-042720-125902">Algorithmic Fairness: Choices, Assumptions, and Definitions</a>.</p>

Given a group of people who share a protected characteristic $$g$$ and an independent variable $$X$$ (such as applicant credentials) *conditional statistical parity* is achieved when given the same values for $$X$$ candidates with and without the protected characteristic have the same chance to get positive outcomes. This is formalized as $$ p(D=1 \| X=x,G=g) = p(D=1 \| X=x,G=  - g) $$ . As this is a difficult criterion to satisfy, when there is access to ground truth (in the case of COMPAS, simply apply the algorithm to prisoners in other counties and use judge ruling as ground truth) we can instead focus on the predicted outcome $$Y$$ and real result. For instance we can examine the False Positive Balance between groups $$ p(D=0 \| Y=1,G=g)=p(D=0 \| Y=1,G=-g) $$ . Other metrics (such as the false negative rate) could similarly be explored.

***

## From Theory To Practice

Given the metrics defined above, a researcher can identify discriminatory algorithms. However, correcting these algorithms may pose a challenge, and  adopting fairer algorithms might encounter pushback if they fall short in other metrics (such as profitability). To overcome this researchers have developed various tools, for instance [Joint Optimization of AI Fairness and Utility](https://arxiv.org/pdf/2002.01621.pdf)<d-cite key="JointOpt"></d-cite> uses an analytic hierarchy process to rank different metrics and optimize the model to satisfy the resulting weighted sum of these metrics. The final results are visualized to allow the user to easily identify high utility models that are also relatively fair (see image).   


<div class="row justify-content-center">
    <div class="col-md-8 align-self-center">
	{% include figure.html path="assets/img/2025-04-28-AI_Fairness_In_The_Wild/jointOpt.png" class="img-fluid" %}
    </div>
	<p class="text-sm-center" style="margin:0em 8em 1em; font-size:0.8em">From <a href="https://arxiv.org/pdf/2002.01621.pdf">Joint Optimization of AI Fairness and Utility: A
	Human-Centered Approach</a>. The Statistical Parity Difference and Weighted Average Odds Difference (SPD and WAOD, two fairness metrics) are visualized for different algorithms, as well as utility 		
	(profit in thousands of dollars).</p>
</div>

A different tool [AI Fairness 360](https://arxiv.org/pdf/1810.01943.pdf)<d-cite key="aif360"></d-cite> enables more hands-on model optimization. The authors of this tool categorize fairness optimization framework as pre, in, and post-processing framework.

- **Fair pre-processing** techniques focus on editing the to-be training data. This is done by rewighing certain data points (examples of individuals in protected classes with desirable outcomes are given higher weights), optimized preprocessing (edit the data directly), learning far representations (use an encoding space that hides protected characteristics), and desperate impact removal (edit features to improve between group fairness while maintaining ingroup ranking).
- **Fair in-processing** techniques focus on learning a fair classifier given the original unchanged data. This can be done using multi-objective optimization with fairness as an objective, adversarial debiasing (training an adversary that tries to infer protected attributes from model prediction), or a prejudice remover (a discrimination aware regularizer).
- **Fair post-processing** assumes you already have a trained biased model. These techniques try to fix the bias using frameworks such as reject option classification (favor/punish points in the model uncertainty boundary depending on the protected attribute) or equalized odds transformation (lear a linear projection of the classifier predicted probabilities to learn outputs that achieve fairness).

{% include figure.html path="assets/img/2025-04-28-AI_Fairness_In_The_Wild/fairness360.png" style="width: 50%;" %}
<p class="text-sm-center" style="margin:0em 8em 1em; font-size:0.8em">From <a href="https://arxiv.org/pdf/1810.01943.pdf">AI Fairness 360</a>. Different frameworks to guarentee algorithmic farirness.</p>

Overall, this tool - developed and suppoeted by IBM - implements over 15 bias mitigation algorithms, 5 fairness metrics, is implemented in both R and python, and even comes with a nicely designed graphical UI. But does this level of polish translate into adoption in real-world applications?

***

## Fairness In The Wild?

While very interesting, AI fairness research seems to - at least so far -  have very limited actual applications outside academia. For instance, while multiple popular magazines [wrote articles about AI fairness 360](https://www.zdnet.com/article/ibms-new-open-source-tools-help-companies-spot-bias-in-advertising/)<d-cite key="aif360"></d-cite>, I was unable to find a single company that is confirmed to be actively using this tool. This underscores a recurring challenge: theoretical advancements, when not paired with principled political action, often fail to drive substantive change. Various articles, such as [Principles alone cannot guarantee ethical AI](https://www.nature.com/articles/s42256-019-0114-4)<d-cite key="Mittelstadt2019PrinciplesAC"></d-cite> have began discussing ways for large institutions in the filed (such as IEEE or ACM) to regulate the development of algorithms. However, as the most severe punishment in the arsinal of these institutions is explusion from the organization, it is perhaps better to look for salvation elsewhere (historically speaking, specific expulsion cases are not publicized and carry minimal career repercussions).

Difficult discussions are necessary to bring AI fairness into practice, and while the outlook seems grim, it is important to rememebr that we (including you the reader) are important voices in the conversation. All the research discussed above has a gradual compounding effect helping push the needle towards a better future, and by reading, citing, and using the developed algorithms we can actively help push a bit harder.
