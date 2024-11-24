---
layout: distill
title: "Mechanistic Interpretability Meets Vision Language Models: Insights and Limitations"
description: "Vision language models (VLMs), such as GPT-4o, have rapidly evolved, demonstrating impressive capabilities across diverse tasks. However, much of the progress in this field has been driven by engineering efforts, with a limited understanding of how these models work. The lack of scientific insight poses challenges to further enhancing their robustness, generalization, and interpretability, especially in high-stakes settings. In this work, we systematically review the use of mechanistic interpretability methods to foster a more scientific and transparent understanding of VLMs. Specifically, we examine five prominent techniques: probing, activation patching, logit lens, sparse autoencoders, and automated explanation. We summarize the key insights these methods provide into how VLMs process information and make decisions. We also discuss critical challenges and limitations that must be addressed to further advance the field."
date: 2025-04-28
future: true
htmlwidgets: true
hidden: false

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
bibliography: 2025-04-28-vlm-understanding.bib  

# Add a table of contents to your post.
#   - make sure that TOC names match the actual section names
#     for hyperlinks within the post to work correctly. 
#   - please use this format rather than manually creating a markdown table of contents.
toc:
  - name: Introduction
  - name: Current Methods
    subsections:
    - name: Probing
    - name: Activation Patching
    - name: Logit Lens
    - name: Sparse Autoencoders
    - name: Automated Explanation
  - name: Future Directions
    subsections:
    - name: From Single Model to Multiple Models
    - name: From Small Models to Large Models
    - name: From Language-Centric to Vision-Centric
    - name: From Static Processes to Dynamic Processes
    - name: From Micro-Level to Macro-Level
  - name: Conclusion

# Below is an example of injecting additional post-specific styles.
# This is used in the 'Layouts' section of this post.
# If you use this post as a template, delete this _styles block.
# _styles: >
#   .fake-img {
#     background: #bbb;
#     border: 1px solid rgba(0, 0, 0, 0.1);
#     box-shadow: 0 0px 4px rgba(0, 0, 0, 0.1);
#     margin-bottom: 12px;
#   }
#   .fake-img p {
#     font-family: monospace;
#     color: white;
#     text-align: left;
#     margin: 12px 0;
#     text-align: center;
#     font-size: 16px;
#   }
_styles: >
  
  .center {
      display: block;
      margin-left: auto;
      margin-right: auto;
  }

  .framed {
    border: 1px var(--global-text-color) dashed !important;
    padding: 20px;
  }

  d-article {
    overflow-x: visible;
  }

  .underline {
    text-decoration: underline;
  }

  .todo{
      display: block;
      margin: 12px 0;
      font-style: italic;
      color: red;
  }
  .todo:before {
      content: "TODO: ";
      font-weight: bold;
      font-style: normal;
  }
  summary {
    color: steelblue;
    font-weight: bold;
  }

  summary-math {
    text-align:center;
    color: black
  }

  [data-theme="dark"] summary-math {
    text-align:center;
    color: white
  }

  details[open] {
  --bg: #e2edfc;
  color: black;
  border-radius: 15px;
  padding-left: 8px;
  background: var(--bg);
  outline: 0.5rem solid var(--bg);
  margin: 0 0 2rem 0;
  font-size: 80%;
  line-height: 1.4;
  }

  [data-theme="dark"] details[open] {
  --bg: #112f4a;
  color: white;
  border-radius: 15px;
  padding-left: 8px;
  background: var(--bg);
  outline: 0.5rem solid var(--bg);
  margin: 0 0 2rem 0;
  font-size: 80%;
  }
  .box-note, .box-warning, .box-error, .box-important {
    padding: 15px 15px 15px 10px;
    margin: 20px 20px 20px 5px;
    border: 1px solid #eee;
    border-left-width: 5px;
    border-radius: 5px 3px 3px 5px;
  }
  d-article .box-note {
    background-color: #eee;
    border-left-color: #2980b9;
  }
  d-article .box-warning {
    background-color: #fdf5d4;
    border-left-color: #f1c40f;
  }
  d-article .box-error {
    background-color: #f4dddb;
    border-left-color: #c0392b;
  }
  d-article .box-important {
    background-color: #d4f4dd;
    border-left-color: #2bc039;
  }
  html[data-theme='dark'] d-article .box-note {
    background-color: #555555;
    border-left-color: #2980b9;
  }
  html[data-theme='dark'] d-article .box-warning {
    background-color: #7f7f00;
    border-left-color: #f1c40f;
  }
  html[data-theme='dark'] d-article .box-error {
    background-color: #800000;
    border-left-color: #c0392b;
  }
  html[data-theme='dark'] d-article .box-important {
    background-color: #006600;
    border-left-color: #2bc039;
  }
  d-article aside {
    border: 1px solid #aaa;
    border-radius: 4px;
    padding: .5em .5em 0;
    font-size: 90%;
  }
  .caption { 
    font-size: 80%;
    line-height: 1.2;
    text-align: left;
  }
---

<div style="display: none">
$$
\definecolor{input}{rgb}{0.42, 0.55, 0.74}
\definecolor{params}{rgb}{0.51,0.70,0.40}
\definecolor{output}{rgb}{0.843, 0.608, 0}
\def\mba{\boldsymbol a}
\def\mbb{\boldsymbol b}
\def\mbc{\boldsymbol c}
\def\mbd{\boldsymbol d}
\def\mbe{\boldsymbol e}
\def\mbf{\boldsymbol f}
\def\mbg{\boldsymbol g}
\def\mbh{\boldsymbol h}
\def\mbi{\boldsymbol i}
\def\mbj{\boldsymbol j}
\def\mbk{\boldsymbol k}
\def\mbl{\boldsymbol l}
\def\mbm{\boldsymbol m}
\def\mbn{\boldsymbol n}
\def\mbo{\boldsymbol o}
\def\mbp{\boldsymbol p}
\def\mbq{\boldsymbol q}
\def\mbr{\boldsymbol r}
\def\mbs{\boldsymbol s}
\def\mbt{\boldsymbol t}
\def\mbu{\boldsymbol u}
\def\mbv{\boldsymbol v}
\def\mbw{\textcolor{params}{\boldsymbol w}}
\def\mbx{\textcolor{input}{\boldsymbol x}}
\def\mby{\boldsymbol y}
\def\mbz{\boldsymbol z}
\def\mbA{\boldsymbol A}
\def\mbB{\boldsymbol B}
\def\mbE{\boldsymbol E}
\def\mbH{\boldsymbol{H}}
\def\mbK{\boldsymbol{K}}
\def\mbP{\boldsymbol{P}}
\def\mbR{\boldsymbol{R}}
\def\mbW{\textcolor{params}{\boldsymbol W}}
\def\mbQ{\boldsymbol{Q}}
\def\mbV{\boldsymbol{V}}
\def\mbtheta{\textcolor{params}{\boldsymbol \theta}}
\def\mbzero{\boldsymbol 0}
\def\mbI{\boldsymbol I}
\def\cF{\mathcal F}
\def\cH{\mathcal H}
\def\cL{\mathcal L}
\def\cM{\mathcal M}
\def\cN{\mathcal N}
\def\cX{\mathcal X}
\def\cY{\mathcal Y}
\def\cU{\mathcal U}
\def\bbR{\mathbb R}
\def\y{\textcolor{output}{y}}
$$
</div>

## Introduction

<!-- VLM introduction -->

<!-- VLM introduction and interpret limitation -->
<!-- While VLMs demonstrate impressive empirical performance, our understanding of how these models process, represent, and integrate visual and linguistic information remains limited. For instance, models like CLIP, which have become the de facto standard for vision encoders, consistently outperform alternatives, yet the mechanisms underlying their success are poorly understood. Similarly, fine-tuning techniques such as LoRA and RLHF enhance task-specific performance but provide little insight into the broader dynamics of multimodal alignment. Meanwhile, VLMs continue to exhibit notable deficiencies, such as struggles with fundamental vision-centric tasks like image classification and a propensity for generating hallucinated outputs. -->

Vision-Language Models (VLMs), such as GPT-4V <d-cite key="2023GPT4VisionSC"></d-cite> or LLaVA <d-cite key="liu2023visual,liu2023improved,liu2024llavanext,li2024llavaonevision"></d-cite>, have achieved remarkable success across a wide range of tasks, including including Image Captioning <d-cite key="vinyals2014show"></d-cite>, Visual Question Answering (VQA)  <d-cite key="agrawal2015vqa"></d-cite>, and Multimodal Reasoning <d-cite key="NEURIPS2022_11332b6b"></d-cite>. These advancements have driven innovation in diverse fields, such as virtual assistants <d-cite key="wu2023visual"></d-cite>, autonomous robotics <d-cite key="driess2023palme"></d-cite>, and medical diagnostics <d-cite key="singhal2023towards"></d-cite>. However, despite their rapid adoption, the internal mechanisms of these models remain largely opaque, raising significant concerns about their reliability, robustness, and interpretability—particularly in high-stakes applications <d-cite key="kolicic2024inherently"></d-cite>.

Interpretability research offers a promising path to address these challenges. Mechanistic interpretability, in particular, seeks to uncover the inner processes of neural networks and explain how specific outputs are generated <d-cite key="saphra2024mechanistic,hastingswoodhouse2024introduction"></d-cite>. By applying these techniques to VLMs, researchers can gain valuable insights into how these models represent, process, store, and integrate visual and linguistic information, advancing both theoretical understanding and practical utility.

In this work, we examine how mechanistic interpretability methods can illuminate the inner workings of VLMs. We review five key techniques—probing <d-cite key="alain2016understanding,hewitt-manning-2019-structural"></d-cite>, activation patching <d-cite key="NEURIPS2020_92650b2e, NEURIPS2022_6f1d43d5"></d-cite>, logit lens <d-cite key="alignmentforumorg2024interpreting"></d-cite>, sparse autoencoders <d-cite key="bricken2023monosemanticity,DBLP:conf/iclr/HubenCRES24"></d-cite>, and automated explanations—detailing their mechanisms, applications, and the insights they provide through concrete examples. These methods help answer critical questions, such as what information is encoded in VLM representations <d-cite key="cao2020behind"></d-cite>, how and when visual and linguistic modalities are integrated <d-cite key="Palit_2023_ICCV,neo2024towards"></d-cite>, and how individual neurons contribute to the model’s decision-making process <d-cite key="huo2024mmneuron,huang2024miner"></d-cite>.

Additionally, we discuss the limitations of current interpretability methods and highlight five key directions for future research: developing approaches that are more generalizable, scalable, vision-centric, dynamic, and capable of macro-level analysis. For instance, the heterogeneity of VLMs calls for interpretability methods that can adapt across diverse models; the micro level of mechanistic interpretability needs to be complemented by a macro-level perspective for a broader understanding. By addressing these challenges, we aim to pave the way for more transparent, reliable, and capable vision language models.

<!-- , unlocking the full potential of VLMs in both research and real-world applications. -->


<!-- By synthesizing findings across diverse interpretability techniques, we aim to provide actionable insights into the underlying principles that govern VLM behavior and guide the development of more transparent, reliable, and capable vision-language systems.



By synthesizing findings across diverse interpretability techniques, we aim to provide actionable insights into the underlying principles that govern VLM behavior and guide the development of more transparent, reliable, and capable vision-language systems. 

 methods, including probing, activation patching, logit lens analysis, sparse autoencoders, and automated explanation techniques, provide a scientific lens to decode how VLMs store and integrate information. These methods uncover mechanisms such as cross-modal alignment, hierarchical information flow, and latent concept representations, while exposing persistent challenges in scalability, generalizability, and practical application. For instance, probing tasks reveal that VLMs often prioritize linguistic over visual information, while activation patching has illuminated the late-stage integration of visual and textual modalities in models like LLaVA.

This paper aims to bridge the gap between empirical advancements and mechanistic understanding by reviewing how interpretability tools can unlock the inner workings of VLMs. By synthesizing findings across diverse interpretability techniques, we provide actionable insights into the underlying principles that govern VLM behavior. We argue that integrating mechanistic insights with empirical observations is essential to addressing challenges in interpretability, robustness, and scalability. This integration has the potential to guide the development of more transparent, reliable, and capable vision-language systems, ultimately advancing both theoretical understanding and real-world applications. -->


<!-- Vision-language models (VLMs), such as GPT-4o, Claude, and Gemini, are transforming multi-modal AI by integrating text and images to perform tasks like image captioning, visual question answering, and multi-modal reasoning. These models are rapidly evolving, achieving widespread adoption across domains like virtual assistants, robotics, and medicine. However, their mechanisms often remain opaque, raising concerns about reliability, robustness, and interpretability—especially in critical applications.

Despite recent empirical advances, understanding how VLMs process and integrate visual-linguistic information remains limited. For instance, while CLIP outperforms other visual encoders, the reasons for its effectiveness are unclear. Similarly, fine-tuning techniques like LoRA improve outcomes, but their broader implications are poorly understood. Furthermore, surprising deficiencies persist, such as failures in image classification and a propensity for hallucinations in generated outputs.

In this work, we review how mechanistic interpretability methods can advance a scientific understanding of VLMs. Techniques like probing, activation patching, logit lens analysis, sparse autoencoders, and automated explanations reveal insights into how VLMs represent, store, and integrate information. These methods illuminate specific mechanisms, such as cross-modal alignment and hierarchical processing, while exposing gaps in scalability and practical application.

We argue for bridging empirical observations with mechanistic insights to better understand and guide VLM development. By formalizing this integration, future research can address challenges in interpretability, robustness, and scalability, paving the way for more transparent and capable vision-language systems. -->

<!-- ## Macro-Level Analysis: Empirical Findings

The release of GPT-4V <d-cite key="2023GPT4VisionSC"></d-cite> has catalyzed significant research interest in Vision-Language Models (VLMs), leading to numerous empirical advances in architectures, training methodologies, data strategies, and evaluations. While these developments have demonstrated what constitutes effective approaches, our theoretical understanding of why these approaches succeed remains limited.

### Architectural Design
<img src="{{ 'assets/img/2025-04-28-vlm-understanding/architecture.png' | relative_url }}" alt="transformer" width="90%" class="l-body rounded z-depth-1 center">
<div class="l-gutter caption" markdown="1">
Self-attention and cross-attention architectures. Figure from <d-cite key="chen2024evlm"></d-cite>.
</div>

Recent architectural explorations in VLMs demonstrate remarkable diversity and sophistication. Cross-attention architectures like Flamingo <d-cite key="NEURIPS2022_960a172b"></d-cite><d-cite key="awadalla2023openflamingo"></d-cite><d-cite key="DBLP:conf/nips/LaurenconSTBSLW23"></d-cite> and LLama 3-V <d-cite key="dubey2024llama"></d-cite> process image tokens through LLM cross-attention mechanisms, while more prevalent decoder-only models such as LLaVA <d-cite key="liu2023visual"></d-cite><d-cite key="liu2023improved"></d-cite><d-cite key="liu2024llavanext"></d-cite><d-cite key="li2024llavaonevision"></d-cite>, InternVL <d-cite key="chen2023internvl"></d-cite><d-cite key="chen2024far"></d-cite>, and Qwen-VL <d-cite key="Qwen-VL"></d-cite><d-cite key="Qwen2VL"></d-cite> utilize a visual encoder→cross-modal connector→LLM approach. Beyond these mainstream approaches, the field has seen hybrid designs like NVLM-H <d-cite key="dai2024nvlm"></d-cite>, which combines cross- and self-attention architectures together, alongside developments in dynamic resolution handling <d-cite key="li2023monkey"></d-cite><d-cite key="ye2023ureader"></d-cite><d-cite key="xue2024xgenmm"></d-cite>, MoE integration <d-cite key="lin2024moellava"></d-cite>, and vision-encoder-free designs <d-cite key="fuyu-8b"></d-cite><d-cite key="wang2024emu3"></d-cite> etc., highlighting the rich diversity of architectural designs in this rapidly evolving field.

While some open-source VLMs now achieve results comparable to commercial models like GPT-4V <d-cite key="2023GPT4VisionSC"></d-cite> and Gemini <d-cite key="team2023gemini"></d-cite><d-cite key="team2024gemini"></d-cite>, fundamental questions about architectural design principles remain unanswered. For example, recent research <d-cite key="tong2024eyes"></d-cite> reveals intriguing differences in task performance between models using DINO <d-cite key="oquab2023dinov2"></d-cite> versus CLIP <d-cite key="DBLP:conf/icml/RadfordKHRGASAM21"></d-cite> as visual encoders, yet the underlying mechanisms driving these differences remain poorly understood.

### Training Recipes

<img src="{{ 'assets/img/2025-04-28-vlm-understanding/training.png' | relative_url }}" alt="transformer" width="90%" class="l-body rounded z-depth-1 center">
<div class="l-gutter caption" markdown="1">
The different stages of training and the types of datasets used. Figure from <d-cite key="laurençon2024building"></d-cite>.
</div>

Training typically involves two stages: pretraining and fine-tuning. Modern training strategies incorporate varying degrees of parameter unfreezing, alongside techniques like LoRA <d-cite key="hu2021lora"></d-cite> to enhance training stability, and DPO <d-cite key="rafailov2023direct"></d-cite> or RLHF <d-cite key="sun2023aligning"></d-cite><d-cite key="yu2023rlhfv"></d-cite><d-cite key="chen2023dress"></d-cite> in the alignment phase to reduce hallucinations. During these stages, progressively higher-quality data is introduced, the maximum image resolution is gradually increased, and more model components are unfrozen <d-cite key="laurençon2024building"></d-cite>.

In pretraining, which aims to establish robust multimodal understanding capabilities, the LLM is typically kept frozen to preserve its linguistic abilities. Some approaches first jointly train the projector and vision encoder <d-cite key="Qwen2VL"></d-cite>, while others focus solely on the newly initialized parameters <d-cite key="laurençon2024building"></d-cite>. The fine-tuning phase involves supervised fine-tuning (SFT) followed by alignment, where the LLM is unfrozen while the vision encoder typically remains frozen.<d-footnote>We need vast text-image pairs (modality mixed) for pretraining and curated academic datasets (domain mixed) for fine-tuning.</d-footnote>

The impact of different training choices extends far beyond benchmark scores. Recent research has revealed that delayed feedback loops can make pretraining ablations misleading <d-cite key="laurençon2024building"></d-cite>, and that scaling alone cannot overcome inherent limitations in CLIP-based models <d-cite key="tong2024eyes"></d-cite>. These findings underscore the need for micro-level analysis to understand how to improve the training process.

### Evaluation

The evaluation of VLMs has evolved to include numerous benchmarks testing various capabilities, including but not limited to knowledge and reasoning <d-cite key="Yue_2024_CVPR"></d-cite><d-cite key="yue2024mmmupro"></d-cite>, document understanding <d-cite key="mathew2020docvqa"></d-cite>, and mathematical problem-solving <d-cite key="DBLP:conf/iclr/LuBX0LH0CG024"></d-cite>. These evaluation frameworks quantitatively assess different aspects of model performance, though the challenge persists: understanding the true depth of what our metrics reveal about model capabilities, and leveraging benchmarks not merely as performance indicators, but as informative experiments that illuminate both the potential and limitations of these systems <d-cite key="miller2024adding"></d-cite>.

---

This macro-level overview reveals both the impressive progress in VLM development and the critical gaps in our theoretical understanding. To develop more transparent, reliable, and capable systems, we must complement these macro-level insights with rigorous micro-level interpretability analysis. The following sections will explore how modern interpretability techniques can help us uncover the scientific principles underlying VLM success. -->

## Current Methods

In this section, we review mechanistic interpretability methods applied to vision language models (VLMs), which aim to uncover the internal processes of these VLMs process visual and language information and explain how they produce specific outputs. Key techniques discussed include probing, activation patching, logit lens analysis, sparse autoencoders, and automated explanations.

<!-- In this section, we explore existing works of mechanistic interpretability method in vision language models (VLMs - seeking to understand the internal processes of neural networks and gain insight into how and why they produce the outputs that they do <d-cite key="saphra2024mechanistic,hastingswoodhouse2024introduction"></d-cite>, which is similar to how neuroscientists study the brain's cognitive processes. These methods aim to uncover the underlying mechanisms of VLMs, providing insights into how they process and integrate visual and linguistic information, including probing, activation patching, logit lens, sparse autoencoders, and automated explanations.

Through our investigation, we find that interpretability in VLMs is still in its early stages. While building intrinsic interpretability or developmental interpretability could offer crucial insights into how these models learn and evolve, it remains largely unexplored in current research. For this study, we focus on six post-hoc interpretability methods that have been successfully applied in VLM research. -->

### Probing

<object data="{{ 'assets/img/2025-04-28-vlm-understanding/probing.svg' | relative_url }}" type="image/svg+xml" width="80%" class="l-body rounded z-depth-1 center"></object>
<div class="l-gutter caption" markdown="1">
Analyzing internal representations through a simple linear classifier.
</div>

Probing <d-cite key="alain2016understanding,hewitt-manning-2019-structural"></d-cite> serves as a powerful diagnostic tool - essentially a microscope for peering into the internal workings of neural networks, helping us understand what information these models actually capture in their representations. At its core, the technique involves **training simple supervised classifiers (probes) to predict specific properties from a model's internal representations.** 

<details markdown="1">
<summary><b>An Example</b></summary>
For instance, in a vision-language model analyzing "a cat sitting on a mat", we might train a linear probe to predict spatial relationships between objects, or whether certain visual attributes are being captured in the textual representations. 

The probe's performance on such tasks reveals whether this information is readily accessible in the model's representations - high accuracy suggests the property is well-encoded, while poor performance indicates the information may be absent or deeply entangled.
</details>

One key advantage of probing is its empirical transparency: by using simple classifiers <d-footnote>single linear probes could suffice in most cases, as their simplicity ensures that strong performance reflects information present in the representations rather than complex learning by the probe itself</d-footnote> as probes, we can directly examine what information is encoded in model representations. 

<aside class="l-body box-warning" markdown="1">
However, it's important to note that high probe accuracy alone doesn't necessarily mean the model actively uses this information during its regular operation - it indicates correlation in information encoding but not necessarily causation in the model's decision-making process.
</aside>

#### Findings

Most research on probing tasks in VLMs focuses on two primary objectives: **identifying the concepts these models struggle to capture** and **assessing the relative importance of visual and linguistic modalities** <d-cite key="golovanevsky2024vlms"></d-cite>. 

Cao et al. <d-cite key="cao2020behind"></d-cite> introduced the VALUE (Vision-And-Language Understanding Evaluation) framework, which developed a set of probing tasks to explain individual layers, heads, and fusion techniques in understanding how pre-training influences learned representations. Their key findings include: (i) pre-trained models tend to prioritize language over vision, a trend consistently observed in the literature; (ii) certain attention heads effectively capture cross-modal interactions; and (iii) attention visualization can reveal interpretable visual relationships.

Building on previous work, researchers have used probing methodologies to examine diverse model capabilities. Studies have investigated various cognitive abilities, including visual semantics <d-cite key="dahlgren-lindstrom-etal-2020-probing"></d-cite>, verb processing <d-cite key="hendricks2021probing,beňová2024beyond"></d-cite>, number sense <d-cite key="kajic2022probing"></d-cite>, and spatial reasoning <d-cite key="pantazopoulos2024lost"></d-cite>, revealing limitations across these domains. Another line of research compared representations at pre-trained and fine-tuned levels <d-cite key="Salin_Farah_Ayache_Favre_2022"></d-cite>, carefully designing datasets to minimize bias and improve multimodal analysis.

While these diverse probing approaches have advanced our understanding of VLMs, probing tasks present several challenges - they must be meticulously crafted to yield meaningful results and are often model-specific. Sometimes, additional models or classifiers are necessary, which limits their broader applicability <d-cite key="vatsa2023adventures"></d-cite>.

<aside class="l-body box-note" markdown="1">
Key Takeaways: 
- Probing is a class of methods for interpreting neural models by assessing whether the model representations encode specific kinds of information
- In VLMs, probing has revealed crucial insights about modality interactions and information encoding patterns
</aside>

### Activation Patching
<object data="{{ 'assets/img/2025-04-28-vlm-understanding/activation.svg' | relative_url }}" type="image/svg+xml" width="100%" class="l-body rounded z-depth-1 center"></object>
<div class="l-gutter caption" markdown="1">
Activation patching experiments comparing model behavior under clean, corrupted, noising, and denoising conditions. We can see that noising and denoising influence the logits' A and B values.
</div>

Activation patching <d-cite key="NEURIPS2020_92650b2e, NEURIPS2022_6f1d43d5"></d-cite> (also known as Casual Tracing, Interchange Intervention, Causal Mediation Analysis) is a powerful interpretability technique for neural networks. It enables researchers to systematically analyze **how different components of a model contribute to its behavior** by modifying specific internal activations while keeping others constant. This approach, grounded in the principle of control variates, provides causal insights into model behavior, helping researchers identify critical components and potential interventions for improving model performance and reliability.. 

#### Methods

The basic process of activation patching involves five steps:
1. **Save Activations:** Run a clean prompt and a corrupted prompt through the model, saving their internal activations.
2. **Select Target Activations:** Identify specific model activations for modification.
3. **Patch Activations:** Replace activations in one run with those saved from another (e.g., clean → corrupted or vice versa).
4. **Rerun Model:** Rerun the model with patched activations to observe changes in behavior.
5. **Analyze Results:** Examine how patched activations affect the model's output to infer their role.

<details markdown="1">
<summary><b>An Example</b></summary>
Consider analyzing a VLM's behavior when identifying objects in an image. Let's say we have:
- Clean input: An image of "a cat sitting on a mat"
- Corrupted input: The same image with Gaussian noise added to the "cat" region

Through activation patching, we could:
1. Run both inputs through the model, saving activations at each layer
2. Replace the corrupted image's activations in specific layers with those from the clean image
3. Observe which layer replacements restore the model's ability to identify "cat"

If replacing activations in layer X restores the correct identification, this suggests layer X is crucial for object recognition. This process helps us understand where and how the model processes object information.
</details>
**Common Approaches**

There are two primary ways to apply activation patching <d-cite key="heimersheim2024use"></d-cite>:

- **Denoising Analysis** involves taking a corrupted prompt, such as one where Gaussian noise has been added to key embeddings, and replacing its activations with those from a clean prompt. By observing which patched activations restore the clean behavior, researchers can identify the components that are **sufficient to correct** the corrupted behavior. For example, this technique can reveal layers where key information is integrated or restored during processing.
- **Noising Analysis**, on the other hand, starts with a clean prompt and replaces its activations with those from a corrupted prompt. By determining which patches disrupt the clean behavior, this method pinpoints the components **necessary to maintain** the correct output. This analysis is particularly useful for identifying which layers or activations play a critical role in preserving the model's functionality.
<details markdown="1">
<summary><b>How to create a corrupted input?</b></summary>
For text inputs, we can introduce perturbations through either Gaussian noise (GN) or Symmetric Token Replacement (STR) <d-cite key="NEURIPS2020_92650b2e, NEURIPS2022_6f1d43d5"></d-cite>, where STR replaces tokens with their semantically similar alternatives. However, since GN pushes inputs out of distribution and disrupts model's internal mechanisms, STR is often preferred <d-cite key="DBLP:conf/iclr/ZhangN24"></d-cite>. For image inputs, we can similarly apply Gaussian noise or use Semantic Image Pairs (SIP) <d-cite key="golovanevsky2024vlms"></d-cite>, a recently introduced approach that serves as the visual domain counterpart to STR. We will discuss SIP in more detail in the Findings section.
</details>
**Methodological Variations** 

Several variations of activation patching have been developed: 
- **Direct Ablations** <d-cite key="DBLP:conf/iclr/NandaCLSS23"></d-cite>: A simpler variant where activations are replaced with zeros or dataset means. While zero ablation shows components critical for network behavior, mean ablation is more natural version of zero ablation.
- **Path Patching** <d-cite key="goldowsky-dill2023localizing"></d-cite>: An extension that traces specific causal pathways through the network, helping understand how information flows between different model components. 
- **Attention Knockout** <d-cite key="geva2023dissecting"></d-cite>: A specialized form focused on analyzing attention mechanisms by selectively blocking attention patterns between tokens. 

#### Findings

<!-- **1. Visual-Linguistic Integration** 
- Activation patching has revealed intricate patterns in how VLMs integrate visual and linguistic information. Through careful experimentation with Gaussian noise injection, Palit et al. <d-cite key="Palit_2023_ICCV"></d-cite> discovered that BLIP's outputs are predominantly influenced by correct image embeddings only in specific layers --- layer 11 of the question encoder and layers 9-11 of the answer decoder. This suggests **either that cross-modal integration is primarily a late-stage process, or that final layers override earlier computations while maintaining weak causal connections.** Interestingly, Neo et al. <d-cite key="neo2024towards"></d-cite> found that representations at visual token positions gradually evolve through layers to align with interpretable textual concepts, suggesting **VLMs naturally refine visual information towards language-like representations even without explicit visual pretraining.** In parallel work, Golovanevsky et al. <d-cite key="golovanevsky2024vlms"></d-cite> developed Semantic Image Pairs (SIP) to enable more precise analysis of how VLMs process semantic information. By modifying single semantic concepts in images (like changing "cat" to "dog"), SIP revealed that cross-attention serves three distinct functions (object detection, suppression, and outlier suppression) and uncovered architectural distinctions: **LLaVA lacks "text-only" attention heads while BLIP has no "vision-only" heads, though both utilize universal heads for cross-modal integration.**

**2. Layer-wise Processing and Information Flow**
- Studies have uncovered a sophisticated hierarchical processing pattern in VLMs. Through causal tracing, Basu et al. <d-cite key="basu2024understanding"></d-cite> found that **models like LLaVA retrieve visual information primarily from early layers (1-4), with consistent summarization in late visual tokens.** Neo et al. <d-cite key="neo2024towards"></d-cite> refined this understanding through attention knockout experiments, showing that **early layers (1-10) integrate broader contextual information, while middle-to-late layers (15-24) extract object-specific details.** Notably, they found minimal impact when blocking attention from visual tokens to the last row, challenging previous assumptions about intermediate summarization steps. Mean ablation studies revealed interesting layer dynamics: Gandelsman et al. <d-cite key="gandelsman2023interpreting"></d-cite> found that **only final layers have significant direct effects on accuracy, with early multihead attention removal having minimal impact.** These findings were extended by Balasubramanian et al. <d-cite key="balasubramanian2024decomposing"></d-cite> to ViTs, showing that while early layers contribute minimally, the final four layers are crucial for maintaining model performance.

**3. Methodological Innovations**
- Recent analytical tools have significantly enhanced our understanding of VLMs. Ben et al. <d-cite key="Ben_Melech_Stan_2024_CVPR"></d-cite> developed LVLM-Interpret, an interactive tool that combines attention knockout with relevancy mapping and causal graph construction to visualize information flow patterns and identify critical image regions.

<aside class="l-body box-note" markdown="1"> 
Key Takeaways: 
- Activation patching provides causal insights into model behavior by modifying specific internal activations while keeping others constant 
- Different variants (e.g., direct ablation, attention knockout) offer complementary perspectives on information flow and processing 
- Key findings reveal: 
	- Cross-modal integration occurs primarily in late layers, with visual information gradually evolving towards language-like representations 
	- VLMs show hierarchical processing: early layers handle context with minimal direct impact, while final layers are crucial 
	- Different architectures exhibit distinct patterns in how they handle cross-modal information 
</aside> -->
**1. Visual-Linguistic Integration**

Recent studies using activation patching have revealed how VLMs combine visual and textual information. Here are the key findings:
- **Layer-Specific Processing in BLIP**: Palit et al. <d-cite key="Palit_2023_ICCV"></d-cite> used Gaussian noise patching to analyze BLIP's processing patterns. They found that image information primarily influence the model's outputs in specific layers: layer 11 of the question encoder and layers 9-11 of the answer decoder. This observation suggests two possibilities:
  - The model might primarily combine visual and text information in its later layers
  - Later layers might play a more decisive role in the final output, while earlier layers provide supporting information
- **Visual-to-Language Transformation in LLaVA**: Neo et al. <d-cite key="neo2024towards"></d-cite> examined how LLaVA processes visual information. They found that representations at visual token positions change systematically across layers, gradually aligning with interpretable textual concepts. This indicates that VLMs can naturally transform visual information into language-like representations, even without specific visual pretraining.
- **Architectural Differences**: Golovanevsky et al. <d-cite key="golovanevsky2024vlms"></d-cite> developed a new method called *Semantic Image Pairs (SIP)* - a method where they make concept changes to images (e.g., changing a "cat" to a "dog") to understand how VLMs process meaning. Their analysis revealed:
  - Cross-attention serves three functions: object detection, suppression, and outlier suppression
  - Different architectures have distinct characteristics:
    - LLaVA lacks "text-only" attention heads
    - BLIP has no "vision-only" heads
    - Both models use universal heads for cross-modal integration

**2. Layer-wise Information Processing**
- **Early vs. Late Layer Functions**: Basu et al. <d-cite key="basu2024understanding"></d-cite> used causal tracing to show that LLaVA primarily retrieves visual information in early layers (1-4), followed by consistent summarization in final visual tokens. Neo et al. <d-cite key="neo2024towards"></d-cite> further investigated this through attention knockout experiments, they found that:
  - Layers 1-10 process broader contextual information
  - Layers 15-24 focus on extracting specific object details
  - Notably, they found that blocking visual token attention to the last row had minimal impact, challenging previous theories about intermediate summarization steps
- **Layer Importance for Model Performance**: Initial studies by Gandelsman et al. <d-cite key="gandelsman2023interpreting"></d-cite> on CLIP showed that final layers have significant direct effects on model accuracy, while early layer modifications (like removing multihead attention) have minimal impact. Balasubramanian et al. <d-cite key="balasubramanian2024decomposing"></d-cite> later extended these findings across a broader range of Vision Transformers, confirming the critical role of the final four layers in model performance.

**3. Analytical Tools**
- Recent analytical tools have significantly enhanced our understanding of VLMs. Ben et al. <d-cite key="Ben_Melech_Stan_2024_CVPR"></d-cite> developed LVLM-Interpret, an interactive tool that combines attention knockout with relevancy mapping and causal graph construction to visualize information flow patterns and identify critical image regions.

<aside class="l-body box-note" markdown="1">
Key Takeaways:
- Activation patching provides causal insights into model behavior by modifying specific internal activations while keeping others constant
- Different variants (e.g., direct ablation, attention knockout) offer complementary perspectives on information flow and processing
- Key findings reveal:
  - Cross-modal integration occurs primarily in late layers, with visual information gradually evolving towards language-like representations
  - VLMs show hierarchical processing: early layers handle context with minimal direct impact, while final layers are crucial since they extract information more related to the task
  - Different architectures exhibit distinct patterns in how they handle cross-modal information
</aside>

### Logit Lens

<object data="{{ 'assets/img/2025-04-28-vlm-understanding/logit_lens.svg' | relative_url }}" type="image/svg+xml" width="80%" class="l-body rounded z-depth-1 center"></object>
<div class="l-gutter caption" markdown="1">
Logit lens implementation showing multiple prediction heads tapping into different model layers.
</div>

The logit lens <d-cite key="alignmentforumorg2024interpreting"></d-cite> serves as a powerful analytical tool for understanding **how neural networks progressively refine their predictions across different layers.** The methodology is elegantly simple yet powerful: it applies the model's final classification layer (unembedding matrix) to intermediate activations, projecting them into vocabulary space. By examining intermediate activations, this technique effectively creating a series of "snapshots" of the model's developing understanding, offering insights into how multimodal models integrate and process information. 

<details markdown="1"> <summary><b>An Example</b></summary> Consider a vision-language model processing an image of "a dog chasing a ball in a park". Using the logit lens, we can observe how the model's prediction confidence evolves: 
- In early layers, the model might show high uncertainty, with similar probabilities for related concepts like "dog", "animal", and "pet" 
- Middle layers start to refine these predictions, showing increased confidence in "dog" while maintaining awareness of context ("park", "ball") 
- Final layers demonstrate sharp predictions focusing on the specific action ("chasing") and precise object relationships This progression reveals how the model gradually builds its understanding from basic visual features to complex scene interpretation.
</details>

#### Findings

1. **Concept Distribution Patterns**
- MMNeuron <d-cite key="huo2024mmneuron"></d-cite> applies logit lens to analyze hidden states of multimodal models like LLaVA-NeXT and InstructBLIP. Through their analysis of decoded vocabulary distributions, they reveal that image tokens generate notably sparser distributions than text tokens. This observation suggests that **image representations are encoded as mixtures of concepts rather than direct word mappings.**

2. **Representations Evolution**
- By examining the entropy of these distributions across layers, Huo et al. <d-cite key="huo2024mmneuron"></d-cite> uncover a distinctive three-stage pattern: **initial feature alignment with high entropy, followed by information processing with sharply declining entropy in middle layers, and finally token selection with slight entropy increase.** More recent work <d-cite key="neo2024towards"></d-cite> further explores how representations at visual token positions evolve through the layers in LLaVa 1.5, finding that **activations in the late layers at each visual token position correspond to token embeddings that describe its original patch object.**

3. **Reduce Hallucinations**
- Building on these insights, Jiang et al. <d-cite key="jiang2024interpreting"></d-cite> demonstrate practical applications of the logit lens by using it to spatially localize objects and perform targeted edits to VLM's latent representations. Their approach effectively reduces hallucinations without compromising the model's overall performance, showcasing how understanding internal representations can lead to concrete improvements in model reliability.

<aside class="l-body box-note" markdown="1"> 
Key Takeaways: 
- Logit lens enables tracking of prediction evolution across model layers
- Through logit lens analysis, researchers discovered that VLMs process images as distributed concept mixtures through a three-stage evolution pattern 
</aside>

### Sparse Autoencoders

<object data="{{ 'assets/img/2025-04-28-vlm-understanding/sae.svg' | relative_url }}" type="image/svg+xml" width="80%" class="l-body rounded z-depth-1 center"></object>
<div class="l-gutter caption" markdown="1">
Integration of Sparse Autoencoder for interpreting and analyzing learned features in transformer models.
</div>

A significant challenge in understanding neural networks is the *superposition* phenomenon <d-cite key="arora2016linear,olah2020zoom"></d-cite>, where individual neurons or layers simultaneously encode multiple features, making their internal representations difficult to interpret <d-cite key="elhage2022toy"></d-cite><d-footnote>Superposition can occur because features tend to be sparsely activated - with only a small subset of features being active at any given moment. </d-footnote>. In late 2023, Sparse Autoencoders (SAEs) was adopted as an effective solution to this problem <d-cite key="bricken2023monosemanticity,DBLP:conf/iclr/HubenCRES24"></d-cite>, offering a valid approach to extract mono-semantic concepts from neural networks.

**SAEs seek to embed the polysemantic representation into a much higher-dimensional space to distangle features.** The technical mechanism of SAEs follows a specific encoding and decoding process <d-cite key="ferrando2024primer"></d-cite>. Given internal network activations $$ z \in \mathbf{R}^d $$, the SAE first expands the representation into a higher-dimensional space while enforcing sparsity through its encoder <d-footnote markdown="1"> The subtraction of $b_{dec}$ serves as a centering term that allows $W_{enc}$ to operate on normalized hidden states. </d-footnote>: 
$$
\begin{equation}
h = \text{ReLU}((z - b_{dec})W_{enc} + b_{enc})
\end{equation} 
$$
The decoder then reconstructs the original input: 
$$
\begin{equation}
\text{SAE}(z) = h W_{dec} + b_{dec}
\end{equation} 
$$
The loss function 
$$
\begin{equation}
\cL(x) = \|\mbz - \text{SAE}(\mbz)\|_2^2 + \alpha \|\mbh(\mbz)\|_1
\end{equation}
$$ 
balances reconstruction accuracy with sparsity, where the $$ L_1 $$ norm term ensures each dimension captures a distinct, interpretable feature.

This method has demonstrated remarkable success in understanding large language models like Claude 3 Sonnet <d-cite key="templeton2024scaling"></d-cite>, GPT-4 <d-cite key="gao2024scaling"></d-cite>, Gemma 2 <d-cite key="lieberum2024gemma"></d-cite> and LLaMA-3.1 <d-cite key="he2024llama"></d-cite>, leading to innovative variants such as Transcoders <d-cite key="dunefsky2024transcoders"></d-cite> and CrossCoders <d-cite key="lindsey2024sparse"></d-cite>. These applications have revealed previously hidden patterns in how language models process and represent information.

More recently, researchers have begun exploring SAEs' potential in Vision Transformers (ViTs) <d-cite key="joseph2023vit"></d-cite><d-cite key="ewingtonpitsos2024suite"></d-cite><d-cite key="DBLP:conf/eccv/RaoMBS24"></d-cite><d-cite key="hugo2024towards"></d-cite>. Despite computational challenges, early results suggest SAEs can efficiently extract interpretable image features with less data compared to their applications in language models, opening new avenues for understanding visual processing in neural networks.

<aside class="l-body box-note" markdown="1">
Key Takeaways: 
- SAEs address the superposition problem by transforming complex neural representations into interpretable features
</aside>


### Automated explanation

While traditional explanation methods focus on highlighting important features in the model's input space, users often care more about understanding the underlying meaning of these features. Automated explanation methods aim to bridge this gap by **translating abstract mathematical representations within neural networks into human-understandable concepts**, without heavily relying on manual analysis. Currently, there are two main approaches to endow such concepts: text-image space alignment and data distribution-based methods.

#### Text-Image Space Alignment

Language serves as a naturally interpretable interface for humans and forms our concept vocabulary, while image representations are inherently less interpretable. The core principle behind Text-Image Space Alignment methods is to establish meaningful connections between neural network's visual features and natural language descriptions. By **mapping neural activations into a shared semantic space** where both textual and visual information coexist, these methods can automatically discover and explain the concepts that drive model behavior.

Recent works have made progress in this direction. Rao et al. <d-cite key="DBLP:conf/eccv/RaoMBS24"></d-cite> developed a method using Sparse Autoencoders (SAEs) that learns interpretable representations from neural network activations. Their approach analyzes the decoder's weight matrix by computing cosine similarities between its columns and word embeddings, effectively translating network representations into human-understandable concepts. SpLiCE <d-cite key="bhalla2024interpreting"></d-cite> proposed a complementary strategy by first establishing a comprehensive semantic concept vocabulary. This method seeks the sparsest possible mapping between these concepts and the CLIP embedding space while maintaining high cosine similarity, ensuring that the identified concepts are both minimal and maximally informative.

<details markdown="1">
<summary><b>An Illustrative Graph</b></summary>
<object data="{{ 'assets/img/2025-04-28-vlm-understanding/concept.svg' | relative_url }}" type="image/svg+xml" width="90%" class="l-body rounded z-depth-1 center"></object>
Find concepts to match with the model's internal representations.
</details>

Beyond using cosine similarity, another approach came from Gandelsman et al. <d-cite key="gandelsman2023interpreting"></d-cite>, who introduced TEXTSPAN - an algorithm that creates a text-labeled basis for attention head outputs in CLIP's vision encoder. By caching vision encoder attention head outputs and strategically selecting text embeddings from a predefined text bank <d-footnote>To be more specific, strategically means TEXTSPAN greedily selects text embeddings from the text bank to maximize the explained variance.</d-footnote>, TEXTSPAN revealed specialized attention heads that capture distinct image properties like "color" and "counting". This discovery enabled targeted interventions for reducing spurious correlations and improving property-based image retrieval.

Building upon this foundation, Balasubramanian et al. <d-cite key="balasubramanian2024decomposing"></d-cite> extended TEXTSPAN's applicability beyond CLIP to ViTs. Their proposed automated representation decomposition method to analyze the computational graph generated during the forward pass. Using this method, they break down internal contributions of models into their final representation and mapping these components to CLIP space, where they then use TEXTSPAN for text-based interpretation.

#### Data Distribution-Based Methods
<object data="{{ 'assets/img/2025-04-28-vlm-understanding/automated.svg' | relative_url }}" type="image/svg+xml" width="80%" class="l-body rounded z-depth-1 center"></object>
<div class="l-gutter caption" markdown="1">
Comparison between supervised categorical analysis and unsupervised automated explanation using LLM/VLM.
</div>

Another effective approach to understanding neural networks' internal mechanisms involves analyzing neuron activation patterns across different input types. This method examines **how neurons respond to various categories of data, revealing specialized neurons and their roles in information processing.** This approach can be broadly categorized into supervised and unsupervised methods.

1. **Supervised Approaches**
- Supervised approaches utilize concept-labeled data to guide the interpretation of neural network components. We're essentially looking for some components that consistently "light up" (activate strongly) when presented with specific types of input. For example, if a specific neuron consistently shows high activation when the network processes a particular category of input (such as images of cats) but remains relatively inactive for other categories, we can classify this neuron as specialized for detecting that category.
- **Neuron Specialization**: Recent research has applied this approach to both language and vision-language models. Tang et al. <d-cite key="tang2024languagespecific"></d-cite> identified *language-specific neurons* in LLMs, while MMNeuron <d-cite key="huo2024mmneuron"></d-cite> made interesting discoveries about *domain-specific neurons* in vision-language models like LLaVA and InstructBLIP. They found that deactivating domain-specific neurons, while significantly perturbing hidden states, doesn't always impact task performance. This suggests that VLMs fail to take full advantage of the domain-specific information in specific domains, which means **VLMs may rely on highly generalized internal representations.** Miner <d-cite key="huang2024miner"></d-cite> further refined this methodology to find *modality-specific neurons*. They also reveal that **modality-specific neurons are primarily concentrated in shallow layers, with most modality information remaining within its original token set.**

2. **Unsupervised Discovery**
- While supervised approaches provide clear and verifiable concept mappings, their reliance on labeled data can limit scalability and may miss concepts not included in the predefined set. Unsupervised discovery methods take a more data-driven approach, identifying meaningful patterns in network activations without requiring concept labels. These approaches typically analyze how different network components respond to various inputs, using techniques like clustering or dimensionality reduction to group similar activation patterns. Recent advances have **integrated language models or vision-language models to automatically generate natural language descriptions of discovered patterns.** This approach offers greater flexibility in concept discovery and can uncover unexpected patterns that might be missed by supervised methods. However, the challenge lies in ensuring the discovered concepts are meaningful and reliable for practical applications <d-cite key="DBLP:conf/blackboxnlp/HuangGDWP23"></d-cite>.
- Recent advances have leveraged large language models like GPT-4 to automatically generate natural language descriptions of discovered patterns <d-cite key="hernandez2022natural,singh2023explaining,bills2023language"></d-cite>. In VLMs, a notable example is MAIA <d-cite key="DBLP:conf/icml/ShahamSWRHA024"></d-cite>, which automates interpretability tasks by composing pretrained modules to conduct experiments on other systems. Given an interpretability query (e.g., "Which neurons in Layer 4 are selective for forested backgrounds?"), MAIA runs experiments to test specific hypotheses, observes outcomes, and iteratively updates its understanding until it can answer the user query.

<aside class="l-body box-note" markdown="1">
Key Takeaways: 
- Automated explanation methods aim to bridge the gap between abstract neural network representations and human-understandable concepts, primarily through two approaches: text-image space alignment and data distribution-based methods.
</aside>




## Future Directions

  

While the above mechanistic interpretability studies have provided significant insights into how vision-language models (VLMs) function, several challenges remain. This section discusses and summarizes these challenges and proposes potential directions for future research.

  

### From Single Model to Multiple Models

  

**Current Situation**: Unlike large language models (LLMs), vision-language models (VLMs) exhibit much greater heterogeneity in terms of architectures, data, and training paradigms. For instance, VLMs can differ significantly in their vision encoders <d-cite key="li2023monkey,ye2023ureader,xue2024xgenmm"></d-cite>, language models <d-cite key="laurençon2024matters"></d-cite>, and the connectors between them—ranging from simple linear layers to visual resamplers or cross-modal attention mechanisms <d-cite key="liu2023visual,NEURIPS2022_960a172b,awadalla2023openflamingo,DBLP:conf/nips/LaurenconSTBSLW23,dubey2024llama"></d-cite>. They also vary in their training data, which may include image-captioning datasets, visual instruction tuning data, or interleaved image-text datasets <d-cite key="laurençon2024building"></d-cite>. Additionally, their training paradigms differ, such as whether they perform alignment <d-cite key="rafailov2023direct,sun2023aligning,yu2023rlhfv,chen2023dress"></d-cite>, or whether the vision encoder is frozen or fine-tuned during training <d-cite key="Qwen-VL,Qwen2VL,lu2024deepseekvl"></d-cite>. This substantial heterogeneity may limit the transferability of findings if interpretability studies are only conducted on a single model.

  

**Path Forward**: Conducting cross-model analyses is essential to verify conclusions across different VLMs and ensure their generalizability. This approach can help identify universal principles applicable across various VLMs, as well as model-specific insights that could lead to tailored improvements.

  

<aside class="l-body box-note" markdown="1">

Summary:

- Perform cross-model analyses to validate findings across different VLMs and ensure their generalizability.

</aside>

  
  

### From Small Models to Large Models

  

**Current Situation**: Current interpretability research in VLMs primarily focuses on smaller-scale models, such as those with 7B parameters. However, larger VLMs often exhibit emergent capabilities that are absent in smaller models. These new capabilities may pose unique challenges for applying interpretability tools to larger models.

  

**Path Forward**: Scaling up interpretability studies to include larger models is critical for understanding how these tools perform at scale and what new insights they might uncover <d-cite key="templeton2024scaling,gao2024scaling"></d-cite>. This effort can deepen our understanding of emergent behaviors and inform the development of interpretability methods suitable for larger models.

  

<aside class="l-body box-note" markdown="1">

Summary:

- Scale up interpretability studies to include larger VLMs to understand emergent capabilities and challenges.

</aside>

  

### From Language-Centric to Vision-Centric

  

**Current Situation**: VLMs differ from LLMs in their handling of visual information. While many LLM interpretability tools have been successful in explaining text-based mechanisms <d-cite key="neo2024towards"></d-cite>, applying these tools directly to VLMs may not suffice due to the richer, more ambiguous nature of visual information <d-cite key="joseph2023vit"></d-cite>. Furthermore, VLMs incorporate vision encoders, language models, and connectors between them, adding layers of complexity to interpretability studies.

**Path Forward**: Developing tools specifically designed for visual contexts is necessary to address the unique challenges posed by vision-based features. Meanwhile, these tools should consider the intricate architectures of VLMs and prioritize analyzing the vision components and vision-language connectors, ensuring that interpretations are accurately attributed to the visual inputs. Additionally, input data used for interpretability should emphasize vision-centric tasks that cannot be easily solved by text-only models, ensuring meaningful insights into how VLMs process visual inputs.

  
<aside class="l-body box-note" markdown="1">

Summary:

- Develop interpretability tools tailored for visual contexts and apply them to vision components to understand how VLMs process visual information.

</aside>

  
  

### From Static Processes to Dynamic Processes

  

**Current Situation**: Interpretability studies often focus on a single checkpoint of a model, ignoring the dynamic nature of information flow during training. For example, VLM training typically involves multiple stages, such as initial alignment using image-captioning data (where only the vision-language connector is tuned) followed by end-to-end fine-tuning with diverse instruction-tuning data. These stages may include phase changes <d-cite key="wei2022emergent"></d-cite> where models gain new capabilities or behaviors, such as transitioning from unimodal pre-trained models to multimodal systems.

  

**Path Forward**: Studying the dynamics of VLM training is crucial for uncovering novel insights. Applying interpretability tools at different checkpoints during training can shed light on phase changes and the evolution of information flow. Insights from these dynamic studies could also resonate with cognitive science research, such as experiments on restoring vision in previously blind individuals.

  

<aside class="l-body box-note" markdown="1">

Summary:

- Study the dynamics of VLM training to understand phase changes and information flow evolution.

</aside>

  
  

### From Micro-Level to Macro-Level

  

**Current Situation**: Interpretability research often focuses on micro-level phenomena, such as individual neurons or layers, to understand how VLMs process information. However, these findings are rarely connected to macro-level behaviors, such as performance variations across tasks or model designs. For example, recent studies show that CLIP/SigLIP vision encoders pre-trained on image-text data outperform those trained purely on images such as DINO when building VLMs <d-cite key="tong2024eyes"></d-cite>. However, the underlying reasons for these differences remain unclear. Similarly, VLMs can struggle with seemingly simple vision-centric tasks like image classification, despite their vision encoders excelling in such tasks <d-cite key="zhang2024visuallygrounded"></d-cite>.

  

**Path Forward**: Bridging the gap between micro-level findings and macro-level behaviors is essential for driving advancements in VLM development. Applying interpretability tools to investigate unresolved macro-level questions—such as why certain vision encoders perform better or why VLMs struggle with specific tasks—can yield actionable insights. For example, probing tools have been employed to link VLM failures on vision-centric tasks to limitations in the vision encoder. Such findings can inform the design of improved vision encoders, potentially combining the strengths of models like CLIP and DINO to overcome these shortcomings.

  

<aside class="l-body box-note" markdown="1">

Summary:

- Connect micro-level insights from interpretability studies to macro-level behaviors to inform VLM design and development.

</aside>


## Conclusion

This work provides a comprehensive review of studies leveraging mechanistic interpretability tools to analyze vision-language models (VLMs), including probing techniques, activation patching, logit lenses, sparse autoencoders, and automated explanation methods. These tools have greatly enhanced our understanding of how VLMs represent, integrate, and process multimodal information. Despite these advancements, several key challenges remain. These include the need for validation across a wider range of VLM architectures and training paradigms, a deeper exploration of information flow dynamics throughout training stages, and a stronger alignment between micro-level insights and macro-level behaviors. Addressing these challenges will pave the way for developing more robust and effective VLMs, advancing both their design and practical applications.


<!-- ## Future Directions

1. **Towards Universality**
- As mechanistic interpretability matures, the field must transition from isolated empirical findings to developing overarching theories and universal reasoning primitives beyond specific circuits, aiming for a comprehensive understanding of AI capabilities <d-cite key="bereska2024mechanistic"></d-cite>. This is especially important to VLMs since architecture design (self-attention, cross attention etc.) and vision transformers come in many forms (CLIP, vanilla ViT and DINO etc.). Experiments are conducted mainly on LLaVA-NeXT and InstructBLIP may not be directly applicable to models that utilize different frameworks <d-cite key="huo2024mmneuron"></d-cite>.

2. **Develop more robust methods to validate interpretability techniques for VLMs**
  - First, we need to rigorously examine whether interpretability methods from LLMs remain effective in VLMs. While Neo et al. <d-cite key="neo2024towards"></d-cite> found that logit lens works surprisingly well on VLMs despite their different training objectives, VLMs' distinct architectural features - such as bidirectional attention, CLS token usage, and patch-based processing - may fundamentally alter how these methods operate or what they reveal <d-cite key="joseph2023vit"></d-cite>.
  - Second, we must address conflicting findings in current VLM interpretability research. For instance, Golovanevsky et al.'s <d-cite key="golovanevsky2024vlms"></d-cite> observation that LLaVA shows consistent logit differences across modalities contradicts earlier findings about textual prior dominance <d-cite key="Salin2022AreVT"></d-cite>. Similarly, Neo et al.'s <d-cite key="neo2024towards"></d-cite> discovery of highly localized object information challenges assumptions about register tokens <d-cite key="DBLP:conf/iclr/DarcetOMB24"></d-cite>. These contradictions highlight the need for more reliable validation approaches.

3. **Understand the complex reality beyond semantic concepts**
- Recent research challenges the fundamental assumption that vision features in AI models can be easily explained through semantic concepts. Unlike language with its standardized vocabulary, visual information lacks a clear "dictionary" of concepts, as a single visual element can simultaneously represent multiple meanings at different levels of abstraction <d-cite key="joseph2023vit"></d-cite>. Current interpretability methods, including saliency maps and counterfactuals, often produce explanations that seem plausible to humans but may not faithfully reflect the model's inner workings <d-cite key="atanasova-etal-2023-faithfulness"></d-cite><d-cite key="agarwal2024faithfulness"></d-cite><d-cite key="madsen2024selfexplanations"></d-cite>. Even semantic decomposition itself has limited applicability, requiring specific conditions such as aligned image and text encoders in CLIP <d-cite key="bhalla2024interpreting"></d-cite> - a constraint that underscores the broader challenge of bridging modality gaps <d-cite key="NEURIPS2022_702f4db7"></d-cite>. The situation is further complicated by model representations being brittle and dataset-specific, often encoding idiosyncratic patterns rather than general, human-interpretable semantic information <d-cite key="bhalla2024interpreting"></d-cite>. These findings collectively indicate the need for more sophisticated methods to understand how visual information is processed and represented in AI models.

4. **SAEs on VLMs**
- Recent work has demonstrated the versatility of Sparse Autoencoders (SAEs) in various tasks including probing, classification, and model steering <d-cite key="kantamneni2024sae"></d-cite><d-cite key="bricken2024using"></d-cite><d-cite key="o'brien2024steering"></d-cite>. However, their application in Vision-Language Models (VLMs) remains relatively unexplored, suggesting significant potential for future research. Notably, existing studies on training SAEs for models like CLIP have used relatively small datasets, yet have achieved promising results <d-cite key="hugo2024towards"></d-cite>. This efficiency in data usage hints at the possibility of scaling SAE training to more diverse VLMs and larger datasets, potentially uncovering new insights about model interpretability and behavioral patterns across different architectures and modalities.

5. **Feature dynamics**
- Understanding feature dynamics in neural networks presents several critical research directions. First, in VLMs, while we know that domain-specific information is not fully utilized, the mechanisms of how features are conveyed or discarded between layers remain poorly understood <d-cite key="huo2024mmneuron"></d-cite>. Second, the process of feature formation during training deserves deeper investigation, particularly given the phenomenon of phase changes - where capabilities can emerge abruptly during training or scaling <d-cite key="wei2022emergent"></d-cite>. These phase changes have significant implications for AI safety, as undesired behaviors could similarly emerge unexpectedly. Studying these transitions provides a unique opportunity to bridge microscopic interpretability with macroscopic scaling laws, potentially offering insights crucial for ensuring safe and controlled development of AI systems <d-cite key="olsson2022context"></d-cite>.

6. **Think more about the big picture**
- While microscopic interpretability studies have provided valuable insights, there's a pressing need to move beyond merely documenting individual model behaviors. Instead, research should focus on translating these findings into actionable improvements in AI system development and deployment, bridging the gap between theoretical understanding and practical applications <d-cite key="DBLP:conf/emnlp/MosbachGBKG24"></d-cite>.

## Conclusion
Vision-Language Models (VLMs) have demonstrated remarkable capabilities in integrating visual and linguistic information. Our survey has examined VLM research through both macro and micro perspectives, revealing our current understanding and future challenges.

At the macro level, while we've seen rapid empirical progress in architectures and training methods, theoretical understanding remains limited. At the micro level, interpretability research has begun illuminating VLM internals through various techniques - from Concept Bottleneck Models to Sparse Autoencoders. However, significant challenges remain, including the need to move beyond isolated findings toward comprehensive theories and the development of more robust validation methods.

The path forward requires balancing empirical advances with deeper theoretical understanding, particularly in understanding how these models process and integrate multimodal information. Success in this endeavor will be crucial for developing more reliable, interpretable, and capable VLMs. -->