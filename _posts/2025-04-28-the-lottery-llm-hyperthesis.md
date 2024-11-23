---
layout: distill
title: The Lottery LLM Hyperthesis
description: In this blog, we present a brief review of some of the recent progresses of LLM related to multi-step reasoning, computational expressivity, external tools, retrieval augmented generation and model compression. We propose a lottery LLM hypothesis that indicates for a given LLM and a task, there exists a small lottery LLM capable of producing the same performance with the original LLM with the assistances of multi-step reasoning and external tools. Introducing a lottery LLM helps to reduce the computational and storage costs. We derive and summarize the crucial abilities that the lottery LLM must possess, hoping to shed some light on the possibility of AGI, which does not need to be a very large LLM as long as the LLM owns some crucial abilities and with the help of external tools and advanced algorithm design.

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

# must be the exact same name as your blogpost
bibliography: 2025-04-28-the-lottery-llm-hyperthesis.bib

# Add a table of contents to your post.
#   - make sure that TOC names match the actual section names
#     for hyperlinks within the post to work correctly. 
#   - please use this format rather than manually creating a markdown table of contents.
toc:
  - name: Equations
  - name: Images and Figures
    subsections:
    - name: Interactive Figures


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

<!-- <d-cite key="aaa"></d-cite> -->
Motivated by reducing the computational and storage costs of LLMs, model compression and KV cache compression have attracted much attentions of researchers. However, previous methods only focus on guaranteeing the performance of compressed LLMs with the similar perplexity on some common sense knowledge QA tasks and the basic arithmetic reasoning tasks. In this blog, we present a brief review of some of the recent progresses of LLM related to retrieval augmented generation, multi-step reasoning, external tools and computational expressivity which significantly improve the performance of LLMs. Then, we propose a lottery LLM hypothesis suggesting that for a given LLM and task, there exists a smaller lottery LLM capable of producing the same performance with the original LLM with the assistances of multi-step reasoning and external tools. We derive and summarize the essential capabilities that the lottery LLM must possess.




 in this blog, we present a brief review of some of the recent progresses of LLM related to multi-step reasoning, computational expressivity, external tools, retrieval augmented generation and model compression. Then, we propose a lottery LLM hypothesis suggesting that for a given LLM and task, there exists a smaller lottery LLM capable of producing the same performance with the original LLM with the assistances of multi-step reasoning and external tools. We derive and summarize the essential capabilities that the lottery LLM must possess,

Introducing a lottery LLM helps to reduce the computational and storage costs. 


We derive and summarize the crucial abilities that the lottery LLM must possess, 



## Huge Computation and Storage Costs of LLMs
LLMs have shown their remarkable ability in natural language conversations,  xxxxxxxxxxxxxxx <d-cite key="aaa"></d-cite>.
To learn the tremendous knowledge in the training datsests, the current advanced LLMs like xxxxx <d-cite key="aaa"></d-cite>have enoumous parameters like $7 \sim 750$ billion. Training such a LLM consumes xxx GPU days with the advanced NVIDIA GPUs<d-cite key="aaa"></d-cite>. This results in the huge electricity consumption, not only occupies the economic costs but also the energy costs<d-cite key="aaa"></d-cite>, raising concerns of the green computing<d-cite key="aaa"></d-cite>. Moreover, providing inference services of LLMs also requires many GPUs and consumes more energy cosits.

To this end, both academic researchers and industrial engineers are trying to compress model pararameters and reduce the model into a smaller one while keeping its performance not change. The typical compression algorithm includes the pruning<d-cite key="aaa"></d-cite> and quantization<d-cite key="aaa"></d-cite>. However, most of the current LLM compression methods only show guaranteed performance with small dataset or the traditional datasets in papers but not in the real-world industrial scenarios. Some recent studies show that the low-precision representations of LLMs will significantly reduce the LLM performance<d-cite key="aaa"></d-cite>, or reduce some crucial abilities of LLMs<d-cite key="aaa"></d-cite>.

Instead of compression a pretrained model, researchers resort to train a small LLM from scratch<d-cite key="aaa"></d-cite>. However, there is still no such a real-world deployed small LLM that has the same performance of LLMs.

Besides the model parameters, some academic works focus on compressing the intermediate features including the KV cache<d-cite key="aaa"></d-cite> and activations from FFNs<d-cite key="aaa"></d-cite> to reduce the computing and storage costs during inference. However, some recent studies show that the LLMs may lose their key abilities under these compressions like xxxxxxxxxxx <d-cite key="aaa"></d-cite>.




## Significant Computational and Storage Costs of LLMs
LLMs have demonstrated remarkable proficiency in natural language processing, enabling sophisticated interactions and understanding of human language. However, to assimilate the vast knowledge contained within training datasets, contemporary advanced LLMs, such as GPT-3 and its successors, comprise an enormous number of parameters, ranging from $7$ to $750$ billion. Training such an LLM requires extensive computational resources, often measured in thousands of GPU days using advanced NVIDIA GPUs. This results in substantial electricity consumption, impacting both economic and energy costs, and raising concerns regarding sustainable computing. Furthermore, providing inference services for LLMs necessitates numerous GPUs and incurs additional energy costs, making it a significant challenge for widespread deployment.

Consequently, both academic researchers and industry engineers are endeavoring to compress model parameters and reduce the model size while maintaining performance. Typical compression algorithms include pruning, which removes redundant parameters, and quantization, which reduces the precision of the model's weights. However, most current LLM compression methods demonstrate assured performance only with small datasets or traditional datasets in academic papers, but not in real-world industrial scenarios. Recent studies indicate that low-precision representations of LLMs can significantly degrade performance, particularly in tasks requiring nuanced understanding or complex reasoning, or diminish some crucial capabilities of LLMs, such as their ability to generalize across diverse contexts.

Instead of compressing a pretrained model, researchers are exploring the training of a small LLM from scratch. This approach aims to inherently design models that are efficient and effective without the need for extensive post-training modifications. However, there is still no real-world deployed small LLM that matches the performance of larger LLMs, highlighting the ongoing challenges in this area of research.

In addition to model parameters, some academic works focus on compressing intermediate features, including the key-value (KV) cache and activations from feedforward networks (FFNs), to reduce computational and storage costs during inference. However, recent studies suggest that LLMs may lose key capabilities under these compressions, such as their ability to maintain context over long conversations or perform complex multi-step reasoning, which are critical for many applications.



```
Some tables and figures here to show the performance limitation of compressed LLMs.
```

Nevertheless, next, with reviewing some progresses of retrival augmented generation, external tool use, and multi-step reasoning, we show the possiblity of allowing small LLMs to own the same performance of their original LLMs.

## Knowledge Retrival: Tackling Redundant and Unreal Knowledge of LLMs

**Redundant Knowledge.** Nowadays, many people even use LLMs to search knowledge like a 百科全书, or check some news or academic research papers like a Internet search engine. Some recent studies show that LLMs have divergence performance on different knowledge retrivals based on the popularity of these knowledge <d-cite key="aaa"></d-cite>. Specifically, many real-world quesion-answer (QA) pairs occupy the small portion of the all QAs, while a small part of QAs are frequently focused by people, showing a long-tail distribution of their popularity<d-cite key="aaa"></d-cite>. LLMs show the higher performance on the high-popularity QAs while low performance on low-popularity QAs. 
**Hallucinated Knowledge.** However, LLMs ofter response the imaginary outputs sampled by its probabilistic language modeling, instead of the factual knowledge that exist in the real world<d-cite key="aaa"></d-cite>. This phenomenon called hallucination has raised significant attentions of researchers<d-cite key="aaa"></d-cite>. There are debates on whether we can completely eliminate it<d-cite key="aaa"></d-cite>, or the hallucination will always exist and cannot be addressed<d-cite key="aaa"></d-cite>. Some studies show that the hallucination itself 必然存在 because it is 大模型推理泛化能力的共生物.


**Retrival Augmented Generation (RAG).** As LLMs have the strong in-context learning ability, they can rely on prompts to response questions instead of only their inner knowledge that are stored in the model parameters. Thus, the external knowledge such as articles, webpages, books and other documents can be inserted into the prompts to allow LLMs to retrive extra factual knowledge<d-cite key="yao2022react"></d-cite> that does not exist in model parameters or avoid hallucination<d-cite key="yao2022react"></d-cite>. Intuitively, the usage of RAG inspires the following research questions:
<blockquote>
Do we need to store all knowledge into LLM parameters if we can use RAG to accurately retrive the factual knowledge from the external knowledge base? If no, which knowledge are needed and which not?
</blockquote>

Considering two extreme cases:
- Storing all knowledge in **model parameters**. If we store all knowledge in the model parameters, the LLMs serve as an oracle machine<d-cite key="aaa"></d-cite> and there is no need to use the RAG. However, it is almost impossible to train such an LLM and the model size will be too large to be deployed.
- Storing all knowledge in **external knowledge base**. If we store all knowledge in the external knowledge base, maybe we can compress the LLM parameters to a very small size and retrive any factual knowledge during inference.

However, LLMs require some basic common knowledge to implement current amazing abilities like reasoning, accurate retrival and so on. We will discuss this problem in the subsequent sections. Thus, compressing all knowledge into the external knowledge base is infeasible. Studying how the learned knowledge and which knowledge to trigger the grokking phenomenon of LLMs is still an open problem<d-cite key="aaa"></d-cite>.

**Trade-off between model size and knowledge base.** Some studies show that the adaptive knowledge retrival is a promising direction to improve the performance of LLMs and may help to find the good trade-off between the knowledge base and model size<d-cite key="aaa"></d-cite>. The adaptive RAG<d-cite key="aaa"></d-cite> shows that the popular knowledge can be stored in the model parameters while the less popular knowledge can be stored in the external knowledge base. 


The core idea of adaptive RAG seems to be related to a classic efficient data structure, the **huffman coding**<d-cite key="aaa"></d-cite>. Specifically, we can view the cost of the knowledge retrival as the prompt length (because the retrived knowledge will be inserted into the prompts). Then, storing knowledge in the model parameters will have a shorter prompt length because LLMs can directly response to questions without need to retrive knowledge from the external knowledge base. And storing the knowledge in the external knowledge base will have a longer prompt length, in other words, the cost of more retrival operations and the longer context length which causes higher computation and storage costs during inference<d-cite key="aaa"></d-cite>. Therefore, the popularity of the knowledge can be seen as the appearance probability as in the Huffman coding. Storing the popular knowledge in the model parameters is more efficient.


**Finetuning v.s. retrival.** Another parallel related question is that whether we should use the finetuning to improve the performance of LLMs in some specifical application domains like legal, finance, medical and so on<d-cite key="aaa"></d-cite>. Because the finetuning may lead to the forgetting problem and the extra training overheads, there is debate on whether we should use the finetuning to improve the performance of LLMs or we can rely on the RAG to achieve the same goal<d-cite key="aaa"></d-cite>. Some recent studies show that the RAG can significantly improve the performance of LLMs on the specific domains<d-cite key="aaa"></d-cite>, achieving similar performance of finetuned LLMs.




**Beyond the RAG.** The document-based knowledge retrival mainly help LLMs to retrive the knowledge of triplets consisting of entity, relation and object like the konwledge graph<d-cite key="aaa"></d-cite>. However, the ability and superb performance of LLMs does not only include retriving the triplet knowledge. LLMs also show amazing abilities like solving the arithmetic problems, playing chess, coding and so on, which are not simple triplet knowledge retrival<d-cite key="aaa"></d-cite>. How to guarantee the reasoning performance of small LLMs is important and cannot be easily addressed bythe document-based knowledge retrival.


<!-- 

Presumably this means that the current LLM probably remembers a lot of knowledge that doesn't need to be remembered, such as what unimportant news happened on a certain day in the training data, and such knowledge can be assisted by the RAG. So there are a lot of redundant parameters that can be thrown out of the big LLM, and we can make the model much smaller by using a small model with the help of external knowledge base/external tool calls.

Then, for the single-step reasoning capability of the big model, the small model may be able to equate it with multi-step reasoning once it has some core capabilities (reference1: the small model is equivalent to a Turing machine with the aid of external memories, reference2: through GoT/agents collaboration/neural symbolic).

We then summarize our conjectures in the paper by guessing what key capabilities the mini-model should have in order to achieve this effect. For example, it would need a good ability to retrival relevant information from context, some ability to copy, rewrite operations (necessary for Turing completeness), and some common sense knowledge (e.g., the ability to recognize which function calls are provided to add, subtract, multiply, or divide two numbers). -->

## External Tools
The advanced LLMs show their remarkable abilities in function calling, which is the ability to call external tools to solve some specific tasks. The external tools can be the Internet search engine<d-cite key="ToolLLM"></d-cite>, the arithmetic calculation functions<d-cite key="NEURIPS2023_d842425e"></d-cite>, system operations<d-cite key="LLM-as-OS"></d-cite><d-cite key="AIOS"></d-cite>, game interfaces and so on, which are formulated into programming function calls<d-cite key="Granite-Function"></d-cite> and passed to LLMs using prompts. According to the function descriptions, LLMs decide to call which function to solve the given problems<d-cite key="Granite-Function"></d-cite>.


<!-- {% include figure.html path="assets/img/2025-04-28-the-lottery-llm-hyperthesis/Tooluse.jpg" class="img-fluid" %} -->

<div class="row mt-3">
    <div class="col-sm mt-3 mt-md-0">
        {% include figure.html path="assets/img/2025-04-28-the-lottery-llm-hyperthesis/Tooluse.jpg" class="img-fluid rounded z-depth-1" %}
    </div>
</div>
<div class="caption">
    The ability of function calling in <d-cite key="NEURIPS2023_d842425e"></d-cite>. The smaller models like GPT-2 finetuned with function calling datsets show much better performance with the large models (GPT-3) that are not specifically tuned with function calling datsets.
</div>

**Arithmetic function callings.** To solve the arithmetic problems, LLMs are trained on the arithmetic datasets<d-cite key="aaa"></d-cite>. However, some simple errors ofter happen during the arithmetic reasoning process like that LLMs may think the 9.11 is larger than 9.9 <d-cite key="aaa"></d-cite>. To this end, some works propose to allow LLMs to generate programs including the arithmetic operations and exploit the external Python interpreter to solve the arithmetic problems<d-cite key="pmlr-v202-gao23f"></d-cite>. Besides, some works propose to exploit the arithmetic function calling to solve the arithmetic problems<d-cite key="he23solving"></d-cite>. The experimental results show that the arithmetic function calling can significantly improve the performance of LLMs on the arithmetic problems<d-cite key="pmlr-v202-gao23f"></d-cite><d-cite key="NEURIPS2023_e3936777"></d-cite>.

**Internet search engine.** To augment LLM knowledge with the online and dynamic updated external knowledge, the Internet search engine is exploited as the external tool<d-cite key="yao2022react"></d-cite><d-cite key="FreshLLMs"></d-cite>. The experimental results show that interacting with the Internet search engine like a simple Wikipedia API can significantly improve the performance of LLMs on the knowledge retrival tasks<d-cite key="yao2022react"></d-cite>.






**LLM Operation System (OS).** By viewing the LLM calls as the system call like the traditional operational system, some recent studies propose to build a new *LLM-as-OS* framework<d-cite key="LLM-as-OS"></d-cite>, which allows LLMs to call the external tools like the apps in the OS. And the recent studies proposes to build the AIOS framework<d-cite key="AIOS"></d-cite> to decouple the LLM calls and system calls, and implementing different managers to help improve the AIOS efficiency. The optimized agent framework from the OS perspective significantly improve both the efficiency and the performance of LLM calls.


<d-cite key="AIOS"></d-cite>
<d-cite key="yao2022react"></d-cite>

<d-cite key="LLM-as-OS"></d-cite>




**Logic Solver.** It is still a debate on whether the LLMs can conduct the logical reasoning like the humans<d-cite key="GSM-Symbolic"></d-cite><d-cite key="CanLLMReason"></d-cite><d-cite key="valmeekam2022large"></d-cite><d-cite key="10.1145/3627673.3679832"></d-cite><d-cite key="xu2023large"></d-cite><d-cite key="arkoudas2023gpt4cantreason"></d-cite>. Some recent studies show that to enhance the reasoning ability of LLMs, the external logic solvers can be exploited to solve the logical reasoning problems<d-cite key="RecursiveReasoning"></d-cite>. In some frameworks, LLMs are responsible to transform the natural language sentences into the logical form and the logic solvers are responsible to solve the logical reasoning problems<d-cite key="han2022folio"></d-cite><d-cite key="pan-etal-2023-logic"></d-cite><d-cite key="wang2024symbolic"></d-cite>. While some frameworks propose to let LLMs summarize the sentences into the premises and conclusions, then aggregate these extracted information into another prompt to let LLMs to conduct the logical reasoning<d-cite key="sun-etal-2024-determlr"></d-cite>. Their experiment results show that thought these symbolic and neural combination, the LLMs can significantly improve their reasoning ability<d-cite key="sun-etal-2024-determlr"></d-cite><d-cite key="pan-etal-2023-logic"></d-cite><d-cite key="wang2024symbolic"></d-cite><d-cite key="Xu2024FaithfulLR"></d-cite><d-cite key="RecursiveReasoning"></d-cite>.



## Computational Expressivity of LLMs

**Basic Transformer Architecture.** The basic transformers without the intermediate decoding steps are shown to have a low computational expressivity<d-cite key="10.1162/tacl_a_00562"></d-cite><d-cite key="chiang2023tighter"></d-cite>, the fairly small circuit complexity class $TC^0$<d-cite key="10.1162/tacl_a_00562"></d-cite>. The basic transformers are far
from Turing-complete: they cannot even solve problems complete for classes larger than $TC^0$ such as simulating automata which is $NC^1$-complete.


**Decoding-based Transformers.** The decoding based transformers generate the next word one by one instead of a single answer. The decoding based transformers can be more powerful than the basic transformers. And their computational expressivity increases along with the length of the decoding steps<d-cite key="merrillexpressive">. This also reveals that why the Chain-of-Thought (CoT) reasoning process<d-cite key="CoT"></d-cite> increases the LLMs' computing expressivity<d-cite key="Reveal-CoT"></d-cite>. Some works show that with the linear steps, the transformers with projected-norm can theoretically simulate a Turing automata<d-cite key="merrillexpressive">. A recent work shows that the autoregressive decoding that allows processing of arbitrarily long input string can achieve simulating a universal Turing machine<d-cite key="Autogressive-Turing"></d-cite>.


**Decoding with External Memory.** Some studies show that the external memory can help to increase the LLM's computational expressivity<d-cite key="deletang2023neural"></d-cite> and endow the LLMs with the approximated Turing completeness<d-cite key="JMLRv2220-302"></d-cite>. And recent works develop the Stack-Attention mechanism to further improve the LLMs's reasoning ability<d-cite key="Stack-Attention"></d-cite>. With the help of the external memory and the 
simple regular expression parsers, transformers can simulate the execution of a universal Turing machine, $U_{15,2}$<d-cite key="Memory-Augmented-Turing"></d-cite>.




## Multi-step Reasoning
<d-cite key="RecursiveReasoning"></d-cite>
The chain-of-thought (CoT) reasoning shows that reasoning with detailed progress can significantly improve the performance of LLMs than the single-step reasoning<d-cite key="CoT"></d-cite>. This is because the single-step reasoning may ignore some crucial intermediate steps that are helpful for solving the problems<d-cite key="CoT"></d-cite>. And the multi-step reasoning process 通过参考 humans' thinking process can significantly improve the performance of LLMs<d-cite key="CoT"></d-cite>.

**Single LLM call.** The CoT is the typical example of a single LLM call, which only use the LLM once. Beyond the explicit prompting to trigger LLMs to conduct detailed reasoning, some recent studies propose to let LLMs conduct advanced searching algorithm in the decoding processes like using the Monte-Carlo Tree Search (MCTS)<d-cite key="MTCS-Decoding"></d-cite> or the Q-star search<d-cite key="chakraborty2024transfer"></d-cite>. And some works propose to use the backtracking algorithm to allow LLMs to regret the previous decisions and improve the final performance<d-cite key="fubreak"></d-cite>.

**Multiple LLM calls.** Some works propose to call the LLM multiple times, which have not dependency between each other. And some correct answers can be found in these multiple LLM calls<d-cite key="brown2024large"></d-cite>. Except the single call of CoT, CoT-SC proposes to let LLMs call the LLM multiple times with CoT and select the best answer to improve the final performance<d-cite key="wangself"></d-cite>. However, these answers do have the direct dependency between each other. To better schedule and decompose the reasoning process, tree-of-thought (ToT) reasoning<d-cite key="ToT"></d-cite> and graph-of-thought (GoT) reasoning<d-cite key="GoT"></d-cite> are proposed to decompose the reasoning process into multiple steps and schedule the reasoning steps in a tree-like structure or graph-like structure. Some works also propose to combine the knowledge graph and let LLMs reason on the graph structure to improve the its reasoning ability<d-cite key="luoreasoning"></d-cite><d-cite key="sunthink"></d-cite>. Using LLMs to structurize the prompts into triplet is also helpful to improve the reasoning ability of LLMs<d-cite key="jiang2023structgpt"></d-cite>. Without a centralized controller, some works propose to use LLMs to simulate multiple agents to collaboratively solve the problem<d-cite key="li2023camel"><d-cite key="hong2024metagpt"></d-cite><d-cite key="liang2023encouraging"></d-cite><d-cite key="duimproving"></d-cite>.


**Planning and Scheduling.** The core idea of the multi-step reasoning is to decompose the original problem into multiple sub-problems and solve the sub-problems one by one. This is a kind of planning and scheduling process. To enable autonomous planning and scheduling, some recent studies propose to use LLMs as a meta agent to conduct the planning and scheduling, in which the original problem is decomposed and this meta agent calls other LLMs to solve the sub-problems according to the scheduling<d-cite key="hong2024metagpt"></d-cite><d-cite key="wu2024autogen"></d-cite><d-cite key="zhoulanguage"></d-cite><d-cite key="wangvoyager"></d-cite>. With the help of external symbolic reasoning, the LLMs can also conduct the planning and scheduling process to solve the problems<d-cite key="RecursiveReasoning"></d-cite>.

## Lottery LLM Hyperthesis

Introduce the lottery LLM hyperthesis.

Give an original language model $f_\theta$ which can receive the input with length $n$, and an input problem $q$ with length $m < n$ and groundtruth $\mu$, a performance measure $P(\cdot)$ to evaluate the performance of the model as $P(f_\theta(q), \mu)$, there exists a smaller language model $g_\phi$ with parameters $\phi$ ($|\phi| < |\theta|$) and the same input length $n$, which can solve the problem $q$ with the performance of $f_\theta$ as

$$P(f_\theta(q), \mu) \leq P( \mathcal{A}_{g_\phi, \mathcal{D}, \mathcal{R}, \mathcal{C}, \mathcal{M}}(q), \mu),$$

in which $\mathcal{A}$ is a reasoning algorithm which may consists of one or multiple times of calling $g_\phi$ with different inputs that consisting of original problem $q$, the documents $d \in \mathcal{D}$ retrived from the external knowledge base $\mathcal{D}$ or the function calls $c \in \mathcal{C}$ retrived from the external tools $\mathcal{C}$ with the retriver $r \in \mathcal{R}$. Here the knowledge base $\mathcal{D}$ is a vector database that stores the vector-documents as key-value pairs. $\mathcal{M}$ is the external memory that store the intermediate results.

We describe the procedures of the reasoning algorithm $\mathcal{A}$ as Algorithm 1, in which the original problem $q$ is solved with a divide-and-conquer strategy. Note that such a dynamic divide-and-conquer methodology is general and can cover many current reasoning algorithms.

```
A pseudo code figure here.
```


**Recurive and Dynamic Scheduling.** Algorithm 1 can be generalized to the tree-based reasoning methods like ToT<d-cite key="zhoulanguage"></d-cite><d-cite key="yao2024tree"></d-cite>, because the recursive design covers the tree search and the branch-or-solve mechanism can be decided based on LLMs dynamically. And the Algorithm 1 also generalizes to graph-based reasoning methods like GoT<d-cite key="besta2024graph"></d-cite><d-cite key="luoreasoning"></d-cite><d-cite key="sunthink"></d-cite>, because the interaction between the different LLMs and the external memory $\mathcal{M}$ can be viewed as the combination in the GoT, in which different outputs of nodes are combined together thus construct the graph structure.


```
A figure here about the tree and graph structure.
```


**External Knowledge and Tools.** In each problem solving step, Algorithm 1 will firstly decides whether to directly solve the problem with the help of the external knowledge base $\mathcal{D}$ or the external tools $\mathcal{C}$. If yes, the Algorithm 1 will use the $g_\phi$ to analyse the problem $q$ and find the necessary knowledge or tools to solve the problem. Then, according to the generated requests, the retriever $\mathcal{R}$ will search external knowledge $d \in \mathcal{D}$ or the tool $c \in \mathcal{C}$ to provide the required results. These supplementary results will be combined with the problem $q$ for the model $g_\phi$ to solve. This design helps to cover exploiting the RAG and external tools like the arithmetic calculation functions, the Internet search engine, logic solvers to address the problem $q$ and so on.


**External Memory.** The external memory $\mathcal{M}$ is used to store the intermediate results during the reasoning process. When solving different sub-problems, the intermediate results can be stored in the external memory and be reused in the later steps. With interacting with the external memory, the Algorithm 1 can recover the reasoing methods with working memory like<d-cite key="wang2024symbolic"></d-cite>. Here we do not restrict the form of the function `Divide_and_Conquer` in the Algorithm 1. Throught dedicated design and programming, the recurisve mechanism can be used to implement the basic operations like the MOV, COPY, JUMP, and WRITE and READ the external memory, thus simulating the Turing machine like<d-cite key="Memory-Augmented-Turing"></d-cite>.


```
A figure here about the external memory.
```



<!--  each reasoning step $\mathcal{A}_i$ is a reasoning algorithm that may consists of one or multiple times of calling $g_\phi$ with different inputs. -->
<!-- Here the external tools can be the Internet search engine, the arithmetic calculation functions, system operations and so on. -->

<d-cite key="aaa"></d-cite>

<!-- Based on the procedures of the Algorithm 1 and the previous design,  -->

Most of previous model compression<d-cite key="aaa"></d-cite> and KV cache compression methods<d-cite key="aaa"></d-cite> only focus on the guaranteeing the model performance on the perplexity metric<d-cite key="aaa"></d-cite> or some downstream tasks like the common sense knowledge<d-cite key="aaa"></d-cite> and the basic arithmetic problems<d-cite key="aaa"></d-cite>. From the above analysis and the procedures of the Algorithm 1, we can see that there are some other crucial abilities that the lottery LLM and other compression methods must take for considering. We summarize the crucial abilities that the lottery LLM should have as follows.

**Ability 1: Retrieval from prompts.** Obviously, the useful information in the prompts that related to address the problem $q$ is crucial for the lottery LLM. After collecting the required external results into the prompt, the LLM $q_\phi$ needs to be able to retrieve the required information from the prompt and avoid the interuption of some irrelevant information. This is related to the retrieval ability of the LLM and its measurement test is like the well-known needle-in-the-haystack test<d-cite key="aaa"></d-cite>. We show that there is a simple and interesting method to endow the LLM with advanced retrieval ability with preprocessing prompts.

```
A figure here to show results of the retrival results from our methods.
```


**Ability 2: Identifying External Required Results.** To accurately find out which external resources to exploit, like searching knowledge or calling the external tools, the LLM $q_\phi$ needs to have the ability to understand and associate the problem $q$ and related sub-problems with the resources. Thus, $q_\phi$ needs to have some basic knowledge about the problem $q$ and the external resources. Also, it needs to have remarkable ability to bind questions with the provided resources. Once the external tools are used well, the performance of the small LLM can be improved significantly. The following table show the results of the arithmetic problems with different LLMs and methods. The PAL<d-cite key="pmlr-v202-gao23f"></d-cite> used the external arithemetic calculation functions to solve the arithmetic problems and significantly improve the performance of the small LLM.

<d-cite key="aaa"></d-cite>

<d-cite key="aaa"></d-cite>


|        | GSM8K | SVAMP | ASDIV |  ADDSUB | MULTIARITH |
|--------|:------:|:-----:|:-----:|:------:|:----------:|
| DIRECT Codex | 19.7  |69.9  | 74.0  |  90.9   | 44.0       |
| CoT UL2-20B  | 4.1   |12.6  | 16.9  | 18.2   | 10.7       |
| CoT LaMDA-137B | 17.1  | 39.9  | 49.0  |  52.9   | 51.8       |
| CoT Codex    | 65.6  | 74.8  | 76.9  |  86.0   | 95.9       |
| CoT PaLM-540B | 56.9  | 79.0  | 73.9  |  91.9   | 94.7       |
| CoT Minerva 540B | 58.8  | -     | -     | -      | -          |
| PAL            | **72.0**  | **79.4**  | **79.6**  | **92.5**   | **99.2**       |

Besides, with provided the external documents, the small LLMs show the superb performance in many QA tasks<d-cite key="aaa"></d-cite>.


```
A table here to show results of the RAG.
```

**Ability 3: Planning and Scheduling.** To split the problem $q$ into multiple sub-problems and solve them one by one, the LLM $q_\phi$ needs to have the ability to plan and schedule the sub-problems. This is crucial for the lottery LLM to solve the complex problems. Thus, the LLM $q_\phi$ needs to have a good understanding of the problem $q$ and the sub-problems. However, the details of solving the sub-problems may not be requied for the LLM $q_\phi$. Because the external resources can be used to solve the sub-problems. And the efficient scheduling ability is also important for the lottery LLM for improving the reasoning efficiency.

**Ability 4: Accurately Approximating Basic Operations.** Like illustrated in the section of the computational expressivity of LLMs, to implement the (approximated) Turing completeness, the LLM $q_\phi$ needs to accurately approximate the basic operations like the MOV, COPY, JUMP, and WRITE and READ the external memory<d-cite key="Autogressive-Turing"></d-cite><d-cite key="Memory-Augmented-Turing"></d-cite>. While these operations may not be directly used in the problem solving, they are crucial for the lottery LLM to serve as a possible meta agent<d-cite key="hong2024metagpt"></d-cite>.


**Ability 5: Long-context Reasoning.** In the single-step reasoning, the longer the context length, the more information the LLM $q_\phi$ can receive and use to address the problems. In the multi-step reasoning, the prompt can be seen as a kind of working memory of the meta agent, or say the planner (controller). Each returned results of the solved sub-problems should be injected into the prompt for the later steps. With the increased problem complexity, the depth of the tree of sub-problems also increases. Thus, the LLM $q_\phi$ needs to have a long-context reasoning ability to support the deep tree reasoning<d-cite key="merrillexpressive"></d-cite><d-cite key="Reveal-CoT"></d-cite>.




<d-cite key="he23solving"></d-cite>
<d-cite key="NEURIPS2023_d842425e"></d-cite>
<d-cite key="pmlr-v202-gao23f"></d-cite>






## Discussion and Broader Impact

The main aim of this blog is to shed light on the possible lottery LLM and summarize the crucial abilities that the lottery LLM should have but missed in the current methods of compressing LLMs and KV cache. The discussion about the redundant knowledge in the LLMs also sheds light on the trade-off between the knowledge and the reasoning ability of LLMs. Once we have the lottery LLM, with the external tools and knowledge base and a strong enough algorithm $\mathcal{A}$, there is the possibility that the lottery LLM can serve as a meta agent to work as the AGI like humans. Its external memory may work as the human's long-term memory, the prompt can work as the short-term memory, while the LLM inference process $g_\phi$ works as the basic thinking process. The external tools and knowledge base can be viewed as commonly used supplementary tools in our daily life.





























































