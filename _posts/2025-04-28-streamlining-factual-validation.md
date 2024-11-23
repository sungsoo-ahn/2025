---
layout: distill
title: "Streamlining Factual Validation: A Scalable Method to Enhance GPT Trustworthiness and Efficiency"
description: As the deployment of Large Language Models (LLMs) like GPT expands across domains, addressing their susceptibility to factual inaccuracies, or hallucinations, is core for reliable performance. In this blog post, we introduce a novel framework that enhances retrieval-augmented generation (RAG) pipelines by integrating LLM summarization, DBSCAN clustering, and vectorized fact storage, achieving a 57.7% reduction in storage size while maintaining near-parity in factual and retrieval accuracy on the PubmedQA dataset. By optimizing the intersection of efficiency, accuracy, and scalability, this framework contributes to the broader goal of advancing trustworthy AI for impactful real-world use cases.
date: 2025-04-28
future: true
htmlwidgets: true
hidden: false

authors:
  - name: Anonymous

bibliography: 2025-04-28-streamlining-factual-validation.bib

toc:
    - name: Introduction
    - name: The Challenge of Factual Accuracy in AI
    - name: Limitations of Existing Approaches
    - name: Theoretical Foundations
    - name: "Our Proposal: Fact-Based Validation for GPT Systems"
    - name: Experimentation and Results
    - name: Broader Implications
    - name: Limitations and Future Work
    - name: Conclusion
  
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
Large Language Models (LLMs), such as GPT, have advanced natural language processing by offering incredible fluency and adaptability <d-cite key="brown2020"></d-cite>. Yet, these models are susceptible to "hallucinations"---outputs that are linguistically coherent but factually incorrect <d-cite key="ji2022, maynez2020"></d-cite>. This issue arises because LLMs optimize for contextual fluency rather than accuracy, a trade-off that becomes increasingly problematic in high-stakes applications such as medicine, where precision is necessary <d-cite key="holtzman2019, esteva2019, ullah2024"></d-cite>.

Existing solutions, including fine-tuning, retrieval-augmented generation (RAG), and post-hoc validation, address specific aspects of this problem but often at the cost of scalability or computational efficiency <d-cite key="guu2020"></d-cite>. These methods frequently fail to account for the growing demand to process unstructured data efficiently while maintaining factual accuracy.

In this blog post, we explore an enhancement to RAG pipelines that combines summarization, clustering with DBSCAN, and vectorized fact storage to address challenges in managing unstructured data <d-cite key="ester1996"></d-cite>. By processing expansive datasets into concise, verifiable formats, this approach optimizes storage efficiency while preserving factual accuracy. Evaluated on the PubmedQA dataset, our framework demonstrates how preprocessing can streamline workflows and improve scalability without compromising retrieval precision, offering new possibilities for real-world applications <d-cite key="jin2019"></d-cite>.

## The Challenge of Factual Accuracy in AI

<span style="font-size: 20px; font-weight: bold;">Hallucinations and Trust Deficit</span>

Hallucinations in LLMs arise from their probabilistic design, which prioritizes predicting plausible sequences over verifying factual accuracy <d-cite key="radford2019"></d-cite>. This limitation is magnified by two interconnected factors:

- **Dynamic and Unstructured Contexts**: LLMs are trained on vast datasets often lacking clear structure or schema. Additionally, knowledge evolves---facts that were accurate during training may become outdated, especially in rapidly advancing fields. Without mechanisms to adapt, models perpetuate inaccuracies <d-cite key="mousavi2024"></d-cite>.
- **Compounded Errors in Reasoning**: When initial premises are flawed, LLMs often propagate these errors across multi-step reasoning tasks, amplifying inaccuracies in their outputs <d-cite key="bender2021"></d-cite>. This cascading effect is particularly problematic for complex queries requiring precise, interdependent answers <d-cite key="wei2022"></d-cite>.

These issues are not only technical inconveniences but fundamental barriers to deploying LLMs in sensitive domains. For instance, incorrect treatment suggestions in medical applications or fabricated legal citations could lead to severe consequences <d-cite key="ji2022"></d-cite>.

## Limitations of Existing Approaches

<span style="font-size: 20px; font-weight: bold;">Fine-Tuning</span>

Fine-tuning involves retraining LLMs on domain-specific datasets, improving their factual accuracy within specific contexts <d-cite key="howard2018"></d-cite>. However, it has inherent trade-offs:

- High resource requirements make it impractical for many organizations and individuals <d-cite key="zhang2022"></d-cite>.
- Static knowledge freezes models in time, necessitating constant retraining to stay relevant <d-cite key="bommasani2021"></d-cite>.
- Specialization often comes at the expense of general-purpose utility, limiting versatility <d-cite key="ruder2019"></d-cite>.

<span style="font-size: 20px; font-weight: bold;">Retrieval-Augmented Generation (RAG)</span>

RAG pipelines integrate external knowledge retrieval into the LLM workflow, providing factual grounding by leveraging curated databases or search engines <d-cite key="ibm2023"></d-cite>. While effective in many cases, RAG systems face challenges:

- **Data Dependence**: Retrieved information may be incomplete, outdated, or biased, directly affecting output reliability <d-cite key="lewis2020"></d-cite>.
- **Validation Gap**: RAG systems retrieve information but do not inherently validate it, leaving unresolved conflicts between retrieved facts and the LLM’s internal predictions <d-cite key="karpukhin2020"></d-cite>.
- **Scalability**: As query complexity increases, retrieved information can become overly sparse, and the increasing contextual length can diminish its enhancement for the LLM, ultimately restricting scalability <d-cite key="laban2024"></d-cite>.

<span style="font-size: 20px; font-weight: bold;">Post-Hoc Validation</span>

Post-hoc validation attempts to correct inaccuracies after generation by cross-referencing outputs with external datasets or models <d-cite key="zhong2024"></d-cite>. While valuable, it struggles from inefficiencies:

- It introduces latency, making it unsuitable for real-time applications.
- Validating every output, even when unnecessary, wastes computational resources <d-cite key="guu2020"></d-cite>.
- It provides limited feedback for refining the underlying generative process, addressing symptoms but not root causes <d-cite key="li2024"></d-cite>.

## Theoretical Foundations

<span style="font-size: 20px; font-weight: bold;">Factual Validation: Core Principles</span>

Factual validation ensures that LLM outputs are verifiable and contextually accurate. This approach focuses on:

- **Granularity**: Decomposing responses into short factual units that allows for precise validation at the component level, reducing the likelihood of undetected inaccuracies <d-cite key="maynez2020"></d-cite>.
- **Scalability**: By leveraging vectorized representations, validation systems can efficiently compare outputs against large, structured knowledge bases <d-cite key="karpukhin2020"></d-cite>.

<span style="font-size: 20px; font-weight: bold;">The Role of Vectorization</span>

Vectorization encodes atomic facts into high-dimensional embeddings, facilitating efficient storage, retrieval, and comparison <d-cite key="johnson2017"></d-cite>. This method allows the system to:

- Quickly identify semantically relevant information from knowledge bases <d-cite key="reimers2019"></d-cite>.
- Enable context-aware validation that adapts to the specific demands of a query <d-cite key="guu2020"></d-cite>.

## Our Proposal: Fact-Based Validation for GPT Systems
To address the challenges of ensuring factual accuracy in GPT systems, we propose a framework that preprocesses data into structured and verifiable units with the aim to enhance the reliability of LLM-generated responses while optimizing storage and retrieval efficiency.

<figure style="text-align: center;">
    <img src="{{ 'assets/img/2025-04-28-streamlining-factual-validation/Method.png' | relative_url }}">
    <figcaption style="font-size: 1em;">This diagram illustrates the workflow of the LLM validity judger. The process involves simplifying the LLM response into facts, fetching nearby facts from a vector database, and assessing the validity of the response based on this context. It then generates an overall validity score with an explanation. The diagram also illustrates a separate system called “New Data Input System” for inputting new data into the vector database, which operates independently from the validity judgment process. The fetching system depicted is the same as existing vector database fetching systems, simply shown for clarity.</figcaption>
</figure>

<span style="font-size: 20px; font-weight: bold;">System Architecture</span>

- **New Data Input System**: Fact Extraction and Storage  
  - **Input**: Multiple text chunks containing factual information.  
  - **Output**: Extracted facts embedded as vector representations, stored in a vector database.

- **LLM Validity Judger**: Prompt and Response Validation  
  - **Input**: A GPT prompt and its corresponding response.  
  - **Output**: A validity rating between 0 and 1 indicating the accuracy of the prompt/response pair, along with optional reasoning.

## Implementation Details
<span style="font-size: 25px; font-weight: bold;">New Data Input System</span>

**Fetching Contexts**: All contexts are fetched from the PubmedQA dataset to serve as the input data for preprocessing <d-cite key="jin2019"></d-cite>.

**Summarization and Compression**: Each context is summarized using GPT-4o-mini, compressing the content into concise paragraphs. This step ensures efficient downstream processing and storage <d-cite key="gpt4omini"></d-cite>.

**Embedding with OpenAI Models**: The summarized contexts are embedded into high-dimensional vector representations using OpenAI's text-embedding-3-large model <d-cite key="openaidocs"></d-cite>.

**Clustering with DBSCAN**: The embeddings are grouped into clusters using DBSCAN, which identifies similar contexts without requiring predefined categories. This step is critical for organizing related data efficiently <d-cite key="ester1996"></d-cite>.

**Context Combination and Summarization**: Context summaries within each cluster are combined and re-summarized into unified representations using GPT-4o-mini, further reducing redundancy and optimizing coherence <d-cite key="gpt4omini"></d-cite>.

**Final Embedding and Storage**: The new summaries are re-embedded with text-embedding-3-large <d-cite key="openaidocs"></d-cite>. These embeddings, along with any unclustered individual summaries, are stored in PineconeDB for retrieval <d-cite key="pinecone"></d-cite>.

<span style="font-size: 25px; font-weight: bold;">LLM Validity Judger</span>

**Decomposing Responses**: GPT processes the prompt and its corresponding response, breaking them into discrete factual statements. Each statement is assigned an importance score (0–1), indicating its relevance to the prompt.

**Proximity Search**: For each statement, the system retrieves five of the closest facts from the vector database through a semantic proximity search <d-cite key="johnson2017"></d-cite>. These facts serve as the context for validation.

**Validity Scoring**: Each statement is evaluated against its contextual facts using GPT, which assigns a validity score between 0 and 1 based on alignment. This evaluation determines how well the statement aligns with retrieved facts <d-cite key="kryscinski2019"></d-cite>.

**Weighted Average Calculation**: The validity scores are aggregated into a weighted average, where the weights are derived from the importance scores of the statements. This weighted score represents the overall factual accuracy of the response.

**Explainability**: If required, GPT generates a rationale for the validity rating, outlining which facts were used, how alignment was determined, and any potential conflicts. This reasoning enhances interpretability and transparency.

<span style="font-size: 20px; font-weight: bold;">Why Not KNN for Clustering?</span>

K-nearest neighbor (KNN) was considered for fact clustering but was ultimately not used due to its reliance on predefined parameters, such as the number of clusters or neighbors—constraints that are impractical for real-world datasets with high diversity <d-cite key="taunk2019"></d-cite>. In contrast, DBSCAN identifies clusters based on density, eliminating the need for prior knowledge of the dataset's structure <d-cite key="ester1996"></d-cite>. This makes DBSCAN more effective for processing unstructured or unknown data distributions. Additionally, KNN's approach risks significant data abstraction or loss when attempting to condense diverse facts into singular representations, worsening the system's ability to maintain factual granularity <d-cite key="cunningham2020"></d-cite>.

<span style="font-size: 20px; font-weight: bold;">Additional Note</span>

If the dataset is preprocessed to isolate one specific fact per entry, we recommend bypassing that step. Instead, the data can directly enter our model, which dynamically groups related facts. This approach preserves the richness of the dataset and streamlines the overall workflow.

## Advantages of This System

- **Granular Verification**: Validates individual facts for higher precision compared to document- or paragraph-level assessments.
- **Simple Integration**: This system can be integrated into existing RAG infrastructure to enhance the overall functionality without requiring infrastructure changes.
- **Efficient Storage**: Compresses data through summarization and clustering, reducing storage needs and retrieval token usage.
- **Fact-Level Accuracy**: Ensures validity by requiring agreement across multiple sources before storing facts.
- **Real-Time Validation**: Enables fast, efficient validation within the generation process.
- **Scalable**: Modular design scales across datasets and domains, ensuring long-term applicability.

## Experimentation and Results

We evaluated our proposed pipeline by benchmarking it against the Traditional Retrieval-Augmented Generation (RAG) Pipeline using the PubmedQA dataset <d-cite key="jin2019"></d-cite>. The evaluation focused on three key metrics: factual accuracy, RAG effectiveness, and storage efficiency, collectively measuring response quality, retrieval precision, and data storage optimization.

<span style="font-size: 20px; font-weight: bold;">Traditional RAG Pipeline</span>

The traditional RAG pipeline was tested under ideal conditions, embedding and retrieving the labeled contexts directly from the PubmedQA dataset <d-cite key="jin2019"></d-cite>. This setup provided perfect access to the correct answer contexts, offering a significant advantage. Despite this, our proposed pipeline—which applies summarization and compression—demonstrates performance comparable to this baseline, underscoring its effectiveness.

The traditional pipeline workflow includes the following steps:
- Fetch all contexts from the PubmedQA dataset <d-cite key="jin2019"></d-cite>.
- Embed each context using OpenAI's `text-embedding-3-large` model.
- Store the embeddings in PineconeDB for retrieval.

<span style="font-size: 20px; font-weight: bold;">Comparative Metrics</span>

| **Metric**           | **Traditional Pipeline** | **Proposed Pipeline** | **Difference**            |
|-----------------------|--------------------------|------------------------|---------------------------|
| Factual Accuracy      | 71.7%                   | 71.2%                 | -0.5%                    |
| RAG Effectiveness     | 99.2%                   | 98.9%                 | -0.3%                    |
| Storage Efficiency    | 1,351 KB                | 571 KB                | -57.7% (Reduction)       |

**Table 1.** Comparison of factual accuracy, RAG effectiveness, and storage efficiency between the traditional pipeline and the proposed pipeline. The proposed pipeline achieves comparable performance in accuracy and effectiveness while significantly reducing storage requirements by 57.7%.

<span style="font-size: 20px; font-weight: bold;">Factual Accuracy</span>

We tested factual accuracy to measure how well the system addressed prompts by integrating all processes, including context retrieval, summarization, and LLM response generation. Using the PubmedQA dataset with 1,000 labeled context-question-answer groups, we queried each question and deemed responses correct if they matched the dataset’s answer (yes, no, or maybe) <d-cite key="jin2019"></d-cite>. The traditional pipeline achieved 71.7% accuracy, while our pipeline achieved 71.2%, a negligible difference of 0.5%. This suggests that summarizing contexts did not hinder the LLM’s ability to generate correct answers. Further improvements in prompt engineering or summary generation could potentially surpass the traditional pipeline’s accuracy.

<span style="font-size: 20px; font-weight: bold;">RAG Effectiveness</span>

RAG effectiveness was evaluated to determine how well each pipeline retrieved the most relevant context for a given query. The PubmedQA dataset contexts were queried using labeled questions, and a retrieval was marked correct if the top result matched the labeled correct context <d-cite key="jin2019"></d-cite>. The traditional pipeline achieved 99.2% RAG effectiveness, while our pipeline achieved 98.9%, a minor 0.3% difference. The minimal reduction shows that summarization and clustering do not compromise retrieval quality. Additionally, granting the proposed pipeline access to original embeddings instead of summaries for labeling could eliminate this gap entirely, further reinforcing its effectiveness.

<span style="font-size: 20px; font-weight: bold;">Storage Efficiency</span>

We measured storage efficiency by calculating the total size of stored contexts in the vector database (excluding vector embeddings). The traditional pipeline required 1,351 KB, whereas our pipeline used only 571 KB, a reduction of 57.7%. This demonstrates the significant compression achieved through summarization and clustering. The benefits are particularly pronounced in unstructured datasets, where redundant data can be consolidated. While PubmedQA is a structured dataset, in more diverse real-world datasets, the proposed pipeline would likely achieve even greater storage savings. This reduced footprint allows for the storage of larger datasets and faster query times, providing scalability without sacrificing usability.

## Broader Implications

The results of this pipeline highlight its potential to advance how RAG systems handle unstructured and large-scale datasets. Its ability to compress and organize data effectively expands the capacity of vector databases, enabling systems to manage larger and more diverse datasets without sacrificing query performance. This scalability is critical for real-time applications and resource-constrained environments like edge computing <d-cite key="yao2024"></d-cite>.

The modular design ensures easy integration with existing RAG systems, supporting seamless adoption while allowing for domain-specific customization. By balancing efficiency, scalability, and accuracy, this framework contributes to more practical and reliable AI systems.

## Limitations and Future Work

1. **Suboptimal Context Fit**:
- The data retrieved may not always align perfectly with the prompt, leading to potential mismatches in utility. 
- **Mitigation**: Introduce additional retrieval steps that identify and remove irrelevant facts, guided by LLM evaluations.

2. **Validity Scoring Challenges**:
- Validity ratings based on retrieved facts may lack precision due to ambiguous LLM prompts or insufficient contextual grounding. 
- **Mitigation**: Provide scoring examples during prompting or fine-tune the LLM to compute scores more reliably from source data.

3. **Context Sensitivity Issues**:
- Facts may be accurate in isolation but misleading when applied in certain contexts, limiting the system's adaptability. 
- **Mitigation**: Extend vector embeddings to include source and context information, enabling the system to incorporate situational nuances.

4. **Potential Source Bias**:
- Bias in the original source documents can propagate into the system, affecting the impartiality of responses. 
- **Mitigation**: Add a preprocessing step to evaluate and mitigate biases in source data before vectorization.

5. **Scope of Accuracy**:
- The system prioritizes factual accuracy without focusing on other dimensions of response quality, such as creative inference or prompt analysis.
- **Mitigation**: Include additional validity metrics, such as evaluating inferential accuracy or alignment with the original prompt, using separate GPT evaluations.

6. **Expanded Dataset Evaluation**:
- Due to time constraints, the system was tested only on PubmedQA and did not undergo evaluation on a broader range of datasets. This limitation restricts the full assessment of the method’s scalability and effectiveness across all data types.
- **Mitigation**: Future work aims on expanding the evaluation to include additional benchmark datasets, such as SimpleQA <d-cite key="simpleqa"></d-cite>. Sources like Wikipedia pages and news articles will be scraped and processed into the vector database to better assess the pipeline's performance under real-world conditions.

## Conclusion

In this blog post, we explored a novel framework for enhancing the efficiency and scalability of RAG pipelines while maintaining high standards of factual accuracy. By incorporating techniques such as summarization, clustering, and vectorized fact storage, we demonstrated how preprocessing can substantially reduce storage and optimize context management.

Our experimentation with the PubmedQA dataset highlighted the pipeline’s ability to perform competitively with traditional methods with ideal access to labeled contexts. These findings accentuate the potential of fact validation strategies to balance accuracy and efficiency, advancing trust in AI systems and paving the way for future innovations in managing unstructured data and large-scale knowledge retrieval.