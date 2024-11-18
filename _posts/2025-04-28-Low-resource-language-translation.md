
---
layout: distill
title: 
description: Language Models are increasingly recognized for their potential in translation tasks. Researchers are actively exploring the most effective architectures, prompting formats, and fine-tuning techniques for these tasks. Further, the use of LLMs in low-resource machine translation is still emerging. This study evaluates methods that combine In-Context Learning (ICL) with fine-tuning scenarios where language data is scarce.

Our strategy involves two steps: first, introducing translation examples from languages closely related to the target language; second, using embeddings to include similar examples to the source language for few-shot ICL. We hypothesize that this approach helps LLMs access their hidden translation capabilities.

We test our methods on two language clusters: Konkani and Tunisian Arabic, using models from three LLM families. The first model is for general chat and instruction following, trained with limited non-English data. The second model is tailored for machine translation tasks, though not specifically with data from the target language clusters. The third model is trained in several languages and is known for its multilingual capabilities.

Our evaluations show that models fine-tuned with this technique perform well given these constraints. Through this paper, we hope to make LLMs more accessible to communities speaking low-resource languages.

date: 2025-04-28
future: true
htmlwidgets: true
hidden: false


bibliography: 2025-04-28-Low-resource-language-translation.bib  


***

## Introduction

A variety of research and development efforts are focused on increasing linguistic diversity in research datasets and boosting performance for Natural Language Processing (NLP) tasks for languages in low-resource settings. For example, the recent No Language Left Behind (NLLB) effort <d-cite key="team2022NoLL"></d-cite> seeks to create datasets and models for massively multilingual Machine Translation (MT) with the stated aim of "eradicating language barriers on a global scale." Other efforts, such as the Bloom Library datasets project <d-cite key="leong-etal-2022-bloom"></d-cite> and Masakhane <d-cite key="nekoto-etal-2020-participatory"></d-cite>, seek to establish the first known baselines for languages with data and tooling developed by speakers of those languages. These and other related efforts generally hope to address the inequalities in the performance of NLP tasks, such as machine translation, across the world’s 7,000+ languages <d-cite key="Eberhard2024Ethnologue"></d-cite>.

At the same time, the explosion of interest and research in generative, Large Language Models (LLMs) has simultaneously demonstrated: (1) impressive multi-task instruction-following capabilities of text generation models, such as LLaMA 2 <d-cite key="Touvron2023Llama2O"></d-cite> or Mistral <d-cite key="Jiang2023Mistral7"></d-cite>, that are fine-tuned using supervised fine-tuning and preference optimization; and (2) the dominance of a small number of resource-rich languages like English and Mandarin in the data used to train these models. Although an increasing number of region-specific and other-than-English-and-Mandarin LLMs are being released (see, for example, the SeaLLM <d-cite key="Nguyen2023SeaLLMsL"></d-cite> and SEA-LION models <d-cite key="lowphansirikul2021wangchanberta"></d-cite> for Southeast Asian languages), the majority of LLM models driving open access model innovation represent a negligible percentage of living languages.

Rather than focusing on the important research directions of dataset creation (to solve data scarcity issues) or language-specific foundation model creation (to increase linguistic diversity in base LLMs), this paper focuses on leveraging the few-shot instruction-following capabilities of LLMs for the task of machine translation in low-resource settings. In cases where there is a desire to translate from a widely spoken language (supported in commercial and open-source MT systems) to a language without high digital language vitality (and not supported in any known commercial and open-source MT systems), we propose leveraging prompts with few-shot examples of machine translation from the source language to both the target language and a higher-resourced, linguistically related pivot language. These prompts demonstrate the adaptation of a machine translation from the pivot language to the target language "in context," such that for new samples (where the adaptation is not known), the LLM generates a likely adaptation to the target language.

Such prompts could be utilized with existing pre-trained models, which would potentially yield useful results if the instruction-following capabilities of the pre-trained LLMs could perform the adaptation task without further fine-tuning. However, prompts formatted in this manner could also be curated into datasets to fine-tune LLMs for the low-resource machine translation task, with the hope that such prompt curation and formatting could reduce the need for large parallel corpora and/or computationally prohibitive training of massively multilingual models.

We demonstrate the approach for two linguistically distinct language clusters: Arabic and Konkani. We find that pre-trained LLMs, with ICL applied, do not produce useful translation in the low-resource language scenario. However, LLMs fine-tuned on small numbers of ICL prompts outperform LLMs trained with zero-shot translation prompts.

We find that when the models are fine-tuned by constructing the prompt as mentioned, they outperform MT benchmarks for these languages.

***

## Related Work

### In-Context Learning

In-Context Learning (ICL) has gained popularity in recent times, significantly enhancing the responses of Large Language Models (LLMs). Unlike traditional approaches where parameters are optimized through training or fine-tuning, LLMs perform tasks merely by conditioning on input-output examples during ICL. This process enables the patterns and structures learned during training to generate appropriate responses. Essentially, the LLM is not explicitly trained to learn from examples in the way we might think. Instead, the pretraining of LLMs focuses on next-token prediction, a foundational task that involves generating the next word based on the context of the words that precede it. Through ICL, latent concepts from the pretraining data are utilized to adapt the foundational next-token prediction capabilities of the LLM, enabling it to perform tasks it was not originally trained for.

The extensive training on vast amounts of text equips the LLM to handle a variety of tasks without further training. When presented with an ICL prompt, the LLM uses this prompt to "locate" a previously learned concept or knowledge within its vast training data. Here, a concept acts as a latent variable, encapsulating various document-level statistics and information that the model has been exposed to during its training phase <d-cite key="xie2021explanation"></d-cite>.

ICL has proven to be useful in translation tasks as well. <d-cite key="moslem2023adaptive"></d-cite> demonstrate its effectiveness, particularly through the use of few-shot ICL for translation tasks. This approach improves real-time adaptive machine translation (MT), offering a potentially new avenue for improving the responsiveness and accuracy of translation systems. They show that the translation quality achieved with few-shot ICL can surpass that of strong encoder-decoder MT systems, especially for languages with abundant resources. They hypothesize that during the unsupervised pretraining phase, the model develops pattern recognition abilities, characteristic of autoregressive decoder-only models. During inference, these patterns are applied to execute translation tasks effectively.

<d-cite key="agrawal2022context"></d-cite> highlight how translation quality and the domain of in-context examples affect the performance of LLMs in translation tasks. It is shown that a single-shot, noisy, unrelated example can detrimentally affect the output quality. Incorporating similar examples, identified based on n-gram overlap with the test source, significantly enhances the translation quality of the outputs.

ICL has also shown significant promise in translation tasks. <d-cite key="moslem2023adaptive"></d-cite> demonstrate its effectiveness, particularly through the use of few-shot ICL, to enhance translation performance. This approach improves real-time adaptive machine translation (MT), presenting a potential avenue for advancing the responsiveness and accuracy of translation systems. Their findings suggest that translation quality achieved with few-shot ICL can surpass even robust encoder-decoder MT systems, especially for languages with abundant resources. They hypothesize that during unsupervised pretraining, the model develops pattern recognition abilities characteristic of autoregressive decoder-only models, which are then applied during inference to perform translation tasks effectively.

<d-cite key="agrawal2022context"></d-cite> emphasize that translation quality and the type of in-context examples used can significantly impact LLM performance in translation tasks. Their study shows that a single-shot, noisy, or unrelated example can degrade output quality, while including more relevant examples—selected based on n-gram overlap with the test source—substantially enhances translation quality.

This method outperforms a strong kNN-MT baseline in half of the tested out-of-domain datasets. Similar examples to the test source are retrieved from a datastore containing pairs of source text and their corresponding translations. Retrieval is facilitated by BM25, an unsupervised and efficient method that provides additional context to the model.

### Fine-Tuning

In the previous section, we primarily focused on improving translations using ICL examples. However, translation performance can also be enhanced by fine-tuning models. <d-cite key="li2023eliciting"></d-cite> show that the ability of LLMs to execute translation tasks depends on their comprehension of translation instructions and the degree of alignment between different languages. By applying multilingual fine-tuning alongside translation instructions, LLMs can effectively learn to translate even between language pairs that were not seen during the instruction tuning phase. Their fine-tuning approach not only improves translation capabilities for language pairs included in the training but also for new, unseen language pairs.

The technique of including ICL examples during fine-tuning has also proven effective beyond translation tasks. <d-cite key="fatehkia2024t"></d-cite> demonstrate that in question-answering tasks, retrieving similar document chunks from embeddings and including them in the prompt of a fine-tuned LLM yields better performance than either method alone.

***

## Methodology

Previous studies have shown that carefully selecting relevant examples can significantly enhance in-context learning (ICL) for translation tasks. Our approach leverages ICL, specifically through few-shot examples, to improve translation performance.

First, we assess the performance of pretrained models on the translation task for the target languages. This initial evaluation establishes a baseline for comparison.

Next, we fine-tune the pretrained models using a consistent prompt format during both training and inference phases. This approach ensures that any observed improvements can be attributed to the fine-tuning process. We prioritize adding sentences that closely match the sentence we aim to translate as few-shot examples in the prompt. By selecting these highly relevant examples, we seek to improve the model’s accuracy in executing the translation task.

To implement our methodology, we first populate a vector database with training samples and calculate cosine similarity scores to identify the examples most similar to the source sentence we wish to translate. We use the all-MiniLM-L12-v2 model to generate vector embeddings <d-cite key="reimers-2019-sentence-bert"></d-cite>. Then, 5 of the most similar sentences, are added to the prompt in the same format as few-shot learning examples.

The overall prompt format is adjusted based on the model in use. The specific prompt templates used are provided in the appendix.

Our experiments are conducted using Unbable Tower models, which are specialized, context-aware translation models fine-tuned from Llama2. Additionally, we utilize Nous Research's fine-tuned Llama3 model <d-cite key="Hermes-2-Pro-Llama-3-8B"></d-cite>, which is a multilingual model. These models possesses unique capabilities, allowing us to evaluate which model performs best in translating to a previously unseen language.

In this study, we also fine-tune the models to assess translation outcomes from English to the target language, maintaining the prompt format as recommended for each original model.

The fine-tuning process leverages training data arranged in the format described above. Using specifically formatted prompts during both training and inference is known to improve model performance. By applying prompts consistently in this format to both pretrained and fine-tuned models, we can assess whether fine-tuning has indeed enhanced performance.

BLEU scores, calculated using SacreBLEU, were obtained for each model on the test set. For comparison with existing machine translation systems, we used Bing to calculate the score for Konkani. For Tunisian Arabic, which is supported by NLLB, the BLEU score was calculated using NLLB.

***

## Experiments

### Language similarity
In this work, we hypothesize that for low-resource target languages, adding a high-resource pivot language—one that the model is likely trained on and shares the same script—can help the model produce more higher quality translations. The choice of a suitable pivot language may depend on factors such as data availability, whether the model has been pretrained on the language, and whether it belongs to the same language family as described in Ethnologue. Quantitatively measuring the similarity between the pivot and target languages to ensure that the data added during in-context learning is relevant, as noisy data can degrade translation quality <d-cite key="agrawal2022context"></d-cite>.

Jaccard Similarity can be used to measure the similarity between the parallel corpus. This metric measures the similarity between two sets, often represented as the ratio of the intersection (shared elements) to the union (total unique elements) of those sets. In this case the measure can be used to check the overlap of words between the pivot language and the target language <d-cite key="Majumdar2022ta"></d-cite>. Jaccard similarity quantifies the vocabulary or structural overlap, yielding a value between 0 and 1. High Jaccard similarity between two languages typically indicates shared vocabulary. We hypothesize that leveraging this overlap, along with selecting pivot languages from language families classified by Ethnologue, increases the likelihood of accurate translations. 

Although other measures of lexical similarity exists, Jaccard similarity is interpretable and allows to measure the similarity between the parallel corpus for low resource languages where the regionally satandardized wordlist is not available (<d-cite key="ethnologue2024"></d-cite> ). 

In our case, a Jaccard similarity score of 62 between Marathi and Konkani suggests a substantial overlap, with the two languages sharing about 62% of their linguistic elements, such as words, phrases, or structures. We also calculated the Jaccard similarity between Tunisian Arabic and Modern Standard Arabic, which resulted in a score of 56%. While Jaccard similarity is not an essential criterion, it supports our selection of effective pivot and source languages. In future work, we aim to explore whether correlation between lexical similarity and translation performance exists.

### Models

In this experiment, we evaluate the performance of two models: Unbabel/TowerInstruct-7B-v0.2 d-cite key="alves2024tower"></d-cite>, and NousResearch/Hermes-2-Pro-Llama-3-8B <d-cite key="Hermes-2-Pro-Llama-3-8B"></d-cite>. The Hermes-2-Pro-Llama-3 model is a fine-tuned version of Llama3, known for its multilingual capabilities. Tower Instruct is fine-tuned from the Llama2 model specifically for translation-related tasks such as Machine Translation (MT) and Automatic Post-Editing (APE) <d-cite key="alves2024tower"></d-cite>. The Tower Base model is first fine-tuned from Llama2 using monolingual and parallel corpora, while Tower Instruct is then fine-tuned from Tower Base with specific instructions for translation tasks. Although Tower Instruct performs well on translation tasks, it is not expected to excel in languages it was not exposed to during training.

Our strategic selection of these models is designed to assess their effectiveness in translating languages outside their initial training sets. Despite Tower Instruct’s specialized training in translation, it has not been directly exposed to the specific low-resource languages focused on in this experiment, offering a unique test of its adaptability to unseen languages.

### Datasets
For this experiment, we focused on two low-resource languages. The first is Konkani, an Indian language spoken in the western part of India, with approximately 2.35 million speakers as of 2011. The second is Tunisian Arabic, spoken in Tunisia, with around 12 million speakers as of 2021.

The Konkani parallel corpus was constructed using a dataset open-sourced by AI4Bharat, which also contributed to the training set for the IndicTrans2 model <d-cite key="gala2023indictrans2"></d-cite>. This corpus includes English, Marathi, and Konkani. Despite having access to a larger dataset, we intentionally limited the training set to approximately 900 records to simulate a low-resource language scenario.

Given the experiment’s reliance on a pivot language included in the ICL examples, Marathi was selected as the pivot language due to its wider prevalence in western India and its linguistic similarity to Konkani.

The test set for Konkani consisted of 200 records.

For Tunisian Arabic, the corpus was derived from the work described in <d-cite key="bouamor-etal-2014-multidialectal"></d-cite>, with Modern Standard Arabic chosen as the pivot language.

The parallel corpus for Tunisian Arabic contained 1,000 records, with 900 used in the training set and 100 used in the test set.

Our aim is to replicate the translation performance for low-resource languages that typically have limited data available. An additional motivation for selecting these languages was their use of non-Latin scripts. By working with a small training or ICL set of around 1,000 records, we explore whether it is possible to enhance translation performance under such resource constraints.

### Trainable parameters
This work deals with scenarios where only few hundred rows of data is available. In such scenarios we try to see what the model size, finetuning technique should be. <d-cite key="zhang2024scaling"></d-cite> highlights that with limited data, finetuning methods like prompt tuning (where embeddings are adjusted) or LoRA (Low-Rank Adaptation) prove particularly effective. With Parameter-Efficient Fine-Tuning (PEFT), even increasing the data yielded modest performance improvements. For instance, using LoRA on the Llama 3 8B model brought the trainable parameters down to 176,242,688, or just 2% of the model’s total parameters. In this work we mainly focus on LORA for finetuning. 

### Tokens 
In this section, we analyze token-related metrics in the training data that may influence model performance for different languages. The token-per-word ratio is an important metric in understanding how effectively a tokenizer captures the linguistic structure of a language, which can influence model performance. <d-cite key="remy2024trans"></d-cite>.

In the first table, we observe that for English, the token-per-word count is approximately 1 for both models, as expected. However, for Marathi and Konkani, the Tower model exhibits a significantly higher token-per-word count compared to the Nous Hermes Llama 3 model. This suggests that the Nous Research model may produce better translations, which aligns with the findings discussed in the results section.

| Language         | Model                              | Model Total Tokens | Model Unique Tokens | Model Avg Token Length |
|------------------|------------------------------------|--------------------|----------------------|------------------------|
| English          | NousResearch/Hermes-2-Pro-Llama-3-8B | 27,136            | 7,624               | 1.341                 |
|                  | Unbabel/TowerInstruct-7B-v0.2      | 31,823            | 6,496               | 1.573                 |
| Marathi          | NousResearch/Hermes-2-Pro-Llama-3-8B | 73,641            | 892                 | 4.144                 |
|                  | Unbabel/TowerInstruct-7B-v0.2      | 139,905           | 128                 | 7.873                 |
| Konkani          | NousResearch/Hermes-2-Pro-Llama-3-8B | 68,777            | 984                 | 4.050                 |
|                  | Unbabel/TowerInstruct-7B-v0.2      | 128,651           | 132                 | 7.576                 |


For the Arabic language cluster, token counts and other metrics are as follows. Similar to the Marathi and Konkani findings, the Nous Research Llama 3 finetune demonstrates a lower token-per-word ratio compared to the Tower model, suggesting potentially better translation performance.


| Language         | Model                              | Model Total Tokens | Model Unique Tokens | Model Avg Token Length |
|------------------|------------------------------------|--------------------|----------------------|------------------------|
| English          | NousResearch/Hermes-2-Pro-Llama-3-8B | 13,185            | 2,829               | 1.024                 |
|                  | Unbabel/TowerInstruct-7B-v0.2      | 14,197            | 2,675               | 1.102                 |
| Modern Standard Arabic              | NousResearch/Hermes-2-Pro-Llama-3-8B | 18,365            | 1,625               | 2.008                 |
|                  | Unbabel/TowerInstruct-7B-v0.2      | 41,359            | 83                  | 4.523                 |
| Tunisian Arabic  | NousResearch/Hermes-2-Pro-Llama-3-8B | 17,418            | 1,585               | 2.051                 |
|                  | Unbabel/TowerInstruct-7B-v0.2      | 39,834            | 78                  | 4.752                 |

Across languages, we can observe that the dataset for Konkani translation is richer compared to the dataset for Tunisian Arabic, as evidenced by the higher English token counts. This richness of the dataset will likely influence model performance.

Analyzing these metrics is critical to assess the quality of the dataset and its potential impact on translation accuracy. In future work, it would be valuable to explore the relationship between token counts in the dataset and model performance, especially for translation tasks to unseen languages. Understanding these dynamics could further improve fine-tuning strategies and dataset construction for multilingual models.





### Parameters
<d-cite key="liu2022few"></d-cite> demonstrate that Parameter-Efficient Fine-Tuning (PEFT) is computationally more efficient than pure ICL, which led us to adopt PEFT for our model fine-tuning process. We used the Huggingface Transformers library and fine-tuned the models with the following hyperparameters: 

#### Trainining Prameters

| Parameter                     | Value             |
|-------------------------------|-------------------|
| `per_device_train_batch_size` | 1                 |
| `num_train_epochs`            | 1.5               |
| `warmup_ratio`                | 0.03              |
| `logging_steps`               | 25                |
| `learning_rate`               | 2e-4              |
| `gradient_accumulation_steps` | 1                 |
| `gradient_checkpointing`      | True              |
| `lr_scheduler_type`           | Cosine            |
| `weight_decay`                | 0.001             |
| `save_strategy`               | No                |
| `optim`                       | `paged_adamw_32bit` |
| `warmup_steps`                | 100               |
| `bf16`                        | True              |

These parameters were selected to optimize both computational efficiency and model performance for fine-tuning on limited data.

#### Quantization
The model was loaded in 4-bit precision using the BitsAndBytes library with the nf4 quantization type. The following table summarizes the configuration used for 4-bit quantization:

| Parameter                 | Value                                       |
|---------------------------|---------------------------------------------|
| `Compute dtype for 4-bit base models`` | float16 (Compute dtype for 4-bit base models) |
| `Quantization type``    | nf4    


#### LoRA 
For fine-tuning, we employed the LoRA configuration, as detailed in the table below:

| Parameter         | Value                                                                  |
|--------------------|------------------------------------------------------------------------|
| `r`               | 64                                                                     |
| `lora_alpha`      | 16                                                                     |
| `lora_dropout`    | 0.1                                                                    |
| `bias`            | none                                                                   |
| `task_type`       | CAUSAL_LM                                                              |
| `target_modules`  | ['k_proj', 'gate_proj', 'v_proj', 'up_proj', 'q_proj', 'o_proj', 'down_proj', 'lm_head'] |


#### Text generation
The following parameters were used to generate the outpput from the finetuned model during the evaluation

| Parameter               | Value  |
|-------------------------|--------|
| `do_sample`            | True   |
| `temperature`          | 0.1    |
| `num_return_sequences` | 1      |
| `max_new_tokens`       | 200    |
| `return_full_text`     | False  |



### Baselines

To establish baselines, we used commercially available machine translation (MT) tools. For Konkani, we chose Bing Translate as our baseline. However, Tunisian Arabic was not supported by most commercial MT systems.

We also fine-tuned the No Languages Left Behind (NLLB) 1.3B distilled model to establish baseline performance on the given data. Since Tunisian Arabic is already included in the pre-trained NLLB model, we observed that fine-tuning it on our training data slightly improved performance. Fine-tuning was conducted for 2,000 steps with a learning rate of 1e-4 and a weight decay of 1e-3 after 1,000 steps.

***

## Results

BLEU scores, calculated using SacreBLEU, were obtained for each model on the test set, which comprised 200 sentences. This test set was evaluated using commercially available MT tools. For Konkani, Bing Translate served as the baseline, with direct translation from English to Konkani without a pivot language. 

In the prompt for In-Context Learning (ICL), few-shot examples were constructed by selecting sentences that closely resemble the source sentence for translation—in this case, English. The prompt included 5 examples of translations from English to Marathi and then to Konkani. Initially, we assessed the performance of the pre-trained models.

The BLEU scores were calculated for the two models. Without any fine-tuning and using only similar examples as few-shot demonstrations, it was observed that the model often did not produce high-quality translations. The responses contained Latin characters and occasionally included comments on the translations. The models were then fine-tuned both with and without the addition of pivot language translations. It was found that incorporating high-resource pivot translations improved the BLEU score and made the translations less noisy.

The results are highlighted below. The more performant general-purpose model, Llama 3, outperformed Tower, a fine-tuned version of Llama 2. Notably, these results are based on only 800 training samples, which suggests that this technique for translations can be widely accessible. Additionally, Konkani was chosen for this study because its script is non-Latin, making it valuable to investigate how language models trained primarily on Latin-character data handle translations for such languages

| **Model**                                 | **BLEU Score** |
|-------------------------------------------|----------------|
| Baseline (Bing Translate)                 | 40.90         |
| Tower Instruct                            | 3.21          |
| Tower Instruct fine-tuned without pivot   | 11.39         |
| Tower Instruct fine-tuned with pivot      | 13.56         |
| Nous Hermes Llama3                        | 7.26           |
| Nous Hermes Llama3 fine-tuned without pivot | 13.22       |
| Nous Hermes Llama3 fine-tuned with pivot  | 18.50         |

**Table:** Konkani language BLEU scores for various models. This table compares the performance of baseline and fine-tuned models with and without a pivot language.

Since NLLB supports Tunisian Arabic, we used it to establish the BLEU score for this language as well, again translating directly from English to Tunisian Arabic without a pivot language.
Tunisian Arabic is supported by NLLB, which was used to establish the baseline for Tunisian Arabic translations. As with Konkani, the performance of Tunisian Arabic translations was first evaluated using a pre-trained model.

Tunisian Arabic is supported by NLLB, which was used as the baseline for Tunisian Arabic translations. Similar to the approach taken with Konkani, the performance of Tunisian Arabic translations was initially evaluated using a pre-trained model.

Following fine-tuning, the models showed improved performance.

To further validate the robustness of the methodology, an expanded test set was created. For Konkani, 200 additional samples were selected from AI4Bharat and tested. These samples were randomly chosen and were not part of the original training dataset. The BLEU score for this expanded test set was calculated. Similarly, for Tunisian Arabic, a parallel corpus of the Bible was constructed at the verse level.

However, the BLEU score for Tunisian Arabic dropped significantly to around 8 when tested on this new corpus. This decline may be attributed to the dissimilarity between the constructed corpus and the training dataset, underscoring the importance of selecting high-quality data that closely resembles the test set when applying translations in practice.

| Model                              | BLEU Score |
|------------------------------------|------------|
| Baseline (NLLB)                    | 4.68       |
| tower instruct                     |        |
| tower instruct fine-tune without pivot |       |
| tower instruct fine-tune with pivot  |       |
| Nous Hermes Llama3                 |        |
| Nous Hermes Llama3 fine-tune without pivot |  |
| Nous Hermes Llama3 fine-tune with pivot |  |

**Table:** Tunisian Arabic language BLEU scores for various models. This table compares the performance of baseline and fine-tuned models with and without adding pivot language.

Following fine-tuning, model performance showed improvement.

To further validate the robustness of our methodology, an expanded test set was created. For the Konkani dataset, 200 additional samples were randomly selected from AI4Bharat and tested. These samples were not part of the original training dataset, and the BLEU score for this expanded test set was calculated. Similarly, for Tunisian Arabic, a parallel corpus was constructed using Bible verses as aligned text at the verse level.



***

## Discussion

Much of the existing research has focused on high-resource languages or those closely related to English. In this paper, we examine translation involving two relatively low-resource languages and offer a practical guide for achieving high translation performance with minimal resources. The fine-tuning process described here is computationally efficient, requiring only 1,000 samples. This approach could make these models more accessible to communities that speak lesser-known languages.

We hypothesize that by adding translation examples that include a pivot language in In-Context Learning (ICL), the model can tap into latent concepts to perform translation tasks effectively, even for languages it has not been previously exposed to.

However, this methodology has certain limitations. One significant challenge is the dependency on identifying a high-resource language that is linguistically similar to the target language. Additionally, building parallel corpora across multiple languages can be challenging due to limited data availability. In our experiments, we also observed that training data quality is crucial; ideally, the data should resemble the types of content on which the translations will be tested.

The results from the expanded test set emphasize the importance of using a high-quality, diverse dataset for both fine-tuning and constructing ICL examples, as these factors critically influence the effectiveness of the methodology.

***

## Acknowledgement
I would like to thank David Dale for his critical input in understanding how best to adapt a language model to translate to low resource languages. 

## Appendix


### Prompt Template


Both the Tower model and Nous Research's Llama 3 model utilize a similar prompt format. The prompt includes the source sentence in English and its translation in a pivot language. For in-context learning, the prompt contains five demonstrations. In each demonstration, the assistant field is pre-filled with the target language translation. These demonstrations are carefully selected sentences that closely resemble the sentence to be translated. In the final instance, the assistant field is left blank. This prompt structure proved to be highly effective for translation tasks of this nature. However, when using this format with the base model, the outputs often included elements like "Note," gibberish, and repetitions. After fine-tuning the model with this format, the generated translations adhered to the expected structure and consistently produced Konkani sentences.

```
<|im_start|>user
APE is a task designed to enhance the quality of the translation by performing minor adjustments
Original (English): [Original text in English]
Translation: [Translaion to  pivot langauge]
Post-edited: 
<|im_end|>
<|im_start|>assistant
[Target language translation generated by the model]
<|im_end|>

```


### Prompt tuning
Prompt tuning, introduced by <d-cite key="lester2021power"></d-cite>, is a mechanism designed to learn soft prompts that condition models to perform specific downstream tasks. According to <d-cite key="zhang2024scaling"></d-cite>, it was observed that when data is limited, prompt tuning can be more effective than traditional finetuning. However, for low-resource languages, the performance of prompt-tuned models was suboptimal. BLEU scores for translations ranged between 2 and 5, with the base model (without any finetuning) outperforming the finetuned model. 

This performance gap is likely because the model was not sufficiently exposed to these low-resource languages during training. As a result, merely applying prompt tuning was insufficient to yield significant improvements. The trainable parameters accounted for just 0.0004% of the Nous Hermes Llama 3 model. For the tower model it was 0.0005%.

#### Training Parameters
The following parameters were used during the training process:

| Parameter   | Value  | 
|-------------|--------|
| `max_length`| 64     |
| `lr`        | 0.03   |
| `num_epochs`| 1      |
| `batch_size`| 8      |

#### Evaluation Parameters
To generate translations during evaluation, the following parameters were applied:

| Parameter         | Value  |
|-------------------|--------|
| `max_new_tokens`  | 750    |
| `temperature`     | 0.1    |
| `do_sample`       | True   |

These parameters are commonly employed in text generation tasks to control output length, randomness, and termination criteria. 

While prompt tuning shows potential, its effectiveness for low-resource languages remains a challenge. Refining the mechanisms to optimize prompt tuning for such scenarios is left for future work.





