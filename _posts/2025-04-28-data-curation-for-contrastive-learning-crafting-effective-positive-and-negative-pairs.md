---
layout: distill
title: A Survey on Data Curation for Contrastive Learning -  Crafting Effective Positive and Negative Pairs
description: Contrastive learning aims to learn a robust representation of the underlying dataset that can be useful for various downstream tasks. It follows a simple yet powerful technique of creating positive and negative pairs of data points (images, text, speech, or multimodal pairs) to learn the representation. The design of positive and negative pairs is crucial in contrastive learning, as it directly impacts the quality of learned representations. Hence, considerable research has been conducted in this direction to create these pairs. This blog post aims to create a taxonomy of existing techniques used in positive and negative pair creation, dive deeper into each methodology to create such pairs, and provide different use cases where each technique can be used along with the trade-offs. Lastly, this blog provides some open-research questions in this field.
date: 2025-04-28
future: true
htmlwidgets: true
hidden: false

# Anonymize when submitting
authors:
   - name: Anonymous
#authors:
#  - name: Albert Einstein
#    url: "https://en.wikipedia.org/wiki/Albert_Einstein"
#    affiliations:
#      name: IAS, Princeton
#  - name: Boris Podolsky
#    url: "https://en.wikipedia.org/wiki/Boris_Podolsky"
#    affiliations:
#      name: IAS, Princeton
#  - name: Nathan Rosen
#    url: "https://en.wikipedia.org/wiki/Nathan_Rosen"
#    affiliations:
#      name: IAS, Princeton

# must be the exact same name as your blogpost
bibliography: 2025-04-28-data-curation-for-contrastive-learning-crafting-effective-positive-and-negative-pairs.bib

# Add a table of contents to your post.
#   - make sure that TOC names match the actual section names
#     for hyperlinks within the post to work correctly. 
#   - please use this format rather than manually creating a markdown table of contents.
toc:
  - name: Contrastive Learning Formulation
  - name: Taxonomy and overview of methodologies to create positive and negative pairs for contrastive learning
    subsections:
    - name: Positive pair creation taxonomy
    - name: Negative pair creation taxonomy
  - name: Crafting Effective Positive Pairs
    subsections:
    - name: Augmentation techniques
    - name: Embedding based techniques
    - name: Synthetic data generation for positive pairs
    - name: Supervised pairing techniques
    - name: Attribute based pairing techniques
    - name: Cross modal positive pairing techniques
  - name: Crafting Effective Negative Pairs
    subsections:
      - name: Hard negative selection
      - name: Removal of false negatives
      - name: Synthetic hard negatives
  - name: Discussion
  - name: Open Research Questions
  - name: Citations

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
## Contrastive Learning Formulation
<!-- Contrastive learning is a self-supervised learning approach that aims to learn representations by contrasting similar (positive) and dissimilar (negative) data pairs. The core idea is to map similar samples closer in the embedding space while pushing dissimilar ones apart. One of the commonly used loss function used for this is the InfoNCE loss which operates on a batch of anchor samples and their corresponding positive and negative pairs. Anchor samples are candidate samples in the dataset with respect to which both positive and negative samples are defined. InfoNCE loss is defined as follows: -->
Contrastive learning is a self-supervised learning approach designed to learn representations by contrasting pairs of data: similar pairs (positive) and dissimilar pairs (negative). The main idea is to position similar samples closer together in the embedding space while pushing dissimilar samples further apart. A commonly used loss function in this context is the InfoNCE loss, which operates on a batch of anchor samples alongside their corresponding positive and negative pairs. Anchor samples serve as reference points in the dataset for defining both positive and negative samples. The InfoNCE loss is defined as follows:

$$
\mathcal{L}_{\text{InfoNCE}} = - \frac{1}{N} \sum_{i=1}^{N} \log \frac{\exp\left(\text{sim}\left(\mathbf{z}_i, \mathbf{z}_i^+\right)/\tau\right)}{\exp\left(\text{sim}\left(\mathbf{z}_i, \mathbf{z}_i^+\right)/\tau\right)  + \sum_{j=1}^{N} \exp\left(\text{sim}\left(\mathbf{z}_i, \mathbf{z}_j\right)/\tau\right)}
$$

$$ z_i$$ is the representation of the anchor sample and $$ z_i^+$$ ​is the representation of the positive sample generated through augmenting the same instance or using a criteria to select another instance in the same dataset.  $$z_j$$ represents the representations of all samples in the batch (including negatives). $$\mathcal{sim}\left(\mathcal{.},\mathcal{.}\right)$$ denotes the similarity function (commonly cosine similarity). $$\tau$$ is the temperature scaling parameter and  $$N$$ is the number of samples in the batch.

<!-- **Given an anchor, a positive pair is either augmented version of the same sample or a distinct instance chosen from the dataset using some criteria. The negatives are other samples in the batch or a memory bank. The loss minimizes the distance between anchor-positive pairs while maximizing the distance between anchor-negative pairs**. This approach enables models to learn robust and meaningful representations without labeled data, making it powerful for downstream tasks like image classification, retrieval, and multimodal alignment. Effective contrastive learning depends on constructing diverse and informative positive and negative pairs, often leveraging techniques like data augmentation, hard negative mining, and synthetic pair generation. -->
Intuitively, **given an anchor, a positive pair is defined as either an augmented version of the same sample or a distinct instance selected from the dataset according to specific criteria. Negative pairs consist of other samples in the batch or from a memory bank. The loss function aims to minimize the distance between anchor-positive pairs while maximizing the distance between anchor-negative pairs.** This approach allows models to learn robust and meaningful representations without the need for labeled data, and has been shown to be highly effective for downstream tasks such as image classification, retrieval, and multimodal alignment. Effective contrastive learning relies on constructing diverse and informative positive and negative pairs, often utilizing techniques such as data augmentation, hard negative mining, and synthetic pair generation. 

In this blog post, we will first establish a taxonomy of existing techniques for creating positive and negative pairs in contrastive learning (as shown in Figure 1), summarizing the key methods within each category. Next, we will provide a detailed explanation of some key techniques in each sub-category of the taxonomy. Finally, we will discuss the trade-offs associated with using different data curation methods, concluding with a discussion on open questions in the area.


## Taxonomy and overview of methodologies to create positive and negative pairs for contrastive learning

{% include figure.html path="assets/img/2025-04-28-data-curation-for-contrastive-learning-crafting-effective-positive-and-negative-pairs/contrastive_taxonomy.png" class="img-fluid" %}
<div class="caption">
    Figure 1: Taxonomy for data curation.
</div>

### Positive pair creation taxonomy
<!-- This section explores techniques for creating positive pairs in contrastive learning, categorizing them into two major groups: same-instance positives and multi-instance positives. 

**Same-instance positives are generated by applying various data augmentations**, such as random crops and color changes, to a single instance. These augmented views are treated as positives, and the model learns to align their representations in the embedding space while avoiding trivial solutions like collapsing all representations to a single point. However, this approach has limitations, as *random augmentations cannot account for diverse variations, such as different viewpoints, object deformations, or semantically similar instances within the same class*. This means that the generalization ability relies heavily on the augmentation pipeline, which may not adequately capture all intrinsic variances. 

To address these shortcomings, **multi-instance positives provide a more diverse set of positive pairs by leveraging different instances** in the dataset. These pairs are generated using predefined criteria, such as *identifying semantically similar instances in embedding space, utilizing supervised labels from an oracle, incorporating online or offline human feedback, associating data across modalities like image-text pairs, creating synthetic positive pairs through generative models like GANs or diffusion models, or selecting pairs based on spatial or temporal similarities*, such as images captured at the same location or time. By offering diverse and meaningful positive pairs, multi-instance approaches allow models to learn richer, more generalizable representations that support a broader range of downstream tasks. -->

This section discusses techniques for creating positive pairs in contrastive learning, which can be categorized into two main groups: same-instance positives and multi-instance positives.



#### Same-instance positives

**Same-instance positives** are generated by applying various data augmentations, such as random cropping and color alterations, to a single instance. These augmented views are considered positive pairs, and the model learns to align their representations in the embedding space. However, this approach has limitations; random augmentations cannot account for diverse variations, such as different viewpoints, object deformations, or semantically similar instances within the same class. Consequently, the model's generalization ability heavily relies on the augmentation pipeline, which may not sufficiently capture all intrinsic variances. This approach is utilized in methods like SimCLR <d-cite key="chen2020simple"></d-cite> and MoCo <d-cite key="he2020momentum"></d-cite> to list a few methods.

<!-- - **Augmentation:** This entails applying different augmentations (e.g. random cropping, color distortion, gaussian blur) to the same data point to create positive pairs. This approach is utilized in methods like SimCLR <d-cite key="chen2020simple"></d-cite> and MoCo <d-cite key="he2020momentum"></d-cite> to list a few methods. -->

#### Multi-instance positives

To address the limitations of using same-instance positive pairs, **multi-instance positives** offer a more diverse set of positive pairs by using different instances from the dataset. These pairs are generated using predefined, often task-specific criteria. Examples include identifying semantically similar instances in the embedding space, utilizing supervised labels from an oracle, incorporating online or offline human feedback, associating data across modalities (such as image-text pairs), creating synthetic positive pairs through generative models like GANs or diffusion models, and selecting pairs based on spatial or temporal similarities, such as images captured in the same location or at the same time. By providing diverse and meaningful positive pairs, multi-instance approaches have been shown to enable models to learn richer, more generalizable representations that are beneficial for a broader range of downstream tasks. Multi-instance positive pair selection techniques lie in the following categories: 

- **Embedding-Based Similarity:** This technique identifies semantically similar samples in the embedding space to form positive pairs. For example, Nearest-Neighbour Contrastive Learning of visual Representations (NNCLR) <d-cite key="dwibedi2021little"></d-cite> retrieves the nearest neighbor of a sample as its positive pair. Similarly, MSF <d-cite key="koohpayegani2021mean"></d-cite>  propose to use the first and k nearest neighbors as the multiple instances positives. All4One <d-cite key="estepa2023all4one"></d-cite> improves MSF by incorporating a centroid contrastive objective to learn contextual information from multiple neighbors using transformer network.

- **Synthetic Data Generation:** This method uses generative models such as generative adversarial network(GAN) <d-cite key="wu2023synthetic"></d-cite> or diffusion models <d-cite key="zeng2024contrastive"></d-cite> to create synthetic data points that are semantically similar but distinct to the original, serving as positive pairs. <d-cite key="wu2023synthetic"></d-cite> is jointly trained with the main model to dynamically customize hard samples based on the training state of the main model. <d-cite key="zeng2024contrastive"></d-cite> replaces the features of the intermediate layers in diffusion model with the semantic features extracted from an anchor image during a random reversed diffusion process.  This results in the generation of images possessing similar semantic content to the anchor image but differing in background and context due to the randomness of features in other layers.

- **Supervised Pairing:** This techniques utilizes label information to create positive pairs from samples of the same class, as seen in Supervised Contrastive Learning (SupCon) <d-cite key="khosla2020supervised"></d-cite>. Another method <d-cite key="ghose2023tailoring"></d-cite> proposes to create pairs and train the model in online manner by using human guided feedback. <d-cite key="wang2022oracle"></d-cite> incorporates human or oracle feedback for a subset of samples to extend the set of positive instance pairs.

- **Attribute-Based Pairing:**  These techniques leverage specific attributes such as spatial location or temporal proximity to form positive pairs. For instance, geographically aligned images captured at different times can be paired <d-cite key="ayush2021geography"></d-cite>. The attributes used to generate optimal views for contrastive representation learning are task-dependent. 

- **Cross-Modal Positives:** Aligns samples across different modalities (e.g., images and text, audio, speech) that correspond to the same semantic content. <d-cite key="radford2021learning"></d-cite>, <d-cite key="wang2022image"></d-cite>, <d-cite key="baevski2020wav2vec"></d-cite>, <d-cite key="li2020unimo"></d-cite>, <d-cite key="morgado2021audio"></d-cite>

### Negative pair creation taxonomy
<!-- Aside from generating diverse positive pairs, it is equally important to carefully select and curate the negative pairs. The common approach of **defining negative pairs as any other instance in the dataset ignoring their semantic content might not be sufficient to learn robust representation**. It can result in false negatives if the other instance is semantically similar to the anchor sample in embedding space or belongs to the same class as the anchor. Hence, selecting multiple instances as negatives might enable the model to learn better representation. As such, there is no clear notion of “informativeness” incorporated in the negative selection process. **A selection mechanism should ideally capture negatives which satisfy properties such as: anchor semantic similarity and representativeness.** -->

In addition to generating a variety of positive pairs, it is also crucial to carefully select and curate negative pairs. The conventional approach of **defining negative pairs as any other instance in the dataset, without considering their semantic content, may not be sufficient for learning robust representations**. This can lead to false negatives if the other instance is semantically similar to the anchor sample in the embedding space or belongs to the same class as the anchor. Therefore, selecting multiple instances as negatives can help the model learn more effective representations. However, the current negative selection process lacks a clear notion of “informativeness.” **Ideally, a selection mechanism should capture negatives that meet specific criteria, such as semantic similarity to the anchor and overall representativeness**. In current literature, negative pair creation fall under the following categories: 

<!-- TODO: explain intuition first before explaining techniques for negative pair creation -->
- **Hard Negative Selection:** <d-cite key="Hardnegativemixing"></d-cite> extends the MoCo v2 framework by adding two sets of hard and harder negatives into the queue. The first set is a convex linear combination of pairs of its hardest existing negatives whereas the second set is created by mixing the negatives with the query. <d-cite key="unremix"></d-cite>, introduce UnReMix, a method designed to enhance contrastive learning by selecting hard negative samples based on three key factors: anchor similarity, model uncertainty and representativeness ensuring that negative samples are similar to the anchor point, making them challenging for the model to distinguish.

- **Removal of False Negatives:** <d-cite key="huynh2022boosting"></d-cite>  introduces methods to identify these false negatives and propose two strategies to mitigate their impact: elimination and attraction. False Negative Elimination identifies potential false negatives and excludes them from the negative sample set, preventing the model from learning misleading distinctions. In False Negative Attraction, instead of excluding false negatives, this strategy reclassifies them as positives, encouraging the model to learn representations that acknowledge their semantic similarity.

- **Synthetic Hard Negatives:**  <d-cite key="dong2024synthetic"></d-cite> proposes a method to enhance contrastive learning by generating synthetic hard negative samples. This approach involves mixing existing negative samples in the feature space to create more challenging negatives, encouraging the model to learn more discriminative representations. To address the issue of false negatives—samples incorrectly labeled as negative but semantically similar to the anchor, the paper incorporates a debiasing mechanism, ensuring the model focuses on truly dissimilar negative samples. 


Next, we dive into details of most commonly used techiques for crafting effective positive and negative pairs.

## Crafting Effective Positive Pairs

### 1. Same instance augmentation techniques
Data augmentation plays a crucial role in self-supervised learning by generating auxiliary versions of input data, which enhances the model’s robustness and ability to generalize. This process involves introducing various transformations to the original data, prompting the learning algorithm to identify consistent underlying patterns. By maintaining semantic integrity while altering visual appearance, augmentation instills a sense of consistency in the model’s representations. 

<!-- TODO: Write something about how all these techniques use same instance positives, but use different model architectures to generate different embeddings. Write here about how other surveys focus on different model architectures for self-supervised learning, but this survey is concerened about how data has been curated for self-supervised learning. -->

- *SimCLR* <d-cite key="chen2020simple"></d-cite> maximizes agreement between augmented views of the same data point using contrastive loss, relying on large batch sizes to sample enough negatives. 

- *MoCo* <d-cite key="he2020momentum"></d-cite> One drawback of SimCLR  is that it requires large negative samples, which can be computationally expensive. MoCo addresses this limitation by using a momentum encoder and memory bank to dynamically maintain a queue of negatives, enabling efficient training with smaller batches. 

- *SimSiam* <d-cite key="chen2021exploring"></d-cite> eliminates the need for negatives altogether, using a stop-gradient mechanism to prevent representation collapse in its Siamese architecture. 

- *BYOL* <d-cite key="grill2020bootstrap"></d-cite> simplifies learning by aligning predictions from an online network with a momentum-maintained target network, achieving strong performance without negatives. 

- *Barlow Twins* <d-cite key="zbontar2021barlow"></d-cite> focuses on redundancy reduction by aligning embeddings and decorrelating feature dimensions, avoiding collapse naturally without negatives or momentum.

- *DINO* <d-cite key="caron2021emerging"></d-cite> combines self-supervised learning with knowledge distillation using a teacher-student framework, producing high-quality embeddings that transfer well across tasks. 

- *VicReg* <d-cite key="bardes2021vicreg"></d-cite> introduces regularization to balance variance, invariance, and decorrelation in embeddings, ensuring quality without negatives or momentum encoders. 

- *SwAV* <d-cite key="caron2020unsupervised"></d-cite> uses clustering to align augmentations by mapping them to shared cluster assignments, achieving robust representations without direct contrastive loss. 

- *CPC* <d-cite key="oord2018representation"></d-cite> leverages contrastive loss in a latent space to predict future data segments, making it particularly effective for time-series tasks. 

- *SEED* <d-cite key="fang2021seed"></d-cite> simplifies training by using teacher-student distillation with pseudo-labels, reducing computational complexity while maintaining strong performance.

### 2. Embedding-based techniques
Nearest-Neighbour Contrastive Learning of visual Representations (NNLCR)  <d-cite key="dwibedi2021little"></d-cite> goes beyond single instance positives, i.e. the instance discrimination task so the model can learn better features that are **invariant to different viewpoints, deformations, and even intra-class variations.** The model is encouraged to generalize to new data-points that may not be covered by the data augmentation scheme at hand. In other words, *nearest neighbors of a sample in the embedding space act as small semantic perturbations that are not imaginary*, i.e. they are representative of actual semantic samples in the dataset. Building upon the SimCLR objective, NNCLR is defined as loss as below:

$$
\mathcal{L}_{\text{NNCLR}} = - \frac{1}{N} \sum_{i=1}^{N} \log \frac{\exp\left(\text{sim}\left(\text{NN}\left(\mathbf{z}_i, \mathcal{Q}\right),  \mathbf{z}_i^+\right) / \tau\right)}{\sum_{j=1}^{N} \exp\left(\text{sim}\left(\text{NN}\left(\mathbf{z}_i, \mathcal{Q}\right), \mathbf{z}_j \right)/ \tau\right)}
$$

where $$NN(z, \mathcal{Q})$$ is the nearest neighbor operator as defined below:


$$
NN(z, \mathcal{Q}) = argmin_{q \in \mathcal{Q}} || z - q ||_{2}
$$

{% include figure.html path="assets/img/2025-04-28-data-curation-for-contrastive-learning-crafting-effective-positive-and-negative-pairs/NNCLR.png" class="img-fluid" %}
<div class="caption">
    Figure 2: Overview of NNCLR training (from <d-cite key="dwibedi2021little"></d-cite>).
</div>
To obtain nearest-neighbors, NNCLR utilizes a support set that keeps embeddings of a subset of the dataset in memory as shown in Figure 2. This support set also gets constantly replenished during training. The support set is different from memory banks <d-cite key="tian2020contrastive"></d-cite>, <d-cite key="wu2018unsupervised"></d-cite> and queues <d-cite key="chen2020improved"></d-cite>, where the stored features are used as negatives. The support set is utilized for nearest neighbor search for retrieving cross-sample positives.


However, relying entirely on the first neighbour holds back the real potential of the approach. Mean Shift for Self-Supervised Learning (MSF) <d-cite key="koohpayegani2021mean"></d-cite> proposes the **use of k neighbours to increase the generalization capability of the model.** MSF employs simple mean-shift algorithm that learns representations by grouping images together without contrasting between them or adopting much of prior on the structure or number of the clusters as shown in Figure 3 below.

{% include figure.html path="assets/img/2025-04-28-data-curation-for-contrastive-learning-crafting-effective-positive-and-negative-pairs/MSF.png" class="img-fluid" %}

<div class="caption">
Figure 3: Similar to BYOL, MSF maintains two encoders (“target” and “online”). The online encoder is updated with gradient descent while the target encoder is the moving average of the online encoder. It augments an image twice and feed to both encoders. It adds the target embedding to the memory bank and look for its nearest neighbors in the memory bank. The target embedding itself will be the first nearest neighbor. It shifts the embedding of the input image towards the mean of its nearest neighbors to minimize the summation of those distances. Ideally, we can average the set of nearest neighbors to come up with a single target embedding, but since averaging is dependent on the choice of loss function, we simply minimize the summation of distances (from <d-cite key="koohpayegani2021mean"></d-cite>).
</div>

It simply “shifts” the embedding of each image to be close to the “mean” of the neighbors of its augmentation. Since the closest neighbor is always another augmentation of the same image, the model architecture is identical to BYOL when using only one nearest neighbor instead of K nearest neighbors. 
<!-- TODO: Last sentence is unclear -->

{% include figure.html path="assets/img/2025-04-28-data-curation-for-contrastive-learning-crafting-effective-positive-and-negative-pairs/knn_each_epoch_MSF.png" class="img-fluid" %}
<div class="caption"> 
Figure 4: For a random query images, this diagram shows how the nearest neighbors evolve at the learning time. Initially, NNs are not semantically quite related, but are close in low-level features. This signal is bootstrapped in the learning method and results in NNs of the late epochs which are mostly semantically related to the query image (from <d-cite key="koohpayegani2021mean"></d-cite>). 
</div>

**MSF has been shown to be very computationally expensive because the objective function needs to be computed for each neighbour (k times).** All4One <d-cite key="estepa2023all4one"></d-cite>  contrasts information from multiple
neighbours in a more efficient way by avoiding multiple computations of the objective function as shown in Figure 5 below.
{% include figure.html path="assets/img/2025-04-28-data-curation-for-contrastive-learning-crafting-effective-positive-and-negative-pairs/nnclr_all4one.png" class="img-fluid" %}

<div class="caption"> 
Figure 5: While NNCLR  only contrast the first neighbour, All4One creates representations that contain contextual information from the k NNs and contrast it in a single objective computation. It proposes the use of a new embedding constructed by a self attention mechanism, such as a transformer encoder, that combines the extracted neighbour representations in a single representation containing contextual information about all of them. Hence, we are able to contrast all the neighbours’ information on a single objective computation. It makes use of a Support Set that actively stores the representations computed during the training <d-cite key="dwibedi2021little"></d-cite> so that it can extract the required neighbours (from <d-cite key="estepa2023all4one"></d-cite>).   
</div>

<!-- TODO: simplify All4One explanation -->
Figure 6 shows that given a sequence of neighbor representations $$ nn^1_i $$, **All4One obtains a single representation $$ c_1 $$ that contains as much information as possible about the input sequence $$ nn^1_i $$ using self-attention based transformer network**. When computing self-attention <d-cite key="vaswani2017attention"></d-cite>, it mixes the representations of the input sequence in a weighted manner to compute an enriched vector of representations. Each element of this enriched vector contains contextual information about all the neighbors in the sequence. During training, for each sequence in $$ NN^1 $$, the process is made up of the following steps: 

(i) for each sequence $$ Seq_i $$ in $$ NN^1 $$, it adds sinusoidal positional encoding <d-cite key="vaswani2017attention"></d-cite>; 

(ii) then, it feeds the transformer encoder $$ \psi $$ with $$ Seq_i $$ ; 

(iii) inside the transformer encoder, self-attention is computed, and a new sequence is returned $$ Seq^c_i $$ ; 

(iv) it selects the first representation $$ Seq^c_{i \, 1} $$ in the returned sequence $$ Seq^c_i $$ as the centroid $$ c_i $$ as it aims to contrast a single representation that contains context information from the rest of the neighbors. 

After selecting the first representation on all sequences, it obtains a batch of representations defined as $$ C^1 $$.
All4One uses three different objective functions that contrast different representations: Centroid objective contrasts the contextual information extracted from multiple neighbors while the Neighbour objective assures diversity <d-cite key="dwibedi2021little"></d-cite>. Additionally, the Feature contrast objective measures the correlation of the generated features and increases their independence.


{% include figure.html path="assets/img/2025-04-28-data-curation-for-contrastive-learning-crafting-effective-positive-and-negative-pairs/All4One.png" class="img-fluid" %}
<div class="caption">
Figure 6: Complete architecture of All4One framework. Feature, Centroid and Neighbour contrast objective functions are indicated by red, purple, and green respectively. All4One uses three different objective functions that contrast different representations: Centroid objective contrasts the contextual information extracted from multiple neighbours while the Neighbour objective assures diversity <d-cite key="dwibedi2021little"></d-cite>. Additionally, the Feature contrast objective measures the correlation of the generated features and increases their independence 
(from  <d-cite key="estepa2023all4one"></d-cite>)
</div>



### 3. Synthetic data generation for positive pairs

In <d-cite key="wu2023synthetic"></d-cite>, the authors propose a data generation framework to improve contrastive learning training by joint sample generation and contrastive learning as shown in Figure 7. **The generator is jointly learned with the main model to dynamically customize hard samples based on the training state of the main model. Besides, a pair of data generators are proposed to generate similar but distinct samples as positive pairs. In joint learning, the hardness of a positive pair is progressively increased by decreasing their similarity.**

{% include figure.html path="assets/img/2025-04-28-data-curation-for-contrastive-learning-crafting-effective-positive-and-negative-pairs/GAN_based_synthetic_framework_image.png" class="img-fluid" %}
<div class="caption">
Figure 7: (Left) Generation of hard samples and hard positive pairs, and the joint learning of generator and the main contrastive model. It generates two similar but distinct raw samples, and use two views of each raw sample (four views in total) as positives, which are then fed into standard contrastive learning frameworks (e.g. SimCLR) for learning. No additional training data is used and no labels are used in the entire training pipeline. (Right) By joint learning, the generated positive pair becomes progressively harder by tracking the training state of the main model. These hard positive pairs help the main model cluster distinct yet similar objects
for better representations (from <d-cite key="wu2023synthetic"></d-cite>).
</div>

<d-cite key="wu2023synthetic"></d-cite> shows that one challenge is that low quality of synthetic samples degrades contrastive learning. They claim that simply using a GAN pre-trained on the same unlabeled dataset as contrastive learning to provide additional synthetic data to the main model cannot effectively improve and even degrades the learned representation. This is because the synthetic data has intrinsically lower quality than the real data <d-cite key="brock2019large"></d-cite> even if labels were available to train a class-conditional generator. When the dataset is unlabeled and the generator is trained in a non class-conditional way, the quality of synthetic data becomes worse (<d-cite key="zhao2020differentiable"></d-cite>, <d-cite key="miyato2018spectral"></d-cite>), which degrades the performance of the main model or only provides marginal benefits. To solve this problem, instead of using the standalone generator and contrastive main model, the model is jointly optimized by formulating a Min-Max game such that they compete with each other. As shown in Figure 7, there are two major components: the hard sample generator (red) and the main contrastive model (blue), which are jointly optimized. Joint learning effectively uses the available unlabeled training data, and no additional training data or labels are used.

One drawback of this GAN based framework in <d-cite key="wu2023synthetic"></d-cite> is that it requires simultaneous training of
the contrastive learning model and GAN, making it unstable and hard to control the quality of the generated positives. On the other hand, as shown in Figure 8 **<d-cite key="zeng2024contrastive"></d-cite> proposed a new approach called Contrastive Learning with Synthetic Positives (CLSP), which integrates additional synthetic positives into conventional contrastive training schema.** These synthetic positives exhibit rich deformations
and diverse backgrounds, allowing the model to emphasize generic features beyond naive data augmentation. Specifically, the diffusion based  method  in <d-cite key="zeng2024contrastive"></d-cite> employs the unconditional diffusion model to generate positives. Apart from their high-fidelity image generation ability, diffusion models have also shown excellent visual representation learning ability, even in the absence of labeled information.

{% include figure.html path="assets/img/2025-04-28-data-curation-for-contrastive-learning-crafting-effective-positive-and-negative-pairs/diffusion.png" class="img-fluid" %}
<div class="caption">
Figure 8: Overview of the proposed CLSP framework, we use a diffusion model to
generate an additional positive to increase the positive diversity for better representation
learning. (b) The t-SNE plot of features extracted from the intermediate layer of the diffusion model trained on CIFAR10. The features are generated at timestamp 50. (c) The generated images only contain background information if intermediate features are masked, suggesting the decoupling of semantic and background information
in different layers of the diffusion model. (d) Using feature interpolation to generate hard positives, the generated images contain similar semantic content to the anchor image but differ in context and background (from <d-cite key="zeng2024contrastive"></d-cite>).
</div>


**CLSP framework <d-cite key="zeng2024contrastive"></d-cite> uses a diffusion model to generate an additional positive sample $$x^3_i$$ and extract the embedding $$z^3_i$$ using the same encoder and projector employed in SimCLR. Given only an unlabeled anchor image, an unconditional diffusion model can generate diverse images that are semantically similar to the anchor image. It  leverages feature interpolation on the features from random sampling $$h$$ and features extracted from the diffusion process of the anchor image $$h_{anchor}$$ to generate new images resembling the anchor image. The interpolation process can be defined as: $$ h = w  * h + (1-w) * h_{anchor}$$**

Where $$w$$ is the interpolation weight controlling the similarity of generated
samples to the anchor image. Specifically, $$w=1$$ means pure random sampling
with no feature interpolation, but, $$w=0$$ does not mean the generated images are
identical to the anchor image because the background-related features remain
random. This approach can be viewed as a specialized form of strong data augmentation,
aiming to enhance the feature diversity of positive samples and guide the model’s attention toward semantically meaningful regions in the image. Practically, generating new positives on the fly for each image during training is unfeasible because of the slow sampling speed of diffusion models. Therefore, this method uses pre-generated a positive candidate set with k(k ≤ 8) samples for each anchor image in the training set and randomly sampled one as the additional positive for
CLSP training. This method can be extended to several frameworks including SimCLR, MoCo, SimSiam, etc.


### 4. Supervised pairing techniques
Supervised contrastive learning (SupCon) <d-cite key="khosla2020supervised"></d-cite> introduces a loss function for supervised learning that builds upon the principles of contrastive self-supervised learning by utilizing information from ground-truth labels. The key novelty of this approach is that it considers multiple positive samples for each anchor, in addition to many negative samples, unlike self-supervised contrastive learning, which typically uses only one positive sample. As shown in Figure 9, **these positive samples are selected from instances of the same class as the anchor, rather than being data augmentations of the anchor, as is done in self-supervised learning.**


{% include figure.html path="assets/img/2025-04-28-data-curation-for-contrastive-learning-crafting-effective-positive-and-negative-pairs/Supcon.png" class="img-fluid" %}
<div class="caption">
Figure 9: Supervised vs. self-supervised contrastive losses: The self-supervised contrastive loss (left, Eq. 1)
contrasts a single positive for each anchor (i.e., an augmented version of the same image) against a set of
negatives consisting of the entire remainder of the batch. The supervised contrastive loss (right), however, contrasts the set of all samples from the same class as positives against the negatives from the remainder of the batch. As demonstrated by the photo of the black and white puppy, taking class label information into account results in an embedding space where elements of the same class are more closely aligned than in the self-supervised case (from <d-cite key="khosla2020supervised"></d-cite>).
</div>

<!-- TODO: Sentence below is unclear -->
The enhancements to contrastive learning discussed focus on generalizing the framework to accommodate multiple positives, increasing the contrastive power with more negatives, and enabling implicit hard positive/negative mining. Supervised contrastive loss formulation is provided below:

$$
\mathcal{L}_{\text{SupCon}} = \sum_{i \in I}  -log\left(\frac{1}{|P(i)|} {\sum_{p \in P\left(i\right)}\frac{\exp\left(\text{sim}\left(\mathbf{z}_i, \mathbf{z}_p\right)/\tau\right)}{\sum_{a \in A\left(i\right)} \exp\left(\text{sim}\left(\mathbf{z}_i, \mathbf{z}_a\right)/\tau\right)}} \right)
$$


Here, $$ P(i) = \left(p \in A(i) : y_{p}^{~} = y_{i}^{~}\right) $$  is the set of indices of all positives in the multiviewed batch distinct from $$ i $$ and $$ P(i) $$ is its cardinality and $$ A(i) = I \setminus {i}$$, where $$i \in I = \{1,2...2N\}$$ is the index of an arbitrary augmented sample.



By including all positives in a batch—augmentation-based samples and others with the same label—in the numerator, **supervised contrastive learning encourages the encoder to align representations within the same class, resulting in more robust clustering compared to traditional single-positive setups. The summation over negatives is retained, consistent with noise contrastive estimation and N-pair losses, with the addition of more negatives improving the ability to distinguish signal from noise, leading to better  representation learning**. Additionally, the loss function's gradient structure inherently emphasizes hard positives and negatives, enabling the model to focus on challenging contrasts without the need for explicit hard mining strategies like triplet loss. These improvements enhance both supervised and self-supervised contrastive learning by making the framework more robust, scalable, and efficient.
<!-- TODO: missing citation for triplet loss, last sentence unclear -->


Building upon this, <d-cite key="ghose2023tailoring"></d-cite> select positive pairs in real-time by observing humans provide limited positive examples but no negative examples (similar to Positive Unlabeled Learning) while working collaboratively with a robot. This approach leverages contrastive learning to develop object representations that align with human expectations as shown in Figure 10. **By incorporating incremental human supervision, the robot can adapt its understanding of objects to meet specific human requirements, demonstrated in a real-time waste sorting task. Positive pairs are formed by grouping images of objects that were selected by the humans to be in the same positive category. This process enables the robot to associate different visual instances of the same object category, improving its generalization capabilities.**
<!-- TODO: cite survey on PU Learning -->

{% include figure.html path="assets/img/2025-04-28-data-curation-for-contrastive-learning-crafting-effective-positive-and-negative-pairs/recyling.png" class="img-fluid" %}
<div class="caption">
Figure 10: Data pairs are constructed using randomly applied data augmentations.
The extracted features are fed into three jointly trained heads. The Instance Projection Head projects the features into a space where each row denotes an augmentation of an image and uses the instance loss to minimize the distance between two augmentations of the same image. The Cluster Projection Head projects feature vectors into a space where each column denotes each instance’s cluster assignments, and the cluster loss minimizes the distance between instances belonging to a cluster. Finally, the Human-Supervised Head minimizes the distance between each human-selected example (from <d-cite key="ghose2023tailoring"></d-cite>).
</div>

The Instance Projection Head is a Multi-Layer Perceptron (MLP) with one hidden layer to map the features of two augmented views of the same image to a latent space similar to SimCLR. The Cluster Projection Head <d-cite key="li2021contrastive"></d-cite>  learns representations by maximizing agreement between two representations (under different augmentations) belonging to a given cluster. This head is an MLP with one hidden layer followed by a softmax operation. It projects a data instance into a latent space whose dimensionality equals the total number of pre-defined cluster $$\mathcal{K}$$, which should be approximately equal to $$\mathcal{C}$$. Intuitively, the cluster projection head tries to partition the embedding space into a pre-specified number of clusters based on the inter-instance similarity between object features. **The human-supervised head aims to guide the representation learning process towards human requirements. Intuitively, it tries to force the representations of the objects selected by the human close to each other in the latent space to serve as a mechanism to help inform the formation of clusters according to the properties of objects important to the human.**
This loss is applied only to the representations of objects selected by the human $$y_h$$. As new objects are selected by the human, this method passes their image ($$x_h$$) through the base encoder to get their feature representation ($$y_h = \mathcal{f}(x_h)$$). If previously there were $$s$$ human-selected objects and the human selects $$t$$ new objects, this method re-calculates the mean of the cluster $$\left(\mu_h^{'}\right) $$ to include the features of the new objects by $$  \mu_h^{'} = \frac{\mu_h*s + \sum_t{y_t}}{s+t}$$. 
The human-supervised loss then minimizes the distance between the new centroid of the human-selected objects and every object selected by the human.

Similarly, Oracle-guided Contrastive Clustering (OCC) <d-cite key="wang2022oracle"></d-cite> is a deep clustering framework designed to incorporate personalized clustering preferences by interactively querying oracles—entities with specific clustering criteria. Traditional deep clustering methods often aim solely to maximize clustering performance, potentially overlooking the unique demands of specific tasks. In contrast, OCC addresses this by engaging with oracles to obtain pairwise "same-cluster" queries, thereby guiding the clustering process according to the desired orientation.

### 5. Attribute-based pairing techniques
Attribute-based pairing entails selecting positive pairs based task-specific criteria. For instance, Geography-aware self supervised learning <d-cite key="ayush2021geography"></d-cite> **use temporal positive pairs from spatially aligned images over time** as shown in Figure 11.


{% include figure.html path="assets/img/2025-04-28-data-curation-for-contrastive-learning-crafting-effective-positive-and-negative-pairs/Geography_aware_framework.png" class="img-fluid" %}

<div class="caption">
Figure 11: Left shows the original MoCo-v2 framework. Right shows the schematic overview of Geography-aware self supervised learning approach (from <d-cite key="ayush2021geography"></d-cite>).
</div>

<!-- Remote sensing data is often geo-located and might provide multiple images of the same location over time. This approach proposed in <d-cite key="ayush2021geography"> enhances self-supervised learning models by incorporating geographic information, particularly in the context of remote sensing and geo-tagged datasets. This method leverages **spatially aligned images over time to create temporal positive pairs in contrastive learning, effectively capturing changes and patterns in geographic data.** In tasks that utilize geo-location data, the model learns representations sensitive to geographic context, improving performance in image classification, object detection, and semantic segmentation. This approach is particularly beneficial for remote sensing applications, where labeled data is scarce but unlabeled, geo-tagged data is abundant.  -->

Remote sensing data is often geo-located and can provide multiple images of the same location over time. The approach proposed by <d-cite key="ayush2021geography"></d-cite> enhances self-supervised learning models by incorporating geographic information, specifically for remote sensing and geo-tagged datasets. This method uses spatially aligned images over time to create temporal positive pairs in contrastive learning, which effectively captures changes and patterns in geographic data. In tasks that leverage geo-location data, the model learns representations that are sensitive to geographic context, thereby improving performance in image classification, object detection, and semantic segmentation. This approach is particularly advantageous for remote sensing applications, where labeled data is scarce, but unlabeled, geo-tagged data is abundant.

<!-- TODO: how does the paragraph below connect to previous paragraph, also it is confusing because MoCo is listed as single instance contrastige learning -->
Given an image $$ x_i^{t_1}$$ collected at time $$t_1$$, we can randomly select another image $$ x_i^{t_2}$$ that is spatially aligned with $$ x_i^{t_1}$$. A perturbation (e.g. random color jittering) as used in MoCo-v2 is applied to the spatially aligned image pair $$ x_i^{t_1}$$ and $$ x_i^{t_2}$$ providing us with a temporal positive pair. Note that when $$t1 = t2$$, the temporal
positive pair is the same as the positive pair used in MoCo-v2.
Given a data sample $$ x_i^{t_1}$$, TemporalInfoNCE objective function can be formulated as follows: 

$$
\mathcal{L}_{\mathbf{z}_i^{t_1}} = - \log \frac{\exp\left(\text{sim}\left(\mathbf{z}_i^{t_1}, \mathbf{z}_i^{t_2}\right)/\tau\right)}{\exp\left(\text{sim}\left(\mathbf{z}_i^{t_1}, \mathbf{z}_i^{t_2}\right)/\tau\right)  + \sum_{j=1}^{N} \exp\left(\text{sim}\left(\mathbf{z}_i^{t_1}, \mathbf{k}_j\right)/\tau\right)}
$$

where $$ z_i^{t_1}$$ and $$ z_i^{t_2}$$ are the encoded representations of the randomly perturbed temporal positive pair $$ x_i^{t_1}$$ and $$ x_i^{t_2}$$. N is number of negative samples, 
$$ \left(k_j \right)_{j=1}^N$$ are the encoded negative pairs and and $$\tau$$ is the temperature hyperparameter.

In remote sensing the content is often more stable across time due to the fixed viewpoint. For instance, a place on ocean is likely to remain as ocean for months or years, in which
case satellite images taken across time at the same location should share high semantic similarities. Even for locations where non-trivial changes do occur over time, certain semantic
similarities could still remain. For instance, key features of a construction site are likely to remain the same even as the appearance changes due to seasonality.


Similarly, <d-cite key="pantazis2021focus"></d-cite>  addresses the challenge of learning effective representations from unlabeled image collections, particularly those captured by static monitoring cameras used in biodiversity studies.  As shown in Figure 12, **this work leverages the natural variations present in sequential images from static cameras, utilizing contextual information such as spatial and temporal relationships to identify high-probability positive pairs—images likely depicting the same visual concept.**

{% include figure.html path="assets/img/2025-04-28-data-curation-for-contrastive-learning-crafting-effective-positive-and-negative-pairs/wildlife.png" class="img-fluid" %}
<div class="caption">
Figure 12: Conventional SSL typically generates augmented pairs of an input source image. Those are then passed to the model (right). Instead, we advocate that the careful selection of image pairs based on their context similarity (e.g. in space or
time), generates more varied and useful image pairs that result in more informative visual representations (from <d-cite key="pantazis2021focus"></d-cite>)
</div>

By focusing on these positive pairs during training, the authors demonstrate that the learned representations are more effective for downstream supervised classification tasks, even with limited human supervision. The approach was evaluated across four different camera trap image collections and three families of self-supervised learning methods, consistently outperforming conventional self-supervised training and transfer learning baselines. This methodology offers a promising avenue for enhancing global biodiversity monitoring efforts by improving species classification accuracy with minimal labeled data.

### 6. Cross-modal positive pairing techniques

<!-- TODO: Write an intro to cross modal pairing before explaining techniques -->

#### a. Image-Text Pairing

Image-text Pairing is fundamental in tasks like image-text retrieval, where the goal is to align visual and textual information effectively. By identifying and leveraging these positive pairs, models can learn shared representations that bridge the gap between modalities, enhancing performance in applications requiring multimodal understanding. This technique is fundamental in models like CLIP <d-cite key="radford2021learning"></d-cite> and BEiT-3 <d-cite key="wang2022image"></d-cite>, which aim to align visual and textual information effectively. Following are two key techniques that utilize image-text pairing:

- *CLIP (Contrastive Language-Image Pre-Training):* CLIP <d-cite key="radford2021learning"></d-cite> is trained on a vast dataset of image-text pairs. It employs a contrastive learning approach where positive pairs (matching images and texts) are brought closer in the embedding space, while negative pairs (non-matching images and texts) are pushed apart. This method enables CLIP to learn a shared representation space for both modalities, facilitating tasks such as zero-shot image classification and cross-modal retrieval. 


- *BEiT-3 (Bidirectional Encoder representation from Image Transformers):* BEiT-3 <d-cite key="wang2022image"></d-cite> extends the BEiT <d-cite key="bao2021beit"></d-cite> framework to handle both vision and vision-language tasks. It introduces a unified masked data modeling objective that operates on monomodal (images or texts) and multimodal (image-text pairs) data. By masking parts of the input and training the model to predict the missing information, BEiT-3 learns representations that capture the relationships between images and texts. This approach leverages cross-modal positive pairing to align visual and textual embeddings within a shared space, enhancing performance across various tasks. 


Both CLIP and BEiT-3 utilize cross-modal positive pairing to bridge the gap between visual and textual modalities, enabling more effective integration and understanding of multimodal information.


#### b. Audio-Image-Text Pairing

AudioCLIP <d-cite key="guzhov2022audioclip"></d-cite> is an extension of the CLIP framework designed to handle audio data alongside image and text modalities. It learns shared embeddings for audio, images, and text, allowing seamless retrieval and classification tasks across these modalities. AudioCLIP maps audio features to the same embedding space as images and text using contrastive learning. Positive pairs include audio-text or audio-image data, while unrelated samples serve as negatives.

#### c. Speech-Audio-Text Pairing

*Wav2Vec <d-cite key="baevski2020wav2vec"></d-cite>* is a family of models for self-supervised representation learning from raw audio waveforms. The wav2vec model is trained by predicting speech units for masked parts of speech audio. It learns basic units that are 25ms long to enable learning of high-level contextualized representations. This enables us to build speech recognition systems that can outperform the best semi-supervised methods, even with 100 times less labeled training data. The model learns a set of speech units, which are shorter than phonemes, to describe the speech audio sequence. Since this set is finite, the units encourage the model to focus on the most important factors to represent the speech audio.

#### d. Audio-Visual Pairing

*Audio-Visual Instance Discrimination (AVID)<d-cite key="morgado2021audio"></d-cite>* is a self-supervised learning approach designed to learn representations from both audio and visual data by leveraging their natural correspondence. Unlike traditional methods that focus on within-modal discrimination, AVID emphasizes cross-modal discrimination, aiming to align audio and visual features effectively. AVID employs a contrastive learning framework where the model learns to associate corresponding audio and visual pairs (positive pairs) while distinguishing them from non-corresponding pairs (negative pairs). This approach enhances the model's ability to capture the inherent relationships between audio and visual modalities.

### Crafting Effective Negative Pairs

<!-- TODO: Write summary of each technique before explaining papers -->
<!-- TODO: Explain how these techniques select positive pairs -->

### 1. Hard Negative Selection 

MoCHi, "(M)ixing (o)f (C)ontrastive (H)ard negat(i)ves" <d-cite key="Hardnegativemixing"></d-cite> creates synthetic hard negatives by combining features of existing hard negatives in the embedding space.  **MoCHi is designed to enhance self-supervised learning by generating challenging negative samples, thereby improving the quality of learned visual representations. Hard negatives—samples that are similar to the anchor but belong to different classes—are particularly valuable as they provide more informative gradients, leading to better feature learning.** Traditional methods often rely on large batch sizes or extensive memory banks to obtain a diverse set of negatives, which can be computationally expensive. The hard negative mixing strategy addresses this by synthesizing hard negatives through feature-level mixing: **for a given anchor, the model identifies existing negatives that are most similar in the embedding space and combines these hard negatives at the feature level to create synthetic negatives that are even closer to the anchor,** increasing the difficulty of the contrastive task.
<!-- TODO: Swap the ideas in the last two sentences: Traditional methods ... end, then MoCHi -->
 This method can be implemented on-the-fly with minimal computational overhead, making it efficient and scalable. By focusing on harder negatives, the model learns more discriminative features, leading to improved performance in downstream tasks such as classification and object detection. Synthesizing hard negatives reduces the need for large batch sizes or memory banks, lowering computational requirements. 

Another method named Uncertainty and Representativeness Mixing (UnReMix) for contrastive training <d-cite key="unremix"></d-cite>, introduces a method designed to enhance contrastive learning by selecting hard negative samples based on three key factors: **anchor similarity, model uncertainty and representativeness** ensuring that negative samples are similar to the anchor point, making them challenging for the model to distinguish. UnReMix utilizes uncertainty to penalize false hard negatives and pairwise distance among negatives to select representative examples as shown in Figure 13.

{% include figure.html path="assets/img/2025-04-28-data-curation-for-contrastive-learning-crafting-effective-positive-and-negative-pairs/Unremix.png" class="img-fluid" %}
<div class="caption">
Figure 13: Overview of the proposed hard negative sampling technique, UnReMix. Given an anchor
and a set of negative samples, UnReMix computes an importance score for each negative sample, by linearly interpolating between gradient-based uncertainty, anchor similarity and representativeness indicators, capturing desirable negative sample properties, i.e., samples that are truly negative (P1), in close vicinity to the anchor (P2) and representative of the sample population (P3) (from <d-cite key="unremix"></d-cite>)
</div>.

- *Anchor Similarity (P1):* This property assesses how closely a potential negative sample resembles the anchor (the reference data point). Incorporating anchor similarity ensures that selected negative samples are challenging, as they are similar to the anchor yet belong to different classes, thereby providing informative gradients for the model.

- *Model Uncertainty (P2):* Model uncertainty evaluates the confidence level of the model's predictions for a given sample. By selecting negative samples where the model exhibits higher uncertainty, the training process focuses on areas where the model is less certain, promoting learning in challenging regions of the data space.

- *Representativeness (P3):* Representativeness measures how well a sample reflects the overall data distribution. Incorporating this property ensures that the selected negative samples are not outliers but are indicative of the broader dataset, leading to more generalized and robust representations.

In summary, UnReMix selects hard negative samples based on calculated importance scores. The
higher the score is, the more informative the sample is assumed to be. The importance scores consist
of three components: (1) a model-based component that utilizes the loss gradients w.r.t. each negative
sample as a measure of uncertainty and approximates P1 by assigning more weight to the negative
examples that lie closer to the decision boundary, (2) a feature-based component that leverages the
feature space geometry via instance similarity to select informative negative samples, that satisfy P2,
and (3) a density-based component that assigns more weight to negatives examples that are more
distant on average from other negative examples in the batch, and satisfies P3.


### 2. Removal of False Negatives
Boosting Contrastive Self-Supervised Learning with False Negative Cancellation <d-cite key="huynh2022boosting"></d-cite> introduces methods to identify these false negatives and propose two strategies to mitigate their impact: elimination and attraction. False negatives are samples from different images with the same semantic content, therefore they should hold certain similarity (e.g., dog features). **A false negative may not be as similar to the anchor as it is to other augmentations of the same image, as each augmentation only holds a specific view of the object.   Contrasting false negatives induces two critical issues in representation learning: discarding semantic information and slow convergence.** As shown in Figure 14, <d-cite key="huynh2022boosting"></d-cite> proposes novel approaches to identify false negatives, as well as two strategies to mitigate their effect, i.e. false negative elimination and attraction. **False Negative Elimination identifies potential false negatives and excludes them from the negative sample set**, preventing the model from learning misleading distinctions. In **False Negative Attraction**, instead of excluding false negatives, **this strategy reclassifies them as positives, encouraging the model to learn representations that acknowledge their semantic similarity.**
{% include figure.html path="assets/img/2025-04-28-data-curation-for-contrastive-learning-crafting-effective-positive-and-negative-pairs/boosting_fn.png" class="img-fluid" %}
<div class="caption">
Figure 14: False negatives in contrastive learning. Without knowledge of labels, automatically selected negative pairs could actually belong to the same semantic category, creating false negatives (from <d-cite key="huynh2022boosting"></d-cite>)
</div>
False negatives hinder the convergence of contrastive learning-based objectives due to the appearance
of contradicting objectives. For instance, in Figure 14, the dog’s head on the left is attracted to its fur (positive pair), but repelled from similar fur of another dog image on the right (negative pair), creating contradicting objectives.

{% include figure.html path="assets/img/2025-04-28-data-curation-for-contrastive-learning-crafting-effective-positive-and-negative-pairs/boosting_fn_2.png" class="img-fluid" %}
<div class="caption">
Figure 15: Overview of the proposed framework. Left: Original definition of the anchor, positive, and negative samples in contrastive learning. Middle: Identification of false negatives (blue). Right: false negative cancellation strategies, i.e. elimination and attraction (from <d-cite key="huynh2022boosting"></d-cite>)
</div>

Figure 15 shows the overview of the method.

- *False Negative Cancellation:* Negative pairs are samples from different images that may or may not possess similar semantic content or visual
features. Consequently, it is possible that some samples k have the same semantic content as the anchor i, and are thus false negatives. As discussed earlier, false negatives give rise to two critical problems in contrastive learning: they discard semantic information and slow convergence

- *False Negative Attraction:* While eliminating false negatives alleviates the undesireable effects of contrasting against them, it ignores information available in what are actually true positives. Minimizing the original contrastive loss  only seeks to attract an anchor to different views of the same image. Including true positives drawn from different images would increase the diversity of the training data and, in turn, has the potential to improve the quality of the learned embeddings.
Indeed, Khosla et al. <d-cite key="khosla2020supervised"></d-cite>  show that supervised contrastive
learning (i.e., where an anchor is attracted to samples having the same semantic label) can be more effective than the traditional supervised cross-entropy loss. Thus, this method proposes to treat the false negatives that have been identified as true positives and attract the anchor to this set.


### 3. Synthetic Hard Negative Samples for Contrastive Learning
Synthetic Hard Negative Samples for Contrastive Learning <d-cite key="dong2024synthetic"></d-cite> proposes a method to enhance contrastive learning by generating synthetic hard negative samples as shown in Figure 16. This approach **involves mixing existing negative samples in the feature space to create more challenging negatives, encouraging the model to learn more discriminative representations. To address the issue of false negatives—samples incorrectly labeled as negative but semantically similar to the anchor, this paper incorporates a debiasing mechanism, ensuring the model focuses on truly dissimilar negative samples.**
 {% include figure.html path="assets/img/2025-04-28-data-curation-for-contrastive-learning-crafting-effective-positive-and-negative-pairs/synthetic.png" class="img-fluid" %}
<div class="caption">
Figure 16: The architecture of SSCL. Given a batch of N images, this method first applies data augmentation and then encode and projection head to obtain two batch feature vectors (2N). For one image, it generates new harder negative samples by synthesizing negative samples from the 2N-2 remaining images. The red dashed arrow indicates the projection of the feature vector.We then combine the original samples with the newly synthesized ones to compute the contrastive loss (from <d-cite key="dong2024synthetic"></d-cite>).
</div>
The process involves the following steps:


- *Identifying Hard Negatives:* For a given anchor sample, the method selects the hardest negative samples from the available negative set based on their similarity to the anchor. These hard negatives are the ones most similar to the anchor but belong to different classes.

- *Synthesizing Harder Negatives:* The selected hard negatives are then combined through linear interpolation to create synthetic negative samples that are even closer to the anchor in the feature space. This interpolation is controlled by a parameter that adjusts the degree of mixing, allowing the generation of negatives with varying levels of hardness.

- *Sampling Synthetic Negatives:* The synthetic hard negatives are sampled by controlling the contrast between the anchor and the negative samples. This ensures that the generated negatives are challenging enough to improve the model's discriminative ability.


Thus, this method not only generates amore diverse set of challenging negative samples but also conducts weighted sampling while ensuring the truthfulness of these negative samples.


## Discussion

In the context of the methods explained above, in this section, we discuss the drawbacks of generating positive pairs from the same instance and the tradeoffs associated with using different techniques to generate positive and negative pairs from multiple instances. 

### Drawback of generating positive pairs from same instance

<!-- *Same-instance positive pairs*, generated through augmentations like random cropping or color distortion, **have inherent limitations in capturing diverse variations such as different viewpoints, object deformations, or semantic similarities across instances within the same class**. These approaches place the burden of generalization on the data augmentation pipeline, which may fail to encompass the full range of real-world data variations, leading to suboptimal representations. To address these limitations, multi-instance positive pair techniques offer diverse alternatives. -->

*Same-instance positive pairs*, generated through augmentations like random cropping or color distortion, **have inherent limitations in capturing diverse variations such as different viewpoints, object deformations, and semantic similarities across instances within the same class**. These methods place the burden of generalization on the data augmentation process, which may not encompass the full range of real-world data variations. This can result in suboptimal representations. Multi-instance positive pair techniques provide diverse alternatives to address these limitations.

### Trade-offs across various techniques to generate positive pair from multiple instances

<!-- **Embedding-based positive pairs utilize semantic similarity** in the embedding space, enabling the model to achieve better generalization.
Embedding-based generation is crucial for tasks where subtle differences within the same class must be captured, such as species classification in biodiversity monitoring or quality assessment in manufacturing. Traditional data augmentations may not provide the nuanced variations needed for fine-grained tasks. By identifying semantically similar samples in the embedding space, **embedding-based methods ensure that positive pairs reflect subtle variations, helping models learn fine-grained features.** For example, in classifying different types of birds or flowers, embedding-based positive pairs allow the model to focus on detailed patterns like feather color or petal shape. Embedding-based methods are ideal for generating hard positives—pairs of samples that are similar but not identical.
However, this approach is **computationally intensive and vulnerable to noisy embeddings, particularly during the early stages of training.** A well-trained embedding model capable of producing high-quality representations for each data sample is essential for its success. The generalization ability of the embedding model on the current dataset significantly impacts the quality of representations learned through contrastive learning.  -->

**Embedding-based positive pairs leverage semantic similarity** in the embedding space, enabling the model to achieve better generalization. This approach is essential for tasks that require capturing subtle differences within the same class, such as species classification in biodiversity monitoring or quality assessment in manufacturing. Traditional data augmentation techniques may not provide the nuanced variations necessary for fine-grained tasks. By identifying semantically similar samples in the embedding space, **embedding-based methods ensure that positive pairs reflect subtle variations, allowing models to learn intricate features.** For instance, embedding-based positive pairs help the model focus on detailed patterns such as feather color or petal shape when classifying different types of birds or flowers. These methods are particularly effective for generating challenging positives—pairs of similar but not identical samples.
However, this approach can be **computationally intensive and may be susceptible to noisy embeddings, especially during the early stages of training.** A well-trained embedding model that can produce high-quality representations for each data sample is crucial for success. The generalization capability of the embedding model on the current dataset significantly influences the quality of the representations learned through contrastive learning.

<!-- **Synthetic data generation** creates challenging positive pairs dynamically, improving model robustness but requiring careful tuning to avoid degradation from low-quality synthetic samples. One major **challenge is the visual fidelity of synthetic samples.** Even with advanced generative models, synthetic images often lack the richness or detail of real-world data, resulting in positive pairs that do not fully capture the desired semantic similarity. Moreover, **contextual discrepancies—differences in background, lighting, or object relationships**—further exacerbate the domain gap, limiting the relevance of synthetic samples in representation learning. Another issue is **semantic misalignment, where synthetic samples might unintentionally introduce artifacts or distortions that diverge from real-world semantics.** These deviations can confuse the model during training, leading to representations that overfit to synthetic peculiarities rather than capturing generalized features. **Addressing domain gap challenges requires novel methods** for both synthetic data generation and contrastive learning. Combining synthetic positives with real-world data, leveraging hybrid training strategies, or incorporating domain adaptation methods can help bridge the gap, enhancing the utility of synthetic data for positive pair generation. Furthermore, **in domains with class imbalances or the absence of negative samples, synthetic data can augment underrepresented classes, improving the model’s ability to learn robust representations.** -->

**Synthetic data generation** dynamically creates challenging positive pairs, enhancing model robustness. However, it requires careful tuning to prevent degradation from low-quality synthetic samples. A significant challenge is the **visual fidelity of synthetic samples.** Even with state-of-the-art generative models, synthetic images often lack the richness and detail of real-world data. This can result in positive pairs that do not fully capture the desired semantic similarity. Additionally, **contextual discrepancies—such as differences in background, lighting, or object relationships**—further widen the domain gap, limiting the relevance of synthetic samples in representation learning.
Another concern is **semantic misalignment,** where synthetic samples may inadvertently introduce artifacts or distortions that diverge from real-world semantics. These deviations can confuse the model during training, leading to representations that overfit to synthetic peculiarities instead of capturing generalized features. **Addressing domain gap challenges requires novel methods** for both synthetic data generation and contrastive learning. Strategies like combining synthetic positives with real-world data, using hybrid training approaches, or incorporating domain adaptation methods can help bridge the gap, enhancing the effectiveness of synthetic data for positive pair generation. Furthermore, **in domains with class imbalances or the lack of negative samples, synthetic data can augment underrepresented classes, improving the model’s ability to learn robust representations.**

<!-- TODO: paragraph below is unclear -->

**Supervised pairing leverages label information to create positive pairs**, providing rich semantic context and ensuring robust learning. This technique is particularly beneficial in scenarios where **labeled data is available and class-specific clustering is essential.** It is useful for tasks such as class-specific retrieval where the goal is to fetch semantically similar items within the same class or tasks requiring discrimination between subtle intra-class variations, such as species identification, product categorization, or medical diagnosis. The biggest drawback is that this method assumes availability of labels which might not be possible to obtain in every scenario.

**Attribute-based pairing** leverages contextual attributes like spatial, temporal, or other task-specific features to create positive pairs. **This approach is especially effective in scenarios where domain-specific context plays a crucial role in learning robust representations**. One challenge associates with this method is  that **attributes may be unevenly distributed across the dataset,** leading to overrepresentation of certain pair types. Another challenges associated with this technique is that the models may overfit to specific attribute values instead of learning generalizable representations.

**Cross-modal pair generation** is useful for multimodal learning which is becoming very common in recent times. However, this method requires availability of multimodal pairs which are semantically aligned with one another else the representations learned might not be meaningful.

### Trade-offs across various techniques to generate negative pair from multiple instances

Techniques like *hard negative mixing* synthesize harder negatives by interpolating between existing negatives, making the task more challenging for the model. However, **overemphasis on hard negatives can lead to overfitting, where the model learns to differentiate subtle, unimportant variations rather than capturing meaningful representations.** A balanced mix of hard and easy negatives is essential for effective training. While hard negatives provide valuable gradients, easy negatives ensure stability and prevent overfitting to challenging examples. Methods like UnReMix address this balance by incorporating representativeness and model uncertainty in the selection process. By ensuring that negatives are both challenging and representative of the broader data distribution, these methods enhance the robustness of learned representations.

**Synthetic negatives offer a scalable solution** for creating diverse and challenging negative pairs but suffer from the same challenge as synthetic data generation for positive pairs. 

Another **challenges with generating negative pairs is computational efficiency.** The computational cost of maintaining a large pool of negatives or synthesizing new ones is a critical consideration. Methods like momentum contrast (MoCo) address this by using memory banks or queues to store embeddings, enabling efficient retrieval of negatives. However, these approaches require careful tuning to balance representation stability and adaptability. Advances in scalable negative sampling techniques are crucial for deploying contrastive learning in large-scale applications.

The definition and **selection of negatives often depend on the domain and task.** For instance, in multimodal contrastive learning, negatives must account for cross-modal discrepancies, such as the semantic gap between text and images. Similarly, in temporal tasks, negatives from non-overlapping time intervals may introduce unwanted biases. Tailoring negative selection strategies to the specific characteristics of the domain is essential for achieving optimal results.

## Open Research Questions

Even though significant work has been done in data curation for contrastive learning, many open questions remain. Below are some of these questions:

### Balancing Diversity vs Relevance in pairs

Balancing diversity and semantic alignment in positive pair creation is a critical challenge in contrastive learning. Both factors are essential: **diversity ensures robustness and generalization, while semantic alignment guarantees meaningful and task-relevant representations.** Striking the right balance between these two can significantly impact the performance of the learned embeddings for a given task. **Diversity improves generalization across unseen domains or test data.** It also  encourages the model to focus on core semantic features rather than superficial patterns. However, **excessive diversity can dilute semantic similarity,** making it harder for the model to learn meaningful representations.
Diverse positive pairs may include instances that, while related, are semantically weakly aligned, leading to noisy embeddings. For instance, in wildlife monitoring, temporally diverse images of the same species might include drastic environmental or seasonal differences, reducing alignment. Conversely, tightly aligned pairs might miss important variations like behavioral changes. For tasks requiring generalization (e.g., unsupervised representation learning), diversity might take precedence. For tasks requiring fine-grained distinctions (e.g., medical imaging), semantic alignment is more critical. Some ways to address this might involve adjusting the balance dynamically during training. Early training phases might prioritize semantic alignment to establish core features. Later phases might introduce diversity to improve robustness. We can define  positives based on similarity scores rather than binary criteria. This allows pairs with partial semantic alignment to contribute proportionally. By carefully navigating this trade-off, models can achieve both robust generalization and meaningful class-specific representations. 

### Dealing with emerging modalities in contrastive learning
As new modalities—such as **LiDAR, hyperspectral imaging, and haptic feedback** —become prominent in various applications, the challenge of integrating these modalities into contrastive learning frameworks emerges. These modalities often come with unique characteristics, data structures, and challenges that require specialized strategies for effective learning. Many **emerging modalities suffer from a lack of large-scale labeled or even unlabeled datasets.** Emerging modalities often exhibit **high levels of noise or variability** due to environmental factors or inherent measurement inaccuracies. Moreover, unlike text or images, **pretrained models for emerging modalities are rare,** making initialization and representation learning more challenging. Future research must focus on scalable, robust, and efficient strategies to handle the diversity and complexity of these new data types.