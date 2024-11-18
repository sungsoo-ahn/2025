---
layout: distill
title: Data Curation for Contrastive Learning- Crafting Effective Positive and Negative Pairs
description: Contrastive learning aims to learn a robust representation of the underlying dataset that can be useful for a variety of downstream tasks. It follows a simple yet powerful technique of creating positive and negative pairs of data points(images, text, speech or multimodal pairs) to learn the representation. The design of positive and negative pairs is crucial in contrastive learning, as it directly impacts the quality of learned representations. Hence, a considerable amount of research has been conducted in this direction to create these pairs. The aim of this blog post is provide a unique perspective on different high level methodlogies to create such pairs and dive deeper into the techniques under each category.
date: 2025-04-28
future: true
htmlwidgets: true
hidden: false

# Anonymize when submitting
# authors:
#   - name: Anonymous

authors:
  - name: Albert Einstein
    url: "https://en.wikipedia.org/wiki/Albert_Einstein"
    affiliations:
      name: IAS, Princeton
  - name: Boris Podolsky
    url: "https://en.wikipedia.org/wiki/Boris_Podolsky"
    affiliations:
      name: IAS, Princeton
  - name: Nathan Rosen
    url: "https://en.wikipedia.org/wiki/Nathan_Rosen"
    affiliations:
      name: IAS, Princeton

# must be the exact same name as your blogpost
bibliography: 2025-04-28-data_curation_for_contrastive_learning.bib

# Add a table of contents to your post.
#   - make sure that TOC names match the actual section names
#     for hyperlinks within the post to work correctly. 
#   - please use this format rather than manually creating a markdown table of contents.
toc:
  - name: Equations
  - name: Images and Figures
    subsections:
    - name: Interactive Figures
  - name: Citations
  - name: Footnotes
  - name: Code Blocks
  - name: Diagrams
  - name: Tweets
  - name: Layouts
  - name: Other Typography?

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
## Contrastive Learning Formulation:
Contrastive learning aims to learn a robust representation of the underlying dataset that can be useful for a variety of downstream tasks. It follows a simple yet powerful technique of creating positive and negative pairs of data points(images, text, speech or multimodal pairs) to learn the representation. The design of positive and negative pairs is crucial in contrastive learning, as it directly impacts the quality of learned representations. Hence, a considerable amount of research has been conducted in this direction to create these pairs. The aim of this blog post is provide a unique  perspective on different high level methodlogies to create such pairs and dive deeper into the techniques under each category.

## Taxonomy and overview of methodologies to create positive and negative pairs for contrastive learning:

### Positive pair creation:
#### Same instance positive
- **Augmentation:** Apply different augmentations (e.g., random cropping, color distortion, gaussian blur) to the same data point to create positive pairs. This approach is utilized in methods like SimCLR <d-cite key="chen2020simple"></d-cite> and MoCo <d-cite key="he2020momentum"></d-cite>.

#### Multiple instances positives
- **Embedding-Based Similarity:** Identify semantically similar samples in the embedding space to form positive pairs. For example, Nearest-Neighbour Contrastive Learning of visual Representations (NNCLR) <d-cite key="dwibedi2021little"></d-cite> retrieves the nearest neighbor of a sample as its positive pair. Similarly, MSF <d-cite key="koohpayegani2021mean"></d-cite>  propose to use the first and k nearest neighbors as the multiple instances positives. All4One <d-cite key="estepa2023all4one"></d-cite> improves them by incorporating a centroid contrastive objective to learn contextual information from multiple neighbors.
- **Synthetic Data Generation:** Use generative models like GANs <d-cite key="wu2023synthetic"></d-cite> or diffusion models <d-cite key="zeng2024contrastive"></d-cite> to create synthetic data points that are semantically similar but distinct to the original, serving as positive pairs. <d-cite key="wu2023synthetic"></d-cite> generates hard samples which are similar but distinct samples as positive pairs. It is jointly trained with the main model to dynamically customize hard samples based on the training state of the main model. <d-cite key="zeng2024contrastive"></d-cite> replaces the features of the intermediate layers with the semantic features extracted from an anchor image during a random reversed diffusion process.  This results in the generation of images possessing similar semantic content to the anchor image but differing in background and context due to the randomness of features in other layers.
- **Supervised Pairing:** Utilize label information to create positive pairs from samples of the same class, as seen in Supervised Contrastive Learning (SupCon) <d-cite key="khosla2020supervised"></d-cite>. <d-cite key="ghose2023tailoring"></d-cite> proposes to create pairs and train the model in online manner by using human guided feedback while the other autonomous agent is performing a task. <d-cite key="wang2022oracle"></d-cite> incorporates human or oracle feedback for a subset of samples to extend the set of positive instance pairs.
- **Attribute-Based Pairing:** Leverage specific attributes such as spatial location or temporal proximity to form positive pairs. For instance, geographically aligned images captured at different times can be paired <d-cite key="ayush2021geography"></d-cite>. The attributes used to generate optimal views for contrastive representation learning are task-dependent. 
- **Cross-Modal Positives:** Align data from different modalities (e.g., images and text) that correspond to the same semantic content.<d-cite key="radford2021learning"></d-cite>, <d-cite key="li2020unimo"></d-cite>. This approach is employed in Vision-Language Models (VLMs).

### Negative pair creation:
- **Hard Negative Selection:** <d-cite key="Hardnegativemixing"></d-cite> creates synthetic hard negatives by combining features of existing hard negatives in the embedding space. <d-cite key="unremix"></d-cite>, introduce UnReMix, a method designed to enhance contrastive learning by selecting hard negative samples based on three key factors: anchor similarity, model uncertainty and representativeness ensuring that negative samples are similar to the anchor point, making them challenging for the model to distinguish. <d-cite key="Hardnegativecontrastive"></d-cite>  propose methods to sample hard negatives—data points that are difficult to distinguish from an anchor point—without relying on label information. Their approach allows for controlling the hardness of negative samples, leading to representations that tightly cluster similar data points and push dissimilar ones apart. 
- **Removal of False Negatives:** <d-cite key="huynh2022boosting"></d-cite>  introduces methods to identify these false negatives and propose two strategies to mitigate their impact: elimination and attraction.
False Negative Elimination identifies potential false negatives and excludes them from the negative sample set, preventing the model from learning misleading distinctions. In False Negative Attraction, instead of excluding false negatives, this strategy reclassifies them as positives, encouraging the model to learn representations that acknowledge their semantic similarity.
- **Synthetic Hard Negative Samples for Contrastive Learning:**  <d-cite key="dong2024synthetic"></d-cite> proposes a method to enhance contrastive learning by generating synthetic hard negative samples. This approach involves mixing existing negative samples in the feature space to create more challenging negatives, encouraging the model to learn more discriminative representations. To address the issue of false negatives—samples incorrectly labeled as negative but semantically similar to the anchor, the paper incorporates a debiasing mechanism, ensuring the model focuses on truly dissimilar negative samples.
- **Negative Sampling Correction:** <d-cite key="wang2024contrastive"></d-cite> propose Positive-Unlabeled Contrastive Learning (PUCL), a method that treats generated negative samples as unlabeled and leverages information from positive samples to correct this bias. The authors demonstrate that the corrected loss in PUCL incurs negligible bias compared to the unbiased contrastive loss, ensuring more accurate representation learning. Empirical results show that PUCL outperforms state-of-the-art methods in various image and graph classification tasks, highlighting its effectiveness in practical applications. 



# Positive Pair Creation Methodologies in Contrastive Learning

## Techniques to generate positive pair using same instance

### Augmentation technique:
Data augmentation plays a crucial role in self-supervised learning by generating auxiliary versions of input data, which enhances the model’s robustness and ability to generalize. This process involves introducing various transformations to the original data, prompting the learning algorithm to identify consistent underlying patterns. By maintaining semantic integrity while altering visual appearance, augmentation instills a sense of consistency in the model’s representations. 

- *SimCLR* <d-cite key="chen2020simple"></d-cite> maximizes agreement between augmented views of the same data point using contrastive loss, relying on large batch sizes to sample enough negatives. 

- *MoCo* <d-cite key="he2020momentum"></d-cite> addresses this limitation by using a momentum encoder and memory bank to dynamically maintain a queue of negatives, enabling efficient training with smaller batches. SimSiam <d-cite key="chen2021exploring"></d-cite> eliminates the need for negatives altogether, using a stop-gradient mechanism to prevent representation collapse in its Siamese architecture. 

- *BYOL* <d-cite key="grill2020bootstrap"></d-cite> further simplifies learning by aligning predictions from an online network with a momentum-maintained target network, achieving strong performance without negatives. 

- *Barlow Twins* <d-cite key="zbontar2021barlow"></d-cite> focuses on redundancy reduction by aligning embeddings and decorrelating feature dimensions, avoiding collapse naturally without negatives or momentum.

- *DINO* <d-cite key="caron2021emerging"></d-cite> combines self-supervised learning with knowledge distillation using a teacher-student framework, producing high-quality embeddings that transfer well across tasks. 

- *VicReg* <d-cite key="bardes2021vicreg"></d-cite> introduces regularization to balance variance, invariance, and decorrelation in embeddings, ensuring quality without negatives or momentum encoders. 

- *SwAV* <d-cite key="caron2020unsupervised"></d-cite> uses clustering to align augmentations by mapping them to shared cluster assignments, achieving robust representations without direct contrastive loss. 

- *CPC* <d-cite key="oord2018representation"></d-cite> leverages contrastive loss in a latent space to predict future data segments, making it particularly effective for time-series tasks. 

- *SEED* <d-cite key="fang2021seed"></d-cite> simplifies training by using teacher-student distillation with pseudo-labels, reducing computational complexity while maintaining strong performance. Collectively, these methods advance self-supervised learning by addressing key challenges such as reliance on negatives, computational efficiency, and representation quality.

## Techniques to generate positive pair using semantically similar but distinct instances:

### 1. Embedding-based technique:
Nearest-Neighbour Contrastive Learning of visual Representations (NNLCR)  <d-cite key="dwibedi2021little"></d-cite> goes beyond single instance positives, i.e. the instance discrimination task so the model can learn better features that are invariant to different viewpoints, deformations, and even intra-class variations. The model is encouraged to generalize to new data-points that may not be covered by the data augmentation scheme at hand. In other words, nearestneighbors of a sample in the embedding space act as small semantic perturbations that are not imaginary, i.e. they are
representative of actual semantic samples in the dataset. To obtain nearest-neighbors, we utilize a
support set that keeps embeddings of a subset of the dataset in memory. This support set also gets constantly replenished during training. Note that our support set is different
from memory banks <d-cite key="tian2020contrastive"></d-cite>, <d-cite key="wu2018unsupervised"></d-cite> and queues <d-cite key="chen2020improved"></d-cite>, where the stored features are used as negatives. They utilize the support set for nearest neighbor search for retrieving cross-sample positives.
{% include figure.html path="assets/img/2025-04-28-data_curation_for_contrastive_learning/NNCLR.png" class="img-fluid" %}
<div class="caption">
    Figure 2: Overview of NNCLR training. <d-cite key="dwibedi2021little"></d-cite>.
</div>

However, relying entirely on the first neighbour holds back the real potential of the approach. Mean Shift for Self-Supervised Learning (MSF) <d-cite key="koohpayegani2021mean"></d-cite> proposes the use of k neighbours to increase the generalization capability of the model. MSF employs simple mean-shift algorithm that learns representations by grouping images together without contrasting between them or adopting much of prior on the structure or number of the clusters. It simply “shifts” the embedding of each image to be close to the “mean” of the neighbors of its augmentation. Since the closest neighbor is always another augmentation of the same image, the model is identical to BYOL when using only one nearest neighbor instead of K nearest neighbors.

{% include figure.html path="assets/img/2025-04-28-data_curation_for_contrastive_learning/MSF.png" class="img-fluid" %}
<div class="caption">
Figure 3: Similar to BYOL, MSF maintains two encoders (“target” and “online”). The online encoder is updated with gradient descent while the target encoder is the moving average of the online encoder. We augment an image twice and feed to both encoders. It adds the target embedding to the memory bank and look for its nearest neighbors in the memory bank. Obviously target embedding itself will be the first nearest neighbor. It shifts the embedding of the input image towards the mean of its nearest neighbors to minimize the summation of those distances. Ideally, we can average the set of nearest neighbors to come up with a single target embedding, but since averaging is dependent on the choice of loss function, we simply minimize the summation of distances. <d-cite key="koohpayegani2021mean"></d-cite>.
</div>


{% include figure.html path="assets/img/2025-04-28-data_curation_for_contrastive_learning/knn_each_epoch_MSF.png" class="img-fluid" %}
<div class="caption"> 
Figure 4: For a random query images, this diagram shows how the nearest neighbors evolve at the learning time. Initially, NNs are not semantically quite related, but are close in low-level features. The accuracy of 1-NN classifier in the initialization is 1.5% which is 15 times larger than random chance (0.1%). This little signal is bootstrapped in our learning method and results in NNs of the late epochs which are mostly semantically related to the query image. <d-cite key="estepa2023all4one"></d-cite>
</div>

MSF suffers from high computation as the objective function needs to be computed for each neighbour (k times). All4One <d-cite key="estepa2023all4one"></d-cite>  contrasts information from multiple
neighbours in a more efficient way by avoiding multiple computations of the objective function.
{% include figure.html path="assets/img/2025-04-28-data_curation_for_contrastive_learning/nnclr_all4one.png" class="img-fluid" %}
<div class="caption"> 
Figure 5: While NNCLR  only contrast the first neighbour, All4One creates representations that contain contextual information from the k NNs and contrast it in a single objective computation. It proposes the use of a new embedding constructed by a self attention mechanism, such as a transformer encoder, that combines the extracted neighbour representations in a single representation containing contextual information about all of them. Hence, we are able to contrast all the neighbours’ information on a single objective computation. It makes use of a Support Set that actively stores the representations computed during the training <d-cite key="dwibedi2021little"></d-cite> so that it can extract the required neighbours. </div>

{% include figure.html path="assets/img/2025-04-28-data_curation_for_contrastive_learning/All4One.png" class="img-fluid" %}
<div class="caption">
Figure 5: Complete architecture of All4One framework. Feature, Centroid and Neighbour contrast objective functions are
indicated by red, purple, and green respectively. All4One uses three different objective functions that contrast different representations: Centroid objective contrasts the contextual information extracted from multiple neighbours while the Neighbour objective assures diversity <d-cite key="dwibedi2021little"></d-cite>. Additionally, the Feature contrast objective measures the correlation of the generated features and increases their independence. <d-cite key="estepa2023all4one"></d-cite>
</div>
Given a sequence of neighbour representations $$ nn^1_i $$ , All4One obtains a single representation $$ c_1 $$ that contains as much information as possible about the input sequence $$ nn^1_i $$ . When computing self-attention <d-cite key="vaswani2017attention"></d-cite>, we mix the representations of the input sequence in a weighted manner so that a new enriched vector of representations is returned. Each element of this enriched vector contains contextual information about all the neighbours in the sequence. During training, for each sequence in $$ NN^1 $$ , the process is made up of the following steps: (i) for each sequence $$ Seq_i $$ in $$ NN^1 $$ , it adds sinusoidal positional encoding <d-cite key="vaswani2017attention"></d-cite>; (ii) then, it feeds the transformer encoder $$ \psi $$ with $$ Seq_i $$ ; (iii) inside the transformer encoder, self-attention is computed and a new sequence is returned $$ Seq^c_i $$ ; (iv) finally, it selects the first representation $$ Seq^c_{i \, 1} $$ in the returned sequence $$ Seq^c_i $$ as our centroid $$ c_i $$ as we aim to contrast a single representation that contains context information from the rest of the neighbours. After selecting the first representation on all sequences, we obtain a batch of representations defined as $$ C^1 $$.
All4One uses three different objective functions that contrast different representations: Centroid objective contrasts the contextual information extracted from multiple neighbours while the Neighbour objective assures diversity <d-cite key="dwibedi2021little"></d-cite>. Additionally, the Feature contrast objective measures the correlation of the generated features and increases their independence.


### 2. Synthetic data generation for positive pairs:

In <d-cite key="wu2023synthetic"></d-cite>, the authors propose a data generation framework with two methods to improve CL training by joint sample generation and contrastive learning. The first approach generates hard samples for the main model. The generator is jointly learned with the main model to dynamically customize hard samples based on the training state of the main model. Besides, a pair of data generators are proposed to generate similar but distinct samples as positive pairs. In joint learning, the hardness of a positive pair is progressively increased by decreasing their similarity.
{% include figure.html path="assets/img/2025-04-28-data_curation_for_contrastive_learning/GAN_based_synthetic_framework_image.png" class="img-fluid" %}
<div class="caption">
Figure 6: (Left) Generation of hard samples and hard positive pairs, and the joint learning of generator and the main contrastive model. We generate two similar but distinct raw samples, and use two views of each raw sample (four views in total) as positives, which are then fed into standard CL frameworks (e.g. SimCLR) for learning. No additional training data is used and no labels are used in the entire training pipeline. (Right) By joint learning, the generated positive pair becomes progressively harder by tracking the training state of the main model. These hard positive pairs help the main model cluster distinct yet similar objects
for better representations. <d-cite key="wu2023synthetic"></d-cite>
</div>
One challenge is that low quality of synthetic samples degrades contrastive learning. Simply using a GAN pre-trained on the same unlabeled dataset as contrastive learning to provide additional synthetic data to the main model cannot effectively improve and even degrades the learned representation. This is because the synthetic data has intrinsically lower quality than the real data <d-cite key="brock2019large"></d-cite> even if labels were available to train a class-conditional generator. When the dataset is unlabeled and the generator is trained in a non class-conditional way, the quality of synthetic data becomes worse (<d-cite key="zhao2020differentiable"></d-cite>; <d-cite key="miyato2018spectral"></d-cite>), which degrades the performance of the main model or only provides marginal benefits. To solve this problem, instead of using the standalone generator and contrastive main model, we jointly optimize them by formulating a Min-Max game such that they compete with each other. As shown in Fig. 6, there are two major components: the hard sample generator (red) and the main contrastive model (blue), which are jointly optimized. Joint learning effectively uses the available unlabeled training data, and no additional training data or labels are used.

<d-cite key="zeng2024contrastive"></d-cite> uses diffusion model to synthetically generate positive pairs. Apart from their high-fidelity image generation ability, diffusion models have also shown excellent visual representation learning ability, even in the absence of labeled information.

{% include figure.html path="assets/img/2025-04-28-data_curation_for_contrastive_learning/diffusion.png" class="img-fluid" %}
<div class="caption">
Figure 7: Overview of the proposed CLSP framework, we use a diffusion model to
generate an additional positive to increase the positive diversity for better representation
learning. (b) The t-SNE plot of features extracted from the intermediate layer of the diffusion model trained on CIFAR10. The features are generated at timestamp 50. (c) The generated images only contain background information if intermediate features are masked, suggesting the decoupling of semantic and background information
in different layers of the diffusion model. (d) Using feature interpolation to generate hard positives, the generated images contain similar semantic content to the anchor image but differ in context and background.. <d-cite key="zeng2024contrastive"></d-cite>
</div>
It uses a diffusion model to generate an additional positive sample $$x^3_i$$ and extract the embedding $$z^3_i$$ using the same encoder and projector employed in SimCLR. Given only an unlabeled anchor image, an unconditional diffusion model can generate diverse images that are semantically similar to the
anchor image. It  leverages feature interpolation on the features from random sampling $$h$$ and features extracted from the diffusion process of the anchor image hanchor to generate new images resembling the anchor image. The interpolation process can be defined as: $$ h = w  * h + (1-w) * h_{anchor}$$


#### Supervised pairing technique:
Supervised contrastive learning (SupCon) <d-cite key="khosla2020supervised"></d-cite> proposes a loss for supervised learning that builds on the contrastive self-supervised literature by leveraging label information. Normalized embeddings from the same class are pulled closer together than embeddings from different classes. Our technical novelty in this work is to
consider many positives per anchor in addition to many negatives (as opposed to self-supervised
contrastive learning which uses only a single positive). These positives are drawn from samples
of the same class as the anchor, rather than being data augmentations of the anchor, as done in
self-supervised learning.
{% include figure.html path="assets/img/2025-04-28-data_curation_for_contrastive_learning/Supcon.png" class="img-fluid" %}
<div class="caption">
Figure 8: Supervised vs. self-supervised contrastive losses: The self-supervised contrastive loss (left, Eq. 1)
contrasts a single positive for each anchor (i.e., an augmented version of the same image) against a set of
negatives consisting of the entire remainder of the batch. The supervised contrastive loss (right), however, contrasts the set of all samples from the same class as positives against the negatives from the remainder of the batch. As demonstrated by the photo of the black and white puppy, taking class label information into account results in an embedding space where elements of the same class are more closely aligned than in the self-supervised case. <d-cite key="khosla2020supervised"></d-cite>
</div>. The enhancements to contrastive learning discussed focus on generalizing the framework to accommodate multiple positives, increasing the contrastive power with more negatives, and enabling implicit hard positive/negative mining. By including all positives in a batch—augmentation-based samples and others with the same label—in the numerator, supervised contrastive learning encourages the encoder to align representations within the same class, resulting in more robust clustering compared to traditional single-positive setups. The summation over negatives is retained, consistent with noise contrastive estimation and N-pair losses, with the addition of more negatives improving the ability to distinguish signal from noise, leading to better  representation learning. Additionally, the loss function's gradient structure inherently emphasizes hard positives and negatives, enabling the model to focus on challenging contrasts without the need for explicit hard mining strategies like triplet loss. These improvements enhance both supervised and self-supervised contrastive learning by making the framework more robust, scalable, and efficient.


In another paper, the authors applies a similar logic for selecting positive pairs in real-time <d-cite key="ghose2023tailoring"></d-cite>. The authors propose a method that allows a robot to learn visual object representations based on a few examples provided by humans. This approach leverages self-supervised learning techniques, such as contrastive learning, to develop object representations that align with human expectations. By incorporating incremental human supervision, the robot can adapt its understanding of objects to meet specific human requirements, enhancing its performance in sorting and recycling tasks. Positive pairs are formed by grouping images of objects that share the same human-provided label. This process enables the robot to associate different visual instances of the same object category, improving its generalization capabilities.

### Attribute-based pairing technique:
Attribute-based pairing entails selecting positive pairs based on criteria defined by end application. For instance, Geography-aware self supervised learning <d-cite key="ayush2021geography"></d-cite> use temporal positive pairs from spatially aligned images over time. 
{% include figure.html path="assets/img/2025-04-28-data_curation_for_contrastive_learning/Geography_aware_framework.png" class="img-fluid" %}
<div class="caption">
Figure 8: Left shows the original MoCo-v2 framework. Right shows the schematic overview of Geography-aware self supervised learning approach. <d-cite key="ayush2021geography"></d-cite>
</div>
Remote sensing data are often geo-located and might provide multiple images of the same location over time. This approach  enhances self-supervised learning models by incorporating geographic information, particularly in the context of remote sensing and geo-tagged datasets. This method leverages spatially aligned images over time to create temporal positive pairs in contrastive learning, effectively capturing changes and patterns in geographic data. By designing tasks that utilize geo-location data, the model learns representations sensitive to geographic context, improving performance in image classification, object detection, and semantic segmentation. This approach is particularly beneficial for remote sensing applications, where labeled data is scarce but unlabeled, geo-tagged data is abundant. It has also been applied to geo-tagged ImageNet images, demonstrating improved performance in various downstream tasks by effectively utilizing geographic information. Incorporating geographic context allows models to learn more meaningful representations, leading to better performance in tasks involving spatial data. Additionally, this approach makes effective use of abundant unlabeled data, reducing the reliance on labeled datasets, which are often expensive and time-consuming to obtain.

Similarly, The paper "Focus on the Positives: Self-Supervised Learning for Biodiversity Monitoring"  <d-cite key="pantazis2021focus"></d-cite>  addresses the challenge of learning effective representations from unlabeled image collections, particularly those captured by static monitoring cameras used in biodiversity studies. 
This study leverages the natural variations present in sequential images from static cameras, utilizing contextual information such as spatial and temporal relationships to identify high-probability positive pairs—images likely depicting the same visual concept. By focusing on these positive pairs during training, the authors demonstrate that the learned representations are more effective for downstream supervised classification tasks, even with limited human supervision. The approach was evaluated across four different camera trap image collections and three families of self-supervised learning methods, consistently outperforming conventional self-supervised training and transfer learning baselines. This methodology offers a promising avenue for enhancing global biodiversity monitoring efforts by improving species classification accuracy with minimal labeled data.

### Cross-modal positive pairing: 
Cross-modal positive pairing involves associating related data points across different sensory modalities, such as linking an image with its corresponding textual description. This technique is fundamental in tasks like image-text retrieval, where the goal is to align visual and textual information effectively. By identifying and leveraging these positive pairs, models can learn shared representations that bridge the gap between modalities, enhancing performance in applications requiring multimodal understanding. For instance, in cross-modal contrastive learning frameworks, positive pairs are used to train models to bring related data points closer in the embedding space while pushing unrelated ones apart, facilitating more accurate retrieval and classification across modalities. This technique is fundamental in models like CLIP and BEiT-3, which aim to align visual and textual information effectively.

*CLIP (Contrastive Language-Image Pre-Training):*

Developed by OpenAI, CLIP is trained on a vast dataset of image-text pairs. It employs a contrastive learning approach where positive pairs (matching images and texts) are brought closer in the embedding space, while negative pairs (non-matching images and texts) are pushed apart. This method enables CLIP to learn a shared representation space for both modalities, facilitating tasks such as zero-shot image classification and cross-modal retrieval. 
ARXIV

*BEiT-3 (Bidirectional Encoder representation from Image Transformers):*

BEiT-3 extends the BEiT framework to handle both vision and vision-language tasks. It introduces a unified masked data modeling objective that operates on monomodal (images or texts) and multimodal (image-text pairs) data. By masking parts of the input and training the model to predict the missing information, BEiT-3 learns representations that capture the relationships between images and texts. This approach leverages cross-modal positive pairing to align visual and textual embeddings within a shared space, enhancing performance across various tasks. 
MICROSOFT

Both CLIP and BEiT-3 utilize cross-modal positive pairing to bridge the gap between visual and textual modalities, enabling more effective integration and understanding of multimodal information.

## Negative Pair Creation Methodologies in Contrastive Learning

### Hard Negative Selection: 
MoCHi, that stands for "(M)ixing (o)f (C)ontrastive (H)ard negat(i)ves". <d-cite key="Hardnegativemixing"></d-cite> creates synthetic hard negatives by combining features of existing hard negatives in the embedding space.  "MoCHi is designed to enhance self-supervised learning by generating challenging negative samples, thereby improving the quality of learned visual representations. In contrastive learning, models are trained to bring similar (positive) samples closer in the embedding space while pushing dissimilar (negative) samples apart. The effectiveness of this approach heavily depends on the selection of negative samples. Hard negatives—samples that are similar to the anchor but belong to different classes—are particularly valuable as they provide more informative gradients, leading to better feature learning. Traditional methods often rely on large batch sizes or extensive memory banks to obtain a diverse set of negatives, which can be computationally expensive. The hard negative mixing strategy addresses this by synthesizing hard negatives through feature-level mixing: for a given anchor, the model identifies existing negatives that are most similar in the embedding space and combines these hard negatives at the feature level to create synthetic negatives that are even closer to the anchor, increasing the difficulty of the contrastive task. This method can be implemented on-the-fly with minimal computational overhead, making it efficient and scalable. By focusing on harder negatives, the model learns more discriminative features, leading to improved performance in downstream tasks such as classification and object detection. Synthesizing hard negatives reduces the need for large batch sizes or memory banks, lowering computational requirements. 

Uncertainty and Representativeness Mixing (UnReMix) for contrastive training <d-cite key="unremix"></d-cite>, introducs a method designed to enhance contrastive learning by selecting hard negative samples based on three key factors: anchor similarity, model uncertainty and representativeness ensuring that negative samples are similar to the anchor point, making them challenging for the model to distinguish. UnReMix utilizes uncertainty to penalize false hard negatives and pairwise distance among negatives
to select representative examples.
{% include figure.html path="assets/img/2025-04-28-data_curation_for_contrastive_learning/Unremix.png" class="img-fluid" %}
<div class="caption">
Figure 9: Overview of the proposed hard negative sampling technique, UnReMix. Given an anchor
$$𝑥_𝑖$$ and a set of negative samples $$X_𝑁 \ 𝑥_𝑖$$ , UnReMix computes an importance score for each negative
sample, by linearly interpolating between gradient-based uncertainty, anchor similarity and representativeness
indicators, capturing desirable negative sample properties, i.e., samples that are truly
negative (P1), in close vicinity to the anchor (P2) and representative of the sample population (P3). <d-cite key="unremix"></d-cite>
</div>
Anchor Similarity: This property assesses how closely a potential negative sample resembles the anchor (the reference data point). Incorporating anchor similarity ensures that selected negative samples are challenging, as they are similar to the anchor yet belong to different classes, thereby providing informative gradients for the model.

Model Uncertainty: Model uncertainty evaluates the confidence level of the model's predictions for a given sample. By selecting negative samples where the model exhibits higher uncertainty, the training process focuses on areas where the model is less certain, promoting learning in challenging regions of the data space.

Representativeness: Representativeness measures how well a sample reflects the overall data distribution. Incorporating this property ensures that the selected negative samples are not outliers but are indicative of the broader dataset, leading to more generalized and robust representations.


### Removal of False Negatives:
<d-cite key="huynh2022boosting"></d-cite>  introduces methods to identify these false negatives and propose two strategies to mitigate their impact: elimination and attraction. Contrasting false negatives
induces two critical issues in representation learning: discarding semantic information and slow convergence. This paper proposes novel approaches to identify false negatives, as well as two strategies to mitigate their effect, i.e. false negative elimination and attraction, while systematically
performing rigorous evaluations to study this problem in detail. False Negative Elimination identifies potential false negatives and excludes them from the negative sample set, preventing the model from learning misleading distinctions. In False Negative Attraction, instead of excluding false negatives, this strategy reclassifies them as positives, encouraging the model to learn representations that acknowledge their semantic similarity.
{% include figure.html path="assets/img/2025-04-28-data_curation_for_contrastive_learning/boosting_fn.png" class="img-fluid" %}
<div class="caption">
Figure 10: False negatives in contrastive learning. Without knowledge of labels, automatically selected negative pairs could actually belong to the same semantic category, creating false negatives. <d-cite key="huynh2022boosting"></d-cite>
</div>
False negatives hinder the convergence of contrastive learning-based objectives due to the appearance
of contradicting objectives. For instance, in Figure 10, the dog’s head on the left is attracted to its fur (positive pair), but repelled from similar fur of another dog image on the right (negative pair), creating contradicting objectives.

{% include figure.html path="assets/img/2025-04-28-data_curation_for_contrastive_learning/boosting_fn_2.png" class="img-fluid" %}
<div class="caption">
Figure 11: Overview of the proposed framework. Left: Original definition of the anchor, positive, and negative samples in contrastive learning. Middle: Identification of false negatives (blue). Right: false negative cancellation strategies, i.e. elimination and attraction. <d-cite key="huynh2022boosting"></d-cite>
</div>

*False Negative Cancellation:*
Negative pairs are samples from different images that may or may not possess similar semantic content or visual
features. Consequently, it is possible that some samples k have the same semantic content as the anchor i, and are thus false negatives. As discussed earlier, false negatives give rise to two critical problems in contrastive learning: they discard semantic information and slow convergence

*False Negative Attraction:*
While eliminating false negatives alleviates the undesireable effects of contrasting against them, it ignores information available in what are actually true positives. Minimizing the original contrastive loss (Eqn. 1) only seeks to attract an anchor to different views of the same image. Including true positives drawn from different images would increase the diversity of the training data and, in turn, has the potential to improve the quality of the learned embeddings.
Indeed, Khosla et al. <d-cite key="khosla2020supervised"></d-cite>  show that supervised contrastive
learning (i.e., where an anchor is attracted to samples having the same semantic label) can be more effective than the traditional supervised cross-entropy loss. Thus, we propose to treat the false negatives that have been identified as true positives and attract the anchor to this set.

False negatives are samples from different images with the same semantic content, therefore they should hold certain similarity (e.g., dog features). A false negative may not be as similar to the anchor as it is to other augmentations of the same image, as each augmentation only holds a specific view of the object.


### Synthetic Hard Negative Samples for Contrastive Learning:###
<d-cite key="dong2024synthetic"></d-cite> proposes a method to enhance contrastive learning by generating synthetic hard negative samples. This approach involves mixing existing negative samples in the feature space to create more challenging negatives, encouraging the model to learn more discriminative representations. To address the issue of false negatives—samples incorrectly labeled as negative but semantically similar to the anchor, the paper incorporates a debiasing mechanism, ensuring the model focuses on truly dissimilar negative samples.
 {% include figure.html path="assets/img/2025-04-28-data_curation_for_contrastive_learning/synthetic.png" class="img-fluid" %}
<div class="caption">
Figure 11: The architecture of SSCL. Given a batch of N images, we first apply data augmentation and then encode and projection head to obtain two batch feature vectors (2N). For one image, we generate new harder negative samples by synthesizing negative samples from the 2N-2 remaining images. The red dashed arrow indicates the projection of the feature vector.We then combine the original samples with the newly synthesized ones to compute the contrastive loss. <d-cite key="dong2024synthetic"></d-cite>
</div>
The process involves the following steps:

Identifying Hard Negatives: For a given anchor sample, the method selects the hardest negative samples from the available negative set based on their similarity to the anchor. These hard negatives are the ones most similar to the anchor but belong to different classes.

Synthesizing Harder Negatives: The selected hard negatives are then combined through linear interpolation to create synthetic negative samples that are even closer to the anchor in the feature space. This interpolation is controlled by a parameter that adjusts the degree of mixing, allowing the generation of negatives with varying levels of hardness.

Sampling Synthetic Negatives: The synthetic hard negatives are sampled by controlling the contrast between the anchor and the negative samples. This ensures that the generated negatives are challenging enough to improve the model's discriminative ability.




# Discussion:

## Drawback of generating positive pairs from same instance:
The random augmentations, such as random crops or color changes, can not provide positive pairs for different viewpoints, deformations of the same object, or even for other similar instances within
a semantic class. The onus of generalization lies heavily on the data augmentation pipeline, which cannot cover all the variances in a given class.

## Trade-off across various techniques to generate positive pair from multiple instances:

## Trade-off across various techniques to generate positive pair from multiple instances:

# Comparative Result:

# Open Research Questions: