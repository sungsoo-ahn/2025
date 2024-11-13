Contrastive learning aims to learn a robust representation of the underlying dataset that can be useful for a variety of downstream tasks. It follows a simple yet powerful technique of creating positive and negative pairs of data points(images, text, speech or multimodal pairs) to learn the representation. The design of positive and negative pairs is crucial in contrastive learning, as it directly impacts the quality of learned representations. Hence, a considerable amount of research has been conducted in this direction to create these pairs. The aim of this blog post is provide a unique  perspective on different high level methodlogies to create such pairs and dive deeper into the techniques under each category.

Taxonomy and overview of methodologies to create positive and negative pairs for contrastive learning:

- Positive pair creation:
    1. Same instance positive
        - Augmentation: Create a positive pair using 2 instances of the same data point(SimCLR, MoCo)
    2. Multiple instances positives
        - K nearest neighbors using embedding based similarity as positives(NNCLR, MSF, ALL4One)
        - Synthetic data generation to generate two data points of distinct identities but similar objects for positive pairs(GAN, Diffusion model)
        - Leveraging labeled data to create positive pairs(SupCon, Generalized SupCon)
        - Attribute based: Spatial location(Geographic aware), Time domain(Wildlife paper)
    3. Positive pairs across multiple modalities
        - VLM, Speech, Text, etc 
    4. Human feedback based
- Negative pair creation:
    1. Removal of False Negatives:
        - False Negative Cancellation and attraction
    2. Hard Negaive Selection: UnReMix(Uncertainty and Representativeness Mixing), CONTRASTIVE LEARNING WITH HARD NEGATIVE SAMPLES, Hard Negative Mixing for Contrastive Learning
    3.  Negative Sampling Correction
    4.  Synthetic negative pairs: Synthetic Hard Negative Samples for Contrastive Learning

Same instance positive
- Augmentation: Create a positive pair using 2 instances of the same data point(SimCLR, MoCo)
This method takes the same data point(image, text) and creates a positive pair by applying 2 different augmentstions to the same instance. Some of the augmentations that can be applied are: Random cropping, masking, Mixup, Cutmix, Copy-past, color distortion, contrast stretching, histogram equalization, etc. Let's say we have a batch of N data points. We generate 2N pairs by applying augmentation to each of the N data points. The NCE pulls the positive pair together and pushes the negative pairs away.

One of the biggest drawbacks of using single instance as positive are as follows:
- Possibility of missing out on good true positives for 2 semantically similar but different  instances. For instance, 2 different data points which have the same label/category will be pushed apart if single instance is used to create positive pairs for contrastive learning
- False negatives: Related to the point above, 2 data points might be semantically similar but will be treated as negatives for contrastive learning, thus introducing false negatives

To alleviate the above issues, it makes sense to use multiple instances for creating positive and negative pairs. 

Multiple instances positive pair creation:
1. K nearest neighbors uses embedding based similarity as positives(NNCLR, MSF, ALL4One):
One of the ways to create positive pair is to find semantically similar samples/data points in embedding space. NNCLR extracts top-1 neighbor to find the semantically similar sample. in the NNCLR methodology, we compute the emebddings of all the samples in the dataset and store them in an embedding bank.  During training, similar to SimLCR, NNCLR generates 2 random augmentations for each  data point  in a batch of N data points, passes it through the feature extractor and projection layer. Once we obtain representation for each of the augmented sample, NNCLR retrieves top-1 neighbor of one of the augmented sample from the embedding bank and uses the retrieved sample along with the other augmented instance as a positive pair. The benefit of doing this is that we can learn represenations using more diverse samples compared to single instance one
A slight modification to this method is to retrieve top-K neightbors(MSF) intead of judt top-1 and use it to contrast the positive pairs. However, this would require us to compute the loss for each of the top-k retrieved pair which might be computationally inefficient. All4One proposes an improvement over this by generating a single unified representation of top-k neighbors by using a self-attention based transformer encoder. This makes it computationally more efficient as we need to compute the the loss only once 

2. Synthetic data generation to generate two data points of distinct identities but similar objects for positive pairs(GAN, Diffusion model)
Instead of retrieving semnatically similar samples in embedding space, some other methods propose synthetically generating positive sample pairs.
GAN based method uses joint training of a GAN network along with contrastive objective to incrementally generate "difficult" positives for the contrastive learning framework.
The input to the GAN is a class label prior. GAN generates a positive pair of images this class label as prior. This method uses min max objective to train the models jointly. GAn incrementally generates hard images.
One drawback of the GAN based method is that it optimizes 2 models together which can lead to issues with model convergence. Diffusion based method propses a way to address this by training 2 models separately.
A well-trained unconditional diffusion models can capture semantic information at
their intermediate layers [4, 49]. During a random reversed diffusion process, we
can replace the features of the intermediate layers with the semantic features
extracted from an anchor image. This results in the generation of images possessing
similar semantic content to the anchor image but differing in background
and context due to the randomness of features in other layers. These
synthetic samples can be considered as “hard” positives of the anchor image because the semantic
similarity could be subtle and extremely distorted. 

3. Attribute based: Spatial location(Geographic aware), Time domain(Wildlife paper)
Sometimes, it might be useful to account for context based on the  end application and incorporate that context to generate the positive pairs. For instance, Geographic aware SSL paper used spatially aligned temporal pairs to learn represenations from unlabled remote sensing images. A Spalially aligned pair of image belongs to the same geographical location on Earth(same Lat,lon coordinates) but is catpured at two different instances in time by the satllite  constellation. It also adds an additional geo-classification loss in addition to the contrastive loss. It uses MoCOv2 framework to learn the representation. 
In a similar vien, [Wildlife paper] proposes a few different methodlogies for creating positive pairs. It propsed to use 2 images from a sequence which are temporally close and embedding similarity based approach. The paper focuses on ecological and wildlife conservation use case wherein they query the data collected using camera traps(wild cams/trail cams). 

4. Positive pairs across multiple modalities
One of the most prominent use case for this is

5. Leveraging labeled data to create positive pairs(SupCon, Generalized SupCon)
- The self-supervised contrastive loss (left, Eq. 1)
contrasts a single positive for each anchor (i.e., an augmented version of the same image) against a set of negatives consisting of the entire remainder of the batch. The supervised contrastive loss (right) considered in this paper (Eq. 2), however, contrasts the set of all samples from the same class as positives against the negatives from the remainder of the batch. As demonstrated by the photo of the black and white puppy, taking class label information into account results in an embedding space where elements of the same class are more closely aligned than in the self-supervised case.
Normalized embeddings from the same class are pulled closer together than embeddings from different classes. Our technical novelty in this work is to consider many positives per anchor in addition to many negatives (as opposed to self-supervised contrastive learning which uses only a single positive). These positives are drawn from samples of the same class as the anchor, rather than being data augmentations of the anchor, as done in self-supervised learning.


Multiple instances negative pair creation:




