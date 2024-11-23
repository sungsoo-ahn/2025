---
layout: distill
title: Active Learning and the Various Flavors of Supervision for Object Detection
description: We reflect on the role of active learning in object detection when combined with other sources of supervision to minimize manual annotation effort. In doing so, we highlight the need to harmonize the approaches so that they can develop their full potential. This can, for example, include an estimate of how many objects our detector might have missed – a quantity most active learning approaches for object detection have so far ignored.
date: 2025-04-28
future: true
htmlwidgets: true
hidden: false

# Anonymize when submitting
authors:
  - name: Anonymous

# must be the exact same name as your blogpost
bibliography: 2025-04-28-active-learning-supervision-object-detection.bib  

# Add a table of contents to your post.
#   - make sure that TOC names match the actual section names
#     for hyperlinks within the post to work correctly. 
#   - please use this format rather than manually creating a markdown table of contents.
toc:
  - name: Supervision for Object Detection
    subsections:
    - name: Failure Modes of Object Detectors
    - name: The Many Flavors of Supervision
  - name: Active Learning Approaches for Object Detection
    subsections:
    - name: Aggregation of Object-Level Scores
    - name: Uncertainty
    - name: Expected Model Change
    - name: Diversity
    - name: Undetectability
    - name: Combining Multiple Criteria
  - name: Object Detection With Minimal Manual Annotation Effort
  - name: Conclusion
---

{% raw %}
<div style="display: none;">
$$\require{mathtools}
\newcommand{\MidSymbol}[1][]{\:#1\:}
\newcommand{\given}{\MidSymbol[\vert]}
$$
</div>
{% endraw %}

The availability of enough annotated data for the model to learn a task has always been a limitation in (deep) machine learning. A large set of full annotations, which present the desired solution, provides a strong supervisory signal to tune the learnable parameters of a model to perform a task well. A lot of research has been done to relax this limitation. There are many approaches to use data more efficiently and provide benefit to models from cheaper types of supervision. This includes the re-use of data with existing annotations <d-cite key="pan2010,zhuang2021"></d-cite>, pre-training with synthetic data <d-cite key="nikolenko2021"></d-cite>, self-training of a model <d-cite key="rosenberg2005,zoph2020"></d-cite>, enforcement of assumptions such as smoothness in input space <d-cite key="sajjadi2016"></d-cite>, usage of weak labels that only give hints to a model <d-cite key="oquab2015,bearman2016,bilen2016"></d-cite>, and learning of useful representations from unlabeled data through auxiliary tasks <d-cite key="doersch2015,zhang2016"></d-cite>. These approaches can help reduce the manual annotation effort. But, in the vast majority of cases, at least a small amount of fully-supervised samples from the problem domain is either necessary to fine-tune the model on that task or can lead to an improvement of the model's performance.

In active learning <d-cite key="settles2009"></d-cite>, a dataset of particularly informative samples is curated in cooperation with a model that learns the actual task. Active learning methods aim to find these examples so that only a fraction of the dataset needs to be labeled by an oracle (e.g. a human annotator), while achieving a model performance similar to learning with the fully annotated dataset.

In this blog post, we explore how cheaper supervision and active learning can be combined to learn the computer vision task of object detection without otherwise large annotation cost. Along the way, we briefly study the peculiarities of object detection and how we can provide cheap supervision to models learning this task. Then, we review common approaches to active learning for object detection and discuss what makes a sample informative. Equipped with this knowledge, we examine object detection settings that effectively reduce annotation cost by combining different types of supervision and active learning.

To convey our most important conclusion early on: In a good interplay with cheap supervision, active learning should select samples that effectively compensate for the weaknesses of the supervision method. A good example is set by Nakamura et al. <d-cite key="nakamura2024"></d-cite> by incorporating a measure of missed detections in their active learning sample selection after their unsupervised domain adaptation approach alone failed to detect many objects in their target data.

## Supervision for Object Detection

Object detection combines localization and classification of objects of interest. Thereby, it differs from other widely studied computer vision tasks like image classification and semantic segmentation. These are both pure classification tasks. They assign a class label to the whole image or, respectively, a class label to every single pixel in the image. In object detection, only regions of the image are classified that contain relevant objects.<d-footnote>One-stage object detectors based on convolutional neural networks (e.g. OverFeat <d-cite key="sermanet2013"></d-cite>, YOLO <d-cite key="redmon2016"></d-cite>, SSD <d-cite key="liu2016"></d-cite>, RetinaNet <d-cite key="lin2017"></d-cite>, FCOS <d-cite key="tian2019"></d-cite>) actually predict class labels and bounding box coordinates for every pixel of their feature map(s). They resemble segmentation models in that regard. Many locations are classified as "background" or "no object" class though and are ignored during the final output of predictions.</d-footnote> To do so, the object detector predicts bounding box coordinates and the class of the object enclosed by that box. Unlike the pure classification tasks, a wrong class prediction is not the only way of failing in object detection.

### Failure Modes of Object Detectors

{% include figure.html path="assets/img/2025-04-28-active-learning-supervision-object-detection/failure_modes.png" class="img-fluid rounded" %}
<div class="caption">
    Figure 1: Demonstration of failure modes of an object detector trained to recognize three types of stuffed animals (cow 🐮, octopus 🐙, panda 🐼).
</div>

It can be regarded as a binary decision whether a location contains an object or not. When an object is present at a location and the detector predicts this region as "background", the detector makes a <mark style="background-color: #facc15">false negative</mark> error; an object that should have been detected is not. In the opposite case, when the detector predicts an object at a location without an actual object of interest, the model makes a <mark style="background-color: #dc2626">false positive</mark> error. Another way to fail making a correct prediction is to assign the <mark style="background-color: #2dd4bf">wrong class</mark> label to an otherwise accurately localized object of interest. The fourth and final failure mode of object detectors is <mark style="background-color: #a855f7">bad localization</mark>. While the object has been correctly assigned to its corresponding class, the predicted bounding box is not tightly centered around the object. Fig. 1 illustrates all of these possibles mistakes.

Wrong class labels and bad localization are counted as false positive errors when calculating the precision and recall of the detector, which are the foundation of mean average precision (mAP) and mean average recall (mAR), the most reported object detection metrics. However, as these mistakes might have different reasons, they might be reduced most easily by different types of supervision, which includes different sets of fully-supervised samples.

### The Many Flavors of Supervision

{% include figure.html path="assets/img/2025-04-28-active-learning-supervision-object-detection/full_supervision.png" class="img-fluid rounded" %}
<div class="caption">
    Figure 2: A sample with full supervision for the object detection task of recognizing different types of stuffed animals.
</div>

Object detection models receive full supervision when they can learn from images and corresponding sets of bounding box coordinates and class labels (see Fig. 2). Ideally, all objects of interest in an image are annotated. Otherwise, regions containing missing objects are learned as "background". Learning from annotated samples that supply the expected output directly is the most common learning setting. Providing such a strong supervisory signal is costly and the annotation effort can vary largely between different samples. During labeling for basic image classification, an annotator assigns a single class label to the entire image. The number of annotations per image in object detection depends on the number of objects present in a scene and can thus vary between zero and many. In experiments by Su et al. <d-cite key="su2012"></d-cite>, the median time to draw a single bounding box on ImageNet <d-cite key="deng2009"></d-cite> images was 25.5&nbsp;s. This includes the loading time of the image, but not the assignment of a class label. In use cases like object detection on satellite images, traffic sign detection, or pedestrian detection, many objects can be present in an image at the same time. Creating complete annotations for only one such image takes up a significant amount of time already, especially when objects are small and exact localization is arguably even more important.

We now go over different approaches that have been developed to learn object detection from supervisory signals that are less costly to provide.

#### Active Learning

Ultimately, the detector must know about the object classes it has to predict and the visual features of the corresponding objects. Active learning <d-cite key="settles2009"></d-cite> helps us focus our annotation effort on those samples that are most informative to our model. The information content of an example is always relative to the observer. For example, a model that has a specific inductive bias built-in might find the information a particular sample provides trivial, while another model that has to learn this issue from data might find the sample highly informative. Active learning involves the model in choosing which samples to annotate. To make this cooperative approach promising, the model already needs to have a basic grasp of the task, e.g. have been trained on a small initial dataset with full supervision. As visualized in Fig. 3, the model then makes predictions for all unlabeled samples and based on an acquisition function, which can incorporate criteria like uncertainty or diversity, the current informativeness of every sample is ranked. The samples deemed most informative are then given to an oracle, which is a human annotator in most cases.<d-footnote>In object detection, it can be useful to also display the model's predictions to the annotator as this may reduce the annotation cost if some predictions only need verification or small adjustments.</d-footnote> The model is then trained again with the increased pool of labeled examples. This procedure is repeated over several rounds until the annotation budget runs out.

{% include figure.html path="assets/img/2025-04-28-active-learning-supervision-object-detection/active_learning.png" class="img-fluid rounded" %}
<div class="caption">
    Figure 3: The active learning loop includes a labeled and an unlabeled pool of samples. The informativeness of unlabeled samples can be derived from the predictions of the trained model. In line with the annotation budget, a set of highly informative samples receives annotations from an oracle and is transferred to the labeled sample pool. The model gets re-trained after the update and the cycle starts again.
</div>

A well-designed active learning approach limits the number of redundant samples to be annotated and can help achieving a high model performance with a fraction of data. Note, however, that a labeled dataset curated in cooperation with a particular model is not necessarily the most informative dataset for a different model <d-cite key="settles2009"></d-cite>.<d-footnote>This limitation can be relaxed through knowledge distillation <d-cite key="hinton2015"></d-cite> approaches such as student-teacher training <d-cite key="romero2015"></d-cite>, where a well-performing model provides training targets for another model that is still learning.</d-footnote>

#### Transfer Learning and Synthetic Data

Transfer learning <d-cite key="pan2010,zhuang2021"></d-cite> is the most common approach to achieve a reasonable model performance with only a small amount of labeled samples in our target domain. Taking advantage of the public availability of large-scale datasets with annotations for various applications, the model is (pre-)trained with a large amount of labeled data on a related "source" task. This can provide a good starting point for the parameters of the model and thereby reduce the required number of labeled samples in the target domain and accelerate training convergence. A ubiquitous approach in computer vision is to initialize a model's feature extraction part, which is often called "backbone", with parameters trained for image classification on huge image databases like ImageNet <d-cite key="deng2009"></d-cite>. In object detection, the COCO <d-cite key="lin2014"></d-cite> dataset is often used to pre-train the whole model (as opposed to only the backbone).

Transfer learning approaches can be categorized as homogeneous and heterogeneous transfer learning <d-cite key="pan2010,zhuang2021"></d-cite>. In the easier, homogeneous case, the feature and label spaces of source and target domain are consistent, while the distributions between the domains might differ. In heterogeneous transfer learning, one or both spaces are not equal. This could mean that we want to detect different object classes in the target domain than learned in the source domain or both domains could exhibit visually distinct features. Depending on the setting, different methods like domain adaptation, which tries to reduce the gap between source and target domain, can be used.

A way to reduce the gap between source and target task at an early stage is the use of high-quality synthetic data <d-cite key="nikolenko2021"></d-cite> specifically for our problem domain as source data. As we govern the data generation process, we get the ground truth labels along with our synthetic samples. We have control over the amount of generated data and other aspects like the class balance. Examples for such an approach use video game engines to generate data for vehicle detection <d-cite key="johnsonRoberson2017"></d-cite> or pedestrian detection <d-cite key="vazquez2014"></d-cite> and tracking <d-cite key="fabbri2021"></d-cite>. Another example combines natural images from the COCO <d-cite key="lin2014"></d-cite> dataset and templates of a country's traffic sign catalogue to generate data for traffic sign detection <d-cite key="tabelini2022"></d-cite>. The need for advanced transfer learning methods to align the features between the domains depends on how closely the synthetic source data imitates the real target data, but training on synthetic data is a promising approach to save annotation cost.

#### Semi-Supervised Learning

Semi-supervised learning uses labeled and unlabeled data to improve the performance of a model beyond the training with only the labeled data. This can only work if the unlabeled data is related to the labeled data. More formally put, it is a necessary condition that the marginal distribution $$ P(X) $$ of the data in input space $$ \mathcal{X} $$ (with $$ X \in \mathcal{X} $$) comprises information about the posterior distribution $$ P(Y \given X) $$, i.e. the probability that a data point $$ x \in X $$ belongs to class $$ y \in Y $$. After training the model fully-supervised with the labeled data, some assumptions are made to propagate labels to unlabeled data. The three most common assumptions regard smoothness in input space, decision boundaries of a classifier in regions of low data density, and the concentration of related data points along lower-dimensional substructures of the high-dimensional input space, known as manifolds <d-cite key="vanEngelen2020"></d-cite>.

{% include figure.html path="assets/img/2025-04-28-active-learning-supervision-object-detection/consistency_regularization.png" class="img-fluid rounded" %}
<div class="caption">
    Figure 4: Consistency regularization encourages same predictions for similar samples. This can be trained even on unlabeled samples through augmented versions of the image.
</div>

The two dominant semi-supervised approaches are consistency regularization and self-training based on pseudo-labels. The smoothness assumption states that similar data points should receive the same prediction. Consistency regularization approaches for object detection like CSD <d-cite key="jeong2019"></d-cite> implement this through a consistency loss that enforces same predictions for differently augmented versions of a sample, as shown in Fig. 4.<d-footnote>Fully-supervised training with data augmentation makes the same assumption, but does not need a consistency loss because the ground truth label is provided.</d-footnote> For unlabeled samples only the consistency loss can be used, while for labeled samples a supervised loss term can be used in addition. Self-training <d-cite key="zoph2020"></d-cite>, on the other hand, uses high-confidence predictions on unlabeled samples as if they were actual annotations, provided by a competent oracle (see Fig. 5). There are many elaborate approaches to improve the quality of these so-called pseudo-labels.

{% include figure.html path="assets/img/2025-04-28-active-learning-supervision-object-detection/self_training.png" class="img-fluid rounded" %}
<div class="caption">
    Figure 5: Self-training trains a model on a few labeled samples and gradually increases the training set size through the incorporation of high-confidence prediction on unlabeled data as pseudo-labels. In the simplest case, the training loop looks like this.
</div>

#### Weakly-Supervised Learning

Instead of full annotations with tight bounding boxes and uniquely assigned class labels, weakly-supervised learning enables model training from coarse labels, which act more like hints to the model. As shown in Fig. 6, such hints could be image-level class labels <d-cite key="oquab2015,bilen2016"></d-cite>, which only hold information regarding present classes, but not where corresponding objects are located.

{% include figure.html path="assets/img/2025-04-28-active-learning-supervision-object-detection/weak_supervision.png" class="img-fluid rounded" %}
<div class="caption">
    Figure 6: Cheap annotations for weakly-supervised learning: image-level class labels (left), point annotations (right).
</div>

Point annotations <d-cite key="bearman2016,chen2021"></d-cite> contain more information, but are still cheaper to create than exact bounding boxes. They just mark a point inside every object and assign a class to that point. For the PASCAL VOC 2012 <d-cite key="everingham2010"></d-cite> dataset, Bearman et al. <d-cite key="bearman2016"></d-cite> report a median of 2.4&nbsp;s to click anywhere on the first instance and 0.9&nbsp;s for additional instances of an object class. Papadopoulos et al. <d-cite key="papadopoulos2017a"></d-cite> determine a mean time of 1.87&nbsp;s to click approximately on the center of an object, which they find to be superior to a point anywhere on the object in their experiments. They use the PASCAL VOC 2007 <d-cite key="everingham2010"></d-cite> dataset and annotator training before the actual labeling.

Other types of weak supervision that have been proposed for object detection include the verification of predicted bounding boxes <d-cite key="papadopoulos2016"></d-cite> and point annotations of the extreme points (top, bottom, left- and right most points) of an object <d-cite key="papadopoulos2017b"></d-cite>.

Typically, a large number of weakly-labeled samples is necessary to train a well-performing model.

#### Self-Supervised Learning

A final type of supervision, before we wrap this section up, aims to learn useful representations through training on auxiliary tasks. Self-supervised learning approaches construct such tasks that use inherent properties of the data like color <d-cite key="zhang2016"></d-cite> (shown in Fig. 7), spatial <d-cite key="doersch2015"></d-cite> and temporal coherence <d-cite key="lee2017"></d-cite>, or invariance to augmentations <d-cite key="chen2020"></d-cite>. Similar to working with synthetic data, the solution to the task just comes with the data. Except that we have to design a suitable task around our real data this time. A well-designed self-supervised learning task encourages the model to learn representations that generalize to the intended downstream task, which is object detection in our case. Note, however, that, during self-supervised pre-training, the model does not know about the objects we care about later on. To make actual predictions for these objects, we must subsequently fine-tune the model with some labeled data.

{% include figure.html path="assets/img/2025-04-28-active-learning-supervision-object-detection/self_supervised.png" class="img-fluid rounded" %}
<div class="caption">
    Figure 7: Colorization as an exemplary auxiliary task from self-supervised learning to learn useful representations.
</div>

Self-supervised learning with a huge amount of unlabeled data has become the default strategy to train large, often transformer-based, foundation models in recent years <d-cite key="caron2021,oquab2024"></d-cite>. These models aim to learn general representations that are useful for a wide variety of computer vision tasks through training with diverse image data. We can then take the so learned feature extractor, freeze it and fine-tune a prediction head on top of it with our own labeled data. However, self-supervised learning does not have to be applied at such scale to learn a useful set of parameters that helps reducing the annotation cost.

Currently, there are relatively few approaches for self-supervised learning of whole detection models <d-cite key="wang2023a"></d-cite>. Many methods focus on feature extractors that can be used as model backbones.

#### Complementary Combinations to Reduce Annotation Cost

In practice, the boundaries between these different types of supervision are not as sharp as presented here. Approaches from one area often borrow methods and ideas from other areas. Several of these methods are complementary to each other. But, while all of them can help reducing the annotation cost, the model still needs some examples with strong enough supervision to gather knowledge about all object classes. Thus, the problem of choosing samples to label persists if we want to use our manual annotation efforts as deliberate as possible.

We now investigate how active learning is applied to object detection. This knowledge of how we choose samples that are deemed most informative helps us later making informed decisions about pairings between cheap supervision and active learning strategies.

## Active Learning Approaches for Object Detection

In this section, we explore common approaches to active learning for object detection. This is not an exhaustive survey of the current state of the art, but rather an introduction to the general thinking behind the choice of the currently most informative samples.<d-footnote>Remember that informativeness is relative to the observer.</d-footnote> Along the way, we highlight some of the difficulties and shortcomings of these methods. It is, for example, quite common to report the evaluation results of a method on public datasets like PASCAL VOC <d-cite key="everingham2010"></d-cite> or COCO <d-cite key="lin2014"></d-cite>. Haussmann et al. <d-cite key="haussmann2020"></d-cite> criticize this practice. They argue that these datasets have been curated to learn object detection in a fully-supervised fashion. Hence, most samples are informative. Besides, this leads to experiments in relatively small-scale settings between hundreds and a few thousand samples. They evaluate their approach on a large-scale internal dataset of traffic scenes that includes many uninformative samples and demonstrate that their approach is beneficial even in a setting that starts with 100k initially labeled samples and adds 200k samples per active learning round three times.

Just a quick heads-up: There are different active learning scenarios. Stream-based active learning picks the most informative samples from a stream of data and sends them to an oracle. Membership query synthesis generates synthetic data that optimally supports the model's learning, which is useful when the target dataset is small. Here, we consider pool-based active learning as a setting where the data collection is already finished and the active learning procedure helps us annotate the least amount of our data while achieving a high-performing model. We also focus on the batch sampling scenario to update the pool of labeled data with multiple samples at once before re-training the model, opposed to selecting informative samples one by one and re-training after each.

While we are on the subject of re-training: We can perform active learning continuously or from scratch after each update of the pool of labeled samples. Training from scratch should give a less biased model. When fine-tuning the model continuously after each pool update, the model sees older samples significantly more often than newer ones.<d-footnote>In addition, fine-tuning could be susceptible to primacy bias <d-cite key="ash2020b,nikishin2022,dohare2024"></d-cite>, a phenomenon where a neural network losses plasticity after learning. Among other things, this seems to be due to the model weights not being a good starting point to learn about a sample that deviates from the already encountered ones.</d-footnote> However, Schmidt et al. <d-cite key="schmidt2020"></d-cite> find that the continuous training strategy outperforms training from scratch in their experiments. Given a sufficiently diverse initial training set, continuous training should be reasonable and less time consuming.

We start our overview of common active learning approaches for object detection by looking at how the aggregation of object-level scores influences the selection of images.

### Aggregation of Object-Level Scores

In most cases, object detectors make more than one detection per image. Many measures of informativeness are calculated on the level of a single object instance. Although there are active learning approaches for object detection that operate on this object-level <d-cite key="wang2018,desai2020,lyu2023"></d-cite> or the level of an object region <d-cite key="laielli2021,liou2023"></d-cite>, which are promising for using the annotation effort even more deliberately, we don't look further into these here. We want to avoid getting lost in too many specific approaches, as this could impair comprehensibility. Instead, we follow the majority of published approaches and expect a score on the image-level to select whole images for annotation. That means, we need to aggregate the scores of multiple objects. Brust et al. <d-cite key="brust2018"></d-cite> present three straightforward aggregation strategies:

1. **A sum of the object-level scores.** This biases the sample selection towards images with many detections, especially if some of them have a high informativeness score. In this setting, a sample with five moderately informative detections might be preferred over a sample with a single highly informative detection.<d-footnote>Apart from this possibly undesired behavior, a sample with more objects is more costly to label.</d-footnote>  We could also compute a weighted sum by assigning a second quantity (e.g. highest class probability of each detection) as a weight to the informativeness scores. Kao et al. <d-cite key="kao2019"></d-cite> provide an example of this in the context of object detection.

2. **An average of all object-level scores.** This limits the influence of the number of detections and can, again, be weighted or unweighted.

3. **Extreme values (minimum or maximum) of the object-level scores.** A lot of information is discarded like this, but it may prove beneficial for some scenarios.

Some approaches to score the informativeness of a sample are based on combining the predictions from multiple models or multiple forward passes of the same model. For these approaches, we must first match the detected objects over those multiple predictions. Each prediction might localize an object a little different. Therefore, we have to associate those predictions that regard the same object to each other. Association is typically done through the calculation of intersection over union (IoU), which is also used for segmentation and visual tracking tasks. The IoU of two sets $$ A $$ and $$ B $$ is defined as

$$ J(A,B) = \mathrm{IoU}(A, B) = \frac{\lvert A \cap B \rvert}{\lvert A \cup B \rvert} = \frac{\lvert A \cap B \rvert}{\lvert A \rvert + \lvert B \rvert - \lvert A \cap B \rvert} , $$

which is the ratio of intersection size and union size, also known as Jaccard index, with $$ 0 \leq \mathrm{IoU}(A, B) \leq 1 $$. An IoU value close to 1 expresses a strong overlap between the two sets. Fig. 8 illustrates the idea. With this statistic, we can associate detections so that those with a high IoU are connected. Then, we can compute the informativeness scores at the object-level before we aggregate them for an image-level score.

{% include figure.html path="assets/img/2025-04-28-active-learning-supervision-object-detection/iou.png" class="img-fluid rounded" %}
<div class="caption">
    Figure 8: Visualization of the intersection area and union area of predictions regarding the octopus 🐙 from two different models.
</div>

Sometimes, especially when detecting small objects, it is possible that two bounding boxes from different predictions address the same object without overlapping. In these cases, the generalized IoU comes in handy. It extends the sets $$ A $$ and $$ B $$ by the smallest convex hull $$ C $$ that encloses both and is defined as

$$ \mathrm{GIoU}(A,B) = \frac{\lvert A \cap B \rvert}{\lvert A \cup B \rvert} - \frac{\lvert C \setminus (A \cup B) \rvert}{|C|} = \mathrm{IoU}(A,B) - \frac{\lvert C \setminus (A \cup B) \rvert}{\lvert C \rvert} . $$

The generalized IoU can take values between -1 and 1, i.e. $$ -1 \leq \mathrm{GIoU}(A, B) \leq 1 $$.

As already mentioned, we have to choose our scoring function, which we study next, and the aggregation strategy carefully to prevent an unintended selection bias. In the worst case, randomly selected samples lead to a better performance, rendering the extra effort of active learning useless.

### Uncertainty

Research on active learning for deep learning on computer vision tasks initially focused on the simpler image classification setting. Since object detectors also classify objects, some of these approaches have been adopted. Later, additional methods were developed that also exploit the localization uncertainty.

#### Classification Uncertainty

Usually, the classification heads of deep learning models provide the log probabilities of all classes normalized by the softmax function as their output. This distribution over possible classes, the predictive probabilities, can be leveraged in several ways to get measures of how confident the model is about its prediction.

The simplest approach is to aggregate the probabilities of the most probable class of each detection in an image and select samples with low confidence scores. On the other hand, neural networks are known to be badly calibrated <d-cite key="guo2017"></d-cite>. A confidence score of 70&nbsp;% does not necessarily translate to predictions that are correct 70&nbsp;% of the time. Neural networks tend to be overconfident in their predictions since this further reduces the training loss.

Using the difference between the probabilities of the most probable and second most probable class is known as margin sampling or 1-vs-2, which is shown Fig. 9. If the model gives a similar confidence to both classes (i.e. the margin is small), this indicates that the model is uncertain about the category of an object. We want to select such small-margin samples during active learning to improve the decision-making for them. Brust et al. <d-cite key="brust2018"></d-cite> tried this approach for object detection.

{% include figure.html path="assets/img/2025-04-28-active-learning-supervision-object-detection/margin_sampling.png" class="img-fluid rounded" %}
<div class="caption">
    Figure 9: Leveraging the class prediction distribution using the example of margin sampling.
</div>

Instead of using only the top-1 or top-2 of the predictive probabilities, we can use the full distribution. Treating the class prediction as a discrete random variable $$ X $$, its entropy quantifies the uncertainty regarding the potential states of $$ X $$. It is defined as

$$ H(X) := - \sum_{x \in X} P(x) \log P(x) . $$

A class prediction with a clear favorite has a low entropy, while a class prediction that resembles a uniform distribution has a high entropy. If the model assigns similar confidence to several classes for a detection, this high-entropy sample should be selected for annotation to help clarify the category of the object. Roy et al. <d-cite key="roy2018"></d-cite> use this measure in the context of object detection.

We can also estimate predictive uncertainty through model ensembles <d-cite key="lakshminarayanan2017"></d-cite>. Combining the predictions of multiple models trained on the same data, but differently initialized, gives us again a distribution of class predicitions whose entropy we can calculate. While the training of multiple models is computationally more expensive, the measure of informativeness becomes more robust this way. If multiple models agree that their predictions for a given sample are uncertain, the chance that labeling this sample brings additional value to the dataset is high. Having multiple models estimate the uncertainty makes the sample selection less tied to a single model, which might be beneficial when the dataset, built in cooperation with this model, is going to be used to train other models in the future.

Another way of using ensemble predictions and entropy is mutual information, as Haussmann et al. <d-cite key="haussmann2020"></d-cite> did. They formulate mutual information as

$$ \mathrm{MI}(X) := H(\bar{X}) - \frac{1}{\lvert E \rvert} \sum_{e \in E} H(X^{(e)}) , $$

with $$ \bar{X} $$ as averaged class prediction over all members of the ensemble $$ E $$ and $$ \lvert E \rvert $$ as the cardinality of the ensemble. The difference between the entropy $$ H(\bar{X}) $$ of the average class probabilities and the average of the entropies of each model's prediction gives the mutual information. This can be used as measure of disagreement between the models in an ensemble.

A common approximation of the ensemble approach is Monte Carlo dropout <d-cite key="gal2016"></d-cite>. Applying dropout <d-cite key="srivastava2014"></d-cite> at test time means that the predicition is made only with a subset of the entire neural network. Some neural units are made temporarily inactive. Multiple forward passes with different neural units dropping out can be viewed as an ensemble. This approximation needs less computational resources to train.

An approach in a similar vein is the uncertainty (or inconsistency) estimation from the stability of predictions under augmentations <d-cite key="elezi2022,yu2022"></d-cite>. Like the consistency regularization approach to semi-supervised learning we discussed earlier, predictions are gathered for the original sample and some augmented variant(s) of it. Then, the Jensen-Shannon divergence of the predictive probabilities is used to measure how uncertain the model is about its prediction. Yu et al. <d-cite key="yu2022"></d-cite> additionally use the IoU as a measure of localization uncertainty in their approach, while Elezi et al. <d-cite key="elezi2022"></d-cite> combine their inconsistency score with a classification uncertainty based on the entropy of the predictive probabilities.

As mentioned earlier, all methods that are based on predictions from multiple models or multiple forward passes of the same model must first find the associations between the detections to calculate an object-level informativeness score. As explained above, this is commonly done via IoU matching.

#### Localization Uncertainty

Informativeness measures for object detection that go beyond image classification can be derived from the uncertainty regarding bounding box coordinates. Kao et al. <d-cite key="kao2019"></d-cite> introduce two such measures.

The first one is named localization tightness and expresses how tightly a bounding box encloses an object. As the ground truth bounding box is not available for unlabeled samples, this is estimated through a proxy. Two-stage object detectors based on convolutional neural networks make the binary decision whether an object centered around a pixel of their feature map(s) is present or not and, if that is the case, provide a region proposal. This initial bounding box guess gets further refined subsequently. The proxy to estimate localization tightness that Kao et al. <d-cite key="kao2019"></d-cite> propose is the IoU of region proposal and the corresponding final prediction (see Fig. 10). It is thus only applicable to two-stage detectors.

{% include figure.html path="assets/img/2025-04-28-active-learning-supervision-object-detection/localization_tightness.png" class="img-fluid rounded" %}
<div class="caption">
    Figure 10: The localization tightness is calculated as IoU of region proposal and final prediction.
</div>

The second informativeness measure by Kao et al. <d-cite key="kao2019"></d-cite> is localization stability. As illustrated in Fig. 11, the original sample and several augmented versions of it are given to the detector. The localization stability is defined as the average IoU of all augmented versions with the original. When aggregating the image-level score, the instances are weighted by their highest class probability. Like that, instances are preferred that have a high localization uncertainty and high probability of containing an object of interest at the same time. While Kao et al. <d-cite key="kao2019"></d-cite> limited the augmentations to different levels of Gaussian noise, Yu et al. <d-cite key="yu2022"></d-cite> applied a wider range of augmentations.

{% include figure.html path="assets/img/2025-04-28-active-learning-supervision-object-detection/localization_stability.png" class="img-fluid rounded" %}
<div class="caption">
    Figure 11: The localization stability is the average IoU of the "clean" reference prediction and several prediction under augmentation.
</div>

Schmidt et al. <d-cite key="schmidt2020"></d-cite> estimate the localization uncertainty from a model ensemble. They match the predictions of multiple models and calculate the minimum IoU between all predictions of the ensemble that concern the same object. They also extend this approach to include class predictions. This should also find samples where the ensemble agrees about the localization but predicts different classes for that object. In their experiments, both variants did not outperform random selection though.

Choi et al. <d-cite key="choi2021"></d-cite> replace the usual prediction heads of deep object detectors with a mixture density network <d-cite key="bishop1994"></d-cite>. From the Gaussian mixture model that it learns, aleatoric and epistemic uncertainties, a common decomposition of predictive uncertainty, for classification and localization can be computed. Aleatoric (or statistical) uncertainty comprises the inherent randomness of a process that leads to variable outcomes. For object detection, a high aleatoric uncertainty could, for example, be caused by sensor noise or occlusion. Epistemic (or systematic) uncertainty is caused by the lack of knowledge. For each detection, Choi et al. <d-cite key="choi2021"></d-cite> compute these two uncertainties per subtask (classification and localization) from a single forward pass. For the aggregation of each uncertainty score, they use the maximum (normalized) uncertainty over the detected objects in an image. They try several strategies to aggregate a unified image-level score from the four uncertainties and find the maximum value of all scores to work best.

### Expected Model Change

Intuitively, samples that cause the detection model to change significantly should be informative. Ash et al. <d-cite key="ash2020a"></d-cite> propose a method to identify such samples, which Haussmann et al. <d-cite key="haussmann2020"></d-cite> adapt to object detection. The average class label predicted by the ensemble is used as the ground truth label. Under this assumption, each ensemble member can compute its loss and a "hallucinated" gradient. The magnitude of the gradient is then used as the uncertainty score.
 
Another approach is the loss prediction module by Yoo and Kweon <d-cite key="yoo2019"></d-cite>. This module is task-agnostic and can be applied to object detection. It is trained along with the actual model and learns to predict the loss value the model's prediction will cause for each sample. It receives intermediate feature maps from the model as input and the loss values observed for labeled samples as supervision. When applying this module to unlabeled samples, the predicted loss values can be used as image-level informativeness scores. The authors note, however, that the accuracy of loss predictions was relatively low when they applied this method to complex tasks such as object detection.

### Diversity

An important consideration when selecting samples for annotation is that they represent the data distribution that underlies our dataset well. When we select many similar samples because the model's predictions are uncertain for them, we improve the model in this specific data region. But the model could probably learn the necessary decision from fewer samples and would overall benefit more from a selection that covers the data distribution more widely. This is especially important when we perform active learning in the batch setting, where we select multiple samples at once for annotation. When we select samples one by one, the model's focus can shift after each sample if the newly added annotation provided enough supervision to learn a previously uncertain decision. When adding a (large) batch of samples to the pool of labeled data at once, we should ensure that they are diverse.

Measuring diversity for images is not straightforward. The usual proxy method is to work with feature vectors obtained from a feature extractor like the detection model's backbone. These representations of all samples can then be used in clustering approaches such as k-means++ <d-cite key="arthur2007"></d-cite> or core-set <d-cite key="sener2018"></d-cite>, a method to best cover the larger distribution of samples with only a subset of them. Like that, data points can be selected that help at representing the distribution. Haussmann et al. <d-cite key="haussmann2020"></d-cite> study this approach in an object detection setting.

The aforementioned approach is usually applied at the image-level, i.e. the similarity of entire images is measured. In object detection, two scenes could look quite similar overall while the present objects are completely different, especially for small objects. Therefore, Yang et al. <d-cite key="yang2024"></d-cite> introduced category conditioned matching similarity. When measuring the similarity of two images, feature vectors of detected objects are calculated and each object is matched to the most similar object of the same category in the other image. The image-level score is then calculated as the average of the object similarities.

### Undetectability

Most prior methods base their score calculation on detections the model made. But one of the failure modes of object detectors are false negatives, objects the model has not detected although it should have. Nakamura et al. <d-cite key="nakamura2024"></d-cite> first explicitly stated this shortcoming in the context of active learning and proposed a solution. They alternately train a false negative prediction module with their detection model. The false negative prediction module receives the features extracted by the detector backbone as input and receives supervision on labeled samples as the number of missed detections. Using the prediction of the false negative prediction module on unlabeled samples provides an image-level score that can be used for sample selection.

### Combining Multiple Criteria

Many of the scores above can be added together to consider the different aspects they address jointly. Quite commonly, scoring and sampling are even distinguished in two phases that consider different aspects <d-cite key="wu2022,yang2024"></d-cite>. Scoring is often based on one or multiple uncertainty scores and determines a ranking of the unlabeled samples. When progressing to the sampling phase, a pool larger than the annotation budget assigned for that active learning round is selected. The samples that will actually be labeled in that round are selected from that pool according to a diversity measure. This ensures that they represent the underlying data distribution well and prevents the annotation of many redundant samples.

It is also common to weight some measures by other quantities, as is done with localization stability <d-cite key="kao2019"></d-cite> and the highest class probability of that object.

Yang et al. <d-cite key="yang2024"></d-cite> propose another way to incorporate both classification and localization. During training with labeled samples, the training difficulty of all predicted boxes is calculated as

$$ q(b \MidSymbol[\vert] \hat{b}) = 1 - P(b \MidSymbol[\vert] \hat{b})^\xi \cdot \mathrm{IoU}(b,\hat{b})^{1-\xi} , $$

where $$ P(b \MidSymbol[\vert] \hat{b}) $$ is the classification probability of the predicted box $$ b $$ with respect to the matched ground-truth box $$ \hat{b} $$ and $$ 0 \leq \xi \leq 1 $$ is a hyperparameter. When scoring unlabeled samples, the entropy of the class prediction of each instance is weighted by the difficulty coefficient corresponding to the most probable class.

Yang et al. <d-cite key="yang2024"></d-cite> also find that the importance of different measures used in active learning varies with the progress of the procedure. In their experiments, uncertainty-based measures were more important in earlier active learning rounds, while in later rounds diversity became more critical.

## Object Detection With Minimal Manual Annotation Effort

Now that we have an understanding of possible types of supervision and the way we can rate informativeness to select samples for annotation during active learning, let's see how we can bring the two together successfully.

Nakamura et al. <d-cite key="nakamura2024"></d-cite> combine unsupervised domain adaptation (UDA), a transfer learning technique that does not use target domain labels, and active learning for object detection. Their UDA method joins self-training with pseudo-labels and adversarial learning for feature space alignment between domains. They found that their model produced many false negative predictions under the domain shift and UDA on its own could not fully eliminate this undesired behavior. As we have seen earlier, most approaches to select informative samples for annotation are based on measures that we take for detected objects. If the main problem of our model is that it misses objects it should detect, we should select some of these samples with false negative predictions. We already looked into the solution of Nakamura et al. when we discussed undetectability as a scoring function. They train a false negative prediction module in alternation with their detection model. The small module learns to predict the number of objects that the detection model has missed by using the features of the detector backbone as input and the number of missing detections on labeled examples as supervision. Combined with uncertainty scores for classification and localization as well as a diversity score, the active learning procedure can focus on the samples that help to reduce false negative errors, the main weakness of the detection model in that setting.

The success of semi-supervised self-training generally depends on the initial dataset and the predictive capabilities the model learns from it. High-confidence predictions are reinforced, while possibly correct, low-confidence predictions are reinforced as "background". Thus, this setting is predestined to need support regarding false negative predictions. At the same time, diversity is important as the self-training model has to generalize from the few labeled data points to the whole underlying data distribution.

Perhaps counterintuitively, when having Goodhart's law<d-footnote>"When a measure becomes a target, it ceases to be a good measure."</d-footnote> in mind, Elezi et al. <d-cite key="elezi2022"></d-cite> find that a consistency-regularizing loss on unlabeled samples improves their active learning approach based on inconsistency under augmentation (and entropy-based classification uncertainty). They argue that the model needs to try inconsistency minimization during training to effectively exploit that information during sample selection. In addition, they use self-training based on pseudo-labels with a modified loss function that mitigates the problem explained above regarding the reinforcement of low-confidence regions as "background".

Cetintas et al. <d-cite key="cetintas2024"></d-cite> use pre-training on synthetic data, self-training with pseudo-labels and active learning in their annotation pipeline for visual tracking. They mitigate the problem of false negative predictions by using over-complete sets of detections. They apply a low confidence threshold and thereby include many false positive predictions in the model output, but increase the chances to find an otherwise lost true positive detection at the same time. Their tracking-by-detection approach uses detections on single frames of a video that have to be associated optimally across frames for successful tracking. By leveraging the available temporal information, many wrong predictions can be filtered out, as there are no corresponding detections on adjacent frames. With this reliance on temporal consistency to remove nonsensical detections, their pipeline successfully reduces annotation cost.

When training an object detector without such additional (temporal) information, allowing many false positive predictions due to a low confidence threshold could introduce an unintended bias to sample selection during active learning.<d-footnote>Unfortunately, the confidence threshold is usually not reported for experiments with active learning for object detection, which complicates an investigation of this speculation. Presumably, the default settings of the used model or deep learning library have been applied in most cases and the authors did not report this parameter as it did not seem relevant.</d-footnote> For example, Cordier et al. <d-cite key="cordier2021"></d-cite> simply select every sample for annotation that contains a detection by their model. In their use case, defect detection for industrial parts, true positive predictions should be rare. Therefore, samples with detections, regardless of whether they are true positive or false positive, are always informative to their model.

Desai et al. <d-cite key="desai2019"></d-cite> propose an extension of the standard active learning loop, as shown in Fig. 12, to leverage weak labels when possible. In their approach, the oracle first provides only weak labels in the form of center point annotations for the selected samples. They use two different approaches to decide then whether the annotation should switch to full supervision. With the hard switch, the mAP of the current model on a validation set is compared to the mAP of the model before the current round of active learning. If the ratio of this difference and the maximum of this difference over all rounds falls below a threshold, indicating that model training stagnates, strong labels are queried in all future rounds. With the soft switch, full supervision can be requested for each sample in the current active learning round individually. If the mean probability of the predicted class over all detected objects in an image is below a threshold, the sample is send again to the annotator to draw bounding boxes. When working with the weak labels, they produce pseudo-labels by selecting those bounding box predictions that have their center closest to the center point annotation. In their experiments, the soft switch approach performs best regarding detection performance (measured by mAP) in relation to annotation time.

{% include figure.html path="assets/img/2025-04-28-active-learning-supervision-object-detection/desai2019.png" class="img-fluid rounded" %}
<div class="caption">
    Figure 12: Desai et al. <d-cite key="desai2019"></d-cite> modify the active learning by querying primarily weak labels from the oracle and adapting the supervision to a stronger signal when necessary.
</div>

Weakly-supervised approaches tend to predict incomplete bounding boxes, which only cover the most discriminative part of the object. In this case, better supervision regarding localization is necessary. While Desai et al. <d-cite key="desai2019"></d-cite> only compare three classification-based scores in their active learning approach with weak labels, the problem of sometimes poor localization can be seen in the examples shown in their paper, which were selected for strong supervision. Some methods have been developed to correct this behavior of weakly-supervised object detection <d-cite key="pan2019,biffi2020"></d-cite>. Vo et al. <d-cite key="vo2022"></d-cite> propose an active learning strategy to address this shortcoming of weakly-supervised learning. For image-level class labels, they find that a weakly-supervised detector often produces box-in-box predictions (see Fig. 13), where either two bounding boxes of different size belong to the same object or a large bounding boxes encloses a group of objects from the same class while some of these have their own bounding boxes in addition. With their method to identify such cases and a diversity score regarding these box-in-box pairs, they can substantially improve their detector with well-directed full supervision.

{% include figure.html path="assets/img/2025-04-28-active-learning-supervision-object-detection/box-in-box.png" class="img-fluid rounded" %}
<div class="caption">
    Figure 13: Weakly-supervised learned models can have difficulties with localization. Box-in-box predictions are shown here as an example.
</div>

We have discussed several approaches that make use of cheap supervision and active learning in conjunction. As we have seen, the configuration of a training pipeline depends largely on the use case and the available information in the data. Generally speaking, it seems promising to leverage labeled (synthetic) data that is close to our problem domain and self-training with pseudo-labels. The quality of the pseudo-labels is of great importance here, which is why all training techniques <d-cite key="mi2022"></d-cite> and information that can lead to their improvement should be utilized. Ideally, this prepares our detection model for the selection of relevant samples during active learning without the need for an initial set of labeled data from our target dataset, as implemented by Wang et al. <d-cite key="wang2023b"></d-cite> and Cetintas et al. <d-cite key="cetintas2024"></d-cite>. If relevant additional data is not available and we cannot generate a synthetic surrogate, self-supervised learning (possibly joint with semi-supervised learning <d-cite key="zhai2019"></d-cite>) can be a good starting point. We have also seen that, at times, it can be sufficient to query the annotator for weak supervision during active learning.

## Conclusion

In this blog post, we explored in which ways we can provide supervision to machine learning models that try to learn object detection. After a general overview of different flavors of supervision, we dived into active learning approaches for object detection and subsequently examined how these complement other supervisory signals. To summarize our key takeaways:

1. **Harmonize active learning and cheap supervision.** Active learning can and should be biased towards selecting samples that effectively compensate for the weaknesses of other supervisory signals in the model training loop. Possible failure modes are false positives (background detected as object), false negatives (missed detections), bad object localization, and wrong object class prediction.

2. **Choose scoring function, aggregation strategy, and sample selection carefully.** Each of these aspects has the potential to turn an active learning pipeline that saves annotation cost into an elaborate selection of samples that cannot outperform a random selection.

3. **Engage complementary approaches in all learning phases.** Active learning does not have to query for full supervision all the time. Weak annotations can sometimes be enough to advance learning.