---
layout: distill
title: Do vision models perceive objects like toddlers ?
description: Despite recent advances in arficial vision systems, humans are still more 
  data-efficient at learning visual representations. Psychophysic experiments suggest 
  that toddlers develop fundamental visual properties between the ages of one and 
  three, which likely affect their perceptual system. 
  Especially, they become able to recognize impoverished variants of daily objects, pay more attention to the shape of 
  an object to categorize it, move objects in specific orientations and progressively 
  generalize over the configural arrangement of objects' parts. The post examines 
  whether these four visual properties also emerge in off-the-shelf machine learning (ML) 
  vision models. Throughout this post, we reproduce and complement previous studies by 
  comparing toddlers and 
  a large set of diverse pre-trained vision models for each visual property. This way, we 
  unveil the interplay between these visual properties and highlight the main differences between ML models and toddlers.
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
bibliography: 2025-04-28-visualprop.bib

# Add a table of contents to your post.
#   - make sure that TOC names match the actual section names
#     for hyperlinks within the post to work correctly. 
#   - please use this format rather than manually creating a markdown table of contents.
toc:
  - name: Introduction
    subsections:
      - name: Motivation
      - name: Scope of the study
  - name: Experiments
    subsections:
      - name: Caricature recognition
      - name: Shape bias
      - name: Side-view bias
      - name: Configural sensitivity
  - name: Conclusion
    subsections:
      - name: Acknowledgments
  - name: Methods
    subsections:
      - name: Models


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

# Introduction

## Motivation
State-of-the-art machine learning vision models learn visual representation by 
learning on millions/billions of independently and identically distributed diverse images.
In comparison, look at toddlers around: they play with/observe almost always the same 
objects in the same playground/home/daycare environments. They likely have not seen 10% of the different dogs of ImageNet and likely
not a single of the 1000 sea snakes of ImageNet-1k. In sum, toddlers likely experience a 
diversity of objects which is several orders of magnitude lower than current models. 
Unlike current machine learning (ML) vision models, they also do not access a massive amount category
labels (or aligned language utterances), adversarial samples and do not massively
color-transform or mask their visual inputs. Yet, they develop strong 
semantic representations that are robust to image distortions, viewpoints, machine-adversarial
samples and different styles (draw, silhouettes...) <d-cite key="wichmann2023deep, 
huber2023developmental"></d-cite>.

What perceptual mechanism underpins such a robust visual system ? Psychophysics 
experiments demonstrated that toddlers progressively reach fundamental visual milestones and develop
visual biases during their second and third year. Specifically,  toddler become able 
to extract the category of simplified objects <d-cite key="smith2003learning"></d-cite> and learn to 
attend to the shapes of novel objects when judging their 
similarity based on their *kind* <d-cite key="diesendruck2003specific"></d-cite>. Simultaneously, 
they also increasingly focus on object's views
that highlight their main axis of elongation <d-cite key="pereira2014main"></d-cite>, 
and later, begin to semantically identify objects based
the configural relation between their parts <d-cite key="augustine2011parts"></d-cite>.
All these developments obviously shape how they perceive the similarity between 
objects and their object recognition abilities over the rest of their life. This raises 
the question of whether off-the-shelf ML models also exhibit these visual properties.


In this blogpost, we add and extend previous studies to investigate the presence of the 
four above-mentioned visual properties in a set of diverse pre-trained models. We include 
diverse models based on 
supervised learning, semi and weak supervision, robust out-of-distribution image 
recognition, self-supervised learning with instance discrimination 
(ID) and masked auto-encoders (MAE), adversarial training, 
vision-language models and egocentric visual training. We also try both convolutional architectures (CNN) and vision transformers (ViT). We 
refer to the end of the blogpost for a complete list 
of models. Our objective is to 1) reproduce and extend prior experiments 
comparing ML to humans to (in)validate prior claims; 2) provide a global picture of 
the (dis)similarities between the visual properties of ML models and toddlers; 3) clarify the interplay between
different visual properties in ML. We will also publicly release our code on github upon
acceptance as an easy-to-use toolbox.


## Scope of the study

We divide our study into four parts, each corresponding to a specific emerging visual
property in toddlers. In the following, we describe each of them and clarify the scope of
our experiments with respect to prior work. Note that we do not aim to faithfully 
reproduce evaluation protocols from developmental studies (which are often based on language). We use alternatives that aim to
assess the presence of a visual property relative to mature toddlers.

{% include figure.html path="assets/img/2025-04-28-visualprop/overview.png"
class="img-fluid" %}
<div class="caption">
Figure 1. Illustration of emerging visual properties in toddlers. Green rectangles means 
that toddlers can match images based on their category (A and B) or focus on specific views (C). Red rectangle (D) means that humans learn to mismatch stimulis.
</div>

#### Caricature recognition

After 18 months, toddlers become able to recognize
impoverished variants of daily-life objects called caricatures, which are
constructed with few simple shapes<d-cite key="smith2003learning"></d-cite> (e.g. Figure 1, A) . The
recognition of these caricatures is related to the development of abstract 
representations and may boost category generalization<d-cite 
key="son2008simplicity"></d-cite>. To the best of our
knowledge, the only study investigating ML models for caricature recognition focuses on
vision-language models <d-cite key="sheybani2024modelvsbaby"></d-cite>. We modify
their experimental protocol to compare a wider range of pre-trained models.

#### Shape bias

When asked to match objects based on their *kind* (but also names etc...), 
toddlers group together novel objects that share the same shape, compared to objects that share the same color 
or texture <d-cite key="diesendruck2003specific"></d-cite> (cf. Figure 1, B). This 
bias begins at around 18 months old and after learning caricature recognition <d-cite 
key="yee2012changes"></d-cite>. This effect has been widely investigated in ML models 
following the introduction of related datasets <d-cite 
key="geirhos2021partial, hermann2020origins"></d-cite>. However, mainstream evaluation
protocols differ from evaluation protocols in the developmental literature <d-cite
key="tartaglini2022developmentally"></d-cite>, which are known to be very sensitive to
experimental details <d-cite key="frigo1998adjective"></d-cite>. Here, we extend
the developmentally-relevant experiments of Tartaglini et al. (2022) to include more
models.

#### Side-view bias

From 18-24 months onward, toddlers learn to manipulate objects
to focus on views showing 1- objects in upright positions; 2- objects' main axis of 
elongation perpendicularly to the line of sight, which we call side view (e.g. Figure 1,
C) <d-cite key="pereira2010early, pereira2014main"></d-cite>. Here, we focus on the 
side-view bias. Evidences suggest this is a perceptual bias (thus, e.g., not a 
motor/grasping bias) <d-cite key="james2014some"></d-cite>, and due to a 
visual inspection of these views while the object being close-to static <d-cite 
key="pereira2014main,lisboa2014looking"></d-cite>. However, its origins and 
consequences on visual learning remain unclear. To the best of our knowledge, we are the first to investigate the
potential specificity of these views for ML models.

#### Configural relation between parts 

Starting at around 20 months, infants increasingly account for the relative position of parts of an object to
categorize it <d-cite key="augustine2011parts"></d-cite>. Such an ability become fully
mature for novel objects later in infants development<d-cite key="mash2006multidimensional"></d-cite> and
is a fundamental component of one of the main theory of object recognition <d-cite key="
hummel1992dynamic"></d-cite>. To the best of our knowledge, two studies treat this 
ability in ML models <d-cite key="baker2022deep,farahat2023novel"></d-cite> with two opposite conclusions. We
discuss their differences and assumptions and extend the experiments of Baker et al. 
(2022) (Figure 1, D) to include more models.

## TL;DR

Our analysis reveals two main visual milestones. The first one is the 
acquisition of the shape bias, which emerges simultaneously to the ability to recognize 
normal objects and caricatures. While the shape bias is also clearly connected with 
side views being better prototypes of object instances, side-views are not prototypical 
enough to explain a side-view bias at the level of toddlers'. This bias is close to 
toddlers' in the strongest models. The second one is the development of 
configural sensitivity, for which adults' (and probably toddlers') level is out-of-reach 
for all 
investigated models. None of the two milestones are achieved by investigated models 
that were trained on a reasonable amount of visual egocentric data. In sum, toddlers' 
visual properties suggest there is room for ML improvements in 1) making the development 
of the shape bias more data-efficient and 2) learning to generalize based on the configural arrangement of parts.
For 1), one should not overlook CNNs architectures, as we found them to show a more 
robust shape bias on novel shapes, compared to ViTs.

# Experiments

## Caricature recognition

The main developmental protocol to evaluate caricature recognition is to ask a subject 
"where is the {object}" among a set of one matching object caricature and non-matching 
object caricature <d-cite key="smith2003learning"></d-cite>. 
This has been recently adapted to vision-language models by giving the object name to the
language encoder and retrieving the closest image embeddings in a set of images <d-cite
key="sheybani2024modelvsbaby"></d-cite>. 

Here, we modify their evaluation protocol to include a broader set of models. We take 
the BabyVsModels dataset <d-cite key="sheybani2024modelvsbaby"></d-cite>, which 
contains 8 categories, each with 5 and 8 
different images for realistic and caricatures, respectively. We create the set of all 
possible triplets, each constructed with a realistic image of
an object, one caricature from the same category and one caricature from a
different category. We compute their image representations and test whether the cosine
similarity between the representation of the realistic image and its 
category matching caricature is higher than the cosine similarity between the 
realistic image representation and the non-matching caricature. We define the 
recognition accuracy as the success rate. We replicate the
process with a harder 8-choices (versus two-choices) classification variant to check 
whether the difficulty of the task impact our results. We use 100 tuples per realistic 
image. Furthermore, to estimate the recognition performance of models on normal images,
we also compute the success rate on realistic images.

{% include figure.html path="assets/img/2025-04-28-visualprop/caricaturesimg.png"
class="img-fluid" %}
<div class="caption">
Figure 2. Illustration of how we compute A) simple caricature recognition accuracy 
and B) the normal image recognition accuracy. 
</div>


In Figure 3, A), we find that diverse models reach a strong caricature recognition 
performance, at the estimated level of toddlers. While supervised 
and vision-language models unsurprisingly perform well. We notably find 
that ID methods (DinoV2, BYOL, AA-SIMCLR) perform reasonably well on caricature 
recognition. Figure 3, B) indicates that the hard caricature recognition accuracy correlates with hard image 
recognition accuracy 
(Pearson score: $0.78$). Overall, we conclude that strong caricature recognition may 
 be a side-effect of strong category recognition and that cutting-edge models 
already reach the level of toddlers.


{% include figure.html path="assets/img/2025-04-28-visualprop/caricatures.png"
class="img-fluid" %}
<div class="caption">
Figure 3. A) Simple caricature and B) hard caricature accuracy against normal image
accuracy recognition. We estimate 24-months toddler performance based on data from 
Pereira et al. (2009). This is consistent with results from Sheybani et al. (2024),  
which suggests that they perform similarly to pre-trained CLIP models. 
</div>

## Shape bias

It seems obvious that the best investigated models leverage objects' shapes to
measure image similarities, as textures and colors cues are absent from caricatures. 
Here, we examine the shape bias of current models, compared to toddlers, and analyze 
the relation between shape bias and caricature recognition.

ML benchmarks used to evaluate the shape bias of models <d-cite key="
geirhos2021partial,hermann2020origins"></d-cite> do not follow the experimental protocols
of developmental studies, as they texture the whole image instead of the object itself <d-cite key="smith2003learning"></d-cite>; 
thus we follow the cue-conflict experimental protocol of Tartaglini et al. (2022). First, we evaluate the shape bias on
common (in-distribution) objects. We reuse masked shapes of common objects used in<d-cite
key="hermann2020origins,tartaglini2022developmentally"></d-cite>, which include 16
categories, each containing 10 masks of different instances, and the 16 textures used
in <d-cite key="tartaglini2022developmentally"></d-cite>. We construct an anchor image
mixing a shape with a texture, a variant built with the same shape but a different texture
and another variant built with the same texture but a different shape. Then, for each
triplet, we define a shape bias success as the embedding of the anchor being closer to the
same shape variant compared to the same texture variant and define the shape bias accuracy
as the success rate. The set of triplets includes all possible combinations.
Second, to evaluate the shape bias on novel objects, we reproduce the same protocol, but
using the 16 novel shapes used by<d-cite key="
tartaglini2022developmentally,parks2020statistical"></d-cite>.

{% include figure.html path="assets/img/2025-04-28-visualprop/shapebiasimg.png"
class="img-fluid" %}
<div class="caption">
Figure 4. Illustration of how we compute the degree of shape bias on A) common objects 
and B) novel shapes. $d$ is the negative cosine similarity.
</div>

In Figure 5, A), we verify that there is a correlation between hard caricature
recognition and the degree of shape bias of a model over common objects (Pearson
score: $0.68$). This is higher than the correlation with hard object recognition ($0.
56$). We also find that language is not a mandatory component of the shape bias as ID 
methods seem to classify object based on
shapes, especially DinoV2, which shows a similar degree of shape bias as vision-language
or supervised models. This is interesting because developmental studies clearly found 
that the productive vocabulary size is a strong predictor of the shape bias <d-cite 
key="gershkoff2004shape,jones2003late"></d-cite>. Our results suggest that one should not 
over-estimate the causal importance of
language for the acquisition of this bias <d-cite key="
gavrikov2024vision,smith2002object"></d-cite>. 


In Tartaglini et al. (2022), the gap of shape bias between toddlers and models increases 
for novel shapes. When reproducing their results, we notice that the performance of a 
random ResNet50 (*Random_RN50*, averaged over three random initialization) decreases 
between common and novel shapes. Since all shapes are *novel* for a random model, this suggests that the 
present novel shapes are inherently less prone to induce a shape-based matching than the 
common objects. To correct this effect, we multiply all scores by the ratio 
$$\frac{\texttt{ShapeBiasCommon}
(\texttt{Random_RN50})}{\texttt{ShapeBiasNovel}(\texttt{Random_RN50})}$$
and show the result in Figure 5, B). We find that the best models perform close 
to toddlers' performance. The best models, Noisy-Student and SWSL, respectively matches 
and outperforms toddlers. A thorough 
look at Figure 5, B) also reveals that CNNs
better generalize the shape bias to novel objects than ViT (e.g. CLIP_RN50 versus 
CLIP_ViT-L/14). This nuances
previous claims of ViTs being more shape biased than CNNs<d-cite 
key="tuli2021convolutional,naseer2021intriguing"></d-cite>. 

Overall, the best ML models have a relatively similar level of shape bias 
than toddlers.

{% include figure.html path="assets/img/2025-04-28-visualprop/shapebias.png"
class="img-fluid" %}
<div class="caption">
Figure 5. Shape bias for A) common objects and B) novel objects. We extract
3-years old toddler performance from Diesendruck et al. (2003) <d-cite 
key="diesendruck2003specific"></d-cite> and
assume the performance would be similar with our set of masks and textures.
</div>

## Side views

The previous section indicates that ML models reach a high degree of shape bias with
common objects. Simultaneously to its emergence, toddlers also focus their attention on
side views of objects <d-cite key="pereira2010early,pereira2014main"></d-cite>, defined as
views showing the main axis of elongation perpendicularly to the line of sight (when the object is in a canonical upright position).
To the best of our knowledge, the reasons for such a
bias are currently unclear. An hypothesis may be that side views are particularly
interesting for toddlers because they unambiguously display the whole shape of the 
object. 

To investigate this question, we leverage the OmniObject3D dataset <d-cite key="
wu2023omniobject3d"></d-cite> which contains 6,000 high-quality textured meshes scanned
from real-world objects, distributed in 190 categories. We specifically use the provided
recordings of 24 in-depth rotations per object in a canonical upright position. We 
preprocess this dataset by removing objects that do not show a clear main axis of elongation (e.g. a symmetric bottle). To do
so, we remove objects for which the ratio between the finest and widest views on the
horizontal axis is below $0.8$. The provided views were uniformly sampled around the yaw
axis, thus we define main side, front, 3/4 views as the views that have the closest 
angular rotation, with respect to a 
canonical side view, to $\\{0, 180\\}$, $\\{-90, 90\\}$, $\\{-135, -45, 45, 135\\}$ 
degrees, 
respectively. To assess the specificity of these views, we define two metrics: 
- the intra-object distance as the 
distance between an object's main view representation and all other views 
representations from the same object;
- the intra-category distance as the distance between an object's main view 
  representations and all other views representations from the same category.

For each object, we retrieve the view that maximizes and minimizes each
metric. Finally, we compute the side-view accuracy as the average number of time the main
side view maximizes these metrics. We similarly compute the 3/4-view and front-view accuracy.
This allows us to assess which orientation is more *prototypical* for an object or a 
category.
  

{% include figure.html path="assets/img/2025-04-28-visualprop/viewsimg.png"
class="img-fluid" %}
<div class="caption">
Figure 6. Illustration of how we compute the side-view accuracy for intra-object 
distance. $d$ refers to the negative cosine similarity.
</div>

First, we compute the Pearson correlation score between the intra-object side-view 
accuracy and our other metrics. We find that the metrics that most highly correlate with 
the side-view accuracy is the shape bias on common objects (Pearson score: $0.86$). 
The correlation is similar for intra-category side-view accuracy (Pearson score: $0.80$).
Visualizing the result in Figure 7 confirms that, the higher the shape bias, the 
more *prototypical* are the side views. Interestingly, AA-SimCLR <d-cite key="aubret2024self"></d-cite>, presents a boost of side-view 
accuracy for a given level of shape bias. This model leverages rotation-action during 
training and has been shown to better generalize visual representations in a 
viewpoint-wise fashion. This suggests that one may increase the 
side-view accuracy of the best-performing models that way. 

{% include figure.html path="assets/img/2025-04-28-visualprop/views.png"
class="img-fluid" %}
<div class="caption">
Figure 7. View accuracies for intra-object and intra-category similarity maximization.
</div>

We did not find raw data about how much relative time toddlers spend on side views, 
however current studies found (on a different sets of objects) that toddlers focus 
more on these views than 3/4 views <d-cite key="pereira2010early"></d-cite>. In our case, 
the 3/4 view accuracy remains out-of-reach for all models. Thus, if our 
hypothesis about the side-view bias turned out to be true, ML models have a far lower 
bias than toddlers. It could be that side-views are very interesting because they 
clearly display the configural arrangement between parts: the head of an animal is 
located at the opposite to its tail, with respect to its body. 

## Configural relation between parts

Paying attention to the shape of an object is different from looking at the relative
position of all parts. It could be that the model only extracts local <d-cite key="baker2020local"></d-cite> or 
intermediate <d-cite key="jarvers2023shape"></d-cite> shape feature for
recognizing categories. The
ability to categorize object based of the configural arrangement of their parts begins to
emerge relatively late in toddlers <d-cite key="augustine2011parts"></d-cite> and matures
only for older children <d-cite key="mash2006multidimensional"></d-cite>. Two lines 
of prior works studied whether ML models rely on the configural relation between parts to compare
images <d-cite key="baker2022deep,farahat2023novel"></d-cite>. For both, the idea is the
same: they try to modify the positions of parts of an image (e.g. a head, a tail, a wheel)
without deterioring the structure of these parts.

In <d-cite key="farahat2023novel"></d-cite>, they train a CNN (with 
supervision) with a specific
architecture that outputs representations corresponding to small parts of an image; then 
they spatially scramble these local representations and train (with supervision) another 
neural network on top of it.
Their assumption is that the first model learns representations of local and coherent 
features: in that case, the scrambling process will mimic a change in relative feature 
positions. They found that the scrambling process hurts the
recognition performance and conclude that current CNN pays attention to configural
relation between parts. To the best of our knowledge, the validity of their assumption 
 is unclear. In addition, their results could be
specific to their CNN and the two-steps training process.

In <d-cite key="baker2022deep,baker2018deep"></d-cite>, they propose to create 
*frankenstein* silhouettes by taking an object silhouette and horizontally flipping 
the upper part of the silhouette, while keeping aligned the boundaries of the silhouette.
The underlying assumption is that the flip operation does not alter the parts. They found that
the recognition performance of ML models is largely unaffected by the flipping process,
unlike adults. We notice that they mostly use animals in their study, such that the 
lower and upper part of the silhouette are well-distinct: the bottom part corresponds to legs and the top part often displays the shape of the face (ears etc.
..).

{% include figure.html path="assets/img/2025-04-28-visualprop/frankenstein.png"
class="img-fluid" %}

Thus, we take the set of stimuli from Baker et al. (2022), which
comprises 9 categories of animals, each containing 40 original and frankenstein 
silhouettes.
To reproduce experiments from Baker et al. (2022) without a classifier, we sample a
normal silhouette from each category and one frankenstein silhouette. We compute the
cosine similarity between the frankenstein silhouettes and all others and define the
accuracy as how often the cosine similarity between the frankenstein and the 
category-matching normal
silhouette is the largest (frankenstein test). To assess the relative
performance, we apply the procedure again after replacing the frankenstein silhouette by
another normal silhouette (normal test). Finally, we compute the 
configural sensitivity as the difference of performance between the success 
rate of the frankenstein test and the success 
rate of the normal test. To show sensitivity to the spatial arrangement of parts, the metric must be lower than $0$.

{% include figure.html path="assets/img/2025-04-28-visualprop/configuralimg.png"
class="img-fluid" %}
<div class="caption">
Figure 9. Illustration of the frankenstein test.
</div>

We first compute the Pearson correlation score between configural sensitivity and
hard image recognition ($-0.58$), hard caricature recognition ($-0.71$) shape bias on 
common objects ($-0.69$) and shape bias on novel objects ($-0.41$). Given that hard 
caricature recognition is the highest correlated metric,
we plot configural sensitivity against hard caricature recognition in Figure 10. We 
observe that the relative performance of current models is very weak. Only SWSL,
CLIP, Swag and DinoV2 are significantly more sensitive than a BagNet-9, a model that 
can not show configural sensitivity by design. All models are less sensitive than 
the estimated adults. We conclude that shape-biased models likely rely on local 
shape cues instead of the global arrangement of parts.


{% include figure.html path="assets/img/2025-04-28-visualprop/configural.png"
class="img-fluid" %}
<div class="caption">
Figure 10. Configural sensitivity of ML models against their shape bias on common objects.
</div>

# Conclusion

In this blogpost, we extended previous studies to investigate whether current ML models 
learn 4 fundamental visual properties that emerge in toddlers. We found that cutting-edge 
models reach the estimated level of a toddler with respect to caricature recognition and the shape bias on common objects. For the shape bias on novel objects, 
models are close to toddlers. However, most of the considered models use 
bio-implausible data: a lot more diverse images, labels, millions of aligned language 
utterances, color-jittered images, adversarial samples... In our study, models trained 
with comparable egocentric visual images (VC-1, VIP, R3M) 
perform poorly on all benchmarks. Despite all these *cheats*, ML 
models still perform poorly on configural sensitivity and the proposed side-view bias 
compared to toddlers and adults, suggesting there is room for improvements. We can not 
conclude on the level of toddlers for configural sensitivity, but, given the emerging 
sensitivity of toddlers <d-cite key="augustine2011parts"></d-cite> and the amplitude of the 
difference between models and adults, we speculate that ML models are 
likely inferior to toddlers.

We found that object recognition abilities correlates with caricature recognition, which 
also correlates with a strong shape bias. This suggests that these properties emerge 
simultaneously in ML models. However, it does not mean that increasing the shape bias 
systematically leads to better recognition abilities, as evidenced by the Noisy-Student 
model (high shape bias, relatively low caricature recognition) and prior works <d-cite 
key="hermann2020origins"></d-cite>. Furthermore, showing a shape bias spurs the side-view 
bias and the configural sensitivity, but it is far from enough 
to reach the level of humans. Thus, we speculate the shape bias may be a first step 
towards a better understanding of the relative arrangement of parts.

From a developmental perspective, prior work found that productive vocabulary size is a 
strong predictor of the shape bias in toddlers <d-cite key="gershkoff2004shape,
jones2003late"></d-cite>. It could be that language aids in building visual 
representations or that strong visual representations are needed to produce language.
Although we can not conclude about a potential causal direction of this relation, we 
discovered that a shape bias similar to toddlers' *can* emerge without language (DinoV2). Thus, 
one should not over-estimate the causal role of language in building visual representations. 

The origin and consequence of the side-view bias for toddlers is mostly unclear in the 
developmental literature. It may emerge because these views are very stable for small 
rotations <d-cite key="perrett1992use"></d-cite> or because they are standardized views for aligning
views of objects <d-cite key="pereira2010early"></d-cite>. 
In both cases, preliminary experiments indicate this bias may 
boost their object recognition abilities <d-cite key="james2001manipulating"></d-cite>.
We investigated for the first time the potential specificity of side-views for ML 
models and found that these views become more object-prototypical when the shape bias 
increases. This supports the hypothesis stating that side-views are standardized views of 
objects and may explain why this bias develops along with the shape 
bias. However, in this case, it remains unclear why side-views are 
still less object-prototypical than 3/4 views on average for ML models. It could be 
that ML models lack some visual property like configural sensitivity or that our dataset is too different 
from the set of 16 simple stimulus objects used in Pereira et al. (2010).

This works presents several limitations. First, the experimental protocol do not 
perfectly follow neither developmental experiments (often based on language), nor machine 
learning protocols (often based on language or supervised labels).
We chose not to rely on language to test a broad set of models beyond vision-language 
models, and we decided to not use supervised heads as it is unclear how it relates to 
toddlers' visual representations. Thus, we assume that, during their task, toddlers 
perform a form of cognitive comparison that resembles our comparison of 
representations. Second, the set of stimuli generally varies between developmental 
studies and ours, allowing to only approximate how a toddler would perform. This is 
especially salient in our study of configural sensitivity: to the best of our knowledge, 
toddlers have not been tested on frankenstein silhouettes. Third, the set of stimulis 
is rather small and often biased (very small for caricatures, only animals for configural 
sensitivity, white background in all datasets...). Despite that, our study provides a global picture of the emergence 
and interplay between toddlers' visual properties in ML models. We hope it will spur 
research in addressing the gap between models and toddlers.





















## Acknowledgments

Removed for anonymous peer-reviewing.


# Methods

## Models

To reproduce and strengthen the claims reported papers and this blogpost, we test a very
diverse set of pre-trained models, including those addressing specific shortcomings of ML
models. We use a mix of convolutional architectures (CNN) and vision transformers (ViT).

As most of previous experiments, we assess standard models based on random 
initializations and supervised ImageNet training for both ResNet50 and ViT-L. We also
consider two different adversarially trained supervised models <d-cite key="salman2020adversarially"></d-cite>, as previous works argued that they share similar
adversarial robustness properties as humans <d-cite 
key="guo2022adversarially"></d-cite>. Finally,
we add three supervised BagNet models that extract representations of small image patchs in the deepest
layer (corresponding to $9\times9$, $17\times17$ and $33\times33$ pixels)<d-cite 
key="brendel2018approximating"></d-cite>; this allows to give hints on whether the local
structure is enough to fulfill a given task.


On the weakly supervised side, we consider a strong ViT-L/16 trained on a large-scale
dataset of hashtags (SWAG) <d-cite key="singh2022revisiting"></d-cite>, four 
vision-language
models based on the openai CLIP with ResNet50 and ViT-L/14<d-cite key="
radford2021learning"></d-cite> and two open CLIP models trained on LAION-5B <d-cite key="
schuhmann2022laionb,cherti2023reproducible,ilharco_gabriel_2021_5143773"></d-cite> 
with a ConvNet-XXLarge and a ViT-G/14; these vision-language models 
show consistent
classification errors as humans <d-cite key="geirhos2021partial"></d-cite> and are very
robust at recognizing caricatures <d-cite key="sheybani2024modelvsbaby"></d-cite>. 

We further add two strong methods for out-of-distribution image 
classification <d-cite key="geirhos2021partial"></d-cite>, namely Noisy-Student <d-cite key="xie2020self"></d-cite> and SWSL <d-cite 
key="yalniz2019billion"></d-cite>. 

We include state-of-the-art self-supervised instance discrimination (ID) methods trained
on ImageNet-1K, namely MoCoV2 <d-cite key="chen2020improved"></d-cite>, BYOL <d-cite key="
grill2020bootstrap"></d-cite> and DinoV2 as a mainstream foundation model<d-cite key="
oquab2024dinov2"></d-cite>. We further include models trained on MVImgNet <d-cite key="
yu2023mvimgnet"></d-cite> to be less sensitive to viewpoints (SimCLR-TT and AA-SimCLR, a
slightly more viewpoint-sensitive variant) <d-cite key="aubret2024self"></d-cite>. We also
add a Masked Auto-Encoder (MAE) trained on ImageNet with a ViT-L/16 <d-cite key="
he2022masked"></d-cite>, as MAEs keep more information than ID methods about the spatial
location of features.

Finally, we evaluate three
vision models trained on robotic and egocentric data, namely VIP <d-cite key="mavip"></d-cite>,
VC-1<d-cite key="majumdar2023we"></d-cite> and R3M <d-cite key="nairr3m"></d-cite>. 
Such a training data is supposed to be relatively close to the visual experience of 
toddlers.




