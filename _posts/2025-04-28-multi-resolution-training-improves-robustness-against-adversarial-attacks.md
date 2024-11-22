---
layout: distill
title: Multi-resolution training improves robustness against adversarial attacks
description: Deep neural networks (DNNs) are now central to critical tasks like traffic sign recognition in autonomous vehicles. However, their vulnerability to adversarial attacks—small but deliberately crafted input perturbations—poses serious risks. To address this, we propose multi-resolution training, a novel method that utilizes lower-resolution information from input images to retain essential features while filtering out adversarial noises. Our approach integrates custom downsampling and upsampling layers into DNNs to enhance robustness. Testing on various DNNs shows effective enhancements in robustness against adversarial attacks, making this technique a promising solution for safer real-world applications.


date: 2025-04-28
future: true
htmlwidgets: true
hidden: false

# Anonymize when submitting
authors:
  - name: Anonymous

# do not fill this in until your post is accepted and you're publishing your camera-ready post!
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
bibliography: 2025-04-28-multi-resolution-training-improves-robustness-against-adversarial-attacks.bib  

# Add a table of contents to your post.
#   - make sure that TOC names match the actual section names
#     for hyperlinks within the post to work correctly. 
#   - please use this format rather than manually creating a markdown table of contents.
toc:
  - name: DNN Vulnerabilities
  - name: Multi-resolution training
    subsections:
    - name: Multi-resolution CNN block
  - name: Experiment
    subsections:
    - name: Adversarial Attacks  
    - name: Results  
    - name: Key Insights 
  - name: Conclusion

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
## DNN Vulnerabilities

Recent research has demonstrated that DNNs are surprisingly vulnerable to adversarial examples, where small modifications to input data can result in significant incorrect predictions<d-cite key="szegedy2014intriguing"></d-cite><d-cite key="goodfellow2015explaining"></d-cite>. This susceptibility arises from DNNs' inherent linearity in high-dimensional spaces and limited generalization. This vulnerability is especially concerning for traffic sign recognition systems, which rely exclusively on DNNs processing camera images without additional error correction sources. 

Many research efforts have been made on attacks targeting these systems, as they are particularly susceptible and easily exploited in real-world scenarios<d-cite key="pavlitska2023adversarial"></d-cite>. Early research into adversarial attacks on traffic sign recognition typically involved physical methods, such as affixing stickers to traffic signs<d-cite key="eykholt2018robust"></d-cite>, replacing signs with versions that included embedded perturbations<d-cite key="pavlitska2023adversarial"></d-cite>, or placing patches directly on camera lenses<d-cite key="zolfi2021translucent"></d-cite>. While these methods can be effective, they require manual effort and are often noticeable to drivers. 

To address these limitations, researchers have shifted their focus to direct attacks on DNN inputs. One of the first developed techniques is the Fast Gradient Sign Method (FGSM), which uses the gradients of the loss function relative to the input data to create perturbations that maximize the loss and result in misclassification<d-cite key="goodfellow2015explaining"></d-cite>. Subsequent developments include the Projected Gradient Descent (PGD) approach<d-cite key="madry2018towards"></d-cite>. Other significant adversarial algorithms are the Jacobian-based Saliency Map Attack (JSMA)<d-cite key="papernot2017practical"></d-cite>, Papernot's attack<d-cite key="papernot2016limitations"></d-cite>, and the Carlini and Wagner (C&W) attack<d-cite key="carlini2017towards"></d-cite>, all contributing to the evolving adversarial machine learning. 


{% include figure.html path="assets/img/2025-04-28-multi-resolution-training-improves-robustness-against-adversarial-attacks/image1.jpg" class="img-fluid" %}

Despite the proliferation of these attack methods, defense and mitigation strategies, particularly for traffic sign classification tasks, have received significantly less attention. The common approach to enhancing the robustness of DNNs is adversarial training, which incorporates adversarial attack samples into the training dataset<d-cite key="zhang2022adversarial"></d-cite>. Training a model with only one or a limited range of adversarial examples leaves it susceptible to other types of attacks, necessitating the inclusion of diverse adversarial examples during training. Consequently, this approach is not universal and can demand significant training effort.

## Multi-resolution training

We propose a novel multi-resolution training approach by enhancing DNN architectures with integration of an additional CNN block prior to the main network. This block processes traffic sign images by first downsampling them to a lower resolution through decimation and then upsampling them back to the original resolution via interpolation. The block outputs either a 3-channel RGB image directly fed into the network or a 6-channel output obtained by concatenating the processed image with the original. The CNN block utilizes layers with filters designed using various downsampling techniques, such as low-pass and Gaussian filtering. This approach effectively reduces the impact of subtle, high-frequency adversarial perturbations while preserving the essential features of traffic signs, thereby improving the robustness of DNNs against adversarial attacks.


{% include figure.html path="assets/img/2025-04-28-multi-resolution-training-improves-robustness-against-adversarial-attacks/image2.jpg" class="img-fluid" %}

### Multi-resolution CNN block
The multi-resolution image process is implemented through a custom-designed CNN block, which includes a downsampling CNN layer with multiple filter options, followed by a standard bilinear interpolation layer for upsampling.

The **first option** utilizes a simple 1D low-pass filter defined as $$h_\text{lp} = [\frac{1}{4}, \frac{1}{2}, \frac{1}{4}]$$, commonly used for its simplicity and effectiveness in smoothing operations, approximating a half-band low-pass filter. The corresponding 2D filter used in the `LP_conv` layer is computed as:

$$ h_\text{lp-2D}[n_\text{1}, n_\text{2}] = h_\text{lp} \cdot h_\text{lp}^T, $$

resulting in a $$3 \times 3$$ kernel:

$$
h_\text{lp-2D} =
\begin{bmatrix}
\frac{1}{16} & \frac{1}{8} & \frac{1}{16} \\
\frac{1}{8} & \frac{1}{4} & \frac{1}{8} \\
\frac{1}{16} & \frac{1}{8} & \frac{1}{16}
\end{bmatrix}.
$$

This 2D filter effectively smooths input image data, making it ideal for downsampling in the multi-resolution framework.

The **second option** is a Gaussian filter, which offers more customization and smoother outputs. The normalized 1D Gaussian filter is defined as:

$$
h_\text{g}[n] = \frac{\exp\left(-0.5 \left(\frac{n}{\sigma}\right)^2\right)}{\sum_{n} \exp\left(-0.5 \left(\frac{n}{\sigma}\right)^2\right)}
$$

where $$n$$ denotes the indices, represented as $ (-\frac{z-1}{2}, \ldots, 0, \ldots, \frac{z-1}{2}) $. $$z$$ is the filter size, and $${\sigma}$$ is the standard deviation, which controls the spread of the filter. The resulting 2D Gaussian filter applied in the `Gaussian_conv` layer is computed as:

$$ h_\text{g-2D}[n_\text{1}, n_\text{2}] = h_\text{g} \cdot h_\text{g}^T $$

By adjusting the kernel size $$z$$ and the standard deviation $${\sigma}$$, the `Gaussian_conv` layer can achieve varying levels of smoothing, offering enhanced flexibility. This design ensures that the multi-resolution training framework adapts to different filtering needs while maintaining computational efficiency.

{% include figure.html path="assets/img/2025-04-28-multi-resolution-training-improves-robustness-against-adversarial-attacks/image3.jpg" class="img-fluid" %}
The diagram highlights the image processing using `LP_conv` and `Gaussian_conv` layers. 

- **LP_conv Layer**:  
   - Utilize downsampling and upsampling to eliminate fine details, including noise, while retaining key structures.  

- **Gaussian_conv Layer**:  
   - Kernel Size: Smaller sizes are computationally efficient, reducing localized noise while preserving edges and fine details. Larger sizes provide broader smoothing, targeting noise over larger areas but with potential detail loss.  
   - Sigma (𝜎): Lower values minimize smoothing, maintaining sharp edges. Higher values deliver more aggressive noise reduction, suitable for removing small noise but may blur finer details.
     
Both methods show effectiveness for removing small noise while preserving essential image features. The Gaussian filter, in particular, allows for targeted noise removal, offering a balance between computational efficiency and detail preservation.

## Experiment
We integrate the designed CNN block into various DNN architectures, including [ResNet18](https://pytorch.org/vision/main/models/generated/torchvision.models.resnet18.html), [MobileNetV2](https://pytorch.org/vision/main/models/generated/torchvision.models.mobilenet_v2.html), and [VGG16](https://pytorch.org/vision/main/models/generated/torchvision.models.vgg16.html). These models are trained on the widely used German Traffic Sign Recognition Benchmark [(GTSRB) dataset](https://pytorch.org/vision/0.17/generated/torchvision.datasets.GTSRB.html), which contains 43 classes of traffic signs, split into 39,209 training images and 12,630 test images. To evaluate the robustness of the trained models, we test them using the FGSM attack and a black-patch attack.

### Adversarial Attacks

- **FGSM attack** 

FGSM generates adversarial examples by slightly perturbing the input data in a way that maximizes the model's prediction error while keeping the perturbation imperceptible to humans. The attack works by exploiting the gradients of the loss function with respect to the input data. By taking a step in the direction of the gradient's sign, the attack aims to increase the loss and mislead the model into making incorrect predictions. The FGSM attack can be expressed mathematically as:

$$
x_{\text{adv}} = x + \epsilon \cdot \text{sign}(\nabla_x J(\theta, x, y))
$$

The adversarial example $$x_{\text{adv}}$$ is generated from the original input $$x$$ using a perturbation factor $$\epsilon$$ to control the level of adversarial noise. The sign of the gradient of the loss $$J$$ (calculated with respect to $$x$$) indicates the direction of modification. $$\theta$$ is the model paramter and $$y$$ indicates the true label. FSGM is considered as a white-box attack as it requires the prior knowdlege of the neural entwork structure and parameters. 

- **Black-Patch Attack**

In addition to the FGSM attack, we apply a black-patch attack that does not require knowledge of the DNN structure. This attack randomly places small black patches on images to simulate both real-world physical attacks (e.g., patches manually applied to traffic signs or cameras) and digital attacks (e.g., patches added to input data).

In the experiments, we used FGSM with $$\epsilon$$ values of 0.01, 0.05, 0.1, and 0.2. For the black-patch attack, we used black patches of size $$3 \times 3$$ pixels and varied the number of patches to 2, 4, 6, and 8 to simulate different levels of perturbations.

{% include figure.html path="assets/img/2025-04-28-multi-resolution-training-improves-robustness-against-adversarial-attacks/image4.jpg" class="img-fluid" %}
_Example of FGSM and black-patch perturbations_

### Results
We developed various models by integrating different combinations of our proposed CNN blocks into the base architectures of ResNet18, MobileNetV2, and VGG16. These models were evaluated based on their classification accuracy on the given test dataset, as illustrated in the plots below. 


{% include figure.html path="assets/img/2025-04-28-multi-resolution-training-improves-robustness-against-adversarial-attacks/image5.jpg" class="img-fluid"%}



The prefixes and suffixes in the model names indicate specific modifications:

- **Prefixes:**
  - **lpf_**: Models integrated with the designed `LPF_conv` block.
  - **gs_**: Models integrated with the designed `Gaussian_conv` block using a filter size of $$3 \times 3$$ and $$\sigma$$ of 0.5.
  - **gm_**: Models integrated with the designed `Gaussian_conv` block using a filter size of $$5 \times 5$$ and $$\sigma$$ of 1.5.
  - **gl_**: Models integrated with the designed `Gaussian_conv` block using a filter size of $$7 \times 7$$ and $$\sigma$$ of 2.5.

- **Suffixes:**
  - **_c3**: The CNN block outputs a processed 3-channel image, which is then fed into the main DNNs for processing.
  - **_c6**: The CNN block concatenates the processed 3-channel image with the original image to create a 6-channel input for the main DNNs (the DNNs are adjusted to accept a 6-channel input for compatibility).



### Key Insights

 **Effectiveness of Enhancements**
 
   The enhanced models consistently outperform their respective baselines  (ResNet18, MobileNetV2, and VGG16) under both FGSM and black box attacks, demonstrating the overall effectiveness of the proposed LPF and Gaussian blocks in improving model robustness.

**Impact of c6 vs. c3 Configurations**

   The plots reveal that, in general, the c3 configuration outperforms the c6 configuration, indicating that directly processing the 3-channel input through the designed CNN block and feeding it into the main DNNs is more effective than concatenating it with the original image. This trend suggests that the standalone processed features provide sufficient robustness without requiring additional raw feature information.

**Performance Against Adversarial Attacks**

   
As adversarial perturbations intensify, such as with a higher ($$\epsilon$$) in the FGSM or an increase in patches during black box attacks, the robustness of enhanced models typically diminishes. However, models equipped with **gl** blocks and **c3** configurations still manage to outperform baseline models in terms of accuracy under both FGSM and black box scenarios. Nonetheless, in specific instances illustrated in figures (d) and (e), this combination shows a quicker decline in performance as perturbations grow. Despite this, alternative methods proposed show greater robustness compared to the baselines, indicating that these enhanced models' ability to effectively mitigate the impact of both gradient-based and localized perturbations.


## Conclusion
DNNs are crucial for tasks like traffic sign recognition but remain vulnerable to adversarial attacks, posing significant challenges for their safe deployment. This blog post introduced a multi-resolution training approach that enhances model robustness by the integration of custom LPF and Gaussian CNN blocks. Our method demonstrated consistent improvements in robustness across different architectures, including ResNet18, MobileNetV2, and VGG16. These findings underscore the potential of multi-resolution training to mitigate adversarial risks, paving the way for safer and more reliable applications of DNNs in real-world scenarios.
