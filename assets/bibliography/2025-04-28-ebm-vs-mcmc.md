---
layout: distill
title: "Bootstrapped Energy Based Models: What are they good for?"
description: Training a generative model with energy or unnormalized density functions is considered an important problem for physical systems such as molecules. This provides a path to train generative models to sample from the much desired Boltzmann distribution in situations of data scarcity. As of late, several generative frameworks have been proposed to target this problem. However, as we show in the following blog post, these methods have not been benchmarked sufficiently well against traditional Markov Chain Monte Carlo (MCMC) methods that are used to sample from energy functions. We take the example of two recent methods (IDEM and IEFM) and show that MCMC outperforms both methods in terms of number of energy evaluations and wall clock time on established baselines. With this, we suggest a “course correction” on the benchmarking of these models and comment on the utility and potential of generative models on these tasks. 
date: 2025-04-28
future: true
htmlwidgets: true

# anonymize when submitting 
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
bibliography: 2025-04-28-ebm-vs-mcmc.bib  

# Add a table of contents to your post.
#   - make sure that TOC names match the actual section names
#     for hyperlinks within the post to work correctly.
toc:
  - name: Introduction 
  - name: "Same Goals, Different Paths: Comparing iEFM and iDEM"
  - name: Benchmark Systems
  - name: Evaluation Method
  - name: Random sampling and reweighting is sufficient for GMMs
  - name: Establishing new baselines for DW4, LJ13 and LJ55 with very long simulations
  - name: 4-particle Double-Well Potential
  - name: 13-Particle Lennard Jones System
  - name: 55-Particle Lennard Jones System
  - name: Closing thoughts - the role that these models have to play
---

## Introduction 

AI for structural biology has a big-data problem. Simply put, there is a lack of sufficiently large datasets to train and deploy models that can generalize across diverse molecular systems and capture the full complexity of biomolecular interactions. Particularly in structure-based drug discovery, where exploring the equilibrium conformational ensemble is inherently challenging, the structural data in the Protein Data Bank (PDB) remains limited. Molecular dynamics (MD) simulations, while valuable, are constrained by sampling inefficiencies and the need for prohibitively long timescales. Addressing these limitations is essential to designing drugs that effectively modulate a target's biophysical processes. Luckily, for physical systems such as molecules we have access to more information than just data to train generative models:

<!-- {% twitter https://x.com/adrian_roitberg/status/1793676191620018398 %} -->

Yes that’s right, we know that for physical systems (molecules), the distribution of states (conformers) are characterized by their energy according the boltzmann distribution $$p(x) \propto exp(-\beta (E(x)))$$, where E(x) is the energy of the state and $$\beta$$ is a constant dependant on temperature. This motivates a paradigm of training generative models that can take advantage of the energy function. 

<div class="row mt-3">
    <div class="col-sm mt-3 mt-md-0">
        {% include figure.html path="assets/img/2025-04-28-ebm-vs-mcmc/ezgif-6-054ae5e4e4.gif" class="img-fluid" %}
    </div>
</div>
<div class="caption">
    Molecules move. Most generative molecular models are trained on static, low-energy structures from the Protein Data Bank (white), but understanding molecular function requires sampling the different conformations (shapes) a molecule can take (green - frames from a molecular dynamics simulation).
</div>

Recently, a surge of deep learning generative models has aimed to address the challenges of sampling and data scarcity by adopting innovative, data-free approaches <d-cite key="sadegh2024idem"></d-cite>. These methods, which one may describe 'self-generative,' leverage a bootstrap procedure whereby the models generate their own data and rely on access to the energy function, to refine their predictions. This paradigm is particularly exciting because it bypasses the traditional reliance on large datasets, making it a promising solution to long-standing barriers in the field. For this assessment, we focus on two such methods: iterative energy-based flow matching (iEFM) and iterative denoising energy matching (iDEM) that have shown state of the art performance on several toy physical systems <d-cite key="woo2024iefm"></d-cite> .

A key consideration to take into account with generative models is to check if they outperform traditional methods for sampling from unnormalized density (energy) functions like MCMC. In this work, we compare IDEM and IEFM to MCMC on the same physical systems they were tested on and show that MCMC outperforms both methods while taking the same number of queries from the energy function. With this result, we suggest a “course correction” on the benchmarking of these models and propose different avenues where the development of these generative models would be useful.

## Same Goals, Different Paths: Comparing iEFM and iDEM

<div class="row mt-3">
    <div class="col-sm mt-3 mt-md-0">
        {% include figure.html path="assets/img/2025-04-28-ebm-vs-mcmc/dem2_cut-1.png" class="img-fluid" %}
    </div>
</div>

iEFM and iDEM are quite similar in their implementation. Both approximate a target vector field via self-normalized importance weights calculated by Monte Carlo estimation. In the case of iEFM, the target vector is a weighted average of the analytical vector field derived from the Flow Matching theory. Whereas in iDEM, the target score field is computed as a weighted average of the gradients of the energy function.


$$\begin{aligned}
u_t(x) &= \frac{\int u_t(x|x_1)p_t(x|x_1)p_1(x_1)dx_1}{\int p_t(x|x_1)p(x_1)dx_1} \\
&= \frac{\int u_t(x|x_1)q(x_1;x,t)p_1(x_1)dx_1}{\int q(x_1;x,t)p(x_1)dx_1} \\
&= \frac{\mathbb{E}_{x_1 \sim q(x_1;x,t)}[u_t(x|x_1)\exp(-\mathcal{E}(x_1|t))]}{\mathbb{E}_{x_1 \sim q(x_1;x,t)}[\exp(-\mathcal{E}(x_1|t))]}
\end{aligned}
\quad
\begin{aligned}
\nabla \log p_t(x_t) &= \frac{((\nabla p_0) * \mathcal{N}(0, \sigma_t^2))(x_t)}{p_t(x_t)} \\
&= \frac{\mathbb{E}_{x_0|t \sim \mathcal{N}(x_t, \sigma_t^2)}[\nabla p_0(x_0|t)]}{\mathbb{E}_{x_0|t \sim \mathcal{N}(x_t, \sigma_t^2)}[p_0(x_0|t)]} \\
&= \frac{\mathbb{E}_{x_0|t \sim \mathcal{N}(x_t, \sigma_t^2)}[\nabla \exp(-\mathcal{E}(x_0|t))]}{\mathbb{E}_{x_0|t \sim \mathcal{N}(x_t, \sigma_t^2)}[\exp(-\mathcal{E}(x_0|t))]}
\end{aligned}$$

In an iterative process, both methods begin by initializing the replay buffer with samples $$x\_T$$ drawn from an isotropic Gaussian distribution. Subsequently, intermediate samples $x\_t$ are generated by sampling from the conditional distribution $$P\_t(x\_t \\mid x\_T)$$, defined according to the specific theoretical framework of each method. The target vector field is then estimated using Monte Carlo estimation, where Boltzmann-weighted averaging is performed over samples $$x\_{T \\mid t}$$, effectively obtained by introducing additional noise to $$x\_t$$. By regressing onto these estimated targets in an inner loop and periodically generating new samples to replenish the buffer in an outer loop, these methods iteratively bootstrap toward sampling the full equilibrium distribution dictated by the energy function.

Developers of both methods, as well as others, have gravitated toward a standard set of toy systems for evaluation and benchmarking. While these systems are useful for early-stage development and illustrative purposes, we caution against over-reliance on them for robust benchmarking. As we will demonstrate, many related works employ MCMC samples of the equilibrium distribution but fail to use MCMC as a meaningful benchmark. In fact, MCMC often produces superior samples with faster runtimes on these systems compared to the proposed methods. Additionally, we show that the metrics commonly used for evaluation are poorly aligned with actual sample quality, further undermining their utility in assessing these methods.

MCMC and Sequential monte carlo (1-2 paragraphs and algorithm block)

To generate samples for the test systems with MCMC, we follow a 2 step procedure. First we generate samples from a simple prior and try to quickly “equilibriate” them to the distribution of the desired density function through a short sequential monte carlo (SMC) run, which is then followed on by regular MCMC for the remaining number of queries from the energy function.

MCMC steps are implemented as simple metropolis monte carlo steps with a symmetric gaussian proposal. The step sizes for the MCMC chains are decided by trying to maximize the value $\alpha*s$, where $\alpha$ is the acceptance rate and $s$ is the standard deviation of the gaussian proposal function.

SMC is implemented by creating intermediates between the prior ($p_o$) and desired density function ($$p_T$$) so that samples go through a short MCMC chain ($M$ steps) before they are reweighted and resampled to represent the distribution of the next density function. The intermediates are created through a simple linear interpolation between the unnormalized density functions. The detailed algorithm for the same is shown in the algorithm block below.



<!-- **_Algorithm 1_**: SMC Sampler for Reequilibration
SMC-PROPOSAL $(P_0, P_1,P_i, P_T), \text{MC steps}: \text{number of chains N}$
1. Sample $\{x_0^{(i)}\}^N_{i=1} \sim P_0$
2. For $t = 1, \ldots, T$ do:
   - Compute importance weights $\{{w_t^{(i)}}\}^N_{i=1}$ as: $ \{\frac{P_t(x_{t-1})}{P_{t-1}(x_{t})}\}_{i=t}^N $
   - Sample $\{{x_t^{(i)}}\}^N_{i=1}$ with replacement from $\{{x_{t-1}}^{(i)}\}^N_{i=1}$ with weights $\{{w_t^{(i)}}\}^N_{i=1}$
   - For $m = 1, \ldots, M$ do:
     > Parallel MCMC step with density $P_T$ to get new $\{{x_t^{(i)}}\}^N_{i=1}$
     > Return $\{x_1^{(i)}\}^N_{i=1}$ -->

<div class="row mt-3">
    <div class="col-sm mt-3 mt-md-0">
        {% include figure.html path="assets/img/2025-04-28-ebm-vs-mcmc/image.png" class="img-fluid" %}
    </div>
</div>

## Benchmark systems

<div class="row mt-3">
    <div class="col-sm mt-3 mt-md-0">
        {% include figure.html path="assets/img/2025-04-28-ebm-vs-mcmc/E_d_DW.png" class="img-fluid" %}
    </div>
    <div class="col-sm mt-3 mt-md-0">
        {% include figure.html path="assets/img/2025-04-28-ebm-vs-mcmc/E_d_LJ.png" class="img-fluid" %}
    </div>
</div>
<div class="caption">
    Caption.
</div>

Original work on iDEM introduced a series of four increasingly complex benchmark systems. The simplest is a 2D Gaussian Mixture Model (GMM) comprising 40 equally-weighted Gaussians with identical covariance matrices and means uniformly distributed across a \[-40, 40\] box, serving as a basic test for mode coverage and sampling quality. The Double Well 4-particle system (DW-4) introduces greater complexity by modeling four particles in 2D space interacting through a pairwise energy function with SE(3) × S4 symmetry properties, incorporating rotation, translation, and particle permutation invariance.

The most challenging benchmarks are two Lennard-Jones (LJ) systems that model interatomic interactions through a combination of attractive and repulsive potentials, supplemented by a harmonic oscillator term that tethers particles to their center of mass. The LJ-13 system models 13 particles in 3D space (39 dimensions), while the LJ-55 system handles 55 particles (165 dimensions), both using standardized parameters rm \= 1, τ \= 1, ϵ \= 1, and oscillator scale c \= 0.5. These systems are particularly demanding due to their high dimensionality and the explosive nature of their score function as interatomic distances approach zero.

Ground truth data for each system was generated using the following procedures:

* (GMM) The test set consists of 1000 samples generated directly from the known mixture distribution using TORCH.RANDOM.SEED(0). Since this is a well-defined mixture of Gaussians, true samples can be generated exactly without approximation.  
* (DW4) The ground truth data comes from MCMC samples provided by Klein et al. (2023b). The paper explicitly notes that while this might not be perfect ground truth, it's considered reasonable for evaluation purposes. They specifically mention avoiding using an older dataset from Garcia Satorras et al. (2021) because it was biased, being generated from a single MCMC chain.  
* (LJ) Like DW-4, these systems use MCMC samples from Klein et al. (2023b) as ground truth. The paper again notes they deliberately avoided using earlier datasets from Garcia Satorras et al. (2021) because these were biased, being generated from single MCMC chains and using only half of the energy function (ELJ/2, as the sum was only calculated for i \< j).

A key point to note is that for the physical systems (DW-4, LJ-13, LJ-55), the ground truth data relies on MCMC sampling since exact sampling is intractable. Only the GMM system has true exact samples since its distribution is explicitly defined.

## Experiment Details

5 models each of IDEM were trained for all the 4 systems: GMM, DW4, LJ13, and LJ55. The IDEM models were trained using the default configs provided on their github repository: [https://github.com/jarridrb/DEM](https://github.com/jarridrb/DEM). We only made one change so that our evaluations were consistent across the board. We ensured the evaluation batch size is equal to a 1000 for all the systems. 

iEFM was reimplemented to align as closely as possible with the algorithm described in the original manuscript. However, the absence of detailed experimental procedures and a publicly available codebase rendered a completely faithful reproduction unattainable. For instance, only the Optimal Transport (OT) variant of iEFM is evaluated here, as we were unable to successfully train the Variance Exploding iEFM. OT-iEFM employs a different noise schedule for the conditional path probabilities compared to iDEM. Specifically, $$P_t(x_t \\mid x\_1)$$ uses a linear schedule $$\sigma(t) = \sqrt{1 - (1 - \sigma_{\min})*t}$$, while $$q(x_1; x_t)$$ employs $$\frac{\sigma(t)}{t}$$, where $$\frac{1}{t}$$ is clamped to avoid division by zero. A max value of 80.0 and 1.3 were used for the GMM and DW4 systems, respectively. By virtue of the optimal transport formulation, OT-iEFM produces straight path trajectories at test time, requiring significantly fewer integration steps than iDEM—30 steps compared to 1000\. Apart from these differences, the following hyperparameters were left unchanged to ensure consistency with iDEM: num\_estimator\_mc\_samples, num\_samples\_to\_generate\_per\_epoch, eval\_batch\_size, optimizer/lr. We also note that the original authors of iEFM did not report any results on the Lennard-Jones systems. In our experiments, we found that OT-iEFM was unable to train successfully on this system.

To run MCMC chains Klein et. al. identified the best value of the MCMC step size for the DW4 and LJ13 systems as 0.5 and 0.025. We found the best value for LJ55 to be 0.0075 and GMMs to be 1.25. Therefore, these are the step sizes we used to run MCMC chains on these four systems. To ensure that our MCMC chains had exactly the same number of energy evaluations as IDEM and IEFM, we ran batch\_size\*num\_mc\_estimates chains in parallel, where num\_mc\_estimates is the number of MC estimates the methods used per sample. We then ensured that the number of MC steps we took are equal to inner\_loop\_iterations\*epochs, where inner\_loop\_iterations is the number of iterations of training the models went through before testing. This resulted in 512,000 parallel chains for GMM, DW4 and LJ13 and 12,800 parallel chains for LJ55. We ran 5 sets of such parallel chains for each system. 

The initial samples for the chains on the DW4 and GMM systems were sampled from a uniform prior, and the initial samples for the LJ systems were sampled from an isotropic gaussian, SMC involved creating 5 intermediate distributions for the DW4 and LJ13 and running the intermediate chains for 1000 steps each. LJ55 required 10 intermediate distributions with 2000 MC steps per distribution. GMMs required no intermediate distributions and no intermediate MCMC steps, just random sampling from a uniform prior and reweighting samples based on boltzmann weights ($w(x) = \exp(-E(x))$) did the trick.

## Evaluation Method 

We evaluate all the methods in two ways: we track the value of metrics as a function of queries from the energy function and time, and also report the metrics on samples generated at the end of training/MCMC runs. All the metrics are computed on batches of 1000 samples. For all our metrics we ensure that the number of queries from the energy function for the MCMC runs exactly match that used by IDEM/IEFM. To compute the results from the samples generated at the end of the runs, we take 5 batches of 1000 samples from 5 independent runs, resulting in mean and std values for the metrics obtained from 25 batches in total. To obtain the compute time of the runs, we take the time required for 1 iteration of training/MCMC sampling and multiply that by the total number of training/MCMC steps. This way we avoid the extra compute time involved in validation steps.

We evaluate all the methods on 3 metrics: W2 distance, energy W2 distance and interatomic distance W2 distance. The W2 distance metric is the standard Wasserstein-2 distance using euclidean distance from sample position and used for all the systems. We compute the energy W2 and interatomic distance W2 for the other systems as the samples themselves are not visualizable and so we need further ways to obtain information of the distribution of samples. The energy W2 distance computes how similar the distributions of energy of the test samples and generated samples are. This provides a notion of how well the sampler is able to capture the ensemble observations of the equilibrium distribution. Finally, the interatomic W2 distance is a projection of the position of samples along 1 dimension and therefore should match very well if the generated samples are close to the distribution of the test set.

## Random sampling and reweighting is sufficient for GMMs

<div class="row mt-3">
    <div class="col-sm mt-3 mt-md-0">
        {% include figure.html path="assets/img/2025-04-28-ebm-vs-mcmc/W2_lineplots-1.png" class="img-fluid" %}
    </div>
    <div class="col-sm mt-3 mt-md-0">
        {% include figure.html path="assets/img/2025-04-28-ebm-vs-mcmc/W2_lineplots_time-1.png" class="img-fluid" %}
    </div>
</div>
<div class="caption">
    Caption.
</div>





For the GMM system we initialize our MC chains using a uniform distribution with limits of \[-45,45\]. Since we are generating 512,000 samples on this 2D surface, we find that it's sufficient to just reweight the samples generated from the uniform prior according to their boltzmann weights. We still continue to run MCMC chains after that for demonstration purposes. The figures above show that W2 scores of the runs as a function of number of training epochs and time. We define an “epoch” of a MCMC chain as the number of energy evaluations that are used to train an epoch of IDEM/IEFM.

| Method | W2 score |
| :---- | :---- |
| Ground truth  | 4.159 \+- 0.604 |
| IDEM | 7.698 \+- 1.442 |
| IEFM | 7.364 \+- 0.955 |
| MCMC | 4.219 \+- 0.569 |

The table above contains the mean and std deviation of the W2 metrics vs the test set with 25 batches of 1000 samples obtained from different methods. The “Ground Truth” are samples generated directly from the GMM, thereby showing what the ideal values of the W2 score is. We see that only the MCMC method is able to get a W2 score similar to that of the ground truth. We also see that samples generated from the MCMC chains also more closely represent the ground truth in the figure below. 

<div class="row mt-3">
    <div class="col-sm mt-3 mt-md-0">
        {% include figure.html path="assets/img/2025-04-28-ebm-vs-mcmc/true_gmm_samples.png" class="img-fluid" %}
    </div>
    <div class="col-sm mt-3 mt-md-0">
        {% include figure.html path="assets/img/2025-04-28-ebm-vs-mcmc/IDEM_gmm_samples.png" class="img-fluid" %}
    </div>
</div>
<div class="row mt-3">
    <div class="col-sm mt-3 mt-md-0">
        {% include figure.html path="assets/img/2025-04-28-ebm-vs-mcmc/MCMC_gmm_samples.png" class="img-fluid" %}
    </div>
    <div class="col-sm mt-3 mt-md-0">
        {% include figure.html path="assets/img/2025-04-28-ebm-vs-mcmc/IEFM_gmm_samples.png" class="img-fluid" %}
    </div>
</div>
<div class="caption">
    Caption.
</div>

The fact that this system is just in 2 dimensions makes sampling from the distribution very easy, Random sampling from a uniform prior and reweighting does the job, so it should not be considered a valid benchmark system for deep learning models for future works: 

## Establishing new baselines for DW4, LJ13 and LJ55 with very long simulations

For the DW4, LJ13 and LJ55 systems we don't have direct access to samples from the equilibrium distribution. Therefore, to create a strong “ground truth” baseline, we decided to run very long simulations on all of these systems and save the final state from all the simulations. For DW4 and LJ13, we ran 50,000 chains in parallel for a million steps, while for LJ55 we ran 12,800 chains for 2.5 million steps. We assume that the combined end states of all these chains represent a distribution very close to the equilibrium distribution and can also be used as a new test set to benchmark on. The W2 and energy W2 scores of the chains with respect to the reference test set, used by IDEM and IEFM, as a function of the number of MC steps are shown in the figures below. The convergence of all the runs on these metrics provides for further evidence that the samples are close to the equilibrium distributions.

<div class="row mt-3">
    <div class="col-sm mt-3 mt-md-0">
        {% include figure.html path="assets/img/2025-04-28-ebm-vs-mcmc/mcmc_dw4_w2-1.png" class="img-fluid" %}
    </div>
    <div class="col-sm mt-3 mt-md-0">
        {% include figure.html path="assets/img/2025-04-28-ebm-vs-mcmc/mcmc_dw4_energyw2-1.png" class="img-fluid" %}
    </div>
</div>
<div class="caption">
    DW4
</div>

<div class="row mt-3">
    <div class="col-sm mt-3 mt-md-0">
        {% include figure.html path="assets/img/2025-04-28-ebm-vs-mcmc/mcmc_lj13_w2-1.png" class="img-fluid" %}
    </div>
    <div class="col-sm mt-3 mt-md-0">
        {% include figure.html path="assets/img/2025-04-28-ebm-vs-mcmc/mcmc_lj13_energyw2-1.png" class="img-fluid" %}
    </div>
</div>
<div class="caption">
    LJ13
</div>

<div class="row mt-3">
    <div class="col-sm mt-3 mt-md-0">
        {% include figure.html path="assets/img/2025-04-28-ebm-vs-mcmc/mcmc_lj55_w2-1.png" class="img-fluid" %}
    </div>
    <div class="col-sm mt-3 mt-md-0">
        {% include figure.html path="assets/img/2025-04-28-ebm-vs-mcmc/mcmc_lj55_energyw2-1.png" class="img-fluid" %}
    </div>
</div>
<div class="caption">
    LJ55
</div>





## 4-particle Double-Well Potential

Examining the top two plots (Image 1), we observe W2 scores tracked over both epochs and wall-clock time. The MCMC method (blue line) stabilizes quickly around a W2 score of 2.1, exhibiting minimal variance, which reflects its robust and consistent sampling performance. In contrast, iDEM (orange line) displays more erratic behavior, with sharp dips and recoveries, though its average performance is similar to MCMC. Meanwhile, iEFM (green line) remains consistently above both methods with higher variance, indicating a lower and less reliable sampling quality. Notably, when plotted against wall-clock time, MCMC’s efficiency stands out, as it reaches and sustains optimal performance much faster than the neural methods.

The energy W2 distance plots (Images 2-3) further underscore the stark differences between the methods. MCMC rapidly converges to a low log energy W2 distance of approximately \-2 and maintains this stability throughout. iDEM, however, takes significantly longer to converge and settles at a higher value of around 0, highlighting its less accurate energy sampling. iEFM performs the worst, with log energy W2 distances hovering near 6, demonstrating fundamental challenges in capturing the system's energy landscape. The time-based comparison again highlights MCMC’s efficiency, achieving superior energy sampling far more quickly than its neural counterparts.

The quantitative results (Image 4\) and distribution plots (Image 5\) provide the most comprehensive insight into performance. Across all metrics, MCMC consistently achieves the best scores when compared to both old and new reference data. The distribution plots reveal key details: while iDEM captures the general shape of the interatomic distance distribution relatively well, iEFM exhibits clear deviations in peak heights and locations. The energy distribution plot paints an even clearer picture—MCMC aligns closely with the reference distribution, whereas iEFM shows substantial discrepancies, particularly in the higher energy regions.

These findings suggest that while neural methods like iDEM and iEFM represent innovative steps forward in molecular sampling, they remain outperformed by the reliability and efficiency of well-tuned MCMC methods. The data strongly indicate that the added complexity of neural approaches does not yet translate into practical advantages over traditional methods, particularly for these benchmark systems.

<div class="row mt-3">
    <div class="col-sm mt-3 mt-md-0">
        {% include figure.html path="assets/img/2025-04-28-ebm-vs-mcmc/dw4_lineplots_epoch-1.png" class="img-fluid" %}
    </div>
    <div class="col-sm mt-3 mt-md-0">
        {% include figure.html path="assets/img/2025-04-28-ebm-vs-mcmc/dw4_lineplots_time-1.png" class="img-fluid" %}
    </div>
</div>
<div class="caption">
    DW4 W2 Score
</div>

<div class="row mt-3">
    <div class="col-sm mt-3 mt-md-0">
        {% include figure.html path="assets/img/2025-04-28-ebm-vs-mcmc/dw4_energyw2_lineplots_epoch-1.png" class="img-fluid" %}
    </div>
    <div class="col-sm mt-3 mt-md-0">
        {% include figure.html path="assets/img/2025-04-28-ebm-vs-mcmc/dw4_energyw2_lineplots_time-1.png" class="img-fluid" %}
    </div>
</div>
<div class="caption">
    Dw4 Energy W2 Distance
</div>

|  | Old Reference (klein et. al) |  |  | New Reference (Long MCMC) |  |  |
| :---- | :---- | :---- | :---- | :---- | :---- | :---- |
| Method | W2 | Energy W2 | Interatomic distance w2 | W2 | Energy W2 | Interatomic distance w2 |
| Long MCMC | 2.080 \+- 0.030 | 0.157 \+- 0.045 | 0.073 \+- 0.012 | 1.883 \+- 0.023 | 0.144 \+- 0.044 | 0.017 \+- 0.007 |
| IDEM | 2.054 \+- 0.027 | 0.435+- 0.153 | 0.139 \+- 0.017 | 1.912+-0.021 | 0.590+- 0.165 | 0.207+-0.014 |
| IEFM | 2.115+- 0.035 | 3.852+-0.959 | 0.214+-0.014 | 1.928+- 0.029 | 3.879+- 0.909 | 0.145+-0.011 |
| MCMC | 2.068 \+- 0.028 | 0.148 \+- 0.048 | 0.067 \+- 0.012 | 1.873 \+- 0.032 | 0.132 \+- 0.030 | 0.022 \+- 0.012 |

<div class="row mt-3">
    <div class="col-sm mt-3 mt-md-0">
        {% include figure.html path="assets/img/2025-04-28-ebm-vs-mcmc/dw4_lineplots_epoch-1.png" class="img-fluid" %}
    </div>
    <div class="col-sm mt-3 mt-md-0">
        {% include figure.html path="assets/img/2025-04-28-ebm-vs-mcmc/dw4_lineplots_time-1.png" class="img-fluid" %}
    </div>
</div>
<div class="caption">
    DW4 W2 Score
</div>
<div class="row mt-3">
    <div class="col-sm mt-3 mt-md-0">
        {% include figure.html path="assets/img/2025-04-28-ebm-vs-mcmc/dw4_lineplots_epoch-1.png" class="img-fluid" %}
    </div>
    <div class="col-sm mt-3 mt-md-0">
        {% include figure.html path="assets/img/2025-04-28-ebm-vs-mcmc/dw4_lineplots_time-1.png" class="img-fluid" %}
    </div>
</div>
<div class="caption">
    DW4 W2 Score
</div>

<div class="row mt-3">
    <div class="col-sm mt-3 mt-md-0">
        {% include figure.html path="assets/img/2025-04-28-ebm-vs-mcmc/dw4_Interatomic_distance_distribution.png" class="img-fluid" %}
    </div>
</div>
<div class="caption">
    DW4 Interatomic Distance Distribution 
</div>

<div class="row mt-3">
    <div class="col-sm mt-3 mt-md-0">
        {% include figure.html path="assets/img/2025-04-28-ebm-vs-mcmc/dw4_Energy_distribution.png" class="img-fluid" %}
    </div>
</div>
<div class="caption">
    DW4 Enegy Distribution 
</div>



## 13-Particle Lennard Jones System

On the LJ13 system we observe that the MCMC runs actually get a higher W2 score than the IDEM runs. However, we posit that for higher dimensional systems, W2 between positions of data samples is not a good measure for the distance between two distributions. This is further evidenced by the stark difference in energy W2 where the MCMC chains do much better than IDEM. 

In Table 3, we also compare both the methods to the very long MCMC chain that has been run for a million steps. Working under the assumption that the long MCMC chain is close to convergence, we can consider the metric values generated by it as the best attainable values on this system. Therefore, we can consider the best methods to be the ones that have the closest metric values to that provided by the long MC chain. From the table, it is visible that the MCMC method does best in this regard. Furthermore, MCMC has much better energy and interatomic distance w2 scores than IDEM on both reference sets indicating a better predictability of ensemble observations.

Finally, in the histogram plots, it's clear that the IDEM energy plot is right shifted as compared to both the reference sets that have good agreement among themselves. Considering all the metrics and distributions, we conclude that MCMC outperforms IDEM on this system.  

<div class="row mt-3">
    <div class="col-sm mt-3 mt-md-0">
        {% include figure.html path="assets/img/2025-04-28-ebm-vs-mcmc/lj13_w2_lineplots_epoch-1.png" class="img-fluid" %}
    </div>
    <div class="col-sm mt-3 mt-md-0">
        {% include figure.html path="assets/img/2025-04-28-ebm-vs-mcmc/lj13_w2_lineplots_time-1.png" class="img-fluid" %}
    </div>
</div>
<div class="caption">
    LJ13 W2 Score
</div>

<div class="row mt-3">
    <div class="col-sm mt-3 mt-md-0">
        {% include figure.html path="assets/img/2025-04-28-ebm-vs-mcmc/lj13_energyw2_lineplots_epoch-1.png" class="img-fluid" %}
    </div>
    <div class="col-sm mt-3 mt-md-0">
        {% include figure.html path="assets/img/2025-04-28-ebm-vs-mcmc/lj13_energyw2_lineplots_time-1.png" class="img-fluid" %}
    </div>
</div>
<div class="caption">
    LJ13 Energy W2 Distance
</div>


|  | Old Reference (klein et. al) |  |  | New Reference (Long MCMC) |  |  |
| :---- | :---- | :---- | :---- | :---- | :---- | :---- |
| Method | W2 | Energy W2 | Interatomic distance w2 | W2 | Energy W2 | Interatomic distance w2 |
| Long MCMC | 4.312 \+- 0.007 | 0.457+-0.120 | 0.004+-0.002 | 4.298 \+- 0.007 | 0.544 \+- 0.167 | 0.005+- 0.002 |
| IDEM | 4.266 \+- 0.007 | 5.204 \+- 1.788 | 0.024+-0.002 | 4.253+- 0.005 | 4.73+- 1.928 | 0.028+-0.003 |
| MCMC | 4.312 \+- 0.005 | 0.516+- 0.119 | 0.003+-0.001 | 4.297 \+- 0.009 | 0.531+- 0.171 | 0.006 \+- 0.002 |


<div class="row mt-3">
    <div class="col-sm mt-3 mt-md-0">
        {% include figure.html path="assets/img/2025-04-28-ebm-vs-mcmc/lj13_interatomic_distance_distribution.png" class="img-fluid" %}
    </div>
</div>
<div class="caption">
    DW4 Interatomic Distance Distribution 
</div>

<div class="row mt-3">
    <div class="col-sm mt-3 mt-md-0">
        {% include figure.html path="assets/img/2025-04-28-ebm-vs-mcmc/lj13_energy_distribution.png" class="img-fluid" %}
    </div>
</div>
<div class="caption">
    DW4 Enegy Distribution 
</div>

## 55-Particle Lennard Jones System

At the time of submitting this blog for review we do not have completed IDEM training runs on the LJ55 systems as we did not anticipate it would take 2.5 days on an L40 GPU to train a model on this system. However, we provide the W2 and Energy W2 metric values plotted against a 1000 epochs and time in the figures below. Both the figures indicate similar trends as seen in previous systems with MCMC outperforming IDEM on relevant metrics and being closer to the ground truth distribution. We will add the values obtained at the end of the training runs and histogram plots to the blog once those training runs have completed.  

<div class="row mt-3">
    <div class="col-sm mt-3 mt-md-0">
        {% include figure.html path="assets/img/2025-04-28-ebm-vs-mcmc/lj55_w2_lineplots-1.png" class="img-fluid" %}
    </div>
</div>
<div class="caption">
    LJ55 W2 Score
</div>

|  | Old Reference (klein et. al) |  |  | New Reference (Long MCMC) |  |  |
| :---- | :---- | :---- | :---- | :---- | :---- | :---- |
| Method | W2 | Energy W2 | Interatomic distance w2 | W2 | Energy W2 | Interatomic distance w2 |
| Long MCMC | 15.923 \+- 0.009 | 0.931 \+- 0.243 | 0.001+- 0.0006 | 15.779 \+- 0.007 | 0.680+-0.211 | 0.001+- 0.0005 |
| MCMC | 16.48 \+- 0.0497 | 0.872 \+- 0.237 | 0.001 \+- 0.0005 | 16.42 \+- 0.026 | 0.907 \+- 0.391 | 0.002+- 0.0009 |


## Closing thoughts \- the role that these models have to play

Our analysis offers a sobering assessment of the current state of neural network-based samplers for molecular systems. Despite the growing interest in methods like iDEM and iEFM, our results show that these approaches fail to outperform well-tuned MCMC methods in terms of both sampling quality and computational efficiency. This finding is particularly significant because, for neural samplers to justify their additional complexity and computational overhead, they must demonstrate clear and consistent advantages over traditional methods—a benchmark that these approaches have yet to meet.

However, this critique should not be viewed as a wholesale rejection of neural approaches in molecular sampling. Rather, it suggests a needed shift in research focus. One promising direction lies in leveraging neural networks' capacity for transfer learning. While MCMC kernels are often system-specific, requiring careful tuning for each new molecular system, neural networks could potentially learn generalizable sampling strategies that transfer across different molecular systems. This transferability could provide a compelling advantage over traditional MCMC methods, even if the absolute sampling efficiency for any single system remains lower.

A critical limitation of current neural approaches like IEFM and IDEM lies in their reliance on weighted averages of noisy samples. This becomes particularly problematic when dealing with real molecular systems, where energy landscapes can be extremely steep and sampling noise can lead to explosive gradients. Future work might address this by reconsidering how noise is incorporated into these models \- for instance, by defining noise on a Riemannian manifold that respects the geometric constraints of molecular systems. Such modifications could help stabilize training and improve sampling efficiency while maintaining the potential benefits of neural approaches.

