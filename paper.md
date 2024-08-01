---
title: 'ISL: A Julia package for training Implicit Generative Models via an Invariant Statistical Loss'
tags:
  - Julia
  - Flux
  - Deep Learning
  - Statistics
authors:
  - name: José Manuel de Frutos Porras
    orcid: 0009-0002-5251-0515
    corresponding: true # (This is how to denote the corresponding author)
    affiliation: "1" # (Multiple affiliations must be quoted)
affiliations:
 - name: Universidad Carlos III de Madrid, Spain
   index: 1
date: 01 August 2024
bibliography: paper.bib
---

# Summary

Implicit generative models have the capability to learn arbitrary complex data distributions. On the downside, training requires telling apart real data from artificially-generated ones using adversarial discriminators, leading to unstable training and mode-dropping issues. As reported by Zahee et al. (2017), even in the one-dimensional (1D) case, training a generative adversarial network (GAN) is challenging and often suboptimal. In this work, we develop a discriminator-free method for training one-dimensional (1D) generative implicit models and subsequently expand this method to accommodate multivariate cases. Our loss function is a discrepancy measure between a suitably chosen transformation of the model samples and a uniform distribution; hence, it is invariant with respect to the true distribution of the data. We first formulate our method for 1D random variables, providing an effective solution for approximate reparameterization of arbitrary complex distributions. Then, we consider the temporal setting (both univariate and multivariate), in which we model the conditional distribution of each sample given the history of the process.

# Statement of need

`ISL` is a Julia package specifically designed for training Implicit Generative Models using the innovative Invariant Statistical Loss. Leveraging the Flux framework, `ISL` enables the implementation of advanced machine learning capabilities, balancing speed with a flexible, user-friendly interface. The API is crafted to simplify common operations such as training 1D models, univariate time series, and multivariate time series.

In addition to its core functionalities, `ISL` offers a suite of utility functions, including support for generative adversarial networks and tools for time series analysis and generation. This makes `ISL` a valuable resource for both machine learning researchers and data scientists/software developers who seek to train their models with this novel approach.

`ISL` has already contributed in a scientific publications [de2024training], underscoring its utility and impact in the field. Its combination of speed, thoughtful design, and robust machine learning functionalities for Implicit Generative Models positions `ISL` as a powerful tool for advancing scientific research and practical applications alike in the area of Implicit Generative Models.

# Methods

Implicit generative models employ an $m$-dimensional latent random variable (r.v.) $\mathbf{z}$ to simulate random samples from a prescribed $n$-dimensional target probability distribution. To be precise, the latent variable undergoes a transformation through a deterministic function $g_{\theta}$, which maps $\mathbb{R}^m \mapsto \mathbb{R}^n$ using the parameter set $\theta$. Given the model capability to generate samples with ease, various techniques can be employed for contrasting two sample collections: one originating from the genuine data distribution and the other from the \jm{model} distribution. This approach essentially constitutes a methodology for \jm{the approximation of probability distributions} via comparison.
Generative adversarial networks (GANs) [goodfellow2014generative], $f$-GANs [nowozin2016f], Wasserstein-GANs (WGANs) [arjovsky2017wasserstein], adversarial variational Bayes (AVB) [mescheder2017adversarial], and \jm{maximum mean-miscrepancy} (MMD) GANs [li2017mmd] are some popular methods that fall within this framework. 

Approximation of 1-dimensional (1D) parametric distributions is a seemingly naive problem for which the above-mentioned models can perform below expectations. In [zaheer2017gan], the authors report that various types of GANs struggle to approximate relatively simple distributions from samples, emerging with MMD-GAN as the most promising technique. However, the latter implements a kernelized extension of a moment-matching criterion defined over a reproducing kernel Hilbert space, and consequently, the objective function is expensive to compute. 

In this work, we introduce a novel approach to train univariate implicit models that relies on a fundamental property of rank statistics. Let $r_1 < r_2 < \cdots < r_k$ be a ranked (ordered) sequence of independent and identically distributed (i.i.d.) samples from the generative model with probability density function (pdf) $tilde{p}$, and let $y$ be a random sample from a pdf $p$. If $\tilde{p} = p$, then $\mathbb{P}(r_{i-1} \leq y < r_{i}) = \frac{1}{K}$ for every $i = 1, \ldots, K+1$, with the convention that $r_0=-\infty$ and $r_{K+1}=\infty$ (see, e.g., [rosenblatt1952remarks] or [elvira2016adapting] for a short explicit proof). This invariant property holds for any continuously distributed data, i.e., for any data with a pdf $p$. Consequently, even if $p$ is unknown, we can leverage this invariance to construct an objective (loss) function.  This objective function eliminates the need for a discriminator, directly measuring the discrepancy of the transformed samples with respect to (w.r.t.) the uniform distribution. The computational cost of evaluating this loss increases linearly with both $K$ and $N$, allowing for low-complexity mini-batch updates. Moreover, the proposed criterion is invariant across true data distributions, hence we refer to the resulting objective function as invariant statistical loss (ISL). Because of this property, the ISL can be exploited to learn multiple modes in mixture models and different time steps when learning temporal processes. Additionally, considering the marginal distributions independently, it is straightforward to extend ISL to the multivariate case.


# Software Description

The `ISL` repository is organized into several directories that encapsulate different aspects of the project, ranging from the core source code and custom functionalities to examples demonstrating the application of the project's capabilities, as well as testing frameworks to ensure reliability.

## Source Code (`src/`)

- **`CustomLossFunction.jl`**: This file contains implementations of the ISL custom loss function tailored for the models developed within the repository.
  
- **`ISL.jl`**: Serves as the main module file of the repository, this file aggregates and exports the functionalities developed in `CustomLossFunction.jl`.

## Examples (`examples/`)

- **`time_series_predictions/`**: This subdirectory showcases how the ISL project's models can be applied to time series prediction tasks. 

- **`Learning1d_distribution/`**: Focuses on the task of learning 1D distributions with the ISL.

## Testing Framework (`test/`)

- **`runtests.jl`**: This script is responsible for running automated tests against the `ISL.jl` module.

# Figures

Figures can be included like this:
![Caption for example figure.\label{fig:example}](figure.png)
and referenced from text using \autoref{fig:example}.

Figure sizes can be customized by adding an optional second parameter:
![Caption for example figure.](figure.png){ width=20% }

# Documentation

Documentation is available at
(https://josemanuel22.github.io/ISL/dev/), where there are worked-out
examples and tutorials on how to use the package.

# Acknowledgements

This work has been supported by the the Office of Naval Research (award N00014-22-1-2647) and Spain's Agencia Estatal de Investigación (refs. PID2021-125159NB-I00 TYCHE and PID2021-123182OB-I00 EPiCENTER). Pablo M. Olmos also acknowledges the support by Comunidad de Madrid under grants IND2022/TIC-23550 and ELLIS Unit Madrid.

# References