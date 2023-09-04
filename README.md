# README

[![Documenter: stable](https://img.shields.io/badge/docs-dev-blue.svg)](https://josemanuel22.github.io/AdaptativeBlockLearning/dev/)


## Abstract

Generative models have been shown to possess the capability to learn the distribution of data, provided that the models have infinite capacity [Generative adversarial net](https://arxiv.org/pdf/1406.2661.pdf), [Wasserstein GAN](https://arxiv.org/pdf/1701.07875.pdf). Empirically, deep neural networks appear to possess 'sufficiently large' capacity and have demonstrated significant success in various applications such as image generation. 

Nevertheless, generative models like Generative Adversarial Networks (GANs) may exhibit limited generalization properties, wherein successful training may not necessarily align the trained distribution with the target distribution according to standard metrics [Generalization and Equilibrium in Generative Adversarial Nets (GANs)](https://arxiv.org/pdf/1703.00573.pdf), [Do GANs actually learn the distribution? An empirical study](https://arxiv.org/pdf/1706.08224.pdf). Even in seemingly straightforward scenarios, such as one-dimensional settings, training generative models like Generative Adversarial Networks can prove to be highly challenging. TThey frequently fail to accurately capture the true distribution and tend to discover only a subset of the modes [GAN Connoisseur: Can GANs Learn Simple 1D Parametric Distributions?](https://chunliangli.github.io/docs/dltp17gan.pdf). 

In this paper, we present a new learning method for one-dimensional settings, inspired by [adaptive particle filters](https://arxiv.org/pdf/1911.01383.pdf), which theoretically guarantees convergence to the true distribution with sufficient capacity. We demonstrate in practice that this new method yields promising results, successfully learning true distributions in a variety of scenarios and mitigating some of the well-known problems that classical generative models present.
