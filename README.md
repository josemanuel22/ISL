# README

[![Documenter: stable](https://img.shields.io/badge/docs-dev-blue.svg)](https://josemanuel22.github.io/AdaptativeBlockLearning/dev/) [![codecov](https://codecov.io/gh/josemanuel22/AdaptativeBlockLearning/graph/badge.svg?token=DDQPSJ9KWQ)](https://codecov.io/gh/josemanuel22/AdaptativeBlockLearning)


## Abstract

Implicit generative models have the capability to learn arbitrary complex data distributions. On the downside, training requires telling apart real data from artificially-generated ones using adversarial discriminators, leading to unstable training and 
%and this is known to lead to both unstable training, due to the inherent min-max optimization problem and 
mode-dropping issues. As reported by Zahee et al. (2017), even in the one-dimensional (1D) case, training a Generative Adversarial Network (GAN) is challenging and often suboptimal. In this work, we develop a  discriminator-free approach to training 1D generative implicit models. Our loss function is a discrepancy measure between a suitably chosen transformation of the model samples and a uniform distribution; hence, it is invariant with respect to the true distribution of the data. We first formulate our method for 1D random variables, providing an effective solution for approximate reparameterization of arbitrary complex distributions. Then, we consider the temporal setting. We model in this case, the conditional distribution of each sample given the history of the process. We demonstrate thought numerical simulations that this new method yields promising results, successfully learning true distributions in a variety of scenarios and mitigating some of the well-known problems that state-of-the-art implicit methods present.
