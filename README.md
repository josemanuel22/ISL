# README

[![Documenter: stable](https://img.shields.io/badge/docs-dev-blue.svg)](https://josemanuel22.github.io/AdaptativeBlockLearning/dev/)


## Abstract

Generative models have been shown to possess the capability to learn the distribution of data, provided that the models have infinite capacity [Generative adversarial net](https://arxiv.org/pdf/1406.2661.pdf), [Wasserstein GAN](https://arxiv.org/pdf/1701.07875.pdf). Empirically, deep neural networks appear to possess 'sufficiently large' capacity and have demonstrated significant success in various applications such as image generation. 

Nevertheless, generative models like Generative Adversarial Networks (GANs) may exhibit limited generalization properties, wherein successful training may not necessarily align the trained distribution with the target distribution according to standard metrics [Generalization and Equilibrium in Generative Adversarial Nets (GANs)](https://arxiv.org/pdf/1703.00573.pdf), [Do GANs actually learn the distribution? An empirical study](https://arxiv.org/pdf/1706.08224.pdf). Even in seemingly straightforward scenarios, such as one-dimensional settings, training generative models like Generative Adversarial Networks can prove to be highly challenging. They frequently fail to accurately capture the true distribution and tend to discover only a subset of the modes [GAN Connoisseur: Can GANs Learn Simple 1D Parametric Distributions?](https://chunliangli.github.io/docs/dltp17gan.pdf). 

In this paper, we present a new learning method for one-dimensional settings, inspired by [adaptive particle filters](https://arxiv.org/pdf/1911.01383.pdf), which theoretically guarantees convergence to the true distribution with sufficient capacity. We demonstrate in practice that this new method yields promising results, successfully learning true distributions in a variety of scenarios and mitigating some of the well-known problems that classical generative models present.

## Introduction

## Background

In particle filtering, the goal is to estimate filtering probabilities by dynamically adjusting the weight of the  particles based on the ground truth. One of the key parameters that significantly affects the convergence and complexity of these methods is the number of particles used. An approach employed in Adaptive Particle Filters is to evalute online certain predictive statistics which are invariant for a broad class of state-space models.  In this class of Monte Carlo algorithm, the number of samples is ajusted periodically. order to assess the convergence of the method, thus allowing for an increase or decrease in the number of particles propagated in subsequent steps of the algorithm. The evaluation of the predictive statistics that lies at the core of the methodology is done by generating fictitious observations, i.e., particles in the observation space.


### Boostrap filter (BF)

A PF is an algorithm that processes the observations {yt}t≥1 sequentially in order to compute Monte Carlo approximations of the sequence of probability measures {πt}t≥1. The simplest algorithm is the so-called bootstrap particle filter (BPF) [11] (see also [32]), which consists of a recursive importance sampling procedure and a resampling step. The term "particle" refers to a Monte Carlo sample in the state space X , which is assigned an importance weight. Below, we outline the BPF algorithm with M particles.


The goal is to evaluate the convergence of the BPF (namely, the accuracy of the approximation pM t (yt)) in real time and, based on the convergence assessment, adapt the computational effort of the algorithm, i.e., the number of used particles M. To that end, we run the BPF in the usual way with a light addition of computations. At each iteration we generate K "fictitious observations", denoted y˜
(1)
t ,..., y˜
(K)
t , from the
approximate predictive pdf pM
t (yt). If the BPF is operating with a small enough level of error, then Theorem 1 states that these fictitious observations come approximately from the same distribution as the acquired observation, i.e., μM
t (dyt) ≈
μt(dyt). In that case, as we explain in Subsection IV-B,
a statistic aK
t can be constructed using yt, y˜
(1)
t ,..., y˜
(K)
t ,
which necessarily has an (approximately) uniform distribution
independently of the specific form of the state-space model
(1)–(3). By collecting a sequence of such statistics, say
aK
t−W+1,...,aK
t for some window size W, one can easily test whether their empirical distribution is close to uniform using standard procedures. The better the approximation μM t ≈ μt generated by the BPF, the better fit with the uniform distribution can be expected.


### Block-adaptive selection of the number of particles.

We propose an algorithm that dynamically adjusts the
number of particles of the filter based on the transformed r.v.
AK,M,t. 




## Proposed Method

We propose a new learning method based on the previous result. As we know from theorem X convergence of the histogram to the uniform distribution garantees that we are sampling from the true distribution given that K and N are suffitienly large.

We aspire to extend this algorithm for the purpose of employing gradient descent. The concept revolves around converting this discrete schema into a continuous one.

Given a neural network which is going to learn the distribution from which the data are draw, we aim to recreate obtain the AK,M,t r.v. but in continous settings. For this porpuse, given a serie of K fictitious simulations generated by the neural network and given a observation y, we can count how many of those fictitious data are smaller than the observation, as follow,

\phi(y) = \sum_{i_1}^{K}σ(y_{i}, y) = \sum_{i_1}^{K}σ(nn(x_{i}), y)

where the σ is the sigmo(., y) function centerred  at y.

We wish now to calculate the  probability distributino of the r.v. AK. based on the previous counting approach in order to construct the histogram associated. For that purpose we used bump function \psi_{k} ∈ C^{\infinity}(R). This way we have reconstructed the aproximated histogram which will be a continious diferntialble function.

We now expressed our new loss function as the diffeenrence between the uniform distribution of the rv AK and the obtained result. The are several way to construct this.



## Experiments

## Conclusion
