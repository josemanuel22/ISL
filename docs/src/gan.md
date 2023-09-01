# GAN

In this repository, we have included a folder with different generative adversarial networks, GANs: [vanilla GAN](https://arxiv.org/pdf/1406.2661.pdf), [WGAN](https://arxiv.org/pdf/1701.07875.pdf), [MMD-GAN](https://arxiv.org/pdf/1705.08584.pdf).

In the first two cases, we have used the implementation from this [repoistory](https://github.com/AdarshKumar712/FluxGAN), with some minor changes. In the last case, we have rewritten the original [code](https://github.com/OctoberChang/MMD-GAN) written in Python to Julia.

The goal is to test that the AdapativeBlockLearning methods can work as regularizers for the solutions proposed by the GANs, providing a solution to the Helvetica scenario.