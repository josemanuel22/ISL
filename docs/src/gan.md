# Generative Adversarial Networks (GANs) Module Overview 

This repository includes a dedicated folder that contains implementations of different Generative Adversarial Networks (GANs), showcasing a variety of approaches within the GAN framework. Our collection includes:

- **Vanilla GAN**: Based on the foundational GAN concept introduced in ["Generative Adversarial Nets"](https://arxiv.org/pdf/1406.2661.pdf) by Goodfellow et al. This implementation adapts and modifies the code from [FluxGAN repository](https://github.com/AdarshKumar712/FluxGAN) to fit our testing needs.

- **WGAN (Wasserstein GAN)**: Implements the Wasserstein GAN as described in ["Wasserstein GAN"](https://arxiv.org/pdf/1701.07875.pdf) by Arjovsky et al., providing an advanced solution to the issue of training stability in GANs. Similar to Vanilla GAN, we have utilized and slightly adjusted the implementation from the [FluxGAN repository](https://github.com/AdarshKumar712/FluxGAN).

- **MMD-GAN (Maximum Mean Discrepancy GAN)**: Our implementation of MMD-GAN is inspired by the paper ["MMD GAN: Towards Deeper Understanding of Moment Matching Network"](https://arxiv.org/pdf/1705.08584.pdf) by Li et al. Unlike the previous models, the MMD-GAN implementation has been rewritten in Julia, transitioning from the original [Python code](https://github.com/OctoberChang/MMD-GAN) provided by the authors.

## Objective

The primary goal of incorporating these GAN models into our repository is to evaluate the effectiveness of ISL (Invariant Statistical Learning) methods as regularizers for GAN-based solutions. Specifically, we aim to address the challenges presented in the "Helvetica scenario," exploring how ISL methods can enhance the robustness and generalization of GANs in generating high-quality synthetic data.

## Implementation Details

For each GAN variant mentioned above, we have made certain adaptations to the original implementations to ensure compatibility with our testing framework and the objectives of the ISL method integration. These modifications range from architectural adjustments to the optimization process, aiming to optimize the performance and efficacy of the ISL regularizers within the GAN context.

We encourage interested researchers and practitioners to explore the implementations and consider the potential of ISL methods in improving GAN architectures. For more detailed insights into the modifications and specific implementation choices, please refer to the code and accompanying documentation within the respective folders for each GAN variant.