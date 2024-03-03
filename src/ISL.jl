__precompile__()

```
Training Implicit Generative Models via an Invariant Statistical Loss addresses the
challenges of training implicit generative models, like Generative Adversarial
Networks(GANs), which often suffer from unstable training and mode-dropping issues.
We propose a novel, discriminator-free approach for training 1-dimensional (1D) generative
implicit models, using a loss function based on the discrepancy between a transformed model
sample distribution and a uniform distribution. This approach is designed to be invariant to
the true data distribution, potentially providing a more stable and effective training
method.We first apply our method to 1D random variables, demonstrating its capacity for
reparameterizing complex distributions. We then extend the approach to temporal settings,
both univariate and multivariate, aiming to model the conditional distribution of each
sample given its historical data. Through numerical simulations, our method shows promising
results in accurately learning true distributions across various scenarios and mitigating
known issues with existing implicit methods.
```
module ISL

using Flux
using StatsBase
using Distributions: Normal, rand
using HypothesisTests: pvalue, ChisqTest
using MLUtils
using Parameters: @with_kw
using ProgressMeter

using StaticArrays

include("CustomLossFunction.jl")

export _sigmoid,
    ψₘ,
    ϕ,
    γ,
    generate_aₖ,
    scalar_diff,
    jensen_shannon_∇,
    jensen_shannon_divergence,
    get_window_of_Aₖ,
    convergence_to_uniform,
    ISLParams,
    AutoISLParams,
    invariant_statistical_loss,
    #invariant_statistical_loss_1,
    #auto_invariant_statistical_loss_2,
    auto_invariant_statistical_loss,
    #auto_invariant_statistical_loss_1,
    HyperParamsTS,
    ts_invariant_statistical_loss_one_step_prediction,
    ts_invariant_statistical_loss,
    ts_invariant_statistical_loss_multivariate
end
