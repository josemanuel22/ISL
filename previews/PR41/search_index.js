var documenterSearchIndex = {"docs":
[{"location":"gan/#GAN","page":"GAN","title":"GAN","text":"","category":"section"},{"location":"gan/","page":"GAN","title":"GAN","text":"In this repository, we have included a folder with different generative adversarial networks, GANs: vanilla GAN, WGAN, MMD-GAN.","category":"page"},{"location":"gan/","page":"GAN","title":"GAN","text":"In the first two cases, we have used the implementation from this repoistory, with some minor changes. In the last case, we have rewritten the original code written in Python to Julia.","category":"page"},{"location":"gan/","page":"GAN","title":"GAN","text":"The goal is to test that the AdapativeBlockLearning methods can work as regularizers for the solutions proposed by the GANs, providing a solution to the Helvetica scenario.","category":"page"},{"location":"benchmark/#Benchmark","page":"Benchmark","title":"Benchmark","text":"","category":"section"},{"location":"benchmark/","page":"Benchmark","title":"Benchmark","text":"This module contains utility functions for conducting tests and generating graphs and statistics from the obtained results.","category":"page"},{"location":"","page":"Home","title":"Home","text":"CurrentModule = AdaptativeBlockLearning","category":"page"},{"location":"#AdaptativeBlockLearning","page":"Home","title":"AdaptativeBlockLearning","text":"","category":"section"},{"location":"","page":"Home","title":"Home","text":"Documentation for AdaptativeBlockLearning.jl","category":"page"},{"location":"","page":"Home","title":"Home","text":"Modules = [AdaptativeBlockLearning]\nOrder   = [:module, :constant, :type, :function]","category":"page"},{"location":"#AdaptativeBlockLearning.AutoAdaptativeHyperParams","page":"Home","title":"AdaptativeBlockLearning.AutoAdaptativeHyperParams","text":"AutoAdaptativeHyperParams\n\nHyperparameters for the method adaptative_block_learning\n\n@with_kw struct AutoAdaptativeHyperParams\n    samples::Int64 = 1000\n    epochs::Int64 = 100\n    η::Float64 = 1e-3\n    max_k::Int64 = 10\n    transform = Normal(0.0f0, 1.0f0)\nend;\n\n\n\n\n\n","category":"type"},{"location":"#AdaptativeBlockLearning.HyperParams","page":"Home","title":"AdaptativeBlockLearning.HyperParams","text":"HyperParams\n\nHyperparameters for the method adaptative_block_learning\n\n@with_kw struct HyperParams\n    samples::Int64 = 1000               # number of samples per histogram\n    K::Int64 = 2                        # number of simulted observations\n    epochs::Int64 = 100                 # number of epochs\n    η::Float64 = 1e-3                   # learning rate\n    transform = Normal(0.0f0, 1.0f0)    # transform to apply to the data\nend;\n\n\n\n\n\n","category":"type"},{"location":"#AdaptativeBlockLearning.HyperParamsTS","page":"Home","title":"AdaptativeBlockLearning.HyperParamsTS","text":"HyperParamsTS\n\nHyperparameters for the method ts_adaptative_block_learning\n\n\n\n\n\n","category":"type"},{"location":"#AdaptativeBlockLearning._sigmoid-Union{Tuple{T}, Tuple{Matrix{T}, T}} where T<:AbstractFloat","page":"Home","title":"AdaptativeBlockLearning._sigmoid","text":"_sigmoid(ŷ, y)\n\nSigmoid function centered at y.\n\n\n\n\n\n","category":"method"},{"location":"#AdaptativeBlockLearning.adaptative_block_learning-Tuple{Any, Any, Any}","page":"Home","title":"AdaptativeBlockLearning.adaptative_block_learning","text":"adaptative_block_learning(model, data, hparams)\n\nCustom loss function for the model. model is a Flux neuronal network model, data is a loader Flux object and hparams is a HyperParams object.\n\nArguments\n\nnn_model::Flux.Chain: is a Flux neuronal network model\ndata::Flux.DataLoader: is a loader Flux object\nhparams::HyperParams: is a HyperParams object\n\n\n\n\n\n","category":"method"},{"location":"#AdaptativeBlockLearning.auto_adaptative_block_learning-Tuple{Any, Any, Any}","page":"Home","title":"AdaptativeBlockLearning.auto_adaptative_block_learning","text":"auto_adaptative_block_learning(model, data, hparams)\n\nCustom loss function for the model.\n\nThis method gradually adapts K (starting from 2) up to max_k (inclusive). The value of K is chosen based on a simple two-sample test between the histogram associated with the obtained result and the uniform distribution.\n\nTo see the value of K used in the test, set the logger level to debug before executing.\n\n#Arguments\n\nmodel::Flux.Chain: is a Flux neuronal network model\ndata::Flux.DataLoader: is a loader Flux object\nhparams::AutoAdaptativeHyperParams: is a AutoAdaptativeHyperParams object\n\n\n\n\n\n","category":"method"},{"location":"#AdaptativeBlockLearning.convergence_to_uniform-Union{Tuple{Vector{T}}, Tuple{T}} where T<:Int64","page":"Home","title":"AdaptativeBlockLearning.convergence_to_uniform","text":"convergence_to_uniform(aₖ)\n\nTest the convergence of the distributino of the window of the rv's Aₖ to a uniform distribution. It is implemented using a Chi-Square test.\n\n\n\n\n\n","category":"method"},{"location":"#AdaptativeBlockLearning.generate_aₖ-Union{Tuple{T}, Tuple{Matrix{T}, T}} where T<:AbstractFloat","page":"Home","title":"AdaptativeBlockLearning.generate_aₖ","text":"generate_aₖ(ŷ, y)\n\nGenerate a one step histogram (Vector{Float}) of the given vector ŷ of K simulted observations and the real data y generate_aₖ(ŷ, y) = ∑ₖ γ(ŷ, y, k)\n\nvecaₖ = _k=0^K γ(y y k) = _k=0^K _i=1^N ψₖ circ ϕ(y yᵢ)\n\n\n\n\n\n","category":"method"},{"location":"#AdaptativeBlockLearning.get_density-NTuple{4, Any}","page":"Home","title":"AdaptativeBlockLearning.get_density","text":"get_density(nn, data, t, m)\n\nGenerate m samples from the model nn given the data data up to time t. Can be used to generate the histogram at time t of the model.\n\n\n\n\n\n","category":"method"},{"location":"#AdaptativeBlockLearning.get_proxy_histogram_loss_ts-NTuple{4, Any}","page":"Home","title":"AdaptativeBlockLearning.get_proxy_histogram_loss_ts","text":"get_proxy_histogram_loss_ts(nn_model, data_xₜ, data_xₜ₊₁, hparams)\n\nGet the loss of the model nn_model on the data data_xₜ and data_xₜ₊₁ using the proxy histogram method.\n\n\n\n\n\n","category":"method"},{"location":"#AdaptativeBlockLearning.get_window_of_Aₖ-Tuple{Any, Any, Any, Int64}","page":"Home","title":"AdaptativeBlockLearning.get_window_of_Aₖ","text":"get_window_of_Aₖ(model, target , K, n_samples)\n\nGenerate a window of the rv's Aₖ for a given model and target function.\n\n\n\n\n\n","category":"method"},{"location":"#AdaptativeBlockLearning.jensen_shannon_∇-Union{Tuple{Vector{T}}, Tuple{T}} where T<:AbstractFloat","page":"Home","title":"AdaptativeBlockLearning.jensen_shannon_∇","text":"jensen_shannon_∇(aₖ)\n\nJensen shannon difference between aₖ vector and uniform distribution vector.\n\n\n\n\n\n","category":"method"},{"location":"#AdaptativeBlockLearning.scalar_diff-Union{Tuple{Vector{T}}, Tuple{T}} where T<:AbstractFloat","page":"Home","title":"AdaptativeBlockLearning.scalar_diff","text":"scalar_diff(aₖ)\n\nScalar difference between aₖ vector and uniform distribution vector.\n\nloss(weights) = langle (a₀ - N(K+1) cdots aₖ - N(K+1)) (a₀ - N(K+1) cdots aₖ - N(K+1))rangle = _k=0^K(a_k - (N(K+1)))^2\n\n\n\n\n\n","category":"method"},{"location":"#AdaptativeBlockLearning.ts_adaptative_block_learning-NTuple{5, Any}","page":"Home","title":"AdaptativeBlockLearning.ts_adaptative_block_learning","text":"ts_adaptative_block_learning(nn_model, Xₜ, Xₜ₊₁, hparams)\n\nCustom loss function for the model nn_model on time series data Xₜ and Xₜ₊₁.\n\n\n\n\n\n","category":"method"},{"location":"#AdaptativeBlockLearning.ts_covariates_adaptative_block_learning-NTuple{5, Any}","page":"Home","title":"AdaptativeBlockLearning.ts_covariates_adaptative_block_learning","text":"ts_covariates_adaptative_block_learning(nn_model, Xₜ, Xₜ₊₁, hparams)\n\nCustom loss function for the model nn_model on time series data Xₜ and Xₜ₊₁. Where Xₜ is a vector of vectors where each vector [x₁, x₂, ..., xₙ] is a time series where we want to predict the value at position x₁ using the values [x₂, ..., xₙ] as covariate and Xₜ₊₁ is a vector.\n\n\n\n\n\n","category":"method"},{"location":"#AdaptativeBlockLearning.γ-Union{Tuple{T}, Tuple{Matrix{T}, T, Int64}} where T<:AbstractFloat","page":"Home","title":"AdaptativeBlockLearning.γ","text":"γ(yₖ, yₙ, m)\n\nCalculate the contribution of ψₘ ∘ ϕ(yₖ, yₙ) to the m bin of the histogram (Vector{Float}).\n\nγ(yₖ yₙ m) = ψₘ circ ϕ(yₖ yₙ)\n\n\n\n\n\n","category":"method"},{"location":"#AdaptativeBlockLearning.γ_fast-Union{Tuple{T}, Tuple{Matrix{T}, T, Int64}} where T<:AbstractFloat","page":"Home","title":"AdaptativeBlockLearning.γ_fast","text":"γ_fast(yₖ, yₙ, m)\n\nApply the γ function to the given parameters. This function is faster than the original γ function because it uses StaticArrays. However because Zygote does not support StaticArrays, this function can not be used in the training process.\n\n\n\n\n\n","category":"method"},{"location":"#AdaptativeBlockLearning.ψₘ-Union{Tuple{T}, Tuple{T, Int64}} where T<:AbstractFloat","page":"Home","title":"AdaptativeBlockLearning.ψₘ","text":"ψₘ(y, m)\n\nBump function centered at m. Implemented as a gaussian function.\n\n\n\n\n\n","category":"method"},{"location":"#AdaptativeBlockLearning.ϕ-Union{Tuple{T}, Tuple{Matrix{T}, T}} where T<:AbstractFloat","page":"Home","title":"AdaptativeBlockLearning.ϕ","text":"ϕ(yₖ, yₙ)\n\nSum of the sigmoid function centered at yₙ applied to the vector yₖ.\n\nϕ(yₖ yₙ) = _i=1^K σ(yₖ^i yₙ)\n\n\n\n\n\n","category":"method"}]
}
