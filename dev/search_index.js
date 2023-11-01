var documenterSearchIndex = {"docs":
[{"location":"gan/#GAN","page":"GAN","title":"GAN","text":"","category":"section"},{"location":"gan/","page":"GAN","title":"GAN","text":"In this repository, we have included a folder with different generative adversarial networks, GANs: vanilla GAN, WGAN, MMD-GAN.","category":"page"},{"location":"gan/","page":"GAN","title":"GAN","text":"In the first two cases, we have used the implementation from this repoistory, with some minor changes. In the last case, we have rewritten the original code written in Python to Julia.","category":"page"},{"location":"gan/","page":"GAN","title":"GAN","text":"The goal is to test that the AdapativeBlockLearning methods can work as regularizers for the solutions proposed by the GANs, providing a solution to the Helvetica scenario.","category":"page"},{"location":"benchmark/#Benchmark","page":"Benchmark","title":"Benchmark","text":"","category":"section"},{"location":"benchmark/","page":"Benchmark","title":"Benchmark","text":"This module contains utility functions for conducting tests and generating graphs and statistics from the obtained results.","category":"page"},{"location":"","page":"Home","title":"Home","text":"CurrentModule = ISL","category":"page"},{"location":"#ISL","page":"Home","title":"ISL","text":"","category":"section"},{"location":"","page":"Home","title":"Home","text":"Documentation for ISL.jl","category":"page"},{"location":"","page":"Home","title":"Home","text":"Modules = [ISL]\nOrder   = [:module, :constant, :type, :function]","category":"page"},{"location":"#ISL.AutoISLParams","page":"Home","title":"ISL.AutoISLParams","text":"AutoISLParams\n\nHyperparameters for the method invariant_statistical_loss\n\n@with_kw struct AutoISLParams\n    samples::Int64 = 1000\n    epochs::Int64 = 100\n    η::Float64 = 1e-3\n    max_k::Int64 = 10\n    transform = Normal(0.0f0, 1.0f0)\nend;\n\n\n\n\n\n","category":"type"},{"location":"#ISL.HyperParamsTS","page":"Home","title":"ISL.HyperParamsTS","text":"HyperParamsTS\n\nHyperparameters for the method ts_adaptative_block_learning\n\n\n\n\n\n","category":"type"},{"location":"#ISL.ISLParams","page":"Home","title":"ISL.ISLParams","text":"ISLParams\n\nHyperparameters for the method adaptative_block_learning\n\n@with_kw struct ISLParams\n    samples::Int64 = 1000               # number of samples per histogram\n    K::Int64 = 2                        # number of simulted observations\n    epochs::Int64 = 100                 # number of epochs\n    η::Float64 = 1e-3                   # learning rate\n    transform = Normal(0.0f0, 1.0f0)    # transform to apply to the data\nend;\n\n\n\n\n\n","category":"type"},{"location":"#ISL._sigmoid-Union{Tuple{T}, Tuple{Matrix{T}, T}} where T<:AbstractFloat","page":"Home","title":"ISL._sigmoid","text":"_sigmoid(ŷ::Matrix{T}, y::T) where {T<:AbstractFloat}\n\nCalculate the sigmoid function centered at y.\n\nArguments\n\nŷ::Matrix{T}: The matrix of values to apply the sigmoid function to.\ny::T: The center value around which the sigmoid function is centered.\n\nReturns\n\nA matrix of the same size as ŷ containing the sigmoid-transformed values.\n\nThis function calculates the sigmoid function for each element in the matrix ŷ centered at the value y. It applies a fast sigmoid transformation with a scaling factor of 10.0.\n\n\n\n\n\n","category":"method"},{"location":"#ISL.auto_invariant_statistical_loss-Tuple{Any, Any, Any}","page":"Home","title":"ISL.auto_invariant_statistical_loss","text":"`auto_invariant_statistical_loss(model, data, hparams)``\n\nCustom loss function for the model.\n\nThis method gradually adapts K (starting from 2) up to max_k (inclusive). The value of K is chosen based on a simple two-sample test between the histogram associated with the obtained result and the uniform distribution.\n\nTo see the value of K used in the test, set the logger level to debug before executing.\n\n#Arguments\n\nmodel::Flux.Chain: is a Flux neuronal network model\ndata::Flux.DataLoader: is a loader Flux object\nhparams::AutoAdaptativeHyperParams: is a AutoAdaptativeHyperParams object\n\n\n\n\n\n","category":"method"},{"location":"#ISL.convergence_to_uniform-Union{Tuple{Vector{T}}, Tuple{T}} where T<:Int64","page":"Home","title":"ISL.convergence_to_uniform","text":"`convergence_to_uniform(aₖ)``\n\nTest the convergence of the distributino of the window of the rv's Aₖ to a uniform distribution. It is implemented using a Chi-Square test.\n\n\n\n\n\n","category":"method"},{"location":"#ISL.generate_aₖ-Union{Tuple{T}, Tuple{Matrix{T}, T}} where T<:AbstractFloat","page":"Home","title":"ISL.generate_aₖ","text":"`generate_aₖ(ŷ, y)``\n\nCalculate the values of the real observation y in each of the components of the approximate histogram with K bins.\n\nArguments\n\nŷ::Matrix{T}: A matrix of simulated observations (each column represents a different simulation).\ny::T: The real data for which the one-step histogram is generated.\n\nReturns\n\naₖ::Vector{Float}: A vector with the values of the real observation y in each of the components of the approximate histogram with K bins.\n\nDetails\n\nThe generate_aₖ function calculates the one-step histogram aₖ as the sum of the contribution of the observation to the subrogate histogram bins. It uses the function γ to calculate the contribution of each observation at each histogram bin. The final histogram is the sum of these contributions.\n\nThe formula for generating aₖ is as follows:\n\naₖ = _k=0^K γ(y y k) = _k=0^K _i=1^N ψₖ(y yᵢ)\n\n\n\n\n\n","category":"method"},{"location":"#ISL.get_window_of_Aₖ-Tuple{Any, Any, Any, Int64}","page":"Home","title":"ISL.get_window_of_Aₖ","text":"`get_window_of_Aₖ(model, target , K, n_samples)``\n\nGenerate a window of the rv's Aₖ for a given model and target function.\n\n\n\n\n\n","category":"method"},{"location":"#ISL.invariant_statistical_loss-Tuple{Any, Any, Any}","page":"Home","title":"ISL.invariant_statistical_loss","text":"`invariant_statistical_loss(model, data, hparams)``\n\nCustom loss function for the model. model is a Flux neuronal network model, data is a loader Flux object and hparams is a HyperParams object.\n\nArguments\n\nnn_model::Flux.Chain: is a Flux neuronal network model\ndata::Flux.DataLoader: is a loader Flux object\nhparams::HyperParams: is a HyperParams object\n\n\n\n\n\n","category":"method"},{"location":"#ISL.jensen_shannon_∇-Union{Tuple{Vector{T}}, Tuple{T}} where T<:AbstractFloat","page":"Home","title":"ISL.jensen_shannon_∇","text":"`jensen_shannon_∇(aₖ)``\n\nJensen shannon difference between aₖ vector and uniform distribution vector.\n\n\n\n\n\n","category":"method"},{"location":"#ISL.scalar_diff-Union{Tuple{Vector{T}}, Tuple{T}} where T<:AbstractFloat","page":"Home","title":"ISL.scalar_diff","text":"`scalar_diff(q)`\n\nScalar difference between the vector representing our subrogate histogram and the uniform distribution vector.\n\nloss = q-1k+1_2 = _k=0^K (qₖ - 1K+1)^2\n\n\n\n\n\n","category":"method"},{"location":"#ISL.ts_invariant_statistical_loss-NTuple{5, Any}","page":"Home","title":"ISL.ts_invariant_statistical_loss","text":"`ts_invariant_statistical_loss(rec, gen, Xₜ, Xₜ₊₁, hparams)``\n\nTrain a model for time series data with statistical invariance loss method.\n\nArguments\n\nrec: The recurrent neural network (RNN) responsible for encoding the time series data.\ngen: The generative model used for generating future time series data.\nXₜ: An array of input time series data at time t.\nXₜ₊₁: An array of target time series data at time t+1.\nhparams::NamedTuple: A structure containing hyperparameters for training. It should include the following fields:\nη::Float64: Learning rate for optimization.\nwindow_size::Int: Size of the sliding window used during training.\nK::Int: Number of samples in the generative model.\nnoise_model: Noise model used for generating random noise.\n\nReturns\n\nlosses::Vector{Float64}: A vector containing the training loss values for each iteration.\n\nDescription\n\nThis function train a model for time series data with statistical invariance loss method. It utilizes a recurrent neural network (rec) to encode the time series data at time t and a generative model (gen) to generate future time series data at time t+1. The training process involves optimizing both the rec and gen models.\n\nThe function iterates through the provided time series data (Xₜ and Xₜ₊₁) in batches, with a sliding window of size window_size.\n\n\n\n\n\n","category":"method"},{"location":"#ISL.γ-Union{Tuple{T}, Tuple{Matrix{T}, T, Int64}} where T<:AbstractFloat","page":"Home","title":"ISL.γ","text":"γ(yₖ::Matrix{T}, yₙ::T, m::Int64) where {T<:AbstractFloat}`\n\nCalculate the contribution of ψₘ ∘ ϕ(yₖ, yₙ) to the m bin of the histogram as a Vector{Float}.\n\nArguments\n\nyₖ::Matrix{T}: A matrix of values for which to compute the contribution.\nyₙ::T: The center value around which the sigmoid and bump functions are centered.\nm::Int64: The bin index for which to calculate the contribution.\n\nReturns\n\nA vector of floating-point values representing the contribution to the m bin of the histogram.\n\nThis function calculates the contribution of the composition of ψₘ and ϕ(yₖ, yₙ) to the m-th bin of the histogram. The result is a vector of floating-point values.\n\nThe contribution is computed according to the formula:\n\nγ(yₖ yₙ m) = ψₘ  ϕ(yₖ yₙ)\n\n\n\n\n\n","category":"method"},{"location":"#ISL.γ_fast-Union{Tuple{T}, Tuple{Matrix{T}, T, Int64}} where T<:AbstractFloat","page":"Home","title":"ISL.γ_fast","text":"γ_fast(yₖ::Matrix{T}, yₙ::T, m::Int64) where {T<:AbstractFloat}`\n\nApply the γ function to the given parameters using StaticArrays for improved performance.\n\nArguments\n\nyₖ::Matrix{T}: A matrix of values for which to compute the contribution.\nyₙ::T: The center value around which the sigmoid and bump functions are centered.\nm::Int64: The bin index for which to calculate the contribution.\n\nReturns\n\nA StaticVector{T} representing the contribution to the m bin of the histogram.\n\nThis function applies the γ function to compute the contribution of the composition of ψₘ and ϕ(yₖ, yₙ) to the m-th bin of the histogram. The result is a StaticVector{T} for improved performance.\n\nPlease note that although this function offers improved performance, it cannot be used in the training process with Zygote because Zygote does not support StaticArrays.\n\n\n\n\n\n","category":"method"},{"location":"#ISL.ψₘ-Union{Tuple{T}, Tuple{T, Int64}} where T<:AbstractFloat","page":"Home","title":"ISL.ψₘ","text":"ψₘ(y::T, m::Int64) where {T<:AbstractFloat}`\n\nCalculate the bump function centered at m, implemented as a Gaussian function.\n\nArguments\n\ny::T: The input value for which to compute the bump function.\nm::Int64: The center point around which the bump function is centered.\n\nReturns\n\nA floating-point value representing the bump function's value at the input y.\n\nThis function calculates the bump function, which is centered at the integer value m. It is implemented as a Gaussian function with a standard deviation of 0.1.\n\n\n\n\n\n","category":"method"},{"location":"#ISL.ϕ-Union{Tuple{T}, Tuple{Matrix{T}, T}} where T<:AbstractFloat","page":"Home","title":"ISL.ϕ","text":"ϕ(yₖ::Matrix{T}, yₙ::T) where {T<:AbstractFloat}\n\nCalculate the sum of sigmoid functions centered at yₙ applied to the vector yₖ.\n\nArguments\n\nyₖ::Matrix{T}: A matrix of values for which to compute the sum of sigmoid functions.\nyₙ::T: The center value around which the sigmoid functions are centered.\n\nReturns\n\nA floating-point value representing the sum of sigmoid-transformed values.\n\nThis function calculates the sum of sigmoid functions, each centered at the value yₙ, applied element-wise to the matrix yₖ. The sum is computed according to the formula:\n\nmath ϕ(yₖ, yₙ) = ∑_{i=1}^K σ(yₖ^i, yₙ)`\n\n\n\n\n\n","category":"method"}]
}
