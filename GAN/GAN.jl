module GAN

using Flux
using Flux.Optimise: update!
using Flux: logitbinarycrossentropy, binarycrossentropy
using Statistics
using Parameters: @with_kw
using Random
using CUDA
using Zygote
using Distributions
using ProgressMeter

include("vanilla_gan.jl")
include("wgan.jl")
include("MMD_GAN/mmd_gan_1d.jl")

export HyperParamsVanillaGan,
    train_vanilla_gan,
    HyperParamsWGAN,
    train_wgan,
    HyperParamsMMD1D,
    train_mmd_gan_1d
end
