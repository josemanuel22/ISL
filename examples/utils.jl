using Plots
using BSON: @save, @load
using ColorSchemes, Colors
using Printf

"""
Macro for executing experiments and logging execution time and memory usage.

This macro is used for running experiments and measuring the execution time and memory allocated during the execution of a given expression (`ex`). It logs information about the execution process, including the start time, end time, elapsed time, and memory allocated.

# Arguments
- `msg::String`: A descriptive message for the experiment being executed.
- `ex::Expr`: The expression representing the experiment to be executed.

# Example
```julia
@test_experiments "Experiment 1" begin
    # Code for the experiment
    result = perform_experiment()
end
```º
"""
macro test_experiments(msg, ex)
    @info "executing $msg"
    start_time = time()
    mem = @allocated quote
        $(esc(ex))
    end
    end_time = time()
    elapsed_time = end_time - start_time
    @info "Execution time for $msg: $elapsed_time seconds, memory allocated: $mem bytes"
end

function format_numbers(x)
    if abs(x) < 0.01
        formatted_x = @sprintf("%.2e", x)
    else
        formatted_x = @sprintf("%.4f", x)
    end
    return formatted_x
end

"""
Calculate KSD metric for a given model.

```math
KSD = \\sup_{x} |\\mathbb{P}_{X}(\\mathcal{x})-\\mathbb{P}_{θ}(x)|
```
"""
function KSD(noise_model, target_model, gen, n_sample, range)
    train_set = rand(target_model, n_sample)
    hist1 = fit(Histogram, train_set, range)

    data = vec(gen(rand(noise_model, n_sample)'))
    hist2 = fit(Histogram, data, range)
    return maximum(abs.(hist1.weights - hist2.weights)) / n_sample
end

"""
Calculate MAE metric for a given model.

```math
MAE = \\int_{-\\infty}^{\\infty} |f(z) - \\tilde{f}(z)| d\\mathbb{P}_{\\mathcal{Z}}(z)
```
"""
function MAE(noise_model, f̂ᵢ, gen, n_sample)
    xᵢ = rand(noise_model, n_sample)
    fᵢ = vec(gen(xᵢ'))
    return mean(abs.(fᵢ .- f̂ᵢ(xᵢ)))
end

"""
Calculate MSE metric for a given model.

```math
MSE = \\int_{-\\infty}^{\\infty} (f(z) - \\tilde{f}(z))^2 d\\mathbb{P}_{\\mathcal{Z}}(z)
```
"""
function MSE(noise_model, f̂ᵢ, gen, n_sample)
    xᵢ = rand(noise_model, n_sample)
    fᵢ = vec(gen(xᵢ'))
    return mean((fᵢ .- f̂ᵢ(xᵢ)) .^ 2)
end

function plot_result(noise_model, target_model, model, n_sample, range)
    x = rand(noise_model, n_sample)
    ŷ = model(x')
    histogram(
        ŷ';
        bins=range,
        xlabel="x",
        ylabel="pdf",
        normalize=:pdf,
        z=2,
        legend=false,
        color=get(ColorSchemes.rainbow, 0.2),
    )
    return plot!(
        x -> pdf(target_model, x),
        range;
        z=1,
        lw=2,
        linecolor=:redsblues,
        formatter=x -> @sprintf("%.2f", x),
        y=(0:0.05:0.5),
        #title=@sprintf("KSD: %0.2f MEA: %0.2f MSE: %0.2f", ksd, mae, mse),
        #titlefontsize=20,
    )
end

function plot_transformation(real_transform, gen, range)
    plot(
        real_transform,
        range;
        xlabel="z noise space",
        ylabel="x target space",
        label="Ideal",
        linecolor=:redsblues,
    )
    y = gen(range')
    return plot!(
        range,
        vec(y);
        legend=:bottomright,
        label="neural network",
        linecolor=get(ColorSchemes.rainbow, 0.2),
        ylims=(-10, 10),
    )
end

"""
Generate and display a global plot summarizing the performance of a generative model.

This function creates a global plot that summarizes the performance of a generative model by displaying various evaluation metrics and visualizations of data transformations and generated results.

# Arguments
- `real_transform`: A function representing the real data transformation.
- `noise_model`: A function representing the noise model used.
- `target_model`: A function representing the target data model.
- `gen`: The generative model to evaluate.
- `n_sample::Int`: The number of samples to generate for evaluation.
- `range_transform`: The range of values for the data transformation plot.
- `range_result`: The range of values for the generated result plot.

# Returns
- A global plot displaying data transformations, generated results, and evaluation metrics.

# Example
```julia
# Define real_transform, noise_model, target_model, gen, n_sample, range_transform, and range_result
plot = plot_global(real_transform, noise_model, target_model, gen, n_sample, range_transform, range_result)
display(plot)
```
"""
function plot_global(
    real_transform, noise_model, target_model, gen, n_sample, range_transform, range_result
)
    ksd = KSD(noise_model, target_model, gen, n_samples, range_result)
    mae = MAE(noise_model, real_transform, gen, n_samples)
    mse = MSE(noise_model, real_transform, gen, n_samples)

    return plot(
        plot_transformation(real_transform, gen, range_transform),
        plot_result(noise_model, target_model, gen, n_sample, range_result);
        plot_title="KSD: " *
                   format_numbers(ksd) *
                   " "^4 *
                   "MEA: " *
                   format_numbers(mae) *
                   " "^4 *
                   "MSE: " *
                   format_numbers(mse),
        plot_titlefontsize=12,
        fmt=:png,
    )
end

function get_incremental_filename(base_name)
    i = 1
    while true
        new_filename = base_name * "-$i.bson"
        if !isfile(new_filename)
            return i
        end
        i += 1
    end
end

"""
Save the GAN model (generator, discriminator, and hyper-parameters) to a BSON file.

This function takes a generator `gen`, a discriminator `dscr`, and a structure `hparams` containing hyper-parameters. It saves both the generator, discriminator, and hyper-parameters to a BSON file. The name of the file is automatically generated based on the hyper-parameters, ensuring uniqueness.

# Arguments
- `gen`: The generator model to be saved.
- `dscr`: The discriminator model to be saved.
- `hparams`: A structure containing hyper-parameters.

# Example
```julia
using BSON

# Create a GAN model and hyper-parameters
generator_model = ...
discriminator_model = ...
hyperparameters = HyperParamsVanillaGan(lr_gen=0.001, n_critic=5, noise_model=..., target_model=...)

# Save the GAN model and hyper-parameters to a file
save_gan_model(generator_model, discriminator_model, hyperparameters)
"""
function save_gan_model(gen, dscr, hparams)
    global gans = Dict(
        HyperParamsVanillaGan => "vgan",
        HyperParamsWGAN => "wgan",
        HyperParamsMMD1D => "mmdgan",
    )

    function getName(hparams)
        gan = gans[typeof(hparams)]
        lr_gen = hparams.lr_gen
        dscr_steps = hparams.n_critic
        noise_model = replace(
            strip(string(hparams.noise_model)),
            "\n" => "",
            r"(K = .*)" => "",
            r"components\[.*\] " => "",
            r"prior = " => "",
            "μ=" => "",
            "σ=" => "",
            r"\{Float.*\}" => "",
        )
        target_model = replace(
            strip(string(hparams.target_model)),
            "\n" => "",
            r"(K = .*)" => "",
            r"components\[.*\] " => "",
            r"prior = " => "",
            "μ=" => "",
            "σ=" => "",
            r"\{Float.*\}" => "",
        )
        basename = "$gan-$noise_model-$target_model-lr_gen=$lr_gen-dscr_steps=$dscr_steps"
        i = get_incremental_filename(basename)
        new_filename = basename * "-$i.bson"
        return new_filename
    end
    name = getName(hparams)
    @info "name file: " * name
    @save name gen dscr hparams
end

"""
Save the adaptive model (generator and hyper-parameters) to a BSON file.

This function takes a generator `gen` and a structure `hparams` containing hyper-parameters. It saves both the generator and hyper-parameters to a BSON file. The name of the file is automatically generated based on the hyper-parameters, ensuring uniqueness.

# Arguments
- `gen`: The generator model to be saved.
- `hparams`: A structure containing hyper-parameters.

# Example
```julia
using BSON

# Create a generator and hyper-parameters
generator_model = ...
hyperparameters = AdaptiveHyperparameters(samples=100, max_k=5, epochs=10, η=0.001)

# Save the model and hyper-parameters to a file
save_adaptative_model(generator_model, hyperparameters)
```
"""
function save_adaptative_model(gen, hparams)
    function getName(hparams)
        samples = hparams.samples
        max_k = hparams.max_k
        epochs = hparams.epochs
        lr = hparams.η
        basename = "samples=$samples-max_k=$max_k-epochs=$epochs-lr=$lr"
        i = get_incremental_filename(basename)
        new_filename = basename * "-$i.bson"
        return new_filename
    end
    name = getName(hparams)
    @info "name file: " * name
    @save name gen hparams
end

"""
Calculate the moving average of a vector.

This function takes a vector `arr` and a window size `window_size`, and returns a new vector
containing the moving averages of `arr` with the given window size.

# Arguments
- `arr::Vector{T}`: The input vector of data.
- `window_size::Int`: The size of the moving window. Must be a positive integer.

# Returns
- `ma::Vector{Float64}`: A vector containing the moving averages.

# Examples
```julia
data = [1.0, 2.0, 3.0, 4.0, 5.0]
window_size = 3
result = moving_average(data, window_size)
# result is [1.0, 1.5, 2.0, 3.0, 4.0]
```
"""
function moving_average(arr::Vector{T}, window_size::Int) where {T}
    if window_size <= 0
        throw(ArgumentError("window_size must be positive"))
    end

    cumsum_arr = cumsum(arr)
    ma = [cumsum_arr[i] / i for i in 1:window_size]

    for i in (window_size + 1):length(arr)
        push!(ma, (cumsum_arr[i] - cumsum_arr[i - window_size]) / window_size)
    end

    return ma
end
