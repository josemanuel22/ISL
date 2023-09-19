using Plots
using BSON: @save, @load
using ColorSchemes, Colors
using Printf

macro test_experiments(msg, ex)
    @info "executing $msg"
    quote
        $(esc(ex))
    end
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
    return plot!(range, vec(y); legend=:bottomright, label="neural network", linecolor=get(ColorSchemes.rainbow, 0.2), ylims=(-10,10))
end

function plot_global(
    real_transform, noise_model, target_model, gen, n_sample, range_transform, range_result
)
    function format_numbers(x)
        if abs(x) < 0.01
            formatted_x = @sprintf("%.2e", x)
        else
            formatted_x = @sprintf("%.4f", x)
        end
        return formatted_x
    end
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
    save_gan_model(gen, dscr, hparams)

Save the model (generator, discriminator and hyper-parameters) in a bson file.
The name of the file is generated based on the hyper-parameters.
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
        noise_model = replace(strip(string(hparams.noise_model)), "\n" => "", r"(K = .*)" => "", r"components\[.*\] " => "", r"prior = " => "", "μ=" => "", "σ=" => "", r"\{Float.*\}" => "")
        target_model = replace(strip(string(hparams.target_model)), "\n" => "", r"(K = .*)" => "", r"components\[.*\] " => "", r"prior = " => "", "μ=" => "", "σ=" => "", r"\{Float.*\}" => "")
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
    save_adaptative_model(gen, hparams)

Save the model (generator and hyper-parameters) in a bson file.
The name of the file is generated based on the hyper-parameters.
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
