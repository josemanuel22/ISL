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

function KSD(noise_model, target_model, n_sample, range)
    train_set = rand(target_model, n_sample)
    hist1 = fit(Histogram, train_set, range)

    data = vec(gen(rand(noise_model, n_sample)'))
    hist2 = fit(Histogram, data, range)
    return maximum(abs.(hist1.weights - hist2.weights)) /
           (n_sample * abs(range[2] - range[1]))
end

function MAE(noise_model, f̂ᵢ, n_sample)
    xᵢ = rand(noise_model, n_sample)
    fᵢ = vec(gen(xᵢ'))
    return mean(abs.(fᵢ .- f̂ᵢ(xᵢ)))
end

function MSE(noise_model, f̂ᵢ, n_sample)
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
        linecolor=get(ColorSchemes.rainbow, 0.2),
    )
    y = gen(range')
    return plot!(range, vec(y); legend=:bottomright, label="GAN", linecolor=:redsblues)
end

function plot_global(
    real_transform, noise_model, target_model, gen, n_sample, range_transform, range_result
)
    ksd = KSD(noise_model, target_model, n_samples, 18:0.1:25)
    mae = MAE(noise_model, x -> quantile.(target_model, cdf(noise_model, x)), n_samples)
    mse = MSE(noise_model, x -> quantile.(target_model, cdf(noise_model, x)), n_samples)

    return plot(
        plot_transformation(real_transform, gen, range_transform),
        plot_result(noise_model, target_model, gen, n_sample, range_result);
        plot_title=@sprintf("KSD: %0.2f     MEA: %0.2f     MSE: %0.2f", ksd, mae, mse),
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

function save_gan_model(gen, dscr, hparams)
    global gans = Dict(
        HyperParamsVanillaGan => "vgan",
        HyperParamsWGAN => "wgan",
        HyperParamsMMD1D => "mmdgan",
    )

    function getName(hparams)
        gan = gans[typeof(hparams)]
        lr_gen = hparams.lr_gen
        dscr_steps = hparams.dscr_steps
        noise_model = string(hparams.noise_model)
        target_model = string(hparams.target_model)
        basename = "$gan-$noise_model-$target_modellr_gen=$lr_gen-dscr_steps=$dscr_steps"
        i = get_incremental_filename(basename)
        new_filename = basename * "-$i.bson"
        return new_filename
    end
    name = getName(hparams)
    @save name gen dscr hparams
end

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
    @save name gen hparams
end
