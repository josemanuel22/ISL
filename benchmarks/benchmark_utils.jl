using Plots
using BSON: @save
using ColorSchemes, Colors
using Printf

macro test_experiments(msg, ex)
    @info "executing $msg"
    quote
        $(esc(ex))
    end
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
        label="GAN",
        linecolor=get(ColorSchemes.rainbow, 0.2),
    )
    y = gen(range')
    return plot!(range, vec(y); legend=:bottomright, label="Ideal", linecolor=:redsblues)
end

function plot_global(
    real_transform,
    noise_model,
    target_model,
    gen,
    n_sample,
    range_transform,
    range_result,
    ksd,
    mae,
    mse,
)
    return plot(
        plot_transformation(real_transform, gen, range_transform),
        plot_result(noise_model, target_model, gen, n_sample, range_result);
        plot_title=@sprintf("KSD: %0.2f     MEA: %0.2f     MSE: %0.2f", ksd, mae, mse),
        plot_titlefontsize=12,
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

function save_gan_model(gen, hyperams)
    global gans = Dict(
        HyperParamsVanillaGan => "vgan",
        HyperParamsWGAN => "wgan",
        HyperParamsMMD1D => "mmdgan",
    )

    function getName(hyperams)
        gan = gans[typeof(hyperams)]
        lr_gen = hyperams.lr_gen
        dscr_steps = hyperams.dscr_steps
        basename = "$gan-lr_gen=$lr_gen-dscr_steps=$dscr_steps"
        i = get_incremental_filename(basename)
        new_filename = basename * "-$i.bson"
        return new_filename
    end
    name = getName(hyperams)
    @save name gen hyperams
end

function save_adaptative_model(gen, hyperams)
    function getName(hyperams)
        samples = hyperams.samples
        K = hyperams.K
        epochs = hyperams.epochs
        lr = hyperams.η
        basename = "samples=$samples-K=$K-epochs=$epochs-lr=$lr"
        i = get_incremental_filename(basename)
        new_filename = basename * "-$i.bson"
        return new_filename
    end
    name = getName(hyperams)
    @save name gen hyperams
end
