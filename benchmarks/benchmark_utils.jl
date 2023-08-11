using Plots
using BSON

macro test_experiments(msg, ex)
    @info "executing $msg"
    quote
        $(esc(ex))
    end
end

function plot_result(noise_model, target_model, model, n_sample, range)
    x = rand(noise_model, n_sample)
    plot(x -> pdf(target_model, x), range)
    ŷ = model(x')
    return histogram!(ŷ'; bins=range, xlabel="x", ylabel="pdf", normalize=:pdf)
end

function plot_transformation(real_transform, gen, range)
    plot(real_transform, range; xlabel="z noise space", ylabel="x target space")
    y = gen(range')
    return plot!(range, vec(y))
end

function save_gan_model(gen, dscr, hyperams)
    global gans = Dict(
        HyperParamsVanillaGan => "vgan",
        HyperParamsWGAN => "wgan",
        HyperParamsMMD1D => "mmdgan",
    )

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

    function getName(hyperams)
        gan = cases[typeof(hyperams)]
        lr_gen = hyperams.lr_gen
        dscr_steps = hyperams.dscr_steps
        basename = "$gan\_lr_gen=$lr_gen\_dscr_steps=$dscr_steps"
        i = get_incremental_filename(base_name)
        new_filename = basename * "-$i.bson"
        return new_filename
    end
    name = getName(hyperams)
    @save name (gen, dscr, hyperams)
end
