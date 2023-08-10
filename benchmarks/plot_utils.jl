using Plots

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
