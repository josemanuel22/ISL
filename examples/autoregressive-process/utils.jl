using LinearAlgebra
using ToeplitzMatrices

ND(xₜ, x̂ₜ) = sum(abs.(xₜ .- x̂ₜ)) / sum(abs.(xₜ))

function RMSE(xₜ, x̂ₜ)
    return sqrt((1 / length(xₜ)) * sum((xₜ .- x̂ₜ) .^ 2)) / ((1 / length(xₜ)) * sum(xₜ))
end

function QLρ(xₜ, x̂ₜ; ρ=0.5)
    return 2 *
           (sum(abs.(xₜ))^-1) *
           sum(ρ .* (xₜ .- x̂ₜ) .* (xₜ .> x̂ₜ) .+ (1 - ρ) .* (x̂ₜ .- xₜ) .* (xₜ .<= x̂ₜ))
end

function get_watson_durbin_test(y, ŷ)
    e = []
    for (yₜ, ŷₜ) in zip(y, ŷ)
        append!(e, yₜ - ŷₜ)
    end
    sum = 0
    for i in 2:2:length(e)
        sum += (e[i] - e[i - 1])^2
    end
    return sum / sum(e .^ 2)
end

function yule_walker(
    x::Vector{Float64};
    order::Int64=1,
    method="adjusted",
    df::Union{Nothing,Int64}=nothing,
    inv=false,
    demean=true,
)
    method in ("adjusted", "mle") ||
        throw(ArgumentError("ACF estimation method must be 'adjusted' or 'MLE'"))

    x = copy(x)
    if demean
        x .-= mean(x)
    end
    n = isnothing(df) ? length(x) : df

    adj_needed = method == "adjusted"

    if ndims(x) > 1 || size(x, 2) != 1
        throw(ArgumentError("Expecting a vector to estimate AR parameters"))
    end

    r = zeros(Float64, order + 1)
    r[1] = sum(x .^ 2) / n
    for k = 1:order
        r[k+1] = sum(x[1:end-k] .* x[k+1:end]) / (n - k * adj_needed)
    end
    R = Toeplitz(r[1:(end - 1)], conj(r[1:(end - 1)]))

    rho = 0
    try
        rho = R \ r[2:end]
    catch err
        if occursin("Singular matrix", string(err))
            @warn "Matrix is singular. Using pinv."
            rho = pinv(R) * r[2:end]
        else
            throw(err)
        end
    end

    sigmasq = r[1] - dot(r[2:end], rho)
    sigma = isnan(sigmasq) || sigmasq <= 0 ? NaN : sqrt(sigmasq)

    if inv
        return rho, sigma, inv(R)
    else
        return rho, sigma
    end
end

function plot_ts(nn_model, xₜ, xₜ₊₁, hparams)
    Flux.reset!(nn_model)
    nn_model([xₜ.data[1]])
    plot(xₜ.data[1:15000]; seriestype=:scatter)
    return plot!(vec(nn_model.([xₜ.data[1:15000]]')...); seriestype=:scatter)
end
