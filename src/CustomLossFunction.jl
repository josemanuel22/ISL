using StatsBase

struct CustomLoss
    K::Int

    function CustomLoss(K::Int)
        new(K)
    end

    scalar_diff = (loss::CustomLoss, a_k) -> sum((a_k - (1 / (loss.K + 1))).^2)
    kl_divergence = (loss::CustomLoss, a_k) -> jensen_shannon_divergence(a_k, fill(1 / (loss.K + 1), 1, loss.K + 1))

end

function jensen_shannon_divergence(p,q)
    ϵ = 1e-3
    p.+=ϵ; q.+=ϵ;
    return 0.5 * (kldivergence(p,q) + kldivergence(q,p))
end

function sigmoid(ŷ, y)
    return σ.((ŷ-y)*10)
end

function ψₘ(y, m)
    stddev = 0.1
    return exp(-0.5 * ((y - m) / stddev)^2)
end

function ϕ(yₖ, yₙ)
    return sum(sigmoid.(yₙ, yₖ))
end

function γ(yₖ, yₙ, m, K)
    eₘ = (n, m) -> [j == m ? 1.0 : 0.0 for j in 1:n]
    return eₘ(K, m) * ψₘ(ϕ(yₖ, yₙ), m)
end