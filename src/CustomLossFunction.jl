struct CustomLoss
    K::Int
    function CustomLoss(K::Int)
        new(K)
    end;
end;

scalar_diff = (loss::CustomLoss, a_k) -> sum((a_k .- (1 ./ (loss.K + 1))) .^2)
jensen_shannon_∇ = (loss::CustomLoss, a_k) -> jensen_shannon_divergence(a_k, fill(1 / (loss.K + 1), 1, loss.K + 1))

function jensen_shannon_divergence(p, q)
    ϵ = 1e-3 # to avoid log(0)
    return 0.5 * (kldivergence(p.+ϵ, q.+ϵ) + kldivergence(q.+ϵ, p.+ϵ))
end;

"""
    sigmoid(ŷ, y)

    Sigmoid function centered at y.
"""
function sigmoid(ŷ, y)
    return sigmoid_fast.((ŷ-y)*10)
end;

"""
    ψₘ(y, m)

    Bump function centered at m. Implemented as a gaussian function.
"""
function ψₘ(y, m)
    stddev = 0.1
    return exp.((-0.5 .* ((y .- m) ./ stddev) .^ 2))
end

"""
    ϕ(yₖ, yₙ)

    Sum of the sigmoid function centered at yₙ applied to the vector yₖ.
"""
function ϕ(yₖ, yₙ)
    return sum(sigmoid.(yₙ, yₖ))
end;

"""
    γ(yₖ, yₙ, m, K)
    
    Calculate the contribution of ψₘ ∘ ϕ(yₖ, yₙ) to the m bin of the histogram (Vector{Float}).
"""
function γ(yₖ, yₙ, m, K)
    eₘ(m) = [j == m ? 1.0 : 0.0 for j in 0:K-1]
    return eₘ(m) * ψₘ(ϕ(yₖ, yₙ), m)
end;

"""
    γ_fast(yₖ, yₙ, m, K)

Apply the γ function to the given parameters. 
This function is faster than the original γ function because it uses StaticArrays.
However because Zygote does not support StaticArrays, this function can not be used in the training process.
"""
function γ_fast(yₖ, yₙ, m, K)
    eₘ(m) = SVector{K, Float64}(j == m ? 1.0 : 0.0 for j in 0:K-1)
    return eₘ(m) * ψₘ(ϕ(yₖ, yₙ), m)
end;

"""
    generate_aₖ(loss, ŷ, y)

    Generate a one step histogram (Vector{Float}) of the given vector ŷ of K simulted observations and the real data y.
    generate_aₖ(loss, ŷ, y) = ∑ₖ γ(ŷ, y, k)
"""
generate_aₖ(loss, ŷ, y) = sum([γ(ŷ, y, k, loss.K+1) for k in 0:loss.K])