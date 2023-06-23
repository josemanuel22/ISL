function get_window_of_Aₖ(model, target , K, n_samples)
    count.([model.(rand(Normal(μ, stddev), K)') .< target(rand(Normal(μ, stddev))) for _ in 1:n_samples])
end;

function convergence_to_uniform(aₖ)
    return pvalue(ChisqTest(aₖ, fill(1/(K+1), K+1)))
end;
