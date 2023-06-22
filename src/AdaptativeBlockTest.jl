function get_window_of_Aₖ(model, target , K, n_samples)
    count.([model.(rand(Normal(μ, stddev), K)') .< target(rand(Normal(μ, stddev))) for _ in 1:n_samples])
end;

