using AdaptativeBlockLearning, Test

tol = 1e-5

@testset "sigmoid" begin
    @test all(_sigmoid([2.6 2.3], 2.) .< [.5 .5])
    @test !all(_sigmoid([2.6 2.3], 2.4) .< [.5 .5])
    @test all(_sigmoid([2.6 2.3], 2.7) .> [.5, .5])
    @test all(_sigmoid([2.6f0 2.3f0], 2.0f0) .< [.5f0, .5f0])
    @test !all(_sigmoid([2.6 2.3], 2.4) .< [.5f0 .5f0])
    @test all(_sigmoid([2.6f0 2.3f0], 2.7f0) .> [.5f0, .5f0])
end;

@testset "ψₘ" begin
    @test ψₘ(1.0, 1) == 1.0
    @test ψₘ(1.5, 1) < 1
    @test ψₘ(0.5, 1) < 1
    #@test isapprox(ψₘ([1.0f0, 2.0f0, 0.0f0], 1), [1.0, 0.0, 0.0], atol=tol)
    #@test isapprox(ψₘ([1.0, 2.0, 0.0], 1), [1.0, 0.0, 0.0], atol=tol)
end;

@testset "ϕ" begin
    @test ϕ([1.0f0 2.0f0 3.1f0 3.9f0], 2.4f0) > 1.0f0
    @test ϕ([1.0f0 2.0f0 3.1f0 3.9f0], 2.4f0) < 3.0f0
    @test ϕ([1.0f0 2.0f0 3.1f0 3.9f0], 3.4f0) > 2.0f0
    @test ϕ([1.0 2.0 3.1 3.9], 2.4) > 1.0
    @test ϕ([1.0 2.0 3.1 3.9], 2.4) < 3.0
    @test ϕ([1.0 2.0 3.1 3.9], 3.4) > 2.0
end;

@testset "γ" begin
    #@test isapprox(γ([1.0, 2.0, 3.1, 3.9], 3.6, 3), [0.0, 0.0, 0.0, 0.9997, 0.0], atol=tol)
    @test isapprox(
        γ([1.0f0 2.0f0 3.1f0 3.9f0], 3.6f0, 3), [0.0, 0.0, 0.0, 0.9997, 0.0], atol=tol
    )
    @test isapprox(γ([1.0 2.0 3.1 3.9], 3.6, 3), [0.0, 0.0, 0.0, 0.9997, 0.0], atol=tol)
end;

@testset "generate aₖ" begin
    #@test isapprox(
    #    generate_aₖ([1.0, 2.0, 3.1, 3.9], 3.6), [0.0, 0.0, 0.0, 0.9997, 0.0], atol=tol
    #)
    @test isapprox(
        generate_aₖ([1.0f0 2.0f0 3.1f0 3.9f0], 3.6f0),
        [0.0, 0.0, 0.0, 0.9997, 0.0],
        atol=tol,
    )
    @test isapprox(
        generate_aₖ([1.0 2.0 3.1 3.9], 3.6), [0.0, 0.0, 0.0, 0.9997, 0.0], atol=tol
    )
end;

@testset "scalar diff" begin
    yₖ = [1. 2. 3. 4.]
    data = 0.5:0.5:4.5
    aₖ = zeros(5)
    for y in data
        aₖ += generate_aₖ(yₖ, y)
    end
    @test isapprox(scalar_diff(aₖ), 3.20004, atol=tol)
end;

@testset "jensen shannon divergence" begin
    @test jensen_shannon_divergence([1., 2.],[1., 2.]) == 0.
    @test jensen_shannon_divergence([1., 2.],[1., 3.]) > 0.
    @test jensen_shannon_divergence([1., 2.],[1., 3.]) < jensen_shannon_divergence([1., 2.],[1., 4.])
    @test jensen_shannon_divergence([1., 3.],[1., 2.]) == jensen_shannon_divergence([1., 2.],[1., 3.])
    @test jensen_shannon_divergence([0., 3.],[1., 3.]) > 0.
end;

@testset "jensen shannon ∇" begin
    yₖ = [1. 2. 3. 4.]
    data = 0.5:0.5:4.5
    aₖ = zeros(5)
    for y in data
        aₖ += generate_aₖ(yₖ, y)
    end
    @test isapprox(jensen_shannon_∇(aₖ./sum(aₖ)), 0., atol=tol)
end;
