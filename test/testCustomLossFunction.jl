using AdaptativeBlockLearning, Test

tol = 1e-5

@testset "sigmoid" begin
    @test sigmoid(1.4, 2) < 0.5
    @test sigmoid(2.6, 2.0) > 0.5
    @test sigmoid([2.6, 2.3], [2.0, 1.5]) > [0.5, 0.5]
end;

@testset "ψₘ" begin
    @test ψₘ(1., 1.) == 1.
    @test ψₘ(1.5, 1.) < 1
    @test ψₘ(0.5, 1.) < 1
end;

@testset "ϕ" begin
    @test ϕ([1.0, 2.0, 3.1, 3.9], 2.4) > 1.
    @test ϕ([1.0, 2.0, 3.1, 3.9], 2.4) < 3.
    @test ϕ([1.0, 2.0, 3.1, 3.9], 3.4) > 2.
end;

@testset "γ" begin
    @test isapprox(γ([1.0, 2.0, 3.1, 3.9], 3.6, 3), [0., 0. , 0., 0.92038, 0.], atol=tol)
end;

@testset "generate aₖ" begin
    @test isapprox(generate_aₖ([1.0, 2.0, 3.1, 3.9], 3.6), [0., 0., 0., 0.92038, 0.], atol=tol)
end;

@testset "scalar diff" begin
    yₖ = [1., 2., 3., 4.]
    data = 0.5:0.5:4.5
    aₖ = zeros(5)
    for y in data
        aₖ += generate_aₖ(yₖ, y)
    end
    @test isapprox(scalar_diff(aₖ), 3.19289, atol=tol)
end;

@testset "jensen shannon divergence" begin
    @test jensen_shannon_divergence([1.,2.],[1.,2.]) == 0.
    @test jensen_shannon_divergence([1.,2.],[1.,3.]) > 0.
    @test jensen_shannon_divergence([1.,2.],[1.,3.]) < jensen_shannon_divergence([1.,2.],[1.,4.])
    @test jensen_shannon_divergence([1.,3.],[1.,2.]) == jensen_shannon_divergence([1.,2.],[1.,3.])
    @test jensen_shannon_divergence([0.,3.],[1.,3.]) > 0.
end;

@testset "jensen shannon ∇" begin
    yₖ = [1., 2., 3., 4.]
    data = 0.5:0.5:4.5
    aₖ = zeros(5)
    for y in data
        aₖ += generate_aₖ(yₖ, y)
    end
    @test isapprox(jensen_shannon_∇(aₖ./sum(aₖ)), 0., atol=tol)
end;

