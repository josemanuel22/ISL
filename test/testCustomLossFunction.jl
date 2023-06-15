using AdaptativeBlockLearning
using Test

tol = 1e-5

@testset "sigmoid" begin
    @test sigmoid(1.4, 2) < 0.5
    @test sigmoid(2.6, 2.0) > 0.5
    @test sigmoid([2.6, 2.3], [2.0, 1.5]) > [0.5, 0.5]
end

@testset "ψₘ" begin
    @test ψₘ(1., 1.) == 1.
    @test ψₘ(1.5, 1.) < 1
    @test ψₘ(0.5, 1.) < 1
end

@testset "ϕ" begin
    @test ϕ([1.0, 2.0, 3.1, 3.9], 2.4) > 1.
    @test ϕ([1.0, 2.0, 3.1, 3.9], 2.4) < 3.
    @test ϕ([1.0, 2.0, 3.1, 3.9], 3.4) > 2.
end

@testset "γ" begin
    @test isapprox(γ([1.0, 2.0, 3.1, 3.9], 3.6, 3, 5), [0., 0. , 0., 0.92038, 0.], atol=tol)
end

@testset "generate aₖ" begin
    loss = CustomLoss(4)
    @test  isapprox(generate_aₖ(loss, [1.0, 2.0, 3.1, 3.9], 3.6), [0., 0., 0., 0.92038, 0.], atol=tol)
end

@testset "scalar diff" begin
    loss = CustomLoss(4)
    yₖ = [1, 2, 3, 4]
    data = 0:0.5:5
    aₖ = zeros(5)
    for y in data
        aₖ += generate_aₖ(loss, yₖ, y)
    end
    @test isapprox(scalar_diff(loss, aₖ), [3.23196, 0.64001, 0.64001, 0.64001, 3.23196], atol=tol)
end

@testset "jensen shannon divergence" begin
    @test jensen_shannon_divergence([1.,2.],[1.,2.]) == 0.
    @test jensen_shannon_divergence([1.,2.],[1.,3.]) > 0.
    @test jensen_shannon_divergence([1.,2.],[1.,3.]) < jensen_shannon_divergence([1.,2.],[1.,4.])
    @test jensen_shannon_divergence([1.,3.],[1.,2.]) == jensen_shannon_divergence([1.,2.],[1.,3.])
    @test jensen_shannon_divergence([0.,3.],[1.,3.]) > 0.
end
