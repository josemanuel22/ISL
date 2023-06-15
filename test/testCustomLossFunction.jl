using AdaptativeBlockLearning:sigmoid,ψₘ,ϕ, γ, CustomLoss, generate_aₖ
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
    @test isapprox(γ([1.0, 2.0, 3.1, 3.9], 3.6, 3, 5), [0., 0. , 0.92038, 0., 0.], atol=tol)
end

@testset "generate aₖ" begin
    loss = CustomLoss(4)
    @test  isapprox(generate_aₖ(loss, [1.0, 2.0, 3.1, 3.9], 3.6), [0., 0., 0.92038, 0., 0.], atol=tol)
end