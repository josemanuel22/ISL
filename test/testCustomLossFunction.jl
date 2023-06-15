using AdaptativeBlockLearning
using Test

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
    @test γ([1.0, 2.0, 3.1, 3.9], 3.6, 3, 5) ≈ [0., 0. , 0.9203893456369119, 0., 0.]
end
