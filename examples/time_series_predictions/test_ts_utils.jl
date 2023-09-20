using Test

tol = 1e-4

include("./ts_utils.jl")

@testset "ND" begin
    @test ND([1, 2, 3], [1, 2, 3]) == 0.0
    @test ND([1, 2, 4], [1, 2, 3]) == 1 / 7
    @test ND([1, 2, 3], [1, 2, 4]) == 1 / 6
    @test ND([2, 3, 7], [4, 2, 5]) == 5 / 12
    @test ND([2, -3, 7], [4, -2, 5]) == 5 / 12
end;

@testset "RMSE" begin
    @test RMSE([1, 2, 3], [1, 2, 3]) == 0.0
    @test RMSE([1, 2, 3], [2, 3, 4]) == sqrt(3)/2
    @test RMSE([1, 2, 3, 5], [2, 3, 4, 0]) == 8*sqrt(7)/11
    @test RMSE([1, 2, 3, -5], [2, 3, 4, 0]) == 8*sqrt(7)/11
end;

@testset "QLρ" begin
    @test QLρ([1, 2, 3], [1, 2, 3]) == 0.0
    @test QLρ([1, 2, 3], [1, 2, 3]; ρ=0.9) == 0.0
    @test QLρ([2, 3, 7], [4, 2, 5]; ρ=0.5) == 5 / 12
    x = rand(5)
    x̂ = rand(5)
    @test QLρ(x, x̂; ρ=0.5) == ND(x, x̂)

    x = .-rand(5)
    x̂ = .-rand(5)
    @test QLρ(x, x̂; ρ=0.5) == ND(x, x̂)

    @test QLρ([2, 3, 7], [4, 2, 5]; ρ=0.9) ≈ 29 / 60
end;

@testset "yule_walker" begin
    rho, sigma = yule_walker([1.0, 2, 3]; order=1)
    @test rho == [0.0]

    rho, sigma = yule_walker([1.0, 2, 3]; order=2)
    @test rho == [0.0, -1.5]

    x = [0.9901178, -0.74795127, 0.44612542, 1.1362954, -0.04040932]
    rho, sigma = yule_walker(x; order=3, method="mle")
    @test rho ≈ [-0.9418963, -0.90335955, -0.33267884]
    @test sigma ≈ 0.44006365345695164

    rho, sigma = yule_walker(x; order=3)
    @test isapprox(rho, [0.10959317, 0.05242324, 1.06587676], atol=tol)
    @test isapprox(sigma, 0.15860522671108127, atol=tol)

    rho, sigma = yule_walker(x; order=5, method="mle")
    @test isapprox(
        rho, [-1.24209771, -1.56893346, -1.16951484, -0.79844781, -0.27598787], atol=tol
    )
    @test isapprox(sigma, 0.3679474002175471, atol=tol)

    x = [
        0.9901178,
        -0.74795127,
        0.44612542,
        1.1362954,
        -0.04040932,
        0.28625813,
        0.88901716,
        -0.1079814,
        -0.33231995,
        0.4607741,
    ]

    rho, sigma = yule_walker(x; order=3, method="mle")
    @test isapprox(
        rho, [-0.4896151627237206, -0.5724647370433921, 0.09083516892540627], atol=tol
    )
    @test isapprox(sigma, 0.4249693094713215, atol=tol)

    x = [
        0.9901178,
        -0.74795127,
        0.44612542,
        1.1362954,
        -0.04040932,
        0.28625813,
        0.88901716,
        -0.1079814,
        -0.33231995,
        0.4607741,
        0.7729643,
        -1.0998684,
        1.098167,
        1.0105597,
        -1.3370227,
        1.239718,
        -0.01393661,
        -0.4790918,
        1.5009186,
        -1.1647809,
    ]

    rho, sigma = yule_walker(x; order=3, method="mle")
    @test isapprox(rho, [-0.82245705, -0.57029742, 0.12166898], atol=tol)
    @test isapprox(sigma, 0.5203501608988023, atol=tol)

    rho, sigma = yule_walker(x; order=3)
    @test isapprox(rho, [-0.93458149, -0.68653741, 0.10161722], atol=tol)
    @test isapprox(sigma, 0.4269012058667671, atol=tol)

    rho, sigma = yule_walker(x; order=5, method="mle")
    @test isapprox(
        rho, [-0.83107755, -0.56407764, 0.20950143, 0.1232321, 0.10249279], atol=tol
    )
    @test isapprox(sigma, 0.5172269743102993, atol=tol)

    rho, sigma = yule_walker(x; order=5)
    @test isapprox(
        rho, [-0.96481241, -0.65359486, 0.31587079, 0.28403115, 0.1913565], atol=tol
    )
    @test isapprox(sigma, 0.41677565377507053, atol=tol)
end;
