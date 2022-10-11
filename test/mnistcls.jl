using Test
using JuliaAlgos: MnistClassifier


@testset "Pass" begin
    @test true
end

@testset "one-hot" begin
    ipt = [2, 4, 6, 1]
    exp = Float32.([
        0.0 0.0 0.0 1.0
        1.0 0.0 0.0 0.0
        0.0 0.0 0.0 0.0
        0.0 1.0 0.0 0.0
        0.0 0.0 0.0 0.0
        0.0 0.0 1.0 0.0
        0.0 0.0 0.0 0.0
        0.0 0.0 0.0 0.0
        0.0 0.0 0.0 0.0
        0.0 0.0 0.0 0.0])
    @test MnistClassifier.onehot(ipt) == exp
end

@testset "one-cold" begin
    ipt = Float32.([
        0.0 0.0 0.0 1.0
        1.0 0.0 0.0 0.0
        0.0 0.0 0.0 0.0
        0.0 1.0 0.0 0.0
        0.0 0.0 0.0 0.0
        0.0 0.0 1.0 0.0
        0.0 0.0 0.0 0.0
        0.0 0.0 0.0 0.0
        0.0 0.0 0.0 0.0
        0.0 0.0 0.0 0.0])
    exp = [2, 4, 6, 1]
    @test MnistClassifier.onecold(ipt) == exp
end

@testset "model" begin
    x = rand(Float32, (28, 28, 1, 5))
    model = MnistClassifier.getmodel(28, 28, 1, 10, 2, 3, 32)
    @test x |> model |> size == (10, 5)
    ŷ = MnistClassifier.getpredictor(model)(x)
    @test ŷ |> size == (5,)
    @test ŷ |> eltype == Int64
end
