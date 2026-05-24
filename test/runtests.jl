using ContinuousTimeEconTools, Test, LinearAlgebra, SparseArrays

testfiles = ["exponential_utility_1d.jl", "log_utility_1d.jl", "regression.jl"]
@testset verbose = true "ContinuousTimeEconTools" begin
    for f in testfiles
        @testset verbose = true "$(f)" begin
            include(f)
        end
    end
end
