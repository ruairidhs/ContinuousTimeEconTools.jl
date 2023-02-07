using ContinuousTimeEconTools, Test, LinearAlgebra, SparseArrays

testfiles = ["regression.jl"]
@testset verbose = true "ContinuousTimeEconTools" begin
    for f in testfiles
        @testset verbose = true "$(f)" begin
            include(f)
        end
    end
end
