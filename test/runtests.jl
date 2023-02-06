using ContinuousTimeEconTools,
      Test,
      LinearAlgebra,
      SparseArrays

testfiles = ["regression.jl", ]#"exponential_utility.jl"]
for f in testfiles
    @testset verbose = true "$(f)" begin
        include(f)
    end
end
