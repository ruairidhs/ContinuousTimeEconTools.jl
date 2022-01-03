using UpwindDifferences
using Test

using SparseArrays
using LinearAlgebra
using DelimitedFiles
using ProgressMeter

function evaluateparameters(ps)
    θ, ρ, r, y, dt = ps[1], ps[2], ps[3], ps[4], ps[5]
    xs = LinRange(0.01, 10.0, 1000)

    reward(x, c) = c ^ (1.0 - θ) / (1.0 - θ) 
    drift(x, c) = r * x + y - c
    policy(x, dv) = dv ^ (-1.0 / θ)
    zerodrift(x) = r * x + y

    R, GF, GB = zeros(1000), zeros(1000), zeros(1000)
    bs = (zeros(1000-1), zeros(1000-1))
    dv = zeros(1000-1)

    function backwardsiterate!(v0, v1)
        #upwind!(R, GF, GB, bs, dv, v1, xs, reward, drift, policy, zerodrift)
        upwind!(R, GF, GB, v1, xs, reward, drift, policy, zerodrift)
        A = sparse(Tridiagonal(-GB[2:end] ./ step(xs), (GB .- GF) ./ step(xs), GF[1:end-1] ./ step(xs)))
        v0 .= ((ρ + 1.0 / dt) * I - A) \ (R .+ v1 ./ dt)
        return v0
    end

    vinit = map(x -> reward(x, zerodrift(x)) ./ ρ, xs)
    vs = (copy(vinit), copy(vinit))
    tol, maxiter = 1e-10, 10000
    err = tol
    for iter in 1:maxiter
        backwardsiterate!(vs[2], vs[1])
        err = maximum(abs.(vs[2] .- vs[1]))
        if err < tol
            return (value = vs[2], iter = iter, err = err, termination = :tolerance) 
        else
            vs[1] .= vs[2]
        end
    end
    return (value = vs[1], iter = maxiter, err = err, termination = :iterations)
end

@testset "UpwindDifferences.jl" begin

    allparameters = readdlm("testdata/hjb_parameters.tsv")
    allvalues = readdlm("testdata/hjb_values.tsv")
    @showprogress for i in axes(allparameters, 1)
        ps = allparameters[i, :]
        vnew = evaluateparameters(ps).value
        vold = allvalues[i, :]
        @test maximum(abs.(vnew .- vold)) < 1e-8
    end
end
