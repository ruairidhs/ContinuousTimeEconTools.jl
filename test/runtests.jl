using UpwindDifferences
using Test,
      LinearAlgebra,
      SparseArrays

@testset "Exponential utility" begin
    θ, y, ρ = 2.0, 1.0, 0.05
    reward(x, c) = -(1/θ) * exp(-θ * c)
    drift(x, c) = y - c
    policy(x, dv) = -(1/θ) * log(dv)
    zerodrift(x) = y

    function backwards_iterate!(v0, v1, dt, A, UR, xs)
        upwind!(UR, v1, xs, reward, drift, policy, zerodrift)
        policy_matrix!(A, UR)
        v0 .= ((ρ + 1 / dt) * I - A) \ (UR.R .+ v1 ./ dt)
        return v0
    end

    function vf_fixed_point(xs)
        n = length(xs)
        v0 = @. (1/ρ) * log(xs + 1.0)
        UR = UpwindResult(v0)
        A = sparse(Tridiagonal(zeros(n-1), zeros(n), zeros(n-1)))

        cache = copy(v0)
        err = 1.0
        iter = 1
        dt = 0.1
        while iter <= 20
            backwards_iterate!(cache, v0, dt, A, UR, xs)
            err = maximum(abs.(cache .- v0))
            v0 .= cache
            iter += 1
        end
        dt = 10.0
        err = 1.0
        iter = 1
        while iter <= 250 && err >= 1e-12
            backwards_iterate!(cache, v0, dt, A, UR, xs)
            err = maximum(abs.(cache .- v0))
            v0 .= cache
            iter += 1
        end
        return v0
    end

    # use a non-regular grid for testing
    xs = zeros(10000)
    xs[2] = 0.0001
    γ = (40.0 / xs[2]) ^ (1 / (length(xs)-1))
    for i in 3:length(xs)
        xs[i] = xs[i-1] * γ
    end

    v = vf_fixed_point(xs)
    function get_numeric_policy(v, xs)
        n = length(xs)
        dvs = zeros(n)
        for i in eachindex(dvs)
            (i == 1 || i == n) && continue
            dvs[i] = (v[i+1] - v[i-1]) / (xs[i+1] - xs[i-1])
        end
        return [policy(x, dv) for (x, dv) in zip(xs[2:n-1], dvs[2:n-1])]
    end

    function get_analytic_policy(xs)
        return y .+ sqrt.(2 * (ρ / θ) * xs) 
    end
    rel_errs = get_numeric_policy(v, xs) ./ get_analytic_policy(xs)[2:end-1] .- 1
    @test maximum(abs.(rel_errs)) <= 0.001
end
