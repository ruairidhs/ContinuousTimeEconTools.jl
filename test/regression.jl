# Test CRRA over Cobb-Douglas utility in multiple dimensions with productivity and housing
using LinearAlgebra
using ContinuousTimeEconTools
using JLD2

## Model specification
function make_grids(specs)
    (xg = range(specs.xg[1]...; length = specs.xg[2]),
     zg = range(specs.zg[1]...; length = specs.zg[2]),
     hg = range(specs.hg[1]...; length = specs.hg[2]),
    )
end

# utility
g(c, h, αh) = h ^ αh * c ^ (1 - αh)
u(x) = log(x)

function reward(_, c, h, params) # h is housing services consumption
    u(g(c, h, params.αh))
end

function policy(_, dv, params)
    base = (1 - params.αh) / dv
    return base
end

function netincome_own(x, z, h, params)
    (; r, y, s) = params
    return r * x + z * y - s * h
end

function netincome_rent(x, z, params)
    (; ϕ, r, ps, s, hs, y) = params
    rent = ϕ * (r * ps + s) * hs
    return r * x + z * y - rent
end

function drift_rent(x, c, z, params)
    return netincome_rent(x, z, params) - c
end

function drift_own(x, c, z, h, params)
    return netincome_own(x, z, h, params) - c
end

function make_Az(zg, λ)
    λi = λ / (length(zg) - 1)
    Az = ones(length(zg), length(zg)) .* λi
    Az[diagind(Az)] .= -λ
    return Az
end

## Value function iteration
function solve_value_own(h, grids, params, Δ)
    (; xg, zg) = grids
    Az = make_exogenous_transition(length(xg), [make_Az(zg, params.λ)]) # expands Az to the x grid
    vinit = (1 / params.ρ) .* log.(xg .+ 1.0)
    vinit = repeat(vinit, 1, length(zg)) # now an nx × nz matrix

    hjb_funcs = ((x, c, zi) -> reward(x, c, h, params),
                 (x, dv, zi) -> policy(x, dv, params),
                 (x, c, zi) -> drift_own(x, c, exp(zg[zi]), h, params),
                 (x, zi) -> netincome_own(x, exp(zg[zi]), h, params),
                )

    iterator = HJBIterator(params.ρ, Δ, Implicit())

    res = invariant_value_function(vinit, xg, Az, hjb_funcs, iterator)
    if res.status != :converged
        error("VF did not converge: err: $(res.err), iters: $(res.iter)")
    else
        return res.value
    end
end

function solve_value_rent(grids, params, Δ)
    (; xg, zg) = grids
    Az = make_exogenous_transition(length(xg), [make_Az(zg, params.λ)]) # expands Az to the x grid
    vinit = (1 / params.ρ) .* log.(xg .+ 1.0)
    vinit = repeat(vinit, 1, length(zg)) # now an nx × nz matrix

    hjb_funcs = ((x, c, zi) -> reward(x, c, params.hs, params),
                 (x, dv, zi) -> policy(x, dv, params),
                 (x, c, zi) -> drift_rent(x, c, exp(zg[zi]), params),
                 (x, zi) -> netincome_rent(x, exp(zg[zi]), params),
                )

    iterator = HJBIterator(params.ρ, Δ, Implicit())

    res = invariant_value_function(vinit, xg, Az, hjb_funcs, iterator)
    if res.status != :converged
        error("VF did not converge: err: $(res.err), iters: $(res.iter)")
    else
        return res.value
    end
end

function solve_value(grids, params, Δ)
    own = mapreduce(h -> solve_value_own(h, grids, params, Δ), (x, y) -> cat(x, y; dims=3), grids.hg)
    rent = solve_value_rent(grids, params, Δ)
    return (rent = rent, own = own)
end

## Regression data generation
GENERATE = false
if GENERATE
    grid_spec = (xg = ((0.0, 5.0), 100), # bounds and length
                 zg = ((-0.2, 0.2), 5),
                 hg = ((0.6, 0.8), 5),
                )

    params = (αh = 0.5, r = 0.02, ρ = 0.05, s = 0.6,
              y = 1.0, ps = 5.0, pb = 5.1, ϕ = 1.2, hs = 0.40,
              λ = 0.05 # total probability of switching productivity state
             )

    value = solve_value(make_grids(grid_spec), params, 5.0)
    jldsave("regression_data/test_1.jld2"; grid_spec, params, value)
end

## Test against the data
loaded = load("regression_data/test_1.jld2")
new_value = solve_value(make_grids(loaded["grid_spec"]), loaded["params"], 5.0)
pc_diff(u, v) = ((u ./ v) .- 1) .* 100

rent_error = pc_diff(new_value.rent, loaded["value"].rent)
@test maximum(abs, rent_error) < 0.1
own_error = pc_diff(new_value.own, loaded["value"].own)
@test maximum(abs, own_error) < 0.1
