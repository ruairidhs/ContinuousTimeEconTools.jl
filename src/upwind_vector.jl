using LoopVectorization,
      BenchmarkTools,
      CairoMakie

T = Float64
const θ = 3.0
const y = 1.0

reward(_, c) = -(1/θ) * exp(-θ * c)
policy(_, dv::T) where {T} = ifelse(dv <= zero(T), 0.0001, -(1/θ) * log(dv))
drift(_, c) = y - c
zd(_) = y

n = 8 * 10_000 + 1
x = vcat([0.0], exp.(range(log(1e-6), log(1), length=n-1)))
v = convert.(T, x .* 5.0 .+ 3.0)
dv = zeros(T, n-1)
cache = zeros(T, n-1)
g, r = map(_ -> zeros(T, n-1), (1,2))
@assert length(dv) % 8 == 0
@assert length(dv) + 1 == length(x)

function vupwind!()
end

# I need to calculate dv
function dv!(dv, v, x)
    @turbo for i in 1:length(dv)
        dv[i] = (v[i+1] - v[i]) / (x[i+1] - x[i])
    end
    return nothing
end

# next step is computing drift and rewards
# first do forward
#dv[i] = dvf[i]; dv[i] = dvb[i+1]

# I need one cache for BF
function fill_forward!(r, g, cache, x, dv)
    x_slice = @view x[1:end-1]
    @views cache .= policy.(x_slice, dv)
    r .= reward.(x_slice, cache)
    g .= drift.(x_slice, cache)
    return nothing
end

dv!(dv, v, x)
@info "max error: $(maximum(abs.(dv .- 5.0)))"


@benchmark fill_forward!(r, g, cache, x, dv)

using LinearAlgebra

bs = 6
A1 = Tridiagonal(ones(bs-1), ones(bs), ones(bs-1))
A2 = 2 .* A1

A = [A1 zeros(bs, bs);
     zeros(bs, bs)  A2]

# and then some Poisson matrix:
Q = [-0.05 0.05;
     0.05 -0.05]

kron(Q, I(bs))

M = A + kron(Q, I(bs))
