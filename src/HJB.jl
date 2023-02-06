abstract type HJBMethod end
struct Explicit <: HJBMethod end
struct Implicit <: HJBMethod end

struct HJBIterator{T, M}
    ρ::T
    Δ::T
    method::M
end

function (HJB::HJBIterator{T, Explicit})(v0, v1, r, A) where {T}
    v0, v1, r = vec(v0), vec(v1), vec(r)
    v0 .= r
    mul!(v0, (HJB.ρ * I - A), v1, 1, -1)
    # now v0 contains (1/Δ)(v1 - v)
    v0 .*= -HJB.Δ
    v0 .+= v1
    return v0
end

function (HJB::HJBIterator{T, Explicit})(v0, v1, vt, r, A, λ) where {T}
    v0, v1, vt, r = vec(v0), vec(v1), vec(vt), vec(r)
    v0 .= r .+ λ .* vt
    mul!(v0, ((HJB.ρ + λ) * I - A), v1, 1, -1)
    # now v0 contains (1/Δ)(v1 - v)
    v0 .*= -HJB.Δ
    v0 .+= v1
    return v0
end

function (HJB::HJBIterator{T, Implicit})(v0, v1, r, A) where {T}
    v0, v1, r = vec(v0), vec(v1), vec(r)
    v0 .= r .+ (1 / HJB.Δ) .* v1
    ldiv!(factorize((HJB.ρ + 1 / HJB.Δ) * I - A), v0)
    return v0
end

function (HJB::HJBIterator{T, Implicit})(v0, v1, vt, r, A, λ) where {T}
    v0, v1, vt, r = vec(v0), vec(v1), vec(vt), vec(r)
    # saves allocations compared to just setting r = r + λ .* vt
    v0 .= r .+ (1 / HJB.Δ) .* v1 .+ λ .* vt
    ldiv!(factorize((HJB.ρ + λ + 1 / HJB.Δ) * I - A), v0)
    return v0
end
