"""
    outer_indices(A::AbstractArray)

Returns an iterator over all axes but the first.
"""
outer_indices(V::AbstractVector) = Base.Iterators.repeated((), 1)
outer_indices(A::AbstractArray{T, N}) where {T, N} =
    Iterators.product((axes(A, i) for i in 2:N)...)

"""
    supnorm(u, v)

Find the largest element-wise difference between two collections.
"""
function supnorm(u, v)
    return maximum(abs.(u .- v))
end

"""
    make_exogenous_transition(nx, Λs)

Build an exogenous state transition matrix.

# Details

Given an endogenous state length `nx` and exogenous transition matrices `Λs`,
compute the overall exogenous transition matrix.
"""
function make_exogenous_transition(nx, Λs)
    return foldl(
        (acc, M) -> kron(I(size(M, 1)), acc) + kron(sparse(M), I(size(acc, 1))),
        Λs;
        init = spzeros(nx, nx),
    )
end
