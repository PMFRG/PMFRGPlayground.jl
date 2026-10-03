"""
Abstract and fundamental types that are used throughout the code.

Notice that we provide types
that would be representable with a single float or integer,
but they represent values that have special meanings
and must not be confused when passing them to functions.

Using types in this way can prevent a whole class of bugs.

A functions that takes a Matsubara frequency and a temperature,
for example,
could be implemented using two Float64 arguments,
but it would be easy to swap the two arguments by mistake
and Julia would not give any error.

When using two different types for these parameters,
we make sure that if we mistakenly swap them
we get a (pre) compilation error.
"""
module BaseTypes

abstract type FlowParameter end


struct Flavour
    f::Int64
end

flavours(nflavours) = (Flavour(i) for i in 1:nflavours)

struct MatsubaraF
    w::Float64
end

# Basic information
Base.:(+)(w1::MatsubaraF, w2::MatsubaraF) = MatsubaraF(w1.w+w2.w)
Base.:(-)(w1::MatsubaraF, w2::MatsubaraF) = MatsubaraF(w1.w-w2.w)

matsubaras(N) = (MatsubaraF(i) for i in -N:N)


# TODO: think - we might need conversion functions
#       between matsubara frequencies and the corresponding
#       integer index in the data structure,
#       or we can bake that into the structures.

struct Temperature <: FlowParameter
    T::Float64
end

MatsubaraF(T::Temperature,n::Int) = pi * T * (2n+1)

struct Lambda <: FlowParameter
    L::Float64
end

"""
This datatype concerns SpinFRGLattices
but it is not provided there
(because it is "structurally" identical to an integeer)
We provide it here because
it is semantically different.
"""
struct Site
    idx::Int64
end

"""
Abstract type for the data structure
containing Sigma,
possibly Complex.

Subtypes need to implement:

- Base.getindex(::SigmaBase,::Flavour,::Flavour,::Site)::Number
- Base.getindex(::SigmaBase,::Colon,::Colon,::Site)::AbstractMatrix
"""
abstract type AbstractSigma{T}
end


"""
Base, unoptimized version of Sigma,
to be used for testing purposes.
"""
struct SigmaBase{T} <: AbstractSigma{T}
    v::Array{T,3}
end
Base.getindex(s::SigmaBase, f1::Flavour, f2::Flavour, i::Site) = s.v[f1.f,f2.f,i.idx]
Base.getindex(s::SigmaBase, ::Colon, ::Colon, i::Site) = s.v[:,:,i.idx]

end
