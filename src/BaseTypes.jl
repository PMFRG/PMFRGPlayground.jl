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

"""
"Sitepair" indices are in Sitesum, in "System/Geometry objects"
"""
abstract type SitePair end

"""
"Site" indices are in PairTypes, in "System/Geometry objects"
"""
abstract type Site end



abstract type FlowParameter end



struct Flavour
    f::Int64
end

flavours(nflavours) = (Flavour(i) for i in 1:nflavours)


#
abstract type MatsubaraF end



struct FermionicMatsubaraF
    w::Float64 # Is this needed, or the other data member is sufficient?
    idx::Int64 # Is this needed, or the other data member is sufficient?
end

struct BosonicMatsubaraF
    w::Float64 # Is this needed, or the other data member is sufficient?
    idx::Int64 # Is this needed, or the other data member is sufficient?
end

Base.+(wa::FermionicMatsubaraF,wb::FermionicMatsubaraF) = BosonicMatsubaraF(wa.w+wb.w)




MatsubaraF(nw::Int) = MatsubaraF(pi*(2*nw+1),nw) # TODO: check, in the Lambda flow this also depends on T
# Basic information
Base.:(+)(w1::MatsubaraF, w2::MatsubaraF) = MatsubaraF(w1.w+w2.w,(w1.idx-1)+(w2.idx-1)+1)
Base.:(-)(w1::MatsubaraF, w2::MatsubaraF) = MatsubaraF(w1.w-w2.w,(w1.idx-1)-(w2.idx-1)+1)

# TODO: fixme, this might depend on the temperature,
#       but we don't know the temperature now.
matsubaras(N) = (MatsubaraF(i,i) for i in -N:N)


# TODO: think - we might need conversion functions
#       between matsubara frequencies and the corresponding
#       integer index in the data structure,
#       or we can bake that into the structures.
#       See above TODO

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
    v::Array{T,4}
end
Base.getindex(s::SigmaBase, f1::Flavour, f2::Flavour, i::Site,w::MatsubaraF ) = s.v[f1.f,f2.f,i.idx,w]
Base.getindex(s::SigmaBase, ::Colon, ::Colon, i::Site,w::MatsubaraF) = s.v[:,:,i.idx,w]


# Tentative, in another file that is specific to spin 3/2...
struct Sigma3half{T} <: AbstractSigma{T}
    v::Array{T,2}
end


"""

"""
abstract type AbstractGamma end

"""
The 4-frequency version of Gamma can be re-expressed
in terms of the 3-frequency version.
"""
function (G::AbstractGamma)(a::Flavour, b::Flavour, c::Flavour, d::Flavour,
                            ij::SitePair,
                            wa::MatsubaraF, wb::MatsubaraF, wc::MatsubaraF, wd::MatsubaraF)

    let s = wa + wb, # PRB 103, 104431 Eq (22)
        t = wa + wc,
        u = wa + wd

        G(a,b,c,d, # Flavours
          ij, # sitepair
          s,t,u) # frequencies
    end
end


end
