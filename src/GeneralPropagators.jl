"""
If time-reversal symmetry holds,
then Sigma is purely imaginary,
and the same can be said for the propagators.

In the most general approach,
we do not yet assume that.
"""
module Propagators
export G_, iG_
import ..BaseTypes: Flavour, MatsubaraF,Temperature, Lambda, FlowParameter, AbstractSigma, Site
import LinearAlgebra: I
using DocStringExtensions

"""
    $(TYPEDSIGNATURES)

    Version of G returning a single value (possibly complex).
"""
G_(f1::Flavour, f2::Flavour, i::Site, w::MatsubaraF, x::FlowParameter, Sigma::AbstractSigma) = G_(i,w,x,Sigma)[f1.f,f2.f]
"""
    $(TYPEDSIGNATURES)

    Version of G returning a square matrix,
    with the number of flavours as dimension
    (possibly complex).
"""
G_(i::Site, w::MatsubaraF, x::FlowParameter,Sigma::AbstractSigma)::AbstractMatrix = inv(Ginv_(w,x,Sigma[:,:,i]))

"""
    $(TYPEDSIGNATURES)

    Version of \$G^{-1}\$ returning a square matrix,
    with the number of flavours as dimension
    (possibly complex).
    T-flow version.

"""
Ginv_(w::MatsubaraF, T::Temperature,Sigma::AbstractMatrix)::AbstractMatrix = im*w*sqrt(T.T)*I - Sigma
DFullGinv_(w::MatsubaraF, T::Temperature,DSigma::AbstractMatrix)::AbstractMatrix = (im  / 2sqrt(T.T))*I - DSigma
DGinv_(w::MatsubaraF, T::Temperature,_::AbstractMatrix)::AbstractMatrix = (im  / 2sqrt(T.T))*I

"""
    $(TYPEDSIGNATURES)

    Version of \$G^{-1}\$ returning a square matrix,
    with the number of flavours as dimension
    (possibly complex).
    Lambda-flow version.

    Note: in this case, `T` enters the equation only through w,
    which has a dependency on `T`: see the "constructor" `MatsubaraF(::Temperature,::Int)`
"""
Ginv_(w::MatsubaraF, L::Lambda,Sigma::AbstractMatrix{Real}) = im*(w+L.L^2/w)*I - Sigma
DFullGinv_(w::MatsubaraF, L::Lambda,DiSigma::AbstractMatrix{Real}) = 2im*L.L/w*I - DiSigma
DGinv_(w::MatsubaraF, L::Lambda,_::AbstractMatrix{Real}) = 2im*L.L/w*I

"""

   If time reversal symmetry is not broken,
   this real-valued function can be used of G_.
   It take the real-value argument `iSigma`.

   $(TYPEDSIGNATURES)

"""
iG_(f1::Flavour, f2::Flavour, i::Site, w::MatsubaraF, x::FlowParameter, iSigma::AbstractSigma) = iG_(i,w,x,iSigma)[f1,f2]

"$(TYPEDSIGNATURES)"
iG_(i::Site, w::MatsubaraF, x::FlowParameter,iSigma::AbstractSigma{Real}) = inv(iGinv_(w,x,iSigma[:,:,i]))


"$(TYPEDSIGNATURES)

  T-Flow method specialization
"
iGinv_(w::MatsubaraF, T::Temperature,iSigma::AbstractMatrix{Real}) = w*sqrt(T.T)*I + iSigma
DFulliGinv_(w::MatsubaraF, T::Temperature,DiSigma::AbstractMatrix{Real}) = w/2sqrt(T.T) + DiSigma
DiGinv_(w::MatsubaraF, T::Temperature,_::AbstractMatrix{Real}) = w/2sqrt(T.T)
"$(TYPEDSIGNATURES)


  Lambda-Flow method specialization.
  (T enters via the w parameter)
"
iGinv_(w::MatsubaraF, L::Lambda,iSigma::AbstractMatrix{Real}) = (w+L.L^2/w)*I + w*iSigma
DFulliGinv_(w::MatsubaraF, L::Lambda,DiSigma::AbstractMatrix{Real}) = (2L.L/w)*I + w*DiSigma
DiGinv_(w::MatsubaraF, L::Lambda,_::AbstractMatrix{Real}) = (2L.L/w)*I


# TODO S(f1::Flavour, f2::Flavour, i::Site, w::MatsubaraF, Sigma) = let G = G_(i,w,Sigma) end

"""
    $(TYPEDSIGNATURES)

    ```math
    S = -\frac{\partial G}{\partial x}
    ```
    with \$x\$ being the flow parameter.
"""
S(f1::Flavour, f2::Flavour, i::Site, w::MatsubaraF, x::FlowParameter, Sigma::AbstractSigma) = S(i,w,x,Sigma)[f1.f,f2.f]
S(i::Site, w::MatsubaraF,x::FlowParameter, Sigma::AbstractSigma) = let G = G_(i,w,x,Sigma)
    G * DGinv(i,w,x,Sigma) * G
end

"""
    $(TYPEDSIGNATURES)

    ```math
    iS = -\frac{\partial iG}{\partial x}
    ```
    with \$x\$ being the flow parameter.
"""
iS(f1::Flavour, f2::Flavour, i::Site, w::MatsubaraF, x::FlowParameter, iSigma::AbstractSigma{Real}) = iS(i,w,x,iSigma)[f1.f,f2.f]
iS(i::Site, w::MatsubaraF,x::FlowParameter, iSigma::AbstractSigma{Real}) = let iG = iG_(i,w,x,iSigma)
    -iG * DiGinv(i,w,x,iSigma) * iG
end


"""
    $(TYPEDSIGNATURES)

    ```math
    S = -\frac{\partial G}{\partial x}
    ```
    with \$x\$ being the flow parameter,
    without ignoring the derivative of Sigma w.r.t
    the flow parameter.
"""
SKat(f1::Flavour, f2::Flavour, i::Site, w::MatsubaraF, x::FlowParameter, Sigma::AbstractSigma, DSigma::AbstractSigma) = SKat(i,w,x,Sigma,DSigma)[f1.f,f2.f]
SKat(i::Site, w::MatsubaraF,x::FlowParameter, Sigma::AbstractSigma,DSigma::AbstractSigma) = let G = G_(i,w,x,Sigma)
    G * DFullGinv_(i,w,x,DSigma) * G
end

"""
    $(TYPEDSIGNATURES)

    ```math
    iS = -\frac{\partial iG}{\partial x}
    ```
    with \$x\$ being the flow parameter.
    without ignoring the derivative of Sigma w.r.t
    the flow parameter.
"""
iSKat(f1::Flavour, f2::Flavour, i::Site, w::MatsubaraF, x::FlowParameter, iSigma::AbstractSigma{Real},DiSigma::AbstractSigma{Real}) = iSKat(i,w,x,iSigma,DiSigma)[f1.f,f2.f]
iSKat(i::Site, w::MatsubaraF,x::FlowParameter, iSigma::AbstractSigma{Real}, DiSigma::AbstractSigma{Real}) = let iG = iG_(i,w,x,iSigma)
    -iG * DiGinv(i,w,x,DiSigma) * iG
end









end
