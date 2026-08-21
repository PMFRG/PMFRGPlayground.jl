include("Flavour.jl")
include("Matsubara.jl")
include("sitesum.jl")

# "Sitepair" indices are in Sitesum, in "System/Geometry objects"
abstract type SitePair end
# "Site" indices are in PairTypes, in "System/Geometry objects"
abstract type Site end


# Propagators
# Notice that these depend on Sigma
# and, in the case of S_kat,
# DSigma.

#! format: off
P_(f1::Flavour, f2::Flavour, f3::Flavour, f4::Flavour, # Flavours
  i::Site, j::Site, # Sites
  w::MatsubaraF, s::MatsubaraF, # Matsubaras
  Sigma, DSigma,

) = S_kat_(f1, f2, i, w, Sigma, DSigma) * G_(f3, f4, j, w + s, Sigma)
#! format: on

# TODO: take, e.g., from Yannik's PMFRG simplified code
G_(f1::Flavour, f2::Flavour, i::Site, w::MatsubaraF, Sigma) = nothing
S_kat_(f1::Flavour, f2::Flavour, i::Site, w::MatsubaraF, Sigma, DSigma) = nothing
S(f1::Flavour, f2::Flavour, i::Site, w::MatsubaraF, Sigma) = nothing


#
#
# Note: all these types are used to reduce the possibility
#       of passing arguments in the wrong order.
# Notes about the equations in latex:
# 1. P's need to have two frequency args
# 2. instead of 1,2,3,4 in Γs we shoul use ω₁ ...
# 3. multiply RHS by δ(ω1+ω2, ω3+ω4)
# And publish this somewhere
# where it can be referenced.

#! format: off
"""
General form of the derivative of the 4-point vertex, $\Gamma$,
with respect to the flow parameter $\Lambda$.

This equation has been derived using these symmetries:
- Time-translation invariance
- Hermiticity
- Local Z(2) gauge symmetry

This equation has not used the following symmetries:
- any flavour symmetry
- time reversal
- any particular total spin

Additional symmetries used
(because of delegating site summation to SpinFRGLattices.jl):
- Spatial translation invariance up to the unit cell

In this form it is given in terms
of the 4 frequencies $\omega_1$, ..., $\omega_4$,
which are not linearly independent.

# Arguments
- `f1::Flavour`, `f2::Flavour`, `f3::Flavour`, `f4::Flavour`: flavours indices for the 4 legs
- `ij::SitePair`: the site pair, as defined in the SpinFRGLattices.jl package
- `w1::MatsubaraF`, `w2::MatsubaraF`, `w3::MatsubaraF`, `w4::MatsubaraF`: Matsubara frequencies for the 4 legs
   (not linearly independent)
- `T::Temperature`: The value of the temperature
- `geometry`: Geometry object containing all the information on the lattice geometry (see `struct Geometry` from SpinFRGLattices.jl)
- `Gamma`: A function representing the 4-point vertex, with the signature
   ```julia
   Gamma(a::Flavour, b::Flavour, c::Flavour, c::Flavour, # flavours
       ij::SitePair, # sitepair
       wa::MatsubaraF, wb::MatsubaraF, wc::MatsubaraF, wd::MatsubaraF, # matsubara frequencies
       )
   ```
- `P`: A function representing the product of the 2-point green function G
       and the single-scale propagator S, with the signature
   ```julia
   P(a::Flavour, b::Flavour, c::Flavour, c::Flavour, # flavours
     i::Site, j::Site, # sites
     w::MatsubaraF,wb::MatsubaraF, # matsubara frequencies
     )

   ```



# Additional notes
## Space translation symmetry
This is embodied by the fact that we are using a "geometry"
object coming from the
[SpinFRGLattices.jl package](https://github.com/NilsNiggemann/SpinFRGLattices.jl.git)

## Additional references
This represents
Eq (4) in Noah's "PMFRG at Full Anisotropy",
or Eq (72) in Siebe's PM-FRG for Heisenberg Spin 3/2.


"""
DGamma_(a::Flavour, b::Flavour, c::Flavour, d::Flavour, # flavours
    ij::SitePair, # sitepair
    wa::MatsubaraF, wb::MatsubaraF, wc::MatsubaraF, wd::MatsubaraF, # matsubaras
    T::Temperature,
    geometry, matsubaras, flavours,
    Gamma, P) = let s = wa + wb, # PRB 103, 104431 Eq (22)
                    t = wa + wc,
                    u = wa + wd,
                    (; i, j) = geometry.PairTypes[ij],
                    ss = geometry.siteSum[:,ij],
                    is_on_site_pair = occursin(ij,geometry.OnSitePairs)

        if is_on_site_pair
        T*sum( # for w in matsubaras
            sum( # for ap,bp,cp,dp
                 @sitesum ss ik ki xk (
                     - Gamma(a, b, ap, dp, # flavours)
                             ik, # sitepair
                             wa,wb,w,-s-w, # matsubaras
                            ) *
                       Gamma(bp,cp,c,d, # flavours
                             ki, # sitepair
                             -w, s+w, wc, wd, # matsubaras
                            ) *
                       P(ap,bp,cp,dp, # flavours
                         xk,xk, # sites
                         -w, -s, # matsubaras
                        )
                     + Gamma(a,c,bp,cp, # flavours
                             ik, # sitepair
                             wa,wc,-w,w-t, # matsubaras
                          ) *
                       Gamma(ap,dp,b,d, # flavours
                             ki, # sitepair
                             w,t-w,wb,wd, # matsubaras
                            ) *
                       P(ap,bp,cp,dp, # flavours
                         xk,xk, # sites
                         -w,t, # matsubaras
                         )
                     - Gamma(a,d,ap,dp, # flavours
                             ik, # sitepair
                             wa,wd,w,-w-u, # matsubaras
                          ) *
                       Gamma(bp,cp,b,c, # flavours
                             ki, # sitepair
                             -w,u+w,wb,wc, # matsubaras
                            ) *
                       P(ap,bp,cp,dp, # flavours
                         xk,xk, # sites
                         -w,-u, # matsubaras
                         )
                     )
                 for ap in flavours, bp in flavours, cp in flavours, dp in flavours)
               for w in matsubaras)
        else
        T*sum( # for w in matsubaras
            sum( # for ap,bp,cp,dp
                (@sitesum ss ik kj xk - Gamma(a, b, ap, dp, # flavours
                                              ik, # sitepair
                                              wa, wb, w, -w-s, # matsubaras
                                             ) *
                                       Gamma(bp, cp, c, d, # flavours
                                             kj, # sitepair
                                             -w, w+s, wc, wd, # matsubaras
                                            ) *
                                       P(ap, bp, cp, dp, # flavours
                                         xk, xk, # sites
                                         -w, -s, # matsubara
                                        )
                 ) + (
                    Gamma(a, bp, c, cp, # flavours
                        ij, # sitepair
                        wa, -w, wc, w - t) # matsubaras
                    * Gamma(b, ap, d, dp, # flavours
                        ij, # sitepair
                        wb, w, wd, -w+t) # matsubaras
                    * P(ap, bp, cp, dp, # flavours
                        i, j,#sites
                        -w, +t) # matsubara
                    ####
                    + Gamma(a, cp, c, bp, # flavours
                        ij, # sitepair
                        wa, w-t, wc, -w) # matsubaras
                    * Gamma(b, dp, d, ap, # flavours
                        ij, # sitepair
                        wb, -w+t, wd, w) # matsubaras
                    * P(ap, bp, cp, dp, # flavours
                        j, i,#sites
                        -w, +t) # matsubara
                ) - (
                    Gamma(a, dp, d, ap, # flavours
                        ij, # sitepair
                        wa, -w-u, wd, w) # matsubaras
                    *Gamma(b, cp, c, bp,#flavours
                        ij, # sitepair
                        wb, w+u, wc, -w) # matsubaras
                    *P(ap, bp, cp, dp,# flavours
                        i, j, # sites
                        -w, -u) # matsubara
                    ####
                    + Gamma(a, ap, d, dp, # flavours
                        ij, # sitepair
                        wa, w, wd, -w-u) # matsubaras
                    * Gamma(b, bp, c, cp, # flavours
                        ij, # sitepair
                        wb, -w, 3, w+u) # matsubaras
                    * P(ap, bp, cp, dp, # flavours
                        j,i, # sites
                        -w, -u) # matsubaras
                    )
                 for ap in flavours, bp in flavours, cp in flavours, dp in flavours)
               for w in matsubaras)
            end # if is_on_site_pair
    end # let s,t,u
#! format: on

DSigma_(a::Flavour,b::Flavour, # flavours
        i::Site, # site
        w::MatsubaraF, # matsubara
        T::Temperature,
        geometry, matsubaras, Gamma, S) = let pairs = [ ij for ij in 1:geometry.Npairs
                                                           if (geometry.Pairtypes[ij].i == i ||
                                                               geometry.Pairtypes[ij].j == i )]

            -T/2* sum( # for wp in matsubaras
                       sum( # for c in flavours, d in flavours
                            sum(
                                let pair = geometry.Pairtypes[ij]
                                    j = if (i == pair.i) pair.j else pair.i end
                            S(c,d,# flavours
                              j,
                                    -wp)*Gamma(c,b,a,b,
                                               ij,
                                               -wp,wp,w,-w)
                                    end
                                for ij in pairs)
                            for c in flavours, d in flavours)
                       for wp in matsubaras)

        end # let
