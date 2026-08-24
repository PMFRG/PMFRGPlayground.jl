import Tullio: @tullio
import LinearAlgebra: I, dot, inv


module Ks
export K, delta5
import LinearAlgebra: I
Kx = [
    0 0 0 -im 0
    0 0 im 0 0
    0 -im 0 0 √3im
    im 0 0 0 0
    0 0 -√3im 0 0
]

Ky = [
    0 0 im 0 0
    0 0 0 im 0
    -im 0 0 0 0
    0 -im 0 0 -√3im
    0 0 0 √3im 0
]

Kz = [
    0 2im 0 0 0
    -2im 0 0 0 0
    0 0 0 im 0
    0 0 -im 0 0
    0 0 0 0 0
]

K = stack((Kx, Ky, Kz))

delta5 = Matrix{Float64}(I[1:5, 1:5])
end

"""
Parametrization basis
and its dual
"""
module GammaBasis
export b, bduals
using Tullio
import LinearAlgebra: dot, inv
using Main.Ks

@tullio b1[a, b, c, d] := delta5[a, b] * delta5[c, d]
@tullio b2[a, b, c, d] := delta5[a, c] * delta5[b, d]
@tullio b3[a, b, c, d] := K[a, b, k] * K[c, d, k]
@tullio b3[a, b, c, d] := delta5[a, d] * delta5[b, c]
@tullio b4[a, b, c, d] := K[a, b, k] * K[c, d, k]
@tullio b5[a, b, c, d] := K[a, c, k] * K[b, d, k]

b = stack((b1, b2, b3, b4, b5))

S = [dot(b[:, :, :, :, i], b[:, :, :, :, j]) for i in 1:5, j in 1:5]

Sinv = inv(S)

@tullio bduals[i, j, k, l, m] := Sinv[m, n] * b[i, j, k, l, n]
end

"""
Parametrization basis
done with 5d levi-civita.

But:
- this is NOT an algebra,
  the contractions produce tensors which are symmetric in some indices
  while everything is antisymmetric in this basis.
- it is not invariant under spin transformation by the Ks

(keeping this as a 'negative result').
"""
module GammaBasisLC
export b, bduals
import Combinatorics: levicivita

epsilon = [ levicivita([a,b,c,d,e]) for a in 1:5, b in 1:5, c in 1:5, d in 1:5, e in 1:5 ]

b = epsilon/sqrt(24)
bduals = b

end


"""
Coefficients that can be used to represent a contraction of two Γs
(that are a linear combination of elements of b)
as a linear combination of elements of b.
"""
module GammaAlgebra
export Ms, Mt_ii, Mt_ij, Mu_ii, Mu_ij, latex_represent_all, print_representation_to_file
using Tullio
import LinearAlgebra: dot
using Main.GammaBasis
using Latexify

convert_to_real_int(M) = Int.(Real.(round.(M)))

# In what follows, comparing with Eq. 75/76 in Siebe's Notes:
# 1 -> i
# 2 -> j
# 3 -> k
# 4 -> l
# 1', 2' -> m
# 3', 4' -> n

# S-channel, ij and ii case
# Terms like
# Γ_{121'4'} Γ_{2'3'34} δ_{1'2'} δ_{3'4'} -> Γ_{ijmn}Γ_{mnkl} ∝ GS_{ijkl}
# The deltas connect the last 2 indices of the first gamma
# with the first 2 indices of the second gamma
@tullio _Ms[x, y, z] := conj(bduals[i, j, k, l, z]) * b[i, j, m, n, x] * b[m, n, k, l, y]
Ms = convert_to_real_int(_Ms)

# T-channel, ii-case
# Terms like
# Γ_{132'3'} Γ_{1'4'24} δ_{1'2'} δ_{3'4'} -> Γ_{ikmn}Γ_{mnjl} ∝ GT_{ijkl}
# The deltas connect index 3 and 4 of the first gamma
# with index 1 and 2 of  the second gamma
@tullio _Mt_ii[x, y, z] := conj(bduals[i, j, k, l, z]) * b[i, k, m, n, x] * b[m, n, j, l, y]
Mt_ii = convert_to_real_int(_Mt_ii)

# T-channel, ij-case
# Terms like
# 1. Γ_{12'33'} Γ_{21'44'} δ_{1'2'} δ_{3'4'} -> Γ_{imkn}Γ_{jmln} ∝ GT_{ijkl}
# 2. Γ_{13'32'} Γ_{24'41'} δ_{1'2'} δ_{3'4'} -> Γ_{inkm}Γ_{jnlm} ∝ GT_{ijkl}
# The deltas connect index 2 and 4 of the first gamma
# with index 2 and 4 of  the second gamma
# Term 2. has same structure
@tullio _Mt_ij[x, y, z] := conj(bduals[i, j, k, l, z]) * b[i, m, k, n, x] * b[j, m, l, n, y]
Mt_ij = convert_to_real_int(_Mt_ij)

# U-channel, ii-case
# Terms like
# Γ_{141'4'} Γ_{2'3'23} δ_{1'2'} δ_{3'4'} -> Γ_{ilmn}Γ_{mnjk} ∝ GU_{ijkl}
# As in the t-case,
# the deltas connect index 2 and 4 of the first gamma
# with index 2 and 4 of the second gamma.
# Terms are equivalent given an exchange of the dummy indices m and n
@tullio _Mu_ii[x, y, z] := conj(bduals[i, j, k, l, z]) * b[i, l, m, n, x] * b[m, n, j, k, y]
Mu_ii = convert_to_real_int(_Mu_ii)

# U-channel, ij-case
# Terms like
# 1. Γ_{14'41'} Γ_{23'32'} δ_{1'2'} δ_{3'4'} -> Γ_{inlm}Γ_{jnkm} ∝ GU_{ijkl}
# 2. Γ_{11'44'} Γ_{22'33'} δ_{1'2'} δ_{3'4'} -> Γ_{imln}Γ_{jmkn} ∝ GU_{ijkl}
# As in the t-case,
# the deltas connect index 2 and 4 of the first gamma
# with index 2 and 4 of the second gamma.
# Terms are equivalent given an exchange of the dummy indices m and n
@tullio _Mu_ij[x, y, z] := conj(bduals[i, j, k, l, z]) * b[i, m, l, n, x] * b[j, m, k, n, y]
Mu_ij = convert_to_real_int(_Mu_ij)

function latex_represent_M(M, tag)
    r = ""
    for i in 1:5
        lhs = tag * "_{$i}"
        r = r * (@latexify $lhs = $(M[:, :, i]))
    end
    return "\\begin{align}\n" * replace(r,
               "\$" => "",
               "\\begin{array}{ccccc}" => "\\begin{pmatrix}",
               "\\left[" => "",
               "\\right]" => "",
               r"\\\\\n.*\\end{array}" => "\n\\end{pmatrix}\\\\",
               r"=.*\n" => "&=") * "\\end{align}\n"
end
function latex_represent_all()
    preamble = """
\\documentclass[11pt]{article}
\\usepackage{amsmath,amssymb}
\\usepackage{bm}
\\usepackage{physics}
\\begin{document}
"""

    r_ij = ("\\subsection*{s-channel, ij matrices}\n\n"
            * latex_represent_M(Ms, "M^{s,ij}")
            * "\\subsection*{t-channel, ij matrices}\n\n"
            * latex_represent_M(Mt_ij, "M^{t,ij}")
            * "\\subsection*{u-channel, ij matrices}\n\n"
            * latex_represent_M(Mu_ij, "M^{u,ij}")
            * "\n\n")

    r_ii = ("\\subsection*{s-channel, ii matrices}\n\n"
            * latex_represent_M(Ms, "M^{s,ii}")
            * "\\subsection*{t-channel, ii matrices}\n\n"
            * latex_represent_M(Mt_ii, "M^{t,ii}")
            * "\\subsection*{u-channel, ii matrices}\n\n"
            * latex_represent_M(Mu_ii, "M^{u,ii}")
            * "\n\n")
    post = """
\\end{document}
"""
    return preamble * r_ij * r_ii * post
end

function print_representation_to_file(fname="m.tex")
    open(fname, "w") do f
        write(f, latex_represent_all())
    end
end
end
using .GammaAlgebra

using Test

function test_K_antisymmetry_and_commutator(K)
    @testset verbose = true "Ks antisymmetry and commutation relations" begin
        function check_antisymmetry(K)
            for i in axes(K, 1), j in axes(K, 2)
                @test K[i, j] == -K[j, i]
            end
        end

        @testset "check antisymmetry of K" begin
            check_antisymmetry(K[:, :, 1])
            check_antisymmetry(K[:, :, 2])
            check_antisymmetry(K[:, :, 3])
        end

        function commutator(A, B)
            @tullio r[i, j] := A[i, k] * B[k, j] - B[i, k] * A[k, j]
        end

        a = randn(5, 5)
        b = randn(5, 5)
        @testset "commutator is antisymmetric" begin
            @test commutator(a, b) ≈ -commutator(b, a)
            @test !(commutator(a, b) ≈ commutator(b, a)) # Commutator is not zero
            @test isapprox(commutator(a, a), zeros(size(a)), atol=1.0e-15)
        end
        @testset "commutator relations for Ks" begin
            @test commutator(K[:, :, 1], K[:, :, 2]) ≈ K[:, :, 3] * im
            @test commutator(K[:, :, 3], K[:, :, 1]) ≈ K[:, :, 2] * im
            @test commutator(K[:, :, 2], K[:, :, 3]) ≈ K[:, :, 1] * im
        end
    end
end

function test_gamma_basis_and_duals(basis, basis_dual)
    @testset verbose = true "Basis" begin
        @testset "basis is invariant for all transformations" begin
            @tullio M[a,b,c,d,ap,bp,cp,dp,i] := (
                  K[a,ap,i]     * delta5[b,bp] * delta5[c,cp] * delta5[d,dp] +
                  delta5[a,ap]  * K[b,bp,i]    * delta5[c,cp] * delta5[d,dp] +
                  delta5[a,ap]  * delta5[b,bp] * K[c,cp,i]    * delta5[d,dp] +
                  delta5[a,ap]  * delta5[b,bp] * delta5[c,cp] * K[d,dp,i])

            for i in 1:3
                for e in 1:5
                    @tullio t[a,b,c,d] := M[a,b,c,d,ap,bp,cp,dp,i]*basis[ap,bp,cp,dp,e]
                    @test all(isapprox.(t , 0, atol=1.0e-15))
                end
            end

        end


        @testset "dual basis is dual to basis" begin
            bbdual_dot = [dot(basis[:, :, :, :, i],
                basis_dual[:, :, :, :, j]) for i in 1:5,
                          j in 1:5]
            println(bbdual_dot)
            @test bbdual_dot ≈ I

        end

        @testset "Decomposition of Γ as a linear combination of elements in b using the dual" begin
            for _ in 1:10
                ws = randn(5)
                @tullio Γ[i, j, k, l] := basis[i, j, k, l, m] * ws[m]

                @testset for i in 1:5
                    @test dot(basis_dual[:, :, :, :, i], Γ) ≈ ws[i]
                end
            end
        end
    end

end

"""
Here we check that any contraction of 2 Γs
(that are a linear combination of the 5 basis vectors)
can be represented faithfully using the coefficient in Ms, Mt_ii, Mt_ij, Mu_ii and Mu_ij
depending on the index contraction scheme.

In the flow equations 5 index contraction schemes are used,
denoted as S, T_ii, T_ij, U_ii, U_ij
"""
function test_gamma_algebra(Ms, Mt_ii, Mt_ij, Mu_ii, Mu_ij, b)
    @testset "Reality" begin
        @test all(isreal(Ms))
        @test all(isreal(Mt_ii))
        @test all(isreal(Mt_ij))
        @test all(isreal(Mu_ii))
        @test all(isreal(Mu_ij))
    end
    @testset "S" begin
        for _ in 1:10
            w1 = randn(5)
            w2 = randn(5)

            @tullio Γ1[i, j, k, l] := b[i, j, k, l, m] * w1[m]
            @tullio Γ2[i, j, k, l] := b[i, j, k, l, m] * w2[m]


            @tullio Γs[i, j, k, l] := Γ1[i, j, m, n] * Γ2[m, n, k, l]
            @tullio Γsexp[i, j, k, l] := Ms[x, y, z] * w1[x] * w2[y] * b[i, j, k, l, z]

            @test Γsexp ≈ Γs
        end
    end

    @testset "T_ii" begin
        for _ in 1:10
            w1 = randn(5)
            w2 = randn(5)

            @tullio Γ1[i, j, k, l] := b[i, j, k, l, m] * w1[m]
            @tullio Γ2[i, j, k, l] := b[i, j, k, l, m] * w2[m]


            @tullio Γt[i, j, k, l] := Γ1[i, k, m, n] * Γ2[m, n, j, l]
            @tullio Γtexp[i, j, k, l] := Mt_ii[x, y, z] * w1[x] * w2[y] * b[i, j, k, l, z]

            @test Γtexp ≈ Γt
        end
    end


    @testset "T_ij" begin
        for _ in 1:10
            w1 = randn(5)
            w2 = randn(5)

            @tullio Γ1[i, j, k, l] := b[i, j, k, l, m] * w1[m]
            @tullio Γ2[i, j, k, l] := b[i, j, k, l, m] * w2[m]


            @tullio Γt[i, j, k, l] := Γ1[i, m, k, n] * Γ2[j, m, l, n]
            @tullio Γtexp[i, j, k, l] := Mt_ij[x, y, z] * w1[x] * w2[y] * b[i, j, k, l, z]

            @test Γtexp ≈ Γt

        end
    end

    @testset "U_ii" begin
        for _ in 1:10
            w1 = randn(5)
            w2 = randn(5)

            @tullio Γ1[i, j, k, l] := b[i, j, k, l, m] * w1[m]
            @tullio Γ2[i, j, k, l] := b[i, j, k, l, m] * w2[m]

            @tullio Γu[i, j, k, l] := Γ1[i, l, m, n] * Γ2[m, n, j, k]
            @tullio Γuexp[i, j, k, l] := Mu_ii[x, y, z] * w1[x] * w2[y] * b[i, j, k, l, z]

            @test Γuexp ≈ Γu
        end
    end

    @testset "U_ij" begin
        for _ in 1:10
            w1 = randn(5)
            w2 = randn(5)

            @tullio Γ1[i, j, k, l] := b[i, j, k, l, m] * w1[m]
            @tullio Γ2[i, j, k, l] := b[i, j, k, l, m] * w2[m]

            @tullio Γu[i, j, k, l] := Γ1[i, m, l, n] * Γ2[j, m, k, n]
            @tullio Γuexp[i, j, k, l] := Mu_ij[x, y, z] * w1[x] * w2[y] * b[i, j, k, l, z]

            @test Γuexp ≈ Γu
        end
    end
end

using .Ks
using .GammaBasis
using .GammaAlgebra

function tests()
    @testset verbose = true begin
        test_K_antisymmetry_and_commutator(K)
        test_gamma_basis_and_duals(b, bduals)
        test_gamma_algebra(Ms, Mt_ii, Mt_ij, Mu_ii, Mu_ij, b)
    end
    return
end

tests()
print_representation_to_file("ms.tex")
