import Tullio: @tullio
import LinearAlgebra: I, dot, inv

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


# Parametrization basis

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


# In what follows, comparing with Eq. 75/76 in Siebe's Notes:
# 1 -> i
# 2 -> j
# 3 -> k
# 4 -> l
# 1', 2' -> m
# 3', 4' -> n

# S-channel
# Terms like
# Γ_{121'4'} Γ_{2'3'34} δ_{1'2'} δ_{3'4'} -> Γ_{ijmn}Γ_{mnkl} ∝ GS_{ijkl}
# The deltas connect the last 2 indices of the first gamma
# with the first 2 indices of the second gamma

@tullio Ms[z, x, y] := conj(bduals[i, j, k, l, z]) * b[i, j, m, n, x] * b[m, n, k, l, y]

# T-channel
# Terms like
# 1. Γ_{12'33'} Γ_{21'44'} δ_{1'2'} δ_{3'4'} -> Γ_{imkn}Γ_{jmln} ∝ GT_{ijkl}
# 2. Γ_{13'32'} Γ_{24'41'} δ_{1'2'} δ_{3'4'} -> Γ_{inkm}Γ_{jnlm} ∝ GT_{ijkl}
# The deltas connect index 2 and 4 of the first gamma
# with index 2 and 4 of  the second gamma
# Term 2. has same structure
@tullio Mt[z, x, y] := conj(bduals[i, j, k, l, z]) * b[i, m, k, n, x] * b[j, m, l, n, y]

# U-channel
# Terms like
# 1. Γ_{14'41'} Γ_{23'32'} δ_{1'2'} δ_{3'4'} -> Γ_{inlm}Γ_{jnkm}
# 2. Γ_{11'44'} Γ_{22'33'} δ_{1'2'} δ_{3'4'} -> Γ_{imln}Γ_{jmkn}
# As in the t-case,
# the deltas connect index 2 and 4 of the first gamma
# with index 2 and 4 of the second gamma.
# Terms are equivalent given an exchange of the dummy indices m and n
@tullio Mu[z, x, y] := conj(bduals[i, j, k, l, z]) * b[i, m, l, n, x] * b[j, m, k, n, y]


using Test

function K_tests(K)
    @testset verbose = true "Ks" begin
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
        end
        @testset "commutator relations for Ks" begin
            @test commutator(K[:, :, 1], K[:, :, 2]) ≈ K[:, :, 3] * im
            @test commutator(K[:, :, 3], K[:, :, 1]) ≈ K[:, :, 2] * im
            @test commutator(K[:, :, 2], K[:, :, 3]) ≈ K[:, :, 1] * im
        end
    end
end

function base_tests(b, bduals)
    @testset verbose = true "Basis" begin
        @testset "bdual is dual to b" begin
            bbdual_dot = [dot(b[:, :, :, :, i],
                bduals[:, :, :, :, j]) for i in 1:5,
                          j in 1:5]
            @test bbdual_dot ≈ I
        end

        @testset "Decomposition of vector into B using Bdual" begin
            for _ in 1:10
                ws = randn(5)
                @tullio x[i, j, k, l] := b[i, j, k, l, m] * ws[m]

                @testset for i in 1:5
                    @test dot(bduals[:, :, :, :, i], x) ≈ ws[i]
                end
            end
        end
    end

end

function M_tests(Ms, Mt, Mu, b)
    @testset "Ms" begin
        for _ in 1:10
            w1 = randn(5)
            w2 = randn(5)

            @tullio Γ1[i, j, k, l] := b[i, j, k, l, m] * w1[m]
            @tullio Γ2[i, j, k, l] := b[i, j, k, l, m] * w2[m]


            @tullio Γs[i, j, k, l] := Γ1[i, j, m, n] * Γ2[m, n, k, l]
            @tullio Γsexp[i, j, k, l] := Ms[z, x, y] * w1[x] * w2[y] * b[i, j, k, l, z]

            @test Γsexp ≈ Γs
        end
    end

    @testset "Mt" begin
        for _ in 1:10
            w1 = randn(5)
            w2 = randn(5)

            @tullio Γ1[i, j, k, l] := b[i, j, k, l, m] * w1[m]
            @tullio Γ2[i, j, k, l] := b[i, j, k, l, m] * w2[m]


            @tullio Γt[i, j, k, l] := Γ1[i, m, k, n] * Γ2[j, m, l, n]
            @tullio Γtexp[i, j, k, l] := Mt[z, x, y] * w1[x] * w2[y] * b[i, j, k, l, z]

            @test Γtexp ≈ Γt

        end
    end

    @testset "Mu" begin
        for _ in 1:10
            w1 = randn(5)
            w2 = randn(5)

            @tullio Γ1[i, j, k, l] := b[i, j, k, l, m] * w1[m]
            @tullio Γ2[i, j, k, l] := b[i, j, k, l, m] * w2[m]

            @tullio Γu[i, j, k, l] := Γ1[i, m, l, n] * Γ2[j, m, k, n]
            @tullio Γuexp[i, j, k, l] := Mu[z, x, y] * w1[x] * w2[y] * b[i, j, k, l, z]

            @test Γuexp ≈ Γu
        end
    end
end



function tests()
    @testset verbose = true begin
        K_tests(K)
        base_tests(b, bduals)
        M_tests(Ms, Mt, Mu, b)
    end
    return
end

tests()
