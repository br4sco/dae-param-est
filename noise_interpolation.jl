import Random

using LinearAlgebra
using CSV, DataFrames

function x_inter(t::Float64,
                 Ts::Float64,
                 A::Array{Float64, 2},
                 B::Array{Float64, 2},
                 x::Array{Array{Float64, 1}, 1},
                 ϵ::Float64=10e-6,
                 x0::Array{Float64, 1}=[0.;0.],
                 t0=0.0)

    n = Int(t÷Ts)           # t lies between t0 + k*Ts and t0 + (k+1)*Ts
    δ = t - (t0 + n*Ts)
    nx = size(A)[1]

    # Values of δ smaller than ϵ are treated as 0
    if δ < ϵ
        if n == 0
            return x0
        else
            return x[n]
        end
    elseif Ts-δ < ϵ
        return x[n+1]
    end

    Mexp    = [-A B*(B'); zeros(size(A)) A']
    MTs     = exp(Mexp*Ts)      # TODO: This really shouldn't be computed in real time
    Mδ      = exp(Mexp*δ)
    MTs_δ   = exp(Mexp*(Ts-δ))
    # AdTs    = MTs[1:nx, 1:nx]
    # Adδ     = Mδ[1:nx, 1:nx]
    # AdTs_δ  = MTs_δ[1:nx, 1:nx]
    # Bd2Ts    = MTs[1:nx, nx+1:end]*AdTs
    # Bd2δ    = Mδ[1:nx, nx+1:end]*Adδ
    # Bd2Ts_δ = MTs_δ[1:nx, nx+1:end]*AdTs_δ
    AdTs    = MTs[nx+1:end, nx+1:end]'
    Adδ     = Mδ[nx+1:end, nx+1:end]'
    AdTs_δ  = MTs_δ[nx+1:end, nx+1:end]'
    Bd2Ts   = Hermitian(AdTs*MTs[1:nx, nx+1:end])
    Bd2δ    = Hermitian(Adδ*Mδ[1:nx, nx+1:end])
    Bd2Ts_δ = Hermitian(AdTs_δ*MTs_δ[1:nx, nx+1:end])
    CTs     = cholesky(Bd2Ts)
    Cδ      = cholesky(Bd2δ)        # Might need to wrap matrices in Hermitian()
    CTs_δ   = cholesky(Bd2Ts_δ)
    BdTs    = CTs.L
    Bdδ     = Cδ.L
    BdTs_δ  = CTs_δ.L

    # t_n = n*Ts, t_np1 = n*Ts + δ, t_np2 = (n+1)*Ts
    # We are thus sampling x(t_np1)
    k_n = n
    k_np2 = n+1

    σ_Ts = BdTs*(BdTs')
    σ_δ = (Bdδ*(Bdδ'))
    σ_Ts_δ = AdTs_δ*σ_δ
    # t_n = n*Ts, t_np1 = n*Ts + δ, t_np2 = (n+1)*Ts
    # We are thus sampling x(t_np1)
    if n > 0
        v_Ts = x[k_np2] - AdTs*x[k_n]
        μ = Adδ*x[k_n] + (σ_Ts_δ')*(σ_Ts\v_Ts)
    else
        # t_0 = 0, t_1 = δ, t_2 = Ts
        v_Ts = x[k_np2] - AdTs*x0
        μ = Adδ*x0 + (σ_Ts_δ')*(σ_Ts\v_Ts)
    end
    # Hermitian()-call might not be necessary, but it probably depends on the
    # model, so I leave it in to ensure that cholesky decomposition will work
    Σ = Hermitian(σ_δ - (σ_Ts_δ')*(σ_Ts\(σ_Ts_δ)))
    CΣ   = cholesky(Σ)
    Σr   = CΣ.L

    # # DEBUG OUTDATED Nice, theory seems to match practice quite well.
    # Σ_test = [σ_np1 σ_np1_z; σ_np1_z' σ_z]
    # println(Σ_test)

    xkδ = μ + Σr*randn(Float64, (nx, 1))
    return xkδ

end
