using Random, LinearAlgebra

function noise_inter(t::Float64,
                     Ts::Float64,
                     A::Array{Float64, 2},
                     B::Array{Float64, 2},
                     x::Array{Array{Float64, 1}, 1},
                     z_inter::Array{Array{Float64, 2}, 1},
                     num_inter_samples::Array{Int64, 1},
                     x0::Array{Float64, 1},
                     ϵ::Float64=10e-12)

    n = Int(t÷Ts)           # t lies between t0 + n*Ts and t0 + (n+1)*Ts
    δ = t - n*Ts
    nx = size(A)[1]
    P = size(z_inter[1])[1]
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

    # MTs     = exp(Mexp*Ts)
    # AdTs    = MTs[nx+1:end, nx+1:end]'
    # Bd2Ts   = Hermitian(AdTs*MTs[1:nx, nx+1:end])
    # CTs     = cholesky(Bd2Ts)
    # BdTs    = CTs.L

    MTs_δ   = exp(Mexp*(Ts-δ))
    AdTs_δ  = MTs_δ[nx+1:end, nx+1:end]'
    Bd2Ts_δ = Hermitian(AdTs_δ*MTs_δ[1:nx, nx+1:end])
    CTs_δ   = cholesky(Bd2Ts_δ)
    BdTs_δ  = CTs_δ.L

    Mδ      = exp(Mexp*δ)
    Adδ     = Mδ[nx+1:end, nx+1:end]'
    Bd2δ    = Hermitian(Adδ*Mδ[1:nx, nx+1:end])
    Cδ      = cholesky(Bd2δ)
    Bdδ     = Cδ.L

    x_l = if (n>0) x[n] else x0 end
    x_u = x[n+1]

    # σ_Ts = BdTs*(BdTs')
    σ_δ = (Bdδ*(Bdδ'))
    σ_Ts_δ = AdTs_δ*σ_δ

    # More efficient way of computing these matrices
    AdTs = AdTs_δ*Adδ
    σ_Ts = AdTs_δ*σ_δ*(AdTs_δ') + BdTs_δ*(BdTs_δ')

    v_Ts = x_u - AdTs*x_l
    μ = Adδ*x_l + (σ_Ts_δ')*(σ_Ts\v_Ts)

    # Hermitian()-call might not be necessary, but it probably depends on the
    # model, so I leave it in to ensure that cholesky decomposition will work
    Σ = Hermitian(σ_δ - (σ_Ts_δ')*(σ_Ts\(σ_Ts_δ)))
    CΣ   = cholesky(Σ)
    Σr   = CΣ.L

    if num_inter_samples[n+1] < P
        white_noise = z_inter[n+1][num_inter_samples[n+1]+1,:]
        num_inter_samples[n+1] += 1
    else
        white_noise = randn(Float64, (nx, 1))
    end

    xkδ = μ + Σr*white_noise
    return xkδ

end
