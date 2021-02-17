import Random

using LinearAlgebra
using CSV, DataFrames

function x_inter(t::Float64, Ts::Float64, A::Array{Float64, 2}, B::Array{Float64, 2}, x::Array{Array{Float64, 1}, 1}, ϵ::Float64=10e-6, x0::Array{Float64, 1}=[0.;0.])
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

    Mexp    = [A B*(B'); zeros(size(A)) -A']
    MTs     = exp(Mexp*Ts)      # TODO: This really shouldn't be computed in real time
    Mδ      = exp(Mexp*δ)
    MTs_δ   = exp(Mexp*(Ts-δ))
    AdTs    = MTs[1:nx, 1:nx]
    Adδ     = Mδ[1:nx, 1:nx]
    AdTs_δ  = MTs_δ[1:nx, 1:nx]
    Bd2Ts    = MTs[1:nx, nx+1:end]*AdTs
    Bd2δ    = Mδ[1:nx, nx+1:end]*Adδ
    Bd2Ts_δ = MTs_δ[1:nx, nx+1:end]*AdTs_δ
    BdTs     = Bd2Ts
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


    # np1 = n+1, np2 = n + 2
    μ_n = (AdTs^(n-1))*x0
    μ_np1   = Adδ*μ_n
    μ_np2 = AdTs_δ*μ_np1
    σ_n = zeros(Float64, nx, nx)
    BdBdT = (BdTs*(BdTs'))
    for j=0:1:n-1
        Adj = (AdTs^j)
        σ_n += Adj*(BdBdT)*(Adj')
    end
    σ_np1 = Adδ*σ_n*(Adδ') + Bdδ*(Bdδ')
    σ_np2 = AdTs_δ*σ_np1*(AdTs_δ') + BdTs_δ*(BdTs_δ')
    σ_np2_np1 = AdTs_δ*σ_np1
    σ_np2_n = AdTs_δ*Adδ*σ_n
    σ_np1_n   = Adδ*σ_n

    σ_np1_z = [σ_np2_np1' σ_np1_n]
    σ_z   = [σ_np2 σ_np2_n; σ_np2_n' σ_n]
    μ_z   = [μ_np2; μ_n]
    if n > 0
        z = [x[k_np2]; x[k_n]]
        μ = μ_np1 + σ_np1_z*(σ_z\(z-μ_z))
        Σ = Hermitian(σ_np1 - σ_np1_z*(σ_z\(σ_np1_z')))
    else
        z = [x[k_np2]; x0]
        μ = μ_np1 + (σ_np2_np1')*( σ_np2\(x[k_np2] - μ_np2) )
        Σ = Hermitian(σ_np1 - (σ_np2_np1')*( σ_np2\σ_np2_np1 ))
    end
    CΣ   = cholesky(Σ)
    Σr   = CΣ.L

    # DEBUG Not yet fully implemented alternative approach
    # σ_Ts = BdTs*(BdTs')
    # σ_δ_Ts = AdTs_δ*(Bdδ*(Bdδ'))
    # # t_n = n*Ts, t_np1 = n*Ts + δ, t_np2 = (n+1)*Ts
    # # We are thus sampling x(t_np1)
    # if n > 0
    #     v_Ts = x[n] - AdTs*x[n-1]         # WHY THIS AND NOT +1 ON BOTH?!?!?!??!
    #     μ = Adδ*x[n-1] + σ_δ_Ts*(σ_Ts\v_Ts)
    # else
    #     # t_0 = 0, t_1 = δ, t_2 = Ts
    #     v_Ts = x[1] - AdTs*x0
    #     μ = Adδ*x0 + σ_δ_Ts*(σ_Ts\v_Ts)
    # end
    # Σ = σ_δ_Ts*(σ_Ts\(σ_δ_Ts'))
    # CΣ   = cholesky(Σ)
    # Σr   = CΣ.L

    # # DEBUG Nice, theory seems to match practice quite well
    # Σ_test = [σ_np1 σ_np1_z; σ_np1_z' σ_z]
    # println(Σ_test)

    xkδ = μ + Σr*randn(Float64, (nx, 1))
    return xkδ

end
