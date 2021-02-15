import Random

using LinearAlgebra
using CSV, DataFrames

function x_inter(t::Float64, Ts::Float64, A::Array{Float64, 2}, B::Array{Float64, 2}, x::Array{Array{Float64, 1}, 1})
    ϵ = 10e-6               # Values of δ smaller than this are treated as 0
    k = Int(t÷Ts)           # t lies between t0 + k*Ts and t0 + (k+1)*Ts
    δ = t - (t0 + k*Ts)
    n = size(A)[1]

    # TODO: Change, must be better way to pass x0
    x0 = [0; 0]

    if δ < ϵ
        if k == 0
            return x0
        else
            return x[k]
        end
    elseif Ts-δ < ϵ
        return x[k+1]
    end


    Mexp    = [A B*(B'); zeros(size(A)) -A']
    MTs     = exp(Mexp*Ts)      # TODO: This really shouldn't be computed in real time
    Mδ      = exp(Mexp*δ)
    MTs_δ   = exp(Mexp*(Ts-δ))
    AdTs    = MTs[1:n, 1:n]
    Adδ     = Mδ[1:n, 1:n]
    AdTs_δ  = MTs_δ[1:n, 1:n]
    Bd2Ts    = MTs[1:n, n+1:end]*AdTs
    Bd2δ    = Mδ[1:n, n+1:end]*Adδ
    Bd2Ts_δ = MTs_δ[1:n, n+1:end]*AdTs_δ
    BdTs     = Bd2Ts
    CTs     = cholesky(Bd2Ts)
    Cδ      = cholesky(Bd2δ)        # Might need to wrap matrices in Hermitian()
    CTs_δ   = cholesky(Bd2Ts_δ)
    BdTs    = CTs.L
    Bdδ     = Cδ.L
    BdTs_δ  = CTs_δ.L

    if k > 0
        # nm1 = n - 1, np1 = n + 1
        μ_nm1 = (AdTs^(k-1))*x0
        μ_n   = Adδ*μ_nm1
        μ_np1 = AdTs_δ*μ_n
        σ_nm1 = zeros(Float64, n, n)
        BdBdT = (BdTs*(BdTs'))
        for j=0:1:k-1
            Adj = (AdTs^j)
            σ_nm1 += Adj*(BdBdT)*(Adj')
        end
        σ_n = Adδ*σ_nm1*(Adδ') + Bdδ*(Bdδ')
        σ_np1 = AdTs_δ*σ_n*(AdTs_δ') + BdTs_δ*(BdTs_δ')
        σ_np1_n = AdTs_δ*σ_n
        σ_np1_nm1 = AdTs_δ*Adδ*σ_nm1
        σ_n_nm1   = Adδ*σ_nm1

        σ_n_z = [σ_np1_n' σ_n_nm1]
        σ_z   = [σ_np1 σ_np1_nm1; σ_np1_nm1' σ_nm1]
        if k > 1
            z = [x[k]; x[k-1]]
        else
            z  = [x[k]; x0]
        end
        μ_z   = [μ_np1; μ_nm1]
        μ = μ_n + σ_n_z*(σ_z\(z-μ_z))
        Σ = Hermitian(σ_n - σ_n_z*(σ_z\(σ_n_z')))
        CΣ   = cholesky(Σ)
        Σr   = CΣ.L
    else
        # k == 0
        μ_1 = Adδ*x0
        σ_1 = Bdδ*(Bdδ')
        μ_2 = AdTs_δ*μ_1
        σ_2 = AdTs*σ_1*(AdTs') + BdTs*(BdTs')
        σ_12 = σ_1*(AdTs')
        μ = μ_1 + σ_12*( σ_2\(x[1]-μ_2) )       # See theory notes for why we have x[1] - μ_2, and not e.g. x[1] - μ_1. It's a question of notation
        Σ = Hermitian(σ_1 - σ_12*(σ_2\(σ_12')))
        CΣ   = cholesky(Σ)
        Σr   = CΣ.L
    end

    # # DEBUG Nice, theory seems to match practice quite well
    # Σ_test = [σ_n σ_n_z; σ_n_z' σ_z]
    # println(Σ_test)

    xkδ = μ + Σr*randn(Float64, (n, 1))
    return xkδ

end
