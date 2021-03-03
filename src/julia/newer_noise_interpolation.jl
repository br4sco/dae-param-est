using Random, LinearAlgebra, Future

mutable struct InterSampleData
    # states[i][p,j] is the j:th element of the p:th state in the i:th interval
    # The number of rows of states[i] should be dynamically updated as new
    # samples are generated
    # sample_times[i][p] is the time associated with the p:th state in the
    # i:th interval. Also here the number of rows should be dynamically updated
    states::Array{Array{Float64,2},1}
    sample_times::Array{Array{Float64,1},1}
    P::Int64    # Number of pre-generated realizations per interval
end

function initialize_isd(P::Int64, N::Int64, nx::Int64)::InterSampleData
    isd_states = [zeros(0,nx) for j=1:N]
    isd_sample_times = [zeros(0) for j=1:N]
    return InterSampleData(isd_states, isd_sample_times, P)
end

function jump_realizations(rng::MersenneTwister, num_realizations::Int64)::MersenneTwister
    steps = num_realizations÷2  # Each step corresponds to generating two Float64 numbers
    remainder = mod(num_realizations, 0:1)
    new_rng = Future.randjump(rng, steps)
    if remainder == 1
        # Move rng state forward one more Float64-realization, sort of like taking half a step
        rand(new_rng)
    end
    return new_rng
end

function noise_inter_new(t::Float64,
                     Ts::Float64,
                     A::Array{Float64, 2},
                     B::Array{Float64, 2},
                     x::Array{Array{Float64, 1}, 1},
                     rng::MersenneTwister,
                     isd::InterSampleData,
                     x0::Array{Float64, 1},
                     ϵ::Float64=10e-12)

    n = Int(t÷Ts)           # t lies between t0 + n*Ts and t0 + (n+1)*Ts
    δ = t - n*Ts
    nx = size(A)[1]
    P = isd.P
    num_inter_samples = size(isd.states[n+1])[1]
    tl = n*Ts
    tu = (n+1)*Ts
    il = 0      # for il>0,   tl = isd.sample_times[n][il]
    iu = P+1    # for iu<P+1, tu = isd.sample_times[n][iu]

    # setting il, tl, iu, tu
    if num_inter_samples > 0
        for p = 1:num_inter_samples
            # interval index = n+1
            t_inter = isd.sample_times[n+1][p]
            if t_inter > tl && t_inter < t
                tl = t_inter
                il = p
            end
            if t_inter < tu && t_inter > t
                tu = t_inter
                iu = p
            end
        end
    end
    δl = t-tl
    δu = tu-t

    # Setting xl and xu
    xl = if (n > 0) x[n] else x0 end
    xu = x[n+1]
    if il > 0
        xl = isd.states[n+1][il,:]
    end
    if iu < P+1
        xu = isd.states[n+1][iu,:]
    end

    # Values of δ smaller than ϵ are treated as 0
    if δl < ϵ
        return xl
    elseif δu < ϵ
        return xu
    end

    Mexp    = [-A B*(B'); zeros(size(A)) A']
    Ml      = exp(Mexp*δl)
    Mu      = exp(Mexp*δu)
    Adl     = Ml[nx+1:end, nx+1:end]'
    Adu     = Mu[nx+1:end, nx+1:end]'
    AdΔ     = Adu*Adl
    B2dl    = Hermitian(Adl*Ml[1:nx, nx+1:end])
    B2du    = Hermitian(Adu*Mu[1:nx, nx+1:end])
    Cl      = cholesky(B2dl)
    Cu      = cholesky(B2du)
    Bdl     = Cl.L
    Bdu     = Cu.L

    σ_l = (Bdl*(Bdl'))
    σ_u = (Bdu*(Bdu'))
    σ_Δ = Adu*σ_l*(Adu') + σ_u
    σ_Δ_l = Adu*σ_l
    v_Δ = xu - AdΔ*xl
    μ = Adl*xl + (σ_Δ_l')*(σ_Δ\v_Δ) # TODO: Might want to double-check that this also covers n=0, but it seems to be the case
    # Hermitian()-call might not be necessary, but it probably depends on the
    # model, so I leave it in to ensure that cholesky decomposition will work
    Σ = Hermitian(σ_l - (σ_Δ_l')*(σ_Δ\(σ_Δ_l)))
    CΣ = cholesky(Σ)
    Σr = CΣ.L

    if num_inter_samples < P
        num_jumps = n*P*nx + (num_inter_samples)*nx
        new_rng = jump_realizations(rng, num_jumps)
        realization = randn(new_rng, Float64, (nx,1))
    else
        @warn "Ran out of pre-generated white noise realizations for interval $(n+1)"
        # TODO: NOTE: WARNING: YOU ARE USING GLOBAL RNG HERE, YOU HAVE TO MAKE
        # SURE THAT IT'S SEEDED APPROPRIATELY OUTSIDE OF THIS FUNCTIONS!!!!!!
        realization = randn(Float64, (nx, 1))
    end

    x_new = μ + Σr*realization

    if num_inter_samples < P
        isd.states[n+1] = [isd.states[n+1]; x_new']
        isd.sample_times[n+1] = [isd.sample_times[n+1]; t]
    end

    return x_new

end
