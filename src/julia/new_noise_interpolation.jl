using Random, LinearAlgebra, Future

mutable struct InterSampleData
    # states[i][p,j] is the j:th element of the p:th state in the i:th interval
    # The number of rows of states[i] should be dynamically updated as new
    # samples are generated
    # sample_times[i][p] is the time associated with the p:th state in the
    # i:th interval. Also here the number of rows should be dynamically updated
    states::Array{Array{Float64,2},1}
    sample_times::Array{Array{Float64,1},1}
    Q::Int64    # Max number of stored samples per interval
    #num_sampled_samples[i] is the number of times a state has been sampled
    # in interval i
    num_sampled_samples::Array{Int64, 1}
    use_interpolation::Bool # true if linear interpolation of states is used
    # instead of conditional sampling when Q stored samples has been surpassed.
    # It improves smoothness of realization in such a scenario.
end

function initialize_isd(Q::Int64, N::Int64, nx::Int64, use_interpolation::Bool)::InterSampleData
    isd_states = [zeros(0,nx) for j=1:N]
    isd_sample_times = [zeros(0) for j=1:N]
    num_samples = zeros(Int64, N)
    return InterSampleData(isd_states, isd_sample_times, Q, num_samples, use_interpolation)
end

function noise_inter(t::Float64,
                     Ts::Float64,       # Sampling time of noise process
                     A::Array{Float64, 2},
                     B::Array{Float64, 2},
                     x::Array{Array{Float64, 1}, 1},
                     #z_inter::Array{Array{Float64, 2}, 1},
                     z_inter::Any,
                     isd::InterSampleData,
                     ϵ::Float64=10e-12,
                     rng::MersenneTwister=Random.default_rng())

    n = Int(t÷Ts)           # t lies between t0 + n*Ts and t0 + (n+1)*Ts
    δ = t - n*Ts
    nx = size(A)[1]
    Q = isd.Q
    # P = size(z_inter[1])[1]
    P = 0
    N = size(isd.states)[1]
    use_interpolation = isd.use_interpolation
    # This case is usually handled by the check further down for δ smaller
    # than ϵ, but if n == N, isd.states[n+1] will give BoundsError, so we
    # need to put this if-statement here to avoid that. We only check for n==N,
    # and not n >= N so that there will be a crash if times after the last
    # sample are requested
    if n == N
        return x[N+1]
    else
        num_stored_samples = size(isd.states[n+1])[1]
    end
    tl = n*Ts
    tu = (n+1)*Ts
    il = 0      # for il>0,   tl = isd.sample_times[n][il]
    iu = Q+1    # for iu<Q+1, tu = isd.sample_times[n][iu]

    # setting il, tl, iu, tu
    if num_stored_samples > 0
        for q = 1:num_stored_samples
            # interval index = n+1
            t_inter = isd.sample_times[n+1][q]
            if t_inter > tl && t_inter <= t
                tl = t_inter
                il = q
            end
            if t_inter < tu && t_inter >= t
                tu = t_inter
                iu = q
            end
        end
    end
    δl = t-tl
    δu = tu-t

    # Setting xl and xu
    xl = x[n+1]     # x[1] == x0
    xu = x[n+2]
    if il > 0
        xl = isd.states[n+1][il,:]
    end
    if iu < Q+1
        xu = isd.states[n+1][iu,:]
    end

    # If no more samples are stored in this interval, allow for the use of
    # linear interpolation instead, to ensure smoothness of realization
    if num_stored_samples >= Q && use_interpolation
        # @warn "Used linear interpolation"   # DEBUG
        return xl + (xu-xl)*(t-tl)/(tu-tl)
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

    if isd.num_sampled_samples[n+1] < P
        white_noise = z_inter[n+1][num_stored_samples+1,:]
        isd.num_sampled_samples[n+1] += 1
    else
        # @warn "Ran out of pre-generated white noise realizations for interval $(n+1)"
        white_noise = randn(rng, Float64, (nx, 1))
    end

    x_new = μ + Σr*white_noise

    if num_stored_samples < Q
        isd.states[n+1] = [isd.states[n+1]; x_new']
        isd.sample_times[n+1] = [isd.sample_times[n+1]; t]
    end

    return x_new

end
