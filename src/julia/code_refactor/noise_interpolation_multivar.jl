module NoiseInterpolation

import Random, LinearAlgebra, Future

export InterSampleWindow, initialize_isw, noise_inter

mutable struct InterSampleWindow
    containers::Array{Array{Float64,2},1}
    sample_times::Array{Array{Float64,1},1}
    num_stored::Array{Int64,1}
    Q::Int64    # Max number of stored samples per interval
    W::Int64    # Number of containers in the array containers
    # TODO: Not sure if this struct is the best place to store use_interpolation
    use_interpolation::Bool # true if linear interpolation of states is used
    # instead of conditional sampling when Q stored samples has been surpassed.
    # It improves smoothness of realization in such a scenario.
    start::Int64    # The interval index to which the "first" container corresponds
    ptr::Int64      # The index of the "first" container in the array containers
end

mutable struct InterSampleData
    # states[i][p,j] is the j:th element of the p:th state in the i:th interval
    # The number of rows of states[i] should be dynamically updated as new
    # samples are generated
    # sample_times[i][p] is the time associated with the p:th state in the
    # i:th interval. Also here the number of rows should be dynamically updated
    states::Array{Array{Float64,2},1}
    sample_times::Array{Array{Float64,1},1}
    Q::Int64    # Max number of stored samples per interval
    use_interpolation::Bool # true if linear interpolation of states is used
    # instead of conditional sampling when Q stored samples has been surpassed.
    # It improves smoothness of realization in such a scenario.
end

function initialize_isw(Q::Int64, W::Int64, nx::Int64,
     use_interpolation::Bool=true)::InterSampleWindow
     if Q > 0
         containers = [zeros(nx, Q)  for j=1:W]
         sample_times = [zeros(Q) for j=1:W]
         num_stored   = zeros(W)
     else
         containers = []
         sample_times = []
         num_stored = []
         W = 0  # W > 0 while Q=0 doesn't make sense, since we don't store anything
     end
     return InterSampleWindow(containers, sample_times, num_stored, Q, W, use_interpolation, 0, 1)
end

function reset_isw!(isw::InterSampleWindow)
    fill!(isw.num_stored, 0)
    isw.start = 0
    isw.ptr = 1
end

function reset_isws!(isws::AbstractArray{InterSampleWindow})
    for i in eachindex(isws)
        reset_isw!(isws[i])
    end
end

function map_to_container(num::Int64, isw::InterSampleWindow)
    return ((num-1)%isw.W) + 1
end

# Since x is originally a 1D array but can be become a 2D array when created
# using matrix-operations, I have specified the input here as an AbstractArray
# instead of requiring a specific dimension of the input array. Hopefully this
# should work well, and I think probably most of our functions should use
# AbstractArrays, since it would make the code support datatypes such as
# transposes and similar. In case this doesn't work, it is simple to convert
# a 2D array into a 1D array using the [:]-operator. According to
# https://stackoverflow.com/questions/63340812/how-to-convert-from-arrayfloat64-2-to-arrayarrayfloat64-1-1-in-julia
# this doesn't use any additional memory, so it shouldn't impact performance negatively.
function add_sample!(x_new::AbstractArray, sample_time::Float64, n::Int64,
    isw::InterSampleWindow)

    if isw.start <= n && n <= isw.start + isw.W - 1
        container_id = map_to_container(isw.ptr + n - isw.start, isw)
    elseif n > isw.start + isw.W - 1
        num_steps = n - isw.start - isw.W + 1
        for step in 1:num_steps
            # isw.containers[map_to_container(isw.ptr+isw.W-1+step, isw)] = zeros(0, size(x_new)[1])
            # isw.sample_times[map_to_container(isw.ptr+isw.W-1+step, isw)] = zeros(0)
            isw.num_stored[map_to_container(isw.ptr+isw.W-1+step, isw)] = 0
        end
        container_id = map_to_container(isw.ptr+isw.W-1+num_steps, isw)
        isw.ptr = map_to_container(isw.ptr+num_steps, isw)
        isw.start += num_steps
    else # n < isw.start
        @warn "Tried to add sample outside of inter-sample window (adding $n to window $(isw.start) -- $(isw.start+isw.W-1))"
        return
    end
    num_stored = isw.num_stored[container_id]
    # Only stores samples if less than Q samples are already stored
    if num_stored < isw.Q
        isw.containers[container_id][:, num_stored+1] = x_new
        isw.sample_times[container_id][num_stored+1]   = sample_time
        isw.num_stored[container_id] += 1
    end
    # println(isw.containers)
    # println("ptr: $(isw.ptr), start: $(isw.start)")
end

# TODO: SHOULD RLY BE 1D ARRAY OF 1D ARRAYS, instead of just an AbstractArray
function get_neighbors(n::Int64, t::Float64, x::AbstractArray,
    Ts::Float64, isw::InterSampleWindow)

    tl = n*Ts
    tu = (n+1)*Ts
    il = 0
    iu = 0

    if isw.W == 0
        return x[n+2], x[n+1], tu-t, t-tl, isw.use_interpolation
    end

    # Finds il, iu, tl, tu, i.e. indices and times of neighboring samples
    should_interpolate = false
    if n >= isw.start && n <= isw.start + isw.W - 1
        idx = map_to_container(isw.ptr + n - isw.start, isw)
        num_stored_samples = isw.num_stored[idx]
        if num_stored_samples > 0
            # TODO: There must be a more efficient search method
            for q=1:num_stored_samples
                t_inter = isw.sample_times[idx][q]
                if t_inter > tl && t_inter <= t
                    tl = t_inter
                    il = q
                end
                if t_inter < tu && t_inter >= t
                    tu = t_inter
                    iu = q
                end
            end
            if num_stored_samples >= isw.Q && isw.use_interpolation
                @warn "Ran out of space to store samples, storing $num_stored_samples"
                # If no more samples can be stored in this interval,
                # conditional sampling should no longer be used, as it will
                # likely produce a non-smooth realization, because none of
                # the new samples will be stored
                should_interpolate = true
            end
        end
    end

    if il > 0
        xl = view(isw.containers[idx], :, il)
    else
        # TODO: For some reason, when using view() here, we get a different
        # data-type on xl which messes up future code.
        # xl = view(x, n+1)
        xl = x[n+1]
    end
    if iu > 0
        xu = view(isw.containers[idx], :, iu)
    else
        # TODO: For some reason, when using view() here, we get a different
        # data-type on xu which messes up future code.
        # xu = view(x, n+2)
        xu = x[n+2]
    end
    δl = t-tl
    δu = tu-t
    return xu, xl, δu, δl, should_interpolate
end

function initialize_isd(Q::Int64, N::Int64, nx::Int64, use_interpolation::Bool)::InterSampleData
    if Q > 0
        isd_states = [zeros(0,nx) for j=1:N]
        isd_sample_times = [zeros(0) for j=1:N]
    else
        isd_states = [zeros(0,nx)]
        isd_sample_times = [zeros(0)]
    end
    return InterSampleData(isd_states, isd_sample_times, Q, use_interpolation)
end

function noise_inter(t::Float64,
                     Ts::Float64,       # Sampling time of noise process
                     a_vec::AbstractArray{Float64, 1},
                     n_in::Int,
                     x::AbstractArray,  # TODO: SHOULD RLY BE 1D ARRAY OF 1D ARRAYS!
                     isw::InterSampleWindow,
                     # num_sampled_per_interval::AbstractArray,
                     # num_times_visited::AbstractArray,
                     ϵ::Float64=10e-8,
                     rng::Random.TaskLocalRNG=Random.default_rng())
                     # rng::MersenneTwister=Random.default_rng())   # VERSION

    n = Int(t÷Ts)           # t lies between t0 + n*Ts and t0 + (n+1)*Ts
    # num_sampled_per_interval[n+1] += 1
    δ = t - n*Ts
    nx = length(a_vec)
    n_out = Int(length(x[1])÷nx)
    Q = isw.Q
    # P = size(z_inter[1])[1]
    P = 0
    # N = size(isd.states)[1]
    use_interpolation = isw.use_interpolation

    # TODO: Update to more efficient use of matrices. Pass C for returning stuff?
    xl = x[n+1]     # x[1] == x0
    xu = x[n+2]
    tl = n*Ts
    tu = (n+1)*Ts
    il = 0      # for il>0,   tl = isd.sample_times[n][il]
    iu = Q+1    # for iu<Q+1, tu = isd.sample_times[n][iu]
    if Q == 0 && use_interpolation
        # @warn "Used linear interpolation"
        # println("base1")
        # num_times_visited[1] += 1
        return xl + (xu-xl)*(t-tl)/(tu-tl)
    end

    #TODO: Q = 0, WE DON'T RLY PRE-ALLOCATE, SO get_neighbors FAILS!!!!!!!!

    xu, xl, δu, δl, should_interpolate = get_neighbors(n, t, x, Ts, isw)

    # If it's not possible to store any more samples for this interval, and
    # use_interpolation == True, the get_neighbors()-functions tells us to use
    # use linear interpolation for this call to ensure a smooth realization.'
    # Also use interpolation if the requested sample is too close to an existing
    # sample, and use_interpolation == true.
    if should_interpolate || ((δl < ϵ || δu < ϵ) && use_interpolation)
        # @warn "Used linear interpolation"   # DEBUG
        # if should_interpolate
        #     println("base2.1")
        # else
        #     println("base2.2")
        # end
        # num_times_visited[2] += 1
        if δl > 0
            return xl + (xu-xl)*δl/(δu+δl)
        else
            return xl
        end
    elseif δl < ϵ
        # println("base3")
        # num_times_visited[3] += 1
        return xl
    elseif δu < ϵ
        # println("base4")
        # num_times_visited[4] += 1
        return xu
    end

    As = LinearAlgebra.diagm(-1 => ones(nx-1,))
    As[1,:] = -a_vec
    BBsT = zeros(nx, nx)            # Bs*(Bs'), independent of parameters
    BBsT[1] = 1

    Mexp    = [-As BBsT; zeros(size(As)) As']
    Ml      = exp(Mexp*δl)
    Mu      = exp(Mexp*δu)
    Adl     = Ml[nx+1:end, nx+1:end]'
    Adu     = Mu[nx+1:end, nx+1:end]'
    # Using view here seems to remove 2 allocations per call, but I'm not sure
    # it provides any performance improvements
    # Adl     = view(Ml, nx+1:2*nx, nx+1:2*nx)'
    # Adu     = view(Mu, nx+1:2*nx, nx+1:2*nx)'
    AdΔ     = Adu*Adl
    σ_l     = LinearAlgebra.Hermitian(Adl*Ml[1:nx, nx+1:end])     # = B2dl
    σ_u     = LinearAlgebra.Hermitian(Adu*Mu[1:nx, nx+1:end])     # = B2du

    # σ_l = B2dl#(Bdl*(Bdl'))
    # σ_u = B2du#(Bdu*(Bdu'))
    σ_Δ = Adu*σ_l*(Adu') + σ_u
    σ_Δ_l = Adu*σ_l
    # Hermitian()-call might not be necessary, but it probably depends on the
    # model, so I leave it in to ensure that cholesky decomposition will work
    Σ = LinearAlgebra.Hermitian(σ_l - (σ_Δ_l')*(σ_Δ\(σ_Δ_l)))
    CΣ = zeros(size(Σ))
    try
        CΣ = LinearAlgebra.cholesky(Σ)
    catch e
        @warn "Cholesky decomposition failed with δ=$(min(δu, δl))"
        println("$e")
        if should_interpolate
            # @warn "Used linear interpolation"   # DEBUG
            return xl + (xu-xl)*δl/(δu+δl)
        elseif δl < δu
            return xl
        else
            return xu
        end
    end
    Σr = kron(LinearAlgebra.Diagonal(ones(n_in)), CΣ.L)
    v_Δ = xu - kron(LinearAlgebra.Diagonal(ones(n_in)), AdΔ)*xl
    μ = zeros(n_out*nx,1)
    for ind = 1:n_out
        rows = (ind-1)*nx+1:ind*nx
        μ[rows] = Adl*xl[rows] + (σ_Δ_l')*(σ_Δ\(xu[rows] - AdΔ*xl[rows]))
    end
    # μ = Adl*xl + (σ_Δ_l')*(σ_Δ\v_Δ) # TODO: Might want to double-check that this also covers n=0, but it seems to be the case

    # if isd.num_sampled_samples[n+1] < P
    #     white_noise = z_inter[n+1][num_stored_samples+1,:]
    #     isd.num_sampled_samples[n+1] += 1
    # else
    #     # @warn "Ran out of pre-generated white noise realizations for interval $(n+1)"
    #     white_noise = randn(rng, Float64, (nx, 1))
    # end
    white_noise = randn(rng, Float64, (nx*n_in, 1))
    x_new = μ + Σr*white_noise
    if isw.W > 0
        add_sample!(x_new, t, n, isw)
    end
    # println("base5")
    # num_times_visited[5] += 1
    return x_new

end

end