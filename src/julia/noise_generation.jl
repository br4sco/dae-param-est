using Statistics, CSV, DataFrames, ControlSystems

struct CT_SS_Model
    # Continuous-time state-space model on the form
    # dx/dt = A*x + B*u
    # y = C*x
    # where x is the state, u is the input and y is the output
    A::Array{Float64, 2}
    B::Array{Float64, 2}
    C::Array{Float64, 2}
    x0::Array{Float64, 1}
end

struct DT_SS_Model
    # Discrete-time state-space model on the form
    # x[k+1] = Ad*x[k] + Bd*u[k]
    # y[k] = Cd*x[k]
    # where x is the state, u is the input and y is the output
    Ad::Array{Float64, 2}
    Bd::Array{Float64, 2}
    Cd::Array{Float64, 2}
    x0::Array{Float64, 1}
    Ts::Float64                     # Sampling period of the system
end

# This struct is deprecated, use DT_SS_Model instead
struct NoiseModel
    # Time-discrete noise model is on the form
    # x[k+1] = Ad*x[k] + Bd*z[k]
    # y[k] = Cd*x[k]
    # where x is the state, z is the white-noise input, and y is the output
    # Ts: The sampling period of the system
    # x0: The initial state that the model starts in
    Ad::Array{Float64, 2}
    Bd::Array{Float64, 2}
    Cd::Array{Float64, 2}
    x0::Array{Float64, 1}
    Ts::Float64
end

function discretize_ct_noise_model(A, B, C, Ts, x0)::DT_SS_Model
    Mexp    = [-A B*(B'); zeros(size(A)) A']
    MTs     = exp(Mexp*Ts)
    AdTs    = MTs[nx+1:end, nx+1:end]'
    Bd2Ts   = Hermitian(AdTs*MTs[1:nx, nx+1:end])
    CholTs  = cholesky(Bd2Ts)
    BdTs    = CholTs.L
    return DT_SS_Model(AdTs, BdTs, C, x0, Ts)
end

function discretize_ct_model(A, B, C, Ts, x0)
    @warn "The function discretize_ct_model() from noise_generation.jl has been deprecated, use discretize_ct_noise_model() instead"
    discretize_ct_noise_model(A, B, C, Ts, x0)
end

function generate_noise(N::Int64, M::Int64, P::Int64, nx::Int64, save_data::Bool=true)
    # N: Number of samples of uniformly sampled noise process after time 0
    # To generate noise also at time 0, we need to generate N+1 noise samples
    # M: Number of different realizations of the noise process
    # P: Number of inter-sample noise samples stored
    # nx: Dimension of each noise sample

    @warn "The function generate_noise() from noise_generation.jl is deprecated, use generate_noise_new() instead"

    # Let z[i,m,j] be the j:th element of the i:th sample of the m:th
    # realization of the process z. Then z_uniform should be interpreted as
    # on the form
    # [z[1,1,1], ..., z[1,1,nx], z[1,2,1]..., z[1,M,nx]
    #   ...
    #  z[N+1,1,1], ...      ...      ...    ..., z[N+1,M,nx]]
    z_uniform = randn(Float64, N+1, nx*M)
    # TODO: Saving z_inter seems to often (but not always) break for
    # 4th degree systems, worth figuring out why... EDIT: Seems simply that
    # it happends if DataFrames become too large

    # Let z[i,m,p,j] be the j:th element of the p:th sample in the i:th
    # inter-sample interval of the m:th realization of the process z.
    # Then z_inter should be interpreted as on the form
    # [z[1,1,1,1], ..., z[1,1,1,nx], z[1,1,2,1]..., z[1,1,P,nx], z[1,2,1,1],... z[1,M,P,nx]
    #   ...
    # [z[N+1,1,1,1], ...     ...      ...      ...      ...      ...       ...    z[N+1,M,P,nx]]
    z_inter   = randn(Float64, N+1, nx*M*P)
    if save_data
        CSV.write("z_uniform.csv", DataFrame(z_uniform))
        # CSV.write("z_inter.csv", DataFrame(z_inter))
        CSV.write("metadata.csv", DataFrame([N M P nx]))
    end
    return z_uniform, z_inter
end

function generate_noise_new(N::Int64, M::Int64, P::Int64, nx::Int64)
    # N: Number of samples of uniformly sampled noise process after time 0
    # M: Number of different realizations of the noise process
    # P: Number of inter-sample noise samples stored
    # nx: Dimension of each noise sample

    # z_all_uniform[m][i][j] is the j:th element of the i:th sample of
    # realization m
    # N+1 since times including 0 and N are included, to match convention
    # used by ControlSystems.jl lsim()-function
    # z_all_uniform = fill(fill(NaN, (N+1, nx)), M)
    z_all_uniform = [ randn(N+1,nx) for m=1:M]
    # z_all_inter[m][i][p][j] is the j:th element of the p:th sample in
    # interval i of realization m
    # z_all_inter = fill(fill(fill(NaN, (P, nx)), N), M)
    z_all_inter = [ [ randn(P,nx) for i=1:N] for m=1:M]

    return z_all_uniform, z_all_inter
end

function load_data(N::Int64, M::Int64, P::Int64, nx::Int64)
    z_uni_mat = CSV.read("z_uniform.csv", DataFrame)
    # z_inter_mat = CSV.read("z_inter.csv", DataFrame)
    z_uniform = [ z_uni_mat[i,j] for i=1:size(z_uni_mat)[1], j=1:size(z_uni_mat)[2]]
    # z_intersample = [ z_inter_mat[i,j] for i=1:size(z_inter_mat)[1], j=1:size(z_inter_mat)[2]]
    return z_uniform#, z_intersample
end

function load_metadata()
    try
        metadata_frame = CSV.read("metadata.csv", DataFrame)
        return [metadata_frame[1,i] for i = 1:size(metadata_frame)[2]]
    catch
        # This happens if e.g. no metadata file exists yet
        return nothing
    end
end

function simulate_noise_process(mdl::DT_SS_Model, data::Array{Float64,2})::Array{Array{Float64, 1}, 2}
    @warn "The function simulate_noise_process() from noise_generation.jl is deprecated, use simulate_noise_process_new() instead"
    (Np1, Mnx) = size(data)
    Ts = mdl.Ts
    N = Np1 - 1     # We have noise for times 0 to N, so a total of N+1 samples
    nx = size(mdl.Ad)[1]
    M = Int(Mnx÷nx)
    sys = ss(mdl.Ad, mdl.Bd, mdl.Cd, 0.0, mdl.Ts)
    t = 0:Ts:N*Ts
    # Allocating space for noise process
    x_process = [ fill(NaN, (nx,)) for i=1:N, m=1:M]
    for m=1:M
        y, t, x = lsim(sys, data[:, (1+(m-1)*nx):m*nx], t, x0=mdl.x0)
        for i=1:N
            x_process[i,m][:] = x[i+1,:]    # i+1, since first elemt of x is at time 0
        end
    end

    return x_process
end

function simulate_noise_process_new(mdl::DT_SS_Model, data::Array{Array{Float64,2}, 1})
    # data[m][i, j] should be the j:th component of the noise at time i of
    # realization m
    M = size(data)[1]
    N = size(data[1])[1]-1
    nx = size(data[1])[2]
    sys = ss(mdl.Ad, mdl.Bd, mdl.Cd, 0.0, mdl.Ts)
    t = 0:mdl.Ts:N*mdl.Ts
    # Allocating space for noise process
    x_process = [ fill(NaN, (nx,)) for i=1:N, m=1:M]
    for m=1:M
        y, t, x = lsim(sys, data[m], t, x0=mdl.x0)
        for i=1:N
            x_process[i,m][:] = x[i+1,:]    # i+1, since first elemt of x is at time 0
        end
    end

    return x_process
end

# DEBUG For checking if the simulated noise process seems to have the expected
# statistical properties
function test_generated_data(mdl::DT_SS_Model, x_process::Array{Array{Float64, 2}})
    (N, M) = size(x_process)
    nx = size(x_process[1,1])[1]
    Ad = mdl.Ad
    Bd = mdl.Bd
    # picking some arbitrary time to look at
    t = 25

    # Computes the correct (theoretical) values for mean, variance, and covariance
    μ = reshape([0;0], (2,1))
    σ_nm1 = zeros((nx, nx))
    for i =1:t-1
        σ_nm1 = Ad*σ_nm1*(Ad') + Bd*(Bd')   # double check!!!!!!!!!!!!!!!
    end
    σ_n = Ad*σ_nm1*(Ad') + Bd*(Bd')
    σ_n_nm1 = Ad*σ_n

    x_nm1 = NaN*ones(M, nx)
    x_n   = NaN*ones(M, nx)
    for m=1:M
        x_nm1[m,:] = x_process[t-1, m]
        x_n[m,:]   = x_process[t, m]
    end
    # Takes mean along the columns, i.e. each row counts as one realization
    m_nm1 = mean(x_nm1, dims=1)
    m_n   = mean(x_n, dims=1)
    s_nm1 = cov(x_nm1, dims=1)
    s_n   = cov(x_n, dims=1)
    s_n_nm1 = cov(x_n, x_nm1, dims=1)

    println("Mean differences:")
    println(μ - m_nm1')
    println(μ - m_n')
    println("Absolute variance differences")
    println(σ_nm1 - s_nm1)
    println(σ_n - s_n)
    println("Relative variance differences")
    println(opnorm(σ_nm1 - s_nm1)/opnorm(σ_nm1))
    println(opnorm(σ_n - s_n)/opnorm(σ_n))
    println("Absolute covariance difference")
    println(σ_n_nm1 - s_n_nm1)
    println("Relative covariance difference")
    println(opnorm(σ_n_nm1 - s_n_nm1)/opnorm(σ_n_nm1))
end
