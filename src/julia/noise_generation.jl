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
    nx      = size(A,1)
    Mexp    = [-A B*(B'); zeros(size(A)) A']
    MTs     = exp(Mexp*Ts)
    AdTs    = MTs[nx+1:end, nx+1:end]'
    Bd2Ts   = Hermitian(AdTs*MTs[1:nx, nx+1:end])
    CholTs  = cholesky(Bd2Ts)
    BdTs    = CholTs.L
    return DT_SS_Model(AdTs, BdTs, C, x0, Ts)
end

# NOTE: z_all_inter is not used by any code anymore, this function can equally
# well be replaced by [ randn(N+1, nx) for m=1:M]
function generate_noise(N::Int64, M::Int64, P::Int64, nx::Int64)
    # N: Number of samples of uniformly sampled noise process after time 0
    # M: Number of different realizations of the noise process
    # P: Number of inter-sample noise samples stored
    # nx: Dimension of each noise sample

    # z_all_uniform[m][i,j] is the j:th element of the i:th sample of
    # realization m
    # N+1 since times including 0 and N are included, to match convention
    # used by ControlSystems.jl lsim()-function
    z_all_uniform = [ randn(N+1, nx) for m=1:M]
    # z_all_inter[m][i][p,j] is the j:th element of the p:th sample in
    # interval i of realization m
    # DEBUG: Generates z_all_inter in this way so that it can easily be compared
    # with newest noise interpolation method. Can't simply transpose randn()
    # since then its datatype stops being Array{Float64, 2}, and the code isn't
    # written generally enough to be able to handle that
    z_all_inter_skew = [ [ randn(nx, P) for i=1:N] for m=1:M]
    z_all_inter = [ [ fill(NaN, P, nx) for i=1:N] for m=1:M]
    for m=1:M
        for i=1:N
            for p=1:P
                for j=1:nx
                    z_all_inter[m][i][p,j] = z_all_inter_skew[m][i][j,p]
                end
            end
        end
    end

    return z_all_uniform, z_all_inter
end

function simulate_noise_process(mdl::DT_SS_Model, data::Array{Array{Float64,2}, 1})
    # data[m][i, j] should be the j:th component of the noise at time i of
    # realization m
    M = size(data)[1]
    N = size(data[1])[1]-1
    nx = size(data[1])[2]
    sys = ss(mdl.Ad, mdl.Bd, mdl.Cd, 0.0, mdl.Ts)
    t = 0:mdl.Ts:N*mdl.Ts
    # Allocating space for noise process
    x_process = [ fill(NaN, (nx,)) for i=1:N+1, m=1:M]
    for m=1:M
        y, t, x = lsim(sys, data[m], t, x0=mdl.x0)
        for i=1:N+1
            x_process[i,m][:] = x[i,:]
        end
    end

    # x_process[i,m][j] is the j:th element of the noise model state at sample
    # i of realization m. Sample 1 corresponds to time 0
    return x_process
end

function simulate_multivar_noise_process(mdl::DT_SS_Model, data::Array{Array{Float64,2}, 1}, n_in::Int)
    # data[m][i, k*nx + j] should be the j:th component of the noise
    # corresponding to input k at time i of realization m
    M = length(data)
    N = size(data[1])[1]-1
    nx = size(mdl.Ad)[1]
    # We only care about the state, not the output, so we ignore the C-matrix
    C_placeholder = zeros(1, nx)

    sys = ss(mdl.Ad, mdl.Bd, C_placeholder, 0.0, mdl.Ts)
    t = 0:mdl.Ts:N*mdl.Ts
    # Allocating space for noise process
    x_process = [ fill(NaN, (nx*n_in,)) for i=1:N+1, m=1:M]
    for ind = 1:n_in
        for m=1:M

            y, t, x = lsim(sys, data[m][:, (ind-1)*nx+1:ind*nx], t, x0=mdl.x0)
            for i=1:N+1
                x_process[i,m][(ind-1)*nx+1:ind*nx] = x[i,:]
            end
        end
    end

    # x_process[i,m][j] is the j:th element of the noise model state at sample
    # i of realization m. Sample 1 corresponds to time 0
    return x_process
end

# function simulate_nonuniform_noise(mdl::CT_SS_Model, z::Array{Array{Float64,1},1}, times::Array{Float64,1})
#     noise = [fill(NaN, size(mdl.x0))]
#     x = mdl.x0
#     noise[1] = x
#     t_prev = times[1]
#     for (i, t) = enumerate(times[2:end])
#         Δt = t - t_prev
#         t_prev = t
#         d_mdl = discretize_ct_noise_model(mdl.A, mdl.B, mdl.C, Δt, x)
#         x = d_mdl.Ad*x + d_mdl.Bd*z[i]
#         noise[i+1] = x
#     end
#     return noise
# end

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
