using Statistics, CSV, DataFrames, ControlSystems, LinearAlgebra

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
    # In MIMO case, we have
end

# =================== Helper Functions ==========================

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

function discretize_ct_noise_model(mdl::CT_SS_Model, Ts::Float64)::DT_SS_Model
    nx = size(mdl.A,1)
    Mexp    = [-mdl.A mdl.B*(mdl.B'); zeros(size(mdl.A)) mdl.A']
    MTs     = exp(Mexp*Ts)
    AdTs    = MTs[nx+1:end, nx+1:end]'
    Bd2Ts   = Hermitian(AdTs*MTs[1:nx, nx+1:end])
    CholTs  = cholesky(Bd2Ts)
    BdTs    = CholTs.L
    return DT_SS_Model(AdTs, BdTs, mdl.C, mdl.x0, Ts)
end

# Converts parameter values for C-matrix as entered into a single vector more
# suitable to be used in the code
function get_c_parameter_vector(c_vals, w_scale, nx::Int, n_out::Int, n_in::Int)
    c_vec = zeros(nx*n_out*n_in)
    for i = 1:n_out
        for j = 1:n_in
            for k = 1:nx
                # Multiplying with w_scale[i] here is equivalent to scaling ith
                # component of the disturbance output (w) with factor w_scale[i]
                c_vec[(j-1)*n_out*nx + (k-1)*n_out + i] = w_scale[i]*c_vals[i,j][k]
            end
        end
    end
    return c_vec
end

function get_ct_disturbance_model(η::Array{Float64,1}, nx::Int, n_out::Int)
    # First nx parameters of η are parameters for A-matrix, the remaining
    # parameters are for the C-matrix
    A = diagm(-1 => ones(nx-1,))
    A[1,:] = -η[1:nx]
    B = zeros(nx,1)
    B[1] = 1.0
    C = reshape(η[nx+1:end], n_out, :)
    x0 = zeros(nx)
    return CT_SS_Model(A, B, C, x0)
end

# ================= Functions simulating disturbance =======================

# These functions simulate the noise process when using a noise model
# following the scalar convention. The scalar case is a special case of the
# multivariate case, so there's no reason to use these functions
# function simulate_noise_process(
#     mdl::DT_SS_Model,
#     data::Array{Array{Float64,2}, 1}
# )
#     # data[m][i, j] should be the j:th component of the noise at time i of
#     # realization m
#     M = size(data)[1]
#     N = size(data[1])[1]-1
#     nx = size(data[1])[2]
#     sys = ss(mdl.Ad, mdl.Bd, mdl.Cd, 0.0, mdl.Ts)
#     t = 0:mdl.Ts:N*mdl.Ts
#     # Allocating space for noise process
#     x_process = [ fill(NaN, (nx,)) for i=1:N+1, m=1:M]
#     for m=1:M
#         y, t, x = lsim(sys, data[m], t, x0=mdl.x0)
#         for i=1:N+1
#             x_process[i,m][:] = x[i,:]
#         end
#     end
#
#     # x_process[i,m][j] is the j:th element of the noise model state at sample
#     # i of realization m. Sample 1 corresponds to time 0
#     return x_process
# end
#
# function simulate_noise_process_mangled(
#     mdl::DT_SS_Model,
#     data::Array{Array{Float64,2}, 1}
# )
#     # data[m][i, j] should be the j:th component of the noise at time i of
#     # realization m
#     M = size(data)[1]
#     N = size(data[1])[1]-1
#     nx = size(data[1])[2]
#     sys = ss(mdl.Ad, mdl.Bd, mdl.Cd, 0.0, mdl.Ts)
#     t = 0:mdl.Ts:N*mdl.Ts
#     # Allocating space for noise process
#     # x_process = [ fill(NaN, (nx,)) for i=1:N+1, m=1:M]
#     x_process = fill(NaN, ((N+1)*nx, M))
#     for m=1:M
#         y, t, x = lsim(sys, data[m], t, x0=mdl.x0)
#         for i=1:N+1
#             x_process[(i-1)*nx+1:i*nx, m] = x[i,:]
#         end
#     end
#
#     # x_process[(i-1)*nx + j, m] is the j:th element of the noise model at
#     # sample i of realization m. Sample 1 corresponds to time 0
#     return x_process
# end

function simulate_noise_process(
    mdl::DT_SS_Model,
    data::Array{Array{Float64,2}, 1}
)
    # data[m][i, k*nx + j] should be the j:th component of the noise
    # corresponding to input k at time i of realization m
    M = length(data)
    N = size(data[1], 1)-1
    nx = size(mdl.Ad, 1)
    n_in = size(mdl.Cd, 2)÷nx
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

function simulate_noise_process_mangled(
    mdl::DT_SS_Model,
    data::Array{Array{Float64,2}, 1},
)
    # data[m][i, k*nx + j] should be the j:th component of the noise
    # corresponding to input k at time i of realization m
    M = length(data)
    N = size(data[1])[1]-1
    nx = size(mdl.Ad)[1]
    n_in = size(mdl.Cd, 2)÷nx
    # We only care about the state, not the output, so we ignore the C-matrix
    C_placeholder = zeros(1, nx)

    sys = ss(mdl.Ad, mdl.Bd, C_placeholder, 0.0, mdl.Ts)
    t = 0:mdl.Ts:N*mdl.Ts
    # Allocating space for noise process
    x_process = fill(NaN, ((N+1)*nx*n_in, M))
    for ind = 1:n_in
        for m=1:M
            # x[i, j] is the j:th element of the state at sample i
            y, t, x = lsim(sys, data[m][:, (ind-1)*nx+1:ind*nx], t, x0=mdl.x0)
            for i=1:N+1
                x_process[(i-1)*n_in*nx + (ind-1)*nx + 1: (i-1)*n_in*nx + ind*nx, m] = x[i,:]
            end
        end
    end

    # x_process[(i-1)*nx*n_in + j, m] is the j:th element of the noise model at
    # sample i of realization m. Sample 1 corresponds to time 0
    return x_process
end

# ============== Functions for generating specific realization ===============


# Used for disturbance
function disturbance_model_1(Ts::Float64)::Tuple{DT_SS_Model, DataFrame}
    nx = 2        # model order
    n_out = 2     # number of outputs
    n_in = 2      # number of inputs
    w_scale = 0.6*ones(n_out)             # noise scale
    # Denominator of every transfer function is given by p(s), where
    # p(s) = s^n + a[1]*s^(n-1) + ... + a[n-1]*s + a[n]
    a_vec = [0.8, 4^2]
    # Transfer function (i,j) has numerator c_[i,j][1]s^{nx-1} + ... + c_[i,j][nx]
    c_ = [zeros(nx) for i=1:n_out, j=1:n_in]
    c_[1,1][nx] = 1 # c_[1,1][:] = vcat(zeros(nx-1), [1])
    c_[2,2][nx] = 1 # c_[2,2][:] = vcat(zeros(nx-1), [1])
    c_vec = get_c_parameter_vector(c_, w_scale, nx, n_out, n_in)
    η0 = vcat(a_vec, c_vec)
    dη = length(η0)
    mdl =
        discretize_ct_noise_model(get_ct_disturbance_model(η0, nx, n_out), Ts)
    return mdl, DataFrame(nx = nx, n_in = n_in, n_out = n_out, η = η0)
end

# Used for input
function disturbance_model_2(Ts::Float64)::Tuple{DT_SS_Model, DataFrame}
    nx = 2        # model order
    n_out = 1     # number of outputs
    n_in = 2      # number of inputs
    w_scale = 0.2*ones(n_out)             # noise scale
    # Denominator of every transfer function is given by p(s), where
    # p(s) = s^n + a[1]*s^(n-1) + ... + a[n-1]*s + a[n]
    a_vec = [0.8, 4^2]
    c_vec = zeros(n_out*nx*n_in)
    # The first state will act as output of the filter
    c_vec[1] = 1
    η0 = vcat(a_vec, c_vec)
    dη = length(η0)
    mdl =
        discretize_ct_noise_model(get_ct_disturbance_model(η0, nx, n_out), Ts)
    return mdl, DataFrame(nx = nx, n_in = n_in, n_out = n_out, η = η0)
end

# Used for scalar disturbance and input
# Used for input
function disturbance_model_3(Ts::Float64)::Tuple{DT_SS_Model, DataFrame}
    ω = 4         # natural freq. in rad/s (tunes freq. contents/fluctuations)
    ζ = 0.1       # damping coefficient (tunes damping)
    nx = 2        # model order
    n_out = 1     # number of outputs
    n_in = 1      # number of inputs
    w_scale = 0.2*ones(n_out)             # noise scale
    # Denominator of every transfer function is given by p(s), where
    # p(s) = s^n + a[1]*s^(n-1) + ... + a[n-1]*s + a[n]
    a_vec = [2*ω*ζ, ω^2]
    c_vec = [0, 1]
    η0 = vcat(a_vec, c_vec)
    dη = length(η0)
    mdl =
        discretize_ct_noise_model(get_ct_disturbance_model(η0, nx, n_out), Ts)
    return mdl, DataFrame(nx = nx, n_in = n_in, n_out = n_out, η = η0)
end
# function disturbance_model_3(Ts::Float64)::Tuple{DT_SS_Model, DataFrame}
#
#     ω = 4           # natural freq. in rad/s (tunes freq. contents/fluctuations)
#     ζ = 0.1          # damping coefficient (tunes damping)
#     d1 = 1.0
#     d2 = 2 * ω * ζ
#     d3 = ω^2
#     a = [1.0]
#     b = [d1, d2, d3]
#     s = ss(tf(a, b))
#     ct_mdl = CT_SS_Model(s.A, s.B, s.C, zeros(size(s.A,2)))
#     mdl = discretize_ct_noise_model(ct_mdl, Ts)
#     return mdl, DataFrame(nx = size(s.A,2), n_in = 1, n_out = 1, n_a = length(a), η = vcat(a, b))
# end

function get_filtered_noise_multivar(gen::Function, Ts::Float64, M::Int, Nw::Int
    )::Tuple{Array{Float64,2}, Array{Float64,2}, DataFrame}

    mdl, metadata = gen(Ts)
    n_tot = size(mdl.Cd,2)

    ZS = [randn(Nw, n_tot) for m = 1:M]
    XW = simulate_noise_process_mangled(mdl, ZS)
    XW, get_system_output_mangled(mdl, XW), metadata
end

# function get_filtered_noise_scalar(gen::Function, Ts::Float64, M::Int, Nw::Int
#     )::Tuple{Array{Float64,2}, Array{Float64,2}, DataFrame}
#
#     mdl, metadata = gen(M, Ts, Nw)
#     nx = size(mdl.Ad,2)
#
#     ZS = [randn(Nw, nx) for m = 1:M]
#     XW = simulate_noise_process_mangled(mdl, ZS)
#     XW, get_system_output_mangled(mdl, XW), metadata
# end

# Converts mangled states to an output vector using the provided model
function get_system_output_mangled(mdl::DT_SS_Model, states::Array{Float64, 2}
    )::Array{Float64, 2}
    M = size(states, 2)
    (n_out, n_tot) = size(mdl.Cd)
    N = size(states, 1)÷n_tot
    output = zeros(N*n_out, M)
    for m=1:M
        for t=1:N
            output[(t-1)*n_out+1:t*n_out, m] = mdl.Cd*states[(t-1)*n_tot+1:t*n_tot, m]
        end
    end
    return output
end

# ====================== Other functions ==========================

# DEBUG For checking if the simulated noise process seems to have the expected
# statistical properties
function test_generated_data(
    mdl::DT_SS_Model,
    x_process::Array{Array{Float64, 2}}
)
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
