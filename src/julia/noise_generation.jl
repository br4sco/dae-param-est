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
end

struct DisturbanceMetaData
    nx::Int
    n_in::Int
    n_out::Int
    η::Array{Float64,1}
    free_par_inds::Array{Int64,1}
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

function discretize_ct_noise_model_with_sensitivities(
    mdl::CT_SS_Model, Ts::Float64, sens_inds::Array{Int64, 1})::DT_SS_Model
    # sens_inds: indices of parameter with respect to which we compute the
    # sensitivity of disturbance output w

    nx = size(mdl.A,1)
    n_out = size(mdl.C, 1)
    n_in  = size(mdl.C, 2)÷nx
    function get_k_j_i(zeta::Int64)::Tuple{Int64, Int64, Int64}
        k = rem( (zeta-1), nx ) + 1
        j = rem( (zeta - k - nx), nx*n_in)÷nx + 1
        i = (zeta - j*nx - k)÷(nx*n_in) + 1
        return k, j, i
    end

    # Indices of free parameters corresponding to "a-vector" in disturbance model
    sens_inds_a = sens_inds[findall(sens_inds .<= nx)]
    q   = length(sens_inds)
    q_a = length(sens_inds_a)
    nx_sens = (1+q_a)*nx
    A_mat = zeros(nx_sens, nx_sens)
    A_mat[1:nx, 1:nx] = mdl.A
    for i in 1:length(sens_inds_a)
        Aη = zeros(nx, nx)
        Aη[1, sens_inds_a[i]] = -1
        A_mat[i*nx+1:(i+1)*nx, i*nx+1:(i+1)*nx] = mdl.A
        A_mat[i*nx+1:(i+1)*nx, 1:nx] = Aη
    end
    B_mat = vcat(mdl.B, zeros(q_a*nx, size(mdl.B, 2)) )
    my_I = Matrix{Float64}(I, q_a*nx, q_a*nx)
    C_mat = zeros((q+1)n_out, nx_sens*n_in)
    for row_block = 1:q+1
        if row_block == 1
            for col_block = 1:n_in
                C_mat[(row_block-1)*n_out+1:row_block*n_out,
                    (col_block-1)*(q_a+1)*nx+1:(col_block-1)*(q_a+1)*nx+nx] =
                    mdl.C[:,(col_block-1)*nx+1:col_block*nx]
                # C_mat[:, (col_block-1)*(q_a+1)*nx+1:col_block*(q_a+1)*nx]
                # = hcat(mdl.C[:,(col_block-1)*nx+1:col_block*nx], zeros(n_out, q_a*nx))
            end
        elseif row_block <= q_a+1
            ind = row_block-1
            for col_block = 1:n_in
                C_mat[(row_block-1)*n_out+1:row_block*n_out,
                    ind*nx+(col_block-1)*(q_a+1)*nx+1:ind*nx+(col_block-1)*(q_a+1)*nx+nx] =
                    mdl.C[:,(col_block-1)*nx+1:col_block*nx]
            end
        else
            k, j, i = get_k_j_i(sens_inds[row_block-1])
            C_mat[(row_block-1)*n_out+i, (j-1)*(q_a+1)*nx+k] = 1
        end
    end

    # for col_block = 1:n_in
    #     C_mat[1:n_out, (col_block-1)*(q_a+1)*nx+1:(col_block-1)*(q_a+1)*nx+nx] =
    #         mdl.C[:,(col_block-1)*nx+1:col_block*nx]
    # end
    # for row_block = 2:1+q_a
    #     C_mat[n_out+(row_block-2)*nx*q_a+1:n_out+(row_block-1)*nx*q_a,
    #         nx+(row_block-2)*(q_a+1)*nx+1:nx+(row_block-2)*(q_a+1)*nx+q_a*nx] =
    #         my_I
    # end

    Mexp    = [-A_mat B_mat*(B_mat'); zeros(size(A_mat)) A_mat']
    MTs     = exp(Mexp*Ts)
    AdTs    = MTs[nx_sens+1:end, nx_sens+1:end]'
    Bd2Ts   = Hermitian(AdTs*MTs[1:nx_sens, nx_sens+1:end])
    CholTs  = cholesky(Bd2Ts)
    BdTs    = CholTs.L
    svd_Bd = svd(Bd2Ts)
    return DT_SS_Model(AdTs, BdTs, C_mat, zeros(nx_sens*n_in), Ts)
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
    N = size(data[1], 1)-1
    nx = size(mdl.Ad, 1)
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

# TODO: Instead of having disturbance metadata as a DataFrame, isn't it better
# to create a custom struct???

# Used for disturbance
function disturbance_model_1(Ts::Float64; bias::Float64=0.0, scale::Float64=0.6)::Tuple{DT_SS_Model, DataFrame}
    nx = 2        # model order
    n_out = 2     # number of outputs
    n_in = 2      # number of inputs
    w_scale = scale*ones(n_out)             # noise scale
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
    return mdl, DataFrame(nx = nx, n_in = n_in, n_out = n_out, η = η0, bias=bias)
end

# Used for input
function disturbance_model_2(Ts::Float64; bias::Float64=0.0, scale::Float64=0.2)::Tuple{DT_SS_Model, DataFrame}
    nx = 2        # model order
    n_out = 1     # number of outputs
    n_in = 2      # number of inputs
    u_scale = scale # input scale
    # Denominator of every transfer function is given by p(s), where
    # p(s) = s^n + a[1]*s^(n-1) + ... + a[n-1]*s + a[n]
    a_vec = [0.8, 4^2]
    c_vec = zeros(n_out*nx*n_in)
    # The first state will act as output of the filter
    c_vec[1] = u_scale
    η0 = vcat(a_vec, Diagonal(w_scale)*c_vec)
    dη = length(η0)
    mdl =
        discretize_ct_noise_model(get_ct_disturbance_model(η0, nx, n_out), Ts)
    return mdl, DataFrame(nx = nx, n_in = n_in, n_out = n_out, η = η0, bias=bias)
end

# Used for scalar disturbance and input
function disturbance_model_3(Ts::Float64; bias::Float64=0.0, scale::Float64=1.0)::Tuple{DT_SS_Model, DataFrame}
    ω = 4         # natural freq. in rad/s (tunes freq. contents/fluctuations)
    ζ = 0.1       # damping coefficient (tunes damping)
    nx = 2        # model order
    n_out = 1     # number of outputs
    n_in = 1      # number of inputs
    w_scale = scale*ones(n_out)             # noise scale
    # Denominator of every transfer function is given by p(s), where
    # p(s) = s^n + a[1]*s^(n-1) + ... + a[n-1]*s + a[n]
    a_vec = [2*ω*ζ, ω^2]
    c_vec = Diagonal(w_scale)*[0, 1]
    η0 = vcat(a_vec, c_vec)
    dη = length(η0)
    mdl =
        discretize_ct_noise_model(get_ct_disturbance_model(η0, nx, n_out), Ts)
    return mdl, DataFrame(nx = nx, n_in = n_in, n_out = n_out, η = η0, bias=bias)
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

function get_filtered_noise(gen::Function, Ts::Float64, M::Int, Nw::Int;
    bias::Float64=0.0, scale::Float64=1.0)::Tuple{Array{Float64,2}, Array{Float64,2}, DataFrame}

    mdl, metadata = gen(Ts, bias=bias, scale=scale)
    n_tot = size(mdl.Cd,2)

    ZS = [randn(Nw, n_tot) for m = 1:M]
    XW = simulate_noise_process_mangled(mdl, ZS)
    XW, get_system_output_mangled(mdl, XW).+ metadata.bias[1], metadata
end

# TODO: Is this even used? Remove?
function get_reactor_debug_input(Ts::Float64, Nw::Int)
    u1(t) = 0.3 + 0.05*sin(t);        # FA
    u2(t) = 3.2;                      # CA0
    u3(t) = 293.15;                   # TA
    u4(t) = 0.3 + 0.02*sin(0.5*t);    # F
    u5(t) = 0.1 + 0.01*sin(2*t);      # Fh
    u6(t) = 313.30;                   # Th
    u(t) = [u1(t), u2(t), u3(t), u4(t), u5(t), u6(t)]

    U = zeros(Nw, 6)
    ts = 0.0:Ts:(Nw-1)*Ts
    for ind = 1:length(ts)
        U[ind,:] = u(ts[ind])
    end
    return U
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
