module NoiseGeneration

# To test which of these are necessary, consider replacing "using" with "import", then one has to write e.g. Random.seed!() to call seed!()
import Random
using DataFrames: DataFrame
using LinearAlgebra: Diagonal, diagm, Hermitian, cholesky, I, LowerTriangular, eigen
using ControlSystems: ss, lsim
# import Statistics, CSV, DataFrames, ControlSystems, LinearAlgebra, Random


seed = 54321    # Important that random samples generated here are independent of those generated in run_experiment.jl
Random.seed!(seed)
struct CT_SS_Model
    # Continuous-time state-space model on the form
    # dx/dt = A*x + B*u
    # y = C*x
    # where x is the state, u is the input and y is the output
    A::Matrix{Float64}
    B::Matrix{Float64}
    C::Matrix{Float64}
    x0::Vector{Float64}
end

struct DT_SS_Model
    # Discrete-time state-space model on the form
    # x[k+1] = Ad*x[k] + Bd*u[k]
    # y[k] = Cd*x[k]
    # where x is the state, u is the input and y is the output
    Ad::Matrix{Float64}
    Bd::Matrix{Float64}
    Cd::Matrix{Float64}
    x0::Vector{Float64}
    Ts::Float64                     # Sampling period of the system
end

struct DisturbanceMetaData
    nx::Int
    nv::Int   # n_tot = nx*nv
    nw::Int
    η::Vector{Float64}
    free_par_inds::Vector{Int}
    # Vector containing lower and upper bound of a disturbance parameter in each row
    free_par_bounds::Matrix{Float64}
    # get_all_ηs encodes what information of the disturbance model is known
    # This function should always return all parameters of the disturbance model,
    # given only the free parameters
    get_all_ηs::Function
    num_rels::Int
    Nw::Int
    δ::Float64
end

# =================== Helper Functions ==========================

demangle_XW(XW::AbstractMatrix{Float64}, n_tot::Int) = [XW[(i-1)*n_tot+1:i*n_tot, m] for i=1:(size(XW,1)÷n_tot), m=1:size(XW,2)]

function Phi(mat_in::Matrix{Float64}, n_tot::Int)::LowerTriangular
    mat = LowerTriangular(mat_in)
    for i=1:n_tot
        mat[i,i] *= 0.5
    end
    return mat
end

# Given index of element in C-matrix, returns row and col of that index
function get_C_row_and_col(ind::Int64, n_tot::Int64)::Tuple{Int64, Int64}
    Ĩ = (ind-1)÷n_tot
    L̃ = ind-1 - Ĩ*n_tot
    Ĩ+1, L̃+1
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

# Applies Corollary 5.1 from my thesis. Just computed the resulting matrices (corresponding to discretization before differentiation)
# and does not return the discrete-time system. Note that the Corollary assumes that B is not parametrized
function get_disc_then_diff_matrices(mdl::CT_SS_Model, Ts::Float64, sens_inds::Vector{Int64})::Tuple{Matrix{Float64}, Matrix{Float64}, Matrix{Float64}}
    # sens_inds: indices of parameter with respect to which we compute the
    # sensitivity of disturbance output w

    nv = size(mdl.B, 2)
    n_tot = size(mdl.A, 1)
    nx = n_tot÷nv
    # Indices of free parameters corresponding to "a-vector" in vector of all disturbance parameters
    sens_inds_a = sens_inds[findall(sens_inds .<= nx)]
    na = length(sens_inds_a)

    Aηa = zeros(na*n_tot, n_tot)
    for ind1 = 1:na
        for ind2 = 1:nv
            Aηa[(ind1-1)n_tot + (ind2-1)nx + 1, (ind2-1)nx + sens_inds_a[ind1]] = -1
        end
    end

    M = [mdl.A                    zeros(n_tot, na*n_tot)                mdl.B*(mdl.B');
         Aηa                kron(Matrix(1.0I, na, na), mdl.A)   zeros(n_tot*na, n_tot);
         zeros(n_tot, n_tot)     zeros(n_tot, n_tot*na)                -mdl.A' ]
    Mexp = exp(M*Ts)
    Ãd   = Mexp[1:n_tot, 1:n_tot]   # Ãd = e^(A*Ts)
    Dd   = Hermitian(Mexp[1:n_tot, (na+1)*n_tot+1:(na+2)*n_tot]*(Ãd'))
    B̃d   = try
        cholesky(Dd).L
    catch PosDefException
        # Due to numerical inaccuracies, Dd can occasionally become indefinite, 
        # at which point we modify the diagonal elements as little as possible
        # but enough to make it positive definite
        eigs = eigen(Dd).values
        @warn "Had to modify Dd with $(-eigs[1])"
        # eigs[1] is the lowest (in this case negative) eigenvale. By subtracting it, 
        # in theory the matrix should become positive SEMIDEFINITE, but because of numerical
        # inaccuracies it has so far become positive definite when I have tested. If it ever
        # throws another PosDefException, it might be worth subtracting twice as much
        try
            cholesky(Dd - eigs[1]*I).L
        catch PosDefException
            @warn "Actually had to increase by even 1e-20"
            cholesky(Dd + (1e-20)*I).L
        end
    end

    # NEW, "inverts"/solves equations with many smaller matrices instead of one huge one. Should be more efficient.
    B̃dηa = zeros(na*n_tot, n_tot)
    for i = 1:na
        #   H = Mexp[i*n_tot+1:(i+1)*n_tot, n_tot*(na+1)+1:n_tot*(2+na)]
        Ddηai = Mexp[i*n_tot+1:(i+1)*n_tot, n_tot*(na+1)+1:n_tot*(2+na)]*(Ãd')
        Ddηai += Ddηai'
        B̃dηa[(i-1)*n_tot+1:i*n_tot,:] = B̃d*Phi((B̃d\Ddηai)/(B̃d'), n_tot)
    end

    return Mexp, B̃d, B̃dηa
end

# Discretizes ct model before differentiation, corresponds to Corollary 5.1 in my Licentiate theisis (i.e. also assumes that B is not parametrized)
# Obtains disturbance model corresponding to Proposition 5.2 in my Licentiate thesis
function discretize_ct_noise_model_disc_then_diff(mdl::CT_SS_Model, Ts::Float64, sens_inds::Vector{Int64})::DT_SS_Model
    Mexp, B̃d, B̃dηa = get_disc_then_diff_matrices(mdl, Ts, sens_inds)

    nv = size(mdl.B, 2)
    nw = size(mdl.C, 1)
    n_tot = size(mdl.A, 1)
    nx = n_tot÷nv
    nη = length(sens_inds)
    na = length(findall(sens_inds .<= nx))

    A_mat = Mexp[1:(na+1)n_tot, 1:(na+1)n_tot]
    B_mat = [B̃d; B̃dηa]
    C_mat = zeros((nη+1)nw, (1+na)*n_tot)
    # Because C-matrix only depends on C-parameters and the disturbance state xw depends only on a-parameters, we have
    # C_mat = [C    0
    #          0   Ina⊗C
    #          Cηc  0]
    C_mat[1:nw, 1:n_tot] = mdl.C
    C_mat[nw+1:(1+na)*nw, n_tot+1:end] = kron(Matrix(1.0I, na, na), mdl.C)
    for ηind = na+1:nη
        # We want to pass the index of the currently considered c-parameter in the C-matrix.
        # sens_inds contains the index of that parameter in η, which contains the additional na
        # parameters corresponding to the A-matrix
        row, col = get_C_row_and_col(sens_inds[ηind]-na, n_tot)  # row and col of the currently considered parameter in mdl.C
        C_mat[ηind*nw + row, col] = 1.0
    end

    return DT_SS_Model(A_mat, B_mat, C_mat, zeros((1+na)n_tot), Ts)
end

# Differentiates ct model before discretization, corresponds to Proposition 5.1 in my Licentiate thesis. 
# Assumes that B is not parametrized, but this does not really simplify the function, just makes some elements zero.
function discretize_ct_noise_model_diff_then_disc( mdl::CT_SS_Model, Ts::Float64, sens_inds::Vector{Int64})::DT_SS_Model
    # sens_inds: indices of parameter with respect to which we compute the
    # sensitivity of disturbance output w

    nv = size(mdl.B, 2)
    nw = size(mdl.C, 1)
    n_tot = size(mdl.A, 1)
    nx = n_tot÷nv
    # Indices of free parameters corresponding to "a-vector" in vector of all disturbance parameters
    sens_inds_a = sens_inds[findall(sens_inds .<= nx)]
    nη   = length(sens_inds)
    na = length(sens_inds_a)

    Aηa = zeros(na*n_tot, n_tot)
    for ind1 = 1:na
        for ind2 = 1:nv
            Aηa[(ind1-1)n_tot + (ind2-1)nx + 1, (ind2-1)nx + sens_inds_a[ind1]] = -1.0
        end
    end

    # Working under the assumption B_θ = 0
    M = [mdl.A                  zeros(n_tot, na*n_tot)                  mdl.B*(mdl.B')          zeros(n_tot, n_tot*na);
         Aηa                kron(Matrix(1.0I, na, na), mdl.A)           zeros(nx*na, nx)       zeros(n_tot*na, n_tot*na);
         zeros(n_tot, n_tot)     zeros(n_tot, n_tot*na)                         -mdl.A'              -Aηa';
         zeros(n_tot*na, n_tot)    zeros(n_tot*na, n_tot*na)           zeros(n_tot*na, n_tot)      kron(Matrix(1.0I, na, na), -mdl.A')]
    Mexp = exp(M*Ts)

    Ad = Mexp[1:n_tot*(na+1), 1:n_tot*(na+1)]
    Dd = Hermitian(Mexp[1:n_tot*(na+1), n_tot*(na+1)+1:2n_tot*(na+1)]*(Ad'))
    Bd = try
        cholesky(Dd).L
    catch PosDefException
        # Due to numerical inaccuracies, Dd can occasionally become indefinite, 
        # at which point we modify the diagonal elements as little as possible
        # but enough to make it positive definite
        eigs = eigen(Dd).values
        @warn "Had to modify Dd with $(-eigs[1])"
        # eigs[1] is the lowest (in this case negative) eigenvale. By subtracting it, 
        # in theory the matrix should become positive SEMIDEFINITE, but because of numerical
        # inaccuracies it has so far become positive definite when I have tested. If it ever
        # throws another PosDefException, it might be worth subtracting twice as much
        try
            cholesky(Dd - eigs[1]*I).L
        catch PosDefException
            @warn "Actually had to increase by even 1e-20"
            cholesky(Dd + (1e-20)*I).L
        end
    end

    C_mat = zeros((nη+1)nw, (1+na)*n_tot)
    # Because C-matrix only depends on C-parameters and the disturbance state xw depends only on a-parameters, we have
    # C_mat = [C    0
    #          0   Ina⊗C
    #          Cηc  0]
    C_mat[1:nw, 1:n_tot] = mdl.C
    C_mat[nw+1:(1+na)*nw, n_tot+1:end] = kron(Matrix(1.0I, na, na), mdl.C)
    for ind = 1:nη-na
        ηind = na + ind
        # We want to pass the index of the currently considered c-parameter in the C-matrix.
        # sens_inds contains the index of that parameter in η, which contains the additional na
        # parameters corresponding to the A-matrix
        row, col = get_C_row_and_col(sens_inds[ηind]-na, n_tot)  # row and col of the currently considered parameter in mdl.C
        C_mat[(na+ind)nw + row, col] = 1.0
    end

    return DT_SS_Model(Ad, Bd, C_mat, zeros((1+na)n_tot), Ts)
end

# Discretizes nominal disturbance model and provides matrices necessary for adjoint method where the disturbance model is 
# approximated by an ODE. Corresponds to Proposition 5.6 in my Licentiate thesis.
# Assumes that B-matrix is not parametrized, i.e. the version of Proposition 5.6 that uses Corollary 5.1
function discretize_ct_noise_model_with_adj_SDEApprox_mats(
    mdl::CT_SS_Model, Ts::Float64, sens_inds::Vector{Int64})::Tuple{DT_SS_Model, Matrix{Float64}, Matrix{Float64}, Matrix{Float64}, Matrix{Float64}}
    # sens_inds: indices of parameter with respect to which we compute the
    # sensitivity of disturbance output w

    @assert (length(sens_inds) > 0) "Make sure at least one disturbance parameter is marked for identification. Can't create model for sensitivity with respect to no parameters."

    nv = size(mdl.B, 2)
    nw = size(mdl.C, 1)
    n_tot = size(mdl.A, 1)
    nx = n_tot÷nv

    # Indices of free parameters corresponding to "a-vector" in disturbance model
    sens_inds_a = sens_inds[findall(sens_inds .<= nx)]
    # sens_inds_c = sens_inds[findall(sens_inds .> nx)]
    nη   = length(sens_inds)
    na = length(sens_inds_a)

    Ǎηa = zeros(na*n_tot, n_tot)
    for i = 1:na
        Ǎηa[(i-1)*n_tot+1, sens_inds_a[i]] = -1.0
    end

    M = [zeros(n_tot, (na+1)*n_tot)      Matrix(I, n_tot, n_tot)      zeros(n_tot, na*n_tot)                zeros(n_tot, n_tot)
         zeros(na*n_tot, (na+1)*n_tot)    zeros(na*n_tot, n_tot)    Matrix(I, na*n_tot, na*n_tot)            zeros(na*n_tot, n_tot)
         zeros(n_tot, (na+1)*n_tot)           mdl.A                 zeros(n_tot, na*n_tot)                mdl.B*(mdl.B');
         zeros(na*n_tot, (na+1)*n_tot)         Ǎηa                  kron(Matrix(1.0I, na, na), mdl.A)   zeros(n_tot*na, n_tot);
         zeros(n_tot, (na+1)*n_tot)          zeros(n_tot, n_tot)        zeros(n_tot, n_tot*na)                -mdl.A' ]

    Mexp = exp(M*Ts)
    Ad   = Mexp[(na+1)*n_tot + 1:(na+2)*n_tot, (na+1)*n_tot + 1:(na+2)*n_tot]
    Dd   = Hermitian(Mexp[(na+1)*n_tot + 1:(na+2)*n_tot, (2na+2)*n_tot + 1:(2na+3)*n_tot]*(Ad'))
    Bd = try
        cholesky(Dd).L
    catch PosDefException
        # Due to numerical inaccuracies, Dd can occasionally become indefinite, 
        # at which point we modify the diagonal elements as little as possible
        # but enough to make it positive definite
        eigs = eigen(Dd).values
        @warn "Had to modify Dd with $(-eigs[1])"
        # eigs[1] is the lowest (in this case negative) eigenvale. By subtracting it, 
        # in theory the matrix should become positive SEMIDEFINITE, but because of numerical
        # inaccuracies it has so far become positive definite when I have tested. If it ever
        # throws another PosDefException, it might be worth subtracting twice as much
        try
            cholesky(Dd - eigs[1]*I).L
        catch PosDefException
            @warn "Actually had to increase by even 1e-20"
            cholesky(Dd + (1e-20)*I).L
        end
    end

    Bdηa = zeros(na*n_tot, n_tot)
    for i = 1:na
        # H = Mexp[(na+i+1)*nx + 1:(na+i+2)*nx, (2na+2)*nx + 1:(2na+3)*nx]
        Ddηai = Mexp[(na+i+1)*n_tot + 1:(na+i+2)*n_tot, (2na+2)*n_tot + 1:(2na+3)*n_tot]*(Ad')
        Ddηai += Ddηai'
        Bdηa[(i-1)*n_tot+1:i*n_tot,:] = Bd*Phi((Bd\Ddηai)/(Bd'), n_tot)
    end

    # Matrices needed for adjoint disturbance estimation
    P = Mexp[1:n_tot, (na+1)*n_tot + 1:(na+2)*n_tot]
    R = Mexp[n_tot+1:(na+1)*n_tot, (na+1)*n_tot + 1: (na+2)*n_tot]
    B̌ηa = kron(Matrix(I,na,na), P) \ (Bdηa - (R/P)*Bd)
    # B̌  = P\Bd

    Čη = zeros(nη*nw, n_tot)
    # Čη = [Čηa; Čηc] = [0; Čηc]
    for ηind = na+1:nη
        # We want to pass the index of the currently considered c-parameter in the C-matrix to get_C_row_and_col
        # sens_inds contains the index of that parameter in η, which contains the additional na
        # parameters corresponding to the A-matrix
        row, col = get_C_row_and_col(sens_inds[ηind]-na, n_tot)  # row and col of the currently considered parameter in mdl.C
        Čη[(ηind-1)nw + row, col] = 1.0
    end

    Ǎη = vcat(Ǎηa, zeros((nη-na)n_tot, n_tot))
    B̌η = vcat(B̌ηa, zeros((nη-na)n_tot, n_tot))

    # Returns non-sensitivity disturbance model and other matrices needed for adjoint disturbance sensitivity
    return DT_SS_Model(Ad, Bd, mdl.C, zeros(n_tot), Ts), Ǎη, B̌η, Čη, mdl.A
end

function discretize_ct_noise_model_with_adj_SDEApprox_mats_Ainvertible(
    mdl::CT_SS_Model, Ts::Float64, sens_inds::Vector{Int64})::Tuple{DT_SS_Model, Matrix{Float64}, Matrix{Float64}, Matrix{Float64}, Matrix{Float64}}
    # sens_inds: indices of parameter with respect to which we compute the
    # sensitivity of disturbance output w

    @assert (length(sens_inds) > 0) "Make sure at least one disturbance parameter is marked for identification. Can't create model for sensitivity with respect to no parameters."

    nv = size(mdl.B, 2)
    nw = size(mdl.C, 1)
    n_tot = size(mdl.A, 1)
    nx = n_tot÷nv

    # Indices of free parameters corresponding to "a-vector" in disturbance model
    sens_inds_a = sens_inds[findall(sens_inds .<= nx)]
    # sens_inds_c = sens_inds[findall(sens_inds .> nx)]
    nη   = length(sens_inds)
    na = length(sens_inds_a)
    # nx_sens = (1+na)*nx

    Aηa = zeros(na*n_tot, n_tot)
    for i = 1:na
        Aηa[(i-1)*n_tot+1, sens_inds_a[i]] = -1.0
    end

    Mexp, B̃d, B̃dηa = get_disc_then_diff_matrices(mdl, Ts, sens_inds)
    Ãd = Mexp[1:n_tot, 1:n_tot]
    Ãdηa = Mexp[n_tot+1:n_tot*(1+na), 1:n_tot]
    M = (Ãd - Matrix(1.0I, n_tot, n_tot))\B̃d
    B̌ηa = Aηa*M + kron(Matrix(I,na,na), mdl.A/(Ãd-Matrix(I,n_tot,n_tot)))*(B̃dηa-Ãdηa*M)

    Čη = zeros(nη*nw, n_tot)
    # Čη = [Čηa; Čηc] = [0; Čηc]
    for ηind = na+1:nη
        # We want to pass the index of the currently considered c-parameter in the C-matrix to get_C_row_and_col
        # sens_inds contains the index of that parameter in η, which contains the additional na
        # parameters corresponding to the A-matrix
        row, col = get_C_row_and_col(sens_inds[ηind]-na, n_tot)  # row and col of the currently considered parameter in mdl.C
        Čη[(ηind-1)nw + row, col] = 1.0
    end

    Ǎη = vcat(Aηa, zeros((nη-na)n_tot, n_tot))
    B̌η = vcat(B̌ηa, zeros((nη-na)n_tot, n_tot))

    # Returns non-sensitivity disturbance model and other matrices needed for adjoint disturbance sensitivity
    return DT_SS_Model(Ãd, B̃d, mdl.C, zeros(n_tot), Ts), Ǎη, B̌η, Čη, mdl.A
end

# ================= Functions simulating disturbance =======================

# TODO: These function do not at all utilize the structure of our model structure
# Write functions that actually use that structure optimally, ideally also parallelize
# simulation of the subsystems

function simulate_noise_process_mangled(mdl::DT_SS_Model, data::Vector{Matrix{Float64}})::Matrix{Float64}
    # data[m][i, j] should be the j:th component of the noise
    # corresponding to time i of realization m
    M = length(data)
    N = size(data[1], 2)-1
    nA = size(mdl.Ad, 1)
    # We only care about the state, not the output, so we ignore the C-matrix
    C_placeholder = zeros(1,nA)

    sys = ss(mdl.Ad, mdl.Bd, C_placeholder, 0.0, mdl.Ts)
    t = 0:mdl.Ts:N*mdl.Ts
    # Allocating space for noise process
    x_process = fill(NaN, ((N+1)*nA, M))

    for m=1:M
        _, t, x = lsim(sys, data[m], t, x0=mdl.x0)
        for i=1:N+1
            # x_process = [xvec1; xvec2; ...; xvecN+1]          # N+1 blocks of size nA
            x_process[(i-1)nA+1 : i*nA, m] = x[:,i]
        end
    end

    # x_process[(i-1)*n_tot + j, m] is the j:th element of the noise model at
    # sample i of realization m. Sample 1 corresponds to time 0
    return x_process
end

function simulate_noise_process(mdl::DT_SS_Model, data::Vector{Matrix{Float64}})::Matrix{Vector{Float64}}
    # data[m][i, j] should be the j:th component of the noise
    # corresponding to time i of realization m
    M = length(data)
    N = size(data[1], 2)-1
    nA = size(mdl.Ad, 1)
    # We only care about the state, not the output, so we ignore the C-matrix
    C_placeholder = zeros(1, nA)

    sys = ss(mdl.Ad, mdl.Bd, C_placeholder, 0.0, mdl.Ts)
    t = 0:mdl.Ts:N*mdl.Ts
    # Allocating space for noise process
    x_process = [fill(NaN, (nA,)) for i=1:N+1, m=1:M]
    for m=1:M
        _, t, x = lsim(sys, data[m], t, x0=mdl.x0)
        for i=1:N+1
            x_process[i,m] = x[:,i]
        end
    end

    # x_process[i,m][j] is the j:th element of the noise model state at sample
    # i of realization m. Sample 1 corresponds to time 0
    return x_process
end

# ============== Functions for generating specific realization ===============

function get_ct_disturbance_model(η::Vector{Float64}, nx::Int, nv::Int)
    # First nx parameters of η are parameters for A-matrix, the remaining
    # parameters are for the C-matrix
    n_tot = nx*nv
    A = diagm(-1 => ones(n_tot-1,))
    B = zeros(n_tot,nv)
    for ind=1:nv
        # A here is actually what I in my licentiate thesis call Φ, and
        # Φ = diag(A,...,A) (nv blocks) where
        # A = [-η[1] -η[2] ⋯ -η[nx]
        #         1     0  ⋯   0
        #         ⋮      ⋮  ⋱    ⋮
        #         0     0  ⋯    0] (i.e. A how I define it in thesis, not here)
        A[(ind-1)nx+1,(ind-1)nx+1:ind*nx] = -η[1:nx]
        # B here is actually what I in my licentiate thesis call Γ, and 
        # Γ = diag(B,...,B) (nv blocks) where
        # B = [1; 0; ...; 0] (nx × 1) (i.e. B how I define it in thesis, not here)
        B[(ind-1)nx+1, ind] = 1.0
    end
    C = reshape(η[nx+1:end], :, n_tot)
    x0 = zeros(n_tot)
    return CT_SS_Model(A, B, C, x0)
    #= With the dimensions we most commonly use, the model becomes
    A = [-a1 -a2
            1   0 ]
    B = [1; 0]
    C = [c1 c2]
    =#
end

# Parameter values for C-matrix as entered are converted into a single vector
# more suitable to be used in the code
function get_c_parameter_vector(c_vals, w_scale, nx::Int, nw::Int, nv::Int)
    c_vec = zeros(nx*nw*nv)
    for i = 1:nw
        for j = 1:nv
            for k = 1:nx
                # Multiplying with w_scale[i] here is equivalent to scaling ith
                # component of the disturbance output (w) with factor w_scale[i]
                c_vec[(j-1)*nw*nx + (k-1)*nw + i] = w_scale[i]*c_vals[i,j][k]
            end
        end
    end
    return c_vec
end

# Used for disturbance
function disturbance_model_1(Ts::Float64; scale::Float64=0.6)::Tuple{DT_SS_Model, Vector{Int}, Vector{Float64}}
    nx = 2        # model order
    nw = 2     # number of outputs
    nv = 2      # number of inputs
    w_scale = scale*ones(nw)             # noise scale
    # Denominator of every transfer function is given by p(s), where
    # p(s) = s^n + a[1]*s^(n-1) + ... + a[n-1]*s + a[n]
    a_vec = [0.8, 4^2]
    # Transfer function (i,j) has numerator c_[i,j][1]s^{nx-1} + ... + c_[i,j][nx]
    c_ = [zeros(nx) for i=1:nw, j=1:nv]
    c_[1,1][nx] = 1 # c_[1,1][:] = vcat(zeros(nx-1), [1])
    c_[2,2][nx] = 1 # c_[2,2][:] = vcat(zeros(nx-1), [1])
    c_vec = get_c_parameter_vector(c_, w_scale, nx, nw, nv)
    η0 = vcat(a_vec, c_vec)
    dη = length(η0)
    mdl =
        discretize_ct_noise_model(get_ct_disturbance_model(η0, nx, nv), Ts)
    # return mdl, DataFrame(nx = nx, nv = nv, nw = nw, η = η0, bias=bias)
    return mdl, [nx, nv, nw], η0
end

# Used for input
function disturbance_model_2(Ts::Float64; scale::Float64=0.2)::Tuple{DT_SS_Model, Vector{Int}, Vector{Float64}}
    nx = 2        # model order
    nw = 1     # number of outputs
    nv = 2      # number of inputs
    u_scale = scale # input scale
    w_scale = scale*ones(nw)             # noise scale
    # Denominator of every transfer function is given by p(s), where
    # p(s) = s^n + a[1]*s^(n-1) + ... + a[n-1]*s + a[n]
    a_vec = [0.8, 4^2]
    c_vec = zeros(nw*nx*nv)
    # The first state will act as output of the filter
    c_vec[1] = u_scale
    η0 = vcat(a_vec, Diagonal(w_scale)*c_vec)
    dη = length(η0)
    mdl =
        discretize_ct_noise_model(get_ct_disturbance_model(η0, nx, nv), Ts)
    # return mdl, DataFrame(nx = nx, nv = nv, nw = nw, η = η0, bias=bias)
    return mdl, [nx, nv, nw], η0
end

# Used for scalar disturbance and input
function disturbance_model_3(Ts::Float64; scale::Float64=1.0)::Tuple{DT_SS_Model, Vector{Int}, Vector{Float64}}
    ω = 4         # natural freq. in rad/s (tunes freq. contents/fluctuations)
    ζ = 0.1       # damping coefficient (tunes damping)
    nx = 2        # model order
    nw = 1     # number of outputs
    nv = 1      # number of inputs
    # Denominator of every transfer function is given by p(s), where
    # p(s) = s^n + a[1]*s^(n-1) + ... + a[n-1]*s + a[n]
    a_vec = [2*ω*ζ, ω^2]
    c_vec = [0, scale]
    η0 = vcat(a_vec, c_vec)
    dη = length(η0)
    mdl =
        discretize_ct_noise_model(get_ct_disturbance_model(η0, nx, nv), Ts)
    # return mdl, DataFrame(nx = nx, nv = nv, nw = nw, η = η0, bias=bias)
    return mdl, [nx, nv, nw], η0
end

# Used for multivariate input for delta-robot
function disturbance_model_4(Ts::Float64; scale::Float64=1.0)::Tuple{DT_SS_Model, Vector{Int}, Vector{Float64}}
    ω = 4         # natural freq. in rad/s (tunes freq. contents/fluctuations)
    ζ = 0.1       # damping coefficient (tunes damping)
    p3 = -2       # The additional pole that is added
    nx = 3        # model order
    nw = 3     # number of outputs
    nv = 1      # number of inputs
    w_scale = scale*ones(nw)             # noise scale
    # Denominator of every transfer function is given by p(s), where
    # p(s) = s^n + a[1]*s^(n-1) + ... + a[n-1]*s + a[n]
    # Old a_vec, i.e. for the 2d-model: a_vec = [2*ω*ζ, ω^2]
    a_vec = [2*ω*ζ-p3, ω^2-p3*2*ω*ζ, -p3*ω^2]
    c_vec = [scale, 0.0, 0.0, 0.0, scale, 0.0, 0.0, 0.0, scale]
    η0 = vcat(a_vec, c_vec)
    dη = length(η0)
    mdl =
        discretize_ct_noise_model(get_ct_disturbance_model(η0, nx, nv), Ts)
    # return mdl, DataFrame(nx = nx, nv = nv, nw = nw, η = η0, bias=bias)
    return mdl, [nx, nv, nw], η0
end

# Used for new multivariate input for delta-robot
function disturbance_model_5(Ts::Float64; scale::Float64=1.0)::Tuple{DT_SS_Model, Vector{Int}, Vector{Float64}}
    ω = 4         # natural freq. in rad/s (tunes freq. contents/fluctuations)
    ζ = 0.1       # damping coefficient (tunes damping)
    p3 = -2       # The additional pole that is added
    nx = 3        # model order
    nw = 3     # number of outputs
    nv = 3      # number of inputs
    # Denominator of every transfer function is given by p(s), where
    # p(s) = s^n + a[1]*s^(n-1) + ... + a[n-1]*s + a[n]
    # Old a_vec, i.e. for the 2d-model: a_vec = [2*ω*ζ, ω^2]
    a_vec = [2*ω*ζ-p3, ω^2-p3*2*ω*ζ, -p3*ω^2]
    c_vec = zeros(3*9)
    c_vec[1] = scale; c_vec[9+4] = scale; c_vec[18+7] = scale
    η0 = vcat(a_vec, c_vec)
    mdl =
        discretize_ct_noise_model(get_ct_disturbance_model(η0, nx, nw), Ts)
    # return mdl, DataFrame(nx = nx, nv = nv, nw = nw, η = η0, bias=bias)
    return mdl, [nx, nv, nw], η0
end

function get_multisine(num::Int; min_amp::Float64=1.0, max_amp::Float64=10.0, min_freq::Float64=1.0, max_freq::Float64=50.0)
    amps = rand(min_amp:0.01:max_amp, num)
    freqs = rand(min_freq:0.1:max_freq, num)
    phases = [-k*(k-1)*pi/num for k=1:num]  # Schröder phases
    t -> sum(
            amps.*(sin.(freqs*t+phases))
        )
end

function get_filtered_noise(gen::Function, Ts::Float64, M::Int, Nw::Int;
    bias::Float64=0.0, scale::Float64=1.0)::Tuple{Matrix{Float64}, Matrix{Float64}, DataFrame}

    mdl, meta_raw, η0 = gen(Ts, scale=scale)
    metadata = DataFrame(nx = meta_raw[1], nv = meta_raw[2], nw = meta_raw[3], η = η0, bias=bias, num_rel = M, Nw=Nw, δ = Ts)
    n_tot = size(mdl.Cd,2)

    # We use Nw+1, since we want samples at t₀, t₁, ..., t_{N_w}, i.e. a total of N_w+1 samples, 
    ZS = [randn(Nw+1, n_tot) for m = 1:M]
    XW = simulate_noise_process_mangled(mdl, ZS, meta_raw[2])
    XW, get_system_output_mangled(mdl, XW).+ bias, metadata
end

# Converts mangled states to an output vector using the provided model
function get_system_output_mangled(mdl::DT_SS_Model, states::Matrix{Float64}
    )::Matrix{Float64}
    M = size(states, 2)
    (nw, n_tot) = size(mdl.Cd)
    N = size(states, 1)÷n_tot
    output = zeros(N*nw, M)
    for m=1:M
        for t=1:N
            output[(t-1)*nw+1:t*nw, m] = mdl.Cd*states[(t-1)*n_tot+1:t*n_tot, m]
        end
    end
    return output
end

# ====================== Other functions ==========================

# DEBUG For checking if the simulated noise process seems to have the expected
# statistical properties
function test_generated_data(
    mdl::DT_SS_Model,
    x_process::Vector{Matrix{Float64}}
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

end