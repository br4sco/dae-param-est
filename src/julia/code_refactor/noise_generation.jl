module NoiseGeneration

# To test which of these are necessary, consider replacing "using" with "import", then one has to write e.g. Random.seed!() to call seed!()
import Random
using DataFrames: DataFrame
using LinearAlgebra: Diagonal, diagm, Hermitian, cholesky, I, LowerTriangular, eigen
using ControlSystems: ss, lsim
# import Statistics, CSV, DataFrames, ControlSystems, LinearAlgebra, Random

export DisturbanceMetaData, demangle_XW, get_filtered_noise, disturbance_model_1, disturbance_model_2, disturbance_model_3, get_ct_disturbance_model
export discretize_ct_noise_model_disc_then_diff, simulate_noise_process_mangled, discretize_ct_noise_model_with_sensitivities_for_adj

seed = 54321    # Important that random samples generated here are independent of those generated in run_experiment.jl
Random.seed!(seed)
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
    n_in::Int   # n_tot = nx*n_in
    n_out::Int  # TODO: Consider renaming n_in -> nv and n_out = nw, to match better my thesis and other work?
    η::Vector{Float64}
    free_par_inds::Vector{Int}
    # Array containing lower and upper bound of a disturbance parameter in each row
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

function Φ(mat_in::Array{Float64,2})
    nx = minimum(size(mat_in))
    Φ(mat_in, nx)
end

function Φ(mat_in::Array{Float64,2}, nx::Int)
    mat = copy(mat_in)
    for i=1:size(mat,1)
        for j=1:size(mat,2)
            i1 = (i-1)%nx+1
            j1 = (j-1)%nx+1
            if i1 > j1
                continue
            elseif i1 == j1
                mat[i,j] = 0.5*mat[i,j]
            else
                mat[i,j] = 0
            end
        end
    end
    return mat
end

# TODO: Don't need all these matrix functions Phi, Φ... Figure out which ones you need and maybe rename!
function Phi(mat_in::Array{Float64,2}, n_tot::Int)::LowerTriangular
    mat = LowerTriangular(mat_in)
    for i=1:n_tot
        mat[i,i] *= 0.5
    end
    return mat
    # mat = zeros(size(mat_in))
    # for i=1:nx
    #     for j=1:nx
    #         if i==j
    #             mat[i,j] = 0.5*mat_in[i,j]
    #         elseif i>j
    #             mat[i,j] = mat_in[i,j]
    #         end
    #     end
    # end
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

    n_in = size(mdl.B, 2)
    n_tot = size(mdl.A, 1)
    nx = n_tot÷n_in
    # Indices of free parameters corresponding to "a-vector" in vector of all disturbance parameters
    sens_inds_a = sens_inds[findall(sens_inds .<= nx)]
    na = length(sens_inds_a)

    Aηa = zeros(na*n_tot, n_tot)
    for ind1 = 1:na
        for ind2 = 1:n_in
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

    n_in = size(mdl.B, 2)
    n_out = size(mdl.C, 1)
    n_tot = size(mdl.A, 1)
    nx = n_tot÷n_in
    nη = length(sens_inds)
    na = length(findall(sens_inds .<= nx))

    A_mat = Mexp[1:(na+1)n_tot, 1:(na+1)n_tot]
    B_mat = [B̃d; B̃dηa]
    C_mat = zeros((nη+1)n_out, (1+na)*n_tot)
    # Because C-matrix only depends on C-parameters and the disturbance state xw depends only on a-parameters, we have
    # C_mat = [C    0
    #          0   Ina⊗C
    #          Cηc  0]
    C_mat[1:n_out, 1:n_tot] = mdl.C
    C_mat[n_out+1:(1+na)*n_out, n_tot+1:end] = kron(Matrix(1.0I, na, na), mdl.C)
    # for ind = 1:nη-na
    for ηind = na+1:nη
        # ηind = na + ind # TODO: Delete
        # We want to pass the index of the currently considered c-parameter in the C-matrix.
        # sens_inds contains the index of that parameter in η, which contains the additional na
        # parameters corresponding to the A-matrix
        row, col = get_C_row_and_col(sens_inds[ηind]-na, n_tot)  # row and col of the currently considered parameter in mdl.C
        C_mat[ηind*n_out + row, col] = 1.0
    end

    return DT_SS_Model(A_mat, B_mat, C_mat, zeros((1+na)n_tot), Ts)
end

# Differentiates ct model before discretization, corresponds to Proposition 5.1 in my Licentiate thesis. 
# Assumes that B is not parametrized, but this does not really simplify the function, just makes some elements zero.
function discretize_ct_noise_model_diff_then_disc( mdl::CT_SS_Model, Ts::Float64, sens_inds::Array{Int64, 1})::DT_SS_Model
    # sens_inds: indices of parameter with respect to which we compute the
    # sensitivity of disturbance output w

    n_in = size(mdl.B, 2)
    n_out = size(mdl.C, 1)
    n_tot = size(mdl.A, 1)
    nx = n_tot÷n_in
    # Indices of free parameters corresponding to "a-vector" in vector of all disturbance parameters
    sens_inds_a = sens_inds[findall(sens_inds .<= nx)]
    nη   = length(sens_inds)
    na = length(sens_inds_a)

    Aηa = zeros(na*n_tot, n_tot)
    for ind1 = 1:na
        for ind2 = 1:n_in
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

    C_mat = zeros((nη+1)n_out, (1+na)*n_tot)
    # Because C-matrix only depends on C-parameters and the disturbance state xw depends only on a-parameters, we have
    # C_mat = [C    0
    #          0   Ina⊗C
    #          Cηc  0]
    C_mat[1:n_out, 1:n_tot] = mdl.C
    C_mat[n_out+1:(1+na)*n_out, n_tot+1:end] = kron(Matrix(1.0I, na, na), mdl.C)
    for ind = 1:nη-na
        ηind = na + ind
        # We want to pass the index of the currently considered c-parameter in the C-matrix.
        # sens_inds contains the index of that parameter in η, which contains the additional na
        # parameters corresponding to the A-matrix
        row, col = get_C_row_and_col(sens_inds[ηind]-na, n_tot)  # row and col of the currently considered parameter in mdl.C
        C_mat[(na+ind)n_out + row, col] = 1.0
    end

    return DT_SS_Model(Ad, Bd, C_mat, zeros((1+na)n_tot), Ts)
end

# Discretizes nominal disturbance model and provides matrices necessary for adjoint method where the disturbance model is 
# approximated by an ODE. Corresponds to Proposition 5.6 in my Licentiate thesis.
# Assumes that B-matrix is not parametrized, i.e. the version of Proposition 5.6 that uses Corollary 5.1
function discretize_ct_noise_model_with_adj_SDEApprox_mats(
    mdl::CT_SS_Model, Ts::Float64, sens_inds::Array{Int64, 1})::Tuple{DT_SS_Model, Matrix{Float64}, Matrix{Float64}, Matrix{Float64}, Matrix{Float64}}
    # sens_inds: indices of parameter with respect to which we compute the
    # sensitivity of disturbance output w

    @assert (length(sens_inds) > 0) "Make sure at least one disturbance parameter is marked for identification. Can't create model for sensitivity with respect to no parameters."

    n_in = size(mdl.B, 2)
    n_out = size(mdl.C, 1)
    n_tot = size(mdl.A, 1)
    nx = n_tot÷n_in

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

    Čη = zeros(nη*n_out, n_tot)
    # Čη = [Čηa; Čηc] = [0; Čηc]
    for ηind = na+1:nη
        # We want to pass the index of the currently considered c-parameter in the C-matrix to get_C_row_and_col
        # sens_inds contains the index of that parameter in η, which contains the additional na
        # parameters corresponding to the A-matrix
        row, col = get_C_row_and_col(sens_inds[ηind]-na, n_tot)  # row and col of the currently considered parameter in mdl.C
        Čη[(ηind-1)n_out + row, col] = 1.0
    end

    Ǎη = vcat(Ǎηa, zeros((nη-na)n_tot, n_tot))
    B̌η = vcat(B̌ηa, zeros((nη-na)n_tot, n_tot))

    # Returns non-sensitivity disturbance model and other matrices needed for adjoint disturbance sensitivity
    return DT_SS_Model(Ad, Bd, mdl.C, zeros(n_tot), Ts), Ǎη, B̌η, Čη, mdl.A
end

# TODO: Might use nx instead of n_tot in some places! Also, this function in particular might not be finished
function discretize_ct_noise_model_with_adj_SDEApprox_mats_Ainvertible(
    mdl::CT_SS_Model, Ts::Float64, sens_inds::Array{Int64, 1})::Tuple{DT_SS_Model, Matrix{Float64}, Matrix{Float64}, Matrix{Float64}}
    # sens_inds: indices of parameter with respect to which we compute the
    # sensitivity of disturbance output w

    @assert (length(sens_inds) > 0) "Make sure at least one disturbance parameter is marked for identification. Can't create model for sensitivity with respect to no parameters."

    nx = size(mdl.A,1)
    n_out = size(mdl.C, 1)
    n_in  = size(mdl.C, 2)÷nx

    # Indices of free parameters corresponding to "a-vector" in disturbance model
    sens_inds_a = sens_inds[findall(sens_inds .<= nx)]
    # sens_inds_c = sens_inds[findall(sens_inds .> nx)]
    nη   = length(sens_inds)
    na = length(sens_inds_a)
    # nx_sens = (1+na)*nx

    Aηa = zeros(na*nx, nx)
    for i = 1:na
        Aηa[(i-1)*nx+1, sens_inds_a[i]] = -1
    end

    # NEW TODO: MAY BE TEST THIS!
    Mexp, B̃d, B̃dηa = get_disc_then_diff_matrices(mdl, Ts, sens_inds)
    Ãdηa = Mexp[nx+1:nx(1+na), nx+1:nx(1+na)]   # TODO: Maybe double-check? I wrote this while tired
    M = (Ãdηa - Matrix(1.0I, nx*na, nx*na))\B̃d
    B̌ηa = Aηa*M + kron(Matrix(I,na,na), A/(Ãd-Matrix(I,nx,nx)))*(B̃dηa-Ãdηa*M)

    # Ǎη = vcat(Aηa, zeros(nη-na, nx))
    # B̌η = vcat(B̌ηa, zeros(nη-na, nx))
    Čη = zeros(n_out*nη, n_in*nx)
    for ind=na+1:nη
        cind = sens_inds[ind]
        tmp = Int(ind÷(nx*n_in))
        row = na + tmp + 1
        col = cind - tmp*nx*n_in
        Čη[row,col] = 1.0
    end

    # Returns non-sensitivity disturbance model and other matrices needed for adjoint disturbance sensitivity
    return DT_SS_Model(Ad, Bd, mdl.C, zeros(nx*n_in), Ts), Aηa, B̌ηa, Čη
end

# ================= Functions simulating disturbance =======================

function simulate_noise_process(mdl::DT_SS_Model, data::Vector{Matrix{Float64}})::Matrix{Vector{Float64}}
    # data[m][i, k*nx + j] should be the j:th component of the noise
    # corresponding to input k at time i of realization m
    M = length(data)
    N = size(data[1], 1)-1
    nx = size(mdl.Ad, 1)
    # When sensitivities are included in disturbance model, nv != nx
    nv = size(mdl.Bd, 2)
    n_in = size(mdl.Cd, 2)÷nx
    # We only care about the state, not the output, so we ignore the C-matrix
    C_placeholder = zeros(1, nx)

    sys = ss(mdl.Ad, mdl.Bd, C_placeholder, 0.0, mdl.Ts)
    t = 0:mdl.Ts:N*mdl.Ts
    # Allocating space for noise process
    x_process = [ fill(NaN, (nx*n_in,)) for i=1:N+1, m=1:M]
    for ind = 1:n_in
        for m=1:M
            _, t, x = lsim(sys, data[m][:, (ind-1)*nv+1:ind*nv]', t, x0=mdl.x0)
            for i=1:N+1
                x_process[i,m][(ind-1)*nx+1:ind*nx] = x[:,i]
            end
        end
    end

    # x_process[i,m][j] is the j:th element of the noise model state at sample
    # i of realization m. Sample 1 corresponds to time 0
    return x_process
end

function simulate_noise_process_mangled(mdl::DT_SS_Model, data::Vector{Matrix{Float64}})::Matrix{Float64}
    # data[m][i, k*nx + j] should be the j:th component of the noise
    # corresponding to input k at time i of realization m
    M = length(data)
    N = size(data[1], 1)-1
    nx = size(mdl.Ad, 1)
    # When sensitivities are included in disturbance model, nv != nx
    nv = size(mdl.Bd, 2)
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
            _, t, x = lsim(sys, data[m][:, (ind-1)*nv+1:ind*nv]', t, x0=mdl.x0)
            for i=1:N+1
                x_process[(i-1)*n_in*nx + (ind-1)*nx + 1: (i-1)*n_in*nx + ind*nx, m] = x[:,i]
            end
        end
    end

    # x_process[(i-1)*nx*n_in + j, m] is the j:th element of the noise model at
    # sample i of realization m. Sample 1 corresponds to time 0
    return x_process
end

# ============== Functions for generating specific realization ===============

function get_ct_disturbance_model(η::Array{Float64,1}, nx::Int, n_in::Int)
    # First nx parameters of η are parameters for A-matrix, the remaining
    # parameters are for the C-matrix
    n_tot = nx*n_in
    A = diagm(-1 => ones(n_tot-1,))
    A[1,:] = -η[1:n_tot]
    B = zeros(n_tot,1)
    B[1] = 1.0
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

# Used for disturbance
# function disturbance_model_1(Ts::Float64; bias::Float64=0.0, scale::Float64=0.6)::Tuple{DT_SS_Model, DataFrame}
function disturbance_model_1(Ts::Float64; scale::Float64=0.6)::Tuple{DT_SS_Model, Vector{Int}, Vector{Float64}}
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
        discretize_ct_noise_model(get_ct_disturbance_model(η0, nx, n_in), Ts)
    # return mdl, DataFrame(nx = nx, n_in = n_in, n_out = n_out, η = η0, bias=bias)
    return mdl, [nx, n_in, n_out], η0
end

# Used for input
# function disturbance_model_2(Ts::Float64; bias::Float64=0.0, scale::Float64=0.2)::Tuple{DT_SS_Model, DataFrame}
function disturbance_model_2(Ts::Float64; scale::Float64=0.2)::Tuple{DT_SS_Model, Vector{Int}, Vector{Float64}}
    nx = 2        # model order
    n_out = 1     # number of outputs
    n_in = 2      # number of inputs
    u_scale = scale # input scale
    w_scale = scale*ones(n_out)             # noise scale
    # Denominator of every transfer function is given by p(s), where
    # p(s) = s^n + a[1]*s^(n-1) + ... + a[n-1]*s + a[n]
    a_vec = [0.8, 4^2]
    c_vec = zeros(n_out*nx*n_in)
    # The first state will act as output of the filter
    c_vec[1] = u_scale
    η0 = vcat(a_vec, Diagonal(w_scale)*c_vec)
    dη = length(η0)
    mdl =
        discretize_ct_noise_model(get_ct_disturbance_model(η0, nx, n_in), Ts)
    # return mdl, DataFrame(nx = nx, n_in = n_in, n_out = n_out, η = η0, bias=bias)
    return mdl, [nx, n_in, n_out], η0
end

# Used for scalar disturbance and input
# function disturbance_model_3(Ts::Float64; bias::Float64=0.0, scale::Float64=1.0)::Tuple{DT_SS_Model, DataFrame}
function disturbance_model_3(Ts::Float64; scale::Float64=1.0)::Tuple{DT_SS_Model, Vector{Int}, Vector{Float64}}
    ω = 4         # natural freq. in rad/s (tunes freq. contents/fluctuations)
    ζ = 0.1       # damping coefficient (tunes damping)
    nx = 2        # model order
    n_out = 1     # number of outputs
    n_in = 1      # number of inputs
    # Denominator of every transfer function is given by p(s), where
    # p(s) = s^n + a[1]*s^(n-1) + ... + a[n-1]*s + a[n]
    a_vec = [2*ω*ζ, ω^2]
    c_vec = [0, scale]
    η0 = vcat(a_vec, c_vec)
    dη = length(η0)
    mdl =
        discretize_ct_noise_model(get_ct_disturbance_model(η0, nx, n_in), Ts)
    # return mdl, DataFrame(nx = nx, n_in = n_in, n_out = n_out, η = η0, bias=bias)
    return mdl, [nx, n_in, n_out], η0
end

# Used for multivariate input for delta-robot
# function disturbance_model_4(Ts::Float64; bias::Float64=0.0, scale::Float64=1.0)::Tuple{DT_SS_Model, DataFrame}
function disturbance_model_4(Ts::Float64; scale::Float64=1.0)::Tuple{DT_SS_Model, Vector{Int}, Vector{Float64}}
    ω = 4         # natural freq. in rad/s (tunes freq. contents/fluctuations)
    ζ = 0.1       # damping coefficient (tunes damping)
    p3 = -2       # The additional pole that is added
    nx = 3        # model order
    n_out = 3     # number of outputs
    n_in = 1      # number of inputs
    w_scale = scale*ones(n_out)             # noise scale
    # Denominator of every transfer function is given by p(s), where
    # p(s) = s^n + a[1]*s^(n-1) + ... + a[n-1]*s + a[n]
    # Old a_vec, i.e. for the 2d-model: a_vec = [2*ω*ζ, ω^2]
    a_vec = [2*ω*ζ-p3, ω^2-p3*2*ω*ζ, -p3*ω^2]
    c_vec = [scale, 0.0, 0.0, 0.0, scale, 0.0, 0.0, 0.0, scale]
    η0 = vcat(a_vec, c_vec)
    dη = length(η0)
    mdl =
        discretize_ct_noise_model(get_ct_disturbance_model(η0, nx, n_in), Ts)
    # return mdl, DataFrame(nx = nx, n_in = n_in, n_out = n_out, η = η0, bias=bias)
    return mdl, [nx, n_in, n_out], η0
end


function get_filtered_noise(gen::Function, Ts::Float64, M::Int, Nw::Int;
    bias::Float64=0.0, scale::Float64=1.0)::Tuple{Array{Float64,2}, Array{Float64,2}, DataFrame}

    mdl, meta_raw, η0 = gen(Ts, scale=scale)
    metadata = DataFrame(nx = meta_raw[1], n_in = meta_raw[2], n_out = meta_raw[3], η = η0, bias=bias, num_rel = M, Nw=Nw, δ = Ts)
    n_tot = size(mdl.Cd,2)

    # We use Nw+1, since we want samples at t₀, t₁, ..., t_{N_w}, i.e. a total of N_w+1 samples, 
    ZS = [randn(Nw+1, n_tot) for m = 1:M]
    XW = simulate_noise_process_mangled(mdl, ZS)
    XW, get_system_output_mangled(mdl, XW).+ bias, metadata
end

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

end