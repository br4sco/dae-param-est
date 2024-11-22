using Statistics, CSV, DataFrames, ControlSystems, LinearAlgebra

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
    n_in::Int
    n_out::Int
    η::Array{Float64,1}
    free_par_inds::Array{Int64,1}
    num_rels::Int
end

# =================== Helper Functions ==========================

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

function Phi(mat_in::Array{Float64,2}, nx::Int)::LowerTriangular
    mat = LowerTriangular(mat_in)
    for i=1:nx
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
    nη   = length(sens_inds)
    na = length(sens_inds_a)
    nx_sens = (1+na)*nx

    Aηa = zeros(na*nx, nx)
    for i = 1:na
        Aηa[(i-1)*nx+1, sens_inds_a[i]] = -1
    end

    M = [mdl.A             zeros(nx, na*nx)                mdl.B*(mdl.B');
         Aηa         kron(Matrix(1.0I, na, na), mdl.A)   zeros(nx*na, nx);
         zeros(nx, nx)     zeros(nx, nx*na)                -mdl.A' ]
    Mexp = exp(M*Ts)
    Ad   = Mexp[1:nx, 1:nx]
    Dd   = Hermitian(Mexp[1:nx, (na+1)*nx+1:(na+2)*nx]*(Ad'))
    Bd   = cholesky(Dd).L
    Adηa = Mexp[nx+1:(na+1)*nx, 1:nx]
    # temp = Mexp[nx+1:(nθ+1)*nx, (nθ+1)*nx+1:(nθ+2)*nx]*(Ad')
    Ddηa = Mexp[nx+1:(na+1)*nx, (na+1)*nx+1:(na+2)*nx]*(Ad')
    for i = 1:na
        Ddηa[(i-1)*nx+1:i*nx, :] += (Ddηa[(i-1)*nx+1:i*nx, :])'
    end
    Φ_arg = ((kron(Matrix(I, na, na), Bd)) \ Ddηa ) / (Bd')
    Bdηa = kron(Matrix(I, na, na), Bd)*Φ( Φ_arg )

    A_mat = [Ad zeros(nx,na*nx); Adηa kron(Matrix(I,na,na), Ad)]
    B_mat = [Bd; Bdηa]
    C_mat = zeros((nη+1)n_out, nx_sens*n_in)
    for row_block = 1:nη+1
        if row_block == 1
            for col_block = 1:n_in
                C_mat[(row_block-1)*n_out+1:row_block*n_out,
                    (col_block-1)*(na+1)*nx+1:(col_block-1)*(na+1)*nx+nx] =
                    mdl.C[:,(col_block-1)*nx+1:col_block*nx]
                # C_mat[:, (col_block-1)*(na+1)*nx+1:col_block*(na+1)*nx]
                # = hcat(mdl.C[:,(col_block-1)*nx+1:col_block*nx], zeros(n_out, na*nx))
            end
        elseif row_block <= na+1
            ind = row_block-1
            for col_block = 1:n_in
                C_mat[(row_block-1)*n_out+1:row_block*n_out,
                    ind*nx+(col_block-1)*(na+1)*nx+1:ind*nx+(col_block-1)*(na+1)*nx+nx] =
                    mdl.C[:,(col_block-1)*nx+1:col_block*nx]
            end
        else
            k, j, i = get_k_j_i(sens_inds[row_block-1])
            C_mat[(row_block-1)*n_out+i, (j-1)*(na+1)*nx+k] = 1
        end
    end

    return DT_SS_Model(A_mat, B_mat, C_mat, zeros(nx_sens*n_in), Ts)
end

# Analytically equivalent to non-alt version, but should be numerically more efficient.
# Instead of computing huge matrices that are then inverted, it inverts several small matrices,
# which are blocks in the huge matrix.
function discretize_ct_noise_model_with_sensitivities_alt(
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
    nη   = length(sens_inds)
    na = length(sens_inds_a)
    nx_sens = (1+na)*nx

    Aηa = zeros(na*nx, nx)
    for i = 1:na
        Aηa[(i-1)*nx+1, sens_inds_a[i]] = -1
    end

    M = [mdl.A             zeros(nx, na*nx)                mdl.B*(mdl.B');
         Aηa         kron(Matrix(1.0I, na, na), mdl.A)   zeros(nx*na, nx);
         zeros(nx, nx)     zeros(nx, nx*na)                -mdl.A' ]
    Mexp = exp(M*Ts)
    Ad   = Mexp[1:nx, 1:nx]
    Dd   = Hermitian(Mexp[1:nx, (na+1)*nx+1:(na+2)*nx]*(Ad'))
    Bd   = cholesky(Dd).L
    # Adηa = Mexp[nx+1:(na+1)*nx, 1:nx]
    # temp = Mexp[nx+1:(nθ+1)*nx, (nθ+1)*nx+1:(nθ+2)*nx]*(Ad')

    # # OLD, creates huge matrices which it then "inverts" (solves equations rather)
    # Ddηa = Mexp[nx+1:(na+1)*nx, (na+1)*nx+1:(na+2)*nx]*(Ad')
    # for i = 1:na
    #     Ddηa[(i-1)*nx+1:i*nx, :] += (Ddηa[(i-1)*nx+1:i*nx, :])'
    # end
    # Φ_arg = ((kron(Matrix(I, na, na), Bd)) \ Ddηa ) / (Bd')
    # Bdηa = kron(Matrix(I, na, na), Bd)*Φ( Φ_arg )

    # NEW, "inverts"/solves equations with many smaller matrices instead of one huge one. Should be more efficient.
    Bdηa = zeros(na*nx, nx)
    for i = 1:na
        # H = Mexp[i*nx+1:(i+1)*nx, nx*(na+1)+1:nx*(2+na)]
        Ddηai = Mexp[i*nx+1:(i+1)*nx, nx*(na+1)+1:nx*(2+na)]*(Ad')
        Ddηai += Ddηai'
        Bdηa[(i-1)*nx+1:i*nx,:] = Bd*Phi((Bd\Ddηai)/(Bd'), nx)
    end

    # A_mat = [Ad zeros(nx,na*nx); Adηa kron(Matrix(I,na,na), Ad)]
    A_mat = Mexp[1:nx*(1+na), 1:nx*(1+na)]
    B_mat = [Bd; Bdηa]
    C_mat = zeros((nη+1)n_out, nx_sens*n_in)
    for row_block = 1:nη+1
        if row_block == 1
            for col_block = 1:n_in
                C_mat[(row_block-1)*n_out+1:row_block*n_out,
                    (col_block-1)*(na+1)*nx+1:(col_block-1)*(na+1)*nx+nx] =
                    mdl.C[:,(col_block-1)*nx+1:col_block*nx]
                # C_mat[:, (col_block-1)*(na+1)*nx+1:col_block*(na+1)*nx]
                # = hcat(mdl.C[:,(col_block-1)*nx+1:col_block*nx], zeros(n_out, na*nx))
            end
        elseif row_block <= na+1
            ind = row_block-1
            for col_block = 1:n_in
                C_mat[(row_block-1)*n_out+1:row_block*n_out,
                    ind*nx+(col_block-1)*(na+1)*nx+1:ind*nx+(col_block-1)*(na+1)*nx+nx] =
                    mdl.C[:,(col_block-1)*nx+1:col_block*nx]
            end
        else
            k, j, i = get_k_j_i(sens_inds[row_block-1])
            C_mat[(row_block-1)*n_out+i, (j-1)*(na+1)*nx+k] = 1
        end
    end

    return DT_SS_Model(A_mat, B_mat, C_mat, zeros(nx_sens*n_in), Ts)
end

# New version, after realizing that in fact the SDE should be differentiated before it is discretized.
# This makes it much simpler to obtain the sensitivity matrices
function discretize_ct_noise_model_with_sensitivities_alt_new(
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
    nη   = length(sens_inds)
    na = length(sens_inds_a)
    nx_sens = (1+na)*nx

    Aηa = zeros(na*nx, nx)
    for i = 1:na
        Aηa[(i-1)*nx+1, sens_inds_a[i]] = -1
    end

    
    # Working under the assumption B_θ = 0
    M = [mdl.A             zeros(nx, na*nx)                mdl.B*(mdl.B')       zeros(nx, nx*na);
         Aηa         kron(Matrix(1.0I, na, na), mdl.A)   zeros(nx*na, nx)       zeros(nx*na, nx*na);
         zeros(nx, nx)     zeros(nx, nx*na)                -mdl.A'              -Aηa';
         zeros(nx*na, nx)    zeros(nx*na, nx*na)           zeros(nx*na, nx)      kron(Matrix(1.0I, na, na), -mdl.A')]
    Mexp = exp(M*Ts)

    Ad   = Mexp[1:nx*(na+1), 1:nx*(na+1)]
    Dd =     Hermitian(Mexp[1:nx*(na+1),nx*(na+1)+1:2nx*(na+1)]*(Ad'))
    Bd = (cholesky(Dd).L)*(Ad')

    C_mat = zeros((nη+1)n_out, nx_sens*n_in)
    for row_block = 1:nη+1
        if row_block == 1
            for col_block = 1:n_in
                C_mat[(row_block-1)*n_out+1:row_block*n_out,
                    (col_block-1)*(na+1)*nx+1:(col_block-1)*(na+1)*nx+nx] =
                    mdl.C[:,(col_block-1)*nx+1:col_block*nx]
                # C_mat[:, (col_block-1)*(na+1)*nx+1:col_block*(na+1)*nx]
                # = hcat(mdl.C[:,(col_block-1)*nx+1:col_block*nx], zeros(n_out, na*nx))
            end
        elseif row_block <= na+1
            ind = row_block-1
            for col_block = 1:n_in
                C_mat[(row_block-1)*n_out+1:row_block*n_out,
                    ind*nx+(col_block-1)*(na+1)*nx+1:ind*nx+(col_block-1)*(na+1)*nx+nx] =
                    mdl.C[:,(col_block-1)*nx+1:col_block*nx]
            end
        else
            k, j, i = get_k_j_i(sens_inds[row_block-1])
            C_mat[(row_block-1)*n_out+i, (j-1)*(na+1)*nx+k] = 1
        end
    end

    return DT_SS_Model(Ad, Bd, C_mat, zeros(nx_sens*n_in), Ts)
end

function discretize_ct_noise_model_with_sensitivities_for_adj(
    mdl::CT_SS_Model, Ts::Float64, sens_inds::Array{Int64, 1})::Tuple{DT_SS_Model, Matrix{Float64}, Matrix{Float64}}
    # sens_inds: indices of parameter with respect to which we compute the
    # sensitivity of disturbance output w

    @assert (length(sens_inds) > 0) "Make sure at least one disturbance parameter is marked for identification. Can't create model for sensitivity with respect to no parameters."

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
    # sens_inds_c = sens_inds[findall(sens_inds .> nx)]
    # nη   = length(sens_inds)
    na = length(sens_inds_a)
    # nx_sens = (1+na)*nx

    Aηa = zeros(na*nx, nx)
    for i = 1:na
        Aηa[(i-1)*nx+1, sens_inds_a[i]] = -1
    end

    M = [zeros(nx, (na+1)*nx)      Matrix(I, nx, nx)      zeros(nx, na*nx)                zeros(nx, nx)
         zeros(na*nx, (na+1)*nx)    zeros(na*nx, nx)   Matrix(I, na*nx, na*nx)            zeros(na*nx, nx)
         zeros(nx, (na+1)*nx)           mdl.A             zeros(nx, na*nx)                mdl.B*(mdl.B');
         zeros(na*nx, (na+1)*nx)         Aηa         kron(Matrix(1.0I, na, na), mdl.A)   zeros(nx*na, nx);
         zeros(nx, (na+1)*nx)          zeros(nx, nx)     zeros(nx, nx*na)                -mdl.A' ]

    Mexp = exp(M*Ts)
    Ad   = Mexp[(na+1)*nx + 1:(na+2)*nx, (na+1)*nx + 1:(na+2)*nx]
    Dd   = Hermitian(Mexp[(na+1)*nx + 1:(na+2)*nx, (2na+2)*nx + 1:(2na+3)*nx]*(Ad'))
    Bd   = cholesky(Dd).L
    Bdηa = zeros(na*nx, nx)
    for i = 1:na
        # H = Mexp[(na+i+1)*nx + 1:(na+i+2)*nx, (2na+2)*nx + 1:(2na+3)*nx]
        Ddηai = Mexp[(na+i+1)*nx + 1:(na+i+2)*nx, (2na+2)*nx + 1:(2na+3)*nx]*(Ad')
        Ddηai += Ddηai'
        Bdηa[(i-1)*nx+1:i*nx,:] = Bd*Phi((Bd\Ddηai)/(Bd'), nx)
    end

    # Matrices needed for adjoint disturbance estimation
    P = Mexp[1:nx, (na+1)*nx + 1:(na+2)*nx]
    R = Mexp[nx+1:(na+1)*nx, (na+1)*nx + 1: (na+2)*nx]
    B̃ηa = kron(Matrix(I,na,na), P) \ (Bdηa - (R/P)*Bd)
    B̃  = P\Bd

    # Returns non-sensitivity disturbance model and other matrices needed for adjoint disturbance sensitivity
    return DT_SS_Model(Ad, Bd, mdl.C, zeros(nx*n_in), Ts), B̃, B̃ηa

    # A_mat = Mexp[(na+1)*nx+1:(2+2na)*nx, (na+1)*nx+1:(2+2na)*nx]
    # B_mat = [Bd; Bdηa]
    # C_mat = zeros((nη+1)n_out, nx_sens*n_in)
    # for row_block = 1:nη+1
    #     if row_block == 1
    #         for col_block = 1:n_in
    #             C_mat[(row_block-1)*n_out+1:row_block*n_out,
    #                 (col_block-1)*(na+1)*nx+1:(col_block-1)*(na+1)*nx+nx] =
    #                 mdl.C[:,(col_block-1)*nx+1:col_block*nx]
    #             # C_mat[:, (col_block-1)*(na+1)*nx+1:col_block*(na+1)*nx]
    #             # = hcat(mdl.C[:,(col_block-1)*nx+1:col_block*nx], zeros(n_out, na*nx))
    #         end
    #     elseif row_block <= na+1
    #         ind = row_block-1
    #         for col_block = 1:n_in
    #             C_mat[(row_block-1)*n_out+1:row_block*n_out,
    #                 ind*nx+(col_block-1)*(na+1)*nx+1:ind*nx+(col_block-1)*(na+1)*nx+nx] =
    #                 mdl.C[:,(col_block-1)*nx+1:col_block*nx]
    #         end
    #     else
    #         k, j, i = get_k_j_i(sens_inds[row_block-1])
    #         C_mat[(row_block-1)*n_out+i, (j-1)*(na+1)*nx+k] = 1
    #     end
    # end

    # # The whole point is that we don't need to simulate disturbance sensitivity forward in time. Why would we then return that model from this function? Just return non-sensitivity model!!
    # return DT_SS_Model(A_mat, B_mat, C_mat, zeros(nx_sens*n_in), Ts), B̃, B̃ηa
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
    #= With the dimensions we most commonly use, the model becomes
    A = [-a1 -a2
          1   0 ]
    B = [1; 0]
    C = [c1 c2]
    =#
end

# ================= Functions simulating disturbance =======================

# NOTE: I don't think this function is used anywhere, and it seems to have
# something funky going on with dimensions and shapes
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
            y, t, x = lsim(sys, data[m][:, (ind-1)*nv+1:ind*nv]', t, x0=mdl.x0)
            # # NOTE: VERSION: The line above here used to be:
            # y, t, x = lsim(sys, data[m][:, (ind-1)*nv+1:ind*nv], t, x0=mdl.x0)    # Input signal wasn't transposed
            # # Changed on 09-08-2022 after it not being used for a long time,
            # # because suddenly transpose was needed. Not sure why (edit: because new version of Julia, but don't know more)
            for i=1:N+1
                x_process[(i-1)*n_in*nx + (ind-1)*nx + 1: (i-1)*n_in*nx + ind*nx, m] = x[:,i]
                # # NOTE: VERSION: The line above here used to be:
                # x_process[(i-1)*n_in*nx + (ind-1)*nx + 1: (i-1)*n_in*nx + ind*nx, m] = x[i,:]     # Elements of x changed around
                # # Changed on 09-08-2022 after it not being used for a long time,
                # # because suddently transpose was needed. Not sure why (edit: because new version of Julia, but don't know more)
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
        discretize_ct_noise_model(get_ct_disturbance_model(η0, nx, n_out), Ts)
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
        discretize_ct_noise_model(get_ct_disturbance_model(η0, nx, n_out), Ts)
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
    w_scale = scale*ones(n_out)             # noise scale
    # Denominator of every transfer function is given by p(s), where
    # p(s) = s^n + a[1]*s^(n-1) + ... + a[n-1]*s + a[n]
    a_vec = [2*ω*ζ, ω^2]
    c_vec = Diagonal(w_scale)*[0, 1]
    η0 = vcat(a_vec, c_vec)
    dη = length(η0)
    mdl =
        discretize_ct_noise_model(get_ct_disturbance_model(η0, nx, n_out), Ts)
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
        discretize_ct_noise_model(get_ct_disturbance_model(η0, nx, n_out), Ts)
    # return mdl, DataFrame(nx = nx, n_in = n_in, n_out = n_out, η = η0, bias=bias)
    return mdl, [nx, n_in, n_out], η0
end


function get_filtered_noise(gen::Function, Ts::Float64, M::Int, Nw::Int;
    bias::Float64=0.0, scale::Float64=1.0)::Tuple{Array{Float64,2}, Array{Float64,2}, DataFrame}

    mdl, meta_raw, η0 = gen(Ts, scale=scale)
    metadata = DataFrame(nx = meta_raw[1], n_in = meta_raw[2], n_out = meta_raw[3], η = η0, bias=bias, num_rel = M)
    n_tot = size(mdl.Cd,2)

    # We used to have Nw+1, since we want samples at t₀, t₁, ..., t_{N_w}, i.e. a total of N_w+1 samples, 
    # but I think I changed the meaning of N_w, so just N_w is what it should be now
    ZS = [randn(Nw, n_tot) for m = 1:M]
    XW = simulate_noise_process_mangled(mdl, ZS)
    XW, get_system_output_mangled(mdl, XW).+ bias, metadata
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
