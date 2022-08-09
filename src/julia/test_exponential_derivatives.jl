using LinearAlgebra

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

function get_ct_disturbance_model(pars::Array{Float64,1}, nx::Int, n_out::Int, u_scale, w_scale)
    # default a is 0.8
    # First nx parameters of η are parameters for A-matrix, the remaining
    # parameters are for the C-matrix
    a_vec = [pars[1], pars[2]]
    c_vec = zeros(n_out*nx*n_in)
    # The first state will act as output of the filter
    c_vec[1] = u_scale
    η = vcat(a_vec, Diagonal(w_scale)*c_vec)
    dη = length(η)

    A = diagm(-1 => ones(nx-1,))
    A[1,:] = -η[1:nx]
    B = zeros(nx,1)
    B[1] = 1.0
    C = reshape(η[nx+1:end], n_out, :)
    x0 = zeros(nx)
    return CT_SS_Model(A, B, C, x0)
end

function get_discretized_disturbances(Mexp, nx, nθ)
    Ad   = Mexp[1:nx, 1:nx]
    Dd   = Mexp[1:nx, (nθ+1)*nx+1:(nθ+2)*nx]*(Ad')
    Adθ  = Mexp[nx+1:(nθ+1)*nx, 1:nx]
    # temp = Mexp[nx+1:(nθ+1)*nx, (nθ+1)*nx+1:(nθ+2)*nx]*(Ad')
    Ddθ = Mexp[nx+1:(nθ+1)*nx, (nθ+1)*nx+1:(nθ+2)*nx]*(Ad')
    for i = 1:nθ
        Ddθ[(i-1)*nx+1:i*nx, :] += (Ddθ[(i-1)*nx+1:i*nx, :])'
    end
    return Ad, Dd, Adθ, Ddθ
end

function get_all_dt_matrices(pars, Ts, scale, nx, n_out, nθ)
    u_scale = scale # input scale
    w_scale = scale*ones(n_out)             # noise scale
    mdl_ct = get_ct_disturbance_model(pars, nx, n_out, u_scale, w_scale)
    mdl_dt = discretize_ct_noise_model(mdl_ct, Ts)
    A = mdl_ct.A
    B = mdl_ct.B
    Aθ = [-1.0 .0; .0 .0; .0 -1; .0 .0]     # NOTE: CHANGE HERE IF Θ CHANGES!
    # Aθ = [-1.0 .0; .0 .0]     # NOTE: CHANGE HERE IF Θ CHANGES!
    # Denominator of every transfer function is given by p(s), where
    # p(s) = s^n + a[1]*s^(n-1) + ... + a[n-1]*s + a[n]

    M = [A                  zeros(nx, nθ*nx)                B*(B');
         Aθ                 kron(Matrix(1.0I, nθ, nθ), A)   zeros(nx*nθ, nx);
         zeros(nx, nx)      zeros(nx, nx*nθ)                -A' ]

    Mexp = exp(M*Ts)

    # @info "M1: $M"
    # @info "Mexp: $Mexp"

    Ad, Dd, Adθ, Ddθ = get_discretized_disturbances(Mexp, nx, nθ)
end

function Φ(mat_in::Array{Float64,2})
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

# function Φ!(mat::Array{Float64,2})
#     for i=1:size(mat,1)
#         for j=1:size(mat,2)
#             i1 = (i-1)%nx+1
#             j1 = (j-1)%nx+1
#             if i1 > j1
#                 continue
#             elseif i1 == j1
#                 mat[i,j] = 0.5*mat[i,j]
#             else
#                 mat[i,j] = 0
#             end
#         end
#     end
#     return mat
# end

Ts = 0.05
scale = 1.0
nx = 2        # model order
n_out = 1     # number of outputs
n_in = 2      # number of inputs
nθ   = 2      # NOTE: CHANGE HERE IF Θ CHANGES

δ = 1e-5#1e-7
Ad1, Dd1, Adθ, Ddθ = get_all_dt_matrices([0.8, 4^2], Ts, scale, nx, n_out, nθ)
Ad2, Dd2, _, _ = get_all_dt_matrices([0.8+δ, 4^2], Ts, scale, nx, n_out, nθ)
Ad3, Dd3, _, _ = get_all_dt_matrices([0.8, 4^2+δ], Ts, scale, nx, n_out, nθ)

# Adθ_est = (Ad2-Ad1)/δ
# Ddθ_est = (Dd2-Dd1)/δ
Adθ_est = vcat((Ad2-Ad1)/δ, (Ad3-Ad1)/δ)
Ddθ_est = vcat((Dd2-Dd1)/δ, (Dd3-Dd1)/δ)

# Yeah, looks like we get good derivative estimates!
println("rel Ad error: $((Adθ-Adθ_est)./Adθ_est)")
println("rel Dd error: $((Ddθ-Ddθ_est)./Ddθ_est)")
# println("Ad error: $(Adθ-Adθ_est)")
# println("Dd error: $(Ddθ-Ddθ_est)")

# test = [1. 2.; 3. 4.; 5. 6.; 7. 8.; 9. 10.; 11. 12.]
# testin = copy(test)
# testout = Φ!(testin)

# θ = 0.8
# Dd1 = [θ 0.1*θ; 0.1*θ 6*θ]
# Dd2 = [(θ+δ) 0.1*(θ+δ); 0.1*(θ+δ) 6*(θ+δ)]
# Ddθ = [1.0 0.1; 0.1 6.0]
# Bd1 = Matrix(cholesky(Dd1).L)
# Bd2 = Matrix(cholesky(Dd2).L)
# Bdθ_est = (Bd2-Bd1)/δ
# temp = (Bd1\Ddθ)/(Bd1')
# Bdθ  = Bd1*Φ(temp)

# Let's try with cholesky too!
Bd1 = cholesky(Hermitian(Dd1)).L        # NOTE: Very important to use .L here!!
Bd2 = cholesky(Hermitian(Dd2)).L
Bd3 = cholesky(Hermitian(Dd3)).L
temp = ((kron(Matrix(I, nθ, nθ), Bd1)) \ Ddθ ) / (Bd1')
Bdθ = kron(Matrix(I, nθ, nθ), Bd1)*Φ( temp )
Bdθ_est = vcat((Bd2-Bd1)/δ, (Bd3-Bd1)/δ)
# Bdθ_est = (Bd2-Bd1)/δ
println("rel Bd error: $((Bdθ-Bdθ_est)./Bdθ_est)")
# println("Bd error: $(Bdθ-Bdθ_est)")

# THIS DOESN'T WORK FOR SOME REASON! fIGURE OUT TOMORROW, TRY WITH SCALAR CASE!!
