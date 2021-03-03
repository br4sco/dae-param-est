using Random, LinearAlgebra
include("noise_generation.jl")
include("noise_interpolation.jl")

struct Special_CT_SS_Model
    A::Array{Float64, 2}
    B::Array{Float64, 2}
    C::Array{Float64, 2}
    D::Array{Float64, 2}
    x0::Array{Float64, 1}
end


function discretize_ss_model(mdl::Special_CT_SS_Model, Ts)
    # NOTE: Assumes invertible A
    Ad   = exp(mdl.A*Ts)
    temp = (mdl.A\ (Ad-I(size(Ad)[1]) ))
    Bd   = temp*mdl.B
    Dd  = temp*mdl.D
    return Ad, Bd, Dd, C
end

function discretize_noise_model(mdl::CT_SS_Model, Ts::Float64)
    nx    = size(mdl.A)[1]
    Mexp  = [-mdl.A mdl.B*(mdl.B'); zeros(size(mdl.A)) mdl.A']
    M     = exp(Mexp*Ts)
    Ad    = M[nx+1:end, nx+1:end]'
    Bd2   = Hermitian(Ad*M[1:nx, nx+1:end])
    Chol  = cholesky(Bd2)
    Bd    = Chol.L
    return Ad, Bd, mdl.C
end

function simulate_system_exactly(mdl::Special_CT_SS_Model,  # System model
                                 noise_mdl::CT_SS_Model,    # Noise model
                                 time::Array{Float64, 1},   # Time vector
                                 u::Array{Float64, 2},      # System input
                                 v::Array{Float64, 2},      # Measurement noise
                                 w::Array{Float64, 2})      # Process noise

    # The input u is assumed to be ZOH, i.e. constant between different values of t
    N  = size(time)[1]
    n  = size(mdl.A)[1]
    m  = size(mdl.C)[1]
    t  = 0.0
    # Element i,j represents the j:th component of the vector at time step i-1
    x  = fill(NaN, (N+1, n))
    y  = fill(NaN, (N+1, m))

    x[1,:] = mdl.x0
    y[1,:] = mdl.C*x[1,:] + v[1,:]

    for k=1:1:size(time)[1]
        δ = time[k] - t
        Ad, Bd, Bdw, Cd = discretize_ss_model(mdl, δ)

        x[k+1,:] = Ad*x[k:k,:]' + Bd*u[k:k,:]' + Bdw*w[k:k,:]'
        y[k+1,:] = Cd*x[k+1:k+1,:]' + v[k+1:k+1,:]'
        t = time[k]
    end
    return y, time, x
end
