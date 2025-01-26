module DynamicalModels

import Interpolations
using DifferentialEquations: DAESolution
using LinearAlgebra: ⋅  # dot product

# TODO: Make it export everything that the user could potentially need! 
export Model, Model_ode, AdjointSDEApproxData, pendulum, pendulum_forward_m, pendulum_forward_k
export get_pendulum_initial, get_pendulum_initial_msens, get_pendulum_initial_Lsens, get_pendulum_initial_ksens, get_pendulum_initial_distsens
export pendulum_adjoint_m, pendulum_adjoint_k_1dist_ODEdist, pendulum_forward_k_1dist

const func_type = Interpolations.Extrapolation

struct Model
    f!::Function                  # residual function
    jac!::Function                # jacobian function
    x0::Vector{Float64}           # initial values of x
    dx0::Vector{Float64}          # initial values of x'
    dvars::Array{Bool, 1}         # bool array indicating differential variables
    ic_check::Vector{Float64}     # residual at t0 (DEBUG)
end

struct Model_ode
    f::Function             # ODE Function
    x0::Vector{Float64}     # initial values of x
end

# When using the version of the adjoint method that approximates the disturbance SDE
# by an ODE to incorporate it into the DAE-system, additional data needs to be passed
# to the model creation function. This data is stored in this struct.
struct AdjointSDEApproxData
    xw::Function
    v::Function
    # TODO: Generally, but especially when having many free disturbance parameters, the below matrices will be very sparse.
    # It would be wise to use a sparse matrix implementation instead. That's a future project though.
    Ǎη::AbstractMatrix{Float64}
    B̌η::AbstractMatrix{Float64}
    Čη::AbstractMatrix{Float64}
    C::AbstractMatrix{Float64}
    ρ::Vector{Float64}  # Contains all disturbance parameters, both free and known
    na::Int64           # Number of the disturbance parameters that correspond to the a-parameters
    nxw::Int64          # Dimension of xw
    nη::Int64           # Number of unknown disturbance parameters
end

################################ PENDULUM MODELS ################################
# NOTE: Used to be called pendulum_new
function pendulum(u::Function, w::Function, pars::Vector{Float64}, model_data::NamedTuple)::Model
    let m = pars[1], L = pars[2], g = pars[3], k = pars[4]

        # the residual function
        function f!(res, dx, x, _, t)
            wt = w(t)
            ut = u(t)
            # Dynamic Equations
            res[1] = dx[1] - x[4] + 2dx[6]*x[1]
            res[2] = dx[2] - x[5] + 2dx[6]*x[2]
            res[3] = m*dx[4] - dx[3]*x[1] + k*abs(x[4])*x[4] - ut[1] - wt[1]^2
            res[4] = m*dx[5] - dx[3]*x[2] + k*abs(x[5])*x[5] + m*g
            res[5] = x[1]^2 + x[2]^2 - L^2
            res[6] = x[4]*x[1] + x[5]*x[2]
            # Equation for obtaining angle
            res[7] = x[7] - atan(x[1] / -x[2])
            nothing
        end

        # Finding consistent initial conditions
        x0, dx0 = get_pendulum_initial(pars, u(0.0)[1], w(0.0)[1], model_data.φ0)

        dvars = vcat(fill(true, 6), [false])

        r0 = zeros(length(x0))
        f!(r0, dx0, x0, [], 0.0)

        # t -> 0.0 is just a dummy function, not to be used
        Model(f!, t -> 0.0, x0, dx0, dvars, r0)
    end
end

# -------------------- Forward sensitivity models ---------------------------

# NOTE: Used to be called pendulum_sensitivity_m
function pendulum_forward_m(u::Function, w::Function, pars::Vector{Float64}, model_data::NamedTuple)::Model
    let m = pars[1], L = pars[2], g = pars[3], k = pars[4]

        # the residual function
        function f!(res, dx, x, _, t)
            wt = w(t)
            ut = u(t)
            # Dynamic Equations
            res[1] = dx[1] - x[4] + 2dx[6]*x[1]
            res[2] = dx[2] - x[5] + 2dx[6]*x[2]
            res[3] = m*dx[4] - dx[3]*x[1] + k*abs(x[4])*x[4] - ut[1] - wt[1]^2
            res[4] = m*dx[5] - dx[3]*x[2] + k*abs(x[5])*x[5] + m*g
            res[5] = x[1]^2 + x[2]^2 - L^2
            res[6] = x[4]*x[1] + x[5]*x[2]
            # Angle of pendulum
            res[7] = x[7] - atan(x[1] / -x[2])
            # Sensitivity with respect to m
            res[8]  = -x[11] + dx[8] + 2x[1]*dx[13] + 2x[8]*dx[6]
            res[9]  = -x[12] + dx[9] + 2x[2]*dx[13] + 2x[9]*dx[6]
            res[10] = 2k*x[11]*abs(x[4]) - x[1]*dx[10] + m*dx[11] - x[8]*dx[3] + dx[4]
            res[11] = 2k*x[12]*abs(x[5]) - x[2]*dx[10] + m*dx[12] - x[9]*dx[3] + g + dx[5]
            res[12] = -2x[8]*x[1] - 2x[9]*x[2]
            res[13] = x[11]*x[1] + x[12]*x[2] + x[8]*x[4] + x[9]*x[5]
            res[14] = x[14] - (x[1]*x[9] - x[2]*x[8])/(L^2)

            nothing
        end
        
        u0 = u(0.0)[1]
        w0 = w(0.0)[1]
        pend0, dpend0 = get_pendulum_initial(pars, u0, w0, model_data.φ0)
        sm0, dsm0 = get_pendulum_initial_msens(pars, u0, w0, model_data.φ0, pend0, dpend0)

        x0  = vcat(pend0, sm0)
        dx0 = vcat(dpend0, dsm0)

        dvars = vcat(fill(true, 6), [false], fill(true, 6), [false])

        r0 = zeros(length(x0))
        f!(r0, dx0, x0, [], 0.0)

        # t -> 0.0 is just a dummy function, not to be used
        Model(f!, t -> 0.0, x0, dx0, dvars, r0)
    end
end

function pendulum_forward_k(u::Function, w::Function, pars::Vector{Float64}, model_data::NamedTuple)::Model
    let m = pars[1], L = pars[2], g = pars[3], k = pars[4]

        # the residual function
        function f!(res, dx, x, _, t)
            wt = w(t)
            ut = u(t)
            # Dynamic Equations
            res[1] = dx[1] - x[4] + 2dx[6]*x[1]
            res[2] = dx[2] - x[5] + 2dx[6]*x[2]
            res[3] = m*dx[4] - dx[3]*x[1] + k*abs(x[4])*x[4] - ut[1] - wt[1]^2
            res[4] = m*dx[5] - dx[3]*x[2] + k*abs(x[5])*x[5] + m*g
            res[5] = x[1]^2 + x[2]^2 - L^2
            res[6] = x[4]*x[1] + x[5]*x[2]
            # Angle of pendulum
            res[7] = x[7] - atan(x[1] / -x[2])
            # Sensitivity Equations (wrt k)
            res[8]  = -x[11] + dx[8] + 2x[1]*dx[13] + 2x[8]*dx[6]
            res[9]  = -x[12] + dx[9] + 2x[2]*dx[13] + 2x[9]*dx[6]
            res[10] = abs(x[4])*x[4] + 2k*x[11]*abs(x[4]) - x[1]*dx[10] + m*dx[11] - x[8]*dx[3]
            res[11] = abs(x[5])*x[5] + 2k*x[12]*abs(x[5]) - x[2]*dx[10] + m*dx[12] - x[9]*dx[3]
            res[12] = -2x[8]*x[1] - 2x[9]*x[2]
            res[13] = x[11]*x[1] + x[12]*x[2] + x[8]*x[4] + x[9]*x[5]
            # Sensitivity of angle of pendulum
            res[14] = x[14] - (x[1]*x[9] - x[8]*x[2])/(L^2)

            nothing
        end

        u0 = u(0.0)[1]
        w0 = w(0.0)[1]
        pend0, dpend0 = get_pendulum_initial(pars, u0, w0, model_data.φ0)
        sk0, dsk0 = get_pendulum_initial_ksens(pars, u0, w0, model_data.φ0, pend0, dpend0)
        x0  = vcat(pend0, sk0)
        dx0 = vcat(dpend0, dsk0)

        dvars = vcat(fill(true, 6), [false], fill(true, 6), [false])

        r0 = zeros(length(x0))
        f!(r0, dx0, x0, [], 0.0)

        # t -> 0.0 is just a dummy function, not to be used
        Model(f!, t -> 0.0, x0, dx0, dvars, r0)
    end
end

function pendulum_forward_k_1dist(u::Function, w::Function, pars::Vector{Float64}, model_data::NamedTuple)::Model
    let m = pars[1], L = pars[2], g = pars[3], k = pars[4]

        # the residual function
        function f!(res, dx, x, _, t)
            wt = w(t)
            ut = u(t)
            # Dynamic Equations
            res[1] = dx[1] - x[4] + 2dx[6]*x[1]
            res[2] = dx[2] - x[5] + 2dx[6]*x[2]
            res[3] = m*dx[4] - dx[3]*x[1] + k*abs(x[4])*x[4] - ut[1] - wt[1]^2
            res[4] = m*dx[5] - dx[3]*x[2] + k*abs(x[5])*x[5] + m*g
            res[5] = x[1]^2 + x[2]^2 - L^2
            res[6] = x[4]*x[1] + x[5]*x[2]
            # Angle of pendulum
            res[7] = x[7] - atan(x[1] / -x[2])

            # Sensitivity with respect to k
            res[8]  = -x[11] + dx[8] + 2x[1]*dx[13] + 2x[8]*dx[6]
            res[9]  = -x[12] + dx[9] + 2x[2]*dx[13] + 2x[9]*dx[6]
            res[10] = 2k*x[11]*abs(x[4]) - x[1]*dx[10] + m*dx[11] - x[8]*dx[3] + abs(x[4])x[4]
            res[11] = 2k*x[12]*abs(x[5]) - x[2]*dx[10] + m*dx[12] - x[9]*dx[3] + abs(x[5])x[5]
            res[12] = -2x[8]*x[1] - 2x[9]*x[2]
            res[13] = x[11]*x[1] + x[12]*x[2] + x[8]*x[4] + x[9]*x[5]
            res[14] = x[14] - (x[1]*x[9] - x[2]*x[8])/(L^2)

            # Sensitivty wrt first disturbance parameter
            # (i.e. parameter corresponding to wt[2])
            res[15] = 2dx[6]x[15] - x[18] + dx[15] + 2x[1]dx[20]
            res[16] = 2dx[6]x[16] - x[19] + dx[16] + 2x[2]dx[20]
            res[17] = -dx[3]x[15] + 2k*abs(x[4])*x[18] - x[1]*dx[17] + m*dx[18] - 2*wt[1]*wt[2]
            res[18] = -dx[3]x[16] + 2k*abs(x[5])*x[19] - x[2]*dx[17] + m*dx[19]
            res[19] = 2x[1]x[15] + 2x[2]x[16]
            res[20] = x[4]x[15] + x[5]x[16] + x[1]x[18] + x[2]x[19]
            # Sensitivity of angle of pendulum to disturbance parameter
            res[21] = x[21] - (x[1]*x[16] - x[2]*x[15])/(L^2)

            nothing
        end

        u0 = u(0.0)[1]
        w0 = w(0.0)[1]
        pend0, dpend0 = get_pendulum_initial(pars, u0, w0, model_data.φ0)
        sk0, dsk0 = get_pendulum_initial_ksens(pars, u0, w0, model_data.φ0, pend0, dpend0)
        r0, dr0 = get_pendulum_initial_distsens(pars, u0, w0, model_data.φ0, pend0, dpend0)

        x0  = vcat(pend0, sk0, r0)
        dx0 = vcat(dpend0, dsk0, dr0)

        dvars = repeat(vcat(fill(true, 6), [false]), 3)

        r0 = zeros(length(x0))
        f!(r0, dx0, x0, [], 0.0)

        # t -> 0.0 is just a dummy function, not to be used
        Model(f!, t -> 0.0, x0, dx0, dvars, r0)
    end
end

# -------------------- Adjoint sensitivity models ---------------------------

# NOTE: Used to be called my_pendulum_adjoint_monly
function pendulum_adjoint_m(_::Function, pars::Vector{Float64}, T::Float64, x::func_type, x2::func_type, y::func_type, dy::func_type, xθ0::Matrix{Float64}, dx::func_type, dx2::func_type)::Tuple{Model,Function}
    let m = pars[1], L = pars[2], g = pars[3], k = pars[4]
        nx, nθ = size(xθ0)
        @assert (nθ == 1) "pendulum_adjoint_m is hard-coded to only handle one parameter m, make sure to pass correct xθ0. Currently passing $nθ parameters."

        FxT = [2dx(T,6)          0.0            0.0   -1.0              0.0             0.0   0.0
                0.0           2*dx(T,6)         0.0    0.0             -1.0             0.0   0.0
                -dx(T,3)         0.0            0.0   2k*abs(x(T,4))    0.0             0.0   0.0
                0.0            -dx(T,3)         0.0   0.0              2k*abs(x(T,5))  0.0   0.0
                2x(T,1)        2x(T,2)          0.0   0.0              0.0             0.0   0.0
                x(T,4)         x(T,5)           0.0   x(T,1)            x(T,2)          0.0   0.0
                x(T,2)/(L^2)  -x(T,1)/(L^2)     0.0   0.0               0.0             0.0   1.0]
        Fdx = t-> vcat([1.0   0.0   0.0          0.0   0.0     2x(t,1)    0.0
                        0.0   1.0   0.0          0.0   0.0     2x(t,2)    0.0
                        0.0   0.0   -x(t,1)      m     0.0     0.0        0.0
                        0.0   0.0   -x(t,2)      0.0   m       0.0        0.0], zeros(3,7))
        dFdxT = vcat([  0.0   0.0  0.0         0.0   0.0   2dx(T,1)    0.0
                        0.0   0.0  0.0         0.0   0.0   2dx(T,2)    0.0
                        0.0   0.0  -dx(T,1)    0.0   0.0   0.0         0.0
                        0.0   0.0  -dx(T,2)    0.0   0.0   0.0         0.0], zeros(3,7))
        FθT = [.0; .0; dx(T,4); dx(T,5)+g; .0; .0; .0 ;;]
        gₓT  = [.0    .0    .0    .0    .0    .0    2(x2(T,7)-y(T,1))/T]
        dgₓT = [.0    .0    .0    .0    .0    .0    2(dx2(T,7)-dy(T,1))/T]


        # The residual function
        function f!(res, dz, z, _, t)
            # ---------- Adjoint system ----------
            # 0 = dλᵀ Fdx + λᵀ(dFdx - Fx) + gₓ                      # Analytical adjoint system
            # 0 = (dz')*Fdx(t) + (z')*(dFdx(t) - Fx(t)) + gₓ(t)     # Matrix form adjoint system
            # Expanded adjoint system residual equations:
            res[1]  = dz[1] - 2*dx(t,6)*z[1] + dx(t,3)*z[3] - 2*x(t,1)*z[5] - x(t,4)*z[6] - (x(t,2)*z[7])/(L^2)
            res[2]  = dz[2] - 2*dx(t,6)*z[2] + dx(t,3)*z[4] - 2*x(t,2)*z[5] - x(t,5)*z[6] + (x(t,1)*z[7])/(L^2)
            res[3]  = -x(t,1)*dz[3] - x(t,2)*dz[4] - dx(t,1)*z[3] - dx(t,2)*z[4]
            res[4]  = m*dz[3] + z[1] - 2*k*abs(x(t,4))*z[3] - x(t,1)*z[6]
            res[5]  = m*dz[4] + z[2] - 2*k*abs(x(t,5))*z[4] - x(t,2)*z[6]
            res[6]  = 2*x(t,1)*dz[1] + 2*x(t,2)*dz[2] + 2*dx(t,1)*z[1] + 2*dx(t,2)*z[2]
            res[7]  = (2*(x2(t,7) - y(t,1)))/T - z[7]
            # ---------- β-equations ----------
            # 0 = dβᵢ - gθᵢ + λᵀ(Fθᵢ + Fw wθᵢ)   
            res[8]  = dz[8] - z[3]*dx(t,4) - z[4]*(dx(t,5)+g)       # For parameter m, only dβᵢ - λᵀFθᵢ
            nothing
        end

        dinds  = 1:4 # Indices of differential variables
        ainds  = 5:7 # Indices of algebraic variables

        ######################## INITIALIZING ADJOINT SYSTEM (Computing terminal values) ####################
        λT, dλT = get_initial_adjoint(dinds, ainds, gₓT, dgₓT, FxT, Fdx(T), dFdxT)
        βT, dβT = get_initial_adjoint_beta(λT, FθT, zeros(nθ))
        z0 = vcat(λT, βT)
        dz0 = vcat(dλT, dβT)

        # Function returning Gθ given adjoint solution
        function get_Gθ(adj_sol::DAESolution)
            # Because the problem is solved backwards in time, the end of the solution corresponds to t=0
            # Gθᵢ = βᵢ(0) + (λᵀFdx xθᵢ)(0)
            adj_sol.u[end][nx+1:nx+nθ] .+ (((adj_sol.u[end][1:nx]')*Fdx(0.0))*xθ0)
        end

        dvars = fill(true, nx+nθ)
        dvars[dinds] .= true
        dvars[end-nθ:end] .= true

        r0 = zeros(length(z0))
        f!(r0, dz0, z0, [], 0.0)
        # @info "r0 is: $r0"

        # t -> 0.0 is just a dummy function, not to be used
        return Model(f!, t -> 0.0, z0, dz0, dvars, r0), get_Gθ
    end
end

function pendulum_adjoint_k_1dist(w::Function, pars::Vector{Float64}, T::Float64, x::func_type, x2::func_type, y::func_type, dy::func_type, xθ0::Matrix{Float64}, dx::func_type, dx2::func_type)::Tuple{Model,Function}
    let m = pars[1], L = pars[2], g = pars[3], k = pars[4]
        nx, nθ = size(xθ0)
        @assert (nθ == 2) "pendulum_adjoint_k_1dist is hard-coded to only handle two parameters, k and one disturbance parameter. Make sure to pass correct xθ0. Currently passing $nθ parameters."

        # Most matrices need only to be evaluated at t=T, because inside f!() we hardcode the equations and don't use these matrices
        FxT = [2dx(T,6)          0.0            0.0   -1.0              0.0             0.0   0.0
                0.0           2*dx(T,6)         0.0    0.0             -1.0             0.0   0.0
                -dx(T,3)         0.0            0.0   2k*abs(x(T,4))    0.0             0.0   0.0
                0.0            -dx(T,3)         0.0   0.0               2k*abs(x(T,5))  0.0   0.0
                2x(T,1)        2x(T,2)          0.0   0.0               0.0             0.0   0.0
                x(T,4)         x(T,5)           0.0   x(T,1)            x(T,2)          0.0   0.0
                x(T,2)/(L^2)  -x(T,1)/(L^2)     0.0   0.0               0.0             0.0   1.0]
        Fdx = t-> vcat([1.0   0.0   0.0          0.0   0.0     2x(t,1)    0.0
                        0.0   1.0   0.0          0.0   0.0     2x(t,2)    0.0
                        0.0   0.0   -x(t,1)      m     0.0     0.0        0.0
                        0.0   0.0   -x(t,2)      0.0   m       0.0        0.0], zeros(3,7))
        dFdxT = vcat([  0.0   0.0  0.0         0.0   0.0   2dx(T,1)    0.0
                        0.0   0.0  0.0         0.0   0.0   2dx(T,2)    0.0
                        0.0   0.0  -dx(T,1)    0.0   0.0   0.0         0.0
                        0.0   0.0  -dx(T,2)    0.0   0.0   0.0         0.0], zeros(3,7))
        FwT = [.0; .0; -2w(T)[1]; .0; .0; .0; .0 ;;]
        FθT = [ .0
                .0
                abs(x(T,4))*x(T,4)
                abs(x(T,5))*x(T,5)
                .0
                .0
                .0 ;;] # NOTE: Just for dynamical parameter
        gₓT  = [.0    .0    .0    .0    .0    .0    2(x2(T,7)-y(T,1))/T]
        dgₓT = [.0    .0    .0    .0    .0    .0    2(dx2(T,7)-dy(T,1))/T]

        # The residual function
        function f!(res, dz, z, _, t)
            # ---------- Adjoint system ----------
            # 0 = dλᵀ Fdx + λᵀ(dFdx - Fx) + gₓ                      # Analytical adjoint system
            # 0 = (dz')*Fdx(t) + (z')*(dFdx(t) - Fx(t)) + gₓ(t)     # Matrix form adjoint system
            # Expanded adjoint system residual equations:
            res[1]  = dz[1] - 2*dx(t,6)*z[1] + dx(t,3)*z[3] - 2*x(t,1)*z[5] - x(t,4)*z[6] - (x(t,2)*z[7])/(L^2)
            res[2]  = dz[2] - 2*dx(t,6)*z[2] + dx(t,3)*z[4] - 2*x(t,2)*z[5] - x(t,5)*z[6] + (x(t,1)*z[7])/(L^2)
            res[3]  = -x(t,1)*dz[3] - x(t,2)*dz[4] - dx(t,1)*z[3] - dx(t,2)*z[4]
            res[4]  = m*dz[3] + z[1] - 2*k*abs(x(t,4))*z[3] - x(t,1)*z[6]
            res[5]  = m*dz[4] + z[2] - 2*k*abs(x(t,5))*z[4] - x(t,2)*z[6]
            res[6]  = 2*x(t,1)*dz[1] + 2*x(t,2)*dz[2] + 2*dx(t,1)*z[1] + 2*dx(t,2)*z[2]
            res[7]  = (2*(x2(t,7) - y(t,1)))/T - z[7]
            # ---------- β-equations ----------
            # 0 = dβᵢ - gθᵢ + λᵀ(Fθᵢ + Fw wθᵢ)                                  # Full analytical form
            res[8] = dz[8] - z[3]*abs(x(t,4))*x(t,4) - z[4]*abs(x(t,5))*x(t,5)  # For parameter k, only dβᵢ - λᵀFθᵢ
            res[9] = dz[9] + 2z[3]*w(t)[1]*w(t)[2]                              # For disturbance parameter, only dβᵢ - λᵀ Fw wθᵢ
            nothing
        end

        dinds = 1:4 # Indices of differential variables
        ainds = 5:7 # Indices of algebraic variables

        ######################## INITIALIZING ADJOINT SYSTEM (Computing terminal values) ####################
        λT, dλT = get_initial_adjoint(dinds, ainds, gₓT, dgₓT, FxT, Fdx(T), dFdxT)
        βT, dβT = get_initial_adjoint_beta(λT, FθT, zeros(1))
        βdistT, dβdistT = get_initial_adjoint_beta_dist(λT, FwT, w(T)[2:2])
        z0 = vcat(λT, βT, [βdistT])
        dz0 = vcat(dλT, dβT, [dβdistT])

        # Function returning Gθ given adjoint solution
        function get_Gθ(adj_sol::DAESolution)::Vector{Float64}
            # Because the problem is solved backwards in time, the end of the solution corresponds to t=0
            # Gθᵢ = βᵢ(0) + (λᵀFdx xθᵢ)(0)
            adj_sol.u[end][nx+1:nx+nθ] .+ (((adj_sol.u[end][1:nx]')*Fdx(0.0))*xθ0)[:]
        end

        dvars = fill(true, nx+nθ)
        dvars[dinds] .= true
        dvars[end-nθ:end] .= true

        r0 = zeros(length(z0))
        f!(r0, dz0, z0, [], 0.0)
        # @info "r0 is: $r0"

        # t -> 0.0 is just a dummy function, not to be used
        return Model(f!, t -> 0.0, z0, dz0, dvars, r0), get_Gθ
    end
end

# -------------------- Adjoint sensitivity models with ODE disturbance model ---------------------------

# NOTE: Additional assumptions are nxw=2, nw=1, only relevant in residual function
function pendulum_adjoint_k_1dist_ODEdist(w::Function, pars::Vector{Float64}, T::Float64, x::func_type, x2::func_type, y::func_type, dy::func_type, xθ0::Matrix{Float64}, dx::func_type, dx2::func_type, ad::AdjointSDEApproxData)::Tuple{Model,Function}
    let m = pars[1], L = pars[2], g = pars[3], k = pars[4]
        nθ = size(xθ0,2)
        n = 7
        # nη = 1
        nx = size(xθ0,1)
        ndist = ad.nxw + length(w(0.0))
        @assert (nθ == 2) "pendulum_adjoint_k_1dist_ODEdist is hard-coded to only handle parameters k, and one disturbance parameter. Make sure to pass correct xθ0 (currently passing for $nθ parameters)"

        # Most matrices need only to be evaluated at t=T, because inside f!() we hardcode the equations and don't use these matrices
        FxT = [2dx(T,6)          0.0            0.0   -1.0              0.0             0.0   0.0
                0.0           2*dx(T,6)         0.0    0.0             -1.0             0.0   0.0
                -dx(T,3)         0.0            0.0   2k*abs(x(T,4))    0.0             0.0   0.0
                0.0            -dx(T,3)         0.0   0.0              2k*abs(x(T,5))  0.0   0.0
                2x(T,1)        2x(T,2)          0.0   0.0              0.0             0.0   0.0
                x(T,4)         x(T,5)           0.0   x(T,1)            x(T,2)          0.0   0.0
                x(T,2)/(L^2)  -x(T,1)/(L^2)     0.0   0.0               0.0             0.0   1.0]
        Fdx = t-> vcat([1.0   0.0   0.0          0.0   0.0     2x(t,1)    0.0
                        0.0   1.0   0.0          0.0   0.0     2x(t,2)    0.0
                        0.0   0.0   -x(t,1)      m     0.0     0.0        0.0
                        0.0   0.0   -x(t,2)      0.0   m       0.0        0.0], zeros(3,7))
        dFdxT = vcat([  0.0   0.0  0.0         0.0   0.0   2dx(T,1)    0.0
                        0.0   0.0  0.0         0.0   0.0   2dx(T,2)    0.0
                        0.0   0.0  -dx(T,1)    0.0   0.0   0.0         0.0
                        0.0   0.0  -dx(T,2)    0.0   0.0   0.0         0.0], zeros(3,7))
        FwT = [.0; .0; -2w(T)[1]; .0; .0; .0; .0 ;;]
        FθT = [ .0
                .0
                abs(x(T,4))*x(T,4)
                abs(x(T,5))*x(T,5)
                .0
                .0
                .0 ;;] # NOTE: Just for dynamical parameter
        gₓT  = [.0    .0    .0    .0    .0    .0    2(x2(T,7)-y(T,1))/T]
        dgₓT = [.0    .0    .0    .0    .0    .0    2(dx2(T,7)-dy(T,1))/T]

        # The residual function
        # Note, everything except the residual function is independent of which disturbance model is used.
        # This residual function assumes that the disturbance model has nxw = 2, nw = 1
        function f!(res, dz, z, _, t)
            # ---------- Nominal adjoint system ----------
            # 0 = dλ̇ᵀ Fdx + λᵀ(dFdx - Fx) + gₓ                      # Analytical adjoint system
            # 0 = (dz')*Fdx(t) + (z')*(dFdx(t) - Fx(t)) + gₓ(t)     # Matrix form adjoint system
            # Expanded adjoint system residual equations:
            res[1]  = dz[1] - 2*dx(t,6)*z[1] + dx(t,3)*z[3] - 2*x(t,1)*z[5] - x(t,4)*z[6] - (x(t,2)*z[7])/(L^2)
            res[2]  = dz[2] - 2*dx(t,6)*z[2] + dx(t,3)*z[4] - 2*x(t,2)*z[5] - x(t,5)*z[6] + (x(t,1)*z[7])/(L^2)
            res[3]  = -x(t,1)*dz[3] - x(t,2)*dz[4] - dx(t,1)*z[3] - dx(t,2)*z[4]
            res[4]  = m*dz[3] + z[1] - 2*k*abs(x(t,4))*z[3] - x(t,1)*z[6]
            res[5]  = m*dz[4] + z[2] - 2*k*abs(x(t,5))*z[4] - x(t,2)*z[6]
            res[6]  = 2*x(t,1)*dz[1] + 2*x(t,2)*dz[2] + 2*dx(t,1)*z[1] + 2*dx(t,2)*z[2]
            res[7]  = (2*(x2(t,7) - y(t,1)))/T - z[7]
            # ---------- Disturbance adjoint system ---------- (This part is hard-coded and assumes nxw=2, nw=1)
            # 0 = dλₓᵀ + λᵀǍ + λwᵀČ                                 # Analytical form
            # 0 = dzₓ' + (zₓ')*Ǎ + (zw')*Č                          # Matrix form
            res[8]  = dz[8] + z[9] - ad.ρ[1]*z[8] + ad.ρ[3]*z[10]
            res[9]  = dz[9] - ad.ρ[2]*z[8] + ad.ρ[4]*z[10]
            # 0 = λwᵀ + λᵀFw                                        # Analytical form              
            # 0 = zw' + (z')*Fw(t)                                  # Matrix form
            res[10] = 2w(t)[1]*z[3] - z[10]
            # ---------- β-equations ----------
            # 0 = dβᵢ - gθᵢ + λᵀFθᵢ + λₓᵀ(Ǎθᵢ*xw(t) + B̌θᵢ*v(t)) + λwᵀČθᵢxw(t)                                     # Analytical form
            res[11]  = dz[11] - z[3]*abs(x(t,4))*x(t,4) - z[4]*abs(x(t,5))*x(t,5)                                 # For parameter k, only dβᵢ + λᵀFθᵢ
            res[12]  = dz[12] + z[8:9]⋅(ad.Ǎη[1:ad.nxw,:]*ad.xw(t) - ad.B̌η[1:ad.nxw,:]*ad.v(t)) + z[10]⋅(ad.Čη*ad.xw(t))        # For disturbance parameter, only  dβᵢ + λₓᵀ(Ǎθᵢ*xw(t) + B̌θᵢ*v(t)) + λwᵀČθᵢxw(t)

            nothing
        end
        
        # Indices 1-4 are differential (d), while 5-7 are algebraic (a)
        dinds  = 1:4
        ainds  = 5:7
        xwinds = 8:9
        ######################## INITIALIZING ADJOINT SYSTEM (Computing terminal values) ####################
        λT, dλT = get_initial_adjoint(dinds, ainds, gₓT, dgₓT, FxT, Fdx(T), dFdxT)
        λdistT, dλdistT = get_initial_adjoint_ODEdist(λT, ad.C, FwT)
        βT, dβT = get_initial_adjoint_beta(λT, FθT, zeros(nθ-ad.nη))
        βdistT, dβdistT = get_initial_adjoint_ODEdistbeta(λdistT[1:ad.nxw], λdistT[ad.nxw+1:end], ad, T, length(w(0.0)))
        z0 = vcat(λT, λdistT, βT, βdistT)
        dz0 = vcat(dλT, dλdistT, dβT, dβdistT)

        # Function returning Gp given adjoint solution
        function get_Gθ(adj_sol::DAESolution)
            # Because the problem is solved backward in time, the end of the solution corresponds to t=0
            # Gθᵢ = βᵢ(0) + (λᵀFdx xθᵢ)(0)
            Gθ = adj_sol.u[end][nx+ndist+1:nx+ndist+nθ] + (((adj_sol.u[end][1:nx]')*Fdx(0.0))*xθ0)[:]
        end

        dvars = fill(false, n+ndist+nθ)
        dvars[dinds] .= true
        dvars[xwinds] .= true
        dvars[end-nθ:end] .= true

        r0 = zeros(length(z0))
        f!(r0, dz0, z0, [], 0.0)
        # @info "r0 is: $r0 here, λ0: $λ0, dλ0: $dλ0"

        # t -> 0.0 is just a dummy function, not to be used
        return Model(f!, t -> 0.0, z0, dz0, dvars, r0), get_Gθ
    end
end

################################ PENDULUM INITIALIZATIONS ################################

function get_pendulum_initial(pars::Vector{Float64}, u0::Float64, w0::Float64, φ0::Float64)::Tuple{Vector{Float64}, Vector{Float64}}
    let m = pars[1], L = pars[2], g = pars[3], k = pars[4]
        x1_0 = L * sin(φ0)
        x2_0 = -L * cos(φ0)
        dx3_0 = (m*g*x2_0 - x1_0*(u0 + w0^2))/(L^2)
        dx4_0 = (dx3_0*x1_0 + u0 + w0^2)/m
        dx5_0 = (dx3_0*x2_0 - m*g)/m

        pend0 = vcat([x1_0, x2_0], zeros(4), [atan(x1_0 / -x2_0)])
        dpend0 = vcat([0., 0., dx3_0, dx4_0, dx5_0], zeros(2))
        pend0, dpend0
    end
end
 
function get_pendulum_initial_msens(pars::Vector{Float64}, u0::Float64, w0::Float64, φ0::Float64, x0::Vector{Float64}, dx0::Vector{Float64})::Tuple{Vector{Float64}, Vector{Float64}}
    let m = pars[1], L = pars[2], g = pars[3], k = pars[4]
        dsm3_0 = g*x0[2]/(L^2)
        dsm4_0 = -(dx0[3]*x0[1] + u0 + w0^2)/(m^2) + (dsm3_0*x0[1])/m
        dsm5_0 = -(dx0[3]*x0[2] - m*g)/(m^2) + (dsm3_0*x0[2] - g)/m
        dsm0 = vcat(zeros(2), [dsm3_0, dsm4_0, dsm5_0, 0., 0.])
        zeros(7), dsm0
    end
end

function get_pendulum_initial_gsens(pars::Vector{Float64}, u0::Float64, w0::Float64, φ0::Float64, x0::Vector{Float64}, dx0::Vector{Float64})::Tuple{Vector{Float64}, Vector{Float64}}
    let m = pars[1], L = pars[2], g = pars[3], k = pars[4]
        dsg3_0 = m*x0[2]/(L^2)
        dsg4_0 = dsg3_0*x0[1]/m
        dsg5_0 = dsg3_0*x0[2]/m - 1
        dsg0 = vcat(zeros(2), [dsg3_0, dsg4_0, dsg5_0], zeros(2))
        zeros(7), dsg0
    end
end

function get_pendulum_initial_Lsens(pars::Vector{Float64}, u0::Float64, w0::Float64, φ0::Float64, x0::Vector{Float64}, dx0::Vector{Float64})::Tuple{Vector{Float64}, Vector{Float64}}
    let m = pars[1], L = pars[2], g = pars[3], k = pars[4]
        dsL3_0 = (-m*g*cos(φ0) - sin(φ0)*(u0+w0^2))/(L^2) - 2*(m*g*x0[2] - x0[1]*(u0+w0^2))/(L^3)
        dsL4_0 = (dsL3_0*x0[1] + dx0[3]*sin(φ0))/m
        dsL5_0 = (dsL3_0*x0[2] - dx0[3]*cos(φ0))/m
        sL0  = vcat([sin(φ0), -cos(φ0)], zeros(5))
        dsL0 = vcat(zeros(2), [dsL3_0, dsL4_0, dsL5_0], zeros(2))
        sL0, dsL0
    end
end

function get_pendulum_initial_ksens(kwargs...)::Tuple{Vector{Float64}, Vector{Float64}}
    zeros(7), zeros(7)
end

# Assumes that pendulum initial conditions are independent of the disturbance parameters
function get_pendulum_initial_distsens(kwargs...)::Tuple{Vector{Float64}, Vector{Float64}}
    zeros(7), zeros(7)
end

# TODO: There's difference compred to thesis! 1. Check thesis derivation! 2. Check if changing it here affects performance!
function get_initial_adjoint(dinds::UnitRange{Int64}, ainds::UnitRange{Int64}, gₓT::Matrix{Float64}, dgₓT::Matrix{Float64}, FxT::Matrix{Float64}, FdxT::Matrix{Float64}, dFdxT::Matrix{Float64})::Tuple{Vector{Float64}, Vector{Float64}}
    λT  = zeros(length(dinds)+length(ainds))
    dλT = zeros(length(dinds)+length(ainds))
    temp = (-gₓT)/vcat(FdxT[dinds,:], -FxT[ainds,:])
    λT[dinds] = zeros(length(dinds))
    λT[ainds] = temp[ainds]
    dλT[dinds] = temp[dinds]                                                                    # Different compared to pg 92 in thesis!!!!!!!!!!!!!!!!!!!!!!!
    # temp = (-dgₓT + (dλT[dinds]')*(FxT[dinds,:] - dFdxT[dinds,:] - FdxT[dinds,:]) + (λT[ainds]')*FxT[ainds,:])/vcat(FdxT[dinds,:], -FxT[ainds,:])     # How I have had it in all experiments
    temp = (-dgₓT + (dλT[dinds]')*(FxT[dinds,:] - dFdxT[dinds,:] - FdxT[dinds,:]) + (λT[ainds]')*FdxT[ainds,:])/vcat(FdxT[dinds,:], -FxT[ainds,:])      # How it should be according to pg 92 in thesis!
    dλT[ainds] = temp[ainds]
    λT, dλT
end

function get_initial_adjoint_ODEdist(λT::Vector{Float64}, C::Matrix{Float64}, FwT::Matrix{Float64})::Tuple{Vector{Float64}, Vector{Float64}}
    λwT = (FwT')*λT
    dλxwT = -(λwT')*C
    dλwT = zeros(length(λwT))
    λxwT = zeros(length(dλxwT))
    vcat(λxwT, λwT[:]), vcat(dλxwT[:], dλwT)
end

function get_initial_adjoint_beta(λT::Vector{Float64}, FθT::Matrix{Float64}, gθ::Vector{Float64})::Tuple{Vector{Float64}, Vector{Float64}}
    # dβᵢ = gθ - λᵀFθ
    zeros(length(gθ)), gθ - ((λT')*FθT)[:]
end

function get_initial_adjoint_ODEdistbeta(λxwT::Vector{Float64}, λwT::Vector{Float64}, ad::AdjointSDEApproxData, T::Float64, nw::Int64)::Tuple{Vector{Float64}, Vector{Float64}}
    dβ = zeros(ad.nη)
    # dβᵢ = - λₓᵀ(Ǎηᵢ*xw + B̌ηᵢ*v) - λwᵀČηᵢ
    for ind=1:ad.nη
        Ǎηᵢ = ad.Ǎη[(ind-1)*ad.nxw+1:ind*ad.nxw,:]
        B̌ηᵢ = ad.B̌η[(ind-1)*ad.nxw+1:ind*ad.nxw,:]
        Čηᵢ = ad.Čη[(ind-1)*nw+1:ind*nw,:]
        dβ[ind] = (λxwT')*(Ǎηᵢ*ad.xw(T) + B̌ηᵢ*ad.v(T)) + (λwT')*Čηᵢ*ad.xw(T)
    end
    zeros(ad.nη), dβ
end

# TODO: Might need to generalize to multivariate case
function get_initial_adjoint_beta_dist(λT::Vector{Float64}, FwT::Matrix{Float64}, wθᵢT::Vector{Float64})::Tuple{Float64, Float64}
    # Assuming parameter parametrizes ONLY disturbance model, we have:
    # dβᵢ = - λᵀ(Fw wθᵢ)
    0.0, - (λT')*FwT*wθᵢT
end

################################ DELTA ROBOT MODELS ################################

end