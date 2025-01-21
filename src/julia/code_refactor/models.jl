module DynamicalModels

import Interpolations
using DifferentialEquations: DAESolution

# TODO: Make it export everything that the user could potentially need! 
export Model, Model_ode, AdjointSDEApproxData, pendulum, pendulum_forward_m, pendulum_forward_k
# export delta_robot_gc, delta_robot_gc_allparsens, delta_robot_gc_foradj_allpar_alldist
# export get_delta_initial_L0sensonly, get_delta_initial_L1sensonly, get_delta_initial_L2sensonly, get_delta_initial_L3sensonly, get_delta_initial_LC1sensonly, get_delta_initial_LC2sensonly
# export get_delta_initial_M1sensonly, get_delta_initial_M2sensonly, get_delta_initial_M3sensonly, get_delta_initial_J1sensonly, get_delta_initial_J2sensonly, get_delta_initial_γsensonly
# export get_delta_initial_comp_with_mats
export get_pendulum_initial, get_pendulum_initial_msens, get_pendulum_initial_Lsens, get_pendulum_initial_ksens, get_pendulum_initial_distsens
export pendulum_adjoint_m, pendulum_adjoint_k_1a_ODEdist, pendulum_forward_k_1a

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
    B̃::AbstractMatrix{Float64}
    B̃ηa::AbstractMatrix{Float64}
    η::Vector{Float64}
    na::Int64           # Number of the disturbance parameters that correspond to the a-parameters
end

################################ PENDULUM MODELS ################################

# NOTE: Used to be called pendulum_new
function pendulum(u::Function, w::Function, θ::Vector{Float64}, model_data::NamedTuple)::Model
    let m = θ[1], L = θ[2], g = θ[3], k = θ[4]

        # the residual function
        function f!(res, dx, x, θ, t)
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
        # Initial values, the pendulum starts at rest
        u0 = u(0.0)[1]
        w0 = w(0.0)[1]
        x1_0 = L * sin(model_data.φ0)
        x2_0 = -L * cos(model_data.φ0)
        dx3_0 = m*g/x2_0    # This is actually hard-coded for vertical position of pendulum, i.e. model_data.φ0=0, and assuming u0+w^2=0
        dx4_0 = -g*tan(model_data.φ0) + (u0 + w0^2)/m

        x0 = vcat([x1_0, x2_0], zeros(4), [atan(x1_0 / -x2_0)])
        dx0 = vcat([0., 0., dx3_0, dx4_0], zeros(3))

        dvars = vcat(fill(true, 6), [false])

        r0 = zeros(length(x0))
        f!(r0, dx0, x0, [], 0.0)

        # t -> 0.0 is just a dummy function, not to be used
        Model(f!, t -> 0.0, x0, dx0, dvars, r0)
    end
end

# NOTE: Used to be called pendulum_sensitivity_m
function pendulum_forward_m(u::Function, w::Function, θ::Vector{Float64}, model_data::NamedTuple)::Model
    let m = θ[1], L = θ[2], g = θ[3], k = θ[4]

        # the residual function
        function f!(res, dx, x, θ, t)
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
        pend0, dpend0 = get_pendulum_initial(θ, u0, w0, model_data.φ0)
        sm0, dsm0 = get_pendulum_initial_msens(θ, u0, w0, model_data.φ0, pend0, dpend0)

        x0  = vcat(pend0, sm0)
        dx0 = vcat(dpend0, dsm0)

        dvars = vcat(fill(true, 6), [false], fill(true, 6), [false])

        r0 = zeros(length(x0))
        f!(r0, dx0, x0, [], 0.0)

        # t -> 0.0 is just a dummy function, not to be used
        Model(f!, t -> 0.0, x0, dx0, dvars, r0)
    end
end

function pendulum_forward_k(u::Function, w::Function, θ::Vector{Float64}, model_data::NamedTuple)::Model
    let m = θ[1], L = θ[2], g = θ[3], k = θ[4]

        # the residual function
        function f!(res, dx, x, θ, t)
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
        pend0, dpend0 = get_pendulum_initial(θ, u0, w0, model_data.φ0)
        sk0, dsk0 = get_pendulum_initial_ksens(θ, u0, w0, model_data.φ0, pend0, dpend0)
        x0  = vcat(pend0, sk0)
        dx0 = vcat(dpend0, dsk0)

        dvars = vcat(fill(true, 6), [false], fill(true, 6), [false])

        r0 = zeros(length(x0))
        f!(r0, dx0, x0, [], 0.0)

        # t -> 0.0 is just a dummy function, not to be used
        Model(f!, t -> 0.0, x0, dx0, dvars, r0)
    end
end

# NOTE: Used to be called my_pendulum_adjoint_monly
function pendulum_adjoint_m(u::Function, w::Function, θ::Vector{Float64}, model_data::NamedTuple, T::Float64, x::func_type, x2::func_type, y::func_type, dy::func_type, xp0::Matrix{Float64}, dx, dx2, N_trans::Int=0)
    # NOTE: A bit ugly to pass sol and sol2 as DAESolution, but dx as a function.
    # But good enough for now, just should be different in final version perhaps
    let m = θ[1], L = θ[2], g = θ[3], k = θ[4]
        np = size(xp0,2)
        @assert (np == 1) "my_pendulum_adjoint_monly is hard-coded to only handle one parameter m, make sure to pass correct xp0. Currently passing $np parameters."
        nx = size(xp0,1)
        # x  = t -> sol(t)
        # x2 = t -> sol2(t)

        Fx = t -> [2dx(t,6)        0               0   -1                  0           0   0
                    0           2*dx(t,6)          0   0                   -1          0   0
                    -dx(t,3)         0             0   2k*abs(x(t,4))     0           0   0
                    0            -dx(t,3)          0   0               2k*abs(x(t,5)) 0   0
                    2x(t,1)      2x(t,2)          0   0                   0           0   0
                    x(t,4)        x(t,5)          0   x(t,1)            x(t,2)      0   0
                    x(t,2)/(L^2)  -x(t,1)/(L^2)   0   0                   0           0   1]
        # (namely the derivative of F with respect to the variable x_p)
        Fdx = t -> vcat([1.0   0   0          0   0   2x(t,1)    0
                         0   1.0   0          0   0   2x(t,2)    0
                         0   0   -x(t,1)   m   0   0           0
                         0   0   -x(t,2)   0   m   0           0], zeros(3,7))
        Fddx = t -> vcat([  0   0  0            0   0   2dx(t,1)    0
                            0   0  0            0   0   2dx(t,2)    0
                            0   0  -dx(t,1)    0   0   0            0
                            0   0  -dx(t,2)    0   0   0            0], zeros(3,7))
        Fp = t -> [ .0
                    .0
                    dx(t,4)
                    dx(t,5)+g
                    .0
                    .0
                    .0]
        gₓ  = t -> [0    0    0    0    0    0    2(x2(t,7)-first(y(t,1)))/T]
        gdₓ = t -> [0    0    0    0    0    0    2(dx2(t,7)-first(dy(t,1)))/T]

        # NOTE: Convention is used that derivatives wrt to θ stack along cols
        # while derivatives wrt to x stack along rows

        ###################### INITIALIZING ADJOINT SYSTEM ####################
        # Indices 1-4 are differential (d), while 5-7 are algebraic (a)
        dinds = 1:4
        ainds = 5:7
        λT  = zeros(7)
        dλT = zeros(7)
        temp = (-gₓ(T))/vcat(Fdx(T)[dinds,:], -Fx(T)[ainds,:])
        λT[dinds] = zeros(length(dinds))
        λT[ainds] = temp[ainds]
        dλT[dinds] = temp[dinds]
        temp = (-gdₓ(T) + (dλT[dinds]')*(Fx(T)[dinds,:] - Fddx(T)[dinds,:] - Fdx(T)[dinds,:]) + (λT[ainds]')*Fx(T)[ainds,:])/vcat(Fdx(T)[dinds,:], -Fx(T)[ainds,:])
        dλT[ainds] = temp[ainds]

        # The residual function
        function f!(res, dz, z, θ, t)
            # Adjoint system
            # Completely unreadabe but most efficient version (still not a huge improvement)
            res[1]  = dz[1] - 2*dx(t,6)*z[1] + dx(t,3)*z[3] - 2*x(t,1)*z[5] - x(t,4)*z[6] - (x(t,2)*z[7])/(L^2)
            res[2]  = dz[2] - 2*dx(t,6)*z[2] + dx(t,3)*z[4] - 2*x(t,2)*z[5] - x(t,5)*z[6] + (x(t,1)*z[7])/(L^2)
            res[3]  = -x(t,1)*dz[3] - x(t,2)*dz[4] - dx(t,1)*z[3] - dx(t,2)*z[4]
            res[4]  = m*dz[3] + z[1] - 2*k*abs(x(t,4))*z[3] - x(t,1)*z[6]
            res[5]  = m*dz[4] + z[2] - 2*k*abs(x(t,5))*z[4] - x(t,2)*z[6]
            res[6]  = 2*x(t,1)*dz[1] + 2*x(t,2)*dz[2] + 2*dx(t,1)*z[1] + 2*dx(t,2)*z[2]
            res[7]  = (2*(x2(t,7) - first(y(t,1))))/T - z[7]
            # res[1]  = dz[1] + 2x(t,1]*dz[6]    - 2dx(t,6]*z[1] + z[4] + 2dx(t,1]*z[6]
            # res[2]  = dz[2] + 2x(t,2]*dz[6]    - 2dx(t,6]*z[2] + z[5] + 2dx(t,2]*z[6]
            # res[3]  = -x(t,1]*dz[3] + m*dz[4]  + dx(t,3]*z[1] - dx(t,1]*z[3] - 2k*abs(x(t,4])*z[4]
            # res[4]  = -x(t,2]*dz[3] + m*dz[5]  + dx(t,3]*z[2] - dx(t,2]*z[3] - 2k*abs(x(t,5])*z[5]
            # res[5]  =                           - 2x(t,1]*z[1] - 2x(t,2]*z[2]
            # res[6]  =                           - x(t,4]*z[1] - x(t,5]*z[2] - x(t,1]*z[4] - x(t,2]*z[5]
            # res[7]  =                           - x(t,2]*z[1]/(L^2) + x(t,1]*z[2]/(L^2) - z[7]                + (2/T)*(x2(t)[7]-y(t)[1])
            # β-equation
            res[8]  = dz[8] - z[3]*dx(t,4) - z[4]*(dx(t,5)+g)
            # res[8]  = dz[8] - z[3]*dx(T-t)[4] - z[4]*(dx(T-t)[5]+g)   # NOTE: SIMPLY WRONG; T-t????
            # res[8]  = dz[8] - z[3]*abs(x(t,4])*x(t,4] - z[4]*abs(x(t,5])*x(t,5]   # from pendelum k-only

            # # Super-readable but less efficient version ALSO NEGATED
            # res[1:7]  = (dz[1:7]')*Fdx(T-t) + (z[1:7]')*(Fddx(T-t) - Fx(T-t)) + gₓ(T-t)
            # res[8] = dz[8] - (Fp(T-t)')*z[1:7]
            nothing
        end

        z0  = vcat(λT[:], zeros(np))
        dz0 = vcat(dλT[:], -(Fp(T)')*λT)

        # Function returning Gp given adjoint solution
        function get_Gp(adj_sol::DAESolution)
            # NOTE: Changes signs to match what I had in my manual calculations, seems correct now
            # Gp = adj_sol.u[end][end-np+1:end] + (((adj_sol.u[end][1:end-np]')*Fdx(0.0))*xp0)[:]
            Gp = adj_sol.u[end][nx+1:nx+np] .+ (((adj_sol.u[end][1:nx]')*Fdx(0.0))*xp0)
            # return 0.0
        end

        function get_Gp_debug(adj_sol::DAESolution)
            integral = adj_sol.u[end][nx+1:nx+np]
            term = (((adj_sol.u[end][1:nx]')*Fdx(0.0))*xp0)
            Gp = integral .+ term
            return Gp, integral, term
        end

        function get_term_debug(adj_sol::DAESolution, xps::Matrix{Float64}, times::AbstractVector{Float64})
            term = zeros(length(adj_sol.u))
            for ind=eachindex(adj_sol.u)
                term[ind] = ((adj_sol.u[end+1-ind][1:nx]')*Fdx(times[ind]))*xps[:,ind]
            end
            return term
        end
        debugs = (get_Gp_debug, get_term_debug)

        dvars = vcat(fill(true, 4), fill(false, 3), fill(true, np))
        # dvars = vcat(fill(true, 4), fill(false, 3))

        r0 = zeros(length(z0))
        f!(r0, dz0, z0, [], 0.0)
        # @info "r0 is: $r0 here, λ0: $λ0, dλ0: $dλ0"

        # t -> 0.0 is just a dummy function, not to be used
        return Model(f!, t -> 0.0, z0, dz0, dvars, r0), get_Gp, debugs
    end
end

# TODO: Models that also returns debugs will probably crash!!!!
# TODO: FIX ASSERTION MESSAGES!
# function pendulum_forward_k_1a(u::Function, w::Function, θ::Vector{Float64}, model_data::NamedTuple, T::Float64, x::func_type, x2::func_type, y::func_type, dy::func_type, xp0::Matrix{Float64}, dx::func_type, dx2::func_type, N_trans::Int=0)
function pendulum_forward_k_1a(u::Function, w::Function, θ::Vector{Float64}, model_data::NamedTuple)::Model
    let m = θ[1], L = θ[2], g = θ[3], k = θ[4]

        # the residual function
        function f!(res, dx, x, θ, t)
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
            # TODO: Analytical formula says it should be x[1]^2+x[2]^2 instead of
            # L^2 (though they should be equal), is it fine to substitute L^2 here?
            res[21] = x[21] - (x[1]*x[16] - x[2]*x[15])/(L^2)

            nothing
        end

        u0 = u(0.0)[1]
        w0 = w(0.0)[1]
        pend0, dpend0 = get_pendulum_initial(θ, u0, w0, model_data.φ0)
        sk0, dsk0 = get_pendulum_initial_ksens(θ, u0, w0, model_data.φ0, pend0, dpend0)
        r0, dr0 = get_pendulum_initial_distsens(θ, u0, w0, model_data.φ0, pend0, dpend0)

        x0  = vcat(pend0, sk0, r0)
        dx0 = vcat(dpend0, dsk0, dr0)

        dvars = repeat(vcat(fill(true, 6), [false]), 3)

        r0 = zeros(length(x0))
        f!(r0, dx0, x0, [], 0.0)

        # t -> 0.0 is just a dummy function, not to be used
        Model(f!, t -> 0.0, x0, dx0, dvars, r0)
    end
end

# TODO: Does this model handle c-parameters as well as a-parameters? If yes, fix naming convention!!!!
# NOTE: Used to be called my_pendulum_adjoint_konly_with_distsensa1
# NOTE Assumes free dynamical parameters are only k. Also identifies a1/a2 from disturbance model
function pendulum_adjoint_k_1a_ODEdist(u::Function, w::Function, θ::Vector{Float64}, _::NamedTuple, T::Float64, x::func_type, x2::func_type, y::func_type, dy::func_type, xp0::Matrix{Float64}, dx::func_type, dx2::func_type, ad::AdjointSDEApproxData)#B̃::Matrix{Float64}, B̃θ::Matrix{Float64}, η::Vector{Float64}, na::Int=1, N_trans::Int=0)
    # NOTE: A bit ugly to pass sol and sol2 as DAESolution, but dx as a function. 
    # But good enough for now, just should be different in final version perhaps
    let m = θ[1], L = θ[2], g = θ[3], k = θ[4]
        np = size(xp0,2)
        nη = 1
        nx = size(xp0,1)
        nw = length(ad.xw(0.0))
        @assert (np == 2) "my_pendulum_adjoint_konly_with_distsensa is hard-coded to only handle parameters k, and a1. Make sure to pass correct xp0"
        ndist = 3   # length of x_w plus length of w (2+1)
        # x  = t -> sol(t)
        # x2 = t -> sol2(t)

        Fx = t -> [2dx(t,6)        0               0   -1                  0           0   0
                    0           2*dx(t,6)          0   0                   -1          0   0
                    -dx(t,3)         0             0   2k*abs(x(t,4))     0           0   0
                    0            -dx(t,3)          0   0               2k*abs(x(t,5)) 0   0
                    2x(t,1)      2x(t,2)          0   0                   0           0   0
                    x(t,4)        x(t,5)          0   x(t,1)            x(t,2)      0   0
                    x(t,2)/(L^2)  -x(t,1)/(L^2)   0   0                   0           0   1]
        # (namely the derivative of F with respect to the variable x_p)
        Fdx = t -> vcat([1   0   0          0   0   2x(t,1)    0
                         0   1   0          0   0   2x(t,2)    0
                         0   0   -x(t,1)   m   0   0           0
                         0   0   -x(t,2)   0   m   0           0], zeros(3,7))
        Fddx = t -> vcat([  0   0  0            0   0   2dx(t,1)    0
                            0   0  0            0   0   2dx(t,2)    0
                            0   0  -dx(t,1)    0   0   0            0
                            0   0  -dx(t,2)    0   0   0            0], zeros(3,7))
        Fp = t -> [.0; .0; abs(x(t,4))*x(t,4); abs(x(t,5))*x(t,5); .0; .0; .0]
        gₓ  = t -> [0    0    0    0    0    0    2(x2(t,7)-y(t,1))/T]
        gdₓ = t -> [0    0    0    0    0    0    2(dx2(t,7)-dy(t,1))/T]

        # NOTE: Convention is used that derivatives wrt to θ stack along cols
        # while derivatives wrt to x stack along rows

        # Creating some of the needed disturbance from η.
        A = [-ad.η[1]  -ad.η[2]; 1.0   0.0]
        C = [ad.η[3]   ad.η[4]]
        Aθ = [-1.0  0.0; 0.0    0.0; 0.0    -1.0; 0.0   0.0]
        Cθ = zeros(1,nw) # Only depends on C-parameters, and we only compute sensitivities wrt to a-parameters. We have 1 disturbance parameter, thus 1 row (I think, added this later)

        ###################### INITIALIZING ADJOINT SYSTEM ####################
        # Indices 1-4 are differential (d), while 5-7 are algebraic (a)
        dinds  = 1:4
        ainds  = 5:7
        λinds  = 1:7
        xwinds = 8:9
        winds  = 10
        λT  = zeros(10)
        dλT = zeros(10)
        temp = (-gₓ(T))/vcat(Fdx(T)[dinds,:], -Fx(T)[ainds,:])
        λT[dinds] = zeros(length(dinds))
        λT[ainds] = temp[ainds]
        dλT[dinds] = temp[dinds]
        temp = (-gdₓ(T) + (dλT[dinds]')*(Fx(T)[dinds,:] - Fddx(T)[dinds,:] - Fdx(T)[dinds,:]) + (λT[ainds]')*Fx(T)[ainds,:])/vcat(Fdx(T)[dinds,:], -Fx(T)[ainds,:])
        dλT[ainds] = temp[ainds]
        # Terminal conditions due to disturbance id
        λT[xwinds]  = zeros(length(xwinds))
        λT[winds]   = 2w(T)[1]*λT[3]
        dλT[xwinds] = (λT[xwinds]')*A + λT[winds]*C
        dλT[winds]  = 2w(T)[1]*dλT[3] + 2(C*(A*ad.xw(T)+ad.B̃*ad.v(T)))[1]*λT[3]

        # the residual function
        # NOTE: Could time-varying coefficients be the problem?? Sure would require more allocations?
        # TODO: If that is the case, we could store x(t) in a static array to avoid re-allocations?
        function f!(res, dz, z, θ, t)
            # Completely unreadabe but most efficient version (still not a huge improvement)
            res[1]  = dz[1] - 2*dx(t,6)*z[1] + dx(t,3)*z[3] - 2*x(t,1)*z[5] - x(t,4)*z[6] - (x(t,2)*z[7])/(L^2)
            res[2]  = dz[2] - 2*dx(t,6)*z[2] + dx(t,3)*z[4] - 2*x(t,2)*z[5] - x(t,5)*z[6] + (x(t,1)*z[7])/(L^2)
            res[3]  = -x(t,1)*dz[3] - x(t,2)*dz[4] - dx(t,1)*z[3] - dx(t,2)*z[4]
            res[4]  = m*dz[3] + z[1] - 2*k*abs(x(t,4))*z[3] - x(t,1)*z[6]
            res[5]  = m*dz[4] + z[2] - 2*k*abs(x(t,5))*z[4] - x(t,2)*z[6]
            res[6]  = 2*x(t,1)*dz[1] + 2*x(t,2)*dz[2] + 2*dx(t,1)*z[1] + 2*dx(t,2)*z[2]
            res[7]  = (2*(x2(t,7) - y(t,1)))/T - z[7]

            # NOTE: η here then has to contain free_parameters and true values for the non-free ones
            res[8]  = dz[8] + z[9] - ad.η[1]*z[8] + ad.η[3]*z[10]
            res[9]  = dz[9] - ad.η[2]*z[8] + ad.η[4]*z[10]
            res[10] = 2w(t)[1]*z[3] - z[10]
            # #          | ------------ from original adjoint system --------------- || ----------------------- new terms ---------------------| 
            # res[11]  = dz[8] - z[3]*abs(x(t)[4])*x(t)[4] - z[4]*abs(x(t)[5])*x(t)[5]  + (z[8:9]')*(Aθ*xw(t) - B̃θ*v(t)) + z[10]*(Cθ*xw(t))
            res[11]  = dz[11] - z[3]*abs(x(t,4))*x(t,4) - z[4]*abs(x(t,5))*x(t,5)   # I don't remember anymore, but those "new terms" above have zero contribution to this line I think
            # @info "Let's check some types and sizes: $(typeof((z[8:9]')*(Aθ[1:nw,:]*ad.xw(t) - ad.B̃ηa[1:nw,:]*ad.v(t)))), size: $(size((z[8:9]')*(Aθ[1:nw,:]*ad.xw(t) - ad.B̃ηa[1:nw,:]*ad.v(t))))"
            # @info "prt2: $(typeof((Aθ[1:nw,:]*ad.xw(t) - ad.B̃ηa[1:nw,:]*ad.v(t)))), $(size((Aθ[1:nw,:]*ad.xw(t) - ad.B̃ηa[1:nw,:]*ad.v(t))))"
            res[12]  = dz[12] + (z[8:9]')*(Aθ[1:nw,:]*ad.xw(t) - ad.B̃ηa[1:nw,:]*ad.v(t)) # + z[10]*(Cθ*xw(t)) # This last part is equal to zero since we don't have C-parameters
            # res[13]  = dz[13] + (z[8:9]')*(Aθ[nw+1:2nw,:]*xw(t) - B̃θ[nw+1:2nw,:]*v(t))

            # # Super-readable but less efficient version
            # res[1:7] = (dz[1:7]')*Fdx(t) + (z[1:7]')*(Fddx(t) - Fx(t)) + gₓ(t)
            # res[8]   = dz[8] - (z[1:7]')*Fp(t)
            nothing
        end
        
        z0  = vcat(λT[:], zeros(np))
        second_temp = Matrix{Float64}(undef, nw, nη)
        third_temp  = Matrix{Float64}(undef, 1, nη)
        for ind = 1:nη   # 2 disturbance parameters
            second_temp[:, ind] = Aθ[(ind-1)nw + 1: ind*nw, :]*ad.xw(T) + ad.B̃ηa[(ind-1)nw + 1: ind*nw, :]*ad.v(T)
            third_temp[:, ind]  = Cθ[ind:ind, :]*ad.xw(T) # NOTE: SCALAR_OUTPUT is assumed
            # second_temp[:, (ind-1)nw + 1: ind*nw] = Aθ[(ind-1)nw + 1: ind*nw, :]*xw(T) + B̃θ[(ind-1)nw + 1: ind*nw, :]*v(T)
            # third_temp[:, (ind-1)nw + 1: ind]  = Cθ[ind:ind, :]*xw(T) # NOTE: SCALAR_OUTPUT is assumed
        end
        my_temp = hcat([(λT[λinds]')*Fp(T)], - (λT[xwinds]')*second_temp - (λT[winds]')*third_temp)
        dz0 = vcat(dλT[:], my_temp[:])
        # dz0 = vcat(dλT[:], (λT[λinds]')*Fp(T) - (λT[xwinds]')*second_temp - (λT[winds]')*third_temp)   # For some reason (λT')*Fp(T) returns a scalar instead of a 1-element matrix, unexpected but desired

        # Function returning Gp given adjoint solution
        function get_Gp(adj_sol::DAESolution)
            Gp = adj_sol.u[end][nx+ndist+1:nx+ndist+np] + (((adj_sol.u[end][1:nx]')*Fdx(0.0))*xp0)[:]     # This is the same as in original adjoint system just because disturbance model has zero initial conditions, independent of parameters
        end

        dvars = vcat(fill(true, 4), fill(false, 3), fill(true, 2), [false], fill(true, np))
        # dvars = vcat(fill(true, 4), fill(false, 3))

        r0 = zeros(length(z0))
        f!(r0, dz0, z0, [], 0.0)
        # @info "r0 is: $r0 here, λ0: $λ0, dλ0: $dλ0"

        # t -> 0.0 is just a dummy function, not to be used
        return Model(f!, t -> 0.0, z0, dz0, dvars, r0), get_Gp
    end
end

# FORADJ PENDULUM NOW! Hopefully EZ. Okay maybe a tomorrow thing, test the others for now
# Everything with this function is wrong (placeholder), except for the name
function pendulum_adjoint_k_1a(u::Function, w::Function, θ::Vector{Float64}, _::NamedTuple, T::Float64, x::func_type, x2::func_type, y::func_type, dy::func_type, xp0::Matrix{Float64}, dx::func_type, dx2::func_type, ad::AdjointSDEApproxData)#B̃::Matrix{Float64}, B̃θ::Matrix{Float64}, η::Vector{Float64}, na::Int=1, N_trans::Int=0)
    # NOTE: A bit ugly to pass sol and sol2 as DAESolution, but dx as a function. 
    # But good enough for now, just should be different in final version perhaps
    let m = θ[1], L = θ[2], g = θ[3], k = θ[4]
        np = size(xp0,2)
        nη = 1
        nx = size(xp0,1)
        nw = length(ad.xw(0.0))
        @assert (np == 2) "my_pendulum_adjoint_konly_with_distsensa is hard-coded to only handle parameters k, and a1. Make sure to pass correct xp0"
        ndist = 3   # length of x_w plus length of w (2+1)
        # x  = t -> sol(t)
        # x2 = t -> sol2(t)

        Fx = t -> [2dx(t,6)        0               0   -1                  0           0   0
                    0           2*dx(t,6)          0   0                   -1          0   0
                    -dx(t,3)         0             0   2k*abs(x(t,4))     0           0   0
                    0            -dx(t,3)          0   0               2k*abs(x(t,5)) 0   0
                    2x(t,1)      2x(t,2)          0   0                   0           0   0
                    x(t,4)        x(t,5)          0   x(t,1)            x(t,2)      0   0
                    x(t,2)/(L^2)  -x(t,1)/(L^2)   0   0                   0           0   1]
        # (namely the derivative of F with respect to the variable x_p)
        Fdx = t -> vcat([1   0   0          0   0   2x(t,1)    0
                         0   1   0          0   0   2x(t,2)    0
                         0   0   -x(t,1)   m   0   0           0
                         0   0   -x(t,2)   0   m   0           0], zeros(3,7))
        Fddx = t -> vcat([  0   0  0            0   0   2dx(t,1)    0
                            0   0  0            0   0   2dx(t,2)    0
                            0   0  -dx(t,1)    0   0   0            0
                            0   0  -dx(t,2)    0   0   0            0], zeros(3,7))
        Fp = t -> [.0; .0; abs(x(t,4))*x(t,4); abs(x(t,5))*x(t,5); .0; .0; .0]
        gₓ  = t -> [0    0    0    0    0    0    2(x2(t,7)-y(t,1))/T]
        gdₓ = t -> [0    0    0    0    0    0    2(dx2(t,7)-dy(t,1))/T]

        # NOTE: Convention is used that derivatives wrt to θ stack along cols
        # while derivatives wrt to x stack along rows

        # Creating some of the needed disturbance from η.
        A = [-ad.η[1]  -ad.η[2]; 1.0   0.0]
        C = [ad.η[3]   ad.η[4]]
        Aθ = [-1.0  0.0; 0.0    0.0; 0.0    -1.0; 0.0   0.0]
        Cθ = zeros(1,nw) # Only depends on C-parameters, and we only compute sensitivities wrt to a-parameters. We have 1 disturbance parameter, thus 1 row (I think, added this later)

        ###################### INITIALIZING ADJOINT SYSTEM ####################
        # Indices 1-4 are differential (d), while 5-7 are algebraic (a)
        dinds  = 1:4
        ainds  = 5:7
        λinds  = 1:7
        xwinds = 8:9
        winds  = 10
        λT  = zeros(10)
        dλT = zeros(10)
        temp = (-gₓ(T))/vcat(Fdx(T)[dinds,:], -Fx(T)[ainds,:])
        λT[dinds] = zeros(length(dinds))
        λT[ainds] = temp[ainds]
        dλT[dinds] = temp[dinds]
        temp = (-gdₓ(T) + (dλT[dinds]')*(Fx(T)[dinds,:] - Fddx(T)[dinds,:] - Fdx(T)[dinds,:]) + (λT[ainds]')*Fx(T)[ainds,:])/vcat(Fdx(T)[dinds,:], -Fx(T)[ainds,:])
        dλT[ainds] = temp[ainds]
        # Terminal conditions due to disturbance id
        λT[xwinds]  = zeros(length(xwinds))
        λT[winds]   = 2w(T)[1]*λT[3]
        dλT[xwinds] = (λT[xwinds]')*A + λT[winds]*C
        dλT[winds]  = 2w(T)[1]*dλT[3] + 2(C*(A*ad.xw(T)+ad.B̃*ad.v(T)))[1]*λT[3]

        # the residual function
        # NOTE: Could time-varying coefficients be the problem?? Sure would require more allocations?
        # TODO: If that is the case, we could store x(t) in a static array to avoid re-allocations?
        function f!(res, dz, z, θ, t)
            # Completely unreadabe but most efficient version (still not a huge improvement)
            res[1]  = dz[1] - 2*dx(t,6)*z[1] + dx(t,3)*z[3] - 2*x(t,1)*z[5] - x(t,4)*z[6] - (x(t,2)*z[7])/(L^2)
            res[2]  = dz[2] - 2*dx(t,6)*z[2] + dx(t,3)*z[4] - 2*x(t,2)*z[5] - x(t,5)*z[6] + (x(t,1)*z[7])/(L^2)
            res[3]  = -x(t,1)*dz[3] - x(t,2)*dz[4] - dx(t,1)*z[3] - dx(t,2)*z[4]
            res[4]  = m*dz[3] + z[1] - 2*k*abs(x(t,4))*z[3] - x(t,1)*z[6]
            res[5]  = m*dz[4] + z[2] - 2*k*abs(x(t,5))*z[4] - x(t,2)*z[6]
            res[6]  = 2*x(t,1)*dz[1] + 2*x(t,2)*dz[2] + 2*dx(t,1)*z[1] + 2*dx(t,2)*z[2]
            res[7]  = (2*(x2(t,7) - y(t,1)))/T - z[7]

            # NOTE: η here then has to contain free_parameters and true values for the non-free ones
            res[8]  = dz[8] + z[9] - ad.η[1]*z[8] + ad.η[3]*z[10]
            res[9]  = dz[9] - ad.η[2]*z[8] + ad.η[4]*z[10]
            res[10] = 2w(t)[1]*z[3] - z[10]
            # #          | ------------ from original adjoint system --------------- || ----------------------- new terms ---------------------| 
            # res[11]  = dz[8] - z[3]*abs(x(t)[4])*x(t)[4] - z[4]*abs(x(t)[5])*x(t)[5]  + (z[8:9]')*(Aθ*xw(t) - B̃θ*v(t)) + z[10]*(Cθ*xw(t))
            res[11]  = dz[11] - z[3]*abs(x(t,4))*x(t,4) - z[4]*abs(x(t,5))*x(t,5)   # I don't remember anymore, but those "new terms" above have zero contribution to this line I think
            # @info "Let's check some types and sizes: $(typeof((z[8:9]')*(Aθ[1:nw,:]*ad.xw(t) - ad.B̃ηa[1:nw,:]*ad.v(t)))), size: $(size((z[8:9]')*(Aθ[1:nw,:]*ad.xw(t) - ad.B̃ηa[1:nw,:]*ad.v(t))))"
            # @info "prt2: $(typeof((Aθ[1:nw,:]*ad.xw(t) - ad.B̃ηa[1:nw,:]*ad.v(t)))), $(size((Aθ[1:nw,:]*ad.xw(t) - ad.B̃ηa[1:nw,:]*ad.v(t))))"
            res[12]  = dz[12] + (z[8:9]')*(Aθ[1:nw,:]*ad.xw(t) - ad.B̃ηa[1:nw,:]*ad.v(t)) # + z[10]*(Cθ*xw(t)) # This last part is equal to zero since we don't have C-parameters
            # res[13]  = dz[13] + (z[8:9]')*(Aθ[nw+1:2nw,:]*xw(t) - B̃θ[nw+1:2nw,:]*v(t))

            # # Super-readable but less efficient version
            # res[1:7] = (dz[1:7]')*Fdx(t) + (z[1:7]')*(Fddx(t) - Fx(t)) + gₓ(t)
            # res[8]   = dz[8] - (z[1:7]')*Fp(t)
            nothing
        end
        
        z0  = vcat(λT[:], zeros(np))
        second_temp = Matrix{Float64}(undef, nw, nη)
        third_temp  = Matrix{Float64}(undef, 1, nη)
        for ind = 1:nη   # 2 disturbance parameters
            second_temp[:, ind] = Aθ[(ind-1)nw + 1: ind*nw, :]*ad.xw(T) + ad.B̃ηa[(ind-1)nw + 1: ind*nw, :]*ad.v(T)
            third_temp[:, ind]  = Cθ[ind:ind, :]*ad.xw(T) # NOTE: SCALAR_OUTPUT is assumed
            # second_temp[:, (ind-1)nw + 1: ind*nw] = Aθ[(ind-1)nw + 1: ind*nw, :]*xw(T) + B̃θ[(ind-1)nw + 1: ind*nw, :]*v(T)
            # third_temp[:, (ind-1)nw + 1: ind]  = Cθ[ind:ind, :]*xw(T) # NOTE: SCALAR_OUTPUT is assumed
        end
        my_temp = hcat([(λT[λinds]')*Fp(T)], - (λT[xwinds]')*second_temp - (λT[winds]')*third_temp)
        dz0 = vcat(dλT[:], my_temp[:])
        # dz0 = vcat(dλT[:], (λT[λinds]')*Fp(T) - (λT[xwinds]')*second_temp - (λT[winds]')*third_temp)   # For some reason (λT')*Fp(T) returns a scalar instead of a 1-element matrix, unexpected but desired

        # Function returning Gp given adjoint solution
        function get_Gp(adj_sol::DAESolution)
            Gp = adj_sol.u[end][nx+ndist+1:nx+ndist+np] + (((adj_sol.u[end][1:nx]')*Fdx(0.0))*xp0)[:]     # This is the same as in original adjoint system just because disturbance model has zero initial conditions, independent of parameters
        end

        dvars = vcat(fill(true, 4), fill(false, 3), fill(true, 2), [false], fill(true, np))
        # dvars = vcat(fill(true, 4), fill(false, 3))

        r0 = zeros(length(z0))
        f!(r0, dz0, z0, [], 0.0)
        # @info "r0 is: $r0 here, λ0: $λ0, dλ0: $dλ0"

        # t -> 0.0 is just a dummy function, not to be used
        return Model(f!, t -> 0.0, z0, dz0, dvars, r0), get_Gp
    end
end


################################ PENDULUM INITIALIZATIONS ################################

function get_pendulum_initial(θ::Vector{Float64}, u0::Float64, w0::Float64, φ0::Float64)
    let m = θ[1], L = θ[2], g = θ[3], k = θ[4]
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
 
function get_pendulum_initial_msens(θ::Vector{Float64}, u0::Float64, w0::Float64, φ0::Float64, x0::Vector{Float64}, dx0::Vector{Float64})
    let m = θ[1], L = θ[2], g = θ[3], k = θ[4]
        dsm3_0 = g*x0[2]/(L^2)
        dsm4_0 = -(dx0[3]*x0[1] + u0 + w0^2)/(m^2) + (dsm3_0*x0[1])/m
        dsm5_0 = -(dx0[3]*x0[2] - m*g)/(m^2) + (dsm3_0*x0[2] - g)/m
        dsm0 = vcat(zeros(2), [dsm3_0, dsm4_0, dsm5_0, 0., 0.])
        zeros(7), dsm0
    end
end

function get_pendulum_initial_gsens(θ::Vector{Float64}, u0::Float64, w0::Float64, φ0::Float64, x0::Vector{Float64}, dx0::Vector{Float64})
    let m = θ[1], L = θ[2], g = θ[3], k = θ[4]
        dsg3_0 = m*x0[2]/(L^2)
        dsg4_0 = dsg3_0*x0[1]/m
        dsg5_0 = dsg3_0*x0[2]/m - 1
        dsg0 = vcat(zeros(2), [dsg3_0, dsg4_0, dsg5_0], zeros(2))
        zeros(7), dsg0
    end
end

function get_pendulum_initial_Lsens(θ::Vector{Float64}, u0::Float64, w0::Float64, φ0::Float64, x0::Vector{Float64}, dx0::Vector{Float64})
    let m = θ[1], L = θ[2], g = θ[3], k = θ[4]
        dsL3_0 = (-m*g*cos(φ0) - sin(φ0)*(u0+w0^2))/(L^2) - 2*(m*g*x0[2] - x0[1]*(u0+w0^2))/(L^3)
        dsL4_0 = (dsL3_0*x0[1] + dx0[3]*sin(φ0))/m
        dsL5_0 = (dsL3_0*x0[2] - dx0[3]*cos(φ0))/m
        sL0  = vcat([sin(φ0), -cos(φ0)], zeros(5))
        dsL0 = vcat(zeros(2), [dsL3_0, dsL4_0, dsL5_0], zeros(2))
        sL0, dsL0
    end
end

function get_pendulum_initial_ksens(kwargs...)
    zeros(7), zeros(7)
end

# Assumes that pendulum initial conditions are independent of the disturbance parameters
function get_pendulum_initial_distsens(kwargs...)
    zeros(7), zeros(7)
end

################################ DELTA ROBOT MODELS ################################

end