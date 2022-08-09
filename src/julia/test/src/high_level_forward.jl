using DiffEqSensitivity, OrdinaryDiffEq, Zygote, ForwardDiff, DifferentialEquations#, ForwardSensitivity

# ======================= Defining useful variables ============================
N = 2000    # Number of samples of output computed
Ts = 0.01   # Time between samples

times = 0:Ts:(N*Ts)
# Model parameters
L = 6.25
m = 0.3
k = 6.25
g = 9.81

# ===========  Functions and structs for using high-level interface ============
struct Model
  f!::Function                  # residual function
  jac!::Function               # jacobian function
  x0::Array{Float64, 1}         # initial values of x
  xp0::Array{Float64, 1}        # initial values of x'
  dvars::Array{Bool, 1}         # bool array indicating differential variables
  ic_check::Array{Float64, 1}   # residual at t0 (DEBUG)
end

function pendulum_new(Φ::Float64, u::Function, w::Function, θ::Array{Float64, 1})::Model
    let m = θ[1], L = θ[2], g = θ[3], k = θ[4]

        # the residual function
        function f!(res, xp, x, θ, t)
            wt = w(t)
            ut = u(t)
            # Dynamic Equations
            res[1] = xp[1] - x[4] + 2xp[6]*x[1]
            res[2] = xp[2] - x[5] + 2xp[6]*x[2]
            res[3] = m*xp[4] - xp[3]*x[1] + k*abs(x[4])*x[4] - ut[1] - wt[1]^2
            res[4] = m*xp[5] - xp[3]*x[2] + k*abs(x[5])*x[5] + m*g
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
        x1_0 = L * sin(Φ)
        x2_0 = -L * cos(Φ)
        dx3_0 = m*g/x2_0
        dx4_0 = -g*tan(Φ) + (u0 + w0^2)/m

        x0 = vcat([x1_0, x2_0], zeros(4), [atan(x1_0 / -x2_0)])
        xp0 = vcat([0., 0., dx3_0, dx4_0], zeros(3))

        dvars = vcat(fill(true, 6), [false])

        r0 = zeros(length(x0))
        f!(r0, xp0, x0, [], 0.0)

        # t -> 0.0 is just a dummy function, not to be used
        Model(f!, t -> 0.0, x0, xp0, dvars, r0)
    end
end

function pendulum_sensitivity_full(Φ::Float64, u::Function, w::Function, θ::Array{Float64, 1})::Model
    let m = θ[1], L = θ[2], g = θ[3], k = θ[4]

        # the residual function
        function f!(res, xp, x, θ, t)
            wt = w(t)
            ut = u(t)
            # Dynamic Equations
            res[1] = xp[1] - x[4] + 2xp[6]*x[1]
            res[2] = xp[2] - x[5] + 2xp[6]*x[2]
            res[3] = m*xp[4] - xp[3]*x[1] + k*abs(x[4])*x[4] - ut[1] - wt[1]^2
            res[4] = m*xp[5] - xp[3]*x[2] + k*abs(x[5])*x[5] + m*g
            res[5] = x[1]^2 + x[2]^2 - L^2
            res[6] = x[4]*x[1] + x[5]*x[2]
            # Angle of pendulum
            res[7] = x[7] - atan(x[1] / -x[2])
            # Sensitivity with respect to m
            res[8]  = -x[11] + xp[8] + 2x[1]*xp[13] + 2x[8]*xp[6]
            res[9]  = -x[12] + xp[9] + 2x[2]*xp[13] + 2x[9]*xp[6]
            res[10] = 2k*x[11]*abs(x[4]) - x[1]*xp[10] + m*xp[11] - x[8]*xp[3] + xp[4]
            res[11] = 2k*x[12]*abs(x[5]) - x[2]*xp[10] + m*xp[12] - x[9]*xp[3] + g + xp[5]
            res[12] = -2x[8]*x[1] - 2x[9]*x[2]
            res[13] = x[11]*x[1] + x[12]*x[2] + x[8]*x[4] + x[9]*x[5]
            res[14] = x[14] - (x[1]*x[9] - x[2]*x[8])/(L^2)

            # Sensitivity with respect to L
            res[15]  = -x[18] + xp[15] + 2x[1]*xp[20] + 2x[15]*xp[6]
            res[16]  = -x[19] + xp[16] + 2x[2]*xp[20] + 2x[16]*xp[6]
            res[17] = 2k*x[18]*abs(x[4]) - x[1]*xp[17] + m*xp[18] - x[15]*xp[3]
            res[18] = 2k*x[19]*abs(x[5]) - x[2]*xp[17] + m*xp[19] - x[16]*xp[3]
            res[19] = -2x[15]*x[1] - 2x[16]*x[2] + 2L
            res[20] = x[18]*x[1] + x[19]*x[2] + x[15]*x[4] + x[16]*x[5]
            res[21] = x[21] - (x[1]*x[16] - x[2]*x[15])/(L^2)

            # Sensitivity with respect to g
            res[22]  = -x[25] + xp[22] + 2x[1]*xp[27] + 2x[22]*xp[6]
            res[23]  = -x[26] + xp[23] + 2x[2]*xp[27] + 2x[23]*xp[6]
            res[24] = 2k*x[25]*abs(x[4]) - x[1]*xp[24] + m*xp[25] - x[22]*xp[3]
            res[25] = 2k*x[26]*abs(x[5]) - x[2]*xp[24] + m*xp[26] - x[23]*xp[3] + m
            res[26] = -2x[22]*x[1] - 2x[23]*x[2]
            res[27] = x[25]*x[1] + x[26]*x[2] + x[22]*x[4] + x[23]*x[5]
            res[28] = x[28] - (x[1]*x[23] - x[2]*x[22])/(L^2)

            # Sensitivity with respect to k
            res[29]  = -x[32] + xp[29] + 2x[1]*xp[34] + 2x[29]*xp[6]
            res[30]  = -x[33] + xp[30] + 2x[2]*xp[34] + 2x[30]*xp[6]
            res[31] = 2k*x[32]*abs(x[4]) - x[1]*xp[31] + m*xp[32] - x[29]*xp[3] + abs(x[4])x[4]
            res[32] = 2k*x[33]*abs(x[5]) - x[2]*xp[31] + m*xp[33] - x[30]*xp[3] + abs(x[5])x[5]
            res[33] = -2x[29]*x[1] - 2x[30]*x[2]
            res[34] = x[32]*x[1] + x[33]*x[2] + x[29]*x[4] + x[30]*x[5]
            res[35] = x[35] - (x[1]*x[30] - x[2]*x[29])/(L^2)

            nothing
        end

        # Finding consistent initial conditions
        # Initial values, the pendulum starts at rest
        u0 = u(0.0)[1]
        w0 = w(0.0)[1]
        x1_0 = L * sin(Φ)
        x2_0 = -L * cos(Φ)
        dx3_0 = m*g/x2_0
        dx4_0 = -g*tan(Φ) + (u0 + w0^2)/m

        pend0 = vcat([x1_0, x2_0], zeros(4), [atan(x1_0 / -x2_0)])
        pendp0 = vcat([0., 0., dx3_0, dx4_0], zeros(3))
        sm0  = zeros(7)
        smp0 = vcat(zeros(3), [-dx4_0/m, -g/m, 0., 0.])
        sL0  = vcat([x1_0/L, x2_0/L], zeros(5))
        sLp0 = vcat([0.,0., -dx3_0/L], zeros(4))
        sg0  = zeros(7)
        sgp0 = vcat(zeros(4), [-1., 0., 0.])
        sk0  = zeros(7)
        skp0 = zeros(7)

        x0  = vcat(pend0, sm0, sL0, sg0, sk0)
        xp0 = vcat(pendp0, smp0, sLp0, sgp0, skp0)

        dvars = repeat(vcat(fill(true, 6), [false]), 5)

        r0 = zeros(length(x0))
        f!(r0, xp0, x0, [], 0.0)

        # t -> 0.0 is just a dummy function, not to be used
        Model(f!, t -> 0.0, x0, xp0, dvars, r0)
    end
end

const abstol = 1e-8
const reltol = 1e-5
const maxiters = Int64(1e8)

# ============= Solving equations using high-level interface ===================
Φ = pi/4
m = 0.3
L = 6.25
g = 9.81
k = 6.25
# θ = [m, L, g, k]
θ = [0.3, 6.25, 9.81, 6.25]
u(t) = 0.0
w(t) = 0.0

mdl = pendulum_new(Φ, u, w, θ)
prob = DAEProblem(mdl.f!, mdl.xp0, mdl.x0, (0, N*Ts), [], differential_vars=mdl.dvars)

mdl_s = pendulum_sensitivity_full(Φ, u, w, θ)
prob_s = DAEProblem(mdl_s.f!, mdl_s.xp0, mdl_s.x0, (0, N*Ts), [], differential_vars=mdl_s.dvars)

function super_solve_func(x0, p)
    prob_ = remake(prob, u0=x0, p=p)
    sol = solve(
      prob,
      saveat = times,
      abstol = abstol,
      reltol = reltol,
      maxiters = maxiters,
      sensealg = QuadratureAdjoint()
    )
    return sol.u[50][1]
end
du01,dp1 = Zygote.gradient(super_solve_func,mdl.x0,θ)

# sol = solve(
#   prob,
#   saveat = times,
#   abstol = abstol,
#   reltol = reltol,
#   maxiters = maxiters,
#   sensealg = ForwardSensitivity()
# )
#
# sol_s = solve(
#   prob_s,
#   saveat = times,
#   abstol = abstol,
#   reltol = reltol,
#   maxiters = maxiters
# )

nothing
