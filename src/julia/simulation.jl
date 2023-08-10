import Interpolations.CubicSplineInterpolation
import Interpolations.LinearInterpolation
import Random

using DifferentialEquations
using Sundials
using DelimitedFiles
using Plots
using ProgressMeter

# include("noise_model.jl")

struct Model
  f!::Function                  # residual function
  jac!::Function               # jacobian function
  x0::Array{Float64, 1}         # initial values of x
  dx0::Array{Float64, 1}        # initial values of x'
  dvars::Array{Bool, 1}         # bool array indicating differential variables
  ic_check::Array{Float64, 1}   # residual at t0 (DEBUG)
end

struct Model_ode
    f::Function             # ODE Function
    x0::Array{Float64, 1}   # initial values of x
end

function interpolation(T::Float64, xs::Array{Float64, 1})
  let ts = range(0, T, length=length(xs))
    LinearInterpolation(ts, xs)
  end
end

function pendulum_new(Φ::Float64, u::Function, w::Function, θ::Array{Float64, 1})::Model
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
        x1_0 = L * sin(Φ)
        x2_0 = -L * cos(Φ)
        dx3_0 = m*g/x2_0
        dx4_0 = -g*tan(Φ) + (u0 + w0^2)/m

        x0 = vcat([x1_0, x2_0], zeros(4), [atan(x1_0 / -x2_0)])
        dx0 = vcat([0., 0., dx3_0, dx4_0], zeros(3))

        dvars = vcat(fill(true, 6), [false])

        r0 = zeros(length(x0))
        f!(r0, dx0, x0, [], 0.0)

        # t -> 0.0 is just a dummy function, not to be used
        Model(f!, t -> 0.0, x0, dx0, dvars, r0)
    end
end

# Sensitivity only with respect to disturbance parameter, no dynamic parameter
function pendulum_dist_sens_1(Φ::Float64, u::Function, w::Function, θ::Array{Float64, 1})::Model
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

            # Sensitivty wrt first disturbance parameter
            # (i.e. parameter corresponding to wt[2])
            res[8] = 2dx[6]x[8] - x[11] + dx[8] + 2x[1]dx[13]
            res[9] = 2dx[6]x[9] - x[12] + dx[9] + 2x[2]dx[13]
            res[10] = -dx[3]x[8] + 2k*abs(x[4])*x[11] - x[1]*dx[10] + m*dx[11] - 2*wt[1]*wt[2]
            res[11] = -dx[3]x[9] + 2k*abs(x[5])*x[12] - x[2]*dx[10] + m*dx[12]
            res[12] = 2x[1]x[8] + 2x[2]x[9]
            res[13] = x[4]x[8] + x[5]x[9] + x[1]x[11] + x[2]x[12]
            # Sensitivity of angle of pendulum to disturbance parameter
            # TODO: Analytical formula says it should be x[1]^2+x[2]^2 instead of
            # L^2 (though they should be equal), is it fine to substitute L^2 here?
            res[14] = x[14] - (x[1]*x[9] - x[2]*x[8])/(L^2)

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
        dpend0 = vcat([0., 0., dx3_0, dx4_0], zeros(3))
        r0 = zeros(7)
        rp0 = zeros(7)

        x0  = vcat(pend0, r0)
        dx0 = vcat(dpend0, rp0)
        # x0 = vcat([x1_0, x2_0], zeros(4), [atan(x1_0 / -x2_0)], zeros(14))
        # dx0 = vcat([0., 0., dx3_0, dx4_0], zeros(17))

        dvars = repeat(vcat(fill(true, 6), [false]), 2)

        r0 = zeros(length(x0))
        f!(r0, dx0, x0, [], 0.0)

        # t -> 0.0 is just a dummy function, not to be used
        Model(f!, t -> 0.0, x0, dx0, dvars, r0)
    end
end

# Sensitivity with respect to two disturbance parameters, no dynamic parameter
function pendulum_dist_sens_2(Φ::Float64, u::Function, w::Function, θ::Array{Float64, 1})::Model
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

            # Sensitivty wrt first disturbance parameter
            # (i.e. parameter corresponding to wt[2])
            res[8] = 2dx[6]x[8] - x[11] + dx[8] + 2x[1]dx[13]
            res[9] = 2dx[6]x[9] - x[12] + dx[9] + 2x[2]dx[13]
            res[10] = -dx[3]x[8] + 2k*abs(x[4])*x[11] - x[1]*dx[10] + m*dx[11] - 2*wt[1]*wt[2]
            res[11] = -dx[3]x[9] + 2k*abs(x[5])*x[12] - x[2]*dx[10] + m*dx[12]
            res[12] = 2x[1]x[8] + 2x[2]x[9]
            res[13] = x[4]x[8] + x[5]x[9] + x[1]x[11] + x[2]x[12]
            # Sensitivity of angle of pendulum to disturbance parameter
            # TODO: Analytical formula says it should be x[1]^2+x[2]^2 instead of
            # L^2 (though they should be equal), is it fine to substitute L^2 here?
            res[14] = x[14] - (x[1]*x[9] - x[2]*x[8])/(L^2)

            # Sensitivty wrt second disturbance parameter
            # (i.e. parameter corresponding to wt[2])
            res[15] = 2dx[6]x[15] - x[18] + dx[15] + 2x[1]dx[20]
            res[16] = 2dx[6]x[16] - x[19] + dx[16] + 2x[2]dx[20]
            res[17] = -dx[3]x[15] + 2k*abs(x[4])*x[18] - x[1]*dx[17] + m*dx[18] - 2*wt[1]*wt[3]
            res[18] = -dx[3]x[16] + 2k*abs(x[5])*x[19] - x[2]*dx[17] + m*dx[19]
            res[19] = 2x[1]x[15] + 2x[2]x[16]
            res[20] = x[4]x[15] + x[5]x[16] + x[1]x[18] + x[2]x[19]
            # Sensitivity of angle of pendulum to disturbance parameter
            # TODO: Analytical formula says it should be x[1]^2+x[2]^2 instead of
            # L^2 (though they should be equal), is it fine to substitute L^2 here?
            res[21] = x[21] - (x[1]*x[16] - x[2]*x[15])/(L^2)

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
        dpend0 = vcat([0., 0., dx3_0, dx4_0], zeros(3))
        r0 = zeros(7)
        rp0 = zeros(7)

        x0  = vcat(pend0, r0, r0)
        dx0 = vcat(dpend0, rp0, rp0)
        # x0 = vcat([x1_0, x2_0], zeros(4), [atan(x1_0 / -x2_0)], zeros(14))
        # dx0 = vcat([0., 0., dx3_0, dx4_0], zeros(17))

        dvars = repeat(vcat(fill(true, 6), [false]), 3)

        r0 = zeros(length(x0))
        f!(r0, dx0, x0, [], 0.0)

        # t -> 0.0 is just a dummy function, not to be used
        Model(f!, t -> 0.0, x0, dx0, dvars, r0)
    end
end

# Sensitivity with respect to "all" disturbance parameters, no dynamic parameter
function pendulum_dist_sens_3(Φ::Float64, u::Function, w::Function, θ::Array{Float64, 1})::Model
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

            # Sensitivty wrt first disturbance parameter
            # (i.e. parameter corresponding to wt[2])
            res[8] = 2dx[6]x[8] - x[11] + dx[8] + 2x[1]dx[13]
            res[9] = 2dx[6]x[9] - x[12] + dx[9] + 2x[2]dx[13]
            res[10] = -dx[3]x[8] + 2k*abs(x[4])*x[11] - x[1]*dx[10] + m*dx[11] - 2*wt[1]*wt[2]
            res[11] = -dx[3]x[9] + 2k*abs(x[5])*x[12] - x[2]*dx[10] + m*dx[12]
            res[12] = 2x[1]x[8] + 2x[2]x[9]
            res[13] = x[4]x[8] + x[5]x[9] + x[1]x[11] + x[2]x[12]
            # Sensitivity of angle of pendulum to disturbance parameter
            # TODO: Analytical formula says it should be x[1]^2+x[2]^2 instead of
            # L^2 (though they should be equal), is it fine to substitute L^2 here?
            res[14] = x[14] - (x[1]*x[9] - x[2]*x[8])/(L^2)

            # Sensitivty wrt second disturbance parameter
            # (i.e. parameter corresponding to wt[2])
            res[15] = 2dx[6]x[15] - x[18] + dx[15] + 2x[1]dx[20]
            res[16] = 2dx[6]x[16] - x[19] + dx[16] + 2x[2]dx[20]
            res[17] = -dx[3]x[15] + 2k*abs(x[4])*x[18] - x[1]*dx[17] + m*dx[18] - 2*wt[1]*wt[3]
            res[18] = -dx[3]x[16] + 2k*abs(x[5])*x[19] - x[2]*dx[17] + m*dx[19]
            res[19] = 2x[1]x[15] + 2x[2]x[16]
            res[20] = x[4]x[15] + x[5]x[16] + x[1]x[18] + x[2]x[19]
            # Sensitivity of angle of pendulum to disturbance parameter
            # TODO: Analytical formula says it should be x[1]^2+x[2]^2 instead of
            # L^2 (though they should be equal), is it fine to substitute L^2 here?
            res[21] = x[21] - (x[1]*x[16] - x[2]*x[15])/(L^2)

            # Sensitivty wrt third disturbance parameter
            # (i.e. parameter corresponding to wt[2])
            res[22] = 2dx[6]x[22] - x[25] + dx[22] + 2x[1]dx[27]
            res[23] = 2dx[6]x[23] - x[26] + dx[23] + 2x[2]dx[27]
            res[24] = -dx[3]x[22] + 2k*abs(x[4])*x[25] - x[1]*dx[24] + m*dx[25] - 2*wt[1]*wt[4]
            res[25] = -dx[3]x[23] + 2k*abs(x[5])*x[26] - x[2]*dx[24] + m*dx[26]
            res[26] = 2x[1]x[22] + 2x[2]x[23]
            res[27] = x[4]x[22] + x[5]x[23] + x[1]x[25] + x[2]x[26]
            # Sensitivity of angle of pendulum to disturbance parameter
            # TODO: Analytical formula says it should be x[1]^2+x[2]^2 instead of
            # L^2 (though they should be equal), is it fine to substitute L^2 here?
            res[28] = x[28] - (x[1]*x[23] - x[2]*x[22])/(L^2)

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
        dpend0 = vcat([0., 0., dx3_0, dx4_0], zeros(3))
        r0 = zeros(7)
        rp0 = zeros(7)

        x0  = vcat(pend0, r0, r0, r0)
        dx0 = vcat(dpend0, rp0, rp0, rp0)
        # x0 = vcat([x1_0, x2_0], zeros(4), [atan(x1_0 / -x2_0)], zeros(14))
        # dx0 = vcat([0., 0., dx3_0, dx4_0], zeros(17))

        dvars = repeat(vcat(fill(true, 6), [false]), 4)

        r0 = zeros(length(x0))
        f!(r0, dx0, x0, [], 0.0)

        # t -> 0.0 is just a dummy function, not to be used
        Model(f!, t -> 0.0, x0, dx0, dvars, r0)
    end
end

function pendulum_sensitivity_k(Φ::Float64, u::Function, w::Function, θ::Array{Float64, 1})::Model
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
            # TODO: Analytical formula says it should be x[1]^2+x[2]^2 instead of
            # L^2 (though they should be equal), is it fine to substitute L^2 here?
            res[14] = x[14] - (x[1]*x[9] - x[8]*x[2])/(L^2)

            # These equations sensitivity are written on the intuitive form. To
            # obtain form correct for this funciton, do the following replacements:
            # s[1] -> x[8],  s[2] -> x[9],  s[3] -> x[10]
            # s[4] -> x[11], s[5] -> x[12], s[6] -> x[13]
            # and the corresponding replacements for sp and dx

            # res[8]  = -s[4] + sp[1] + 2x[1]*sp[6] + 2s[1]*dx[6]
            # res[9]  = -s[5] + sp[2] + 2x[2]*sp[6] + 2s[2]*dx[6]
            # res[10]  = abs(x[4])*x[4] + 2k*s[4]*abs(x[4]) - x[1]*sp[3] + m*sp[4] - s[1]*dx[3]
            # res[11] = abs(x[5])*x[5] + 2k*s[5]*abs(x[5]) - x[2]*sp[3] + m*sp[5] - s[2]*dx[3]
            # res[12] = -2s[1]*x[1] - 2s[2]*x[2]
            # res[13] = s[4]*x[1] + s[5]*x[2] + s[1]*x[4] + s[2]*x[5]

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

        x0 = vcat([x1_0, x2_0], zeros(4), [atan(x1_0 / -x2_0)], zeros(7))
        dx0 = vcat([0., 0., dx3_0, dx4_0], zeros(10))

        dvars = vcat(fill(true, 6), [false], fill(true, 6), [false])

        r0 = zeros(length(x0))
        f!(r0, dx0, x0, [], 0.0)

        # t -> 0.0 is just a dummy function, not to be used
        Model(f!, t -> 0.0, x0, dx0, dvars, r0)
    end
end

function pendulum_sensitivity_L(Φ::Float64, u::Function, w::Function, θ::Array{Float64, 1})::Model
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
            # Sensitivity equations for L
            res[8]  = dx[8] - x[11] + 2x[8]dx[6] + 2dx[13]x[1]
            res[9]  = dx[9] - x[12] + 2x[9]dx[6] + 2dx[13]x[2]
            res[10] = 2k*x[11]*abs(x[4]) + m*dx[11] - x[8]dx[3] - dx[10]x[1]
            res[11] = 2k*x[12]*abs(x[5]) + m*dx[12] - x[9]dx[3] - dx[10]x[2]
            res[12] = 2x[8]x[1] + 2x[9]x[2] - 2L
            res[13] = x[8]x[4] + x[11]x[1] + x[9]x[5] + x[12]x[2]
            # Sensitivity of angle of pendulum to L
            # TODO: Analytical formula says it should be x[1]^2+x[2]^2 instead of
            # L^2 (though they should be equal), is it fine to substitute L^2 here?
            res[14] = x[14] - (x[1]*x[9] - x[8]*x[2])/(L^2)

            # These equations sensitivity are written on the intuitive form. To
            # obtain form correct for this funciton, do the following replacements:
            # s[1] -> x[8],  s[2] -> x[9],  s[3] -> x[10]
            # s[4] -> x[11], s[5] -> x[12], s[6] -> x[13]
            # t[1] -> x[15],  t[2] -> x[16],  t[3] -> x[17]
            # t[4] -> x[18], t[5] -> x[19], t[6] -> x[20]
            # and the corresponding replacements for sp/dx and tp/dx

            # res[8]  = -s[4] + sp[1] + 2x[1]*sp[6] + 2s[1]*dx[6]
            # res[9]  = -s[5] + sp[2] + 2x[2]*sp[6] + 2s[2]*dx[6]
            # res[10]  = abs(x[4])*x[4] + 2k*s[4]*abs(x[4]) - x[1]*sp[3] + m*sp[4] - s[1]*dx[3]
            # res[11] = abs(x[5])*x[5] + 2k*s[5]*abs(x[5]) - x[2]*sp[3] + m*sp[5] - s[2]*dx[3]
            # res[12] = -2s[1]*x[1] - 2s[2]*x[2]
            # res[13] = s[4]*x[1] + s[5]*x[2] + s[1]*x[4] + s[2]*x[5]
            # res[14] = x[14] - (x[1]*s[2] - x[2]*s[1])/(L^2)

            # MATLAB form
            # res[15]  = tp[1] - t[4] + 2t[1]dx[6] + 2tp[6]x[1]
            # res[16]  = tp[2] - t[5] + 2t[2]dx[6] + 2tp[6]x[2]
            # res[17] = 2k*t[4]*abs(x[4]) + m*tp[4] - t[1]dx[3] - tp[3]x[1]
            # res[18] = 2k*t[5]*abs(x[5]) + m*tp[5] - t[2]dx[3] - tp[3]x[2]
            # res[19] = 2L -2t[1]x[1] - 2t[2]x[2]
            # res[20] = t[1]x[4] + t[4]x[1] + t[2]x[5] + t[5]x[2]
            # res[21] = (x[1]*t[2] - x[2]*t[1])/(L^2)

            # Hand-derived form
            # res[15]  = -t[4] + tp[1] + 2x[1]*tp[6] + 2t[1]*dx[6]
            # res[16]  = -t[5] + tp[2] + 2x[2]*tp[6] + 2t[2]*dx[6]
            # res[17] = 2k*t[4]*abs(x[4]) - x[1]*tp[3] + m*tp[4] - t[1]*dx[3]
            # res[18] = 2k*t[5]*abs(x[5]) - x[2]*tp[3] + m*tp[5] - t[2]*dx[3]
            # res[19] = -2t[1]*x[1] - 2t[2]*x[2] + 2*L
            # res[20] = t[4]*x[1] + t[5]*x[2] + t[1]*x[4] + t[2]*x[5]

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
        dpend0 = vcat([0., 0., dx3_0, dx4_0], zeros(3))
        # pend_dvar = vcat(fill(true, 6))
        sL0  = vcat([x1_0/L, x2_0/L], zeros(5))
        dsL0 = vcat([0.,0., -dx3_0/L], zeros(4))
        # if x1_0 != 0.0
        #     t0  = vcat([L/(2x1_0), L/(2x2_0)], zeros(4), [(x1_0^2-x2_0^2)/(2L*x1_0*x2_0)])
        #     tp0 = vcat([0,0], [(-L*dx3_0)/(2x1_0*x2_0)], zeros(4))
        # else
        #     t0  = vcat([0.0, L/x2_0], zeros(5))
        #     tp0 = vcat([0,0], [(-L*dx3_0)/(x2_0^2)], zeros(4))
        # end

        x0  = vcat(pend0, sL0)
        dx0 = vcat(dpend0, dsL0)
        # x0 = vcat([x1_0, x2_0], zeros(4), [atan(x1_0 / -x2_0)], zeros(14))
        # dx0 = vcat([0., 0., dx3_0, dx4_0], zeros(17))

        dvars = vcat(fill(true, 6), [false], fill(true, 6), [false])

        r0 = zeros(length(x0))
        f!(r0, dx0, x0, [], 0.0)

        # t -> 0.0 is just a dummy function, not to be used
        Model(f!, t -> 0.0, x0, dx0, dvars, r0)
    end
end

function pendulum_sensitivity_m(Φ::Float64, u::Function, w::Function, θ::Array{Float64, 1})::Model
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

            # These equations sensitivity are written on the intuitive form. To
            # obtain form correct for this funciton, do the following replacements:
            # s[1] -> x[8],  s[2] -> x[9],  s[3] -> x[10]
            # s[4] -> x[11], s[5] -> x[12], s[6] -> x[13]
            # t[1] -> x[15],  t[2] -> x[16],  t[3] -> x[17]
            # t[4] -> x[18], t[5] -> x[19], t[6] -> x[20]
            # and the corresponding replacements for sp/dx and tp/dx

            # res[8]  = -s[4] + sp[1] + 2x[1]*sp[6] + 2s[1]*dx[6]
            # res[9]  = -s[5] + sp[2] + 2x[2]*sp[6] + 2s[2]*dx[6]
            # res[10]  = abs(x[4])*x[4] + 2k*s[4]*abs(x[4]) - x[1]*sp[3] + m*sp[4] - s[1]*dx[3]
            # res[11] = abs(x[5])*x[5] + 2k*s[5]*abs(x[5]) - x[2]*sp[3] + m*sp[5] - s[2]*dx[3]
            # res[12] = -2s[1]*x[1] - 2s[2]*x[2]
            # res[13] = s[4]*x[1] + s[5]*x[2] + s[1]*x[4] + s[2]*x[5]
            # res[14] = x[14] - (x[1]*s[2] - x[2]*s[1])/(L^2)

            # MATLAB form
            # res[15]  = tp[1] - t[4] + 2t[1]dx[6] + 2tp[6]x[1]
            # res[16]  = tp[2] - t[5] + 2t[2]dx[6] + 2tp[6]x[2]
            # res[17] = 2k*t[4]*abs(x[4]) + m*tp[4] - t[1]dx[3] - tp[3]x[1]
            # res[18] = 2k*t[5]*abs(x[5]) + m*tp[5] - t[2]dx[3] - tp[3]x[2]
            # res[19] = 2L -2t[1]x[1] - 2t[2]x[2]
            # res[20] = t[1]x[4] + t[4]x[1] + t[2]x[5] + t[5]x[2]
            # res[21] = (x[1]*t[2] - x[2]*t[1])/(L^2)

            # Hand-derived form
            # res[15]  = -t[4] + tp[1] + 2x[1]*tp[6] + 2t[1]*dx[6]
            # res[16]  = -t[5] + tp[2] + 2x[2]*tp[6] + 2t[2]*dx[6]
            # res[17] = 2k*t[4]*abs(x[4]) - x[1]*tp[3] + m*tp[4] - t[1]*dx[3]
            # res[18] = 2k*t[5]*abs(x[5]) - x[2]*tp[3] + m*tp[5] - t[2]*dx[3]
            # res[19] = -2t[1]*x[1] - 2t[2]*x[2] + 2*L
            # res[20] = t[4]*x[1] + t[5]*x[2] + t[1]*x[4] + t[2]*x[5]

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
        dpend0 = vcat([0., 0., dx3_0, dx4_0], zeros(3))
        # pend_dvar = vcat(fill(true, 6))
        sm0  = zeros(7)
        dsm0 = vcat(zeros(3), [-dx4_0/m, -g/m, 0., 0.])
        # if x1_0 != 0.0
        #     t0  = vcat([L/(2x1_0), L/(2x2_0)], zeros(4), [(x1_0^2-x2_0^2)/(2L*x1_0*x2_0)])
        #     tp0 = vcat([0,0], [(-L*dx3_0)/(2x1_0*x2_0)], zeros(4))
        # else
        #     t0  = vcat([0.0, L/x2_0], zeros(5))
        #     tp0 = vcat([0,0], [(-L*dx3_0)/(x2_0^2)], zeros(4))
        # end

        x0  = vcat(pend0, sm0)
        dx0 = vcat(dpend0, dsm0)
        # x0 = vcat([x1_0, x2_0], zeros(4), [atan(x1_0 / -x2_0)], zeros(14))
        # dx0 = vcat([0., 0., dx3_0, dx4_0], zeros(17))

        dvars = vcat(fill(true, 6), [false], fill(true, 6), [false])

        r0 = zeros(length(x0))
        f!(r0, dx0, x0, [], 0.0)

        # t -> 0.0 is just a dummy function, not to be used
        Model(f!, t -> 0.0, x0, dx0, dvars, r0)
    end
end

function pendulum_sensitivity_g(Φ::Float64, u::Function, w::Function, θ::Array{Float64, 1})::Model
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
            # Sensitivity equations for g
            res[8]  = dx[8] - x[11] + 2x[8]dx[6] + 2dx[13]x[1]
            res[9]  = dx[9] - x[12] + 2x[9]dx[6] + 2dx[13]x[2]
            res[10] = 2k*x[11]*abs(x[4]) + m*dx[11] - x[8]dx[3] - dx[10]x[1]
            res[11] = 2k*x[12]*abs(x[5]) + m*dx[12] - x[9]dx[3] - dx[10]x[2] + m
            res[12] = 2x[8]x[1] + 2x[9]x[2]
            res[13] = x[8]x[4] + x[11]x[1] + x[9]x[5] + x[12]x[2]
            # Sensitivity of angle of pendulum to L
            # TODO: Analytical formula says it should be x[1]^2+x[2]^2 instead of
            # L^2 (though they should be equal), is it fine to substitute L^2 here?
            res[14] = x[14] - (x[1]*x[9] - x[8]*x[2])/(L^2)

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
        dpend0 = vcat([0., 0., dx3_0, dx4_0], zeros(3))
        # pend_dvar = vcat(fill(true, 6))
        sg0  = zeros(7)
        dsg0 = vcat(zeros(5), [-1., 0.])
        # if x1_0 != 0.0
        #     t0  = vcat([L/(2x1_0), L/(2x2_0)], zeros(4), [(x1_0^2-x2_0^2)/(2L*x1_0*x2_0)])
        #     tp0 = vcat([0,0], [(-L*dx3_0)/(2x1_0*x2_0)], zeros(4))
        # else
        #     t0  = vcat([0.0, L/x2_0], zeros(5))
        #     tp0 = vcat([0,0], [(-L*dx3_0)/(x2_0^2)], zeros(4))
        # end

        x0  = vcat(pend0, sg0)
        dx0 = vcat(dpend0, dsg0)
        # x0 = vcat([x1_0, x2_0], zeros(4), [atan(x1_0 / -x2_0)], zeros(14))
        # dx0 = vcat([0., 0., dx3_0, dx4_0], zeros(17))

        dvars = vcat(fill(true, 6), [false], fill(true, 6), [false])

        r0 = zeros(length(x0))
        f!(r0, dx0, x0, [], 0.0)

        # t -> 0.0 is just a dummy function, not to be used
        Model(f!, t -> 0.0, x0, dx0, dvars, r0)
    end
end

# NOTE: Doesn't actually support changing the value of pi
function pendulum_sensitivity_deb(Φ::Float64, u::Function, w::Function, θ::Array{Float64, 1})::Model
    let m = θ[1], L = θ[2], g = θ[3], k = θ[4]

        my_ind = 2
        i = my_ind

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
            # Sensitivity equations for p1                      # here
            res[8]  = dx[8] - x[11] + 2x[8]dx[6] + 2dx[13]x[1]  + Int(i==1)
            res[9]  = dx[9] - x[12] + 2x[9]dx[6] + 2dx[13]x[2] + Int(i==2)
            res[10] = 2k*x[11]*abs(x[4]) + m*dx[11] - x[8]dx[3] - dx[10]x[1] + Int(i==3)
            res[11] = 2k*x[12]*abs(x[5]) + m*dx[12] - x[9]dx[3] - dx[10]x[2] + Int(i==4)
            res[12] = 2x[8]x[1] + 2x[9]x[2] + Int(i==5)
            res[13] = x[8]x[4] + x[11]x[1] + x[9]x[5] + x[12]x[2] + Int(i==6)
            # Sensitivity of angle of pendulum to L
            # TODO: Analytical formula says it should be x[1]^2+x[2]^2 instead of
            # L^2 (though they should be equal), is it fine to substitute L^2 here?
            res[14] = x[14] - (x[1]*x[9] - x[8]*x[2])/(L^2) + Int(i==7)

            nothing
        end

        # # ORIGINAL
        # # Finding consistent initial conditions
        # # Initial values, the pendulum starts at rest
        # u0 = u(0.0)[1]
        # w0 = w(0.0)[1]
        # x1_0 = L * sin(Φ)
        # x2_0 = -L * cos(Φ)
        # dx3_0 = m*g/x2_0
        # dx4_0 = -g*tan(Φ) + (u0 + w0^2)/m
        # pend0 = vcat([x1_0, x2_0], zeros(4), [atan(x1_0 / -x2_0)])
        # dpend0 = vcat([0., 0., dx3_0, dx4_0], zeros(3))
        # ALLOWING NON-ZERO P
        pvec = zeros(7); pvec[i] = 0.
        u0 = u(0.0)[1]
        w0 = w(0.0)[1]
        L_new = sqrt(L^2 - pvec[5])
        x1_0 = L_new * sin(Φ)
        x2_0 = -L_new * cos(Φ)
        x5_0 = -pvec[6]/x2_0
        pend0 = vcat([x1_0, x2_0], zeros(2), [x5_0, 0.0, atan(x1_0 / -x2_0)-pvec[7]])
        dx2_0 = x5_0 - pvec[2]
        dx4_0 = (u0 + w0^2 - pvec[3])/m
        dx5_0 = (-k*abs(x5_0)*x5_0 - m*g - pvec[4])/m
        dpend0 = [-pvec[1], dx2_0, 0.0, dx4_0, dx5_0, 0.0, 0.0]

        s1_0 = 0.0#equivalent -Int(i==5)*x1_0/(2*L^2)
        s2_0 = -Int(i==5)/(2x2_0)  # Very different, is it right?
        s5_0 = -Int(i==6)/(x2_0)
        s0  = [s1_0, s2_0, 0.0, 0.0, s5_0, 0.0, -Int(i==7)]
        ds1_0 = -Int(i==1)
        ds2_0 = -Int(i==2) + s5_0
        ds4_0 = -Int(i==3)/m#Almost equivalent : (Int(i==3)+dx3_0*s1_0)/m
        # ds5_0 = (-Int(i==4)+dx3_0*s2_0)/m
        ds5_0 = (-Int(i==4)+0.0*s2_0)/m
        ds7_0 = Int(i==1)*x2_0/(L^2)#Not even close: (x1_0*(Int(i==1)+s4_0) - x2_0*(Int(i==2)+s5_0))/(L^2)
        ds0 = [ds1_0, ds2_0, 0.0, ds4_0, ds5_0, 0.0, ds7_0]

        # res should be:
        x = vcat(pend0, s0)
        dx = vcat(dpend0, ds0)
        my_res = dx[8] - x[11] + 2x[8]dx[6] + 2dx[13]x[1]  + dx[1] - x[4] + 2dx[6]*x[1]

        # if x1_0 != 0.0
        #     t0  = vcat([L/(2x1_0), L/(2x2_0)], zeros(4), [(x1_0^2-x2_0^2)/(2L*x1_0*x2_0)])
        #     tp0 = vcat([0,0], [(-L*dx3_0)/(2x1_0*x2_0)], zeros(4))
        # else
        #     t0  = vcat([0.0, L/x2_0], zeros(5))
        #     tp0 = vcat([0,0], [(-L*dx3_0)/(x2_0^2)], zeros(4))
        # end

        x0  = vcat(pend0, s0)
        dx0 = vcat(dpend0, ds0)
        # x0 = vcat([x1_0, x2_0], zeros(4), [atan(x1_0 / -x2_0)], zeros(14))
        # dx0 = vcat([0., 0., dx3_0, dx4_0], zeros(17))

        dvars = vcat(fill(true, 6), [false], fill(true, 6), [false])

        r0 = zeros(length(x0))
        f!(r0, dx0, x0, [], 0.0)
        # @info "r0 is $r0 ! debsens"

        # t -> 0.0 is just a dummy function, not to be used
        Model(f!, t -> 0.0, x0, dx0, dvars, r0)
    end
end

# NOTE: Doesn't actually support changing the value of pi
function pendulum_sensitivity_deb_0p01(Φ::Float64, u::Function, w::Function, θ::Array{Float64, 1})::Model
    let m = θ[1], L = θ[2], g = θ[3], k = θ[4]

        my_ind = 2
        i = my_ind

        # the residual function
        function f!(res, dx, x, θ, t)
            wt = w(t)
            ut = u(t)
            # Dynamic Equations
            res[1] = dx[1] - x[4] + 2dx[6]*x[1] + Int(i==1)*0.01
            res[2] = dx[2] - x[5] + 2dx[6]*x[2] + Int(i==2)*0.01
            res[3] = m*dx[4] - dx[3]*x[1] + k*abs(x[4])*x[4] - ut[1] - wt[1]^2 + Int(i==3)*0.01
            res[4] = m*dx[5] - dx[3]*x[2] + k*abs(x[5])*x[5] + m*g + Int(i==4)*0.01
            res[5] = x[1]^2 + x[2]^2 - L^2 + Int(i==5)*0.01
            res[6] = x[4]*x[1] + x[5]*x[2] + Int(i==6)*0.01
            # Angle of pendulum
            res[7] = x[7] - atan(x[1] / -x[2]) + Int(i==7)*0.01
            # Sensitivity equations for p1                      # here
            res[8]  = dx[8] - x[11] + 2x[8]dx[6] + 2dx[13]x[1]  + Int(i==1)
            res[9]  = dx[9] - x[12] + 2x[9]dx[6] + 2dx[13]x[2] + Int(i==2)
            res[10] = 2k*x[11]*abs(x[4]) + m*dx[11] - x[8]dx[3] - dx[10]x[1] + Int(i==3)
            res[11] = 2k*x[12]*abs(x[5]) + m*dx[12] - x[9]dx[3] - dx[10]x[2] + Int(i==4)
            res[12] = 2x[8]x[1] + 2x[9]x[2] + Int(i==5)
            res[13] = x[8]x[4] + x[11]x[1] + x[9]x[5] + x[12]x[2] + Int(i==6)
            # Sensitivity of angle of pendulum to L
            # TODO: Analytical formula says it should be x[1]^2+x[2]^2 instead of
            # L^2 (though they should be equal), is it fine to substitute L^2 here?
            res[14] = x[14] - (x[1]*x[9] - x[8]*x[2])/(L^2) + Int(i==7)

            nothing
        end

        # # ORIGINAL
        # # Finding consistent initial conditions
        # # Initial values, the pendulum starts at rest
        # u0 = u(0.0)[1]
        # w0 = w(0.0)[1]
        # x1_0 = L * sin(Φ)
        # x2_0 = -L * cos(Φ)
        # dx3_0 = m*g/x2_0
        # dx4_0 = -g*tan(Φ) + (u0 + w0^2)/m
        # pend0 = vcat([x1_0, x2_0], zeros(4), [atan(x1_0 / -x2_0)])
        # dpend0 = vcat([0., 0., dx3_0, dx4_0], zeros(3))
        # ALLOWING NON-ZERO P
        pvec = zeros(7); pvec[i] = 0.01
        u0 = u(0.0)[1]
        w0 = w(0.0)[1]
        L_new = sqrt(L^2 - pvec[5])
        x1_0 = L_new * sin(Φ)
        x2_0 = -L_new * cos(Φ)
        x5_0 = -pvec[6]/x2_0
        pend0 = vcat([x1_0, x2_0], zeros(2), [x5_0, 0.0, atan(x1_0 / -x2_0)-pvec[7]])
        dx2_0 = x5_0 - pvec[2]
        dx4_0 = (u0 + w0^2 - pvec[3])/m
        dx5_0 = (-k*abs(x5_0)*x5_0 - m*g - pvec[4])/m
        dpend0 = [-pvec[1], dx2_0, 0.0, dx4_0, dx5_0, 0.0, 0.0]

        s1_0 = 0.0#equivalent -Int(i==5)*x1_0/(2*L^2)
        s2_0 = -Int(i==5)/(2x2_0)  # Very different, is it right?
        s5_0 = -Int(i==6)/(x2_0)
        s0  = [s1_0, s2_0, 0.0, 0.0, s5_0, 0.0, -Int(i==7)]
        ds1_0 = -Int(i==1)
        ds2_0 = -Int(i==2) + s5_0
        ds4_0 = -Int(i==3)/m#Almost equivalent : (Int(i==3)+dx3_0*s1_0)/m
        # ds5_0 = (-Int(i==4)+dx3_0*s2_0)/m
        ds5_0 = (-Int(i==4)+0.0*s2_0)/m
        ds7_0 = Int(i==1)*x2_0/(L^2)#Not even close: (x1_0*(Int(i==1)+s4_0) - x2_0*(Int(i==2)+s5_0))/(L^2)
        ds0 = [ds1_0, ds2_0, 0.0, ds4_0, ds5_0, 0.0, ds7_0]
        # s1_0 = -Int(i==5)*x1_0/(2*L^2)
        # s2_0 = Int(i==5)*x2_0/(2*L^2)
        # s4_0 = Int(i==6)/(2x1_0)
        # s5_0 = Int(i==6)/(2x2_0)
        # s0  = [s1_0, s2_0, 0.0, s4_0, s5_0, 0.0, -Int(i==7)]
        # ds1_0 = Int(i==1) + s4_0
        # ds2_0 = Int(i==2) + s5_0
        # ds4_0 = (Int(i==3)+dx3_0*s1_0)/m
        # ds5_0 = (Int(i==4)+dx3_0*s2_0)/m
        # ds7_0 = (x1_0*(Int(i==1)+s4_0) - x2_0*(Int(i==2)+s5_0))/(L^2)
        # ds0 = [ds1_0, ds2_0, 0.0, ds4_0, ds5_0, 0.0, ds7_0]

        # res should be:
        x = vcat(pend0, s0)
        dx = vcat(dpend0, ds0)
        my_res = dx[8] - x[11] + 2x[8]dx[6] + 2dx[13]x[1]  + dx[1] - x[4] + 2dx[6]*x[1]

        # if x1_0 != 0.0
        #     t0  = vcat([L/(2x1_0), L/(2x2_0)], zeros(4), [(x1_0^2-x2_0^2)/(2L*x1_0*x2_0)])
        #     tp0 = vcat([0,0], [(-L*dx3_0)/(2x1_0*x2_0)], zeros(4))
        # else
        #     t0  = vcat([0.0, L/x2_0], zeros(5))
        #     tp0 = vcat([0,0], [(-L*dx3_0)/(x2_0^2)], zeros(4))
        # end

        x0  = vcat(pend0, s0)
        dx0 = vcat(dpend0, ds0)
        # x0 = vcat([x1_0, x2_0], zeros(4), [atan(x1_0 / -x2_0)], zeros(14))
        # dx0 = vcat([0., 0., dx3_0, dx4_0], zeros(17))

        dvars = vcat(fill(true, 6), [false], fill(true, 6), [false])

        r0 = zeros(length(x0))
        f!(r0, dx0, x0, [], 0.0)
        # @info "r0 is $r0, for this deb one"

        # t -> 0.0 is just a dummy function, not to be used
        Model(f!, t -> 0.0, x0, dx0, dvars, r0)
    end
end

function pendulum_sensitivity_Lk(Φ::Float64, u::Function, w::Function, θ::Array{Float64, 1})::Model
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
            # Sensitivity equations for L
            res[8]  = dx[8] - x[11] + 2x[8]dx[6] + 2dx[13]x[1]
            res[9]  = dx[9] - x[12] + 2x[9]dx[6] + 2dx[13]x[2]
            res[10] = 2k*x[11]*abs(x[4]) + m*dx[11] - x[8]dx[3] - dx[10]x[1]
            res[11] = 2k*x[12]*abs(x[5]) + m*dx[12] - x[9]dx[3] - dx[10]x[2]
            res[12] = 2x[8]x[1] + 2x[9]x[2] - 2L
            res[13] = x[8]x[4] + x[11]x[1] + x[9]x[5] + x[12]x[2]
            # Sensitivity of angle of pendulum to L
            # TODO: Analytical formula says it should be x[1]^2+x[2]^2 instead of
            # L^2 (though they should be equal), is it fine to substitute L^2 here?
            res[14] = x[14] - (x[1]*x[9] - x[8]*x[2])/(L^2)
            # Sensitivity equations for k
            res[15]  = -x[18] + dx[15] + 2x[1]*dx[20] + 2x[15]*dx[6]
            res[16]  = -x[19] + dx[16] + 2x[2]*dx[20] + 2x[16]*dx[6]
            res[17] = abs(x[4])*x[4] + 2k*x[18]*abs(x[4]) - x[1]*dx[17] + m*dx[18] - x[15]*dx[3]
            res[18] = abs(x[5])*x[5] + 2k*x[19]*abs(x[5]) - x[2]*dx[17] + m*dx[19] - x[16]*dx[3]
            res[19] = 2x[15]*x[1] + 2x[16]*x[2]
            res[20] = x[18]*x[1] + x[19]*x[2] + x[15]*x[4] + x[16]*x[5]
            # Sensitivity of angle of pendulum to k
            # TODO: Analytical formula says it should be x[1]^2+x[2]^2 instead of
            # L^2 (though they should be equal), is it fine to substitute L^2 here?
            res[21] = x[21] - (x[1]*x[16] - x[15]*x[2])/(L^2)

            # These equations sensitivity are written on the intuitive form. To
            # obtain form correct for this funciton, do the following replacements:
            # s[1] -> x[8],  s[2] -> x[9],  s[3] -> x[10]
            # s[4] -> x[11], s[5] -> x[12], s[6] -> x[13]
            # t[1] -> x[15],  t[2] -> x[16],  t[3] -> x[17]
            # t[4] -> x[18], t[5] -> x[19], t[6] -> x[20]
            # and the corresponding replacements for sp/dx and tp/dx

            # res[8]  = -s[4] + sp[1] + 2x[1]*sp[6] + 2s[1]*dx[6]
            # res[9]  = -s[5] + sp[2] + 2x[2]*sp[6] + 2s[2]*dx[6]
            # res[10]  = abs(x[4])*x[4] + 2k*s[4]*abs(x[4]) - x[1]*sp[3] + m*sp[4] - s[1]*dx[3]
            # res[11] = abs(x[5])*x[5] + 2k*s[5]*abs(x[5]) - x[2]*sp[3] + m*sp[5] - s[2]*dx[3]
            # res[12] = -2s[1]*x[1] - 2s[2]*x[2]
            # res[13] = s[4]*x[1] + s[5]*x[2] + s[1]*x[4] + s[2]*x[5]
            # res[14] = x[14] - (x[1]*s[2] - x[2]*s[1])/(L^2)

            # MATLAB form
            # res[15]  = tp[1] - t[4] + 2t[1]dx[6] + 2tp[6]x[1]
            # res[16]  = tp[2] - t[5] + 2t[2]dx[6] + 2tp[6]x[2]
            # res[17] = 2k*t[4]*abs(x[4]) + m*tp[4] - t[1]dx[3] - tp[3]x[1]
            # res[18] = 2k*t[5]*abs(x[5]) + m*tp[5] - t[2]dx[3] - tp[3]x[2]
            # res[19] = 2L -2t[1]x[1] - 2t[2]x[2]
            # res[20] = t[1]x[4] + t[4]x[1] + t[2]x[5] + t[5]x[2]
            # res[21] = (x[1]*t[2] - x[2]*t[1])/(L^2)

            # Hand-derived form
            # res[15]  = -t[4] + tp[1] + 2x[1]*tp[6] + 2t[1]*dx[6]
            # res[16]  = -t[5] + tp[2] + 2x[2]*tp[6] + 2t[2]*dx[6]
            # res[17] = 2k*t[4]*abs(x[4]) - x[1]*tp[3] + m*tp[4] - t[1]*dx[3]
            # res[18] = 2k*t[5]*abs(x[5]) - x[2]*tp[3] + m*tp[5] - t[2]*dx[3]
            # res[19] = -2t[1]*x[1] - 2t[2]*x[2] + 2*L
            # res[20] = t[4]*x[1] + t[5]*x[2] + t[1]*x[4] + t[2]*x[5]

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
        dpend0 = vcat([0., 0., dx3_0, dx4_0], zeros(3))
        # pend_dvar = vcat(fill(true, 6))
        s0  = zeros(7)
        sp0 = zeros(7)
        t0 = vcat([sin(Φ), -cos(Φ)], zeros(5))
        tp0 = vcat([0.,0.], [-dx3_0/L], zeros(4))
        # if x1_0 != 0.0
        #     t0  = vcat([L/(2x1_0), L/(2x2_0)], zeros(4), [(x1_0^2-x2_0^2)/(2L*x1_0*x2_0)])
        #     tp0 = vcat([0,0], [(-L*dx3_0)/(2x1_0*x2_0)], zeros(4))
        # else
        #     t0  = vcat([0.0, L/x2_0], zeros(5))
        #     tp0 = vcat([0,0], [(-L*dx3_0)/(x2_0^2)], zeros(4))
        # end

        x0  = vcat(pend0, s0, t0)
        dx0 = vcat(dpend0, sp0, tp0)
        # x0 = vcat([x1_0, x2_0], zeros(4), [atan(x1_0 / -x2_0)], zeros(14))
        # dx0 = vcat([0., 0., dx3_0, dx4_0], zeros(17))

        dvars = vcat(fill(true, 6), [false], fill(true, 6), [false], fill(true, 6), [false])

        r0 = zeros(length(x0))
        f!(r0, dx0, x0, [], 0.0)

        # t -> 0.0 is just a dummy function, not to be used
        Model(f!, t -> 0.0, x0, dx0, dvars, r0)
    end
end

function pendulum_sensitivity_full(Φ::Float64, u::Function, w::Function, θ::Array{Float64, 1})::Model
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

            # Sensitivity with respect to L
            res[15]  = -x[18] + dx[15] + 2x[1]*dx[20] + 2x[15]*dx[6]
            res[16]  = -x[19] + dx[16] + 2x[2]*dx[20] + 2x[16]*dx[6]
            res[17] = 2k*x[18]*abs(x[4]) - x[1]*dx[17] + m*dx[18] - x[15]*dx[3]
            res[18] = 2k*x[19]*abs(x[5]) - x[2]*dx[17] + m*dx[19] - x[16]*dx[3]
            res[19] = -2x[15]*x[1] - 2x[16]*x[2] + 2L
            res[20] = x[18]*x[1] + x[19]*x[2] + x[15]*x[4] + x[16]*x[5]
            res[21] = x[21] - (x[1]*x[16] - x[2]*x[15])/(L^2)

            # Sensitivity with respect to g
            res[22]  = -x[25] + dx[22] + 2x[1]*dx[27] + 2x[22]*dx[6]
            res[23]  = -x[26] + dx[23] + 2x[2]*dx[27] + 2x[23]*dx[6]
            res[24] = 2k*x[25]*abs(x[4]) - x[1]*dx[24] + m*dx[25] - x[22]*dx[3]
            res[25] = 2k*x[26]*abs(x[5]) - x[2]*dx[24] + m*dx[26] - x[23]*dx[3] + m
            res[26] = -2x[22]*x[1] - 2x[23]*x[2]
            res[27] = x[25]*x[1] + x[26]*x[2] + x[22]*x[4] + x[23]*x[5]
            res[28] = x[28] - (x[1]*x[23] - x[2]*x[22])/(L^2)

            # Sensitivity with respect to k
            res[29]  = -x[32] + dx[29] + 2x[1]*dx[34] + 2x[29]*dx[6]
            res[30]  = -x[33] + dx[30] + 2x[2]*dx[34] + 2x[30]*dx[6]
            res[31] = 2k*x[32]*abs(x[4]) - x[1]*dx[31] + m*dx[32] - x[29]*dx[3] + abs(x[4])x[4]
            res[32] = 2k*x[33]*abs(x[5]) - x[2]*dx[31] + m*dx[33] - x[30]*dx[3] + abs(x[5])x[5]
            res[33] = -2x[29]*x[1] - 2x[30]*x[2]
            res[34] = x[32]*x[1] + x[33]*x[2] + x[29]*x[4] + x[30]*x[5]
            res[35] = x[35] - (x[1]*x[30] - x[2]*x[29])/(L^2)

            # The default template used to generate these equations was:
            # res[8]  = -s[4] + sp[1] + 2x[1]*sp[6] + 2s[1]*dx[6]
            # res[9]  = -s[5] + sp[2] + 2x[2]*sp[6] + 2s[2]*dx[6]
            # res[10]  = 2k*s[4]*abs(x[4]) - x[1]*sp[3] + m*sp[4] - s[1]*dx[3]
            # res[11] = 2k*s[5]*abs(x[5]) - x[2]*sp[3] + m*sp[5] - s[2]*dx[3]
            # res[12] = -2s[1]*x[1] - 2s[2]*x[2]
            # res[13] = s[4]*x[1] + s[5]*x[2] + s[1]*x[4] + s[2]*x[5]
            # res[14] = x[14] - (x[1]*s[2] - x[2]*s[1])/(L^2)
            # Note that these edxressions don't include parameter-specific terms

            # To obtain the equations written on the form suitable for this
            # function, use the following substitutions:
            # sm[1] -> x[8],  sm[2] -> x[9],  sm[3] -> x[10]
            # sm[4] -> x[11], sm[5] -> x[12], sm[6] -> x[13]
            # sL[1] -> x[15],  sL[2] -> x[16],  sL[3] -> x[17]
            # sL[4] -> x[18], sL[5] -> x[19], sL[6] -> x[20]
            # sg[1] -> x[22],  sg[2] -> x[23],  sg[3] -> x[24]
            # sg[4] -> x[25], sg[5] -> x[26], sg[6] -> x[27]
            # sk[1] -> x[29],  sk[2] -> x[30],  sk[3] -> x[31]
            # sk[4] -> x[32], sk[5] -> x[33], sk[6] -> x[34]

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
        dpend0 = vcat([0., 0., dx3_0, dx4_0], zeros(3))
        sm0  = zeros(7)
        dsm0 = vcat(zeros(3), [-dx4_0/m, -g/m, 0., 0.])
        sL0  = vcat([x1_0/L, x2_0/L], zeros(5))
        dsL0 = vcat([0.,0., -dx3_0/L], zeros(4))
        sg0  = zeros(7)
        sgp0 = vcat(zeros(4), [-1., 0., 0.])
        sk0  = zeros(7)
        dsk0 = zeros(7)
        # t0 = vcat([sin(Φ), -cos(Φ)], zeros(5))
        # tp0 = vcat([0.,0.], [-dx3_0/L], zeros(4))
        # if x1_0 != 0.0
        #     t0  = vcat([L/(2x1_0), L/(2x2_0)], zeros(4), [(x1_0^2-x2_0^2)/(2L*x1_0*x2_0)])
        #     tp0 = vcat([0,0], [(-L*dx3_0)/(2x1_0*x2_0)], zeros(4))
        # else
        #     t0  = vcat([0.0, L/x2_0], zeros(5))
        #     tp0 = vcat([0,0], [(-L*dx3_0)/(x2_0^2)], zeros(4))
        # end

        x0  = vcat(pend0, sm0, sL0, sg0, sk0)
        dx0 = vcat(dpend0, dsm0, dsL0, sgp0, dsk0)
        # x0 = vcat([x1_0, x2_0], zeros(4), [atan(x1_0 / -x2_0)], zeros(14))
        # dx0 = vcat([0., 0., dx3_0, dx4_0], zeros(17))

        dvars = repeat(vcat(fill(true, 6), [false]), 5)

        r0 = zeros(length(x0))
        f!(r0, dx0, x0, [], 0.0)

        # t -> 0.0 is just a dummy function, not to be used
        Model(f!, t -> 0.0, x0, dx0, dvars, r0)
    end
end

function pendulum_sensitivity_sans_g(Φ::Float64, u::Function, w::Function, θ::Array{Float64, 1})::Model
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

            # Sensitivity with respect to L
            res[15]  = -x[18] + dx[15] + 2x[1]*dx[20] + 2x[15]*dx[6]
            res[16]  = -x[19] + dx[16] + 2x[2]*dx[20] + 2x[16]*dx[6]
            res[17] = 2k*x[18]*abs(x[4]) - x[1]*dx[17] + m*dx[18] - x[15]*dx[3]
            res[18] = 2k*x[19]*abs(x[5]) - x[2]*dx[17] + m*dx[19] - x[16]*dx[3]
            res[19] = -2x[15]*x[1] - 2x[16]*x[2] + 2L
            res[20] = x[18]*x[1] + x[19]*x[2] + x[15]*x[4] + x[16]*x[5]
            res[21] = x[21] - (x[1]*x[16] - x[2]*x[15])/(L^2)

            # Sensitivity with respect to k
            res[22]  = -x[25] + dx[22] + 2x[1]*dx[27] + 2x[22]*dx[6]
            res[23]  = -x[26] + dx[23] + 2x[2]*dx[27] + 2x[23]*dx[6]
            res[24] = 2k*x[25]*abs(x[4]) - x[1]*dx[24] + m*dx[25] - x[22]*dx[3] + abs(x[4])x[4]
            res[25] = 2k*x[26]*abs(x[5]) - x[2]*dx[24] + m*dx[26] - x[23]*dx[3] + abs(x[5])x[5]
            res[26] = -2x[22]*x[1] - 2x[23]*x[2]
            res[27] = x[25]*x[1] + x[26]*x[2] + x[22]*x[4] + x[23]*x[5]
            res[28] = x[28] - (x[1]*x[23] - x[2]*x[22])/(L^2)

            # The default template used to generate these equations was:
            # res[8]  = -s[4] + sp[1] + 2x[1]*sp[6] + 2s[1]*dx[6]
            # res[9]  = -s[5] + sp[2] + 2x[2]*sp[6] + 2s[2]*dx[6]
            # res[10]  = 2k*s[4]*abs(x[4]) - x[1]*sp[3] + m*sp[4] - s[1]*dx[3]
            # res[11] = 2k*s[5]*abs(x[5]) - x[2]*sp[3] + m*sp[5] - s[2]*dx[3]
            # res[12] = -2s[1]*x[1] - 2s[2]*x[2]
            # res[13] = s[4]*x[1] + s[5]*x[2] + s[1]*x[4] + s[2]*x[5]
            # res[14] = x[14] - (x[1]*s[2] - x[2]*s[1])/(L^2)
            # Note that these edxressions don't include parameter-specific terms

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
        dpend0 = vcat([0., 0., dx3_0, dx4_0], zeros(3))
        sm0  = zeros(7)
        dsm0 = vcat(zeros(3), [-dx4_0/m, -g/m, 0., 0.])
        sL0  = vcat([x1_0/L, x2_0/L], zeros(5))
        dsL0 = vcat([0.,0., -dx3_0/L], zeros(4))
        sk0  = zeros(7)
        dsk0 = zeros(7)
        # t0 = vcat([sin(Φ), -cos(Φ)], zeros(5))
        # tp0 = vcat([0.,0.], [-dx3_0/L], zeros(4))
        # if x1_0 != 0.0
        #     t0  = vcat([L/(2x1_0), L/(2x2_0)], zeros(4), [(x1_0^2-x2_0^2)/(2L*x1_0*x2_0)])
        #     tp0 = vcat([0,0], [(-L*dx3_0)/(2x1_0*x2_0)], zeros(4))
        # else
        #     t0  = vcat([0.0, L/x2_0], zeros(5))
        #     tp0 = vcat([0,0], [(-L*dx3_0)/(x2_0^2)], zeros(4))
        # end

        x0  = vcat(pend0, sm0, sL0, sk0)
        dx0 = vcat(dpend0, dsm0, dsL0, dsk0)
        # x0 = vcat([x1_0, x2_0], zeros(4), [atan(x1_0 / -x2_0)], zeros(14))
        # dx0 = vcat([0., 0., dx3_0, dx4_0], zeros(17))

        dvars = repeat(vcat(fill(true, 6), [false]), 4)

        r0 = zeros(length(x0))
        f!(r0, dx0, x0, [], 0.0)

        # t -> 0.0 is just a dummy function, not to be used
        Model(f!, t -> 0.0, x0, dx0, dvars, r0)
    end
end

function pendulum_sensitivity_sans_g_with_dist_sens_2(Φ::Float64, u::Function, w::Function, θ::Array{Float64, 1})::Model
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

            # Sensitivity with respect to L
            res[15]  = -x[18] + dx[15] + 2x[1]*dx[20] + 2x[15]*dx[6]
            res[16]  = -x[19] + dx[16] + 2x[2]*dx[20] + 2x[16]*dx[6]
            res[17] = 2k*x[18]*abs(x[4]) - x[1]*dx[17] + m*dx[18] - x[15]*dx[3]
            res[18] = 2k*x[19]*abs(x[5]) - x[2]*dx[17] + m*dx[19] - x[16]*dx[3]
            res[19] = -2x[15]*x[1] - 2x[16]*x[2] + 2L
            res[20] = x[18]*x[1] + x[19]*x[2] + x[15]*x[4] + x[16]*x[5]
            res[21] = x[21] - (x[1]*x[16] - x[2]*x[15])/(L^2)

            # Sensitivity with respect to k
            res[22]  = -x[25] + dx[22] + 2x[1]*dx[27] + 2x[22]*dx[6]
            res[23]  = -x[26] + dx[23] + 2x[2]*dx[27] + 2x[23]*dx[6]
            res[24] = 2k*x[25]*abs(x[4]) - x[1]*dx[24] + m*dx[25] - x[22]*dx[3] + abs(x[4])x[4]
            res[25] = 2k*x[26]*abs(x[5]) - x[2]*dx[24] + m*dx[26] - x[23]*dx[3] + abs(x[5])x[5]
            res[26] = -2x[22]*x[1] - 2x[23]*x[2]
            res[27] = x[25]*x[1] + x[26]*x[2] + x[22]*x[4] + x[23]*x[5]
            res[28] = x[28] - (x[1]*x[23] - x[2]*x[22])/(L^2)

            # Sensitivty wrt first disturbance parameter
            # (i.e. parameter corresponding to wt[2])
            res[29] = 2dx[6]x[29] - x[32] + dx[29] + 2x[1]dx[34]
            res[30] = 2dx[6]x[30] - x[33] + dx[30] + 2x[2]dx[34]
            res[31] = -dx[3]x[29] + 2k*abs(x[4])*x[32] - x[1]*dx[31] + m*dx[32] - 2*wt[1]*wt[2]
            res[32] = -dx[3]x[30] + 2k*abs(x[5])*x[33] - x[2]*dx[31] + m*dx[33]
            res[33] = 2x[1]x[29] + 2x[2]x[30]
            res[34] = x[4]x[29] + x[5]x[30] + x[1]x[32] + x[2]x[33]
            # Sensitivity of angle of pendulum to disturbance parameter
            # TODO: Analytical formula says it should be x[1]^2+x[2]^2 instead of
            # L^2 (though they should be equal), is it fine to substitute L^2 here?
            res[35] = x[35] - (x[1]*x[30] - x[2]*x[29])/(L^2)

            # Sensitivty wrt second disturbance parameter
            # (i.e. parameter corresponding to wt[3])
            res[36] = 2dx[6]x[36] - x[39] + dx[36] + 2x[1]dx[41]
            res[37] = 2dx[6]x[37] - x[40] + dx[37] + 2x[2]dx[41]
            res[38] = -dx[3]x[36] + 2k*abs(x[4])*x[39] - x[1]*dx[38] + m*dx[39] - 2*wt[1]*wt[3]
            res[39] = -dx[3]x[37] + 2k*abs(x[5])*x[40] - x[2]*dx[38] + m*dx[40]
            res[40] = 2x[1]x[36] + 2x[2]x[37]
            res[41] = x[4]x[36] + x[5]x[37] + x[1]x[39] + x[2]x[40]
            # Sensitivity of angle of pendulum to disturbance parameter
            # TODO: Analytical formula says it should be x[1]^2+x[2]^2 instead of
            # L^2 (though they should be equal), is it fine to substitute L^2 here?
            res[42] = x[42] - (x[1]*x[37] - x[2]*x[36])/(L^2)

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
        dpend0 = vcat([0., 0., dx3_0, dx4_0], zeros(3))
        sm0  = zeros(7)
        dsm0 = vcat(zeros(3), [-dx4_0/m, -g/m, 0., 0.])
        sL0  = vcat([x1_0/L, x2_0/L], zeros(5))
        dsL0 = vcat([0.,0., -dx3_0/L], zeros(4))
        sk0  = zeros(7)
        dsk0 = zeros(7)
        r0 = zeros(7)
        rp0 = zeros(7)
        # t0 = vcat([sin(Φ), -cos(Φ)], zeros(5))
        # tp0 = vcat([0.,0.], [-dx3_0/L], zeros(4))
        # if x1_0 != 0.0
        #     t0  = vcat([L/(2x1_0), L/(2x2_0)], zeros(4), [(x1_0^2-x2_0^2)/(2L*x1_0*x2_0)])
        #     tp0 = vcat([0,0], [(-L*dx3_0)/(2x1_0*x2_0)], zeros(4))
        # else
        #     t0  = vcat([0.0, L/x2_0], zeros(5))
        #     tp0 = vcat([0,0], [(-L*dx3_0)/(x2_0^2)], zeros(4))
        # end

        x0  = vcat(pend0, sm0, sL0, sk0, r0, r0)
        dx0 = vcat(dpend0, dsm0, dsL0, dsk0, rp0, rp0)
        # x0 = vcat([x1_0, x2_0], zeros(4), [atan(x1_0 / -x2_0)], zeros(14))
        # dx0 = vcat([0., 0., dx3_0, dx4_0], zeros(17))

        dvars = repeat(vcat(fill(true, 6), [false]), 6)

        r0 = zeros(length(x0))
        f!(r0, dx0, x0, [], 0.0)

        # t -> 0.0 is just a dummy function, not to be used
        Model(f!, t -> 0.0, x0, dx0, dvars, r0)
    end
end

function pendulum_sensitivity_sans_g_with_dist_sens_3(Φ::Float64, u::Function, w::Function, θ::Array{Float64, 1})::Model
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

            # Sensitivity with respect to L
            res[15]  = -x[18] + dx[15] + 2x[1]*dx[20] + 2x[15]*dx[6]
            res[16]  = -x[19] + dx[16] + 2x[2]*dx[20] + 2x[16]*dx[6]
            res[17] = 2k*x[18]*abs(x[4]) - x[1]*dx[17] + m*dx[18] - x[15]*dx[3]
            res[18] = 2k*x[19]*abs(x[5]) - x[2]*dx[17] + m*dx[19] - x[16]*dx[3]
            res[19] = -2x[15]*x[1] - 2x[16]*x[2] + 2L
            res[20] = x[18]*x[1] + x[19]*x[2] + x[15]*x[4] + x[16]*x[5]
            res[21] = x[21] - (x[1]*x[16] - x[2]*x[15])/(L^2)

            # Sensitivity with respect to k
            res[22]  = -x[25] + dx[22] + 2x[1]*dx[27] + 2x[22]*dx[6]
            res[23]  = -x[26] + dx[23] + 2x[2]*dx[27] + 2x[23]*dx[6]
            res[24] = 2k*x[25]*abs(x[4]) - x[1]*dx[24] + m*dx[25] - x[22]*dx[3] + abs(x[4])x[4]
            res[25] = 2k*x[26]*abs(x[5]) - x[2]*dx[24] + m*dx[26] - x[23]*dx[3] + abs(x[5])x[5]
            res[26] = -2x[22]*x[1] - 2x[23]*x[2]
            res[27] = x[25]*x[1] + x[26]*x[2] + x[22]*x[4] + x[23]*x[5]
            res[28] = x[28] - (x[1]*x[23] - x[2]*x[22])/(L^2)

            # Sensitivty wrt first disturbance parameter
            # (i.e. parameter corresponding to wt[2])
            res[29] = 2dx[6]x[29] - x[32] + dx[29] + 2x[1]dx[34]
            res[30] = 2dx[6]x[30] - x[33] + dx[30] + 2x[2]dx[34]
            res[31] = -dx[3]x[29] + 2k*abs(x[4])*x[32] - x[1]*dx[31] + m*dx[32] - 2*wt[1]*wt[2]
            res[32] = -dx[3]x[30] + 2k*abs(x[5])*x[33] - x[2]*dx[31] + m*dx[33]
            res[33] = 2x[1]x[29] + 2x[2]x[30]
            res[34] = x[4]x[29] + x[5]x[30] + x[1]x[32] + x[2]x[33]
            # Sensitivity of angle of pendulum to disturbance parameter
            # TODO: Analytical formula says it should be x[1]^2+x[2]^2 instead of
            # L^2 (though they should be equal), is it fine to substitute L^2 here?
            res[35] = x[35] - (x[1]*x[30] - x[2]*x[29])/(L^2)

            # Sensitivty wrt second disturbance parameter
            # (i.e. parameter corresponding to wt[3])
            res[36] = 2dx[6]x[36] - x[39] + dx[36] + 2x[1]dx[41]
            res[37] = 2dx[6]x[37] - x[40] + dx[37] + 2x[2]dx[41]
            res[38] = -dx[3]x[36] + 2k*abs(x[4])*x[39] - x[1]*dx[38] + m*dx[39] - 2*wt[1]*wt[3]
            res[39] = -dx[3]x[37] + 2k*abs(x[5])*x[40] - x[2]*dx[38] + m*dx[40]
            res[40] = 2x[1]x[36] + 2x[2]x[37]
            res[41] = x[4]x[36] + x[5]x[37] + x[1]x[39] + x[2]x[40]
            # Sensitivity of angle of pendulum to disturbance parameter
            # TODO: Analytical formula says it should be x[1]^2+x[2]^2 instead of
            # L^2 (though they should be equal), is it fine to substitute L^2 here?
            res[42] = x[42] - (x[1]*x[37] - x[2]*x[36])/(L^2)

            # Sensitivty wrt third disturbance parameter
            # (i.e. parameter corresponding to wt[4])
            res[43] = 2dx[6]x[43] - x[46] + dx[43] + 2x[1]dx[48]
            res[44] = 2dx[6]x[44] - x[47] + dx[44] + 2x[2]dx[48]
            res[45] = -dx[3]x[43] + 2k*abs(x[4])*x[46] - x[1]*dx[45] + m*dx[46] - 2*wt[1]*wt[4]
            res[46] = -dx[3]x[44] + 2k*abs(x[5])*x[47] - x[2]*dx[45] + m*dx[47]
            res[47] = 2x[1]x[43] + 2x[2]x[44]
            res[48] = x[4]x[43] + x[5]x[44] + x[1]x[46] + x[2]x[47]
            # Sensitivity of angle of pendulum to disturbance parameter
            # TODO: Analytical formula says it should be x[1]^2+x[2]^2 instead of
            # L^2 (though they should be equal), is it fine to substitute L^2 here?
            res[49] = x[49] - (x[1]*x[44] - x[2]*x[43])/(L^2)

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
        dpend0 = vcat([0., 0., dx3_0, dx4_0], zeros(3))
        sm0  = zeros(7)
        dsm0 = vcat(zeros(3), [-dx4_0/m, -g/m, 0., 0.])
        sL0  = vcat([x1_0/L, x2_0/L], zeros(5))
        dsL0 = vcat([0.,0., -dx3_0/L], zeros(4))
        sk0  = zeros(7)
        dsk0 = zeros(7)
        r0 = zeros(7)
        rp0 = zeros(7)
        # t0 = vcat([sin(Φ), -cos(Φ)], zeros(5))
        # tp0 = vcat([0.,0.], [-dx3_0/L], zeros(4))
        # if x1_0 != 0.0
        #     t0  = vcat([L/(2x1_0), L/(2x2_0)], zeros(4), [(x1_0^2-x2_0^2)/(2L*x1_0*x2_0)])
        #     tp0 = vcat([0,0], [(-L*dx3_0)/(2x1_0*x2_0)], zeros(4))
        # else
        #     t0  = vcat([0.0, L/x2_0], zeros(5))
        #     tp0 = vcat([0,0], [(-L*dx3_0)/(x2_0^2)], zeros(4))
        # end

        x0  = vcat(pend0, sm0, sL0, sk0, r0, r0, r0)
        dx0 = vcat(dpend0, dsm0, dsL0, dsk0, rp0, rp0, rp0)
        # x0 = vcat([x1_0, x2_0], zeros(4), [atan(x1_0 / -x2_0)], zeros(14))
        # dx0 = vcat([0., 0., dx3_0, dx4_0], zeros(17))

        dvars = repeat(vcat(fill(true, 6), [false]), 7)

        r0 = zeros(length(x0))
        f!(r0, dx0, x0, [], 0.0)

        # t -> 0.0 is just a dummy function, not to be used
        Model(f!, t -> 0.0, x0, dx0, dvars, r0)
    end
end

function pendulum_sensitivity_sans_g_with_dist_sens_1(Φ::Float64, u::Function, w::Function, θ::Array{Float64, 1})::Model
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

            # Sensitivity with respect to L
            res[15]  = -x[18] + dx[15] + 2x[1]*dx[20] + 2x[15]*dx[6]
            res[16]  = -x[19] + dx[16] + 2x[2]*dx[20] + 2x[16]*dx[6]
            res[17] = 2k*x[18]*abs(x[4]) - x[1]*dx[17] + m*dx[18] - x[15]*dx[3]
            res[18] = 2k*x[19]*abs(x[5]) - x[2]*dx[17] + m*dx[19] - x[16]*dx[3]
            res[19] = -2x[15]*x[1] - 2x[16]*x[2] + 2L
            res[20] = x[18]*x[1] + x[19]*x[2] + x[15]*x[4] + x[16]*x[5]
            res[21] = x[21] - (x[1]*x[16] - x[2]*x[15])/(L^2)

            # Sensitivity with respect to k
            res[22]  = -x[25] + dx[22] + 2x[1]*dx[27] + 2x[22]*dx[6]
            res[23]  = -x[26] + dx[23] + 2x[2]*dx[27] + 2x[23]*dx[6]
            res[24] = 2k*x[25]*abs(x[4]) - x[1]*dx[24] + m*dx[25] - x[22]*dx[3] + abs(x[4])x[4]
            res[25] = 2k*x[26]*abs(x[5]) - x[2]*dx[24] + m*dx[26] - x[23]*dx[3] + abs(x[5])x[5]
            res[26] = -2x[22]*x[1] - 2x[23]*x[2]
            res[27] = x[25]*x[1] + x[26]*x[2] + x[22]*x[4] + x[23]*x[5]
            res[28] = x[28] - (x[1]*x[23] - x[2]*x[22])/(L^2)

            # Sensitivty wrt first disturbance parameter
            # (i.e. parameter corresponding to wt[2])
            res[29] = 2dx[6]x[29] - x[32] + dx[29] + 2x[1]dx[34]
            res[30] = 2dx[6]x[30] - x[33] + dx[30] + 2x[2]dx[34]
            res[31] = -dx[3]x[29] + 2k*abs(x[4])*x[32] - x[1]*dx[31] + m*dx[32] - 2*wt[1]*wt[2]
            res[32] = -dx[3]x[30] + 2k*abs(x[5])*x[33] - x[2]*dx[31] + m*dx[33]
            res[33] = 2x[1]x[29] + 2x[2]x[30]
            res[34] = x[4]x[29] + x[5]x[30] + x[1]x[32] + x[2]x[33]
            # Sensitivity of angle of pendulum to disturbance parameter
            # TODO: Analytical formula says it should be x[1]^2+x[2]^2 instead of
            # L^2 (though they should be equal), is it fine to substitute L^2 here?
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
        dpend0 = vcat([0., 0., dx3_0, dx4_0], zeros(3))
        sm0  = zeros(7)
        dsm0 = vcat(zeros(3), [-dx4_0/m, -g/m, 0., 0.])
        sL0  = vcat([x1_0/L, x2_0/L], zeros(5))
        dsL0 = vcat([0.,0., -dx3_0/L], zeros(4))
        sk0  = zeros(7)
        dsk0 = zeros(7)
        r0 = zeros(7)
        rp0 = zeros(7)
        # t0 = vcat([sin(Φ), -cos(Φ)], zeros(5))
        # tp0 = vcat([0.,0.], [-dx3_0/L], zeros(4))
        # if x1_0 != 0.0
        #     t0  = vcat([L/(2x1_0), L/(2x2_0)], zeros(4), [(x1_0^2-x2_0^2)/(2L*x1_0*x2_0)])
        #     tp0 = vcat([0,0], [(-L*dx3_0)/(2x1_0*x2_0)], zeros(4))
        # else
        #     t0  = vcat([0.0, L/x2_0], zeros(5))
        #     tp0 = vcat([0,0], [(-L*dx3_0)/(x2_0^2)], zeros(4))
        # end

        x0  = vcat(pend0, sm0, sL0, sk0, r0)
        dx0 = vcat(dpend0, dsm0, dsL0, dsk0, rp0)
        # x0 = vcat([x1_0, x2_0], zeros(4), [atan(x1_0 / -x2_0)], zeros(14))
        # dx0 = vcat([0., 0., dx3_0, dx4_0], zeros(17))

        dvars = repeat(vcat(fill(true, 6), [false]), 5)

        r0 = zeros(length(x0))
        f!(r0, dx0, x0, [], 0.0)

        # t -> 0.0 is just a dummy function, not to be used
        Model(f!, t -> 0.0, x0, dx0, dvars, r0)
    end
end

function pendulum_sensitivity_full_with_dist_sens_2(Φ::Float64, u::Function, w::Function, θ::Array{Float64, 1})::Model
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

            # Sensitivity with respect to L
            res[15]  = -x[18] + dx[15] + 2x[1]*dx[20] + 2x[15]*dx[6]
            res[16]  = -x[19] + dx[16] + 2x[2]*dx[20] + 2x[16]*dx[6]
            res[17] = 2k*x[18]*abs(x[4]) - x[1]*dx[17] + m*dx[18] - x[15]*dx[3]
            res[18] = 2k*x[19]*abs(x[5]) - x[2]*dx[17] + m*dx[19] - x[16]*dx[3]
            res[19] = -2x[15]*x[1] - 2x[16]*x[2] + 2L
            res[20] = x[18]*x[1] + x[19]*x[2] + x[15]*x[4] + x[16]*x[5]
            res[21] = x[21] - (x[1]*x[16] - x[2]*x[15])/(L^2)

            # Sensitivity with respect to g
            res[22]  = -x[25] + dx[22] + 2x[1]*dx[27] + 2x[22]*dx[6]
            res[23]  = -x[26] + dx[23] + 2x[2]*dx[27] + 2x[23]*dx[6]
            res[24] = 2k*x[25]*abs(x[4]) - x[1]*dx[24] + m*dx[25] - x[22]*dx[3]
            res[25] = 2k*x[26]*abs(x[5]) - x[2]*dx[24] + m*dx[26] - x[23]*dx[3] + m
            res[26] = -2x[22]*x[1] - 2x[23]*x[2]
            res[27] = x[25]*x[1] + x[26]*x[2] + x[22]*x[4] + x[23]*x[5]
            res[28] = x[28] - (x[1]*x[23] - x[2]*x[22])/(L^2)

            # Sensitivity with respect to k
            res[29]  = -x[32] + dx[29] + 2x[1]*dx[34] + 2x[29]*dx[6]
            res[30]  = -x[33] + dx[30] + 2x[2]*dx[34] + 2x[30]*dx[6]
            res[31] = 2k*x[32]*abs(x[4]) - x[1]*dx[31] + m*dx[32] - x[29]*dx[3] + abs(x[4])x[4]
            res[32] = 2k*x[33]*abs(x[5]) - x[2]*dx[31] + m*dx[33] - x[30]*dx[3] + abs(x[5])x[5]
            res[33] = -2x[29]*x[1] - 2x[30]*x[2]
            res[34] = x[32]*x[1] + x[33]*x[2] + x[29]*x[4] + x[30]*x[5]
            res[35] = x[35] - (x[1]*x[30] - x[2]*x[29])/(L^2)

            # 22->36, 23->37, 24->38, 25->39, 26->40, 27->41, 28->42

            # DISTURBANCE SENSITIVITIES
            res[36] = 2dx[6]x[36] - x[39] + dx[36] + 2x[1]dx[41]
            res[37] = 2dx[6]x[37] - x[40] + dx[37] + 2x[2]dx[41]
            res[38] = -dx[3]x[36] + 2k*abs(x[4])*x[39] - x[1]*dx[38] + m*dx[39] - 2*wt[1]*wt[2]
            res[39] = -dx[3]x[37] + 2k*abs(x[5])*x[40] - x[2]*dx[38] + m*dx[40]
            res[40] = -2x[1]x[36] - 2x[2]x[37]
            res[41] = x[4]x[36] + x[5]x[37] + x[1]x[39] + x[2]x[40]
            # Sensitivity of angle of pendulum to disturbance parameter
            # TODO: Analytical formula says it should be x[1]^2+x[2]^2 instead of
            # L^2 (though they should be equal), is it fine to substitute L^2 here?
            res[42] = x[42] - (x[1]*x[37] - x[2]*x[36])/(L^2)

            # 29->43, 30->44, 31->45, 32->46, 33->47, 34->48, 35->49

            res[43] = 2dx[6]x[43] - x[46] + dx[43] + 2x[1]dx[48]
            res[44] = 2dx[6]x[44] - x[47] + dx[44] + 2x[2]dx[47]
            res[45] = -dx[3]x[43] + 2k*abs(x[4])*x[46] - x[1]*dx[45] + m*dx[46] - 2*wt[1]*wt[3]
            res[46] = -dx[3]x[44] + 2k*abs(x[5])*x[47] - x[2]*dx[45] + m*dx[47]
            res[47] = -2x[1]x[43] - 2x[2]x[44]
            # res[26] = 2L -2x[15]x[1] - 2x[16]x[2]
            res[48] = x[4]x[43] + x[5]x[44] + x[1]x[46] + x[2]x[47]
            # Sensitivity of angle of pendulum to disturbance parameter
            # TODO: Analytical formula says it should be x[1]^2+x[2]^2 instead of
            # L^2 (though they should be equal), is it fine to substitute L^2 here?
            res[49] = x[49] - (x[1]*x[44] - x[2]*x[43])/(L^2)

            # The default template used to generate these equations was:
            # res[8]  = -s[4] + sp[1] + 2x[1]*sp[6] + 2s[1]*dx[6]
            # res[9]  = -s[5] + sp[2] + 2x[2]*sp[6] + 2s[2]*dx[6]
            # res[10]  = 2k*s[4]*abs(x[4]) - x[1]*sp[3] + m*sp[4] - s[1]*dx[3]
            # res[11] = 2k*s[5]*abs(x[5]) - x[2]*sp[3] + m*sp[5] - s[2]*dx[3]
            # res[12] = -2s[1]*x[1] - 2s[2]*x[2]
            # res[13] = s[4]*x[1] + s[5]*x[2] + s[1]*x[4] + s[2]*x[5]
            # res[14] = x[14] - (x[1]*s[2] - x[2]*s[1])/(L^2)
            # Note that these edxressions don't include parameter-specific terms

            # To obtain the equations written on the form suitable for this
            # function, use the following substitutions:
            # sm[1] -> x[8],  sm[2] -> x[9],  sm[3] -> x[10]
            # sm[4] -> x[11], sm[5] -> x[12], sm[6] -> x[13]
            # sL[1] -> x[15],  sL[2] -> x[16],  sL[3] -> x[17]
            # sL[4] -> x[18], sL[5] -> x[19], sL[6] -> x[20]
            # sg[1] -> x[22],  sg[2] -> x[23],  sg[3] -> x[24]
            # sg[4] -> x[25], sg[5] -> x[26], sg[6] -> x[27]
            # sk[1] -> x[29],  sk[2] -> x[30],  sk[3] -> x[31]
            # sk[4] -> x[32], sk[5] -> x[33], sk[6] -> x[34]

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
        dpend0 = vcat([0., 0., dx3_0, dx4_0], zeros(3))
        sm0  = zeros(7)
        dsm0 = vcat(zeros(3), [-dx4_0/m, -g/m, 0., 0.])
        sL0  = vcat([x1_0/L, x2_0/L], zeros(5))
        dsL0 = vcat([0.,0., -dx3_0/L], zeros(4))
        sg0  = zeros(7)
        sgp0 = vcat(zeros(4), [-1., 0., 0.])
        sk0  = zeros(7)
        dsk0 = zeros(7)
        # t0 = vcat([sin(Φ), -cos(Φ)], zeros(5))
        # tp0 = vcat([0.,0.], [-dx3_0/L], zeros(4))
        # if x1_0 != 0.0
        #     t0  = vcat([L/(2x1_0), L/(2x2_0)], zeros(4), [(x1_0^2-x2_0^2)/(2L*x1_0*x2_0)])
        #     tp0 = vcat([0,0], [(-L*dx3_0)/(2x1_0*x2_0)], zeros(4))
        # else
        #     t0  = vcat([0.0, L/x2_0], zeros(5))
        #     tp0 = vcat([0,0], [(-L*dx3_0)/(x2_0^2)], zeros(4))
        # end

        x0  = vcat(pend0, sm0, sL0, sg0, sk0, zeros(7))
        dx0 = vcat(dpend0, dsm0, dsL0, sgp0, dsk0, zeros(7))
        # x0 = vcat([x1_0, x2_0], zeros(4), [atan(x1_0 / -x2_0)], zeros(14))
        # dx0 = vcat([0., 0., dx3_0, dx4_0], zeros(17))

        dvars = repeat(vcat(fill(true, 6), [false]), 6)

        r0 = zeros(length(x0))
        f!(r0, dx0, x0, [], 0.0)

        # t -> 0.0 is just a dummy function, not to be used
        Model(f!, t -> 0.0, x0, dx0, dvars, r0)
    end
end

function pendulum_sensitivity_Lk_with_dist_sens_1(Φ::Float64, u::Function, w_comp::Function, θ::Array{Float64, 1})::Model
    let m = θ[1], L = θ[2], g = θ[3], k = θ[4]
        # NOTE: In this function w_comp is edxected to have two elements: the
        # first should just be the disturbance w, and the second the sensitivity
        # of the disturbance w to the disturbance model parameter

        # the residual function
        function f!(res, dx, x, θ, t)
            wt = w_comp(t)
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
            # Sensitivity equations for L
            res[8]  = dx[8] - x[11] + 2x[8]dx[6] + 2dx[13]x[1]
            res[9]  = dx[9] - x[12] + 2x[9]dx[6] + 2dx[13]x[2]
            res[10] = 2k*x[11]*abs(x[4]) + m*dx[11] - x[8]dx[3] - dx[10]x[1]
            res[11] = 2k*x[12]*abs(x[5]) + m*dx[12] - x[9]dx[3] - dx[10]x[2]
            res[12] = 2x[8]x[1] + 2x[9]x[2] - 2L
            res[13] = x[8]x[4] + x[11]x[1] + x[9]x[5] + x[12]x[2]
            # Sensitivity of angle of pendulum to L
            # TODO: Analytical formula says it should be x[1]^2+x[2]^2 instead of
            # L^2 (though they should be equal), is it fine to substitute L^2 here?
            res[14] = x[14] - (x[1]*x[9] - x[8]*x[2])/(L^2)

            # Sensitivity equations for k
            res[15]  = -x[18] + dx[15] + 2x[1]*dx[20] + 2x[15]*dx[6]
            res[16]  = -x[19] + dx[16] + 2x[2]*dx[20] + 2x[16]*dx[6]
            res[17] = abs(x[4])*x[4] + 2k*x[18]*abs(x[4]) - x[1]*dx[17] + m*dx[18] - x[15]*dx[3]
            res[18] = abs(x[5])*x[5] + 2k*x[19]*abs(x[5]) - x[2]*dx[17] + m*dx[19] - x[16]*dx[3]
            res[19] = 2x[15]*x[1] + 2x[16]*x[2]
            res[20] = x[18]*x[1] + x[19]*x[2] + x[15]*x[4] + x[16]*x[5]
            # Sensitivity of angle of pendulum to k
            # TODO: Analytical formula says it should be x[1]^2+x[2]^2 instead of
            # L^2 (though they should be equal), is it fine to substitute L^2 here?
            res[21] = x[21] - (x[1]*x[16] - x[15]*x[2])/(L^2)

            # Sensitivty wrt first disturbance parameter
            # (i.e. parameter corresponding to wt[2])
            res[22] = 2dx[6]x[22] - x[25] + dx[22] + 2x[1]dx[27]
            res[23] = 2dx[6]x[23] - x[26] + dx[23] + 2x[2]dx[27]
            res[24] = -dx[3]x[22] + 2k*abs(x[4])*x[25] - x[1]*dx[24] + m*dx[25] - 2*wt[1]*wt[2]
            res[25] = -dx[3]x[23] + 2k*abs(x[5])*x[26] - x[2]*dx[24] + m*dx[26]
            res[26] = 2x[1]x[22] + 2x[2]x[23]
            res[27] = x[4]x[22] + x[5]x[23] + x[1]x[25] + x[2]x[26]
            # Sensitivity of angle of pendulum to disturbance parameter
            # TODO: Analytical formula says it should be x[1]^2+x[2]^2 instead of
            # L^2 (though they should be equal), is it fine to substitute L^2 here?
            res[28] = x[28] - (x[1]*x[23] - x[2]*x[22])/(L^2)

            nothing
        end

        # Finding consistent initial conditions
        # Initial values, the pendulum starts at rest
        u0 = u(0.0)[1]
        w0 = w_comp(0.0)[1]
        x1_0 = L * sin(Φ)
        x2_0 = -L * cos(Φ)
        dx3_0 = m*g/x2_0
        dx4_0 = -g*tan(Φ) + (u0 + w0^2)/m

        pend0 = vcat([x1_0, x2_0], zeros(4), [atan(x1_0 / -x2_0)])
        dpend0 = vcat([0., 0., dx3_0, dx4_0], zeros(3))
        # pend_dvar = vcat(fill(true, 6))
        s0  = zeros(7)
        sp0 = zeros(7)
        t0 = vcat([sin(Φ), -cos(Φ)], zeros(5))
        tp0 = vcat([0.,0.], [-dx3_0/L], zeros(4))
        r0 = zeros(7)
        rp0 = zeros(7)
        # if x1_0 != 0.0
        #     t0  = vcat([L/(2x1_0), L/(2x2_0)], zeros(4), [(x1_0^2-x2_0^2)/(2L*x1_0*x2_0)])
        #     tp0 = vcat([0,0], [(-L*dx3_0)/(2x1_0*x2_0)], zeros(4))
        # else
        #     t0  = vcat([0.0, L/x2_0], zeros(5))
        #     tp0 = vcat([0,0], [(-L*dx3_0)/(x2_0^2)], zeros(4))
        # end

        x0  = vcat(pend0, t0, s0, r0)
        dx0 = vcat(dpend0, tp0, sp0, rp0)
        # x0 = vcat([x1_0, x2_0], zeros(4), [atan(x1_0 / -x2_0)], zeros(14))
        # dx0 = vcat([0., 0., dx3_0, dx4_0], zeros(17))

        dvars = vcat(fill(true, 6), [false], fill(true, 6), [false], fill(true, 6), [false], fill(true, 6), [false])

        r0 = zeros(length(x0))
        f!(r0, dx0, x0, [], 0.0)

        # t -> 0.0 is just a dummy function, not to be used
        Model(f!, t -> 0.0, x0, dx0, dvars, r0)
    end
end

function pendulum_sensitivity_Lk_with_dist_sens_2(Φ::Float64, u::Function, w_comp::Function, θ::Array{Float64, 1})::Model
    let m = θ[1], L = θ[2], g = θ[3], k = θ[4]
        # NOTE: In this function w_comp is edxected to have two elements: the
        # first should just be the disturbance w, and the second the sensitivity
        # of the disturbance w to the disturbance model parameter

        # the residual function
        function f!(res, dx, x, θ, t)
            wt = w_comp(t)
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
            # Sensitivity equations for L
            res[8]  = dx[8] - x[11] + 2x[8]dx[6] + 2dx[13]x[1]
            res[9]  = dx[9] - x[12] + 2x[9]dx[6] + 2dx[13]x[2]
            res[10] = 2k*x[11]*abs(x[4]) + m*dx[11] - x[8]dx[3] - dx[10]x[1]
            res[11] = 2k*x[12]*abs(x[5]) + m*dx[12] - x[9]dx[3] - dx[10]x[2]
            res[12] = 2x[8]x[1] + 2x[9]x[2] - 2L
            res[13] = x[8]x[4] + x[11]x[1] + x[9]x[5] + x[12]x[2]
            # Sensitivity of angle of pendulum to L
            # TODO: Analytical formula says it should be x[1]^2+x[2]^2 instead of
            # L^2 (though they should be equal), is it fine to substitute L^2 here?
            res[14] = x[14] - (x[1]*x[9] - x[8]*x[2])/(L^2)

            # Sensitivity equations for k
            res[15]  = -x[18] + dx[15] + 2x[1]*dx[20] + 2x[15]*dx[6]
            res[16]  = -x[19] + dx[16] + 2x[2]*dx[20] + 2x[16]*dx[6]
            res[17] = abs(x[4])*x[4] + 2k*x[18]*abs(x[4]) - x[1]*dx[17] + m*dx[18] - x[15]*dx[3]
            res[18] = abs(x[5])*x[5] + 2k*x[19]*abs(x[5]) - x[2]*dx[17] + m*dx[19] - x[16]*dx[3]
            res[19] = 2x[15]*x[1] + 2x[16]*x[2]
            res[20] = x[18]*x[1] + x[19]*x[2] + x[15]*x[4] + x[16]*x[5]
            # Sensitivity of angle of pendulum to k
            # TODO: Analytical formula says it should be x[1]^2+x[2]^2 instead of
            # L^2 (though they should be equal), is it fine to substitute L^2 here?
            res[21] = x[21] - (x[1]*x[16] - x[15]*x[2])/(L^2)

            # Sensitivty wrt first disturbance parameter
            # (i.e. parameter corresponding to wt[2])
            res[22] = 2dx[6]x[22] - x[25] + dx[22] + 2x[1]dx[27]
            res[23] = 2dx[6]x[23] - x[26] + dx[23] + 2x[2]dx[27]
            res[24] = -dx[3]x[22] + 2k*abs(x[4])*x[25] - x[1]*dx[24] + m*dx[25] - 2*wt[1]*wt[2]
            res[25] = -dx[3]x[23] + 2k*abs(x[5])*x[26] - x[2]*dx[24] + m*dx[26]
            res[26] = 2x[1]x[22] + 2x[2]x[23]
            res[27] = x[4]x[22] + x[5]x[23] + x[1]x[25] + x[2]x[26]
            # Sensitivity of angle of pendulum to disturbance parameter
            # TODO: Analytical formula says it should be x[1]^2+x[2]^2 instead of
            # L^2 (though they should be equal), is it fine to substitute L^2 here?
            res[28] = x[28] - (x[1]*x[23] - x[2]*x[22])/(L^2)

            # Sensitivty wrt second disturbance parameter
            # (i.e. parameter corresponding to wt[3])
            res[29] = 2dx[6]x[29] - x[32] + dx[29] + 2x[1]dx[34]
            res[30] = 2dx[6]x[30] - x[33] + dx[30] + 2x[2]dx[34]
            res[31] = -dx[3]x[29] + 2k*abs(x[4])*x[32] - x[1]*dx[31] + m*dx[32] - 2*wt[1]*wt[3]
            res[32] = -dx[3]x[30] + 2k*abs(x[5])*x[33] - x[2]*dx[31] + m*dx[33]
            res[33] = 2x[1]x[29] + 2x[2]x[30]
            res[34] = x[4]x[29] + x[5]x[30] + x[1]x[32] + x[2]x[33]
            # Sensitivity of angle of pendulum to disturbance parameter
            # TODO: Analytical formula says it should be x[1]^2+x[2]^2 instead of
            # L^2 (though they should be equal), is it fine to substitute L^2 here?
            res[35] = x[35] - (x[1]*x[30] - x[2]*x[29])/(L^2)

            nothing
        end

        # Finding consistent initial conditions
        # Initial values, the pendulum starts at rest
        u0 = u(0.0)[1]
        w0 = w_comp(0.0)[1]
        x1_0 = L * sin(Φ)
        x2_0 = -L * cos(Φ)
        dx3_0 = m*g/x2_0
        dx4_0 = -g*tan(Φ) + (u0 + w0^2)/m

        pend0 = vcat([x1_0, x2_0], zeros(4), [atan(x1_0 / -x2_0)])
        dpend0 = vcat([0., 0., dx3_0, dx4_0], zeros(3))
        # pend_dvar = vcat(fill(true, 6))
        s0  = zeros(7)
        sp0 = zeros(7)
        t0 = vcat([sin(Φ), -cos(Φ)], zeros(5))
        tp0 = vcat([0.,0.], [-dx3_0/L], zeros(4))
        r0 = zeros(7)
        rp0 = zeros(7)
        # if x1_0 != 0.0
        #     t0  = vcat([L/(2x1_0), L/(2x2_0)], zeros(4), [(x1_0^2-x2_0^2)/(2L*x1_0*x2_0)])
        #     tp0 = vcat([0,0], [(-L*dx3_0)/(2x1_0*x2_0)], zeros(4))
        # else
        #     t0  = vcat([0.0, L/x2_0], zeros(5))
        #     tp0 = vcat([0,0], [(-L*dx3_0)/(x2_0^2)], zeros(4))
        # end

        x0  = vcat(pend0, t0, s0, r0, r0)
        dx0 = vcat(dpend0, tp0, sp0, rp0, rp0)
        # x0 = vcat([x1_0, x2_0], zeros(4), [atan(x1_0 / -x2_0)], zeros(14))
        # dx0 = vcat([0., 0., dx3_0, dx4_0], zeros(17))

        dvars = vcat(fill(true, 6), [false], fill(true, 6), [false], fill(true, 6), [false], fill(true, 6), [false], fill(true, 6), [false])

        r0 = zeros(length(x0))
        f!(r0, dx0, x0, [], 0.0)

        # t -> 0.0 is just a dummy function, not to be used
        Model(f!, t -> 0.0, x0, dx0, dvars, r0)
    end
end

function pendulum_sensitivity_k_with_dist_sens_1(Φ::Float64, u::Function, w::Function, θ::Array{Float64, 1})::Model
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

        # Finding consistent initial conditions
        # Initial values, the pendulum starts at rest
        u0 = u(0.0)[1]
        w0 = w(0.0)[1]
        x1_0 = L * sin(Φ)
        x2_0 = -L * cos(Φ)
        dx3_0 = m*g/x2_0
        dx4_0 = -g*tan(Φ) + (u0 + w0^2)/m

        pend0 = vcat([x1_0, x2_0], zeros(4), [atan(x1_0 / -x2_0)])
        dpend0 = vcat([0., 0., dx3_0, dx4_0], zeros(3))
        sk0  = zeros(7)
        dsk0 = zeros(7)
        r0 = zeros(7)
        rp0 = zeros(7)

        x0  = vcat(pend0, sk0, r0)
        dx0 = vcat(dpend0, dsk0, rp0)
        # x0 = vcat([x1_0, x2_0], zeros(4), [atan(x1_0 / -x2_0)], zeros(14))
        # dx0 = vcat([0., 0., dx3_0, dx4_0], zeros(17))

        dvars = repeat(vcat(fill(true, 6), [false]), 3)

        r0 = zeros(length(x0))
        f!(r0, dx0, x0, [], 0.0)

        # t -> 0.0 is just a dummy function, not to be used
        Model(f!, t -> 0.0, x0, dx0, dvars, r0)
    end
end

function pendulum_sensitivity_k_with_dist_sens_2(Φ::Float64, u::Function, w::Function, θ::Array{Float64, 1})::Model
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

            # Sensitivty wrt second disturbance parameter
            # (i.e. parameter corresponding to wt[2])
            res[22] = 2dx[6]x[22] - x[25] + dx[22] + 2x[1]dx[27]
            res[23] = 2dx[6]x[23] - x[26] + dx[23] + 2x[2]dx[27]
            res[24] = -dx[3]x[22] + 2k*abs(x[4])*x[25] - x[1]*dx[24] + m*dx[25] - 2*wt[1]*wt[3]
            res[25] = -dx[3]x[23] + 2k*abs(x[5])*x[26] - x[2]*dx[24] + m*dx[26]
            res[26] = 2x[1]x[22] + 2x[2]x[23]
            res[27] = x[4]x[22] + x[5]x[23] + x[1]x[25] + x[2]x[26]
            # Sensitivity of angle of pendulum to disturbance parameter
            # TODO: Analytical formula says it should be x[1]^2+x[2]^2 instead of
            # L^2 (though they should be equal), is it fine to substitute L^2 here?
            res[28] = x[28] - (x[1]*x[23] - x[2]*x[22])/(L^2)

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
        dpend0 = vcat([0., 0., dx3_0, dx4_0], zeros(3))
        sk0  = zeros(7)
        dsk0 = zeros(7)
        r0 = zeros(7)
        rp0 = zeros(7)

        x0  = vcat(pend0, sk0, r0, r0)
        dx0 = vcat(dpend0, dsk0, rp0, rp0)
        # x0 = vcat([x1_0, x2_0], zeros(4), [atan(x1_0 / -x2_0)], zeros(14))
        # dx0 = vcat([0., 0., dx3_0, dx4_0], zeros(17))

        dvars = repeat(vcat(fill(true, 6), [false]), 4)

        r0 = zeros(length(x0))
        f!(r0, dx0, x0, [], 0.0)

        # t -> 0.0 is just a dummy function, not to be used
        Model(f!, t -> 0.0, x0, dx0, dvars, r0)
    end
end

function pendulum_sensitivity_k_with_dist_sens_3(Φ::Float64, u::Function, w::Function, θ::Array{Float64, 1})::Model
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

            # Sensitivty wrt second disturbance parameter
            # (i.e. parameter corresponding to wt[2])
            res[22] = 2dx[6]x[22] - x[25] + dx[22] + 2x[1]dx[27]
            res[23] = 2dx[6]x[23] - x[26] + dx[23] + 2x[2]dx[27]
            res[24] = -dx[3]x[22] + 2k*abs(x[4])*x[25] - x[1]*dx[24] + m*dx[25] - 2*wt[1]*wt[3]
            res[25] = -dx[3]x[23] + 2k*abs(x[5])*x[26] - x[2]*dx[24] + m*dx[26]
            res[26] = 2x[1]x[22] + 2x[2]x[23]
            res[27] = x[4]x[22] + x[5]x[23] + x[1]x[25] + x[2]x[26]
            # Sensitivity of angle of pendulum to disturbance parameter
            # TODO: Analytical formula says it should be x[1]^2+x[2]^2 instead of
            # L^2 (though they should be equal), is it fine to substitute L^2 here?
            res[28] = x[28] - (x[1]*x[23] - x[2]*x[22])/(L^2)

            # Sensitivty wrt third disturbance parameter
            # (i.e. parameter corresponding to wt[2])
            res[29] = 2dx[6]x[29] - x[32] + dx[29] + 2x[1]dx[34]
            res[30] = 2dx[6]x[30] - x[33] + dx[30] + 2x[2]dx[34]
            res[31] = -dx[3]x[29] + 2k*abs(x[4])*x[32] - x[1]*dx[31] + m*dx[32] - 2*wt[1]*wt[4]
            res[32] = -dx[3]x[30] + 2k*abs(x[5])*x[33] - x[2]*dx[31] + m*dx[33]
            res[33] = 2x[1]x[29] + 2x[2]x[30]
            res[34] = x[4]x[29] + x[5]x[30] + x[1]x[32] + x[2]x[33]
            # Sensitivity of angle of pendulum to disturbance parameter
            # TODO: Analytical formula says it should be x[1]^2+x[2]^2 instead of
            # L^2 (though they should be equal), is it fine to substitute L^2 here?
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
        dpend0 = vcat([0., 0., dx3_0, dx4_0], zeros(3))
        sk0  = zeros(7)
        dsk0 = zeros(7)
        r0 = zeros(7)
        rp0 = zeros(7)

        x0  = vcat(pend0, sk0, r0, r0, r0)
        dx0 = vcat(dpend0, dsk0, rp0, rp0, rp0)
        # x0 = vcat([x1_0, x2_0], zeros(4), [atan(x1_0 / -x2_0)], zeros(14))
        # dx0 = vcat([0., 0., dx3_0, dx4_0], zeros(17))

        dvars = repeat(vcat(fill(true, 6), [false]), 5)

        r0 = zeros(length(x0))
        f!(r0, dx0, x0, [], 0.0)

        # t -> 0.0 is just a dummy function, not to be used
        Model(f!, t -> 0.0, x0, dx0, dvars, r0)
    end
end

function my_pendulum_adjoint(u::Function, w::Function, θ::Vector{Float64}, T::Float64, sol::DAESolution, sol2::DAESolution, y::Function, dy::Function, xp0::Vector{Float64}, dx::Function, dx2::Function, N_trans::Int=0)
    # NOTE: A bit ugly to pass sol and sol2 as DAESolution, but dx as a function.
    # But good enough for now, just should be different in final version perhaps
    let m = θ[1], L = θ[2], g = θ[3], k = θ[4]
        np = size(xp0,2)
        @assert (np == 3) "my_pendulum_adjoint is hard-coded to only handle exactly three parameters (all dynamical), make sure to pass correct xp0"
        nx = size(xp0,1)
        x  = t -> sol(t)
        x2 = t -> sol2(t)

        Fx = t -> [2dx(t)[6]        0               0   -1                  0           0   0
                    0           2*dx(t)[6]          0   0                   -1          0   0
                    -dx(t)[3]         0             0   2k*abs(x(t)[4])     0           0   0
                    0            -dx(t)[3]          0   0               2k*abs(x(t)[5]) 0   0
                    2x(t)[1]      2x(t)[2]          0   0                   0           0   0
                    x(t)[4]        x(t)[5]          0   x(t)[1]            x(t)[2]      0   0
                    x(t)[2]/(L^2)  -x(t)[1]/(L^2)   0   0                   0           0   1]
        # (namely the derivative of F with respect to the variable x_p)
        Fdx = t -> vcat([1   0   0          0   0   2x(t)[1]    0
                         0   1   0          0   0   2x(t)[2]    0
                         0   0   -x(t)[1]   m   0   0           0
                         0   0   -x(t)[2]   0   m   0           0], zeros(3,7))
        Fddx = t -> vcat([  0   0  0            0   0   2dx(t)[1]    0
                            0   0  0            0   0   2dx(t)[2]    0
                            0   0  -dx(t)[1]    0   0   0            0
                            0   0  -dx(t)[2]    0   0   0            0], zeros(3,7))
        Fp = t -> [ .0          .0          .0
                    .0          .0          .0
                    dx(t)[4]    .0  abs(x(t)[4])*x(t)[4]
                    dx(t)[5]+g  .0  abs(x(t)[5])*x(t)[5]
                    .0           L          .0
                    .0          .0          .0
                    .0          .0          .0]
        gₓ  = t -> [0    0    0    0    0    0    2(x2(t)[7]-y(t))/T]
        gdₓ = t -> [0    0    0    0    0    0    2(dx2(t)[7]-dy(t))/T]

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

        # the residual function
        # NOTE: Could time-varying coefficients be the problem?? Sure would require more allocations?
        # TODO: If that is the case, we could store x(t) in a static array to avoid re-allocations?
        function f!(res, dz, z, θ, t)
            # TODO: Complete
            # # Completely unreadabe but most efficient version (still not a huge improvement)
            # res[1]  = dx(T-t)[3]*z[3] - dz[1] - 2*dx(T-t)[6]*z[1] + 2*z[5]*x(T-t)[1] - z[6]*x(T-t)[4] - (z[7]*x(T-t)[2])/(L^2)
            # res[2]  = dx(T-t)[3]*z[4] - dz[2] - 2*dx(T-t)[6]*z[2] + 2*z[5]*x(T-t)[2] - z[6]*x(T-t)[5] + (z[7]*x(T-t)[1])/(L^2)
            # res[3]  = dz[3]*x(T-t)[1] - dx(T-t)[2]*z[4] - dx(T-t)[1]*z[3] + dz[4]*x(T-t)[2]
            # res[4]  = z[1] - dz[3]*m - z[6]*x(T-t)[1] - 2*k*z[3]*abs(x(T-t)[4])
            # res[5]  = z[2] - dz[4]*m - z[6]*x(T-t)[2] - 2*k*z[4]*abs(x(T-t)[5])
            # res[6]  = 2*dx(T-t)[1]*z[1] + 2*dx(T-t)[2]*z[2] - 2*dz[1]*x(T-t)[1] - 2*dz[2]*x(T-t)[2]
            # res[7]  = (2*(x2(T-t)[7] - y(T-t)))/T - z[7]
            # res[8]  = dz[8] + dz[3]*abs(x(T-t)[4])*x(T-t)[4] + dz[4]*abs(x(T-t)[5])*x(T-t)[5]

            # TODO: Complete
            # # Medium readable but quite efficient version
            # λ  = z[1:nx]
            # dλ = dz[1:nx]
            # β  = z[nx+1:end]
            # dβ = dz[nx+1:end]
            # res[1]  = dx(T-t)[3]*λ[3] - dλ[1] - 2*dx(T-t)[6]*λ[1] + 2*λ[5]*x(T-t)[1] - λ[6]*x(T-t)[4] - (λ[7]*x(T-t)[2])/(L^2)
            # res[2]  = dx(T-t)[3]*λ[4] - dλ[2] - 2*dx(T-t)[6]*λ[2] + 2*λ[5]*x(T-t)[2] - λ[6]*x(T-t)[5] + (λ[7]*x(T-t)[1])/(L^2)
            # res[3]  = dλ[3]*x(T-t)[1] - dx(T-t)[2]*λ[4] - dx(T-t)[1]*λ[3] + dλ[4]*x(T-t)[2]
            # res[4]  = λ[1] - dλ[3]*m - λ[6]*x(T-t)[1] - 2*k*λ[3]*abs(x(T-t)[4])
            # res[5]  = λ[2] - dλ[4]*m - λ[6]*x(T-t)[2] - 2*k*λ[4]*abs(x(T-t)[5])
            # res[6]  = 2*dx(T-t)[1]*λ[1] + 2*dx(T-t)[2]*λ[2] - 2*dλ[1]*x(T-t)[1] - 2*dλ[2]*x(T-t)[2]
            # res[7]  = (2*(x2(T-t)[7] - y(T-t)))/T - λ[7]
            # res[8]  = dβ[1] + dλ[3]*abs(x(T-t)[4])*x(T-t)[4] + dλ[4]*abs(x(T-t)[5])*x(T-t)[5]
            # # Readable but slightly less efficient version
            # res[1:7] = -(dλ')*Fdx(T-t) + (λ')*(Fddx(T-t)-Fx(T-t)) + gₓ(T-t)
            # res[8]  = dβ[1]+(λ')*Fp(T-t)

            # Super-readable but less efficient version
            res[1:7]  = (-dz[1:7]')*Fdx(T-t) + (z[1:7]')*(Fddx(T-t) - Fx(T-t)) + gₓ(T-t)
            # res[8:10] = -dz[8:10] - (z[1:7]')*Fp(T-t)
            res[8:10] = -dz[8:10] - (Fp(T-t)')*z[1:7]
            nothing
        end

        # Solving backwards in time =>
        # 1. Initial conditions will equal terminal conditions
        # 2. All derivatives have to be negated
        λ0  = λT[:]
        dλ0 = -dλT[:]
        # z0 = λ0
        # dz0 = dλ0
        z0  = vcat(λ0, zeros(np))
        # dz0 = vcat(dλ0, -((λT')*Fp(T))')
        dz0 = vcat(dλ0, -(Fp(T)')*λT)
        # TODO: Can we delete these row below here perhaps?
        # z0  = vcat(λ0, 0.0)
        # dz0 = vcat(dλ0, λ0[1])   # For some reasing (λT')*Fp(T) returns a scalar instead of a 1-element matrix, unexpected but desired

        if N_trans > 0
            @warn "The returned function get_Gp() doesn't fully support N_trans > 0, as sensitivity of internal variables not known at any other time than t=0. A non-rigorous approximation is used instead."
        end
        # Function returning Gp given adjoint solution
        function get_Gp(adj_sol::DAESolution)
            # NOTE: Changes signs to match what I had in my manual calculations, seems correct now
            # Gp = adj_sol.u[end][end-np+1:end] + (((adj_sol.u[end][1:end-np]')*Fdx(0.0))*xp0)[:]
            Gp = adj_sol.u[end-N_trans][nx+1:nx+np] + (((adj_sol.u[end][1:nx]')*Fdx(0.0))*xp0)[:]
            # return 0.0
        end

        dvars = vcat(fill(true, 4), fill(false, 3), fill(true, np))
        # dvars = vcat(fill(true, 4), fill(false, 3))

        r0 = zeros(length(z0))
        f!(r0, dz0, z0, [], 0.0)
        # @info "r0 is: $r0 here, λ0: $λ0, dλ0: $dλ0"

        # t -> 0.0 is just a dummy function, not to be used
        return Model(f!, t -> 0.0, z0, dz0, dvars, r0), get_Gp
    end
end

function NEGATED_my_pendulum_adjoint_monly(u::Function, w::Function, θ::Vector{Float64}, T::Float64, x::Function, x2::Function, y::Function, dy::Function, xp0::Vector{Float64}, dx::Function, dx2::Function, N_trans::Int=0)
    # NOTE: A bit ugly to pass sol and sol2 as DAESolution, but dx as a function.
    # But good enough for now, just should be different in final version perhaps
    let m = θ[1], L = θ[2], g = θ[3], k = θ[4]
        np = size(xp0,2)
        @assert (np == 1) "my_pendulum_adjoint_monly is hard-coded to only handle one parameter m, make sure to pass correct xp0"
        nx = size(xp0,1)
        # x  = t -> sol(t)
        # x2 = t -> sol2(t)

        Fx = t -> [2dx(t)[6]        0               0   -1                  0           0   0
                    0           2*dx(t)[6]          0   0                   -1          0   0
                    -dx(t)[3]         0             0   2k*abs(x(t)[4])     0           0   0
                    0            -dx(t)[3]          0   0               2k*abs(x(t)[5]) 0   0
                    2x(t)[1]      2x(t)[2]          0   0                   0           0   0
                    x(t)[4]        x(t)[5]          0   x(t)[1]            x(t)[2]      0   0
                    x(t)[2]/(L^2)  -x(t)[1]/(L^2)   0   0                   0           0   1]
        # (namely the derivative of F with respect to the variable x_p)
        Fdx = t -> vcat([1   0   0          0   0   2x(t)[1]    0
                         0   1   0          0   0   2x(t)[2]    0
                         0   0   -x(t)[1]   m   0   0           0
                         0   0   -x(t)[2]   0   m   0           0], zeros(3,7))
        Fddx = t -> vcat([  0   0  0            0   0   2dx(t)[1]    0
                            0   0  0            0   0   2dx(t)[2]    0
                            0   0  -dx(t)[1]    0   0   0            0
                            0   0  -dx(t)[2]    0   0   0            0], zeros(3,7))
        Fp = t -> [ .0
                    .0
                    dx(t)[4]
                    dx(t)[5]+g
                    .0
                    .0
                    .0]
        gₓ  = t -> [0    0    0    0    0    0    2(x2(t)[7]-y(t))/T]
        gdₓ = t -> [0    0    0    0    0    0    2(dx2(t)[7]-dy(t))/T]

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

        # the residual function
        # NOTE: Could time-varying coefficients be the problem?? Sure would require more allocations?
        # TODO: If that is the case, we could store x(t) in a static array to avoid re-allocations?
        function f!(res, dz, z, θ, t)
            # TODO: Complete
            # Completely unreadabe but most efficient version (still not a huge improvement)
            res[1]  = dx(T-t)[3]*z[3] - dz[1] - 2*dx(T-t)[6]*z[1] - 2*z[5]*x(T-t)[1] - z[6]*x(T-t)[4] - (z[7]*x(T-t)[2])/(L^2)
            res[2]  = dx(T-t)[3]*z[4] - dz[2] - 2*dx(T-t)[6]*z[2] - 2*z[5]*x(T-t)[2] - z[6]*x(T-t)[5] + (z[7]*x(T-t)[1])/(L^2)
            res[3]  = dz[3]*x(T-t)[1] - dx(T-t)[2]*z[4] - dx(T-t)[1]*z[3] + dz[4]*x(T-t)[2]
            res[4]  = z[1] - dz[3]*m - z[6]*x(T-t)[1] - 2*k*z[3]*abs(x(T-t)[4])
            res[5]  = z[2] - dz[4]*m - z[6]*x(T-t)[2] - 2*k*z[4]*abs(x(T-t)[5])
            res[6]  = 2*dx(T-t)[1]*z[1] + 2*dx(T-t)[2]*z[2] - 2*dz[1]*x(T-t)[1] - 2*dz[2]*x(T-t)[2]
            res[7]  = (2*(x2(T-t)[7] - y(T-t)))/T - z[7]
            res[8]  = dz[8] + z[3]*dx(T-t)[4] + z[4]*(dx(T-t)[5]+g)

            # # Super-readable but less efficient version
            # res[1:7]  = (-dz[1:7]')*Fdx(T-t) + (z[1:7]')*(Fddx(T-t) - Fx(T-t)) + gₓ(T-t)
            # res[8] = dz[8] + (Fp(T-t)')*z[1:7]
            nothing
        end

        # Solving backwards in time =>
        # 1. Initial conditions will equal terminal conditions
        # 2. All derivatives have to be negated
        λ0  = λT[:]
        dλ0 = -dλT[:]
        # z0 = λ0
        # dz0 = dλ0
        z0  = vcat(λ0, zeros(np))
        # dz0 = vcat(dλ0, -((λT')*Fp(T))')
        dz0 = vcat(dλ0, -(Fp(T)')*λT)
        # TODO: Can we delete these row below here perhaps?
        # z0  = vcat(λ0, 0.0)
        # dz0 = vcat(dλ0, λ0[1])   # For some reasing (λT')*Fp(T) returns a scalar instead of a 1-element matrix, unexpected but desired

        if N_trans > 0
            @warn "The returned function get_Gp() doesn't fully support N_trans > 0, as sensitivity of internal variables not known at any other time than t=0. A non-rigorous approximation is used instead."
        end
        # Function returning Gp given adjoint solution
        function get_Gp(adj_sol::DAESolution)
            # NOTE: Changes signs to match what I had in my manual calculations, seems correct now
            # Gp = adj_sol.u[end][end-np+1:end] + (((adj_sol.u[end][1:end-np]')*Fdx(0.0))*xp0)[:]
            Gp = adj_sol.u[end-N_trans][nx+1:nx+np] .+ (((adj_sol.u[end][1:nx]')*Fdx(0.0))*xp0)
            # return 0.0
        end

        dvars = vcat(fill(true, 4), fill(false, 3), fill(true, np))
        # dvars = vcat(fill(true, 4), fill(false, 3))

        r0 = zeros(length(z0))
        f!(r0, dz0, z0, [], 0.0)
        # @info "r0 is: $r0 here, λ0: $λ0, dλ0: $dλ0"

        # t -> 0.0 is just a dummy function, not to be used
        return Model(f!, t -> 0.0, z0, dz0, dvars, r0), get_Gp
    end
end

function my_pendulum_adjoint_monly(u::Function, w::Function, θ::Vector{Float64}, T::Float64, x::Function, x2::Function, y::Function, dy::Function, xp0::Vector{Float64}, dx::Function, dx2::Function, N_trans::Int=0)
    # NOTE: A bit ugly to pass sol and sol2 as DAESolution, but dx as a function.
    # But good enough for now, just should be different in final version perhaps
    let m = θ[1], L = θ[2], g = θ[3], k = θ[4]
        np = size(xp0,2)
        @assert (np == 1) "my_pendulum_adjoint_monly is hard-coded to only handle one parameter m, make sure to pass correct xp0"
        nx = size(xp0,1)
        # x  = t -> sol(t)
        # x2 = t -> sol2(t)

        Fx = t -> [2dx(t)[6]        0               0   -1                  0           0   0
                    0           2*dx(t)[6]          0   0                   -1          0   0
                    -dx(t)[3]         0             0   2k*abs(x(t)[4])     0           0   0
                    0            -dx(t)[3]          0   0               2k*abs(x(t)[5]) 0   0
                    2x(t)[1]      2x(t)[2]          0   0                   0           0   0
                    x(t)[4]        x(t)[5]          0   x(t)[1]            x(t)[2]      0   0
                    x(t)[2]/(L^2)  -x(t)[1]/(L^2)   0   0                   0           0   1]
        # (namely the derivative of F with respect to the variable x_p)
        Fdx = t -> vcat([1   0   0          0   0   2x(t)[1]    0
                         0   1   0          0   0   2x(t)[2]    0
                         0   0   -x(t)[1]   m   0   0           0
                         0   0   -x(t)[2]   0   m   0           0], zeros(3,7))
        Fddx = t -> vcat([  0   0  0            0   0   2dx(t)[1]    0
                            0   0  0            0   0   2dx(t)[2]    0
                            0   0  -dx(t)[1]    0   0   0            0
                            0   0  -dx(t)[2]    0   0   0            0], zeros(3,7))
        Fp = t -> [ .0
                    .0
                    dx(t)[4]
                    dx(t)[5]+g
                    .0
                    .0
                    .0]
        gₓ  = t -> [0    0    0    0    0    0    2(x2(t)[7]-y(t))/T]
        gdₓ = t -> [0    0    0    0    0    0    2(dx2(t)[7]-dy(t))/T]

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

        # the residual function
        # NOTE: Could time-varying coefficients be the problem?? Sure would require more allocations?
        # TODO: If that is the case, we could store x(t) in a static array to avoid re-allocations?
        function f!(res, dz, z, θ, t)
            # TODO: Complete
            # Completely unreadabe but most efficient version (still not a huge improvement)
            res[1]  = dz[1] - 2*dx(t)[6]*z[1] + dx(t)[3]*z[3] - 2*x(t)[1]*z[5] - x(t)[4]*z[6] - (x(t)[2]*z[7])/(L^2)
            res[2]  = dz[2] - 2*dx(t)[6]*z[2] + dx(t)[3]*z[4] - 2*x(t)[2]*z[5] - x(t)[5]*z[6] + (x(t)[1]*z[7])/(L^2)
            res[3]  = -x(t)[1]*dz[3] - x(t)[2]*dz[4] - dx(t)[1]*z[3] - dx(t)[2]*z[4]
            res[4]  = m*dz[3] + z[1] - 2*k*abs(x(t)[4])*z[3] - x(t)[1]*z[6]
            res[5]  = m*dz[4] + z[2] - 2*k*abs(x(t)[5])*z[4] - x(t)[2]*z[6]
            res[6]  = 2*x(t)[1]*dz[1] + 2*x(t)[2]*dz[2] + 2*dx(t)[1]*z[1] + 2*dx(t)[2]*z[2]
            res[7]  = (2*(x2(t)[7] - y(t)))/T - z[7]
            # res[1]  = dz[1] + 2x(t)[1]*dz[6]    - 2dx(t)[6]*z[1] + z[4] + 2dx(t)[1]*z[6]
            # res[2]  = dz[2] + 2x(t)[2]*dz[6]    - 2dx(t)[6]*z[2] + z[5] + 2dx(t)[2]*z[6]
            # res[3]  = -x(t)[1]*dz[3] + m*dz[4]  + dx(t)[3]*z[1] - dx(t)[1]*z[3] - 2k*abs(x(t)[4])*z[4]
            # res[4]  = -x(t)[2]*dz[3] + m*dz[5]  + dx(t)[3]*z[2] - dx(t)[2]*z[3] - 2k*abs(x(t)[5])*z[5]
            # res[5]  =                           - 2x(t)[1]*z[1] - 2x(t)[2]*z[2]
            # res[6]  =                           - x(t)[4]*z[1] - x(t)[5]*z[2] - x(t)[1]*z[4] - x(t)[2]*z[5]
            # res[7]  =                           - x(t)[2]*z[1]/(L^2) + x(t)[1]*z[2]/(L^2) - z[7]                + (2/T)*(x2(t)[7]-y(t))
            res[8]  = dz[8] - z[3]*dx(t)[4] - z[4]*(dx(t)[5]+g)
            # res[8]  = dz[8] - z[3]*dx(T-t)[4] - z[4]*(dx(T-t)[5]+g)   # NOTE: SIMPLY WRONG; T-t????
            # res[8]  = dz[8] - z[3]*abs(x(t)[4])*x(t)[4] - z[4]*abs(x(t)[5])*x(t)[5]   # from pendelum k-only

            # # Super-readable but less efficient version ALSO NEGATED
            # res[1:7]  = (dz[1:7]')*Fdx(T-t) + (z[1:7]')*(Fddx(T-t) - Fx(T-t)) + gₓ(T-t)
            # res[8] = dz[8] - (Fp(T-t)')*z[1:7]
            nothing
        end

        # z0 = λ0
        # dz0 = dλ0
        z0  = vcat(λT[:], zeros(np))
        # dz0 = vcat(dλ0, -((λT')*Fp(T))')
        dz0 = vcat(dλT[:], -(Fp(T)')*λT)
        # TODO: Can we delete these row below here perhaps?
        # z0  = vcat(λ0, 0.0)
        # dz0 = vcat(dλ0, λ0[1])   # For some reasing (λT')*Fp(T) returns a scalar instead of a 1-element matrix, unexpected but desired

        if N_trans > 0
            @warn "The returned function get_Gp() doesn't fully support N_trans > 0, as sensitivity of internal variables not known at any other time than t=0. A non-rigorous approximation is used instead."
        end
        # Function returning Gp given adjoint solution
        function get_Gp(adj_sol::DAESolution)
            # NOTE: Changes signs to match what I had in my manual calculations, seems correct now
            # Gp = adj_sol.u[end][end-np+1:end] + (((adj_sol.u[end][1:end-np]')*Fdx(0.0))*xp0)[:]
            @warn "adj_sols: $(adj_sol.u[end][nx+1:nx+np]), $(adj_sol.u[1][nx+1:nx+np])"
            Gp = adj_sol.u[end-N_trans][nx+1:nx+np] .+ (((adj_sol.u[end][1:nx]')*Fdx(0.0))*xp0)
            # return 0.0
        end

        dvars = vcat(fill(true, 4), fill(false, 3), fill(true, np))
        # dvars = vcat(fill(true, 4), fill(false, 3))

        r0 = zeros(length(z0))
        f!(r0, dz0, z0, [], 0.0)
        # @info "r0 is: $r0 here, λ0: $λ0, dλ0: $dλ0"

        # t -> 0.0 is just a dummy function, not to be used
        return Model(f!, t -> 0.0, z0, dz0, dvars, r0), get_Gp
    end
end

# Removes factor 1/T from gₓ, should be added back during Gp computation
function DEBUG_my_pendulum_adjoint_monly(u::Function, w::Function, θ::Vector{Float64}, T::Float64, x::Function, x2::Function, y::Function, dy::Function, xp0::Vector{Float64}, dx::Function, dx2::Function, N_trans::Int=0)
    # NOTE: A bit ugly to pass sol and sol2 as DAESolution, but dx as a function.
    # But good enough for now, just should be different in final version perhaps
    let m = θ[1], L = θ[2], g = θ[3], k = θ[4]
        np = size(xp0,2)
        @assert (np == 1) "my_pendulum_adjoint_monly is hard-coded to only handle one parameter m, make sure to pass correct xp0"
        nx = size(xp0,1)
        # x  = t -> sol(t)
        # x2 = t -> sol2(t)

        Fx = t -> [2dx(t)[6]        0               0   -1                  0           0   0
                    0           2*dx(t)[6]          0   0                   -1          0   0
                    -dx(t)[3]         0             0   2k*abs(x(t)[4])     0           0   0
                    0            -dx(t)[3]          0   0               2k*abs(x(t)[5]) 0   0
                    2x(t)[1]      2x(t)[2]          0   0                   0           0   0
                    x(t)[4]        x(t)[5]          0   x(t)[1]            x(t)[2]      0   0
                    x(t)[2]/(L^2)  -x(t)[1]/(L^2)   0   0                   0           0   1]
        # (namely the derivative of F with respect to the variable x_p)
        Fdx = t -> vcat([1   0   0          0   0   2x(t)[1]    0
                         0   1   0          0   0   2x(t)[2]    0
                         0   0   -x(t)[1]   m   0   0           0
                         0   0   -x(t)[2]   0   m   0           0], zeros(3,7))
        Fddx = t -> vcat([  0   0  0            0   0   2dx(t)[1]    0
                            0   0  0            0   0   2dx(t)[2]    0
                            0   0  -dx(t)[1]    0   0   0            0
                            0   0  -dx(t)[2]    0   0   0            0], zeros(3,7))
        Fp = t -> [ .0
                    .0
                    dx(t)[4]
                    dx(t)[5]+g
                    .0
                    .0
                    .0]
        gₓ  = t -> [0    0    0    0    0    0    2(x2(t)[7]-y(t))]
        gdₓ = t -> [0    0    0    0    0    0    2(dx2(t)[7]-dy(t))]

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

        # the residual function
        # NOTE: Could time-varying coefficients be the problem?? Sure would require more allocations?
        # TODO: If that is the case, we could store x(t) in a static array to avoid re-allocations?
        function f!(res, dz, z, θ, t)
            # TODO: Complete
            # Completely unreadabe but most efficient version (still not a huge improvement)
            res[1]  = dz[1] - 2*dx(t)[6]*z[1] + dx(t)[3]*z[3] - 2*x(t)[1]*z[5] - x(t)[4]*z[6] - (x(t)[2]*z[7])/(L^2)
            res[2]  = dz[2] - 2*dx(t)[6]*z[2] + dx(t)[3]*z[4] - 2*x(t)[2]*z[5] - x(t)[5]*z[6] + (x(t)[1]*z[7])/(L^2)
            res[3]  = -x(t)[1]*dz[3] - x(t)[2]*dz[4] - dx(t)[1]*z[3] - dx(t)[2]*z[4]
            res[4]  = m*dz[3] + z[1] - 2*k*abs(x(t)[4])*z[3] - x(t)[1]*z[6]
            res[5]  = m*dz[4] + z[2] - 2*k*abs(x(t)[5])*z[4] - x(t)[2]*z[6]
            res[6]  = 2*x(t)[1]*dz[1] + 2*x(t)[2]*dz[2] + 2*dx(t)[1]*z[1] + 2*dx(t)[2]*z[2]
            res[7]  = (2*(x2(t)[7] - y(t)))/T - z[7]
            # res[1]  = dz[1] + 2x(t)[1]*dz[6]    - 2dx(t)[6]*z[1] + z[4] + 2dx(t)[1]*z[6]
            # res[2]  = dz[2] + 2x(t)[2]*dz[6]    - 2dx(t)[6]*z[2] + z[5] + 2dx(t)[2]*z[6]
            # res[3]  = -x(t)[1]*dz[3] + m*dz[4]  + dx(t)[3]*z[1] - dx(t)[1]*z[3] - 2k*abs(x(t)[4])*z[4]
            # res[4]  = -x(t)[2]*dz[3] + m*dz[5]  + dx(t)[3]*z[2] - dx(t)[2]*z[3] - 2k*abs(x(t)[5])*z[5]
            # res[5]  =                           - 2x(t)[1]*z[1] - 2x(t)[2]*z[2]
            # res[6]  =                           - x(t)[4]*z[1] - x(t)[5]*z[2] - x(t)[1]*z[4] - x(t)[2]*z[5]
            # res[7]  =                           - x(t)[2]*z[1]/(L^2) + x(t)[1]*z[2]/(L^2) - z[7]                + (2/T)*(x2(t)[7]-y(t))
            res[8]  = dz[8] - z[3]*dx(t)[4] - z[4]*(dx(t)[5]+g)
            # res[8]  = dz[8] - z[3]*dx(T-t)[4] - z[4]*(dx(T-t)[5]+g)   # NOTE: SIMPLY WRONG; T-t????
            # res[8]  = dz[8] - z[3]*abs(x(t)[4])*x(t)[4] - z[4]*abs(x(t)[5])*x(t)[5]   # from pendelum k-only

            # # Super-readable but less efficient version
            # res[1:7]  = (dz[1:7]')*Fdx(T-t) + (z[1:7]')*(Fddx(T-t) - Fx(T-t)) + gₓ(T-t)
            # res[8] = dz[8] - (Fp(T-t)')*z[1:7]
            nothing
        end

        # z0 = λ0
        # dz0 = dλ0
        z0  = vcat(λT[:], zeros(np))
        # dz0 = vcat(dλ0, -((λT')*Fp(T))')
        dz0 = vcat(dλT[:], -(Fp(T)')*λT)
        # TODO: Can we delete these row below here perhaps?
        # z0  = vcat(λ0, 0.0)
        # dz0 = vcat(dλ0, λ0[1])   # For some reasing (λT')*Fp(T) returns a scalar instead of a 1-element matrix, unexpected but desired

        if N_trans > 0
            @warn "The returned function get_Gp() doesn't fully support N_trans > 0, as sensitivity of internal variables not known at any other time than t=0. A non-rigorous approximation is used instead."
        end
        # Function returning Gp given adjoint solution
        function get_Gp(adj_sol::DAESolution)
            # NOTE: Changes signs to match what I had in my manual calculations, seems correct now
            # Gp = adj_sol.u[end][end-np+1:end] + (((adj_sol.u[end][1:end-np]')*Fdx(0.0))*xp0)[:]
            @warn "adj_sols: $(adj_sol.u[end][nx+1:nx+np]), $(adj_sol.u[1][nx+1:nx+np])"
            Gp = (adj_sol.u[end-N_trans][nx+1:nx+np] .+ (((adj_sol.u[end][1:nx]')*Fdx(0.0))*xp0))/T
            # return 0.0
        end

        dvars = vcat(fill(true, 4), fill(false, 3), fill(true, np))
        # dvars = vcat(fill(true, 4), fill(false, 3))

        r0 = zeros(length(z0))
        f!(r0, dz0, z0, [], 0.0)
        # @info "r0 is: $r0 here, λ0: $λ0, dλ0: $dλ0"

        # t -> 0.0 is just a dummy function, not to be used
        return Model(f!, t -> 0.0, z0, dz0, dvars, r0), get_Gp
    end
end

function NEGATED_my_pendulum_adjoint_Lonly(u::Function, w::Function, θ::Vector{Float64}, T::Float64, x::Function, x2::Function, y::Function, dy::Function, xp0::Vector{Float64}, dx::Function, dx2::Function, N_trans::Int=0)
    # NOTE: A bit ugly to pass sol and sol2 as DAESolution, but dx as a function.
    # But good enough for now, just should be different in final version perhaps
    let m = θ[1], L = θ[2], g = θ[3], k = θ[4]
        np = size(xp0,2)
        @assert (np == 1) "my_pendulum_adjoint_Lonly is hard-coded to only handle one parameter L, make sure to pass correct xp0"
        nx = size(xp0,1)
        # x  = t -> sol(t)
        # x2 = t -> sol2(t)

        Fx = t -> [2dx(t)[6]        0               0   -1                  0           0   0
                    0           2*dx(t)[6]          0   0                   -1          0   0
                    -dx(t)[3]         0             0   2k*abs(x(t)[4])     0           0   0
                    0            -dx(t)[3]          0   0               2k*abs(x(t)[5]) 0   0
                    2x(t)[1]      2x(t)[2]          0   0                   0           0   0
                    x(t)[4]        x(t)[5]          0   x(t)[1]            x(t)[2]      0   0
                    x(t)[2]/(L^2)  -x(t)[1]/(L^2)   0   0                   0           0   1]
        # (namely the derivative of F with respect to the variable x_p)
        Fdx = t -> vcat([1   0   0          0   0   2x(t)[1]    0
                         0   1   0          0   0   2x(t)[2]    0
                         0   0   -x(t)[1]   m   0   0           0
                         0   0   -x(t)[2]   0   m   0           0], zeros(3,7))
        Fddx = t -> vcat([  0   0  0            0   0   2dx(t)[1]    0
                            0   0  0            0   0   2dx(t)[2]    0
                            0   0  -dx(t)[1]    0   0   0            0
                            0   0  -dx(t)[2]    0   0   0            0], zeros(3,7))
        Fp = t -> [ .0
                    .0
                    .0
                    .0
                    -2L   # How does L^2 turn into L? Is there no 2 missing or something?
                    .0
                    .0] # Was last element not dependent on L? Didn't we divide by e.g. L^2?
        gₓ  = t -> [0    0    0    0    0    0    2(x2(t)[7]-y(t))/T]
        gdₓ = t -> [0    0    0    0    0    0    2(dx2(t)[7]-dy(t))/T]

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

        # the residual function
        # NOTE: Could time-varying coefficients be the problem?? Sure would require more allocations?
        # TODO: If that is the case, we could store x(t) in a static array to avoid re-allocations?
        function f!(res, dz, z, θ, t)
            # Completely unreadabe but most efficient version (still not a huge improvement)
            res[1]  = dx(T-t)[3]*z[3] - dz[1] - 2*dx(T-t)[6]*z[1] - 2*z[5]*x(T-t)[1] - z[6]*x(T-t)[4] - (z[7]*x(T-t)[2])/(L^2)
            res[2]  = dx(T-t)[3]*z[4] - dz[2] - 2*dx(T-t)[6]*z[2] - 2*z[5]*x(T-t)[2] - z[6]*x(T-t)[5] + (z[7]*x(T-t)[1])/(L^2)
            res[3]  = dz[3]*x(T-t)[1] - dx(T-t)[2]*z[4] - dx(T-t)[1]*z[3] + dz[4]*x(T-t)[2]
            res[4]  = z[1] - dz[3]*m - z[6]*x(T-t)[1] - 2*k*z[3]*abs(x(T-t)[4])
            res[5]  = z[2] - dz[4]*m - z[6]*x(T-t)[2] - 2*k*z[4]*abs(x(T-t)[5])
            res[6]  = 2*dx(T-t)[1]*z[1] + 2*dx(T-t)[2]*z[2] - 2*dz[1]*x(T-t)[1] - 2*dz[2]*x(T-t)[2]
            res[7]  = (2*(x2(T-t)[7] - y(T-t)))/T - z[7]
            res[8]  = dz[8] - 2*L*z[5]

            # # Super-readable but less efficient version
            # res[1:7]  = (-dz[1:7]')*Fdx(T-t) + (z[1:7]')*(Fddx(T-t) - Fx(T-t)) + gₓ(T-t)
            # res[8] = dz[8] + (Fp(T-t)')*z[1:7]
            nothing
        end

        # Solving backwards in time =>
        # 1. Initial conditions will equal terminal conditions
        # 2. All derivatives have to be negated
        λ0  = λT[:]
        dλ0 = -dλT[:]
        # z0 = λ0
        # dz0 = dλ0
        z0  = vcat(λ0, zeros(np))
        # dz0 = vcat(dλ0, -((λT')*Fp(T))')
        dz0 = vcat(dλ0, -(Fp(T)')*λT)
        # TODO: Can we delete these row below here perhaps?
        # z0  = vcat(λ0, 0.0)
        # dz0 = vcat(dλ0, λ0[1])   # For some reasing (λT')*Fp(T) returns a scalar instead of a 1-element matrix, unexpected but desired

        if N_trans > 0
            @warn "The returned function get_Gp() doesn't fully support N_trans > 0, as sensitivity of internal variables not known at any other time than t=0. A non-rigorous approximation is used instead."
        end
        # Function returning Gp given adjoint solution
        function get_Gp(adj_sol::DAESolution)
            # NOTE: Changes signs to match what I had in my manual calculations, seems correct now
            # Gp = adj_sol.u[end][end-np+1:end] + (((adj_sol.u[end][1:end-np]')*Fdx(0.0))*xp0)[:]
            # Gp = adj_sol.u[end-N_trans][nx+1:nx+np] + (((adj_sol.u[end][1:nx]')*Fdx(0.0))*xp0)[:]
            Gp = adj_sol.u[end-N_trans][nx+1:nx+np] .+ (((adj_sol.u[end][1:nx]')*Fdx(0.0))*xp0)
            # return 0.0
        end

        dvars = vcat(fill(true, 4), fill(false, 3), fill(true, np))
        # dvars = vcat(fill(true, 4), fill(false, 3))

        r0 = zeros(length(z0))
        f!(r0, dz0, z0, [], 0.0)
        # @info "r0 is: $r0 here, λ0: $λ0, dλ0: $dλ0"

        # t -> 0.0 is just a dummy function, not to be used
        return Model(f!, t -> 0.0, z0, dz0, dvars, r0), get_Gp
    end
end

function my_pendulum_adjoint_Lonly(u::Function, w::Function, θ::Vector{Float64}, T::Float64, x::Function, x2::Function, y::Function, dy::Function, xp0::Vector{Float64}, dx::Function, dx2::Function, N_trans::Int=0)
    # NOTE: A bit ugly to pass sol and sol2 as DAESolution, but dx as a function.
    # But good enough for now, just should be different in final version perhaps
    let m = θ[1], L = θ[2], g = θ[3], k = θ[4]
        np = size(xp0,2)
        @assert (np == 1) "my_pendulum_adjoint_Lonly is hard-coded to only handle one parameter L, make sure to pass correct xp0"
        nx = size(xp0,1)
        # x  = t -> sol(t)
        # x2 = t -> sol2(t)

        Fx = t -> [2dx(t)[6]        0               0   -1                  0           0   0
                    0           2*dx(t)[6]          0   0                   -1          0   0
                    -dx(t)[3]         0             0   2k*abs(x(t)[4])     0           0   0
                    0            -dx(t)[3]          0   0               2k*abs(x(t)[5]) 0   0
                    2x(t)[1]      2x(t)[2]          0   0                   0           0   0
                    x(t)[4]        x(t)[5]          0   x(t)[1]            x(t)[2]      0   0
                    x(t)[2]/(L^2)  -x(t)[1]/(L^2)   0   0                   0           0   1]
        # (namely the derivative of F with respect to the variable x_p)
        Fdx = t -> vcat([1   0   0          0   0   2x(t)[1]    0
                         0   1   0          0   0   2x(t)[2]    0
                         0   0   -x(t)[1]   m   0   0           0
                         0   0   -x(t)[2]   0   m   0           0], zeros(3,7))
        Fddx = t -> vcat([  0   0  0            0   0   2dx(t)[1]    0
                            0   0  0            0   0   2dx(t)[2]    0
                            0   0  -dx(t)[1]    0   0   0            0
                            0   0  -dx(t)[2]    0   0   0            0], zeros(3,7))
        Fp = t -> [ .0
                    .0
                    .0
                    .0
                    -2L   # How does L^2 turn into L? Is there no 2 missing or something?
                    .0
                    .0] # Was last element not dependent on L? Didn't we divide by e.g. L^2?
        gₓ  = t -> [0    0    0    0    0    0    2(x2(t)[7]-y(t))/T]
        gdₓ = t -> [0    0    0    0    0    0    2(dx2(t)[7]-dy(t))/T]

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

        # the residual function
        # NOTE: Could time-varying coefficients be the problem?? Sure would require more allocations?
        # TODO: If that is the case, we could store x(t) in a static array to avoid re-allocations?
        function f!(res, dz, z, θ, t)
            # Completely unreadabe but most efficient version (still not a huge improvement)
            res[1]  = dz[1] - 2*dx(t)[6]*z[1] + dx(t)[3]*z[3] - 2*x(t)[1]*z[5] - x(t)[4]*z[6] - (x(t)[2]*z[7])/(L^2)
            res[2]  = dz[2] - 2*dx(t)[6]*z[2] + dx(t)[3]*z[4] - 2*x(t)[2]*z[5] - x(t)[5]*z[6] + (x(t)[1]*z[7])/(L^2)
            res[3]  = -x(t)[1]*dz[3] - x(t)[2]*dz[4] - dx(t)[1]*z[3] - dx(t)[2]*z[4]
            res[4]  = m*dz[3] + z[1] - 2*k*abs(x(t)[4])*z[3] - x(t)[1]*z[6]
            res[5]  = m*dz[4] + z[2] - 2*k*abs(x(t)[5])*z[4] - x(t)[2]*z[6]
            res[6]  = 2*x(t)[1]*dz[1] + 2*x(t)[2]*dz[2] + 2*dx(t)[1]*z[1] + 2*dx(t)[2]*z[2]
            res[7]  = (2*(x2(t)[7] - y(t)))/T - z[7]
            # res[1]  = dz[1] + 2x(t)[1]*dz[6]    - 2dx(t)[6]*z[1] + z[4] + 2dx(t)[1]*z[6]
            # res[2]  = dz[2] + 2x(t)[2]*dz[6]    - 2dx(t)[6]*z[2] + z[5] + 2dx(t)[2]*z[6]
            # res[3]  = -x(t)[1]*dz[3] + m*dz[4]  + dx(t)[3]*z[1] - dx(t)[1]*z[3] - 2k*abs(x(t)[4])*z[4]
            # res[4]  = -x(t)[2]*dz[3] + m*dz[5]  + dx(t)[3]*z[2] - dx(t)[2]*z[3] - 2k*abs(x(t)[5])*z[5]
            # res[5]  =                           - 2x(t)[1]*z[1] - 2x(t)[2]*z[2]
            # res[6]  =                           - x(t)[4]*z[1] - x(t)[5]*z[2] - x(t)[1]*z[4] - x(t)[2]*z[5]
            # res[7]  =                           - x(t)[2]*z[1]/(L^2) + x(t)[1]*z[2]/(L^2) - z[7]                + (2/T)*(x2(t)[7]-y(t))
            res[8]  = dz[8] + 2*L*z[5]

            # # Super-readable but less efficient version
            # res[1:7]  = (dz[1:7]')*Fdx(T-t) + (z[1:7]')*(Fddx(T-t) - Fx(T-t)) + gₓ(T-t)
            # res[8] = dz[8] - (Fp(T-t)')*z[1:7]
            nothing
        end

        # z0 = λ0
        # dz0 = dλ0
        z0  = vcat(λT[:], zeros(np))
        # dz0 = vcat(dλ0, -((λT')*Fp(T))')
        dz0 = vcat(dλT[:], -(Fp(T)')*λT)
        # TODO: Can we delete these row below here perhaps?
        # z0  = vcat(λ0, 0.0)
        # dz0 = vcat(dλ0, λ0[1])   # For some reasing (λT')*Fp(T) returns a scalar instead of a 1-element matrix, unexpected but desired

        if N_trans > 0
            @warn "The returned function get_Gp() doesn't fully support N_trans > 0, as sensitivity of internal variables not known at any other time than t=0. A non-rigorous approximation is used instead."
        end
        # Function returning Gp given adjoint solution
        function get_Gp(adj_sol::DAESolution)
            # NOTE: Changes signs to match what I had in my manual calculations, seems correct now
            # Gp = adj_sol.u[end][end-np+1:end] + (((adj_sol.u[end][1:end-np]')*Fdx(0.0))*xp0)[:]
            # Gp = adj_sol.u[end-N_trans][nx+1:nx+np] + (((adj_sol.u[end][1:nx]')*Fdx(0.0))*xp0)[:]
            Gp = adj_sol.u[end-N_trans][nx+1:nx+np] .+ (((adj_sol.u[end][1:nx]')*Fdx(0.0))*xp0)
            # return 0.0
        end

        dvars = vcat(fill(true, 4), fill(false, 3), fill(true, np))
        # dvars = vcat(fill(true, 4), fill(false, 3))

        r0 = zeros(length(z0))
        f!(r0, dz0, z0, [], 0.0)
        # @info "r0 is: $r0 here, λ0: $λ0, dλ0: $dλ0"

        # t -> 0.0 is just a dummy function, not to be used
        return Model(f!, t -> 0.0, z0, dz0, dvars, r0), get_Gp
    end
end

# NOTE Assumes free dynamical parameters are only k
function NEGATED_my_pendulum_adjoint_konly(u::Function, w::Function, θ::Vector{Float64}, T::Float64, x::Function, x2::Function, y::Function, dy::Function, xp0::Vector{Float64}, dx::Function, dx2::Function, N_trans::Int=0)
    # NOTE: A bit ugly to pass sol and sol2 as DAESolution, but dx as a function.
    # But good enough for now, just should be different in final version perhaps
    let m = θ[1], L = θ[2], g = θ[3], k = θ[4]
        np = size(xp0,2)
        @assert (np == 1) "my_pendulum_adjoint_konly is hard-coded to only handle one parameter k, make sure to pass correct xp0"
        nx = size(xp0,1)
        # x  = t -> sol(t)
        # x2 = t -> sol2(t)

        Fx = t -> [2dx(t)[6]        0               0   -1                  0           0   0
                    0           2*dx(t)[6]          0   0                   -1          0   0
                    -dx(t)[3]         0             0   2k*abs(x(t)[4])     0           0   0
                    0            -dx(t)[3]          0   0               2k*abs(x(t)[5]) 0   0
                    2x(t)[1]      2x(t)[2]          0   0                   0           0   0
                    x(t)[4]        x(t)[5]          0   x(t)[1]            x(t)[2]      0   0
                    x(t)[2]/(L^2)  -x(t)[1]/(L^2)   0   0                   0           0   1]
        # (namely the derivative of F with respect to the variable x_p)
        Fdx = t -> vcat([1   0   0          0   0   2x(t)[1]    0
                         0   1   0          0   0   2x(t)[2]    0
                         0   0   -x(t)[1]   m   0   0           0
                         0   0   -x(t)[2]   0   m   0           0], zeros(3,7))
        Fddx = t -> vcat([  0   0  0            0   0   2dx(t)[1]    0
                            0   0  0            0   0   2dx(t)[2]    0
                            0   0  -dx(t)[1]    0   0   0            0
                            0   0  -dx(t)[2]    0   0   0            0], zeros(3,7))
        Fp = t -> [.0; .0; abs(x(t)[4])*x(t)[4]; abs(x(t)[5])*x(t)[5]; .0; .0; .0]
        gₓ  = t -> [0    0    0    0    0    0    2(x2(t)[7]-y(t))/T]
        gdₓ = t -> [0    0    0    0    0    0    2(dx2(t)[7]-dy(t))/T]

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

        # the residual function
        # NOTE: Could time-varying coefficients be the problem?? Sure would require more allocations?
        # TODO: If that is the case, we could store x(t) in a static array to avoid re-allocations?
        function f!(res, dz, z, θ, t)
            # Completely unreadabe but most efficient version (still not a huge improvement)
            res[1]  = dx(T-t)[3]*z[3] - dz[1] - 2*dx(T-t)[6]*z[1] - 2*z[5]*x(T-t)[1] - z[6]*x(T-t)[4] - (z[7]*x(T-t)[2])/(L^2)
            res[2]  = dx(T-t)[3]*z[4] - dz[2] - 2*dx(T-t)[6]*z[2] - 2*z[5]*x(T-t)[2] - z[6]*x(T-t)[5] + (z[7]*x(T-t)[1])/(L^2)
            res[3]  = dz[3]*x(T-t)[1] - dx(T-t)[2]*z[4] - dx(T-t)[1]*z[3] + dz[4]*x(T-t)[2]
            res[4]  = z[1] - dz[3]*m - z[6]*x(T-t)[1] - 2*k*z[3]*abs(x(T-t)[4])
            res[5]  = z[2] - dz[4]*m - z[6]*x(T-t)[2] - 2*k*z[4]*abs(x(T-t)[5])
            res[6]  = 2*dx(T-t)[1]*z[1] + 2*dx(T-t)[2]*z[2] - 2*dz[1]*x(T-t)[1] - 2*dz[2]*x(T-t)[2]
            res[7]  = (2*(x2(T-t)[7] - y(T-t)))/T - z[7]
            res[8]  = dz[8] + z[3]*abs(x(T-t)[4])*x(T-t)[4] + z[4]*abs(x(T-t)[5])*x(T-t)[5]

            # # Medium readable but quite efficient version
            # λ  = z[1:nx]
            # dλ = dz[1:nx]
            # β  = z[nx+1:end]
            # dβ = dz[nx+1:end]
            # res[1]  = dx(T-t)[3]*λ[3] - dλ[1] - 2*dx(T-t)[6]*λ[1] - 2*λ[5]*x(T-t)[1] - λ[6]*x(T-t)[4] - (λ[7]*x(T-t)[2])/(L^2)
            # res[2]  = dx(T-t)[3]*λ[4] - dλ[2] - 2*dx(T-t)[6]*λ[2] - 2*λ[5]*x(T-t)[2] - λ[6]*x(T-t)[5] + (λ[7]*x(T-t)[1])/(L^2)
            # res[3]  = dλ[3]*x(T-t)[1] - dx(T-t)[2]*λ[4] - dx(T-t)[1]*λ[3] + dλ[4]*x(T-t)[2]
            # res[4]  = λ[1] - dλ[3]*m - λ[6]*x(T-t)[1] - 2*k*λ[3]*abs(x(T-t)[4])
            # res[5]  = λ[2] - dλ[4]*m - λ[6]*x(T-t)[2] - 2*k*λ[4]*abs(x(T-t)[5])
            # res[6]  = 2*dx(T-t)[1]*λ[1] + 2*dx(T-t)[2]*λ[2] - 2*dλ[1]*x(T-t)[1] - 2*dλ[2]*x(T-t)[2]
            # res[7]  = (2*(x2(T-t)[7] - y(T-t)))/T - λ[7]
            # res[8]  = dβ[1] + λ[3]*abs(x(T-t)[4])*x(T-t)[4] + λ[4]*abs(x(T-t)[5])*x(T-t)[5]

            # # Readable but slightly less efficient version
            # λ  = z[1:nx]
            # dλ = dz[1:nx]
            # β  = z[nx+1:end]
            # dβ = dz[nx+1:end]
            # res[1:7] = -(dλ')*Fdx(T-t) + (λ')*(Fddx(T-t)-Fx(T-t)) + gₓ(T-t)
            # res[8]  = dβ[1]+(λ')*Fp(T-t)

            # # Super-readable but less efficient version
            # res[1:7] = (-dz[1:7]')*Fdx(T-t) + (z[1:7]')*(Fddx(T-t) - Fx(T-t)) + gₓ(T-t)
            # res[8]   = dz[8] + (z[1:7]')*Fp(T-t)
            nothing
        end

        # Solving backwards in time =>
        # 1. Initial conditions will equal terminal conditions
        # 2. All derivatives have to be negated
        λ0  = λT[:]
        dλ0 = -dλT[:]
        # z0 = λ0
        # dz0 = dλ0
        z0  = vcat(λ0, zeros(np))
        dz0 = vcat(dλ0, -(λT')*Fp(T))   # For some reasing (λT')*Fp(T) returns a scalar instead of a 1-element matrix, unexpected but desired
        # TODO: Can we delete these row below here perhaps?
        # z0  = vcat(λ0, 0.0)
        # dz0 = vcat(dλ0, λ0[1])   # For some reasing (λT')*Fp(T) returns a scalar instead of a 1-element matrix, unexpected but desired

        if N_trans > 0
            @warn "The returned function get_Gp() doesn't fully support N_trans > 0, as sensitivity of internal variables not known at any other time than t=0. A non-rigorous approximation is used instead."
        end
        # Function returning Gp given adjoint solution
        function get_Gp(adj_sol::DAESolution)
            # NOTE: Changes signs to match what I had in my manual calculations, seems correct now
            # Gp = adj_sol.u[end][end-np+1:end] + (((adj_sol.u[end][1:end-np]')*Fdx(0.0))*xp0)[:]
            # Gp = adj_sol.u[end-N_trans][nx+1:nx+np] + (((adj_sol.u[end][1:nx]')*Fdx(0.0))*xp0)[:]
            Gp = adj_sol.u[end-N_trans][nx+1:nx+np] .+ (((adj_sol.u[end][1:nx]')*Fdx(0.0))*xp0)
            # return 0.0
        end

        dvars = vcat(fill(true, 4), fill(false, 3), fill(true, np))
        # dvars = vcat(fill(true, 4), fill(false, 3))

        r0 = zeros(length(z0))
        f!(r0, dz0, z0, [], 0.0)
        # @info "r0 is: $r0 here, λ0: $λ0, dλ0: $dλ0"

        # t -> 0.0 is just a dummy function, not to be used
        return Model(f!, t -> 0.0, z0, dz0, dvars, r0), get_Gp
    end
end

# NOTE Assumes free dynamical parameters are only k
function my_pendulum_adjoint_konly(u::Function, w::Function, θ::Vector{Float64}, T::Float64, x::Function, x2::Function, y::Function, dy::Function, xp0::Vector{Float64}, dx::Function, dx2::Function, N_trans::Int=0)
    # NOTE: A bit ugly to pass sol and sol2 as DAESolution, but dx as a function.
    # But good enough for now, just should be different in final version perhaps
    let m = θ[1], L = θ[2], g = θ[3], k = θ[4]
        np = size(xp0,2)
        @assert (np == 1) "my_pendulum_adjoint_konly is hard-coded to only handle one parameter k, make sure to pass correct xp0"
        nx = size(xp0,1)
        # x  = t -> sol(t)
        # x2 = t -> sol2(t)

        Fx = t -> [2dx(t)[6]        0               0   -1                  0           0   0
                    0           2*dx(t)[6]          0   0                   -1          0   0
                    -dx(t)[3]         0             0   2k*abs(x(t)[4])     0           0   0
                    0            -dx(t)[3]          0   0               2k*abs(x(t)[5]) 0   0
                    2x(t)[1]      2x(t)[2]          0   0                   0           0   0
                    x(t)[4]        x(t)[5]          0   x(t)[1]            x(t)[2]      0   0
                    x(t)[2]/(L^2)  -x(t)[1]/(L^2)   0   0                   0           0   1]
        # (namely the derivative of F with respect to the variable x_p)
        Fdx = t -> vcat([1   0   0          0   0   2x(t)[1]    0
                         0   1   0          0   0   2x(t)[2]    0
                         0   0   -x(t)[1]   m   0   0           0
                         0   0   -x(t)[2]   0   m   0           0], zeros(3,7))
        Fddx = t -> vcat([  0   0  0            0   0   2dx(t)[1]    0
                            0   0  0            0   0   2dx(t)[2]    0
                            0   0  -dx(t)[1]    0   0   0            0
                            0   0  -dx(t)[2]    0   0   0            0], zeros(3,7))
        Fp = t -> [.0; .0; abs(x(t)[4])*x(t)[4]; abs(x(t)[5])*x(t)[5]; .0; .0; .0]
        gₓ  = t -> [0    0    0    0    0    0    2(x2(t)[7]-y(t))/T]
        gdₓ = t -> [0    0    0    0    0    0    2(dx2(t)[7]-dy(t))/T]

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

        # the residual function
        # NOTE: Could time-varying coefficients be the problem?? Sure would require more allocations?
        # TODO: If that is the case, we could store x(t) in a static array to avoid re-allocations?
        function f!(res, dz, z, θ, t)
            # Completely unreadabe but most efficient version (still not a huge improvement)
            res[1]  = dz[1] - 2*dx(t)[6]*z[1] + dx(t)[3]*z[3] - 2*x(t)[1]*z[5] - x(t)[4]*z[6] - (x(t)[2]*z[7])/(L^2)
            res[2]  = dz[2] - 2*dx(t)[6]*z[2] + dx(t)[3]*z[4] - 2*x(t)[2]*z[5] - x(t)[5]*z[6] + (x(t)[1]*z[7])/(L^2)
            res[3]  = -x(t)[1]*dz[3] - x(t)[2]*dz[4] - dx(t)[1]*z[3] - dx(t)[2]*z[4]
            res[4]  = m*dz[3] + z[1] - 2*k*abs(x(t)[4])*z[3] - x(t)[1]*z[6]
            res[5]  = m*dz[4] + z[2] - 2*k*abs(x(t)[5])*z[4] - x(t)[2]*z[6]
            res[6]  = 2*x(t)[1]*dz[1] + 2*x(t)[2]*dz[2] + 2*dx(t)[1]*z[1] + 2*dx(t)[2]*z[2]
            res[7]  = (2*(x2(t)[7] - y(t)))/T - z[7]
            # res[1]  = dz[1] + 2x(t)[1]*dz[6]    - 2dx(t)[6]*z[1] + z[4] + 2dx(t)[1]*z[6]
            # res[2]  = dz[2] + 2x(t)[2]*dz[6]    - 2dx(t)[6]*z[2] + z[5] + 2dx(t)[2]*z[6]
            # res[3]  = -x(t)[1]*dz[3] + m*dz[4]  + dx(t)[3]*z[1] - dx(t)[1]*z[3] - 2k*abs(x(t)[4])*z[4]
            # res[4]  = -x(t)[2]*dz[3] + m*dz[5]  + dx(t)[3]*z[2] - dx(t)[2]*z[3] - 2k*abs(x(t)[5])*z[5]
            # res[5]  =                           - 2x(t)[1]*z[1] - 2x(t)[2]*z[2]
            # res[6]  =                           - x(t)[4]*z[1] - x(t)[5]*z[2] - x(t)[1]*z[4] - x(t)[2]*z[5]
            # res[7]  =                           - x(t)[2]*z[1]/(L^2) + x(t)[1]*z[2]/(L^2) - z[7]                + (2/T)*(x2(t)[7]-y(t))
            res[8]  = dz[8] - z[3]*abs(x(t)[4])*x(t)[4] - z[4]*abs(x(t)[5])*x(t)[5]

            # # Super-readable but less efficient version
            # res[1:7] = (dz[1:7]')*Fdx(t) + (z[1:7]')*(Fddx(t) - Fx(t)) + gₓ(t)
            # res[8]   = dz[8] - (z[1:7]')*Fp(t)
            nothing
        end

        z0  = vcat(λT[:], zeros(np))
        dz0 = vcat(dλT[:], (λT')*Fp(T))   # For some reasing (λT')*Fp(T) returns a scalar instead of a 1-element matrix, unexpected but desired
        # TODO: Can we delete these row below here perhaps?
        # z0  = vcat(λ0, 0.0)
        # dz0 = vcat(dλ0, λ0[1])   # For some reasing (λT')*Fp(T) returns a scalar instead of a 1-element matrix, unexpected but desired

        if N_trans > 0
            @warn "The returned function get_Gp() doesn't fully support N_trans > 0, as sensitivity of internal variables not known at any other time than t=0. A non-rigorous approximation is used instead."
        end
        # Function returning Gp given adjoint solution
        function get_Gp(adj_sol::DAESolution)
            # NOTE: Changes signs to match what I had in my manual calculations, seems correct now
            # Gp = adj_sol.u[end][end-np+1:end] + (((adj_sol.u[end][1:end-np]')*Fdx(0.0))*xp0)[:]
            # Gp = adj_sol.u[end-N_trans][nx+1:nx+np] + (((adj_sol.u[end][1:nx]')*Fdx(0.0))*xp0)[:]
            Gp = adj_sol.u[end-N_trans][nx+1:nx+np] .+ (((adj_sol.u[end][1:nx]')*Fdx(0.0))*xp0)
            # return 0.0
        end

        dvars = vcat(fill(true, 4), fill(false, 3), fill(true, np))
        # dvars = vcat(fill(true, 4), fill(false, 3))

        r0 = zeros(length(z0))
        f!(r0, dz0, z0, [], 0.0)
        # @info "r0 is: $r0 here, λ0: $λ0, dλ0: $dλ0"

        # t -> 0.0 is just a dummy function, not to be used
        return Model(f!, t -> 0.0, z0, dz0, dvars, r0), get_Gp
    end
end

# NOTE Assumes free dynamical parameters are only k
function crazystab_pendulum_adjoint_konly(u::Function, w::Function, θ::Vector{Float64}, T::Float64, x::Function, x2::Function, y::Function, dy::Function, xp0::Vector{Float64}, dx::Function, dx2::Function, N_trans::Int=0)
    # NOTE: A bit ugly to pass sol and sol2 as DAESolution, but dx as a function.
    # But good enough for now, just should be different in final version perhaps
    let m = θ[1], L = θ[2], g = θ[3], k = θ[4]
        np = size(xp0,2)
        @assert (np == 1) "crazystab_pendulum_adjoint_konly is hard-coded to only handle one parameter k, make sure to pass correct xp0"
        nx = size(xp0,1)
        # x  = t -> sol(t)
        # x2 = t -> sol2(t)

        # NOTE: Convention is used that derivatives wrt to θ stack along cols
        # while derivatives wrt to x stack along rows

        Fp = t -> [abs(x(t)[4])*x(t)[4]; abs(x(t)[5])*x(t)[5]; .0; .0; .0; .0; .0]  # NOTE: Actually not quite right, since it only should have 6 elements and not 7, but doesn't matter in this particular case

        ###################### INITIALIZING ADJOINT SYSTEM ####################
        # Indices 1-4 are differential (d), while 5-7 are algebraic (a)
        zT  = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, (2*x2(T)[7]-y(T))/T] # z = [λ1, λ2, λ3, λ4, λ5, d1, λ7]
        dzT = [0.0, 0.0, 0.0, x(T)[2]*zT[7]/(L^2), -x(T)[1]*λT[7]/(L^2), -x(t)[2]*zT[7]/(m*L^2), 2(dx2(T)[7]-dy(T))/T]

        # the residual function
        # NOTE: Could time-varying coefficients be the problem?? Sure would require more allocations?
        # TODO: If that is the case, we could store x(t) in a static array to avoid re-allocations?
        function f!(res, dz, z, θ, t)
            # NEW
            res[1]  = dz[4] + x(t)[3]*z[1] - 2*x(t)[1]*z[3] - x(t)[2]*z[7]/(L^2)
            res[2]  = dz[5] + x(t)[3]*z[2] - 2*x(t)[2]*z[3] + x(t)[1]*z[7]/(L^2)
            res[3]  = z[1]*x(t)[1] + z[2]*x(t)[2]
            res[4]  = z[4] + m*z[6] - 2*k*abs(x(t)[4])*z[1] - 2*k*abs(x(t)[5])*z[2]
            res[5]  = m*dz[2] + z[5]
            res[6]  = z[1]*x(t)[4] + z[2]*x(t)[5] + x(t)[2]*dz[2] + x(t)[1]*z[6]
            res[7]  = -z[7] + 2(x(t)[7]-y(t))/T
            res[8]  = dz[8] - z[1]*abs(x(t)[4])*x(t)[4] - z[2]*abs(x(t)[5])*x(t)[5]
            # # OLD
            # # Completely unreadabe but most efficient version (still not a huge improvement)
            # res[1]  = dz[1] - 2*dx(t)[6]*z[1] + dx(t)[3]*z[3] - 2*x(t)[1]*z[5] - x(t)[4]*z[6] - (x(t)[2]*z[7])/(L^2)
            # res[2]  = dz[2] - 2*dx(t)[6]*z[2] + dx(t)[3]*z[4] - 2*x(t)[2]*z[5] - x(t)[5]*z[6] + (x(t)[1]*z[7])/(L^2)
            # res[3]  = -x(t)[1]*dz[3] - x(t)[2]*dz[4] - dx(t)[1]*z[3] - dx(t)[2]*z[4]
            # res[4]  = m*dz[3] + z[1] - 2*k*abs(x(t)[4])*z[3] - x(t)[1]*z[6]
            # res[5]  = m*dz[4] + z[2] - 2*k*abs(x(t)[5])*z[4] - x(t)[2]*z[6]
            # res[6]  = 2*x(t)[1]*dz[1] + 2*x(t)[2]*dz[2] + 2*dx(t)[1]*z[1] + 2*dx(t)[2]*z[2]
            # res[7]  = (2*(x2(t)[7] - y(t)))/T - z[7]
            # # res[1]  = dz[1] + 2x(t)[1]*dz[6]    - 2dx(t)[6]*z[1] + z[4] + 2dx(t)[1]*z[6]
            # # res[2]  = dz[2] + 2x(t)[2]*dz[6]    - 2dx(t)[6]*z[2] + z[5] + 2dx(t)[2]*z[6]
            # # res[3]  = -x(t)[1]*dz[3] + m*dz[4]  + dx(t)[3]*z[1] - dx(t)[1]*z[3] - 2k*abs(x(t)[4])*z[4]
            # # res[4]  = -x(t)[2]*dz[3] + m*dz[5]  + dx(t)[3]*z[2] - dx(t)[2]*z[3] - 2k*abs(x(t)[5])*z[5]
            # # res[5]  =                           - 2x(t)[1]*z[1] - 2x(t)[2]*z[2]
            # # res[6]  =                           - x(t)[4]*z[1] - x(t)[5]*z[2] - x(t)[1]*z[4] - x(t)[2]*z[5]
            # # res[7]  =                           - x(t)[2]*z[1]/(L^2) + x(t)[1]*z[2]/(L^2) - z[7]                + (2/T)*(x2(t)[7]-y(t))
            # res[8]  = dz[8] - z[3]*abs(x(t)[4])*x(t)[4] - z[4]*abs(x(t)[5])*x(t)[5]

            # # # Super-readable but less efficient version
            # # res[1:7] = (dz[1:7]')*Fdx(t) + (z[1:7]')*(Fddx(t) - Fx(t)) + gₓ(t)
            # # res[8]   = dz[8] - (z[1:7]')*Fp(t)
            nothing
        end

        z0  = vcat(zT[:], zeros(np))
        dz0 = vcat(dzT[:], (zT')*Fp(T))   # For some reason (λT')*Fp(T) returns a scalar instead of a 1-element matrix, unexpected but desired
        # TODO: Can we delete these rows below here perhaps?
        # z0  = vcat(λ0, 0.0)
        # dz0 = vcat(dλ0, λ0[1])   # For some reason (λT')*Fp(T) returns a scalar instead of a 1-element matrix, unexpected but desired

        if N_trans > 0
            @warn "The returned function get_Gp() doesn't fully support N_trans > 0, as sensitivity of internal variables not known at any other time than t=0. A non-rigorous approximation is used instead."
        end
        # Function returning Gp given adjoint solution
        function get_Gp(adj_sol::DAESolution)
            # NOTE: Changes signs to match what I had in my manual calculations, seems correct now
            # Gp = adj_sol.u[end][end-np+1:end] + (((adj_sol.u[end][1:end-np]')*Fdx(0.0))*xp0)[:]
            # Gp = adj_sol.u[end-N_trans][nx+1:nx+np] + (((adj_sol.u[end][1:nx]')*Fdx(0.0))*xp0)[:]
            Gp = adj_sol.u[end-N_trans][nx+1:nx+np] .+ (((adj_sol.u[end][1:nx]')*Fdx(0.0))*xp0)
            # return 0.0
        end

        dvars = vcat([false, true, false, true, true, false, false], fill(true, np))
        # dvars = vcat(fill(true, 4), fill(false, 3))

        r0 = zeros(length(z0))
        f!(r0, dz0, z0, [], 0.0)
        # @info "r0 is: $r0 here, λ0: $λ0, dλ0: $dλ0"

        # t -> 0.0 is just a dummy function, not to be used
        return Model(f!, t -> 0.0, z0, dz0, dvars, r0), get_Gp
    end
end

function my_pendulum_adjoint_gonly(u::Function, w::Function, θ::Vector{Float64}, T::Float64, x::Function, x2::Function, y::Function, dy::Function, xp0::Vector{Float64}, dx::Function, dx2::Function, N_trans::Int=0)
    # NOTE: A bit ugly to pass sol and sol2 as DAESolution, but dx as a function.
    # But good enough for now, just should be different in final version perhaps
    let m = θ[1], L = θ[2], g = θ[3], k = θ[4]
        np = size(xp0,2)
        @assert (np == 1) "my_pendulum_adjoint_Lonly is hard-coded to only handle one parameter L, make sure to pass correct xp0"
        nx = size(xp0,1)
        # x  = t -> sol(t)
        # x2 = t -> sol2(t)

        Fx = t -> [2dx(t)[6]        0               0   -1                  0           0   0
                    0           2*dx(t)[6]          0   0                   -1          0   0
                    -dx(t)[3]         0             0   2k*abs(x(t)[4])     0           0   0
                    0            -dx(t)[3]          0   0               2k*abs(x(t)[5]) 0   0
                    2x(t)[1]      2x(t)[2]          0   0                   0           0   0
                    x(t)[4]        x(t)[5]          0   x(t)[1]            x(t)[2]      0   0
                    x(t)[2]/(L^2)  -x(t)[1]/(L^2)   0   0                   0           0   1]
        # (namely the derivative of F with respect to the variable x_p)
        Fdx = t -> vcat([1   0   0          0   0   2x(t)[1]    0
                         0   1   0          0   0   2x(t)[2]    0
                         0   0   -x(t)[1]   m   0   0           0
                         0   0   -x(t)[2]   0   m   0           0], zeros(3,7))
        Fddx = t -> vcat([  0   0  0            0   0   2dx(t)[1]    0
                            0   0  0            0   0   2dx(t)[2]    0
                            0   0  -dx(t)[1]    0   0   0            0
                            0   0  -dx(t)[2]    0   0   0            0], zeros(3,7))
        Fp = t -> [ .0
                    .0
                    .0
                     m
                    .0
                    .0
                    .0] # Was last element not dependent on L? Didn't we divide by e.g. L^2?
        gₓ  = t -> [0    0    0    0    0    0    2(x2(t)[7]-y(t))/T]
        gdₓ = t -> [0    0    0    0    0    0    2(dx2(t)[7]-dy(t))/T]

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

        # the residual function
        # NOTE: Could time-varying coefficients be the problem?? Sure would require more allocations?
        # TODO: If that is the case, we could store x(t) in a static array to avoid re-allocations?
        function f!(res, dz, z, θ, t)
            # Completely unreadabe but most efficient version (still not a huge improvement)
            res[1]  = dz[1] - 2*dx(t)[6]*z[1] + dx(t)[3]*z[3] - 2*x(t)[1]*z[5] - x(t)[4]*z[6] - (x(t)[2]*z[7])/(L^2)
            res[2]  = dz[2] - 2*dx(t)[6]*z[2] + dx(t)[3]*z[4] - 2*x(t)[2]*z[5] - x(t)[5]*z[6] + (x(t)[1]*z[7])/(L^2)
            res[3]  = -x(t)[1]*dz[3] - x(t)[2]*dz[4] - dx(t)[1]*z[3] - dx(t)[2]*z[4]
            res[4]  = m*dz[3] + z[1] - 2*k*abs(x(t)[4])*z[3] - x(t)[1]*z[6]
            res[5]  = m*dz[4] + z[2] - 2*k*abs(x(t)[5])*z[4] - x(t)[2]*z[6]
            res[6]  = 2*x(t)[1]*dz[1] + 2*x(t)[2]*dz[2] + 2*dx(t)[1]*z[1] + 2*dx(t)[2]*z[2]
            res[7]  = (2*(x2(t)[7] - y(t)))/T - z[7]
            # res[1]  = dz[1] + 2x(t)[1]*dz[6]    - 2dx(t)[6]*z[1] + z[4] + 2dx(t)[1]*z[6]
            # res[2]  = dz[2] + 2x(t)[2]*dz[6]    - 2dx(t)[6]*z[2] + z[5] + 2dx(t)[2]*z[6]
            # res[3]  = -x(t)[1]*dz[3] + m*dz[4]  + dx(t)[3]*z[1] - dx(t)[1]*z[3] - 2k*abs(x(t)[4])*z[4]
            # res[4]  = -x(t)[2]*dz[3] + m*dz[5]  + dx(t)[3]*z[2] - dx(t)[2]*z[3] - 2k*abs(x(t)[5])*z[5]
            # res[5]  =                           - 2x(t)[1]*z[1] - 2x(t)[2]*z[2]
            # res[6]  =                           - x(t)[4]*z[1] - x(t)[5]*z[2] - x(t)[1]*z[4] - x(t)[2]*z[5]
            # res[7]  =                           - x(t)[2]*z[1]/(L^2) + x(t)[1]*z[2]/(L^2) - z[7]                + (2/T)*(x2(t)[7]-y(t))
            res[8]  = dz[8] - m*z[4]

            # # Super-readable but less efficient version
            # res[1:7]  = (dz[1:7]')*Fdx(T-t) + (z[1:7]')*(Fddx(T-t) - Fx(T-t)) + gₓ(T-t)
            # res[8] = dz[8] - (Fp(T-t)')*z[1:7]
            nothing
        end

        # z0 = λ0
        # dz0 = dλ0
        z0  = vcat(λT[:], zeros(np))
        # dz0 = vcat(dλ0, -((λT')*Fp(T))')
        dz0 = vcat(dλT[:], -(Fp(T)')*λT)
        # TODO: Can we delete these row below here perhaps?
        # z0  = vcat(λ0, 0.0)
        # dz0 = vcat(dλ0, λ0[1])   # For some reasing (λT')*Fp(T) returns a scalar instead of a 1-element matrix, unexpected but desired

        if N_trans > 0
            @warn "The returned function get_Gp() doesn't fully support N_trans > 0, as sensitivity of internal variables not known at any other time than t=0. A non-rigorous approximation is used instead."
        end
        # Function returning Gp given adjoint solution
        function get_Gp(adj_sol::DAESolution)
            # NOTE: Changes signs to match what I had in my manual calculations, seems correct now
            # Gp = adj_sol.u[end][end-np+1:end] + (((adj_sol.u[end][1:end-np]')*Fdx(0.0))*xp0)[:]
            # Gp = adj_sol.u[end-N_trans][nx+1:nx+np] + (((adj_sol.u[end][1:nx]')*Fdx(0.0))*xp0)[:]
            Gp = adj_sol.u[end-N_trans][nx+1:nx+np] .+ (((adj_sol.u[end][1:nx]')*Fdx(0.0))*xp0)
            # return 0.0
        end

        dvars = vcat(fill(true, 4), fill(false, 3), fill(true, np))
        # dvars = vcat(fill(true, 4), fill(false, 3))

        r0 = zeros(length(z0))
        f!(r0, dz0, z0, [], 0.0)
        # @info "r0 is: $r0 here, λ0: $λ0, dλ0: $dλ0"

        # t -> 0.0 is just a dummy function, not to be used
        return Model(f!, t -> 0.0, z0, dz0, dvars, r0), get_Gp
    end
end

function my_pendulum_adjoint_deb(u::Function, w::Function, θ::Vector{Float64}, T::Float64, x::Function, x2::Function, y::Function, dy::Function, xp0::Vector{Float64}, dx::Function, dx2::Function, N_trans::Int=0)
    # NOTE: A bit ugly to pass sol and sol2 as DAESolution, but dx as a function.
    # But good enough for now, just should be different in final version perhaps
    let m = θ[1], L = θ[2], g = θ[3], k = θ[4]
        np = size(xp0,2)
        @assert (np == 1) "my_pendulum_adjoint_Lonly is hard-coded to only handle one parameter L, make sure to pass correct xp0"
        nx = size(xp0,1)
        # x  = t -> sol(t)
        # x2 = t -> sol2(t)

        # @warn "my_pendulum_adjoint_deb. m: $m, L: $L, g: $g, k: $k"

        my_ind = 2
        i = my_ind

        Fx = t -> [2dx(t)[6]        0               0   -1                  0           0   0
                    0           2*dx(t)[6]          0   0                   -1          0   0
                    -dx(t)[3]         0             0   2k*abs(x(t)[4])     0           0   0
                    0            -dx(t)[3]          0   0               2k*abs(x(t)[5]) 0   0
                    2x(t)[1]      2x(t)[2]          0   0                   0           0   0
                    x(t)[4]        x(t)[5]          0   x(t)[1]            x(t)[2]      0   0
                    x(t)[2]/(L^2)  -x(t)[1]/(L^2)   0   0                   0           0   1]
        # (namely the derivative of F with respect to the variable x_p)
        Fdx = t -> vcat([1   0   0          0   0   2x(t)[1]    0
                         0   1   0          0   0   2x(t)[2]    0
                         0   0   -x(t)[1]   m   0   0           0
                         0   0   -x(t)[2]   0   m   0           0], zeros(3,7))
        Fddx = t -> vcat([  0   0  0            0   0   2dx(t)[1]    0
                            0   0  0            0   0   2dx(t)[2]    0
                            0   0  -dx(t)[1]    0   0   0            0
                            0   0  -dx(t)[2]    0   0   0            0], zeros(3,7))
        Fp = t ->       [ Int(i==1)
                          Int(i==2)
                          Int(i==3)
                          Int(i==4)
                          Int(i==5)
                          Int(i==6)
                          Int(i==7)]
        gₓ  = t -> [0    0    0    0    0    0    2(x2(t)[7]-y(t))/T]
        gdₓ = t -> [0    0    0    0    0    0    2(dx2(t)[7]-dy(t))/T]

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

        # the residual function
        # NOTE: Could time-varying coefficients be the problem?? Sure would require more allocations?
        # TODO: If that is the case, we could store x(t) in a static array to avoid re-allocations?
        function f!(res, dz, z, θ, t)
            # Completely unreadabe but most efficient version (still not a huge improvement)
            res[1]  = dz[1] - 2*dx(t)[6]*z[1] + dx(t)[3]*z[3] - 2*x(t)[1]*z[5] - x(t)[4]*z[6] - (x(t)[2]*z[7])/(L^2)
            res[2]  = dz[2] - 2*dx(t)[6]*z[2] + dx(t)[3]*z[4] - 2*x(t)[2]*z[5] - x(t)[5]*z[6] + (x(t)[1]*z[7])/(L^2)
            res[3]  = -x(t)[1]*dz[3] - x(t)[2]*dz[4] - dx(t)[1]*z[3] - dx(t)[2]*z[4]
            res[4]  = m*dz[3] + z[1] - 2*k*abs(x(t)[4])*z[3] - x(t)[1]*z[6]
            res[5]  = m*dz[4] + z[2] - 2*k*abs(x(t)[5])*z[4] - x(t)[2]*z[6]
            res[6]  = 2*x(t)[1]*dz[1] + 2*x(t)[2]*dz[2] + 2*dx(t)[1]*z[1] + 2*dx(t)[2]*z[2]
            res[7]  = (2*(x2(t)[7] - y(t)))/T - z[7]
            res[8]  = dz[8] - z[i]

            # β(t) = ∫_t^T - λ'Fθ dt = ∫_T^t λ'Fθ => dβ = λ'Fθ
            # => dβ - λ'Fθ = 0.0

            # Super-readable but less efficient version
            # res[1:7]  = (dz[1:7]')*Fdx(t) + (z[1:7]')*(Fddx(t) - Fx(t)) + gₓ(t)
            # res[8] = dz[8] - (Fp(t)')*z[1:7]
            nothing
        end


        # z0 = λ0
        # dz0 = dλ0
        z0  = vcat(λT[:], zeros(np))
        # dz0 = vcat(dλ0, -((λT')*Fp(T))')
        dz0 = vcat(dλT[:], -(Fp(T)')*λT)
        # TODO: Can we delete these row below here perhaps?
        # z0  = vcat(λ0, 0.0)
        # dz0 = vcat(dλ0, λ0[1])   # For some reasing (λT')*Fp(T) returns a scalar instead of a 1-element matrix, unexpected but desired

        if N_trans > 0
            @warn "The returned function get_Gp() doesn't fully support N_trans > 0, as sensitivity of internal variables not known at any other time than t=0. A non-rigorous approximation is used instead."
        end
        # Function returning Gp given adjoint solution
        function get_Gp(adj_sol::DAESolution)
            # NOTE: Changes signs to match what I had in my manual calculations, seems correct now
            # Gp = adj_sol.u[end][end-np+1:end] + (((adj_sol.u[end][1:end-np]')*Fdx(0.0))*xp0)[:]
            # Gp = adj_sol.u[end-N_trans][nx+1:nx+np] + (((adj_sol.u[end][1:nx]')*Fdx(0.0))*xp0)[:]
            Gp = adj_sol.u[end-N_trans][nx+1:nx+np] .+ (((adj_sol.u[end][1:nx]')*Fdx(0.0))*xp0)
            # return 0.0
        end

        dvars = vcat(fill(true, 4), fill(false, 3), fill(true, np))
        # dvars = vcat(fill(true, 4), fill(false, 3))

        r0 = zeros(length(z0))
        f!(r0, dz0, z0, [], 0.0)
        # @info "r0 is: $r0 here, λ0: $λ0, dλ0: $dλ0"

        # t -> 0.0 is just a dummy function, not to be used
        return Model(f!, t -> 0.0, z0, dz0, dvars, r0), get_Gp
    end
end

function pendulum_adj_stepbystep_k(Φ::Float64, u::Function, w::Function, θ::Array{Float64, 1}, y::Function, λ::Function, dλ::Function, T::Float64)::Model
    @warn "pendulum_adj_stepbystep_k only adapted for k-parameter currently, no others"
    let m = θ[1], L = θ[2], g = θ[3], k = θ[4]

        # x  = t -> sol(t)
        Fx = (x, dx) -> [2dx[6]        0               0   -1                  0           0   0
                          0           2*dx[6]          0   0                   -1          0   0
                          -dx[3]         0             0   2k*abs(x[4])        0           0   0
                          0          -dx[3]            0   0               2k*abs(x[5])    0   0
                          2x[1]      2x[2]             0   0                   0           0   0
                          x[4]        x[5]             0   x[1]              x[2]          0   0
                          x[2]/(L^2)  -x[1]/(L^2)      0   0                   0           0   1]
        # (namely the derivative of F with respect to the variable x_p)
        Fdx = (x, dx) -> vcat([1   0   0          0   0   2x[1]    0
                               0   1   0          0   0   2x[2]    0
                               0   0   -x[1]   m   0   0           0
                               0   0   -x[2]   0   m   0           0], zeros(3,7))
        Fddx = (x, dx) -> vcat([  0   0  0            0   0   2dx[1]    0
                                  0   0  0            0   0   2dx[2]    0
                                  0   0  -dx[1]       0   0   0         0
                                  0   0  -dx[2]       0   0   0         0], zeros(3,7))
        Fp = (x, dx) -> [0
                        0
                        abs(x[4])*x[4]
                        abs(x[5])*x[5]
                        0
                        0
                        0]
       gₓ = (x, dx, t) -> [0.; 0.; 0.; 0.; 0.; 0.; 2(x[7]-y(t))/T]
       # λ = [1.; 1.; 1.; 1.; 1.; 1.; 1.]

       #TODO: DON'T PASS T. BUT ALSO DON'T CALL THIS gₓ!!!!

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
            # TODO: Analytical formula says it should be x[1]^2+x[2]^2 instead of
            # L^2 (though they should be equal), is it fine to substitute L^2 here?
            res[14] = x[14] - (x[1]*x[9] - x[8]*x[2])/(L^2)

            # res[15] = dx[15] - (y(t) - y(t;θ))^2 (point 1)
            res[15] = dx[15] - (y(t)-x[7])^2
            # res[16] = dx[16] - 2(y(t;θ)-y(t))*y_θ(t) (point 2)
            res[16] = dx[16] - 2(x[7]-y(t))*x[14]

            # # Comment out and just use explicilty
            # xθ  = x[8:14]
            # dxθ = dx[8:14]
            # res[17] = dx[17] - 2(y(t;θ)-y(t))*y_θ(t) - (λ')*(Fx(t)*xθ + Fdx(t)*dxθ + Fp(t)) (point 3)
            res[17] = dx[17] - 2(x[7]-y(t))*x[14] + (λ(t)')*(Fx(x, dx)*x[8:14] + Fdx(x, dx)*dx[8:14] + Fp(x, dx))

            # res[18] = dx[18] + (λ')*Fp(x, dx) + ( (λ')*(Fx(x,dx)-Fddx(x,dx)) - (dλ')*Fdx(x,dx) - gx(x,dx,t)' )*xθ (point 4)
            res[18] = dx[18] + (λ(t)')*Fp(x, dx) + ( (λ(t)')*(Fx(x,dx)-Fddx(x,dx)) - (dλ(t)')*Fdx(x,dx) - gₓ(x,dx,t)' )*x[8:14]

            # res[19] = dx[19] + (λ')*Fp(x, dx) (point 5)
            res[19] = dx[19] + (λ(t)')*Fp(x, dx)
            # res[20] = dx[20]  + ( (λ')*(Fx(x,dx)-Fddx(x,dx)) - (dλ')*Fdx(x,dx) - gx(x,dx,t)' )*xθ
            res[20] = dx[20] + ( (λ(t)')*(Fx(x,dx)-Fddx(x,dx)) - (dλ(t)')*Fdx(x,dx) - gₓ(x,dx,t)' )*x[8:14]

            # res[21] = dx[21] - (gx(x,dx,t)')*xθ + (λ(t)')*( Fx(x,dx)*xθ + Fdx(x,dx)*dxθ + Fp(x,dx)
            res[21] = dx[21] - (gₓ(x,dx,t)')*x[8:14] + (λ(t)')*( Fx(x,dx)*x[8:14] + Fdx(x,dx)*dx[8:14] + Fp(x,dx) )

            # res[22] = dx[22] - (λ(t)')*Fdx(t)*dxθ - ( (dλ(t)')*Fdx(t) + (λ(t)')*Fddx(t) )*xθ
            res[22] = dx[22] - (λ(t)')*Fdx(x,dx)*dx[8:14] - ( (dλ(t)')*Fdx(x,dx) + (λ(t)')*Fddx(x,dx) )*x[8:14]
            # res[22] = dx[22] - 2*sin(t)*cos(t)*(t^3) - (sin(t)^2)*3*t^2

            # res[23:29] = dx[23:29] - ( (dλ(t)')*Fdx(x,dx) + (λ(t)')*Fddx(x,dx) )[:]

            # Shared between 3.5 and 4
            res[23] = dx[23] - (gₓ(x,dx,t)')*x[8:14] + (λ(t)')*( Fx(x,dx)*x[8:14] + Fp(x,dx) )
            # Extra for 3.5 (for m and k only)
            res[24] = dx[24] + (λ(t)')*Fdx(x,dx)*dx[8:14]
            # Extra for 4 (for m and k only)
            res[25] = dx[25] - ( (dλ(t)')*Fdx(x,dx) + (λ(t)')*Fddx(x,dx) )*x[8:14]

            # NOTE: LoL, these are just negated versions of the above, clearly not very equal!
            # For replacement using partial integration
            # LHS
            res[26] = dx[26] - (λ(t)')*Fdx(x,dx)*dx[8:14]
            # RHS (without term, assumed zero, which is the case for m and correct λ)
            res[27] = dx[27] + ((dλ(t)')*Fdx(x,dx) + (λ(t)')*Fddx(x,dx) )*x[8:14]

            # (point 3.9) (rearranged point 4 for easier comparison with point 3.5)
            res[28] = dx[28] - (gₓ(x,dx,t)')*x[8:14] + (λ(t)')*(Fx(x,dx)*x[8:14] + Fp(x, dx)) - ( (dλ(t)')*Fdx(x,dx) + (λ(t)')*Fddx(x,dx) )*x[8:14]

            # REALLY SMAL STUFF; JUST DELETE
            res[29] = dx[29] - (dλ(t)')*Fdx(x,dx)*x[8:14]
            res[30] = dx[30] - (λ(t)')*Fddx(x,dx)*x[8:14]

            # res[30]    = dx[30] - 2*sin(t)*cos(t)*cos(t-0.3) + (sin(t)^2)*sin(t-0.3)

            # This is the equations for my_pendulum_adjoint_konly
            # res[1:7] = (-dz[1:7]')*Fdx(T-t) + (z[1:7]')*(Fddx(T-t) - Fx(T-t)) + gₓ(T-t)
            # <=> (z[1:7]')*(Fddx(T-t) - Fx(T-t)) - (dz[1:7]')*Fdx(T-t) + gₓ(T-t)
            # <=> (z[1:7]')*(Fx(T-t) - Fddx(T-t)) + (dz[1:7]')*Fdx(T-t) - gₓ(T-t)
            # THERE IS A SIGN INCONSISTENCY!!!!!!!!!!!!!!!!
            # Yes becuase that one is solved backwards you silly...

            # These equations sensitivity are written on the intuitive form. To
            # obtain form correct for this funciton, do the following replacements:
            # s[1] -> x[8],  s[2] -> x[9],  s[3] -> x[10]
            # s[4] -> x[11], s[5] -> x[12], s[6] -> x[13]
            # and the corresponding replacements for sp and dx

            # res[8]  = -s[4] + sp[1] + 2x[1]*sp[6] + 2s[1]*dx[6]
            # res[9]  = -s[5] + sp[2] + 2x[2]*sp[6] + 2s[2]*dx[6]
            # res[10]  = abs(x[4])*x[4] + 2k*s[4]*abs(x[4]) - x[1]*sp[3] + m*sp[4] - s[1]*dx[3]
            # res[11] = abs(x[5])*x[5] + 2k*s[5]*abs(x[5]) - x[2]*sp[3] + m*sp[5] - s[2]*dx[3]
            # res[12] = -2s[1]*x[1] - 2s[2]*x[2]
            # res[13] = s[4]*x[1] + s[5]*x[2] + s[1]*x[4] + s[2]*x[5]

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

        pend0  = vcat([x1_0, x2_0], zeros(4), [atan(x1_0 / -x2_0)])
        dpend0 = [0., 0., dx3_0, dx4_0, 0., 0., 0.]
        s0     = zeros(7)
        sp0    = zeros(7)
        G0    = 0.0
        dG0   = (y(0.0)-pend0[7])^2
        Gp0   = 0.0
        dGp0  = 2(pend0[7]-y(0.0))*0.0
        Gp02  = 0.0
        dGp02 = dGp0 - (λ(0.)')*(Fx(pend0, dpend0)*s0 + Fdx(pend0, dpend0)*sp0 + Fp(pend0, dpend0))
        Gp03  = 0.0
        dGp03 = -(λ(0.)')*Fp(pend0, dpend0) - ( (λ(0.)')*(Fx(pend0,dpend0)-Fddx(pend0,dpend0)) -(dλ(0.)')*Fdx(pend0,dpend0) - gₓ(pend0,dpend0,0.0)' )*s0
        Gp04  = 0.0
        dGp04 = -(λ(0.)')*Fp(pend0, dpend0)
        Gp04b  = 0.0
        dGp04b = - ( (λ(0.)')*(Fx(pend0,dpend0)-Fddx(pend0,dpend0)) -(dλ(0.)')*Fdx(pend0,dpend0) - gₓ(pend0,dpend0,0.0)' )*s0
        Gp035  = 0.0
        dGp035 = (gₓ(pend0,dpend0,0.0)')*s0 - (λ(0.0)')*( Fx(pend0,dpend0)*s0 + Fdx(pend0,dpend0)*sp0 + Fp(pend0,dpend0) )
        partial    = 0.0
        dpartial   = (λ(0.0)')*Fdx(pend0,dpend0)*sp0 + ( (dλ(0.0)')*Fdx(pend0,dpend0) + (λ(0.0)')*Fddx(pend0,dpend0) )*s0

        # NOTE: ONLY MAKE SENSE FOR k AND m SINCE ONLY THEN APPROPRIATE QUANTITIES ARE ZERO
        common   = 0.0
        dcommon  = (gₓ(pend0,dpend0,0.0)')*s0 - (λ(0.)')*( Fx(pend0,dpend0)*s0 + Fp(pend0,dpend0) )
        extra35  = 0.0
        dextra35 = -(λ(0.)')*Fdx(pend0,dpend0)*sp0
        extra4   = 0.0
        dextra4  = ( (dλ(0.0)')*Fdx(pend0,dpend0) + (λ(0.0)')*Fddx(pend0,dpend0) )*s0

        partrepL  = 0.0
        dpartrepL = (λ(0.)')*Fdx(pend0,dpend0)*sp0
        partrepR  = 0.0
        dpartrepR = -((dλ(0.)')*Fdx(pend0,dpend0) + (λ(0.)')*Fddx(pend0,dpend0) )*s0

        oscdeb = 0.0
        doscdeb = (gₓ(pend0,dpend0,0.0)')*s0 - (λ(0.0)')*(Fx(pend0,dpend0)*s0 + Fp(pend0, dpend0)) + (λ(0.0)')*Fddx(pend0,dpend0)*s0 + (dλ(0.0)')*Fdx(pend0,dpend0)*s0
        res29  = 0.0
        dres29 = (dλ(0.0)')*Fdx(pend0,dpend0)*s0
        res30  = 0.0
        dres30 = (λ(0.0)')*Fddx(pend0,dpend0)*s0

        # b1  = 0.0
        # db1 = 0.0

        # tue_deb    = (λ(0.0)')*Fdx(pend0, dpend0)
        # dtue_deb   = (dλ(0.0)')*Fdx(pend0, dpend0) + (λ(0.0)')*Fddx(pend0, dpend0)

        x0   = vcat(pend0, s0, G0, Gp0, Gp02, Gp03, Gp04, Gp04b, Gp035, partial, common, extra35, extra4, partrepL, partrepR, oscdeb, res29, res30)#, b1)
        dx0  = vcat(dpend0, sp0, dG0, dGp0, dGp02, dGp03, dGp04, dGp04b, dGp035, dpartial, dcommon, dextra35, dextra4, dpartrepL, dpartrepR, doscdeb, dres29, dres30)#, db1)

        dvars = vcat(fill(true, 6), [false], fill(true, 6), [false, true, true, true, true, true, true, true, true], fill(true, 8))

        r0 = zeros(length(x0))
        f!(r0, dx0, x0, [], 0.0)

        # t -> 0.0 is just a dummy function, not to be used
        Model(f!, t -> 0.0, x0, dx0, dvars, r0)
    end
end

function pendulum_adj_stepbystep_L(Φ::Float64, u::Function, w::Function, θ::Array{Float64, 1}, y::Function, λ::Function, dλ::Function, T::Float64)::Model
    @warn "pendulum_adj_stepbystep_L only adapted for L-parameter currently, no others"
    let m = θ[1], L = θ[2], g = θ[3], k = θ[4]

        # x  = t -> sol(t)
        Fx = (x, dx) -> [2dx[6]        0               0   -1                  0           0   0
                          0           2*dx[6]          0   0                   -1          0   0
                         -dx[3]        0               0   2k*abs(x[4])        0           0   0
                          0          -dx[3]            0   0               2k*abs(x[5])    0   0
                          2x[1]       2x[2]            0   0                   0           0   0
                          x[4]        x[5]             0   x[1]               x[2]         0   0
                          x[2]/(L^2)  -x[1]/(L^2)      0   0                   0           0   1]
        # (namely the derivative of F with respect to the variable x_p)
        Fdx = (x, dx) -> vcat([1   0   0          0   0   2x[1]    0
                               0   1   0          0   0   2x[2]    0
                               0   0   -x[1]   m   0   0           0
                               0   0   -x[2]   0   m   0           0], zeros(3,7))
        Fddx = (x, dx) -> vcat([  0   0  0            0   0   2dx[1]    0
                                  0   0  0            0   0   2dx[2]    0
                                  0   0  -dx[1]    0   0   0            0
                                  0   0  -dx[2]    0   0   0            0], zeros(3,7))
        Fp = (x, dx) -> [0
                         0
                         0
                         0
                        -2L
                         0
                         0]
       gₓ = (x, dx, t) -> [0.; 0.; 0.; 0.; 0.; 0.; 2(x[7]-y(t))/T]
       # λ = [1.; 1.; 1.; 1.; 1.; 1.; 1.]

       #TODO: DON'T PASS T. BUT ALSO DON'T CALL THIS gₓ!!!! EDIT: Why...?

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

            # Sensitivity equations for L
            res[8]  = dx[8] - x[11] + 2x[8]dx[6] + 2dx[13]x[1]
            res[9]  = dx[9] - x[12] + 2x[9]dx[6] + 2dx[13]x[2]
            res[10] = 2k*x[11]*abs(x[4]) + m*dx[11] - x[8]dx[3] - dx[10]x[1]
            res[11] = 2k*x[12]*abs(x[5]) + m*dx[12] - x[9]dx[3] - dx[10]x[2]
            res[12] = 2x[8]x[1] + 2x[9]x[2] - 2L
            res[13] = x[8]x[4] + x[11]x[1] + x[9]x[5] + x[12]x[2]
            # Sensitivity of angle of pendulum to L
            # TODO: Analytical formula says it should be x[1]^2+x[2]^2 instead of
            # L^2 (though they should be equal), is it fine to substitute L^2 here?
            res[14] = x[14] - (x[1]*x[9] - x[8]*x[2])/(L^2)

            # res[15] = dx[15] - (y(t) - y(t;θ))^2 (point 1)
            res[15] = dx[15] - (y(t)-x[7])^2
            # res[16] = dx[16] - 2(y(t;θ)-y(t))*y_θ(t) (point 2)
            res[16] = dx[16] - 2(x[7]-y(t))*x[14]

            # # Comment out and just use explicilty
            # xθ  = x[8:14]
            # dxθ = dx[8:14]
            # res[17] = dx[17] - 2(y(t;θ)-y(t))*y_θ(t) - (λ')*(Fx(t)*xθ + Fdx(t)*dxθ + Fp(t)) (point 3)
            res[17] = dx[17] - 2(x[7]-y(t))*x[14] + (λ(t)')*(Fx(x, dx)*x[8:14] + Fdx(x, dx)*dx[8:14] + Fp(x, dx))

            # res[18] = dx[18] + (λ')*Fp(x, dx) + ( (λ')*(Fx(x,dx)-Fddx(x,dx)) - (dλ')*Fdx(x,dx) - gx(x,dx,t)' )*xθ (point 4)
            res[18] = dx[18] + (λ(t)')*Fp(x, dx) + ( (λ(t)')*(Fx(x,dx)-Fddx(x,dx)) - (dλ(t)')*Fdx(x,dx) - gₓ(x,dx,t)' )*x[8:14]

            # res[19] = dx[19] + (λ')*Fp(x, dx) (point 5)
            res[19] = dx[19] + (λ(t)')*Fp(x, dx)
            # res[20] = dx[20]  + ( (λ')*(Fx(x,dx)-Fddx(x,dx)) - (dλ')*Fdx(x,dx) - gx(x,dx,t)' )*xθ
            res[20] = dx[20] + ( (λ(t)')*(Fx(x,dx)-Fddx(x,dx)) - (dλ(t)')*Fdx(x,dx) - gₓ(x,dx,t)' )*x[8:14]

            # res[21] = dx[21] - (gx(x,dx,t)')*xθ + (λ(t)')*( Fx(x,dx)*xθ + Fdx(x,dx)*dxθ + Fp(x,dx
            res[21] = dx[21] - (gₓ(x,dx,t)')*x[8:14] + (λ(t)')*( Fx(x,dx)*x[8:14] + Fdx(x,dx)*dx[8:14] + Fp(x,dx) )

            # res[22] = dx[22] - (λ(t)')*Fdx(t)*dxθ - ( (dλ(t)')*Fdx(t) + (λ(t)')*Fddx(t) )*xθ
            res[22] = dx[22] - (λ(t)')*Fdx(x,dx)*dx[8:14] - ( (dλ(t)')*Fdx(x,dx) + (λ(t)')*Fddx(x,dx) )*x[8:14]

            # Shared between 3.5 and 4
            res[23] = dx[23] - (gₓ(x,dx,t)')*x[8:14] + (λ(t)')*( Fx(x,dx)*x[8:14] + Fp(x,dx) )
            # Extra for 3.5 (for m and k only)
            res[24] = dx[24] + (λ(t)')*Fdx(x,dx)*dx[8:14]
            # Extra for 4 (for m and k only)
            res[25] = dx[25] - ( (dλ(t)')*Fdx(x,dx) + (λ(t)')*Fddx(x,dx) )*x[8:14]

            # NOTE: LoL, these are just negated versions of the above, clearly not very equal!
            # For replacement using partial integration
            # LHS
            res[26] = dx[26] - (λ(t)')*Fdx(x,dx)*dx[8:14]
            # RHS (without term, assumed zero, which is the case for m and correct λ)
            res[27] = dx[27] + ((dλ(t)')*Fdx(x,dx) + (λ(t)')*Fddx(x,dx) )*x[8:14]

            # (point 3.9) (rearranged point 4 for easier comparison with point 3.5)
            res[28] = dx[28] - (gₓ(x,dx,t)')*x[8:14] + (λ(t)')*(Fx(x,dx)*x[8:14] + Fp(x, dx)) - ( (dλ(t)')*Fdx(x,dx) + (λ(t)')*Fddx(x,dx) )*x[8:14]

            # REALLY SMAL STUFF; JUST DELETE
            res[29] = dx[29] - (dλ(t)')*Fdx(x,dx)*x[8:14]
            res[30] = dx[30] - (λ(t)')*Fddx(x,dx)*x[8:14]

            # This is the equations for my_pendulum_adjoint_konly
            # res[1:7] = (-dz[1:7]')*Fdx(T-t) + (z[1:7]')*(Fddx(T-t) - Fx(T-t)) + gₓ(T-t)
            # <=> (z[1:7]')*(Fddx(T-t) - Fx(T-t)) - (dz[1:7]')*Fdx(T-t) + gₓ(T-t)
            # <=> (z[1:7]')*(Fx(T-t) - Fddx(T-t)) + (dz[1:7]')*Fdx(T-t) - gₓ(T-t)
            # THERE IS A SIGN INCONSISTENCY!!!!!!!!!!!!!!!!
            # Yes becuase that one is solved backwards you silly...

            # These equations sensitivity are written on the intuitive form. To
            # obtain form correct for this funciton, do the following replacements:
            # s[1] -> x[8],  s[2] -> x[9],  s[3] -> x[10]
            # s[4] -> x[11], s[5] -> x[12], s[6] -> x[13]
            # and the corresponding replacements for sp and dx

            # res[8]  = -s[4] + sp[1] + 2x[1]*sp[6] + 2s[1]*dx[6]
            # res[9]  = -s[5] + sp[2] + 2x[2]*sp[6] + 2s[2]*dx[6]
            # res[10]  = abs(x[4])*x[4] + 2k*s[4]*abs(x[4]) - x[1]*sp[3] + m*sp[4] - s[1]*dx[3]
            # res[11] = abs(x[5])*x[5] + 2k*s[5]*abs(x[5]) - x[2]*sp[3] + m*sp[5] - s[2]*dx[3]
            # res[12] = -2s[1]*x[1] - 2s[2]*x[2]
            # res[13] = s[4]*x[1] + s[5]*x[2] + s[1]*x[4] + s[2]*x[5]

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

        pend0  = vcat([x1_0, x2_0], zeros(4), [atan(x1_0 / -x2_0)])
        dpend0 = [0., 0., dx3_0, dx4_0, 0., 0., 0.]
        s0  = vcat([x1_0/L, x2_0/L], zeros(5))
        ds0 = vcat([0.,0., -dx3_0/L], zeros(4))
        G0    = 0.0
        dG0   = (y(0.0)-pend0[7])^2
        Gp0   = 0.0
        dGp0  = 2(pend0[7]-y(0.0))*0.0
        Gp02  = 0.0
        dGp02 = dGp0 - (λ(0.)')*(Fx(pend0, dpend0)*s0 + Fdx(pend0, dpend0)*ds0 + Fp(pend0, dpend0))
        Gp03  = 0.0
        dGp03 = -(λ(0.)')*Fp(pend0, dpend0) - ( (λ(0.)')*(Fx(pend0,dpend0)-Fddx(pend0,dpend0)) -(dλ(0.)')*Fdx(pend0,dpend0) - gₓ(pend0,dpend0,0.0)' )*s0
        Gp04  = 0.0
        dGp04 = -(λ(0.)')*Fp(pend0, dpend0)
        Gp04b  = 0.0
        dGp04b = - ( (λ(0.)')*(Fx(pend0,dpend0)-Fddx(pend0,dpend0)) -(dλ(0.)')*Fdx(pend0,dpend0) - gₓ(pend0,dpend0,0.0)' )*s0
        Gp035  = 0.0
        dGp035 = (gₓ(pend0,dpend0,0.0)')*s0 - (λ(0.0)')*( Fx(pend0,dpend0)*s0 + Fdx(pend0,dpend0)*ds0 + Fp(pend0,dpend0) )
        partial    = 0.0
        dpartial   = (λ(0.0)')*Fdx(pend0,dpend0)*ds0 + ( (dλ(0.0)')*Fdx(pend0,dpend0) + (λ(0.0)')*Fddx(pend0,dpend0) )*s0

        # tue_deb    = (λ(0.0)')*Fdx(pend0, dpend0)
        # dtue_deb   = (dλ(0.0)')*Fdx(pend0, dpend0) + (λ(0.0)')*Fddx(pend0, dpend0)
        # NOTE: ONLY MAKE SENSE FOR k AND m SINCE ONLY THEN APPROPRIATE QUANTITIES ARE ZERO
        common   = 0.0
        dcommon  = (gₓ(pend0,dpend0,0.0)')*s0 - (λ(0.)')*( Fx(pend0,dpend0)*s0 + Fp(pend0,dpend0) )
        extra35  = 0.0
        dextra35 = -(λ(0.)')*Fdx(pend0,dpend0)*ds0
        extra4   = 0.0
        dextra4  = ( (dλ(0.0)')*Fdx(pend0,dpend0) + (λ(0.0)')*Fddx(pend0,dpend0) )*s0

        partrepL  = 0.0
        dpartrepL = (λ(0.)')*Fdx(pend0,dpend0)*ds0
        partrepR  = 0.0
        dpartrepR = -((dλ(0.)')*Fdx(pend0,dpend0) + (λ(0.)')*Fddx(pend0,dpend0) )*s0
        oscdeb = 0.0
        doscdeb = (gₓ(pend0,dpend0,0.0)')*s0 - (λ(0.0)')*(Fx(pend0,dpend0)*s0 + Fp(pend0, dpend0)) + (λ(0.0)')*Fddx(pend0,dpend0)*s0 + (dλ(0.0)')*Fdx(pend0,dpend0)*s0
        res29  = 0.0
        dres29 = (dλ(0.0)')*Fdx(pend0,dpend0)*s0
        res30  = 0.0
        dres30 = (λ(0.0)')*Fddx(pend0,dpend0)*s0

        x0   = vcat(pend0, s0, G0, Gp0, Gp02, Gp03, Gp04, Gp04b, Gp035, partial, common, extra35, extra4, partrepL, partrepR, oscdeb, res29, res30)
        dx0  = vcat(dpend0, ds0, dG0, dGp0, dGp02, dGp03, dGp04, dGp04b, dGp035, dpartial, dcommon, dextra35, dextra4, dpartrepL, dpartrepR, doscdeb, dres29, dres30)

        dvars = vcat(fill(true, 6), [false], fill(true, 6), [false, true, true, true, true, true, true, true, true], fill(true, 8))
        # dvars = vcat(fill(true, 6), [false], fill(true, 6), [false, true, true, true, true, true, true, true, true], fill(true, 7))

        r0 = zeros(length(x0))
        f!(r0, dx0, x0, [], 0.0)
        # @info "r0 is $r0"

        # t -> 0.0 is just a dummy function, not to be used
        Model(f!, t -> 0.0, x0, dx0, dvars, r0)
    end
end

function pendulum_adj_stepbystep_m(Φ::Float64, u::Function, w::Function, θ::Array{Float64, 1}, y::Function, λ::Function, dλ::Function, T::Float64)::Model
    @warn "pendulum_adj_stepbystep_m only adapted for m-parameter currently, no others"
    let m = θ[1], L = θ[2], g = θ[3], k = θ[4]

        # x  = t -> sol(t)
        Fx = (x, dx) -> [2dx[6]        0               0   -1                  0           0   0
                          0           2*dx[6]          0   0                   -1          0   0
                         -dx[3]        0               0   2k*abs(x[4])        0           0   0
                          0          -dx[3]            0   0               2k*abs(x[5])    0   0
                          2x[1]       2x[2]            0   0                   0           0   0
                          x[4]        x[5]             0   x[1]               x[2]         0   0
                          x[2]/(L^2)  -x[1]/(L^2)      0   0                   0           0   1]
        # (namely the derivative of F with respect to the variable x_p)
        Fdx = (x, dx) -> vcat([1   0   0          0   0   2x[1]    0
                               0   1   0          0   0   2x[2]    0
                               0   0   -x[1]   m   0   0           0
                               0   0   -x[2]   0   m   0           0], zeros(3,7))
        Fddx = (x, dx) -> vcat([  0   0  0            0   0   2dx[1]    0
                                  0   0  0            0   0   2dx[2]    0
                                  0   0  -dx[1]    0   0   0            0
                                  0   0  -dx[2]    0   0   0            0], zeros(3,7))
        Fp = (x, dx) -> [ .0
                  .0
                  dx[4]
                  dx[5]+g
                  .0
                  .0
                  .0]
       gₓ = (x, dx, t) -> [0.; 0.; 0.; 0.; 0.; 0.; 2(x[7]-y(t))/T]
       # λ = [1.; 1.; 1.; 1.; 1.; 1.; 1.]

       #TODO: DON'T PASS T. BUT ALSO DON'T CALL THIS gₓ!!!! EDIT: Why...?

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

            # res[15] = dx[15] - (y(t) - y(t;θ))^2 (point 1)
            res[15] = dx[15] - (y(t)-x[7])^2
            # res[16] = dx[16] - 2(y(t;θ)-y(t))*y_θ(t) (point 2)
            res[16] = dx[16] - 2(x[7]-y(t))*x[14]

            # # Comment out and just use explicilty
            # xθ  = x[8:14]
            # dxθ = dx[8:14]
            # res[17] = dx[17] - 2(y(t;θ)-y(t))*y_θ(t) - (λ')*(Fx(t)*xθ + Fdx(t)*dxθ + Fp(t)) (point 3)
            res[17] = dx[17] - 2(x[7]-y(t))*x[14] + (λ(t)')*(Fx(x, dx)*x[8:14] + Fdx(x, dx)*dx[8:14] + Fp(x, dx))

            # res[18] = dx[18] + (λ')*Fp(x, dx) + ( (λ')*(Fx(x,dx)-Fddx(x,dx)) - (dλ')*Fdx(x,dx) - gx(x,dx,t)' )*xθ (point 4)
            res[18] = dx[18] + (λ(t)')*Fp(x, dx) + ( (λ(t)')*(Fx(x,dx)-Fddx(x,dx)) - (dλ(t)')*Fdx(x,dx) - gₓ(x,dx,t)' )*x[8:14]

            # res[19] = dx[19] + (λ')*Fp(x, dx) (point 5)
            res[19] = dx[19] + (λ(t)')*Fp(x, dx)# - 12.1e-7
            # res[20] = dx[20]  + ( (λ')*(Fx(x,dx)-Fddx(x,dx)) - (dλ')*Fdx(x,dx) - gx(x,dx,t)' )*xθ
            res[20] = dx[20] + ( (λ(t)')*(Fx(x,dx)-Fddx(x,dx)) - (dλ(t)')*Fdx(x,dx) - gₓ(x,dx,t)' )*x[8:14] # (bonus point (remainder?))

            # # res[21] = dx[21] - (gx(x,dx,t)')*xθ + (λ(t)')*( Fx(x,dx)*xθ + Fdx(x,dx)*dxθ + Fp(x,dx
            # res[21] = dx[21] - (gₓ(x,dx,t)')*x[8:14] + (λ(t)')*( Fx(x,dx)*x[8:14] + Fdx(x,dx)*dx[8:14] + Fp(x,dx) ) # (point 3.5)

            # res[22] = dx[22] - (λ(t)')*Fdx(t)*dxθ - ( (dλ(t)')*Fdx(t) + (λ(t)')*Fddx(t) )*xθ
            res[22] = dx[22] - (λ(t)')*Fdx(x,dx)*dx[8:14] - ( (dλ(t)')*Fdx(x,dx) + (λ(t)')*Fddx(x,dx) )*x[8:14]     # (point 6)

            # res[23:29] = dx[23:29] - ( (dλ(t)')*Fdx(x,dx) + (λ(t)')*Fddx(x,dx) )[:]

            # Shared between 3.5 and 4
            res[23] = dx[23] - (gₓ(x,dx,t)')*x[8:14] + (λ(t)')*( Fx(x,dx)*x[8:14] + Fp(x,dx) )
            # Extra for 3.5 (for m and k only)
            res[24] = dx[24] + (λ(t)')*Fdx(x,dx)*dx[8:14]
            # Extra for 4 (for m and k only)
            res[25] = dx[25] - ( (dλ(t)')*Fdx(x,dx) + (λ(t)')*Fddx(x,dx) )*x[8:14]

            # For replacement using partial integration
            # LHS
            res[26] = dx[26] - (λ(t)')*Fdx(x,dx)*dx[8:14]
            # RHS (without term, assumed zero, which is the case for m and correct λ)
            res[27] = dx[27] + ((dλ(t)')*Fdx(x,dx) + (λ(t)')*Fddx(x,dx) )*x[8:14]

            # res[21] = dx[21] - (gx(x,dx,t)')*xθ + (λ(t)')*(Fx(x,dx)*xθ + Fdx(x,dx)*dxθ + Fp(x,dx
            res[21] = dx[21] - (gₓ(x,dx,t)')*x[8:14] + (λ(t)')*(Fx(x,dx)*x[8:14] + Fp(x,dx) + Fdx(x,dx)*dx[8:14] ) # (point 3.5)

            # (point 3.9) (rearranged point 4 for easier comparison with point 3.5)
            res[28] = dx[28] - (gₓ(x,dx,t)')*x[8:14] + (λ(t)')*(Fx(x,dx)*x[8:14] + Fp(x, dx)) - ( (dλ(t)')*Fdx(x,dx) + (λ(t)')*Fddx(x,dx) )*x[8:14]

            # REALLY SMAL STUFF; JUST DELETE
            res[29] = dx[29] - (dλ(t)')*Fdx(x,dx)*x[8:14]
            res[30] = dx[30] - (λ(t)')*Fddx(x,dx)*x[8:14]

            # This is the equations for my_pendulum_adjoint_konly
            # res[1:7] = (-dz[1:7]')*Fdx(T-t) + (z[1:7]')*(Fddx(T-t) - Fx(T-t)) + gₓ(T-t)
            # <=> (z[1:7]')*(Fddx(T-t) - Fx(T-t)) - (dz[1:7]')*Fdx(T-t) + gₓ(T-t)
            # <=> (z[1:7]')*(Fx(T-t) - Fddx(T-t)) + (dz[1:7]')*Fdx(T-t) - gₓ(T-t)
            # THERE IS A SIGN INCONSISTENCY!!!!!!!!!!!!!!!!
            # Yes becuase that one is solved backwards you silly...

            # These equations sensitivity are written on the intuitive form. To
            # obtain form correct for this funciton, do the following replacements:
            # s[1] -> x[8],  s[2] -> x[9],  s[3] -> x[10]
            # s[4] -> x[11], s[5] -> x[12], s[6] -> x[13]
            # and the corresponding replacements for sp and dx

            # res[8]  = -s[4] + sp[1] + 2x[1]*sp[6] + 2s[1]*dx[6]
            # res[9]  = -s[5] + sp[2] + 2x[2]*sp[6] + 2s[2]*dx[6]
            # res[10]  = abs(x[4])*x[4] + 2k*s[4]*abs(x[4]) - x[1]*sp[3] + m*sp[4] - s[1]*dx[3]
            # res[11] = abs(x[5])*x[5] + 2k*s[5]*abs(x[5]) - x[2]*sp[3] + m*sp[5] - s[2]*dx[3]
            # res[12] = -2s[1]*x[1] - 2s[2]*x[2]
            # res[13] = s[4]*x[1] + s[5]*x[2] + s[1]*x[4] + s[2]*x[5]

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

        pend0  = vcat([x1_0, x2_0], zeros(4), [atan(x1_0 / -x2_0)])
        dpend0 = [0., 0., dx3_0, dx4_0, 0., 0., 0.]
        sm0  = zeros(7)
        dsm0 = vcat(zeros(3), [-dx4_0/m, -g/m, 0., 0.])
        G0    = 0.0
        dG0   = (y(0.0)-pend0[7])^2
        Gp0   = 0.0
        dGp0  = 2(pend0[7]-y(0.0))*sm0[7]
        Gp02  = 0.0
        dGp02 = dGp0 - (λ(0.)')*(Fx(pend0, dpend0)*sm0 + Fdx(pend0, dpend0)*dsm0 + Fp(pend0, dpend0))
        Gp03  = 0.0
        dGp03 = -(λ(0.)')*Fp(pend0, dpend0) - ( (λ(0.)')*(Fx(pend0,dpend0)-Fddx(pend0,dpend0)) -(dλ(0.)')*Fdx(pend0,dpend0) - gₓ(pend0,dpend0,0.0)' )*sm0
        Gp04  = 0.0
        dGp04 = -(λ(0.)')*Fp(pend0, dpend0)
        Gp04b  = 0.0
        dGp04b = - ( (λ(0.)')*(Fx(pend0,dpend0)-Fddx(pend0,dpend0)) -(dλ(0.)')*Fdx(pend0,dpend0) - gₓ(pend0,dpend0,0.0)' )*sm0
        Gp035  = 0.0
        dGp035 = (gₓ(pend0,dpend0,0.0)')*sm0 - (λ(0.0)')*( Fx(pend0,dpend0)*sm0 + Fdx(pend0,dpend0)*dsm0 + Fp(pend0,dpend0) )
        partial    = 0.0
        dpartial   = (λ(0.0)')*Fdx(pend0,dpend0)*dsm0 + ( (dλ(0.0)')*Fdx(pend0,dpend0) + (λ(0.0)')*Fddx(pend0,dpend0) )*sm0

        # tue_deb    = (λ(0.0)')*Fdx(pend0, dpend0)
        # dtue_deb   = (dλ(0.0)')*Fdx(pend0, dpend0) + (λ(0.0)')*Fddx(pend0, dpend0)
        # NOTE: ONLY MAKE SENSE FOR k AND m SINCE ONLY THEN APPROPRIATE QUANTITIES ARE ZERO
        common   = 0.0
        dcommon  = (gₓ(pend0,dpend0,0.0)')*sm0 - (λ(0.)')*( Fx(pend0,dpend0)*sm0 + Fp(pend0,dpend0) )
        extra35  = 0.0
        dextra35 = -(λ(0.)')*Fdx(pend0,dpend0)*dsm0
        extra4   = 0.0
        dextra4  = ( (dλ(0.0)')*Fdx(pend0,dpend0) + (λ(0.0)')*Fddx(pend0,dpend0) )*sm0

        partrepL  = 0.0
        dpartrepL = (λ(0.)')*Fdx(pend0,dpend0)*dsm0
        partrepR  = 0.0
        dpartrepR = -((dλ(0.)')*Fdx(pend0,dpend0) + (λ(0.)')*Fddx(pend0,dpend0) )*sm0
        oscdeb = 0.0
        doscdeb = (gₓ(pend0,dpend0,0.0)')*sm0 - (λ(0.0)')*(Fx(pend0,dpend0)*sm0 + Fp(pend0, dpend0)) + (λ(0.0)')*Fddx(pend0,dpend0)*sm0 + (dλ(0.0)')*Fdx(pend0,dpend0)*sm0
        res29  = 0.0
        dres29 = (dλ(0.0)')*Fdx(pend0,dpend0)*sm0
        res30  = 0.0
        dres30 = (λ(0.0)')*Fddx(pend0,dpend0)*sm0

        x0   = vcat(pend0, sm0, G0, Gp0, Gp02, Gp03, Gp04, Gp04b, Gp035, partial, common, extra35, extra4, partrepL, partrepR, oscdeb, res29, res30)
        dx0  = vcat(dpend0, dsm0, dG0, dGp0, dGp02, dGp03, dGp04, dGp04b, dGp035, dpartial, dcommon, dextra35, dextra4, dpartrepL, dpartrepR, doscdeb, dres29, dres30)

        dvars = vcat(fill(true, 6), [false], fill(true, 6), [false, true, true, true, true, true, true, true, true], fill(true, 8))

        r0 = zeros(length(x0))
        f!(r0, dx0, x0, [], 0.0)
        # @info "r0 is $r0"

        # t -> 0.0 is just a dummy function, not to be used
        Model(f!, t -> 0.0, x0, dx0, dvars, r0)
    end
end

function pendulum_adj_stepbystep_g(Φ::Float64, u::Function, w::Function, θ::Array{Float64, 1}, y::Function, λ::Function, dλ::Function, T::Float64)::Model
    @warn "pendulum_adj_stepbystep_L only adapted for L-parameter currently, no others"
    let m = θ[1], L = θ[2], g = θ[3], k = θ[4]

        # x  = t -> sol(t)
        Fx = (x, dx) -> [2dx[6]        0               0   -1                  0           0   0
                          0           2*dx[6]          0   0                   -1          0   0
                         -dx[3]        0               0   2k*abs(x[4])        0           0   0
                          0          -dx[3]            0   0               2k*abs(x[5])    0   0
                          2x[1]       2x[2]            0   0                   0           0   0
                          x[4]        x[5]             0   x[1]               x[2]         0   0
                          x[2]/(L^2)  -x[1]/(L^2)      0   0                   0           0   1]
        # (namely the derivative of F with respect to the variable x_p)
        Fdx = (x, dx) -> vcat([1   0   0          0   0   2x[1]    0
                               0   1   0          0   0   2x[2]    0
                               0   0   -x[1]   m   0   0           0
                               0   0   -x[2]   0   m   0           0], zeros(3,7))
        Fddx = (x, dx) -> vcat([  0   0  0            0   0   2dx[1]    0
                                  0   0  0            0   0   2dx[2]    0
                                  0   0  -dx[1]    0   0   0            0
                                  0   0  -dx[2]    0   0   0            0], zeros(3,7))
        Fp = (x, dx) -> [0
                         0
                         0
                         m
                         0
                         0
                         0]
       gₓ = (x, dx, t) -> [0.; 0.; 0.; 0.; 0.; 0.; 2(x[7]-y(t))/T]
       # λ = [1.; 1.; 1.; 1.; 1.; 1.; 1.]

       #TODO: DON'T PASS T. BUT ALSO DON'T CALL THIS gₓ!!!! EDIT: Why...?

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

            # Sensitivity equations for L
            res[8]  = dx[8] - x[11] + 2x[8]dx[6] + 2dx[13]x[1]
            res[9]  = dx[9] - x[12] + 2x[9]dx[6] + 2dx[13]x[2]
            res[10] = 2k*x[11]*abs(x[4]) + m*dx[11] - x[8]dx[3] - dx[10]x[1]
            res[11] = 2k*x[12]*abs(x[5]) + m*dx[12] - x[9]dx[3] - dx[10]x[2] + m
            res[12] = 2x[8]x[1] + 2x[9]x[2]
            res[13] = x[8]x[4] + x[11]x[1] + x[9]x[5] + x[12]x[2]
            # Sensitivity of angle of pendulum to L
            # TODO: Analytical formula says it should be x[1]^2+x[2]^2 instead of
            # L^2 (though they should be equal), is it fine to substitute L^2 here?
            res[14] = x[14] - (x[1]*x[9] - x[8]*x[2])/(L^2)

            # res[15] = dx[15] - (y(t) - y(t;θ))^2 (point 1)
            res[15] = dx[15] - (y(t)-x[7])^2
            # res[16] = dx[16] - 2(y(t;θ)-y(t))*y_θ(t) (point 2)
            res[16] = dx[16] - 2(x[7]-y(t))*x[14]

            # # Comment out and just use explicilty
            # xθ  = x[8:14]
            # dxθ = dx[8:14]
            # res[17] = dx[17] - 2(y(t;θ)-y(t))*y_θ(t) - (λ')*(Fx(t)*xθ + Fdx(t)*dxθ + Fp(t)) (point 3)
            res[17] = dx[17] - 2(x[7]-y(t))*x[14] + (λ(t)')*(Fx(x, dx)*x[8:14] + Fdx(x, dx)*dx[8:14] + Fp(x, dx))

            # res[18] = dx[18] + (λ')*Fp(x, dx) + ( (λ')*(Fx(x,dx)-Fddx(x,dx)) - (dλ')*Fdx(x,dx) - gx(x,dx,t)' )*xθ (point 4)
            res[18] = dx[18] + (λ(t)')*Fp(x, dx) + ( (λ(t)')*(Fx(x,dx)-Fddx(x,dx)) - (dλ(t)')*Fdx(x,dx) - gₓ(x,dx,t)' )*x[8:14]

            # res[19] = dx[19] + (λ')*Fp(x, dx) (point 5)
            res[19] = dx[19] + (λ(t)')*Fp(x, dx)
            # res[20] = dx[20]  + ( (λ')*(Fx(x,dx)-Fddx(x,dx)) - (dλ')*Fdx(x,dx) - gx(x,dx,t)' )*xθ
            res[20] = dx[20] + ( (λ(t)')*(Fx(x,dx)-Fddx(x,dx)) - (dλ(t)')*Fdx(x,dx) - gₓ(x,dx,t)' )*x[8:14]

            # res[21] = dx[21] - (gx(x,dx,t)')*xθ + (λ(t)')*( Fx(x,dx)*xθ + Fdx(x,dx)*dxθ + Fp(x,dx
            res[21] = dx[21] - (gₓ(x,dx,t)')*x[8:14] + (λ(t)')*( Fx(x,dx)*x[8:14] + Fdx(x,dx)*dx[8:14] + Fp(x,dx) )

            # res[22] = dx[22] - (λ(t)')*Fdx(t)*dxθ - ( (dλ(t)')*Fdx(t) + (λ(t)')*Fddx(t) )*xθ
            res[22] = dx[22] - (λ(t)')*Fdx(x,dx)*dx[8:14] - ( (dλ(t)')*Fdx(x,dx) + (λ(t)')*Fddx(x,dx) )*x[8:14]

            # Shared between 3.5 and 4
            res[23] = dx[23] - (gₓ(x,dx,t)')*x[8:14] + (λ(t)')*( Fx(x,dx)*x[8:14] + Fp(x,dx) )
            # Extra for 3.5 (for m and k only)
            res[24] = dx[24] + (λ(t)')*Fdx(x,dx)*dx[8:14]
            # Extra for 4 (for m and k only)
            res[25] = dx[25] - ( (dλ(t)')*Fdx(x,dx) + (λ(t)')*Fddx(x,dx) )*x[8:14]

            # NOTE: LoL, these are just negated versions of the above, clearly not very equal!
            # For replacement using partial integration
            # LHS
            res[26] = dx[26] - (λ(t)')*Fdx(x,dx)*dx[8:14]
            # RHS (without term, assumed zero, which is the case for m and correct λ)
            res[27] = dx[27] + ((dλ(t)')*Fdx(x,dx) + (λ(t)')*Fddx(x,dx) )*x[8:14]

            # (point 3.9) (rearranged point 4 for easier comparison with point 3.5)
            res[28] = dx[28] - (gₓ(x,dx,t)')*x[8:14] + (λ(t)')*(Fx(x,dx)*x[8:14] + Fp(x, dx)) - ( (dλ(t)')*Fdx(x,dx) + (λ(t)')*Fddx(x,dx) )*x[8:14]

            # REALLY SMAL STUFF; JUST DELETE
            res[29] = dx[29] - (dλ(t)')*Fdx(x,dx)*x[8:14]
            res[30] = dx[30] - (λ(t)')*Fddx(x,dx)*x[8:14]

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

        pend0  = vcat([x1_0, x2_0], zeros(4), [atan(x1_0 / -x2_0)])
        dpend0 = [0., 0., dx3_0, dx4_0, 0., 0., 0.]
        s0  = zeros(7)
        ds0 = vcat(zeros(5), [-1., 0.])
        G0    = 0.0
        dG0   = (y(0.0)-pend0[7])^2
        Gp0   = 0.0
        dGp0  = 2(pend0[7]-y(0.0))*0.0
        Gp02  = 0.0
        dGp02 = dGp0 - (λ(0.)')*(Fx(pend0, dpend0)*s0 + Fdx(pend0, dpend0)*ds0 + Fp(pend0, dpend0))
        Gp03  = 0.0
        dGp03 = -(λ(0.)')*Fp(pend0, dpend0) - ( (λ(0.)')*(Fx(pend0,dpend0)-Fddx(pend0,dpend0)) -(dλ(0.)')*Fdx(pend0,dpend0) - gₓ(pend0,dpend0,0.0)' )*s0
        Gp04  = 0.0
        dGp04 = -(λ(0.)')*Fp(pend0, dpend0)
        Gp04b  = 0.0
        dGp04b = - ( (λ(0.)')*(Fx(pend0,dpend0)-Fddx(pend0,dpend0)) -(dλ(0.)')*Fdx(pend0,dpend0) - gₓ(pend0,dpend0,0.0)' )*s0
        Gp035  = 0.0
        dGp035 = (gₓ(pend0,dpend0,0.0)')*s0 - (λ(0.0)')*( Fx(pend0,dpend0)*s0 + Fdx(pend0,dpend0)*ds0 + Fp(pend0,dpend0) )
        partial    = 0.0
        dpartial   = (λ(0.0)')*Fdx(pend0,dpend0)*ds0 + ( (dλ(0.0)')*Fdx(pend0,dpend0) + (λ(0.0)')*Fddx(pend0,dpend0) )*s0

        # tue_deb    = (λ(0.0)')*Fdx(pend0, dpend0)
        # dtue_deb   = (dλ(0.0)')*Fdx(pend0, dpend0) + (λ(0.0)')*Fddx(pend0, dpend0)
        # NOTE: ONLY MAKE SENSE FOR k AND m SINCE ONLY THEN APPROPRIATE QUANTITIES ARE ZERO
        common   = 0.0
        dcommon  = (gₓ(pend0,dpend0,0.0)')*s0 - (λ(0.)')*( Fx(pend0,dpend0)*s0 + Fp(pend0,dpend0) )
        extra35  = 0.0
        dextra35 = -(λ(0.)')*Fdx(pend0,dpend0)*ds0
        extra4   = 0.0
        dextra4  = ( (dλ(0.0)')*Fdx(pend0,dpend0) + (λ(0.0)')*Fddx(pend0,dpend0) )*s0

        partrepL  = 0.0
        dpartrepL = (λ(0.)')*Fdx(pend0,dpend0)*ds0
        partrepR  = 0.0
        dpartrepR = -((dλ(0.)')*Fdx(pend0,dpend0) + (λ(0.)')*Fddx(pend0,dpend0) )*s0
        oscdeb = 0.0
        doscdeb = (gₓ(pend0,dpend0,0.0)')*s0 - (λ(0.0)')*(Fx(pend0,dpend0)*s0 + Fp(pend0, dpend0)) + (λ(0.0)')*Fddx(pend0,dpend0)*s0 + (dλ(0.0)')*Fdx(pend0,dpend0)*s0
        res29  = 0.0
        dres29 = (dλ(0.0)')*Fdx(pend0,dpend0)*s0
        res30  = 0.0
        dres30 = (λ(0.0)')*Fddx(pend0,dpend0)*s0

        x0   = vcat(pend0, s0, G0, Gp0, Gp02, Gp03, Gp04, Gp04b, Gp035, partial, common, extra35, extra4, partrepL, partrepR, oscdeb, res29, res30)
        dx0  = vcat(dpend0, ds0, dG0, dGp0, dGp02, dGp03, dGp04, dGp04b, dGp035, dpartial, dcommon, dextra35, dextra4, dpartrepL, dpartrepR, doscdeb, dres29, dres30)

        dvars = vcat(fill(true, 6), [false], fill(true, 6), [false, true, true, true, true, true, true, true, true], fill(true, 8))
        # dvars = vcat(fill(true, 6), [false], fill(true, 6), [false, true, true, true, true, true, true, true, true], fill(true, 7))

        r0 = zeros(length(x0))
        f!(r0, dx0, x0, [], 0.0)
        # @info "r0 is $r0"

        # t -> 0.0 is just a dummy function, not to be used
        Model(f!, t -> 0.0, x0, dx0, dvars, r0)
    end
end

function pendulum_adj_stepbystep_deb(Φ::Float64, u::Function, w::Function, θ::Array{Float64, 1}, y::Function, λ::Function, dλ::Function, T::Float64)::Model
    @warn "pendulum_adj_stepbystep_deb only adapted for pi-parameter currently, no others"
    let m = θ[1], L = θ[2], g = θ[3], k = θ[4]

        my_ind = 2
        i = my_ind
        # @warn "pendulum_adj_stepbystep_deb. m: $m, L: $L, g: $g, k: $k"


        # x  = t -> sol(t)
        Fx = (x, dx) -> [2dx[6]        0               0   -1                  0           0   0
                          0           2*dx[6]          0   0                   -1          0   0
                         -dx[3]        0               0   2k*abs(x[4])        0           0   0
                          0          -dx[3]            0   0               2k*abs(x[5])    0   0
                          2x[1]       2x[2]            0   0                   0           0   0
                          x[4]        x[5]             0   x[1]               x[2]         0   0
                          x[2]/(L^2)  -x[1]/(L^2)      0   0                   0           0   1]
        # (namely the derivative of F with respect to the variable x_p)
        Fdx = (x, dx) -> vcat([1   0   0          0   0   2x[1]    0
                               0   1   0          0   0   2x[2]    0
                               0   0   -x[1]   m   0   0           0
                               0   0   -x[2]   0   m   0           0], zeros(3,7))
        Fddx = (x, dx) -> vcat([  0   0  0            0   0   2dx[1]    0
                                  0   0  0            0   0   2dx[2]    0
                                  0   0  -dx[1]    0   0   0            0
                                  0   0  -dx[2]    0   0   0            0], zeros(3,7))
        Fp = (x, dx) -> [ Int(i==1)
                          Int(i==2)
                          Int(i==3)
                          Int(i==4)
                          Int(i==5)
                          Int(i==6)
                          Int(i==7)]
       gₓ = (x, dx, t) -> [0.; 0.; 0.; 0.; 0.; 0.; 2(x[7]-y(t))/T]
       # λ = [1.; 1.; 1.; 1.; 1.; 1.; 1.]
       pvec = zeros(7); pvec[i] = 0.
       u0 = u(0.0)[1]
       w0 = w(0.0)[1]
       L_new = sqrt(L^2 - pvec[5])
       x1_0 = L_new * sin(Φ)
       x2_0 = -L_new * cos(Φ)
       x5_0 = -pvec[6]/x2_0
       pend0 = vcat([x1_0, x2_0], zeros(2), [x5_0, 0.0, atan(x1_0 / -x2_0)-pvec[7]])
       dx2_0 = x5_0 - pvec[2]
       dx4_0 = (u0 + w0^2 - pvec[3])/m
       dx5_0 = (-k*abs(x5_0)*x5_0 - m*g - pvec[4])/m
       dpend0 = [-pvec[1], dx2_0, 0.0, dx4_0, dx5_0, 0.0, 0.0]

       s1_0 = 0.0#equivalent -Int(i==5)*x1_0/(2*L^2)
       s2_0 = -Int(i==5)/(2x2_0)  # Very different, is it right?
       s5_0 = -Int(i==6)/(x2_0)
       s0  = [s1_0, s2_0, 0.0, 0.0, s5_0, 0.0, -Int(i==7)]
       ds1_0 = -Int(i==1)
       ds2_0 = -Int(i==2) + s5_0
       ds4_0 = -Int(i==3)/m#Almost equivalent : (Int(i==3)+dx3_0*s1_0)/m
       # ds5_0 = (-Int(i==4)+dx3_0*s2_0)/m
       ds5_0 = (-Int(i==4)+0.0*s2_0)/m
       ds7_0 = Int(i==1)*x2_0/(L^2)#Not even close: (x1_0*(Int(i==1)+s4_0) - x2_0*(Int(i==2)+s5_0))/(L^2)
       ds0 = [ds1_0, ds2_0, 0.0, ds4_0, ds5_0, 0.0, ds7_0]

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
            # Sensitivity equations for p1                      # here
            res[8]  = dx[8] - x[11] + 2x[8]dx[6] + 2dx[13]x[1]  + Int(i==1)
            res[9]  = dx[9] - x[12] + 2x[9]dx[6] + 2dx[13]x[2] + Int(i==2)
            res[10] = 2k*x[11]*abs(x[4]) + m*dx[11] - x[8]dx[3] - dx[10]x[1] + Int(i==3)
            res[11] = 2k*x[12]*abs(x[5]) + m*dx[12] - x[9]dx[3] - dx[10]x[2] + Int(i==4)
            res[12] = 2x[8]x[1] + 2x[9]x[2] + Int(i==5)
            res[13] = x[8]x[4] + x[11]x[1] + x[9]x[5] + x[12]x[2] + Int(i==6)
            # Sensitivity of angle of pendulum to L
            # TODO: Analytical formula says it should be x[1]^2+x[2]^2 instead of
            # L^2 (though they should be equal), is it fine to substitute L^2 here?
            res[14] = x[14] - (x[1]*x[9] - x[8]*x[2])/(L^2) + Int(i==7)

            # res[15] = dx[15] - (y(t) - y(t;θ))^2 (point 1)
            res[15] = dx[15] - (y(t)-x[7])^2
            # res[16] = dx[16] - 2(y(t;θ)-y(t))*y_θ(t) (point 2)
            res[16] = dx[16] - 2(x[7]-y(t))*x[14]

            # # Comment out and just use explicilty
            # xθ  = x[8:14]
            # dxθ = dx[8:14]
            # res[17] = dx[17] - 2(y(t;θ)-y(t))*y_θ(t) - (λ')*(Fx(t)*xθ + Fdx(t)*dxθ + Fp(t)) (point 3)
            res[17] = dx[17] - 2(x[7]-y(t))*x[14] + (λ(t)')*(Fx(x, dx)*x[8:14] + Fdx(x, dx)*dx[8:14] + Fp(x, dx))

            # res[18] = dx[18] + (λ')*Fp(x, dx) + ( (λ')*(Fx(x,dx)-Fddx(x,dx)) - (dλ')*Fdx(x,dx) - gx(x,dx,t)' )*xθ (point 4)
            res[18] = dx[18] + (λ(t)')*Fp(x, dx) + ( (λ(t)')*(Fx(x,dx)-Fddx(x,dx)) - (dλ(t)')*Fdx(x,dx) - gₓ(x,dx,t)' )*x[8:14]

            # res[19] = dx[19] + (λ')*Fp(x, dx) (point 5)
            res[19] = dx[19] + (λ(t)')*Fp(x, dx)
            # res[20] = dx[20]  + ( (λ')*(Fx(x,dx)-Fddx(x,dx)) - (dλ')*Fdx(x,dx) - gx(x,dx,t)' )*xθ
            res[20] = dx[20] + ( (λ(t)')*(Fx(x,dx)-Fddx(x,dx)) - (dλ(t)')*Fdx(x,dx) - gₓ(x,dx,t)' )*x[8:14]

            # res[21] = dx[21] - (gx(x,dx,t)')*xθ + (λ(t)')*( Fx(x,dx)*xθ + Fdx(x,dx)*dxθ + Fp(x,dx
            res[21] = dx[21] - (gₓ(x,dx,t)')*x[8:14] + (λ(t)')*( Fx(x,dx)*x[8:14] + Fdx(x,dx)*dx[8:14] + Fp(x,dx) )

            # res[22] = dx[22] - (λ(t)')*Fdx(t)*dxθ - ( (dλ(t)')*Fdx(t) + (λ(t)')*Fddx(t) )*xθ
            res[22] = dx[22] - (λ(t)')*Fdx(x,dx)*dx[8:14] - ( (dλ(t)')*Fdx(x,dx) + (λ(t)')*Fddx(x,dx) )*x[8:14]

            # Shared between 3.5 and 4
            res[23] = dx[23] - (gₓ(x,dx,t)')*x[8:14] + (λ(t)')*( Fx(x,dx)*x[8:14] + Fp(x,dx) )
            # Extra for 3.5 (for m and k only)
            res[24] = dx[24] + (λ(t)')*Fdx(x,dx)*dx[8:14]
            # Extra for 4 (for m and k only)
            res[25] = dx[25] - ( (dλ(t)')*Fdx(x,dx) + (λ(t)')*Fddx(x,dx) )*x[8:14]

            # NOTE: LoL, these are just negated versions of the above, clearly not very equal!
            # For replacement using partial integration
            # LHS
            res[26] = dx[26] - (λ(t)')*Fdx(x,dx)*dx[8:14]
            # RHS (without term, assumed zero, which is the case for m and correct λ)
            res[27] = dx[27] + ((dλ(t)')*Fdx(x,dx) + (λ(t)')*Fddx(x,dx) )*x[8:14]

            # (point 3.9) (rearranged point 4 for easier comparison with point 3.5)
            res[28] = dx[28] - (gₓ(x,dx,t)')*x[8:14] + (λ(t)')*(Fx(x,dx)*x[8:14] + Fp(x, dx)) - ( (dλ(t)')*Fdx(x,dx) + (λ(t)')*Fddx(x,dx) )*x[8:14]

            # REALLY SMAL STUFF; JUST DELETE
            res[29] = dx[29] - (dλ(t)')*Fdx(x,dx)*x[8:14]
            res[30] = dx[30] - (λ(t)')*Fddx(x,dx)*x[8:14]

            nothing
        end

        # # ORIGINAL
        # # Finding consistent initial conditions
        # # Initial values, the pendulum starts at rest
        # u0 = u(0.0)[1]
        # w0 = w(0.0)[1]
        # x1_0 = L * sin(Φ)
        # x2_0 = -L * cos(Φ)
        # dx3_0 = m*g/x2_0
        # dx4_0 = -g*tan(Φ) + (u0 + w0^2)/m
        # FOR DEB
        pvec = zeros(7); pvec[i] = 0.
        u0 = u(0.0)[1]
        w0 = w(0.0)[1]
        L_new = sqrt(L^2 - pvec[5])
        x1_0 = L_new * sin(Φ)
        x2_0 = -L_new * cos(Φ)
        x5_0 = -pvec[6]/x2_0
        pend0 = vcat([x1_0, x2_0], zeros(2), [x5_0, 0.0, atan(x1_0 / -x2_0)-pvec[7]])
        dx2_0 = x5_0 - pvec[2]
        dx4_0 = (u0 + w0^2 - pvec[3])/m
        dx5_0 = (-k*abs(x5_0)*x5_0 - m*g - pvec[4])/m
        dpend0 = [-pvec[1], dx2_0, 0.0, dx4_0, dx5_0, 0.0, 0.0]

        s1_0 = 0.0#equivalent -Int(i==5)*x1_0/(2*L^2)
        s2_0 = -Int(i==5)/(2x2_0)  # Very different, is it right?
        s5_0 = -Int(i==6)/(x2_0)
        s0  = [s1_0, s2_0, 0.0, 0.0, s5_0, 0.0, -Int(i==7)]
        ds1_0 = -Int(i==1)
        ds2_0 = -Int(i==2) + s5_0
        ds4_0 = -Int(i==3)/m#Almost equivalent : (Int(i==3)+dx3_0*s1_0)/m
        # ds5_0 = (-Int(i==4)+dx3_0*s2_0)/m
        ds5_0 = (-Int(i==4)+0.0*s2_0)/m
        ds7_0 = Int(i==1)*x2_0/(L^2)#Not even close: (x1_0*(Int(i==1)+s4_0) - x2_0*(Int(i==2)+s5_0))/(L^2)
        ds0 = [ds1_0, ds2_0, 0.0, ds4_0, ds5_0, 0.0, ds7_0]

        # # ORIGINAL, for deb above
        # pend0  = vcat([x1_0, x2_0], zeros(4), [atan(x1_0 / -x2_0)])
        # dpend0 = [0., 0., dx3_0, dx4_0, 0., 0., 0.]
        # s0  = zeros(7)
        # ds0 = zeros(7)
        G0    = 0.0
        dG0   = (y(0.0)-pend0[7])^2
        Gp0   = 0.0
        dGp0  = 2(pend0[7]-y(0.0))*0.0
        Gp02  = 0.0
        dGp02 = dGp0 - (λ(0.)')*(Fx(pend0, dpend0)*s0 + Fdx(pend0, dpend0)*ds0 + Fp(pend0, dpend0))
        Gp03  = 0.0
        dGp03 = -(λ(0.)')*Fp(pend0, dpend0) - ( (λ(0.)')*(Fx(pend0,dpend0)-Fddx(pend0,dpend0)) -(dλ(0.)')*Fdx(pend0,dpend0) - gₓ(pend0,dpend0,0.0)' )*s0
        Gp04  = 0.0
        dGp04 = -(λ(0.)')*Fp(pend0, dpend0)
        Gp04b  = 0.0
        dGp04b = - ( (λ(0.)')*(Fx(pend0,dpend0)-Fddx(pend0,dpend0)) -(dλ(0.)')*Fdx(pend0,dpend0) - gₓ(pend0,dpend0,0.0)' )*s0
        Gp035  = 0.0
        dGp035 = (gₓ(pend0,dpend0,0.0)')*s0 - (λ(0.0)')*( Fx(pend0,dpend0)*s0 + Fdx(pend0,dpend0)*ds0 + Fp(pend0,dpend0) )
        partial    = 0.0
        dpartial   = (λ(0.0)')*Fdx(pend0,dpend0)*ds0 + ( (dλ(0.0)')*Fdx(pend0,dpend0) + (λ(0.0)')*Fddx(pend0,dpend0) )*s0

        # tue_deb    = (λ(0.0)')*Fdx(pend0, dpend0)
        # dtue_deb   = (dλ(0.0)')*Fdx(pend0, dpend0) + (λ(0.0)')*Fddx(pend0, dpend0)
        # NOTE: ONLY MAKE SENSE FOR k AND m SINCE ONLY THEN APPROPRIATE QUANTITIES ARE ZERO
        common   = 0.0
        dcommon  = (gₓ(pend0,dpend0,0.0)')*s0 - (λ(0.)')*( Fx(pend0,dpend0)*s0 + Fp(pend0,dpend0) )
        extra35  = 0.0
        dextra35 = -(λ(0.)')*Fdx(pend0,dpend0)*ds0
        extra4   = 0.0
        dextra4  = ( (dλ(0.0)')*Fdx(pend0,dpend0) + (λ(0.0)')*Fddx(pend0,dpend0) )*s0

        partrepL  = 0.0
        dpartrepL = (λ(0.)')*Fdx(pend0,dpend0)*ds0
        partrepR  = 0.0
        dpartrepR = -((dλ(0.)')*Fdx(pend0,dpend0) + (λ(0.)')*Fddx(pend0,dpend0) )*s0
        oscdeb = 0.0
        doscdeb = (gₓ(pend0,dpend0,0.0)')*s0 - (λ(0.0)')*(Fx(pend0,dpend0)*s0 + Fp(pend0, dpend0)) + (λ(0.0)')*Fddx(pend0,dpend0)*s0 + (dλ(0.0)')*Fdx(pend0,dpend0)*s0
        res29  = 0.0
        dres29 = (dλ(0.0)')*Fdx(pend0,dpend0)*s0
        res30  = 0.0
        dres30 = (λ(0.0)')*Fddx(pend0,dpend0)*s0

        x0   = vcat(pend0, s0, G0, Gp0, Gp02, Gp03, Gp04, Gp04b, Gp035, partial, common, extra35, extra4, partrepL, partrepR, oscdeb, res29, res30)
        dx0  = vcat(dpend0, ds0, dG0, dGp0, dGp02, dGp03, dGp04, dGp04b, dGp035, dpartial, dcommon, dextra35, dextra4, dpartrepL, dpartrepR, doscdeb, dres29, dres30)

        dvars = vcat(fill(true, 6), [false], fill(true, 6), [false, true, true, true, true, true, true, true, true], fill(true, 8))
        # dvars = vcat(fill(true, 6), [false], fill(true, 6), [false, true, true, true, true, true, true, true, true], fill(true, 7))

        r0 = zeros(length(x0))
        f!(r0, dx0, x0, [], 0.0)
        # @info "r0 is $r0"

        # t -> 0.0 is just a dummy function, not to be used
        Model(f!, t -> 0.0, x0, dx0, dvars, r0)
    end
end

function deleteme_stepbystep_deb(Φ::Float64, u::Function, w::Function, θ::Array{Float64, 1}, y::Function, λ::Function, dλ::Function, T::Float64)::Model
    @warn "pendulum_adj_stepbystep_deb only adapted for pi-parameter currently, no others"
    let m = θ[1], L = θ[2], g = θ[3], k = θ[4]

        my_ind = 2
        i = my_ind
        # @warn "pendulum_adj_stepbystep_deb. m: $m, L: $L, g: $g, k: $k"


        # x  = t -> sol(t)
        Fx = (x, dx) -> [2dx[6]        0               0   -1                  0           0   0
                          0           2*dx[6]          0   0                   -1          0   0
                         -dx[3]        0               0   2k*abs(x[4])        0           0   0
                          0          -dx[3]            0   0               2k*abs(x[5])    0   0
                          2x[1]       2x[2]            0   0                   0           0   0
                          x[4]        x[5]             0   x[1]               x[2]         0   0
                          x[2]/(L^2)  -x[1]/(L^2)      0   0                   0           0   1]
        # (namely the derivative of F with respect to the variable x_p)
        Fdx = (x, dx) -> vcat([1   0   0          0   0   2x[1]    0
                               0   1   0          0   0   2x[2]    0
                               0   0   -x[1]   m   0   0           0
                               0   0   -x[2]   0   m   0           0], zeros(3,7))
        Fddx = (x, dx) -> vcat([  0   0  0            0   0   2dx[1]    0
                                  0   0  0            0   0   2dx[2]    0
                                  0   0  -dx[1]    0   0   0            0
                                  0   0  -dx[2]    0   0   0            0], zeros(3,7))
        Fp = (x, dx) -> [ Int(i==1)
                          Int(i==2)
                          Int(i==3)
                          Int(i==4)
                          Int(i==5)
                          Int(i==6)
                          Int(i==7)]
       gₓ = (x, dx, t) -> [0.; 0.; 0.; 0.; 0.; 0.; 2(x[7]-y(t))/T]
       # λ = [1.; 1.; 1.; 1.; 1.; 1.; 1.]
       pvec = zeros(7); pvec[i] = 0.
       u0 = u(0.0)[1]
       w0 = w(0.0)[1]
       L_new = sqrt(L^2 - pvec[5])
       x1_0 = L_new * sin(Φ)
       x2_0 = -L_new * cos(Φ)
       x5_0 = -pvec[6]/x2_0
       pend0 = vcat([x1_0, x2_0], zeros(2), [x5_0, 0.0, atan(x1_0 / -x2_0)-pvec[7]])
       dx2_0 = x5_0 - pvec[2]
       dx4_0 = (u0 + w0^2 - pvec[3])/m
       dx5_0 = (-k*abs(x5_0)*x5_0 - m*g - pvec[4])/m
       dpend0 = [-pvec[1], dx2_0, 0.0, dx4_0, dx5_0, 0.0, 0.0]

       s1_0 = 0.0#equivalent -Int(i==5)*x1_0/(2*L^2)
       s2_0 = -Int(i==5)/(2x2_0)  # Very different, is it right?
       s5_0 = -Int(i==6)/(x2_0)
       s0  = [s1_0, s2_0, 0.0, 0.0, s5_0, 0.0, -Int(i==7)]
       ds1_0 = -Int(i==1)
       ds2_0 = -Int(i==2) + s5_0
       ds4_0 = -Int(i==3)/m#Almost equivalent : (Int(i==3)+dx3_0*s1_0)/m
       # ds5_0 = (-Int(i==4)+dx3_0*s2_0)/m
       ds5_0 = (-Int(i==4)+0.0*s2_0)/m
       ds7_0 = Int(i==1)*x2_0/(L^2)#Not even close: (x1_0*(Int(i==1)+s4_0) - x2_0*(Int(i==2)+s5_0))/(L^2)
       ds0 = [ds1_0, ds2_0, 0.0, ds4_0, ds5_0, 0.0, ds7_0]

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
            # Sensitivity equations for p1                      # here
            res[8]  = dx[8] - x[11] + 2x[8]dx[6] + 2dx[13]x[1]  + Int(i==1)
            res[9]  = dx[9] - x[12] + 2x[9]dx[6] + 2dx[13]x[2] + Int(i==2)
            res[10] = 2k*x[11]*abs(x[4]) + m*dx[11] - x[8]dx[3] - dx[10]x[1] + Int(i==3)
            res[11] = 2k*x[12]*abs(x[5]) + m*dx[12] - x[9]dx[3] - dx[10]x[2] + Int(i==4)
            res[12] = 2x[8]x[1] + 2x[9]x[2] + Int(i==5)
            res[13] = x[8]x[4] + x[11]x[1] + x[9]x[5] + x[12]x[2] + Int(i==6)
            # Sensitivity of angle of pendulum to L
            # TODO: Analytical formula says it should be x[1]^2+x[2]^2 instead of
            # L^2 (though they should be equal), is it fine to substitute L^2 here?
            res[14] = x[14] - (x[1]*x[9] - x[8]*x[2])/(L^2) + Int(i==7)

            # res[15] = dx[15] - (y(t) - y(t;θ))^2 (point 1)
            res[15] = dx[15] - (y(t)-x[7])^2
            # res[16] = dx[16] - 2(y(t;θ)-y(t))*y_θ(t) (point 2)
            res[16] = dx[16] - 2(x[7]-y(t))*x[14]
            nothing
        end

        # # ORIGINAL
        # # Finding consistent initial conditions
        # # Initial values, the pendulum starts at rest
        # u0 = u(0.0)[1]
        # w0 = w(0.0)[1]
        # x1_0 = L * sin(Φ)
        # x2_0 = -L * cos(Φ)
        # dx3_0 = m*g/x2_0
        # dx4_0 = -g*tan(Φ) + (u0 + w0^2)/m
        # FOR DEB
        pvec = zeros(7); pvec[i] = 0.
        u0 = u(0.0)[1]
        w0 = w(0.0)[1]
        L_new = sqrt(L^2 - pvec[5])
        x1_0 = L_new * sin(Φ)
        x2_0 = -L_new * cos(Φ)
        x5_0 = -pvec[6]/x2_0
        pend0 = vcat([x1_0, x2_0], zeros(2), [x5_0, 0.0, atan(x1_0 / -x2_0)-pvec[7]])
        dx2_0 = x5_0 - pvec[2]
        dx4_0 = (u0 + w0^2 - pvec[3])/m
        dx5_0 = (-k*abs(x5_0)*x5_0 - m*g - pvec[4])/m
        dpend0 = [-pvec[1], dx2_0, 0.0, dx4_0, dx5_0, 0.0, 0.0]

        s1_0 = 0.0#equivalent -Int(i==5)*x1_0/(2*L^2)
        s2_0 = -Int(i==5)/(2x2_0)  # Very different, is it right?
        s5_0 = -Int(i==6)/(x2_0)
        s0  = [s1_0, s2_0, 0.0, 0.0, s5_0, 0.0, -Int(i==7)]
        ds1_0 = -Int(i==1)
        ds2_0 = -Int(i==2) + s5_0
        ds4_0 = -Int(i==3)/m#Almost equivalent : (Int(i==3)+dx3_0*s1_0)/m
        # ds5_0 = (-Int(i==4)+dx3_0*s2_0)/m
        ds5_0 = (-Int(i==4)+0.0*s2_0)/m
        ds7_0 = Int(i==1)*x2_0/(L^2)#Not even close: (x1_0*(Int(i==1)+s4_0) - x2_0*(Int(i==2)+s5_0))/(L^2)
        ds0 = [ds1_0, ds2_0, 0.0, ds4_0, ds5_0, 0.0, ds7_0]

        # # ORIGINAL, for deb above
        # pend0  = vcat([x1_0, x2_0], zeros(4), [atan(x1_0 / -x2_0)])
        # dpend0 = [0., 0., dx3_0, dx4_0, 0., 0., 0.]
        # s0  = zeros(7)
        # ds0 = zeros(7)
        G0    = 0.0
        dG0   = (y(0.0)-pend0[7])^2
        Gp0   = 0.0
        dGp0  = 2(pend0[7]-y(0.0))*0.0

        x0   = vcat(pend0, s0, G0, Gp0)
        dx0  = vcat(dpend0, ds0, dG0, dGp0)

        dvars = vcat(fill(true, 6), [false], fill(true, 6), [false, true, true])
        # dvars = vcat(fill(true, 6), [false], fill(true, 6), [false, true, true, true, true, true, true, true, true], fill(true, 7))

        r0 = zeros(length(x0))
        f!(r0, dx0, x0, [], 0.0)
        # @info "r0 is $r0"

        # t -> 0.0 is just a dummy function, not to be used
        Model(f!, t -> 0.0, x0, dx0, dvars, r0)
    end
end

function pendulum_adj_stepbystep_m_alt(Φ::Float64, u::Function, w::Function, θ::Array{Float64, 1}, y::Function, λ::Function, dλ::Function, x::Function, dx::Function, T::Float64)::Model
    @warn "pendulum_adj_stepbystep_m only adapted for m-parameter currently, no others"
    let m = θ[1], L = θ[2], g = θ[3], k = θ[4]

        # x  = t -> sol(t)
        Fx = t -> [2dx(t)[6]        0                0   -1                   0           0   0
                      0           2*dx(t)[6]         0   0                   -1           0   0
                     -dx(t)[3]        0              0   2k*abs(x(t)[4])      0           0   0
                      0          -dx(t)[3]           0   0               2k*abs(x(t)[5])  0   0
                      2x(t)[1]       2x(t)[2]        0   0                   0            0   0
                      x(t)[4]        x(t)[5]         0   x(t)[1]          x(t)[2]         0   0
                      x(t)[2]/(L^2)  -x(t)[1]/(L^2)  0   0                   0            0   1]
        # (namely the derivative of F with respect to the variable x_p)
        Fdx = t -> vcat([1   0   0          0   0   2x(t)[1]    0
                         0   1   0          0   0   2x(t)[2]    0
                         0   0   -x(t)[1]   m   0   0           0
                         0   0   -x(t)[2]   0   m   0           0], zeros(3,7))
        Fddx = t -> vcat([  0   0  0            0   0   2dx(t)[1]    0
                            0   0  0            0   0   2dx(t)[2]    0
                            0   0  -dx(t)[1]    0   0   0            0
                            0   0  -dx(t)[2]    0   0   0            0], zeros(3,7))
        Fp = t -> [ .0
                  .0
                  dx(t)[4]
                  dx(t)[5]+g
                  .0
                  .0
                  .0]
       gₓ = t -> [0.; 0.; 0.; 0.; 0.; 0.; 2(x(t)[7]-y(t))/T]
       # λ = [1.; 1.; 1.; 1.; 1.; 1.; 1.]

       #TODO: DON'T PASS T. BUT ALSO DON'T CALL THIS gₓ!!!! EDIT: Why...?

        # the residual function
        function f!(res, dz, z, θ, t)
            wt = w(t)
            ut = u(t)

            # res[15] = dx[15] - (y(t) - y(t;θ))^2 (point 1)
            res[1] = dz[1] - (y(t)-x(t)[7])^2
            # res[2] = dx[2] - 2(y(t;θ)-y(t))*y_θ(t) (point 2)
            res[2] = dz[2] - 2(x(t)[7]-y(t))*x(t)[14]

            # # Comment out and just use explicilty
            # xθ  = x[8:14]
            # dxθ = dx[8:14]
            # res[3] = dx[3] - 2(y(t;θ)-y(t))*y_θ(t) - (λ')*(Fx(t)*xθ + Fdx(t)*dxθ + Fp(t)) (point 3)
            res[3] = dz[3] - 2(x(t)[7]-y(t))*x(t)[14] + (λ(t)')*(Fx(t)*x(t)[8:14] + Fdx(x, dx)*dx(t)[8:14] + Fp(t))

            # res[4] = dx[4] + (λ')*Fp(x, dx) + ( (λ')*(Fx(t)-Fddx(t)) - (dλ')*Fdx(t) - gx(x,dx,t)' )*xθ (point 4)
            res[4] = dz[4] + (λ(t)')*Fp(t) + ( (λ(t)')*(Fx(t)-Fddx(t)) - (dλ(t)')*Fdx(t) - gₓ(t)' )*x(t)[8:14]

            # res[5] = dx[5] + (λ')*Fp(x, dx) (point 5)
            res[5] = dz[5] + (λ(t)')*Fp(t)
            # res[6] = dx[6]  + ( (λ')*(Fx(t)-Fddx(t)) - (dλ')*Fdx(t) - gx(x,dx,t)' )*xθ
            res[6] = dz[6] + ( (λ(t)')*(Fx(t)-Fddx(t)) - (dλ(t)')*Fdx(t) - gₓ(t)' )*x(t)[8:14] # (bonus point (remainder?))

            # res[7] = dx[7] - (gx(x,dx,t)')*xθ + (λ(t)')*( Fx(t)*xθ + Fdx(t)*dxθ + Fp(x,dx
            res[7] = dz[7] - (gₓ(t)')*x(t)[8:14] + (λ(t)')*( Fx(t)*x[8:14] + Fdx(t)*dx(t)[8:14] + Fp(t) ) # (point 3.5)

            # res[8] = dx[8] - (λ(t)')*Fdx(t)*dxθ - ( (dλ(t)')*Fdx(t) + (λ(t)')*Fddx(t) )*xθ
            res[8] = dz[8] - (λ(t)')*Fdx(t)*dx(t)[8:14] - ( (dλ(t)')*Fdx(t) + (λ(t)')*Fddx(t) )*x(t)[8:14]     # (point 6)

            # res[23:29] = dx[23:29] - ( (dλ(t)')*Fdx(t) + (λ(t)')*Fddx(t) )[:]

            # Shared between 3.5 and 4
            res[9] = dz[9] - (gₓ(t)')*x(t)[8:14] + (λ(t)')*( Fx(t)*x(t)[8:14] + Fp(t) )
            # Extra for 3.5 (for m and k only)
            res[10] = dz[10] + (λ(t)')*Fdx(t)*dx(t)[8:14]
            # Extra for 4 (for m and k only)
            res[11] = dz[11] - ( (dλ(t)')*Fdx(t) + (λ(t)')*Fddx(t) )*x(t)[8:14]

            # For replacement using partial integration
            # LHS
            res[12] = dz[12] - (λ(t)')*Fdx(t)*dx(t)[8:14]
            # RHS (without term, assumed zero, which is the case for m and correct λ)
            res[13] = dz[13] + ((dλ(t)')*Fdx(t) + (λ(t)')*Fddx(t) )*x(t)[8:14]

            # This is the equations for my_pendulum_adjoint_konly
            # res[1:7] = (-dz[1:7]')*Fdx(T-t) + (z[1:7]')*(Fddx(T-t) - Fx(T-t)) + gₓ(T-t)
            # <=> (z[1:7]')*(Fddx(T-t) - Fx(T-t)) - (dz[1:7]')*Fdx(T-t) + gₓ(T-t)
            # <=> (z[1:7]')*(Fx(T-t) - Fddx(T-t)) + (dz[1:7]')*Fdx(T-t) - gₓ(T-t)
            # THERE IS A SIGN INCONSISTENCY!!!!!!!!!!!!!!!!
            # Yes becuase that one is solved backwards you silly...

            # These equations sensitivity are written on the intuitive form. To
            # obtain form correct for this funciton, do the following replacements:
            # s[1] -> x[8],  s[2] -> x[9],  s[3] -> x[10]
            # s[4] -> x[11], s[5] -> x[12], s[6] -> x[13]
            # and the corresponding replacements for sp and dx

            # res[8]  = -s[4] + sp[1] + 2x[1]*sp[6] + 2s[1]*dx[6]
            # res[9]  = -s[5] + sp[2] + 2x[2]*sp[6] + 2s[2]*dx[6]
            # res[10]  = abs(x[4])*x[4] + 2k*s[4]*abs(x[4]) - x[1]*sp[3] + m*sp[4] - s[1]*dx[3]
            # res[11] = abs(x[5])*x[5] + 2k*s[5]*abs(x[5]) - x[2]*sp[3] + m*sp[5] - s[2]*dx[3]
            # res[12] = -2s[1]*x[1] - 2s[2]*x[2]
            # res[13] = s[4]*x[1] + s[5]*x[2] + s[1]*x[4] + s[2]*x[5]

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

        pend0  = vcat([x1_0, x2_0], zeros(4), [atan(x1_0 / -x2_0)])
        dpend0 = [0., 0., dx3_0, dx4_0, 0., 0., 0.]
        sm0  = zeros(7)
        dsm0 = vcat(zeros(3), [-dx4_0/m, -g/m, 0., 0.])
        G0    = 0.0
        dG0   = (y(0.0)-pend0[7])^2
        Gp0   = 0.0
        dGp0  = 2(pend0[7]-y(0.0))*sm0[7]
        Gp02  = 0.0
        dGp02 = dGp0 - (λ(0.)')*(Fx(0.0)*sm0 + Fdx(0.0)*dsm0 + Fp(0.0))
        Gp03  = 0.0
        dGp03 = -(λ(0.)')*Fp(0.0) - ( (λ(0.)')*(Fx(0.0)-Fddx(0.0)) -(dλ(0.)')*Fdx(0.0) - gₓ(0.0)' )*sm0
        Gp04  = 0.0
        dGp04 = -(λ(0.)')*Fp(0.0)
        Gp04b  = 0.0
        dGp04b = - ( (λ(0.)')*(Fx(0.0)-Fddx(0.0)) -(dλ(0.)')*Fdx(0.0) - gₓ(0.0)' )*sm0
        Gp035  = 0.0
        dGp035 = (gₓ(0.0)')*sm0 - (λ(0.0)')*( Fx(0.0)*sm0 + Fdx(0.0)*dsm0 + Fp(0.0) )
        partial    = 0.0
        dpartial   = (λ(0.0)')*Fdx(0.0)*dsm0 + ( (dλ(0.0)')*Fdx(0.0) + (λ(0.0)')*Fddx(0.0) )*sm0

        # tue_deb    = (λ(0.0)')*Fdx(0.0)
        # dtue_deb   = (dλ(0.0)')*Fdx(0.0) + (λ(0.0)')*Fddx(0.0)
        # NOTE: ONLY MAKE SENSE FOR k AND m SINCE ONLY THEN APPROPRIATE QUANTITIES ARE ZERO
        common   = 0.0
        dcommon  = (gₓ(0.0)')*sm0 - (λ(0.)')*( Fx(0.0)*sm0 + Fp(0.0) )
        extra35  = 0.0
        dextra35 = -(λ(0.)')*Fdx(0.0)*dsm0
        extra4   = 0.0
        dextra4  = ( (dλ(0.0)')*Fdx(0.0) + (λ(0.0)')*Fddx(0.0) )*sm0

        partrepL  = 0.0
        dpartrepL = (λ(0.)')*Fdx(0.0)*dsm0
        partrepR  = 0.0
        dpartrepR = -((dλ(0.)')*Fdx(0.0) + (λ(0.)')*Fddx(0.0) )*sm0

        x0   = vcat(G0, Gp0, Gp02, Gp03, Gp04, Gp04b, Gp035, partial, common, extra35, extra4, partrepL, partrepR)
        dx0  = vcat(dG0, dGp0, dGp02, dGp03, dGp04, dGp04b, dGp035, dpartial, dcommon, dextra35, dextra4, dpartrepL, dpartrepR)

        dvars = vcat([true, true, true, true, true, true, true, true], fill(true, 5))

        r0 = zeros(length(x0))
        f!(r0, dx0, x0, [], 0.0)
        # @info "r0 is $r0"

        # t -> 0.0 is just a dummy function, not to be used
        Model(f!, t -> 0.0, x0, dx0, dvars, r0)
    end
end

# FOR SOME REASON, IT TELLS ME THIS IS UNSTABLE WHEN i TRY TO SOLVE IT.
# TODO: Delete me
function partial_debugging(Φ::Float64, u::Function, w::Function, θ::Array{Float64, 1}, y::Function, λ::Function, dλ::Function, T::Float64)::Model
    function f!(res, dx, x, θ, t)
        res[1]    = dx[1] - 2*sin(t)*cos(t)*cos(t-0.3) + (sin(t)^2)*sin(t-0.3)
        nothing
    end

    x0   = [0.0]
    dx0  = [0.0]

    dvars = [true]

    r0 = zeros(length(x0))
    f!(r0, dx0, x0, [], 0.0)

    # t -> 0.0 is just a dummy function, not to be used
    Model(f!, t -> 0.0, x0, dx0, dvars, r0)
end

function model_mohamed(Φ::Float64, u::Function, w::Function, θ::Array{Float64,1})
    # NOTE: Φ not used, just passed to have consistent interface
    function f!(res, dx, x, p, t)
        wt = w(t)[1]
        ut = u(t)[1]
        # Dynamic Equations
        # @info "sum: $(ut+wt), θ: $θ, x1: $(x[1])"   # TODO: Sum is not correct here, gotta figure out why!
        res[1] = dx[1] + θ[1]*x[1] + ut + wt
        res[2] = x[2] + 2/( (θ[1]*x[1]+ut+wt)^2 + 1 )
        nothing
    end

    # Finding consistent initial conditions
    # Initial values, the pendulum starts at rest
    u0 = u(0.0)[1]
    w0 = w(0.0)[1]
    x10 = -(u0+w0)/θ[1]
    x0 = [x10, -2/( (θ[1]*x10+u0+w0)^2 + 1 )]
    dx0 = [0.0, 0.0]

    dvars = [true, false]

    r0 = zeros(length(x0))
    f!(r0, dx0, x0, [], 0.0)

    # t -> 0.0 is just a dummy function, not to be used
    Model(f!, t -> 0.0, x0, dx0, dvars, r0)
end

function mohamed_sens(Φ::Float64, u::Function, w::Function, p::Array{Float64,1})
    θ = p[1]
    # NOTE: Φ not used, just passed to have consistent interface
    zeta(x, dx, t) = (θ*x[1] + u(t)[1] + w(t)[1])^2 + 1
    dzeta_dx1(x, dx, t) = 2θ*(θ*x[1]+u(t)[1]+w(t)[1])
    dzeta_dθ(x, dx, t)  = 2x[1]*(θ*x[1]+u(t)[1]+w(t)[1])
    # zeta(t) = (θ*x(t)[1] + u(t)[1] + w(t)[1])^2 + 1
    # dzeta_dx1(t) = 2θ*(θ*x(t)[1]+u(t)[1]+w(t)[1])
    # dzeta_dθ(t)  = 2x(t)[1]*(θ*x(t)[1]+u(t)[1]+w(t)[1])

    # @warn "This function doesn't generate a correct sensitivty model, only x0 is correct"
    function f!(res, dx, x, p, t)
        wt = w(t)[1]
        ut = u(t)[1]
        # Dynamic Equations
        res[1] = dx[1] + θ*x[1] + ut + wt
        # res[2] = x[2] + 2/( (θ*x[1]+ut+wt)^2 + 1 )
        res[2] = x[2] + 2/zeta(x, dx, t)
        # Sensitivity equations
        res[3] = dx[3] + θ*x[3] + x[1]
        res[4] = x[4] - 2*( dzeta_dx1(x,dx,t)*x[3]+dzeta_dθ(x,dx,t) )/((zeta(x,dx,t))^2)
        nothing
    end

    # Finding consistent initial conditions
    u0 = u(0.0)[1]
    w0 = w(0.0)[1]
    x10 = -(u0+w0)/θ
    xp10 = (u0+w0)/(θ^2)
    xp20 = 2*( 2*θ*(θ*x10+u0+w0)*xp10 + 2*x10*(θ*x10+u0+w0) )/(((θ*x10 + u0 + w0)^2+1)^2)
    x0 = [x10, -2/( (θ*x10+u0+w0)^2 + 1 ), xp10, xp20]
    dx0 = [0.0, 0.0, 0.0, 0.0]

    dvars = [true, false, true, false]

    r0 = zeros(length(x0))
    f!(r0, dx0, x0, [], 0.0)
    # @info "r0 is $r0"

    # t -> 0.0 is just a dummy function, not to be used
    Model(f!, t -> 0.0, x0, dx0, dvars, r0)
end

# Just writing it from scratch since a lot of time has passed and I've learned some stuff
function mohamed_adjoint_new(u::Function, w::Function, p::Array{Float64, 1}, T::Float64, x::Function, x2::Function, y::Function, dy::Function, xp0::Vector{Float64}, dx::Function, dx2::Function)
    θ = p[1]
    np = size(xp0, 2)
    nx = size(xp0,1)
    # x  = t -> sol(t)
    # x2 = t -> sol2(t)
    zeta(t) = (θ*x(t)[1] + u(t)[1] + w(t)[1])^2 + 1
    dzeta_dx1(t) = 2θ*(θ*x(t)[1]+u(t)[1]+w(t)[1])
    dzeta_dθ(t)  = 2x(t)[1]*(θ*x(t)[1]+u(t)[1]+w(t)[1])

    Fx(t)   = [θ 0.0; -(2/(zeta(t)^2))*dzeta_dx1(t) 1.0]
    Fdx(t)  = [1.0 0.0; 0.0 0.0]
    Fddx(t) = zeros(2,2)
    Fθ(t)   = [x(t)[1]; -(2/(zeta(t)^2))*dzeta_dθ(t)]
    # # FOR x2 AS OUTPUT
    # gₓ(t)   = [0  2(x2(t)[2]-y(t))/T]'
    # dgₓ(t)  = [0  2(dx2(t)[2]-dy(t))/T]'
    # FOR x1 AS OUTPUT
    gₓ(t)   = [2(x2(t)[1]-y(t))/T  0]'
    dgₓ(t)  = [2(dx2(t)[1]-dy(t))/T  0]'

    ###################### INITIALIZING ADJOINT SYSTEM ####################
    # Index 1 is differential (d), while index 2 is algebraic (a)
    λT  = zeros(2)
    dλT = zeros(2)
    λT[1] = 0.0
    # λT[2] = gₓ(T)/F_xa^a = gₓ(T)
    λT[2] = gₓ(T)[2]
    dλT[1] = -(λT')*[-θ; (2/(zeta(T)^2))*dzeta_dx1(T)] - gₓ(T)[1]
    dλT[2] = dgₓ(T)[2]
    #######################################################################

    # the residual function
    function f!(res, dz, z, θ, t)
        # Dynamic Equations
        # res[1] = (dλ')*Fdx(t) + (λ')*(Fddx(t)-Fx(t)) + gₓ(t)
        # res[2] = dβ + (λ')*Fθ
        res[1:2] = (dz[1:2]')*Fdx(t) + (z[1:2]')*(Fddx(t)-Fx(t)) + gₓ(t)'
        res[3] = dz[3] - (z[1:2]')*Fθ(t)
        # res[3] = dz[3] + (z[1:2]')*Fθ(t)    # WRONG
        nothing
    end

    function get_Gp(adj_sol::DAESolution)
        # Gp = adj_sol.u[end][end-np+1:end] + (((adj_sol.u[end][1:end-np]')*Fdx(0.0))*xp0)[:]
        Gp = adj_sol.u[end][nx+1:nx+np] .+ (((adj_sol.u[end][1:nx]')*Fdx(0.0))*xp0)
    end

    # λ0  = λT[:]
    # dλ0 = -dλT[:]
    β0  = 0.0
    dβ0 = -(λT[:]')*Fθ(T)
    # x0  = λ0
    # dx0 = dλ0
    z0  = vcat(λT[:],β0)
    dz0 = vcat(dλT[:], dβ0)

    @warn "AND IN HERE: $z0, $dz0"

    # dvars = [true, false]
    dvars = vcat([true, false], [true])

    # r0 = zeros(length(λ0))
    # f!(r0, dλ0, λ0, [], 0.0)
    r0 = zeros(length(z0))
    f!(r0, dz0, z0, [], 0.0)
    # @info "r0 is: $r0"

    # t -> 0.0 is just a dummy function, not to be used
    Model(f!, t -> 0.0, z0, dz0, dvars, r0), get_Gp
end

function mohamed_λ1(u::Function, w::Function, p::Array{Float64, 1}, T::Float64, x::Function, x2::Function, y::Function, dy::Function, xp0::Vector{Float64}, dx::Function, dx2::Function)
    θ = p[1]
    np = size(xp0, 2)
    nx = size(xp0,1)
    # x  = t -> sol(t)
    # x2 = t -> sol2(t)
    zeta(t) = (θ*x(t)[1] + u(t)[1] + w(t)[1])^2 + 1
    dzeta_dx1(t) = 2θ*(θ*x(t)[1]+u(t)[1]+w(t)[1])
    dzeta_dθ(t)  = 2x(t)[1]*(θ*x(t)[1]+u(t)[1]+w(t)[1])

    Fx(t)   = [θ 0.0; -(2/(zeta(t)^2))*dzeta_dx1(t) 1.0]
    Fdx(t)  = [1.0 0.0; 0.0 0.0]
    Fddx(t) = zeros(2,2)
    Fθ(t)   = [x(t)[1]; -(2/(zeta(t)^2))*dzeta_dθ(t)]
    # # FOR x2 AS OUTPUT
    # gₓ(t)   = [0  2(x2(t)[2]-y(t))/T]'
    # dgₓ(t)  = [0  2(dx2(t)[2]-dy(t))/T]'
    # FOR x1 AS OUTPUT
    gₓ(t)   = [2(x2(t)[1]-y(t))/T  0]'
    dgₓ(t)  = [2(dx2(t)[1]-dy(t))/T  0]'

    ###################### INITIALIZING ADJOINT SYSTEM ####################
    # Index 1 is differential (d), while index 2 is algebraic (a)
    λT  = zeros(2)
    dλT = zeros(2)
    λT[1] = 0.0
    # λT[2] = gₓ(T)/F_xa^a = gₓ(T)
    λT[2] = gₓ(T)[2]
    dλT[1] = -(λT')*[-θ; (2/(zeta(T)^2))*dzeta_dx1(T)] - gₓ(T)[1]
    dλT[2] = dgₓ(T)[2]
    #######################################################################

    # the residual function
    function f!(res, dz, z, p, t)
        res[1] = dz[1] - θ*z[1] + (2/T)*(x2(t)[1]-y(t))
        # res[1] = (dz[1:2]')*Fdx(t) + (z[1:2]')*(Fddx(t)-Fx(t)) + gₓ(t)'
        nothing
    end

    z0  = λT[1:1]#vcat(λT[:],β0)
    dz0 = dλT[1:1]#vcat(dλT[:], dβ0)

    @warn "INIT CONDS IN HERE: $z0, $dz0"

    # dvars = [true, false]
    dvars = [true]

    # r0 = zeros(length(λ0))
    # f!(r0, dλ0, λ0, [], 0.0)
    r0 = zeros(length(z0))
    f!(r0, dz0, z0, [], 0.0)
    # @info "r0 is: $r0"

    # t -> 0.0 is just a dummy function, not to be used
    Model(f!, t -> 0.0, z0, dz0, dvars, r0)
end

# function mohamed_stepbystep(Φ::Float64, u::Function, w::Function, p::Array{Float64, 1}, y::Function, λ::Function, dλ::Function, T::Float64)
#     θ = p[1]
#     # NOTE: Φ not used, just passed to have consistent interface
#     zeta(dx, x, t) = (θ*x[1] + u(t)[1] + w(t)[1])^2 + 1
#     dzeta_dx1(dx, x, t) = 2θ*(θ*x[1]+u(t)[1]+w(t)[1])
#     dzeta_dθ(dx, x, t)  = 2x[1]*(θ*x[1]+u(t)[1]+w(t)[1])
#
#     # ORG
#     θ = p[1]
#     # np = size(xp0, 2)
#     # nx = size(xp0,1)
#     zeta(x,dx,t) = (θ*x[1] + u(t)[1] + w(t)[1])^2 + 1
#     dzeta_dx1(x,dx,t) = 2θ*(θ*x[1]+u(t)[1]+w(t)[1])
#     dzeta_dθ(x,dx,t)  = 2x[1]*(θ*x[1]+u(t)[1]+w(t)[1])
#     #
#     # Fx(x,dx,t)   = [θ 0.0; -(2/(zeta(x,dx,t)^2))*dzeta_dx1(x,dx,t) 1.0]
#     # Fdx(x,dx)  = [1.0 0.0; 0.0 0.0]
#     # Fddx(x,dx) = zeros(2,2)
#     # Fp(x,dx,t)   = [x[1]; -(2/(zeta(x,dx,t)^2))*dzeta_dθ(x,dx,t)]
#     # gₓ(x,dx,t)   = [0; 2(x[2]-y(t))/T]
#     # dgₓ(x,dx,t)  = [0; 2(dx[2]-dy(t))/T]
#     # # zeta(t) = (θ*x(t)[1] + u(t)[1] + w(t)[1])^2 + 1
#     # # dzeta_dx1(t) = 2θ*(θ*x(t)[1]+u(t)[1]+w(t)[1])
#     # # dzeta_dθ(t)  = 2x(t)[1]*(θ*x(t)[1]+u(t)[1]+w(t)[1])
#     # #
#     # # Fx(t)   = [θ 0.0; -(2/(zeta(t)^2))*dzeta_dx1(t) 1.0]
#     # # Fdx(t)  = [1.0 0.0; 0.0 0.0]
#     # # Fddx(t) = zeros(2,2)
#     # # Fθ(t)   = [x(t)[1]; -(2/(zeta(t)^2))*dzeta_dθ(t)]
#     # # gₓ(t)   = [0  2(x2(t)[2]-y(t))/T]'
#     # # dgₓ(t)  = [0  2(dx2(t)[2]-dy(t))/T]'
#
#
#     # @warn "This function doesn't generate a correct sensitivty model, only x0 is correct"
#     function f!(res, dx, x, p, t)
#         wt = w(t)[1]
#         ut = u(t)[1]
#         # Dynamic Equations
#         res[1] = dx[1] + θ*x[1] + ut + wt
#         res[2] = x[2] + 2/( (θ*x[1]+ut+wt)^2 + 1 )
#         # Sensitivity equations
#         # res[3] = dx_θ[1] + p*x_θ[1] + x[1]
#         # res[4] = -2*zeta(x,dx,t)*dzeta_dx1(x,dx,t)*x_θ[1] + x_θ[2] -2 *zeta(x,dx,t)*dzeta_dθ(x,dx,t)
#         res[3] = dx[3] + θ*x[3] + x[1]
#         res[4] = -2*zeta(x,dx,t)*dzeta_dx1(x,dx,t)*x[3] + x[4] - 2*zeta(x,dx,t)*dzeta_dθ(x,dx,t)
#         nothing
#     end
#
#     # Finding consistent initial conditions
#     u0 = u(0.0)[1]
#     w0 = w(0.0)[1]
#     x10 = -(u0+w0)/θ
#     xp10 = (u0+w0)/(θ^2)
#     xp20 = 2*( 2*θ*(θ*x10+u0+w0)*xp10 + 2*x10*(θ*x10+u0+w0) )/(((θ*x10 + u0 + w0)^2+1)^2)
#     x0 = [x10, -2/( (θ*x10+u0+w0)^2 + 1 ), xp10, xp20]
#     dx0 = [0.0, 0.0, 0.0, 0.0]
#     @warn "INITIAL VALS SENS: $(x0), $(dx0)"
#
#     dvars = [true, false, true, false]
#
#     r0 = zeros(length(x0))
#     f!(r0, dx0, x0, [], 0.0)
#     # @info "r0 is $r0"
#
#     # t -> 0.0 is just a dummy function, not to be used
#     Model(f!, t -> 0.0, x0, dx0, dvars, r0)
# end

function mohamed_stepbystep(Φ::Float64, u::Function, w::Function, p::Array{Float64, 1}, y::Function, λ::Function, dλ::Function, T::Float64)
    θ = p[1]
    # np = size(xp0, 2)
    # nx = size(xp0,1)
    zeta(x,dx,t) = (θ*x[1] + u(t)[1] + w(t)[1])^2 + 1
    dzeta_dx1(x,dx,t) = 2θ*(θ*x[1]+u(t)[1]+w(t)[1])
    dzeta_dθ(x,dx,t)  = 2x[1]*(θ*x[1]+u(t)[1]+w(t)[1])

    Fx(x,dx,t)   = [θ 0.0; -(2/(zeta(x,dx,t)^2))*dzeta_dx1(x,dx,t) 1.0]
    Fdx(x,dx)  = [1.0 0.0; 0.0 0.0]
    Fddx(x,dx) = zeros(2,2)
    Fp(x,dx,t)   = [x[1]; -(2/(zeta(x,dx,t)^2))*dzeta_dθ(x,dx,t)]
    # FOR x2 AS OUTPUT
    # gₓ(x,dx,t)   = [0; 2(x[2]-y(t))/T]
    # dgₓ(x,dx,t)  = [0; 2(dx[2]-dy(t))/T]
    # FOR x1 AS OUTPUT
    gₓ(x,dx,t)   = [2(x[1]-y(t))/T; 0]
    dgₓ(x,dx,t)  = [2(dx[1]-dy(t))/T; 0]

    # the residual function
    function f!(res, dx, x, p, t)
        wt = w(t)[1]
        ut = u(t)[1]
        # NOTE: MAKE SURE TO ALSO CHANGE gₓ WHEN CHANGING THESE!
        z  = x[1]#x[2]
        zθ = x[3]#x[4]
        # Dynamic Equations
        res[1] = dx[1] + θ*x[1] + ut + wt
        res[2] = x[2] + 2/( (θ*x[1]+ut+wt)^2 + 1 )
        # Sensitivity equations
        res[3] = dx[3] + θ*x[3] + x[1]
        res[4] = x[4] - 2*( dzeta_dx1(x,dx,t)*x[3]+dzeta_dθ(x,dx,t) )/((zeta(x,dx,t))^2)

        # res[5] = dx[5] - (y(t) - y(t;θ))^2 (point 1)
        res[5] = dx[5] - (y(t)-z)^2
        # res[6] = dx[6] - 2(y(t;θ)-y(t))*y_θ(t) (point 2)
        res[6] = dx[6] - 2(z-y(t))*zθ

        # res[7] = dx[7] - 2(y(t;θ)-y(t))*y_θ(t) - (λ')*(Fx(t)*xθ + Fdx(t)*dxθ + Fp(t)) (point 3)
        res[7] = dx[7] - 2(z-y(t))*zθ + (λ(t)')*(Fx(x, dx, t)*x[3:4] + Fdx(x, dx)*dx[3:4] + Fp(x, dx, t))

        # res[8] = dx[8] + (λ')*Fp(x, dx) + ( (λ')*(Fx(x,dx)-Fddx(x,dx)) - (dλ')*Fdx(x,dx) - gx(x,dx,t)' )*xθ (point 4)
        res[8] = dx[8] + (λ(t)')*Fp(x, dx, t) + ( (λ(t)')*(Fx(x,dx,t)-Fddx(x,dx)) - (dλ(t)')*Fdx(x,dx) - gₓ(x,dx,t)' )*x[3:4]

        # res[9] = dx[9] + (λ')*Fp(x, dx) (point 5)
        res[9] = dx[9] + (λ(t)')*Fp(x, dx, t)       # NOTE: This one is the incorrect one!
        # (bonus point) i.e. remainder
        # res[10] = dx[10]  + ( (λ')*(Fx(x,dx)-Fddx(x,dx)) - (dλ')*Fdx(x,dx) - gx(x,dx,t)' )*xθ
        res[10] = dx[10] + ( (λ(t)')*(Fx(x,dx,t)-Fddx(x,dx)) - (dλ(t)')*Fdx(x,dx) - gₓ(x,dx,t)' )*x[3:4]

        # (point 3.5)
        # res[11] = dx[11] - (gx(x,dx,t)')*xθ + (λ(t)')*( Fx(x,dx)*xθ + Fdx(x,dx)*dxθ + Fp(x,dx) )
        res[11] = dx[11] - (gₓ(x,dx,t)')*x[3:4] + (λ(t)')*( Fx(x,dx,t)*x[3:4] + Fdx(x,dx)*dx[3:4] + Fp(x,dx,t) )

        # res[12] = dx[12] - (λ(t)')*Fdx(t)*dxθ - ( (dλ(t)')*Fdx(t) + (λ(t)')*Fddx(t) )*xθ
        res[12] = dx[12] - (λ(t)')*Fdx(x,dx)*dx[3:4] - ( (dλ(t)')*Fdx(x,dx) + (λ(t)')*Fddx(x,dx) )*x[3:4]

        # !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
        # NOTE: PERHAPS NOT APPLICABLE FOR MOHMOD! !!!!!!!!!!!!!!!!!!!!!!!!!!!!
        # !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
        # Shared between 3.5 and 4
        res[13] = dx[13] - (gₓ(x,dx,t)')*x[3:4] + (λ(t)')*( Fx(x,dx,t)*x[3:4] + Fp(x,dx,t) )
        # Extra for 3.5 (for m and k only)  # NOTE: PERHAPS NOT APPLICABLE FOR MOHMOD!
        res[14] = dx[14] + (λ(t)')*Fdx(x,dx)*dx[3:4]
        # Extra for 4 (for m and k only)    # NOTE: PERHAPS NOT APPLICABLE FOR MOHMOD!
        res[15] = dx[15] - ( (dλ(t)')*Fdx(x,dx) + (λ(t)')*Fddx(x,dx) )*x[3:4]

        # NOTE: LoL, these are just negated versions of the above, clearly not very equal!
        # For replacement using partial integration
        # LHS
        res[16] = dx[16] - (λ(t)')*Fdx(x,dx)*dx[3:4]
        # RHS (without term, assumed zero, which is the case for m and correct λ)
        res[17] = dx[17] + ((dλ(t)')*Fdx(x,dx) + (λ(t)')*Fddx(x,dx) )*x[3:4]

        # FRIDAY BONUS!
        # Remainder from λ-equation (2D in this case)
        res[18:19] = dx[18:19] + ((λ(t)')*(Fx(x,dx,t)-Fddx(x,dx)) - (dλ(t)')*Fdx(x,dx) - gₓ(x,dx,t)')[:]
        res[20]    = dx[20] + dλ(t)[1] - θ*λ(t)[1] + gₓ(x,dx,t)[1]
        nothing
    end

    u0 = u(0.0)[1]
    w0 = w(0.0)[1]
    x10 = -(u0+w0)/θ
    x20 = -2/( (θ*x10+u0+w0)^2 + 1 )
    dx10  = 0.0
    dx20  = 0.0
    xp10 = (u0+w0)/(θ^2)
    xp20 = 2*( 2*θ*(θ*x10+u0+w0)*dx10 + 2*x10*(θ*x10+u0+w0) )/(((θ*x10 + u0 + w0)^2+1)^2)
    x0   = [x10, x20]
    xp0  = [xp10, xp20]
    dxp10 = 0.0
    dxp20 = 0.0
    dx0   = [dx10, dx20]
    dxp0  = [dxp10, dxp20]

    G0    = 0.0
    dG0   = (y(0.0)-x20)^2
    Gp0   = 0.0
    dGp0  = 2(x20-y(0.0))*xp0[2]
    Gp02  = 0.0
    dGp02 = dGp0 - (λ(0.)')*(Fx(x0, dx0, 0.)*xp0 + Fdx(x0, dx0)*dxp0 + Fp(x0, dx0, 0.))
    Gp03  = 0.0
    # # dGp03 = -(λ(0.)')*Fp(pend0, dpend0) - ( (λ(0.)')*(Fx(pend0,dpend0)-Fddx(pend0,dpend0)) -(dλ(0.)')*Fdx(pend0,dpend0) - gₓ(pend0,dpend0,0.0)' )*s0
    dGp03 = -(λ(0.)')*Fp(x0, dx0,0.) - first( ( (λ(0.)')*(Fx(x0,dx0,0.)-Fddx(x0,dx0)) -(dλ(0.)')*Fdx(x0,dx0) - gₓ(x0,dx0,0.0)' )*xp0 )
    Gp04  = 0.0
    dGp04 = -(λ(0.)')*Fp(x0, dx0, 0.)
    Gp04b  = 0.0
    dGp04b = - ( (λ(0.)')*(Fx(x0,dx0,0.)-Fddx(x0,dx0)) -(dλ(0.)')*Fdx(x0,dx0) - gₓ(x0,dx0,0.0)' )*xp0
    Gp035  = 0.0
    dGp035 = (gₓ(x0,dx0,0.0)')*xp0 - (λ(0.0)')*( Fx(x0,dx0,0.)*xp0 + Fdx(x0,dx0)*dxp0 + Fp(x0,dx0,0.) )
    partial    = 0.0
    dpartial   = (λ(0.0)')*Fdx(x0,dx0)*dxp0 + ( (dλ(0.0)')*Fdx(x0,dx0) + (λ(0.0)')*Fddx(x0,dx0) )*xp0

    # NOTE: ONLY MAKE SENSE FOR k AND m SINCE ONLY THEN APPROPRIATE QUANTITIES ARE ZERO
    common   = 0.0
    dcommon  = (gₓ(x0,dx0,0.0)')*xp0 - (λ(0.)')*( Fx(x0,dx0,0.)*xp0 + Fp(x0,dx0,0.) )
    extra35  = 0.0
    dextra35 = -(λ(0.)')*Fdx(x0,dx0)*dxp0
    extra4   = 0.0
    dextra4  = ( (dλ(0.0)')*Fdx(x0,dx0) + (λ(0.0)')*Fddx(x0,dx0) )*xp0

    partrepL  = 0.0
    dpartrepL = (λ(0.)')*Fdx(x0,dx0)*dxp0
    partrepR  = 0.0
    dpartrepR = -((dλ(0.)')*Fdx(x0,dx0) + (λ(0.)')*Fddx(x0,dx0) )*xp0

    # FRIDAY DEBUG
    vecres1_0 = 0.0
    vecres2_0 = 0.0
    temp = -(λ(0.)')*(Fx(x0,dx0,0.0)-Fddx(x0,dx0)) + (dλ(0.0)')*Fdx(x0,dx0) + gₓ(x0,dx0,0.)'
    dvecres1_0 = temp[1]
    dvecres2_0 = temp[2]

    res1_0     = 0.0
    dres1_0    = -dλ(0.)[1] + θ*λ(0.)[1] - gₓ(x0,dx0,0.)[1]

    x0 = [x10, x20, xp10, xp20, G0, Gp0, Gp02, Gp03, Gp04, Gp04b, Gp035, partial, common, extra35, extra4, partrepL, partrepR, vecres1_0, vecres2_0, res1_0]
    dx0 = [dx10, dx20, dxp10, dxp20, dG0, dGp0, dGp02, dGp03, dGp04, dGp04b, dGp035, dpartial, dcommon, dextra35, dextra4, dpartrepL, dpartrepR, dvecres1_0, dvecres2_0, dres1_0]
    dvars = vcat([true, false, true, false], fill(true, 16))
    # dvars = vcat([true, false, true, false], fill(true, 2))
    # x0 =  [x10, x20, xp10, xp20]
    # dx0 = [dx10, dx20, dxp10, dxp20]
    # dvars = [true, false, true, false]

    # r0 = zeros(length(λ0))
    # f!(r0, dλ0, λ0, [], 0.0)
    r0 = zeros(length(x0))
    f!(r0, dx0, x0, [], 0.0)
    # @info "r0 is: $r0"

    # t -> 0.0 is just a dummy function, not to be used
    Model(f!, t -> 0.0, x0, dx0, dvars, r0)
end

function mohamed_an_stepbystep(u__::Function, w__::Function, θs::Array{Float64, 1}, x_::Function, dx_::Function, xθ_::Function, dxθ_::Function, y_::Function, λ_::Function, dλ_::Function, T::Float64)
    θ = θs[1]

    u_(t) = u__(t)[1]
    w_(t) = w__(t)[1]
    # TODO: Evaluate if we need any of these!
    zeta(x,dx,t) = (θ*x[1] + u_(t) + w_(t))^2 + 1
    dzeta_dx1(x,dx,t) = 2θ*(θ*x[1]+u_(t)+w_(t))
    dzeta_dθ(x,dx,t)  = 2x[1]*(θ*x[1]+u_(t)+w_(t))

    Fx(x,dx,t)   = [θ 0.0; -(2/(zeta(x,dx,t)^2))*dzeta_dx1(x,dx,t) 1.0]
    Fdx(x,dx)  = [1.0 0.0; 0.0 0.0]
    Fddx(x,dx) = zeros(2,2)
    Fp(x,dx,t)   = [x[1]; -(2/(zeta(x,dx,t)^2))*dzeta_dθ(x,dx,t);;]
    # FOR x2 AS OUTPUT
    # gₓ(x,dx,t)   = [0; 2(x[2]-y(t))/T]
    # # dgₓ(x,dx,t)  = [0; 2(dx[2]-dy(t))/T]
    # FOR x1 AS OUTPUT
    gₓ(x,dx,t)   = [2(x[1]-y_(t))/T; 0;;]
    # dgₓ(x,dx,t)  = [2(dx[1]-dy_(t))/T; 0]

    # the residual function
    function f!(res, dz, z, p, t)
        w = w_(t)
        u = u_(t)
        x = x_(t)
        dx = dx_(t)
        xθ = xθ_(t)
        dxθ = dxθ_(t)
        y = y_(t)
        λ = λ_(t)
        dλ = dλ_(t)
        # NOTE: USING x1 AS OUTPUT CURRENTLY
        # (point 1), cost
        res[1] = dz[1] - (1/T)*(y-x[1])^2
        # (point 2), cost sensitivity

        res[2] = dz[2] - first((gₓ(x,dx,t)')*xθ)
        # (point 3), cost sensitivity with λ-term that's always zero
        res[3] = dz[3] - first((gₓ(x,dx,t)')*xθ + (λ')*(Fx(x,dx,t)*xθ + Fdx(x,dx)*dxθ + Fp(x,dx,t)))
        # (point 4), cost sensitivity after partial integration
        res[4] = dz[4] + first((λ')*Fp(x,dx,t) + ( (λ')*(Fx(x,dx,t)-Fddx(x,dx)) - (dλ')*Fdx(x,dx)-gₓ(x,dx,t)' )*xθ)   # NOTE: Needs term added to it to be correct
        # (point 5), cost sensitivity, assuming that λ satisfied adjoint equation
        res[5] = dz[5] + first((λ')*Fp(x,dx,t))  # NOTE: Needs term added to it to be correct
        nothing
    end

    w = w_(0.)
    u = u_(0.)
    x = x_(0.)
    dx = dx_(0.)
    xθ = xθ_(0.)
    dxθ = dxθ_(0.)
    y = y_(0.)
    λ = λ_(0.)
    dλ = dλ_(0.)
    z0  = zeros(5)
    dz0_1 = (1/T)*(y-x[1])^2
    dz0_2 = first((gₓ(x,dx,0.)')*xθ)
    dz0_3 = first((gₓ(x,dx,0.)')*xθ - (λ')*(Fx(x,dx,0.)*xθ + Fdx(x,dx)*dxθ + Fp(x,dx,0.)))
    dz0_4 = first(-(λ')*Fp(x,dx,0.) - ( (λ')*(Fx(x,dx,0.)-Fddx(x,dx)) - (dλ')*Fdx(x,dx) - gₓ(x,dx,0.)' )*xθ)
    dz0_5 = first(-(λ')*Fp(x,dx,0.))
    dz0 = [dz0_1, dz0_2, dz0_3, dz0_4, dz0_5]

    dvars = fill(true, 5)
    r0 = zeros(length(z0))
    f!(r0, dz0, z0, [], 0.0)
    # @info "r0 is: $r0"

    # t -> 0.0 is just a dummy function, not to be used
    Model(f!, t -> 0.0, z0, dz0, dvars, r0)
end

# TODO: Make into ODE instead?
function mohamed_analytical(u::Function, w::Function, θ::Array{Float64,1})
    # NOTE: Φ not used, just passed to have consistent interface
    function f!(res, dx, x, p, t)
        wt = w(t)[1]
        ut = u(t)[1]
        # Dynamic Equations
        res[1] = dx[1] - exp(θ[1]*t)*(ut+wt)
        res[2] = dx[2] - t*exp(θ[1]*t)*(ut+wt)
        nothing
    end

    # Finding consistent initial conditions
    # Initial values, the pendulum starts at rest
    u0 = u(0.0)[1]
    w0 = w(0.0)[1]
    x0 = zeros(2)
    dx0 = [u0+w0, 0.0]

    dvars = [true, true]

    r0 = zeros(length(x0))
    f!(r0, dx0, x0, [], 0.0)

    # t -> 0.0 is just a dummy function, not to be used
    Model(f!, t -> 0.0, x0, dx0, dvars, r0)
end

# DEBUG, TODO: Delete
function trivial_model(Φ::Float64, u::Function, w_comp::Function, θ::Array{Float64, 1})::Model
    let m = θ[1], L = θ[2], g = θ[3], k = θ[4]
        # NOTE: In this function w_comp is edxected to have two elements: the
        # first should just be the disturbance w, and the second the sensitivity
        # of the disturbance w to the disturbance model parameter

        # the residual function
        function f!(res, dx, x, θ, t)
            wt = w_comp(t)
            ut = u(t)
            # Dynamic equation
            res[1] = dx[1] + x[1] - wt[1]^2
            # First sensitivity equation
            res[2] = dx[2] + x[2] - 2*wt[1]*wt[2]
            # Second sensitivity equation
            res[3] = dx[3] + x[3] - 2*wt[1]*wt[3]
            nothing
        end

        # Finding consistent initial conditions
        # Initial values, the pendulum starts at rest
        u0 = u(0.0)[1]
        w0 = w_comp(0.0)[1]
        x0  = [0.0, 0.0, 0.0]
        dx0 = [w0^2, 0.0, 0.0]

        dvars = [true, true, true]

        r0 = zeros(length(x0))
        f!(r0, dx0, x0, [], 0.0)

        # t -> 0.0 is just a dummy function, not to be used
        Model(f!, t -> 0.0, x0, dx0, dvars, r0)
    end
end

function fast_heat_transfer_reactor(V0::Float64, T0::Float64, u::Function, w::Function, θ::Array{Float64, 1})::Model
    let k0 = θ[1], k1 = θ[2], k2 = θ[3], k3 = θ[4], k4 = θ[5]

        # the residual function
        function f!(res, dx, x, θ, t)
            wt = w(t)
            ut = u(t)
            # Dynamic Equations
            res[1] = dx[1] - ut[1] + ut[4]
            res[2] = dx[2] - ut[1]*(ut[2]-x[2])/x[1] + k0*TEMPECSPRESSION(-k1/x[4])x[2]
            res[3] = dx[3] + ut[1]x[3]/x[1] - k0*TEMPECSPRESSION(-k1/x[4])x[2]
            res[4] = x[6] - k3*x[5]/x[1] - ut[1]*(ut[3]-x[4])/x[1] + k0*k2*TEMPECSPRESSION(-k1/x[4])*x[2]
            res[5] = dx[4] + k3*x[5]/k4 - ut[5]*(ut[6]-x[4])/k4 - wt[1]^2
            res[6] = x[6] - dx[4]
            nothing
        end

        # Finding consistent initial conditions
        # Initial values, with starting volume V0 and starting temperature T0
        u0 = u(0.0)
        w0 = w(0.0)
        x5_0 = (V0*u0[5]*(u0[6]-T0) - k4*u0[1]*(u0[3]-T0))/((V0+k4)k3)
        dx4_0 = -k3*x5_0/k4 + u0[5]*(u0[6]-T0)/k4
        dx6_0 = -k3*x5_0/V0^2 - u0[1]*dx4_0/V0 - u0[1]*(u0[3]-T0)/V0^2 + k0*k2*u0[1]*u0[2]TEMPECSPRESSION(-k1/T0)/V0

        x0 = [V0; 0.; 0.; T0; x5_0; dx4_0]
        dx0 = [u0[1]-u0[4]; u0[1]u0[2]/V0; 0.; dx4_0; 0.; dx6_0]

        dvars = vcat(fill(true, 4), [false, false])

        r0 = zeros(length(x0))
        f!(r0, dx0, x0, [], 0.0)

        # t -> 0.0 is just a dummy function, not to be used
        Model(f!, t -> 0.0, x0, dx0, dvars, r0)
    end
end

function pendulum_ode(Φ::Float64, u::Function, w::Function, θ::Array{Float64, 1})::Model_ode
    let m = θ[1], L = θ[2], g = θ[3], k = θ[4]
        # Similarly to the DAE-implementation, we don't use the ability of passing
        # the parameters to the function f. Instead, we create a new problem for
        # new values of the parameters by calling pendulum_ode() again when
        # f(x,p,t) = [x[2]; (m^2)*(L^2)*g*cos(x[1])-k*(L^2)*x[2]^2 + u(t)[1] + w(t)[1]^2]
        f(x,p,t) = [x[2]; (m^2)*(L^2)*g*cos(x[1])-k*(L^2)*x[2]*abs(x[2]) + u(t)[1] + w(t)[1]^2]
        x0 = [Φ; 0.0]
        return Model_ode(f, x0)
    end
end

function pendulum_sensitivity_ode(Φ::Float64, u::Function, w::Function, θ::Array{Float64, 1})::Model_ode
    let m = θ[1], L = θ[2], g = θ[3], k = θ[4]
        # Similarly to the DAE-implementation, we don't use the ability of passing
        # the parameters to the function f. Instead, we create a new problem for
        # new values of the parameters by calling pendulum_ode() again when
        # parameters change. Therefore, the argument p is unused.
        f(x,p,t) = [x[2];
                    (m^2)*(L^2)*g*cos(x[1])-k*(L^2)*x[2]*abs(x[2]) + u(t)[1] + w(t)[1]^2
                    x[4]
                    -(m^2)*(L^2)*g*sin(x[1])*x[3]-2*k*(L^2)*abs(x[2])*x[4] - (L^2)*x[2]*abs(x[2])]
        x0 = [Φ; 0.0; 0.0; 0.0]
        return Model_ode(f, x0)
    end
end

function simple_model_sens(u::Function, w::Function, θ::Array{Float64, 1})::Model

    function f(out, dx, x, p, t)
        wt = w(t)
        ut = u(t)
        out[1] = dx[1] + θ[1]*x[1] - ut[1] - wt[1]      # Equation 1
        out[2] = x[2] - x[1]^2                          # Equation 2     x
        out[3] = θ[1]*x[3] + dx[3] + x[1]               # Sensitivity of x₁
        out[4] = -2x[1]*x[3] + x[4]                     # Sensitivity of x₂
    end

    x₀  = [0.0, 0.0, 0.0, 0.0]
    dx₀ = [0.0, 0.0, 0.0, 0.0]
    dvars = [true, false, true, false]

    Model(f, x -> x, x₀, dx₀, dvars, [0.0])
end

function simple_model(u::Function, w::Function, θ::Array{Float64, 1})::Model

    function f(out, dx, x, p, t)
        wt = w(t)
        ut = u(t)
        out[1] = dx[1] + θ[1]*x[1] - ut[1] - wt[1]      # Equation 1
        out[2] = x[2] - x[1]^2                          # Equation 2     x                  # Sensitivity of x₂
    end

    x₀  = [0.0, 0.0]
    dx₀ = [0.0, 0.0]
    dvars = [true, false]

    Model(f, x -> x, x₀, dx₀, dvars, [0.0])
end

# T = 200.0
# m = pendulum2(pi/4, t -> 0., t -> 0.1sin(t), [0.3, 3.0, 9.81, 0.1])
# prob = DAEProblem(m.f!, m.dx0, m.x0, (0, T), [])
# sol = solve(prob, IDA(), abstol = 1e-7, reltol = 1e-7, maxiters = Int(1e10), saveat = 0:0.05:T)
# plot(sol, vars=[1, 2])

function simulation_plots(T, sols, vars; kwargs...)
  ps = [plot() for var in vars]
  np = length(ps)

  for sol in sols
    for p = 1:np
      plot!(ps[p], sol, tspan=(0.0, T), vars=[vars[p]]; kwargs...)
    end
  end

  ps
end

function problem(m::Model, N::Int, Ts::Float64)
  T = N * Ts
  # ff = DAEFunction(m.f!, jac = m.jac!)
  # ff = DAEFunction{true,true}(m.f!)
  DAEProblem(m.f!, m.dx0, m.x0, (0, T), [], differential_vars=m.dvars)
end

function problem_reverse(m::Model, N::Int, Ts::Float64)
    T = N * Ts
    # ff = DAEFunction(m.f!, jac = m.jac!)
    # ff = DAEFunction{true,true}(m.f!)
    DAEProblem(m.f!, m.dx0, m.x0, (T, 0), [], differential_vars=m.dvars)
end

function problem_ode(m::Model_ode, N::Int, Ts::Float64)
    T = N * Ts
    ODEProblem(m.f, m.x0, (0,T), [])
end

function solve(prob; kwargs...)
  DifferentialEquations.solve(prob, IDA(); kwargs...)
end

# TODO: Might be worth specifying solver instead of letting it be picked automatically
function solve_ode(prob; kwargs...)
    DifferentialEquations.solve(prob; kwargs...)
end

function apply_outputfun(h, sol)
  if sol.retcode != :Success
    throw(ErrorException("Solution retcode: $(sol.retcode)"))
  end
  # NOTE: There are alternative, more recommended, ways of accessing solution
  # than through sol.u: https://diffeq.sciml.ai/stable/basics/solution/
  # TODO: If this doesn't work when h returns scalar value instead of vector,
  # then you should separate scalar case. Then this line would be only map(h, sol.u)
  # vcat(map(h, sol.u)...)    # vcat(*...) makes it so that, if output is multivariate, it is all stacked in one big vector
  map(h, sol.u)
end

function apply_two_outputfun(h1, h2, sol)
    if sol.retcode != :Success
      throw(ErrorException("Solution retcode: $(sol.retcode)"))
    end
    # NOTE: There are alternative, more recommended, ways of accessing solution
    # than through sol.u: https://diffeq.sciml.ai/stable/basics/solution/
    map(h1, sol.u), map(h2, sol.u)
end

function apply_outputfun_mvar(h, sol)
  if sol.retcode != :Success
    throw(ErrorException("Solution retcode: $(sol.retcode)"))
  end

  out_mat = zeros(length(sol.u), length(h(sol.u[1])))
  for ind in 1:length(sol.u)
      out_mat[ind,:] = h(sol.u[ind])
  end
  # map(h, sol.u)
  return out_mat
end

function apply_two_outputfun_mvar(h1, h2, sol)
    if sol.retcode != :Success
      throw(ErrorException("Solution retcode: $(sol.retcode)"))
    end

    # Since there might be several parameters (h2 might have a multi-dimensional
    # output), the output corresponding to h2 has to be stored in a matirx.
    # The rows of the matrix correspond to different values of t, and the
    # columns the different indices of the multidimensional output
    out_mat2 = zeros(length(sol.u), length(h2(sol.u[1])))
    for ind in 1:length(sol.u)
        # out_mat1[ind,:] = h1(sol.u[ind])
        out_mat2[ind,:] = h2(sol.u[ind])
    end
    # NOTE: There are alternative, more recommended, ways of accessing solution
    # than through sol.u: https://diffeq.sciml.ai/stable/basics/solution/
    # map(h1, sol.u), map(h2, sol.u)
    # return out_mat1, out_mat2
    # NOTE: Currently supporting several parameters but only scalar output
    return map(h1, sol.u), out_mat2
end

function apply_two_outputfun_twomvar(h1, h2, sol)
    if sol.retcode != :Success
      throw(ErrorException("Solution retcode: $(sol.retcode)"))
    end

    # Since there might be several parameters (h2 might have a multi-dimensional
    # output), the output corresponding to h2 has to be stored in a matirx.
    # The rows of the matrix correspond to different values of t, and the
    # columns the different indices of the multidimensional output
    out_mat2 = zeros(length(sol.u), length(h2(sol.u[1])))
    out_mat1 = zeros(length(sol.u), length(h1(sol.u[1])))
    for ind in 1:length(sol.u)
        # out_mat1[ind,:] = h1(sol.u[ind])
        out_mat2[ind,:] = h2(sol.u[ind])
        out_mat1[ind,:] = h1(sol.u[ind])
    end
    # NOTE: There are alternative, more recommended, ways of accessing solution
    # than through sol.u: https://diffeq.sciml.ai/stable/basics/solution/
    # map(h1, sol.u), map(h2, sol.u)
    # return out_mat1, out_mat2
    # NOTE: Currently supporting several parameters but only scalar output
    return out_mat1, out_mat2
end

function solve_in_parallel(solve, is)
  M = length(is)
  p = Progress(M, 1, "Running $(M) simulations...", 50)
  y1 = solve(is[1])
  Y = zeros(length(y1), M)
  Y[:, 1] += y1
  next!(p)
  Threads.@threads for m = 2:M
      y = solve(is[m])
      Y[:, m] .+= y
      next!(p)
  end
  Y
end

function solve_adj_in_parallel(solve, is)
  M = length(is)
  p = Progress(M, 1, "Running $(M) simulations...", 50)
  Gp1 = solve(is[1])
  Gps = zeros(length(Gp1), M)
  Gps[:, 1] += Gp1
  next!(p)
  Threads.@threads for m = 2:M
      Gp = solve(is[m])
      Gps[:, m] .+= Gp
      next!(p)
  end
  Gps
end

# Handles multivariate outputs
function solve_in_parallel_multivar(solve, is)
  M = length(is)
  p = Progress(M, 1, "Running $(M) simulations...", 50)
  y1 = solve(is[1])
  ny = length(y1[1])
  Y = zeros(ny*length(y1), M)
  Y[:,1] += vcat(y1...) # Flattens the array
  next!(p)
  Threads.@threads for m = 2:M
      y = vcat(solve(is[m])...) # Flattens the array of arrays returned by solve()
      Y[:,m] .+= y[:]
      next!(p)
  end
  Y, ny
end

function solve_in_parallel_sens(solve, is)
    M = length(is)
    p = Progress(M, 1, "Running $(M) simulations...", 50)
    y1, sens1 = solve(is[1])
    Ysens = [Array{Float64,2}(undef, length(sens1), length(sens1[1])) for m=1:M]
    Y = zeros(length(y1), M)
    Y[:,1] = y1
    Ysens[1] = sens1
    next!(p)
    Threads.@threads for m = 2:M
        y, sens = solve(is[m])
        Y[:,m] += y
        Ysens[m] = sens
        next!(p)
    end
    Y, Ysens
  end

function get_sol_in_parallel(solve, is)
    M = length(is)
    p = Progress(M, 1, "Running $(M) simulations...", 50)
    solutions = Array{DAESolution, 1}(undef, M)
    # y1 = solve(is[1])
    # Y = zeros(length(y1), M)
    # Y[:, 1] += y1
    next!(p)
    Threads.@threads for m = 1:M
        solutions[m] = solve(is[m])
        next!(p)
    end
    solutions
end
# Old API

function simulate(m::Model, N::Int, Ts::Float64; kwargs...)
  let
    T = N * Ts
    saveat = 0:Ts:T

    prob = problem(m, N, Ts)

    solve(prob, saveat = saveat; kwargs...)
  end
end

function simulate_h(m::Model, N::Int, Ts::Float64, h::Function)
  sol = simulate(m, N, Ts)
  apply_outputfun(h, sol)
end

function simulate_m(mk_model::Function, N::Int, Ts::Float64)
  function f(m)
    model = mk_model(m)
    simulate(model, N, Ts)
  end
end

function simulate_h_m(
  mk_model::Function, N::Int, Ts::Float64, h::Function, ms::Array{Int, 1}
)::Array{Float64, 2}

  M = length(ms)
  # Y = hcat([[Threads.Atomic{Float64}(0.0) for i=1:(N+1)] for j=1:M]...)
  Y = zeros(N+1, M)
  p = Progress(M, 1, "Running $(M) simulations...", 50)
  Threads.@threads for m = 1:M
    model = mk_model(ms[m])
    y = simulate_h(model, N, Ts, h)
    # for n = 1:(N+1)
    # Threads.atomic_add!(Y[k, m], y[k])
    Y[:, m] .+= y
    # end
    next!(p)
  end
  # map(y -> y[], Y)
  Y
end
