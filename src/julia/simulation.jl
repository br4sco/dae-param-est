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

function pendulum_sensitivity(Φ::Float64, u::Function, w::Function, θ::Array{Float64, 1})::Model
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
        pendp0 = vcat([0., 0., dx3_0, dx4_0], zeros(3))
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
        dx0 = vcat(pendp0, sp0, tp0)
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
        pendp0 = vcat([0., 0., dx3_0, dx4_0], zeros(3))
        sm0  = zeros(7)
        smp0 = vcat(zeros(3), [-dx4_0/m, -g/m, 0., 0.])
        sL0  = vcat([x1_0/L, x2_0/L], zeros(5))
        sLp0 = vcat([0.,0., -dx3_0/L], zeros(4))
        sg0  = zeros(7)
        sgp0 = vcat(zeros(4), [-1., 0., 0.])
        sk0  = zeros(7)
        skp0 = zeros(7)
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
        dx0 = vcat(pendp0, smp0, sLp0, sgp0, skp0)
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
        pendp0 = vcat([0., 0., dx3_0, dx4_0], zeros(3))
        sm0  = zeros(7)
        smp0 = vcat(zeros(3), [-dx4_0/m, -g/m, 0., 0.])
        sL0  = vcat([x1_0/L, x2_0/L], zeros(5))
        sLp0 = vcat([0.,0., -dx3_0/L], zeros(4))
        sk0  = zeros(7)
        skp0 = zeros(7)
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
        dx0 = vcat(pendp0, smp0, sLp0, skp0)
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
        pendp0 = vcat([0., 0., dx3_0, dx4_0], zeros(3))
        sm0  = zeros(7)
        smp0 = vcat(zeros(3), [-dx4_0/m, -g/m, 0., 0.])
        sL0  = vcat([x1_0/L, x2_0/L], zeros(5))
        sLp0 = vcat([0.,0., -dx3_0/L], zeros(4))
        sk0  = zeros(7)
        skp0 = zeros(7)
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
        dx0 = vcat(pendp0, smp0, sLp0, skp0, rp0, rp0)
        # x0 = vcat([x1_0, x2_0], zeros(4), [atan(x1_0 / -x2_0)], zeros(14))
        # dx0 = vcat([0., 0., dx3_0, dx4_0], zeros(17))

        dvars = repeat(vcat(fill(true, 6), [false]), 6)

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
        pendp0 = vcat([0., 0., dx3_0, dx4_0], zeros(3))
        sm0  = zeros(7)
        smp0 = vcat(zeros(3), [-dx4_0/m, -g/m, 0., 0.])
        sL0  = vcat([x1_0/L, x2_0/L], zeros(5))
        sLp0 = vcat([0.,0., -dx3_0/L], zeros(4))
        sk0  = zeros(7)
        skp0 = zeros(7)
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
        dx0 = vcat(pendp0, smp0, sLp0, skp0, rp0)
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
        pendp0 = vcat([0., 0., dx3_0, dx4_0], zeros(3))
        sm0  = zeros(7)
        smp0 = vcat(zeros(3), [-dx4_0/m, -g/m, 0., 0.])
        sL0  = vcat([x1_0/L, x2_0/L], zeros(5))
        sLp0 = vcat([0.,0., -dx3_0/L], zeros(4))
        sg0  = zeros(7)
        sgp0 = vcat(zeros(4), [-1., 0., 0.])
        sk0  = zeros(7)
        skp0 = zeros(7)
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
        dx0 = vcat(pendp0, smp0, sLp0, sgp0, skp0, zeros(7))
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
        pendp0 = vcat([0., 0., dx3_0, dx4_0], zeros(3))
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
        dx0 = vcat(pendp0, tp0, sp0, rp0)
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
        pendp0 = vcat([0., 0., dx3_0, dx4_0], zeros(3))
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
        dx0 = vcat(pendp0, tp0, sp0, rp0, rp0)
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
        pendp0 = vcat([0., 0., dx3_0, dx4_0], zeros(3))
        sk0  = zeros(7)
        skp0 = zeros(7)
        r0 = zeros(7)
        rp0 = zeros(7)

        x0  = vcat(pend0, sk0, r0)
        dx0 = vcat(pendp0, skp0, rp0)
        # x0 = vcat([x1_0, x2_0], zeros(4), [atan(x1_0 / -x2_0)], zeros(14))
        # dx0 = vcat([0., 0., dx3_0, dx4_0], zeros(17))

        dvars = repeat(vcat(fill(true, 6), [false]), 3)

        r0 = zeros(length(x0))
        f!(r0, dx0, x0, [], 0.0)

        # t -> 0.0 is just a dummy function, not to be used
        Model(f!, t -> 0.0, x0, dx0, dvars, r0)
    end
end

# global debug_counter
# debug_counter = 0

# NOTE Assumes free dynamical parameters are m, L, and k
# TODO: I think we can change all occurences of Array{Float64,1} to Vector{Float64}, pretty sure they are the same type, one just looks nicer (introduced in Julia 1.7 I think)
function pendulum_adjoint(u::Function, w::Function, θ::Array{Float64, 1}, T::Float64, sol::DAESolution, sol2::DAESolution, y::Function, xp0::Array{Float64, 2})
    let m = θ[1], L = θ[2], g = θ[3], k = θ[4]
        np = size(xp0,2)
        x  = t -> sol(t)
        x2 = t -> sol2(t)
        dx = t -> sol(t, Val{1})  # TODO: Does this give same results as sol.up???? NOTE: Nope, sol.up is Nothing, and this just uses finite differences)
        Fx = t -> [2dx(t)[6]        0               0   -1                  0           0   0
                    0           2*dx(t)[6]          0   0                   -1          0   0
                    -dx(t)[3]         0             0   2k*abs(x(t)[4])     0           0   0
                    0          -dx(t)[3]            0   0               2k*abs(x(t)[5]) 0   0
                    -2x(t)[1]     -2x(t)[2]         0   0                   0           0   0
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
        Fp = t -> [0            0           0
                   0            0           0
                   dx(t)[4]     0       abs(x(t)[4])*x(t)[4]
                   dx(t)[5]+g   0       abs(x(t)[5])*x(t)[5]
                   0            2L          0
                   0            0           0
                   0            0           0]
        gₓ = t -> [0    0    0    0    0    0    2(x(t)[7]-y(t))/T]'
        λT  = -gₓ(T)
        dλT = [-x(T)[2]*λT[7]/(L^2)   x(T)[1]*λT[7]/(L^2)  0   0   0   0   0]'

        println("Adjoint residual: $((dλT')*Fdx(T) + (λT')*(Fx(T)-Fddx(T)) + gₓ(T)')")
        println("Terminal constraint: $((λT')*(Fdx(T)))")

        # the residual function
        function f!(res, dx, x, θ, t)
            # global debug_counter
            # debug_counter += 1
            #
            # if debug_counter % 10000 == 0
            #     @info "counter: $debug_counter"
            # end
            # Dynamic Equations
            # res[1] = -dλ⋅Fdx(T-t)[:,1] + λ⋅(Fx(T-t)[:,1].-Fddx(T-t)[:,1]) + gₓ(T-t)[1]
            # # res[2] = -dλ⋅Fdx(T-t)[:,2] + λ⋅(Fx(T-t)[:,2].-Fddx(T-t)[:,2]) + gₓ(T-t)[2]
            # res[2] = λ⋅(Fx(T-t)[:,2].-Fddx(T-t)[:,2]) + gₓ(T-t)[2]
            # @info "Called this!"
            len = length(x)
            λ  = x[1:len-np]
            dλ = dx[1:len-np]
            β  = x[len-np+1:end]
            βp = dx[len-np+1:end]
            for i=1:len-np
                res[i] = -dλ⋅Fdx(T-t)[:,i] + λ⋅(Fddx(T-t)[:,i].-Fx(T-t)[:,i]) + gₓ(T-t)[i]
            end
            # Could be written simpler and less general, since np=1 for this model
            for j = 1:np
                res[len-np+j] = -βp[j] + (λ')*Fp(T-t)[:,j]
            end
            nothing
        end

        λ0  = λT[:]
        dλ0 = -dλT[:]
        x0  = vcat(λ0, zeros(np))
        dx0 = vcat(dλ0, (-(Fp(T)')*λT)[:])

        # Function returning Gp given adjoint solution
        function get_Gp(adj_sol::DAESolution)
            Gp = -adj_sol.u[end][end-np+1:end] - (((adj_sol.u[end][1:end-np]')*Fdx(0))*xp0)[:]
        end

        # dvars = [true, false]
        dvars = vcat(fill(true, 4), fill(false, 3), fill(true, np))

        r0 = zeros(length(x0))
        # f!(r0, dλ0, λ0, [], 0.0)
        f!(r0, dx0, x0, [], 0.0)

        # t -> 0.0 is just a dummy function, not to be used
        Model(f!, t -> 0.0, x0, dx0, dvars, r0), get_Gp
    end
end

function model_mohamed(Φ::Float64, u::Function, w::Function, θ::Array{Float64,1})
    # NOTE: Φ not used, just passed to have consistent interface
    function f!(res, dx, x, θ, t)
        wt = w(t)[1]
        ut = u(t)[1]
        # Dynamic Equations
        res[1] = dx[1] + p*x[1] + ut + wt
        res[2] = x[2] + 2/( (p*x[1]+ut+wt)^2 + 1 )
        nothing
    end

    p = θ[1]
    # Finding consistent initial conditions
    # Initial values, the pendulum starts at rest
    u0 = u(0.0)[1]
    w0 = w(0.0)[1]
    x10 = -(u0+w0)/p
    x0 = [x10, -2/( (p*x10+u0+w0)^2 + 1 )]
    dx0 = [0.0, 0.0]

    dvars = [true, false]

    r0 = zeros(length(x0))
    f!(r0, dx0, x0, [], 0.0)

    # t -> 0.0 is just a dummy function, not to be used
    Model(f!, t -> 0.0, x0, dx0, dvars, r0)
end

function mohamed_sens(Φ::Float64, u::Function, w::Function, θ::Array{Float64,1})
    @warn "This function doesn't generate a correct sensitivty model, only x0 is correct"
    # NOTE: Φ not used, just passed to have consistent interface
    function f!(res, dx, x, θ, t)
        nothing
    end

    p = θ[1]
    # Finding consistent initial conditions
    # Initial values, the pendulum starts at rest
    u0 = u(0.0)[1]
    w0 = w(0.0)[1]
    x10 = -(u0+w0)/p
    xp10 = (u0+w0)/(p^2)
    xp20 = 2*( 2*p*(p*x10+u0+w0)*xp10 + 2*x10*(p*x10+u0+w0) )/(((p*x10 + u0 + w0)^2+1)^2)
    x0 = [x10, -2/( (p*x10+u0+w0)^2 + 1 ), xp10, xp20]
    dx0 = [0.0, 0.0, 0.0, 0.0]

    dvars = [true, false, true, false]

    r0 = zeros(length(x0))
    f!(r0, dx0, x0, [], 0.0)

    # t -> 0.0 is just a dummy function, not to be used
    Model(f!, t -> 0.0, x0, dx0, dvars, r0)
end

function mohamed_adjoint(u::Function, w::Function, θ::Array{Float64, 1}, T::Float64, sol::DAESolution, sol2::DAESolution, y::Function, xp0::Array{Float64,2})
    p = θ[1]
    np = size(xp0, 2)
    x  = t -> sol(t)
    x2 = t -> sol2(t)
    dx = t -> sol(t, Val{1})  # TODO: Does this give same results as sol.up???? NOTE: Nope, sol.up is Nothing, and this just uses finite differences)
    zeta(t) = (p*x(t)[1] + u(t)[1] + w(t)[1])^2 + 1
    dzeta_dx1(t) = 2p*(p*x(t)[1]+u(t)[1]+w(t)[1])
    dzeta_dp(t)  = 2x(t)[1]*(p*x(t)[1]+u(t)[1]+w(t)[1])

    Fx(t)   = [p 0.0; -(2/(zeta(t)^2))*dzeta_dx1(t) 1.0]
    Fdx(t)  = [1.0 0.0; 0.0 0.0]
    Fddx(t) = zeros(2,2)
    Fp(t)   = [x(t)[1]; -(2/(zeta(t)^2))*dzeta_dp(t)]
    gₓ(t)   = [0  2(x2(t)[2]-y(t))/T]'

    # the residual function
    function f!(res, dx, x, θ, t)
        # Dynamic Equations
        # res[1] = -dλ⋅Fdx(T-t)[:,1] + λ⋅(Fx(T-t)[:,1].-Fddx(T-t)[:,1]) + gₓ(T-t)[1]
        # # res[2] = -dλ⋅Fdx(T-t)[:,2] + λ⋅(Fx(T-t)[:,2].-Fddx(T-t)[:,2]) + gₓ(T-t)[2]
        # res[2] = λ⋅(Fx(T-t)[:,2].-Fddx(T-t)[:,2]) + gₓ(T-t)[2]
        len = length(x)
        λ  = x[1:len-np]
        dλ = dx[1:len-np]
        β  = x[len-np+1:end]
        βp = dx[len-np+1:end]
        for i=1:len-np
            res[i] = -dλ⋅Fdx(T-t)[:,i] + λ⋅(Fddx(T-t)[:,i].-Fx(T-t)[:,i]) + gₓ(T-t)[i]
        end
        # Could be written simpler and less general, since np=1 for this model
        for j = 1:np
            res[len-np+j] = -βp[j] + (λ')*Fp(T-t)[:,j]
        end
        nothing
    end

    λT  = gₓ(T)
    # NOTE: Since dλ[2] doesn't appear in the adjoint system (λ[2] is a algebraic
    # variable) it seems that the terminal value for dλ[2] doesn't matter, and
    # can be set to anything, saving us the hassle of computing the derivative of
    # the true output NOTE: This should be generalizable to all systems in our
    # approach
    dλT = [-(2/(zeta(T)^2))*dzeta_dx1(T)*(gₓ(T)[2]); 0.0]

    println("Adjoint residual: $((dλT')*Fdx(T) + (λT')*(Fddx(T)-Fx(T)) + gₓ(T)')")
    println("Terminal constraint: $((λT')*(Fdx(T)))")

    λ0  = λT[:]
    dλ0 = -dλT[:]
    # x0  = λ0
    # dx0 = dλ0
    x0  = vcat(λ0, zeros(np))
    dx0 = vcat(dλ0, (-(Fp(T)')*λT)[:])

    function get_Gp(adj_sol::DAESolution)
        Gp = -adj_sol.u[end][end-np+1:end] - (((adj_sol.u[end][1:end-np]')*Fdx(0))*xp0)[:]
    end

    # dvars = [true, false]
    dvars = vcat([true, false], fill(true, np))

    # r0 = zeros(length(λ0))
    # f!(r0, dλ0, λ0, [], 0.0)
    r0 = zeros(length(x0))
    f!(r0, dx0, x0, [], 0.0)

    # t -> 0.0 is just a dummy function, not to be used
    Model(f!, t -> 0.0, x0, dx0, dvars, r0), get_Gp
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
