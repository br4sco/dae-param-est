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
  xp0::Array{Float64, 1}        # initial values of x'
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

function pendulum_multivar(Φ::Float64, u::Function, w::Function, θ::Array{Float64, 1})::Model
    let m = θ[1], L = θ[2], g = θ[3], k = θ[4]

      # the residual function
      function f!(res, zp, z, θ, t)
        wt = w(t)
        ut = u(t)
        res[1] = z[2] - zp[1] - zp[6] * 2z[1]
        res[2] = z[4] - zp[3] - zp[6] * 2z[3]
        res[3] = zp[2] - zp[5] * z[1] / m + (k / m) * z[2] * abs(z[2]) - (ut[1] + wt[1] * wt[1]) / m
        res[4] = zp[4] - zp[5] * z[3] / m + (k / m) * z[4] * abs(z[4]) + g - (wt[2]*wt[2]) / m
        res[5] = z[1]^2 + z[3]^2 - L^2
        res[6] = 2(z[1] * z[2] + z[3] * z[4])
        res[7] = z[7] - atan(z[1] / -z[3])
        nothing
      end

      function jac!(J, zp, z, p, gamma, t)
        J[1,1] = -gamma - 2zp[6]
        J[1,2] = 1
        J[1,3] = 0
        J[1,4] = 0
        J[1,5] = 0
        J[1,6] = -gamma * 2z[1]
        J[1,7] = 0

        J[2,1] = 0
        J[2,2] = 0
        J[2,3] = -gamma - 2zp[6]
        J[2,4] = 1
        J[2,5] = 0
        J[2,6] = -gamma * 2z[3]
        J[2,7] = 0

        J[3,1] = -zp[5] / m
        J[3,2] = gamma + (k / m) * (z[2] == 0 ? 0 : z[2]^2 / abs(z[2]))
        J[3,3] = 0
        J[3,4] = 0
        J[3,5] = -gamma * z[1] / m
        J[3,6] = 0
        J[3,7] = 0

        J[4,1] = 0
        J[4,2] = 0
        J[4,3] = -zp[5] / m
        J[4,4] = gamma + (k / m) * (z[4] == 0 ? 0 : z[4]^2 / abs(z[4]))
        J[4,5] = -gamma * z[3] / m
        J[4,6] = 0
        J[4,7] = 0

        J[5,1] = 2z[1]
        J[5,2] = 0
        J[5,3] = 2z[3]
        J[5,4] = 0
        J[5,5] = 0
        J[5,6] = 0
        J[5,7] = 0

        J[6,1] = 2z[2]
        J[6,2] = 2z[1]
        J[6,3] = 2z[4]
        J[6,4] = 2z[3]
        J[6,5] = 0
        J[6,6] = 0
        J[6,7] = 0

        J[7,1] = z[3] / L
        J[7,2] = 0
        J[7,3] = -z[1] / L
        J[7,4] = 0
        J[7,5] = 0
        J[7,6] = 0
        J[7,7] = 1
        nothing
      end

      # initial values, the pendulum starts at rest

      u0 = u(0.0)
      w0 = w(0.0)
      x0 = L * sin(Φ)
      y0 = -L * cos(Φ)
      λ0 = m * g * cos(Φ) + (u0[1] + w0[1]) * cos(Φ)    # NOTE: Why isn't w0 squared here? //Robert
      xpp0 = x0 * λ0 / m + (u0[1] + w0[1]) / m
      ypp0 = y0 * λ0 / m - g + w0[2] / m
      z0 = [
        x0,                     # x
        0.,                     # x'
        y0,                     # y
        0.,                     # y'
        0.,                     # int λ
        0.,                     # int μ
        atan(x0 / -y0),
      ]

      zp0 = [
        0.,                     # x'
        xpp0,                   # x''
        0.,                     # y'
        ypp0,                   # y''
        λ0,                     # λ
        0.,                     # μ
        0.,                     # u'
      ]

      dvars = Bool[
        1,
        1,
        1,
        1,
        1,
        1,
        0,
      ]

      r0 = zeros(length(z0))
      f!(r0, zp0, z0, [], 0.0)

      Model(f!, jac!, z0, zp0, dvars, r0)
    end
end

function pendulum_sensitivity(Φ::Float64, u::Function, w::Function, θ::Array{Float64, 1})::Model
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
            # Sensitivity Equations
            res[8]  = -x[11] + xp[8] + 2x[1]*xp[13] + 2x[8]*xp[6]
            res[9]  = -x[12] + xp[9] + 2x[2]*xp[13] + 2x[9]*xp[6]
            res[10] = abs(x[4])*x[4] + 2k*x[11]*abs(x[4]) - x[1]*xp[10] + m*xp[11] - x[8]*xp[3]
            res[11] = abs(x[5])*x[5] + 2k*x[12]*abs(x[5]) - x[2]*xp[10] + m*xp[12] - x[9]*xp[3]
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
            # and the corresponding replacements for sp and xp

            # res[8]  = -s[4] + sp[1] + 2x[1]*sp[6] + 2s[1]*xp[6]
            # res[9]  = -s[5] + sp[2] + 2x[2]*sp[6] + 2s[2]*xp[6]
            # res[10]  = abs(x[4])*x[4] + 2k*s[4]*abs(x[4]) - x[1]*sp[3] + m*sp[4] - s[1]*xp[3]
            # res[11] = abs(x[5])*x[5] + 2k*s[5]*abs(x[5]) - x[2]*sp[3] + m*sp[5] - s[2]*xp[3]
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
        xp0 = vcat([0., 0., dx3_0, dx4_0], zeros(10))

        dvars = vcat(fill(true, 6), [false], fill(true, 6), [false])

        r0 = zeros(length(x0))
        f!(r0, xp0, x0, [], 0.0)

        # t -> 0.0 is just a dummy function, not to be used
        Model(f!, t -> 0.0, x0, xp0, dvars, r0)
    end
end

function pendulum_sensitivity2(Φ::Float64, u::Function, w::Function, θ::Array{Float64, 1})::Model
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
            # Sensitivity equations for k
            res[8]  = -x[11] + xp[8] + 2x[1]*xp[13] + 2x[8]*xp[6]
            res[9]  = -x[12] + xp[9] + 2x[2]*xp[13] + 2x[9]*xp[6]
            res[10] = abs(x[4])*x[4] + 2k*x[11]*abs(x[4]) - x[1]*xp[10] + m*xp[11] - x[8]*xp[3]
            res[11] = abs(x[5])*x[5] + 2k*x[12]*abs(x[5]) - x[2]*xp[10] + m*xp[12] - x[9]*xp[3]
            res[12] = -2x[8]*x[1] - 2x[9]*x[2]
            res[13] = x[11]*x[1] + x[12]*x[2] + x[8]*x[4] + x[9]*x[5]
            # Sensitivity of angle of pendulum to k
            # TODO: Analytical formula says it should be x[1]^2+x[2]^2 instead of
            # L^2 (though they should be equal), is it fine to substitute L^2 here?
            res[14] = x[14] - (x[1]*x[9] - x[8]*x[2])/(L^2)
            # # Sensitivity equations for L
            res[15]  = xp[15] - x[18] + 2x[15]xp[6] + 2xp[20]x[1]
            res[16]  = xp[16] - x[19] + 2x[16]xp[6] + 2xp[20]x[2]
            res[17] = 2k*x[18]*abs(x[4]) + m*xp[18] - x[15]xp[3] - xp[17]x[1]
            res[18] = 2k*x[19]*abs(x[5]) + m*xp[19] - x[16]xp[3] - xp[17]x[2]
            res[19] = 2L -2x[15]x[1] - 2x[16]x[2]
            res[20] = x[15]x[4] + x[18]x[1] + x[16]x[5] + x[19]x[2]
            # Sensitivity of angle of pendulum to L
            # TODO: Analytical formula says it should be x[1]^2+x[2]^2 instead of
            # L^2 (though they should be equal), is it fine to substitute L^2 here?
            res[21] = x[21] - (x[1]*x[16] - x[15]*x[2])/(L^2)

            # These equations sensitivity are written on the intuitive form. To
            # obtain form correct for this funciton, do the following replacements:
            # s[1] -> x[8],  s[2] -> x[9],  s[3] -> x[10]
            # s[4] -> x[11], s[5] -> x[12], s[6] -> x[13]
            # t[1] -> x[15],  t[2] -> x[16],  t[3] -> x[17]
            # t[4] -> x[18], t[5] -> x[19], t[6] -> x[20]
            # and the corresponding replacements for sp/xp and tp/xp

            # res[8]  = -s[4] + sp[1] + 2x[1]*sp[6] + 2s[1]*xp[6]
            # res[9]  = -s[5] + sp[2] + 2x[2]*sp[6] + 2s[2]*xp[6]
            # res[10]  = abs(x[4])*x[4] + 2k*s[4]*abs(x[4]) - x[1]*sp[3] + m*sp[4] - s[1]*xp[3]
            # res[11] = abs(x[5])*x[5] + 2k*s[5]*abs(x[5]) - x[2]*sp[3] + m*sp[5] - s[2]*xp[3]
            # res[12] = -2s[1]*x[1] - 2s[2]*x[2]
            # res[13] = s[4]*x[1] + s[5]*x[2] + s[1]*x[4] + s[2]*x[5]
            # res[14] = x[14] - (x[1]*s[2] - x[2]*s[1])/(L^2)

            # MATLAB form
            # res[15]  = tp[1] - t[4] + 2t[1]xp[6] + 2tp[6]x[1]
            # res[16]  = tp[2] - t[5] + 2t[2]xp[6] + 2tp[6]x[2]
            # res[17] = 2k*t[4]*abs(x[4]) + m*tp[4] - t[1]xp[3] - tp[3]x[1]
            # res[18] = 2k*t[5]*abs(x[5]) + m*tp[5] - t[2]xp[3] - tp[3]x[2]
            # res[19] = 2L -2t[1]x[1] - 2t[2]x[2]
            # res[20] = t[1]x[4] + t[4]x[1] + t[2]x[5] + t[5]x[2]
            # res[21] = (x[1]*t[2] - x[2]*t[1])/(L^2)

            # Hand-derived form
            # res[15]  = -t[4] + tp[1] + 2x[1]*tp[6] + 2t[1]*xp[6]
            # res[16]  = -t[5] + tp[2] + 2x[2]*tp[6] + 2t[2]*xp[6]
            # res[17] = 2k*t[4]*abs(x[4]) - x[1]*tp[3] + m*tp[4] - t[1]*xp[3]
            # res[18] = 2k*t[5]*abs(x[5]) - x[2]*tp[3] + m*tp[5] - t[2]*xp[3]
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
        xp0 = vcat(pendp0, sp0, tp0)
        # x0 = vcat([x1_0, x2_0], zeros(4), [atan(x1_0 / -x2_0)], zeros(14))
        # xp0 = vcat([0., 0., dx3_0, dx4_0], zeros(17))

        dvars = vcat(fill(true, 6), [false], fill(true, 6), [false], fill(true, 6), [false])

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

            # The default template used to generate these equations was:
            # res[8]  = -s[4] + sp[1] + 2x[1]*sp[6] + 2s[1]*xp[6]
            # res[9]  = -s[5] + sp[2] + 2x[2]*sp[6] + 2s[2]*xp[6]
            # res[10]  = 2k*s[4]*abs(x[4]) - x[1]*sp[3] + m*sp[4] - s[1]*xp[3]
            # res[11] = 2k*s[5]*abs(x[5]) - x[2]*sp[3] + m*sp[5] - s[2]*xp[3]
            # res[12] = -2s[1]*x[1] - 2s[2]*x[2]
            # res[13] = s[4]*x[1] + s[5]*x[2] + s[1]*x[4] + s[2]*x[5]
            # res[14] = x[14] - (x[1]*s[2] - x[2]*s[1])/(L^2)
            # Note that these expressions don't include parameter-specific terms

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
        xp0 = vcat(pendp0, smp0, sLp0, sgp0, skp0)
        # x0 = vcat([x1_0, x2_0], zeros(4), [atan(x1_0 / -x2_0)], zeros(14))
        # xp0 = vcat([0., 0., dx3_0, dx4_0], zeros(17))

        dvars = repeat(vcat(fill(true, 6), [false]), 5)

        r0 = zeros(length(x0))
        f!(r0, xp0, x0, [], 0.0)

        # t -> 0.0 is just a dummy function, not to be used
        Model(f!, t -> 0.0, x0, xp0, dvars, r0)
    end
end

function pendulum_sensitivity_full_with_dist_sens_2(Φ::Float64, u::Function, w::Function, θ::Array{Float64, 1})::Model
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

            # 22->36, 23->37, 24->38, 25->39, 26->40, 27->41, 28->42

            # DISTURBANCE SENSITIVITIES
            res[36] = 2xp[6]x[36] - x[39] + xp[36] + 2x[1]xp[41]
            res[37] = 2xp[6]x[37] - x[40] + xp[37] + 2x[2]xp[41]
            res[38] = -xp[3]x[36] + 2k*abs(x[4])*x[39] - x[1]*xp[38] + m*xp[39] - 2*wt[1]*wt[2]
            res[39] = -xp[3]x[37] + 2k*abs(x[5])*x[40] - x[2]*xp[38] + m*xp[40]
            res[40] = -2x[1]x[37] - 2x[2]x[37]
            res[41] = x[4]x[36] + x[5]x[37] + x[1]x[39] + x[2]x[40]
            # Sensitivity of angle of pendulum to disturbance parameter
            # TODO: Analytical formula says it should be x[1]^2+x[2]^2 instead of
            # L^2 (though they should be equal), is it fine to substitute L^2 here?
            res[42] = x[42] - (x[1]*x[37] - x[2]*x[36])/(L^2)

            # 29->43, 30->44, 31->45, 32->46, 33->47, 34->48, 35->49

            # res[43] = 2xp[6]x[43] - x[46] + xp[43] + 2x[1]xp[48]
            # res[44] = 2xp[6]x[44] - x[47] + xp[44] + 2x[2]xp[47]
            # res[45] = -xp[3]x[43] + 2k*abs(x[4])*x[46] - x[1]*xp[45] + m*xp[46] - 2*wt[1]*wt[3]
            # res[46] = -xp[3]x[44] + 2k*abs(x[5])*x[47] - x[2]*xp[45] + m*xp[47]
            # res[47] = -2x[1]x[43] - 2x[2]x[44]
            # # res[26] = 2L -2x[15]x[1] - 2x[16]x[2]
            # res[48] = x[4]x[43] + x[5]x[44] + x[1]x[46] + x[2]x[47]
            # # Sensitivity of angle of pendulum to disturbance parameter
            # # TODO: Analytical formula says it should be x[1]^2+x[2]^2 instead of
            # # L^2 (though they should be equal), is it fine to substitute L^2 here?
            # res[49] = x[49] - (x[1]*x[44] - x[2]*x[43])/(L^2)

            # The default template used to generate these equations was:
            # res[8]  = -s[4] + sp[1] + 2x[1]*sp[6] + 2s[1]*xp[6]
            # res[9]  = -s[5] + sp[2] + 2x[2]*sp[6] + 2s[2]*xp[6]
            # res[10]  = 2k*s[4]*abs(x[4]) - x[1]*sp[3] + m*sp[4] - s[1]*xp[3]
            # res[11] = 2k*s[5]*abs(x[5]) - x[2]*sp[3] + m*sp[5] - s[2]*xp[3]
            # res[12] = -2s[1]*x[1] - 2s[2]*x[2]
            # res[13] = s[4]*x[1] + s[5]*x[2] + s[1]*x[4] + s[2]*x[5]
            # res[14] = x[14] - (x[1]*s[2] - x[2]*s[1])/(L^2)
            # Note that these expressions don't include parameter-specific terms

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
        xp0 = vcat(pendp0, smp0, sLp0, sgp0, skp0, zeros(7))
        # x0 = vcat([x1_0, x2_0], zeros(4), [atan(x1_0 / -x2_0)], zeros(14))
        # xp0 = vcat([0., 0., dx3_0, dx4_0], zeros(17))

        dvars = repeat(vcat(fill(true, 6), [false]), 6)

        r0 = zeros(length(x0))
        f!(r0, xp0, x0, [], 0.0)

        # t -> 0.0 is just a dummy function, not to be used
        Model(f!, t -> 0.0, x0, xp0, dvars, r0)
    end
end

function pendulum_sensitivity_sans_g(Φ::Float64, u::Function, w::Function, θ::Array{Float64, 1})::Model
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

            # Sensitivity with respect to k
            res[22]  = -x[25] + xp[22] + 2x[1]*xp[27] + 2x[22]*xp[6]
            res[23]  = -x[26] + xp[23] + 2x[2]*xp[27] + 2x[23]*xp[6]
            res[24] = 2k*x[25]*abs(x[4]) - x[1]*xp[24] + m*xp[25] - x[22]*xp[3] + abs(x[4])x[4]
            res[25] = 2k*x[26]*abs(x[5]) - x[2]*xp[24] + m*xp[26] - x[23]*xp[3] + abs(x[5])x[5]
            res[26] = -2x[22]*x[1] - 2x[23]*x[2]
            res[27] = x[25]*x[1] + x[26]*x[2] + x[22]*x[4] + x[23]*x[5]
            res[28] = x[28] - (x[1]*x[23] - x[2]*x[22])/(L^2)

            # The default template used to generate these equations was:
            # res[8]  = -s[4] + sp[1] + 2x[1]*sp[6] + 2s[1]*xp[6]
            # res[9]  = -s[5] + sp[2] + 2x[2]*sp[6] + 2s[2]*xp[6]
            # res[10]  = 2k*s[4]*abs(x[4]) - x[1]*sp[3] + m*sp[4] - s[1]*xp[3]
            # res[11] = 2k*s[5]*abs(x[5]) - x[2]*sp[3] + m*sp[5] - s[2]*xp[3]
            # res[12] = -2s[1]*x[1] - 2s[2]*x[2]
            # res[13] = s[4]*x[1] + s[5]*x[2] + s[1]*x[4] + s[2]*x[5]
            # res[14] = x[14] - (x[1]*s[2] - x[2]*s[1])/(L^2)
            # Note that these expressions don't include parameter-specific terms

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
        xp0 = vcat(pendp0, smp0, sLp0, skp0)
        # x0 = vcat([x1_0, x2_0], zeros(4), [atan(x1_0 / -x2_0)], zeros(14))
        # xp0 = vcat([0., 0., dx3_0, dx4_0], zeros(17))

        dvars = repeat(vcat(fill(true, 6), [false]), 4)

        r0 = zeros(length(x0))
        f!(r0, xp0, x0, [], 0.0)

        # t -> 0.0 is just a dummy function, not to be used
        Model(f!, t -> 0.0, x0, xp0, dvars, r0)
    end
end

function pendulum_sensitivity2_with_dist_sens_c(Φ::Float64, u::Function, w_comp::Function, θ::Array{Float64, 1})::Model
    let m = θ[1], L = θ[2], g = θ[3], k = θ[4]
        # NOTE: In this function w_comp is expected to have two elements: the
        # first should just be the disturbance w, and the second the sensitivity
        # of the disturbance w to the disturbance model parameter

        # the residual function
        function f!(res, xp, x, θ, t)
            wt = w_comp(t)
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
            # Sensitivity equations for k
            res[8]  = -x[11] + xp[8] + 2x[1]*xp[13] + 2x[8]*xp[6]
            res[9]  = -x[12] + xp[9] + 2x[2]*xp[13] + 2x[9]*xp[6]
            res[10] = abs(x[4])*x[4] + 2k*x[11]*abs(x[4]) - x[1]*xp[10] + m*xp[11] - x[8]*xp[3]
            res[11] = abs(x[5])*x[5] + 2k*x[12]*abs(x[5]) - x[2]*xp[10] + m*xp[12] - x[9]*xp[3]
            res[12] = -2x[8]*x[1] - 2x[9]*x[2]
            res[13] = x[11]*x[1] + x[12]*x[2] + x[8]*x[4] + x[9]*x[5]
            # Sensitivity of angle of pendulum to k
            # TODO: Analytical formula says it should be x[1]^2+x[2]^2 instead of
            # L^2 (though they should be equal), is it fine to substitute L^2 here?
            res[14] = x[14] - (x[1]*x[9] - x[8]*x[2])/(L^2)
            # # Sensitivity equations for L
            res[15]  = xp[15] - x[18] + 2x[15]xp[6] + 2xp[20]x[1]
            res[16]  = xp[16] - x[19] + 2x[16]xp[6] + 2xp[20]x[2]
            res[17] = 2k*x[18]*abs(x[4]) + m*xp[18] - x[15]xp[3] - xp[17]x[1]
            res[18] = 2k*x[19]*abs(x[5]) + m*xp[19] - x[16]xp[3] - xp[17]x[2]
            res[19] = 2L -2x[15]x[1] - 2x[16]x[2]
            res[20] = x[15]x[4] + x[18]x[1] + x[16]x[5] + x[19]x[2]
            # Sensitivity of angle of pendulum to L
            # TODO: Analytical formula says it should be x[1]^2+x[2]^2 instead of
            # L^2 (though they should be equal), is it fine to substitute L^2 here?
            res[21] = x[21] - (x[1]*x[16] - x[15]*x[2])/(L^2)

            res[22] = 2xp[6]x[22] - x[25] + xp[22] + 2x[1]xp[27]
            res[23] = 2xp[6]x[23] - x[26] + xp[23] + 2x[2]xp[27]
            res[24] = -xp[3]x[22] + 2k*abs(x[4])*x[25] - x[1]*xp[24] + m*xp[25] - 2*wt[1]*wt[2]
            res[25] = -xp[3]x[23] + 2k*abs(x[5])*x[26] - x[2]*xp[24] + m*xp[26]
            res[26] = -2x[1]x[22] - 2x[2]x[23]
            # res[26] = 2L -2x[15]x[1] - 2x[16]x[2]
            res[27] = x[4]x[22] + x[5]x[23] + x[1]x[25] + x[2]x[26]
            # Sensitivity of angle of pendulum to disturbance parameter
            # TODO: Analytical formula says it should be x[1]^2+x[2]^2 instead of
            # L^2 (though they should be equal), is it fine to substitute L^2 here?
            res[28] = x[28] - (x[1]*x[23] - x[2]*x[22])/(L^2)

            # These equations sensitivity are written on the intuitive form. To
            # obtain form correct for this funciton, do the following replacements:
            # s[1] -> x[8],  s[2] -> x[9],  s[3] -> x[10]
            # s[4] -> x[11], s[5] -> x[12], s[6] -> x[13]
            # t[1] -> x[15],  t[2] -> x[16],  t[3] -> x[17]
            # t[4] -> x[18], t[5] -> x[19], t[6] -> x[20]
            # r[1] -> x[22], r[2] -> x[23], r[3] -> x[24]
            # r[4] -> x[25], r[5] -> x[26], r[6] -> x[27]
            # and the corresponding replacements for sp/xp and tp/xp

            # res[8]  = -s[4] + sp[1] + 2x[1]*sp[6] + 2s[1]*xp[6]
            # res[9]  = -s[5] + sp[2] + 2x[2]*sp[6] + 2s[2]*xp[6]
            # res[10]  = abs(x[4])*x[4] + 2k*s[4]*abs(x[4]) - x[1]*sp[3] + m*sp[4] - s[1]*xp[3]
            # res[11] = abs(x[5])*x[5] + 2k*s[5]*abs(x[5]) - x[2]*sp[3] + m*sp[5] - s[2]*xp[3]
            # res[12] = -2s[1]*x[1] - 2s[2]*x[2]
            # res[13] = s[4]*x[1] + s[5]*x[2] + s[1]*x[4] + s[2]*x[5]
            # res[14] = (x[1]*s[2] - x[2]*s[1])/(L^2)

            # MATLAB form
            # res[15]  = tp[1] - t[4] + 2t[1]xp[6] + 2tp[6]x[1]
            # res[16]  = tp[2] - t[5] + 2t[2]xp[6] + 2tp[6]x[2]
            # res[17] = 2k*t[4]*abs(x[4]) + m*tp[4] - t[1]xp[3] - tp[3]x[1]
            # res[18] = 2k*t[5]*abs(x[5]) + m*tp[5] - t[2]xp[3] - tp[3]x[2]
            # res[19] = 2L -2t[1]x[1] - 2t[2]x[2]
            # res[20] = t[1]x[4] + t[4]x[1] + t[2]x[5] + t[5]x[2]
            # res[21] = (x[1]*t[2] - x[2]*t[1])/(L^2)

            # Hand-derived form ( This form looks a bit stupid honestly)
            # res[15]  = -t[4] + tp[1] + 2x[1]*tp[6] + 2t[1]*xp[6]
            # res[16]  = -t[5] + tp[2] + 2x[2]*tp[6] + 2t[2]*xp[6]
            # res[17] = 2k*t[4]*abs(x[4]) - x[1]*tp[3] + m*tp[4] - t[1]*xp[3]
            # res[18] = 2k*t[5]*abs(x[5]) - x[2]*tp[3] + m*tp[5] - t[2]*xp[3]
            # res[19] = -2t[1]*x[1] - 2t[2]*x[2] + 2*L
            # res[20] = t[4]*x[1] + t[5]*x[2] + t[1]*x[4] + t[2]*x[5]
            # res[21] = (x[1]*t[2] - x[2]*t[1])/(L^2)

            # Sensitivities with respect to disturbance parameters
            # res[22] = 2xp[6]r[1] - r[4] + rp[1] + 2x[1]rp[6]
            # res[23] = 2xp[6]r[2] - r[5] + rp[2] + 2x[2]rp[6]
            # res[24] = -xp[3]r[1] + 2k*abs(x[4])*r[4] - x[1]*rp[3] + m*rp[4] + 2*w[1]wη
            # res[25] = -xp[3]r[2] + 2k*abs(x[5])*r[5] - x[2]*rp[3] + m*rp[5]
            # res[26] = -2x[1]r[1] - 2x[2]r[2]
            # res[27] = x[4]r[1] + x[5]r[2] + x[1]r[4] + x[2]r[5]
            # res[28] = (x[1]*r[2] - x[2]*r[1])/(L^2)
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

        x0  = vcat(pend0, s0, t0, r0)
        xp0 = vcat(pendp0, sp0, tp0, rp0)
        # x0 = vcat([x1_0, x2_0], zeros(4), [atan(x1_0 / -x2_0)], zeros(14))
        # xp0 = vcat([0., 0., dx3_0, dx4_0], zeros(17))

        dvars = vcat(fill(true, 6), [false], fill(true, 6), [false], fill(true, 6), [false], fill(true, 6), [false])

        r0 = zeros(length(x0))
        f!(r0, xp0, x0, [], 0.0)

        # t -> 0.0 is just a dummy function, not to be used
        Model(f!, t -> 0.0, x0, xp0, dvars, r0)
    end
end

function pendulum_sensitivity2_with_dist_sens_2(Φ::Float64, u::Function, w_comp::Function, θ::Array{Float64, 1})::Model
    let m = θ[1], L = θ[2], g = θ[3], k = θ[4]
        # NOTE: In this function w_comp is expected to have two elements: the
        # first should just be the disturbance w, and the second the sensitivity
        # of the disturbance w to the disturbance model parameter

        # the residual function
        function f!(res, xp, x, θ, t)
            wt = w_comp(t)
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
            # Sensitivity equations for k
            res[8]  = -x[11] + xp[8] + 2x[1]*xp[13] + 2x[8]*xp[6]
            res[9]  = -x[12] + xp[9] + 2x[2]*xp[13] + 2x[9]*xp[6]
            res[10] = abs(x[4])*x[4] + 2k*x[11]*abs(x[4]) - x[1]*xp[10] + m*xp[11] - x[8]*xp[3]
            res[11] = abs(x[5])*x[5] + 2k*x[12]*abs(x[5]) - x[2]*xp[10] + m*xp[12] - x[9]*xp[3]
            res[12] = -2x[8]*x[1] - 2x[9]*x[2]
            res[13] = x[11]*x[1] + x[12]*x[2] + x[8]*x[4] + x[9]*x[5]
            # Sensitivity of angle of pendulum to k
            # TODO: Analytical formula says it should be x[1]^2+x[2]^2 instead of
            # L^2 (though they should be equal), is it fine to substitute L^2 here?
            res[14] = x[14] - (x[1]*x[9] - x[8]*x[2])/(L^2)
            # # Sensitivity equations for L
            res[15]  = xp[15] - x[18] + 2x[15]xp[6] + 2xp[20]x[1]
            res[16]  = xp[16] - x[19] + 2x[16]xp[6] + 2xp[20]x[2]
            res[17] = 2k*x[18]*abs(x[4]) + m*xp[18] - x[15]xp[3] - xp[17]x[1]
            res[18] = 2k*x[19]*abs(x[5]) + m*xp[19] - x[16]xp[3] - xp[17]x[2]
            res[19] = 2L -2x[15]x[1] - 2x[16]x[2]
            res[20] = x[15]x[4] + x[18]x[1] + x[16]x[5] + x[19]x[2]
            # Sensitivity of angle of pendulum to L
            # TODO: Analytical formula says it should be x[1]^2+x[2]^2 instead of
            # L^2 (though they should be equal), is it fine to substitute L^2 here?
            res[21] = x[21] - (x[1]*x[16] - x[15]*x[2])/(L^2)

            res[22] = 2xp[6]x[22] - x[25] + xp[22] + 2x[1]xp[27]
            res[23] = 2xp[6]x[23] - x[26] + xp[23] + 2x[2]xp[27]
            res[24] = -xp[3]x[22] + 2k*abs(x[4])*x[25] - x[1]*xp[24] + m*xp[25] - 2*wt[1]*wt[2]
            res[25] = -xp[3]x[23] + 2k*abs(x[5])*x[26] - x[2]*xp[24] + m*xp[26]
            res[26] = -2x[1]x[22] - 2x[2]x[23]
            res[27] = x[4]x[22] + x[5]x[23] + x[1]x[25] + x[2]x[26]
            # Sensitivity of angle of pendulum to disturbance parameter
            # TODO: Analytical formula says it should be x[1]^2+x[2]^2 instead of
            # L^2 (though they should be equal), is it fine to substitute L^2 here?
            res[28] = x[28] - (x[1]*x[23] - x[2]*x[22])/(L^2)

            res[29] = 2xp[6]x[29] - x[32] + xp[29] + 2x[1]xp[34]
            res[30] = 2xp[6]x[30] - x[33] + xp[30] + 2x[2]xp[33]
            res[31] = -xp[3]x[29] + 2k*abs(x[4])*x[32] - x[1]*xp[31] + m*xp[32] - 2*wt[1]*wt[3]
            res[32] = -xp[3]x[30] + 2k*abs(x[5])*x[33] - x[2]*xp[31] + m*xp[33]
            res[33] = -2x[1]x[29] - 2x[2]x[30]
            # res[26] = 2L -2x[15]x[1] - 2x[16]x[2]
            res[34] = x[4]x[29] + x[5]x[30] + x[1]x[32] + x[2]x[33]
            # Sensitivity of angle of pendulum to disturbance parameter
            # TODO: Analytical formula says it should be x[1]^2+x[2]^2 instead of
            # L^2 (though they should be equal), is it fine to substitute L^2 here?
            res[35] = x[35] - (x[1]*x[30] - x[2]*x[29])/(L^2)

            # These equations sensitivity are written on the intuitive form. To
            # obtain form correct for this funciton, do the following replacements:
            # s[1] -> x[8],  s[2] -> x[9],  s[3] -> x[10]
            # s[4] -> x[11], s[5] -> x[12], s[6] -> x[13]
            # t[1] -> x[15],  t[2] -> x[16],  t[3] -> x[17]
            # t[4] -> x[18], t[5] -> x[19], t[6] -> x[20]
            # r[1] -> x[22], r[2] -> x[23], r[3] -> x[24]
            # r[4] -> x[25], r[5] -> x[26], r[6] -> x[27]
            # and the corresponding replacements for sp/xp and tp/xp

            # res[8]  = -s[4] + sp[1] + 2x[1]*sp[6] + 2s[1]*xp[6]
            # res[9]  = -s[5] + sp[2] + 2x[2]*sp[6] + 2s[2]*xp[6]
            # res[10]  = abs(x[4])*x[4] + 2k*s[4]*abs(x[4]) - x[1]*sp[3] + m*sp[4] - s[1]*xp[3]
            # res[11] = abs(x[5])*x[5] + 2k*s[5]*abs(x[5]) - x[2]*sp[3] + m*sp[5] - s[2]*xp[3]
            # res[12] = -2s[1]*x[1] - 2s[2]*x[2]
            # res[13] = s[4]*x[1] + s[5]*x[2] + s[1]*x[4] + s[2]*x[5]
            # res[14] = (x[1]*s[2] - x[2]*s[1])/(L^2)

            # MATLAB form
            # res[15]  = tp[1] - t[4] + 2t[1]xp[6] + 2tp[6]x[1]
            # res[16]  = tp[2] - t[5] + 2t[2]xp[6] + 2tp[6]x[2]
            # res[17] = 2k*t[4]*abs(x[4]) + m*tp[4] - t[1]xp[3] - tp[3]x[1]
            # res[18] = 2k*t[5]*abs(x[5]) + m*tp[5] - t[2]xp[3] - tp[3]x[2]
            # res[19] = 2L -2t[1]x[1] - 2t[2]x[2]
            # res[20] = t[1]x[4] + t[4]x[1] + t[2]x[5] + t[5]x[2]
            # res[21] = (x[1]*t[2] - x[2]*t[1])/(L^2)

            # Hand-derived form ( This form looks a bit stupid honestly)
            # res[15]  = -t[4] + tp[1] + 2x[1]*tp[6] + 2t[1]*xp[6]
            # res[16]  = -t[5] + tp[2] + 2x[2]*tp[6] + 2t[2]*xp[6]
            # res[17] = 2k*t[4]*abs(x[4]) - x[1]*tp[3] + m*tp[4] - t[1]*xp[3]
            # res[18] = 2k*t[5]*abs(x[5]) - x[2]*tp[3] + m*tp[5] - t[2]*xp[3]
            # res[19] = -2t[1]*x[1] - 2t[2]*x[2] + 2*L
            # res[20] = t[4]*x[1] + t[5]*x[2] + t[1]*x[4] + t[2]*x[5]
            # res[21] = (x[1]*t[2] - x[2]*t[1])/(L^2)

            # Sensitivities with respect to disturbance parameters
            # res[22] = 2xp[6]r[1] - r[4] + rp[1] + 2x[1]rp[6]
            # res[23] = 2xp[6]r[2] - r[5] + rp[2] + 2x[2]rp[6]
            # res[24] = -xp[3]r[1] + 2k*abs(x[4])*r[4] - x[1]*rp[3] + m*rp[4] + 2*w[1]wη
            # res[25] = -xp[3]r[2] + 2k*abs(x[5])*r[5] - x[2]*rp[3] + m*rp[5]
            # res[26] = -2x[1]r[1] - 2x[2]r[2]
            # res[27] = x[4]r[1] + x[5]r[2] + x[1]r[4] + x[2]r[5]
            # res[28] = (x[1]*r[2] - x[2]*r[1])/(L^2)
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

        x0  = vcat(pend0, s0, t0, r0, r0)
        xp0 = vcat(pendp0, sp0, tp0, rp0, rp0)
        # x0 = vcat([x1_0, x2_0], zeros(4), [atan(x1_0 / -x2_0)], zeros(14))
        # xp0 = vcat([0., 0., dx3_0, dx4_0], zeros(17))

        dvars = vcat(fill(true, 6), [false], fill(true, 6), [false], fill(true, 6), [false], fill(true, 6), [false], fill(true, 6), [false])

        r0 = zeros(length(x0))
        f!(r0, xp0, x0, [], 0.0)

        # t -> 0.0 is just a dummy function, not to be used
        Model(f!, t -> 0.0, x0, xp0, dvars, r0)
    end
end

function fast_heat_transfer_reactor(V0::Float64, T0::Float64, u::Function, w::Function, θ::Array{Float64, 1})::Model
    let k0 = θ[1], k1 = θ[2], k2 = θ[3], k3 = θ[4], k4 = θ[5]

        # the residual function
        function f!(res, xp, x, θ, t)
            wt = w(t)
            ut = u(t)
            # Dynamic Equations
            res[1] = xp[1] - ut[1] + ut[4]
            res[2] = xp[2] - ut[1]*(ut[2]-x[2])/x[1] + k0*exp(-k1/x[4])x[2]
            res[3] = xp[3] + ut[1]x[3]/x[1] - k0*exp(-k1/x[4])x[2]
            res[4] = x[6] - k3*x[5]/x[1] - ut[1]*(ut[3]-x[4])/x[1] + k0*k2*exp(-k1/x[4])*x[2]
            res[5] = xp[4] + k3*x[5]/k4 - ut[5]*(ut[6]-x[4])/k4# - 0.001*wt[1]
            res[6] = x[6] - xp[4]
            nothing
        end

        # Finding consistent initial conditions
        # Initial values, with starting volume V0 and starting temperature T0
        u0 = u(0.0)
        w0 = w(0.0)
        x5_0 = (V0*u0[5]*(u0[6]-T0) - k4*u0[1]*(u0[3]-T0))/((V0+k4)k3)
        xp4_0 = -k3*x5_0/k4 + u0[5]*(u0[6]-T0)/k4
        xp6_0 = -k3*x5_0/V0^2 - u0[1]*xp4_0/V0 - u0[1]*(u0[3]-T0)/V0^2 + k0*k2*u0[1]*u0[2]exp(-k1/T0)/V0

        x0 = [V0; 0.; 0.; T0; x5_0; xp4_0]
        xp0 = [u0[1]-u0[4]; u0[1]u0[2]/V0; 0.; xp4_0; 0.; xp6_0]

        dvars = vcat(fill(true, 4), [false, false])

        r0 = zeros(length(x0))
        f!(r0, xp0, x0, [], 0.0)

        # t -> 0.0 is just a dummy function, not to be used
        Model(f!, t -> 0.0, x0, xp0, dvars, r0)
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
# prob = DAEProblem(m.f!, m.xp0, m.x0, (0, T), [])
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
  DAEProblem(m.f!, m.xp0, m.x0, (0, T), [], differential_vars=m.dvars)
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

  map(h, sol.u)
end

function apply_two_outputfun(h1, h2, sol)
    if sol.retcode != :Success
      throw(ErrorException("Solution retcode: $(sol.retcode)"))
    end

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

    # out_mat1 = zeros(length(sol.u), length(h1(sol.u[1])))
    out_mat2 = zeros(length(sol.u), length(h2(sol.u[1])))
    for ind in 1:length(sol.u)
        # out_mat1[ind,:] = h1(sol.u[ind])
        out_mat2[ind,:] = h2(sol.u[ind])
    end
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
