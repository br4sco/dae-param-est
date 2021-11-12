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

# Simple model of a pendulum in Cartesian coordinates on stabilized index-1
# form.
function pendulum(Φ::Float64, u::Function, w::Function, θ::Array{Float64, 1})::Model
    let m = θ[1], L = θ[2], g = θ[3], k = θ[4]

      # the residual function
      function f!(res, zp, z, θ, t)
        wt = w(t)
        ut = u(t)
        res[1] = z[2] - zp[1] - zp[6] * 2z[1]
        res[2] = z[4] - zp[3] - zp[6] * 2z[3]
        res[3] = zp[2] - zp[5] * z[1] / m + (k / m) * z[2] * abs(z[2]) - (ut[1] + wt[1] * wt[1]) / m
        res[4] = zp[4] - zp[5] * z[3] / m + (k / m) * z[4] * abs(z[4]) + g
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
      ypp0 = y0 * λ0 / m - g

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

# Old API

function simulate(m::Model, N::Int, Ts::Float64; kwargs...)
  let
    T = N * Ts
    saveat = 0:Ts:T

    prob = problem(m, N, Ts)

    solve(prob, saveat = saveat; kwargs...)
  end
end

# function simulate_xw(xw::XW, m::Model, N::Int, Ts::Float64)
#   T = N * Ts
#   saveat = 0:Ts:T

#   prob = DAEProblem(
#     m.f!, m.xp0, m.x0, (0, T), [], differential_vars=m.dvars)

#   integrator = init(
#     prob,
#     IDA(),
#     abstol = abstol,
#     reltol = reltol,
#     maxiters = maxiters,
#     saveat = saveat
#   )

#   for i in integrator
#     xw.x = xw.next_x
#     xw.t = integrator.t
#     xw.k += 1
#   end

#   integrator.sol
# end

# mk_w = discrete_time_noise_model_1(10000000, 10, 10.0)
# xw = XW(zeros(2), zeros(2), 0.0, 1)
# m = pendulum(pi/4, t -> 0., mk_w(xw, 1), [0.3, 3.0, 9.81, 0.1])
# sol = simulate_xw(xw, m, 100, 0.05)

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
