import Interpolations.CubicSplineInterpolation
import Random

using DifferentialEquations
using Sundials
using DelimitedFiles
using Plots
using ProgressMeter

include("noise_model.jl")

struct Model
  f!::Function                  # residual function
  x0::Array{Float64, 1}         # initial values of x
  xp0::Array{Float64, 1}        # initial values of x'
  dvars::Array{Bool, 1}         # bool array indicating differential variables
  ic_check::Array{Float64, 1}   # residual at t0 (DEBUG)
end

function interpolation(Ts::Float64, N::Int, xs::Array{Float64, 1})
  let ts = range(0, N * Ts, length=length(xs))
    CubicSplineInterpolation(ts, xs)
  end
end

# Simple model of a pendulum in Cartesian coordinates on stabilized index-1
# form.
function pendulum(Φ::Float64, u::Function, w::Function, θ::Array{Float64, 1})::Model
    let m = θ[1], L = θ[2], g = θ[3], k = θ[4]
      # the residual function
      function f!(out, zp, z, θ, t)
        x = z[1]
        xp = zp[1]
        vx = z[2]
        vxp = zp[2]

        y = z[3]
        yp = zp[3]
        vy = z[4]
        vyp = zp[4]

        λ = zp[5]
        μ = zp[6]

        ut = z[7]
        wt = z[8]

        out[1] = vx - xp - μ * 2x
        out[2] = vy - yp - μ * 2y
        out[3] = vxp - λ * x / m + k * vx * abs(vx) / m - (ut + wt) / m
        out[4] = vyp - λ * y / m + k * vy * abs(vy) / m + g
        out[5] = x^2 + y^2 - L^2
        out[6] = 2(x * vx + y * vy)
        out[7] = ut - u(t)
        out[8] = wt - w(t)
      end

      # initial values, the pendulum starts at rest

      u0 = u(0.0)
      w0 = w(0.0)
      x0 = L * sin(Φ)
      y0 = -L * cos(Φ)
      λ0 = m * g * cos(Φ) + (u0 + w0) * cos(Φ)
      xpp0 = x0 * λ0 / m + (u0 + w0) / m
      ypp0 = y0 * λ0 / m - g

      z0 = [
        x0,                     # x
        0.,                     # x'
        y0,                     # y
        0.,                     # y'
        0.,                     # int λ
        0.,                     # int μ
        u0,                     # u
        w0,                     # w
      ]

      zp0 = [
        0.,                     # x'
        xpp0,                   # x''
        0.,                     # y'
        ypp0,                   # y''
        λ0,                     # λ
        0.,                     # μ
        0.,                     # u'
        0.,                     # w'
      ]

      dvars = Bool[
        1,
        1,
        1,
        1,
        1,
        1,
        0,
        0,
      ]

      r0 = zeros(length(z0))
      f!(r0, zp0, z0, [], 0.0)

      Model(f!, z0, zp0, dvars, r0)
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

const abstol = 1e-7
const reltol = 1e-7
const maxiters = Int(1e10)

function simulate(m::Model, N::Int, Ts::Float64)
  let
    T = N * Ts
    saveat = 0:Ts:T

    prob = DAEProblem(
      m.f!, m.xp0, m.x0, (0, T), [], differential_vars=m.dvars)

    solve(
      prob,
      IDA(),
      abstol = abstol,
      reltol = reltol,
      maxiters = maxiters,
      saveat = saveat
    )
  end
end

function simulate_xw(xw::XW, m::Model, N::Int, Ts::Float64)
  T = N * Ts
  saveat = 0:Ts:T

  prob = DAEProblem(
    m.f!, m.xp0, m.x0, (0, T), [], differential_vars=m.dvars)

  integrator = init(
    prob,
    IDA(),
    abstol = abstol,
    reltol = reltol,
    maxiters = maxiters,
    saveat = saveat
  )

  for i in integrator
    xw.x = xw.next_x
    xw.t = integrator.t
    xw.k += 1
  end

  integrator.sol
end

# mk_w = discrete_time_noise_model_1(10000000, 10, 10.0)
# xw = XW(zeros(2), zeros(2), 0.0, 1)
# m = pendulum(pi/4, t -> 0., mk_w(xw, 1), [0.3, 3.0, 9.81, 0.1])
# sol = simulate_xw(xw, m, 100, 0.05)

function simulate_h(m::Model, N::Int, Ts::Float64, h::Function)
  sol = simulate(m, N, Ts)

  if sol.retcode != :Success
    return ones(N+1) * Inf
  end

  return map(h, sol.u)
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
  Y = hcat([[Threads.Atomic{Float64}(0.0) for i=1:(N+1)] for j=1:M]...)
  Y = zeros(N+1, M)
  p = Progress(M, 1, "Running $(M) simulations...", 50)
  @inbounds Threads.@threads for m = 1:M
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
