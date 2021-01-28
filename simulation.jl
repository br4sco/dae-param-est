import Interpolations.CubicSplineInterpolation
import Random

using DifferentialEquations
using Sundials
using DelimitedFiles
using Plots

export Model, SimParams, interpolation, pendulum, simulate1

struct Model
  f!::Function                  # residual function
  x0::Array{Float64, 1}         # initial values of x
  xp0::Array{Float64, 1}        # initial values of x'
  dvars::Array{Bool, 1}         # bool array indicating differential variables
  h::Function                   # output function (single output)
  plottrace::Function           # function to plot a single simulation trace
                                # for debugging purposes
end

struct SimParams
  T0::Float64                   # simulation start time
  ΔT::Float64                   # simulation output stepsize
  T::Float64                    # simulation end time
end

function interpolation(t0::Float64, tend::Float64, xs::Array{Float64, 1})
  let ts = range(t0, tend, length=length(xs))
    CubicSplineInterpolation(ts, xs)
  end
end

# Simple model of a pendulum in Cartesian coordinates on index-1 form
function pendulum(u::Function, w::Function, θ::Array{Float64, 1})::Model

  function sw(t)
    0.7 * w(t)
  end

  let m = θ[1], L = θ[2], g = θ[3]
    # the residual function
    function f!(out, xp, x, p, t)
      out[1] = x[2] - xp[1]
      out[2] = x[4] - xp[3]
      out[3] = m * xp[2] + x[5] * x[1] / L
      out[4] = m * xp[4] + x[5] * x[3] / L + g + u(t) + sw(t)
      out[5] = x[1] * xp[2] + x[2]^2 + x[3] * xp[4] + x[4]^2
    end

    # initial values, the pendulum starts at x=L, y=0 at rest
    x0 = [
      L,                          # x
      0.,                         # x'
      0.,                         # y
      0.,                         # y'
      0.                          # λ
    ]

    xp0 = [
      0.,                         # x'
      0.,                         # x''
      0.,                         # y'
      -(g + u(T0) + sw(T0)) / m,  # y''
      0.                          # λ'
    ]

    dvars = Bool[1, 1, 1, 1, 0]

    # the output function
    function h(x)
      x[5]                      # we observe the tension λ in the pendulum arm
    end

    # plots a single simulation trace of x, y, λ
    function plottrace(sol)
      plot(sol,
           tspan=(T0 + 0.0001, T),
           layout=(3,1),
           vars=[(0,1), (0,3), (0,5)])
    end

    Model(f!, x0, xp0, dvars, h, plottrace)
  end
end

# simulates once
function simulate1(m::Model,
                   sp::SimParams,
                   θ::Array{Float64, 1},
                   doplot=false)


  prob = DAEProblem(m.f!, m.xp0, m.x0, (sp.T0, sp.T), θ, differential_vars=m.dvars)
  sol = solve(prob, IDA())

  if sol.retcode != :Success
    return ones(length(m.x0)) * Inf
  end

  if doplot                     # for debugging purposes
    display(m.plottrace(sol))
  end

  return map(m.h, sol(sp.T0:sp.ΔT:sp.T).u)
end
