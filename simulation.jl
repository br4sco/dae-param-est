import Interpolations.CubicSplineInterpolation
import Random

using DifferentialEquations
using Sundials
using DelimitedFiles
using Plots

struct Model
  f!::Function                  # residual function
  x0::Array{Float64, 1}         # initial values of x
  xp0::Array{Float64, 1}        # initial values of x'
  dvars::Array{Bool, 1}         # bool array indicating differential variables
  h::Function                   # output function (single output)
  plottrace::Function           # function to plot a single simulation trace
                                # for debugging purposes
end

struct TimeParams
  T0::Float64                   # simulation start time
  ΔT::Float64                   # simulation output stepsize
  N::Int                        # number of steps
end

function endT(tp)
  tp.N*tp.ΔT+tp.T0
end

function time_range(tp::TimeParams)
  tp.T0:tp.ΔT:endT(tp)
end

function interpolation(tp::TimeParams, xs::Array{Float64, 1})
  let ts = range(tp.T0, endT(tp), length=length(xs))
    CubicSplineInterpolation(ts, xs)
  end
end

# Simple model of a pendulum in Cartesian coordinates on index-1 form
function pendulum(u::Function, w::Function, T0::Float64)::Function
  function model(θ::Array{Float64, 1})::Model
    let m = θ[1], L = θ[2], g = θ[3], k = θ[4]
      # the residual function
      function f!(out, xp, x, θ, t)
        out[1] = x[2] - xp[1]
        out[2] = x[4] - xp[3]
        out[3] = m * xp[2] + x[5] * x[1] / L + k * x[2]
        out[4] = m * xp[4] + x[5] * x[3] / L + k * x[4] + m * g + u(t) + w(t)
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
        -(g + u(T0) + w(T0)) / m,   # y''
        0.                          # λ'
      ]

      dvars = Bool[1, 1, 1, 1, 0]

      # the output function
      function h(x)
        x[5]                    # we observe the tension λ in the pendulum arm
      end

      # plots a single simulation trace of x, y, λ
      function plottrace(sol, T)
        plot(sol,
             tspan=(T0 + 0.0001, T),
             layout=(3,1),
             vars=[(0,1), (0,3), (0,5)])
      end

      Model(f!, x0, xp0, dvars, h, plottrace)
    end
  end
end

function pendulumK(u, w, T0)
  let p = pendulum(u, w, T0)
    θ -> p([1., 1., 1., θ[1]])
  end
end

# simulates once
function simulate1(mk_model::Function,
                   tp::TimeParams,
                   θ::Array{Float64, 1},
                   doplot=false)

  m = mk_model(θ)

  saveat = time_range(tp)

  prob = DAEProblem(
    m.f!, m.xp0, m.x0, (tp.T0, endT(tp)), [], differential_vars=m.dvars, saveat=saveat)

  sol = solve(prob, IDA())

  if sol.retcode != :Success
    return ones(length(m.x0)) * Inf
  end

  if doplot                     # for debugging purposes
    display(m.plottrace(sol, T))
  end

  return map(m.h, sol.u)
end
