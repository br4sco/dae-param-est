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

Random.seed!(1234)              # set the seed

const N = 100                   # number of steps
const M = 2                     # number of noise realizations
const K = 2                     # number of θ samples

const T0 = 0.                   # simulation start time
const T = 10.                   # simulation end time
const ΔT = (T - T0) / (N - 1)   # simulation output stepsize

const NZ = 1                    # dimension of white noise
const NU = 1                    # dimension of inputs
const Nθ = 3                    # dimension of θ

const US = rand(10, NU)         # input data (TODO: temporary implementation)
const ZS = rand(10, NZ, M)      # white noise data (TODO: temporary implementation)
const ΘS = rand(Nθ, K)          # paramter data (TODO: temporary implementation)

function interpolation(t0::Float64, tend::Float64, xs::Array{Float64, 1})
  let ts = range(t0, tend, length=length(xs))
    CubicSplineInterpolation(ts, xs)
  end
end

# Simple model of a pendulum in Cartesian coordinates on index-1 form
function pendulum(z::Array{Float64, 2}, θ::Array{Float64, 1})::Model

  # the input function is simply an interpolation over the first column in the
  # input data
  function u(t)
    interpolation(T0, T, US[:,1])(t)
  end

  # the process noise (TODO: implement appropriate linear filter model, given
  # white noise [z] and [θ])
  function w(t)
    interpolation(T0, T, z[:,1])(t)
  end

  let m = θ[1], L = θ[2], g = θ[3]
    # the residual function
    function f!(out, xp, x, p, t)
      out[1] = x[2] - xp[1]
      out[2] = x[4] - xp[3]
      out[3] = m * xp[2] + x[5] * x[1] / L + w(t)
      out[4] = m * xp[4] + x[5] * x[3] / L + g + u(t)
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
      -w(T0) / m,                 # x''
      0.,                         # y'
      -(g + u(T0)) / m,           # y''
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

# Simulates at fixed parameter θ and process noise realization given fixed z
# and returns output y
function simulate1(z::Array{Float64, 2},
                   θ::Array{Float64, 1},
                   doplot=false)::Array{Float64, 1}

  m = pendulum(z, θ)
  prob = DAEProblem(m.f!, m.xp0, m.x0, (T0, T), θ, differential_vars=m.dvars)
  sol = solve(prob, IDA())
  if doplot                     # for debugging purposes
    display(m.plottrace(sol))
  end
  map(m.h, sol(T0:ΔT:T).u)
end

# Simulates at fixed parameter θ over all realizations of z and returns output
# averaged over the noise realizations
function simulateM(θ::Array{Float64, 1})::Array{Float64, 1}
  yhat = zeros(N)
  for m = 1:M
    z = ZS[:,:,m]
    yhat += simulate1(z, θ)
  end
  yhat / M
end

# Simulates over all θ samples and returns the corresponding outputs in an N×K
# matrix
function simulateK()::Array{Float64, 2}
  Yhats = zeros(N,K)
  for k = 1:K
    θ = ΘS[:,k]
    Yhats[:,k] += simulateM(θ)
  end
  Yhats
end
