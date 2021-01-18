import Interpolations.CubicSplineInterpolation
using DifferentialEquations
using Sundials
using DelimitedFiles
using Plots

struct IC
  x0::Array{Float64, 1}
  xp0::Array{Float64, 1}
  dvars::Array{Bool, 1}
end

const TSTART = 0.
const TEND = 10.

const NW = 1
const NU = 1
const M = 2

const WS = reshape(readdlm("ws.csv", ','), (:, NW, M))
const US = readdlm("us.csv", ',')
const THETAS = readdlm("theta.csv", ',')

const N = size(THETAS, 2)

function ts(n)
  TSTART:((TEND - TSTART) / n):TEND
end

function interpolation(xs::Array{Float64, 1})
  let f = CubicSplineInterpolation(ts(length(xs) - 1), xs)
    t -> f(t)
  end
end

function pendulumF(u::Function, w::Function)::DAEFunction
  function f(out, xp, x, theta, t)
    out[1] = x[2] - xp[1]
    out[2] = x[4] - xp[3]
    out[3] = theta[1] * xp[2] + x[5] * x[1] / theta[2] + w(t)
    out[4] = theta[1] * xp[4] + x[5] * x[3] / theta[2] + theta[3] + u(t)
    out[5] = x[1] * xp[2] + x[2]^2 + x[3] * xp[4] + x[4]^2
  end
  f
end

function pendulumIC(u0::Float64,w0::Float64, theta::Array{Float64, 1}, t0::Float64)::IC
  x0 = [theta[1], 0., 0., 0., 0.]
  xp0 = [0., 0., -w0 / theta[2], -(theta[3] + u0) / theta[2], 0.]
  dvars = Bool[1, 1, 1, 1, 0]
  IC(x0, xp0, dvars)
end

function pendulumPlot(sol)
  plot(sol, tspan=(TSTART + 0.0001, TEND), layout=(3,1), vars=[(0,1), (0,3), (0,5)])
end


function simulate1(u::Function, w::Function, theta::Array{Float64, 1})
  f = pendulumF(u, w)
  ic = pendulumIC(u(TSTART), w(TSTART), theta, TSTART)
  prob = DAEProblem(f, ic.x0, ic.xp0, (TSTART, TEND), theta, differential_vars=ic.dvars)
  solve(prob, IDA())
end

function simulateM(theta::Array{Float64, 1}, ms::UnitRange{Int}, doplot=false)
  u = interpolation(US[:, 1])
  plot_array = Any[]
  for m = ms
    w = interpolation(WS[:, 1, m])
    sol = simulate1(u, w, theta)
    if doplot
      push!(plot_array, pendulumPlot(sol))
    end
    print(typeof(reduce(hcat, sol(ts(10)).u)))
  end
  if doplot
    display(plot(plot_array...))
  end
end

# w = interpolation(WS[:,1,1])
# u = interpolation(US[:,1])
# theta = [1., 2., 3.]

# sol = simulate1(u, w, theta)
# plot(sol, tspan=(0.001, 10.0), layout=(3,1), vars=[(0,1),(0,3),(0,5)])
