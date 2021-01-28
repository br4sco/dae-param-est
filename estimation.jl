include("spectral_mc.jl")
include("simulation.jl")

using ProgressMeter
using LsqFit

Random.seed!(1234)              # set the seed

struct EstProblem
  mk_model::Function
  ZS::Array{Float64, 2}
  sp::SimParams
  ys::Array{Float64, 1}
  θ::Array{Float64, 1}
  θ0::Array{Float64, 1}
end

function mk_yhat(mk_model::Function,
                 sp::SimParams,
                 ZS::Array{Float64, 2})::Function

   function f(θ::Array{Float64, 1})::Array{Float64, 1}
     N = length(ys)
     M = size(ZS, 2)
     yhats = zeros(N)
     p = Progress(M, 1, "Simulating...", 50)
     Threads.@threads for m = 1:M
       z = ZS[:, m]
       model = mk_model(θ, z)
       yhats += simulate1(model, sp, θ)
       next!(p)
     end
     yhats
   end
end

function mk_lm_model(sp::SimParams, yhat::Function)::Function
  y = identity
  θold::Array{Float64, 1} = ones(3) * Inf
  function model(t::Float64, θ::Array{Float64, 1})
    if θ != θold
      y = interpolation(sp.T0, sp.T, yhat(θ))
      θold = θ
    end
    y(t)
  end
  model
end

const PROBLEM1 = begin
  us = [0; rand(10)]

  function u(t::Float64)::Float64
    interpolation(T0, T, us)(t)
  end

  θ = [1., 1., 1.]              # true parameters
  θ0 = [0.2, 1.1, 0.7]          # inital guess

  N = 1000                      # number of steps
  T0 = 0.0                      # simulation start time
  T = 19.0                      # simulation end time
  ΔT = (T - T0) / (N - 1)       # simulation output stepsize

  sp = SimParams(T0, ΔT, T)

  # plot(T0:ΔT:T, u)
  nm = SPECTRAL_MC_NOISE_1

  mk_model = (θ, z) -> pendulum(u, t -> nm.w(z, t), θ) # physical model

  z = rand(nm.K)
  # plot(T0:ΔT:T, t -> nm.w(z, t))

  ys = simulate1(mk_model(θ, z), sp, θ)  # output data

  M = 100
  ZS = rand(nm.K, M)            # noise realizations

  EstProblem(mk_model, ZS, sp, ys, θ, θ0)
end

p = PROBLEM1
yhat = mk_yhat(p.mk_model, p.sp, p.ZS)
fit = curve_fit((t, θ) -> yhat(θ), p.sp.T0:p.sp.ΔT:p.sp.T, p.ys, p.θ0)
fit.params
