include("spectral_mc.jl")
include("noise_interpolation.jl")
include("simulation.jl")

using ProgressMeter
using LsqFit
using DelimitedFiles
using Statistics
using LaTeXStrings
using Distributions
using JLD

Random.seed!(1234)              # set the seed

struct EstProblem
  yhat::Function                # Function yhat(θ) that estimates y given ZS,
                                # where θ are the model paramaters.
  yhat_baseline::Function       # baseline of yhat(θ).
  tp::TimeParams                # Time parameters.
  u::Function                   # Input function
  w::Function                   # Disturbance of the true system.
  ys::Array{Float64, 1}         # Measurement of the true system.
  θ::Array{Float64, 1}          # Parameters of the true system.
  θ0::Array{Float64, 1}         # Initial guess of the parameters.
end

const Us = [0; rand(10)]        # Fix randomness of input function

# Returns a function y(θ), the observations given θ averaged over the M noise
# realizations, on the time interval given by tp. The function mk_mk_model
# returns a function that constructs the model of interest given θ. Simulations
# are run in parallel, start julia as `Julia --threads n` where n is the number
# of threads you wish to use.
function mk_yhatM(mk_mk_model::Function, tp::TimeParams, M::Int)::Function
  let N = tp.N
    function f(θ::Array{Float64, 1})::Array{Float64, 1}
      yhats = [Threads.Atomic{Float64}(0.0) for i in 1:(N+1)]
      @info "θ set to $(θ)"
      p = Progress(M, 1, "Running $(M) simulations...", 50)
      @inbounds Threads.@threads for m = 1:M
        mk_model = mk_mk_model(m)
        y = simulate1(mk_model, tp, θ)
        for k = 1:(N+1)
          Threads.atomic_add!(yhats[k], y[k])
        end
        next!(p)
      end
      map(y -> y[], yhats) / M
    end
  end
end

# Returns a function y(θ), the observations given θ on the time interval given
# by tp. The function mk_model constructs the model of interest given θ.
function mk_yhat1(mk_model::Function, tp::TimeParams)
  function f(θ::Array{Float64, 1})::Array{Float64, 1}
    simulate1(mk_model, tp, θ)
  end
end

function problem1(wscale, M)
  let
    θ = [1.]
    θ0 = [0.1]
    σ = 0.01

    N = 100
    T0 = 0.0
    ΔT = 0.05
    tp = TimeParams(T0, ΔT, N)

    let
      u = t -> interpolation(tp, Us)(t)

      θ = [1.]
      θ0 = [0.1]
      σ = 0.01

      N = 100
      T0 = 0.0
      ΔT = 0.05
      tp = TimeParams(T0, ΔT, N)

      nm = spectral_mc_noise_model_1(M)
      w = (m, t) -> wscale * nm(m, t)

      nm_true = spectral_mc_noise_model_1(1)
      w_true = t -> wscale * nm_true(1, t)

      model = m -> pendulumK(u, t -> w(m, t), T0)
      model_true = pendulumK(u, w_true, T0)

      ys = simulate1(model_true, tp, θ)
      ys += σ*rand(Normal(), length(ys))

      yhat_baseline = mk_yhat1(pendulumK(u, t -> 0.0, T0), tp)
      yhat = mk_yhatM(model, tp, M)

      EstProblem(yhat, yhat_baseline, tp, u, w_true, ys, θ, θ0)
    end
  end
end

function run_est(p, yhat, label)
  d = Dict()

  d["name"] = label
  d["theta"] = p.θ
  d["theta0"] = p.θ0

  ws = map(p.w, collect(time_range(p.tp)))
  snr = sum(p.ys.^2) / sum(ws.^2)

  d["snr"] = snr

  @info "Fitting θ"
  fit = curve_fit((t, θ) -> yhat(θ), time_range(p.tp), p.ys, p.θ0)
  @info "converged: $(fit.converged)"
  @info "Found params: $(fit.param)"
  d["converged"] = fit.converged
  d["thetahat"] = fit.param

  return d
end

function experiment(problem, M, wscale, n, name)
  runs = []
  for i = 1:n
    @info "run $(i) of $(n)"
    p = problem(wscale, M)
    push!(runs, run_est(p, p.yhat_baseline, "baseline"))
    push!(runs, run_est(p, p.yhat, "$(M)"))
  end

  save("$(name).jld", "runs", runs)

  @info "baseline: $(map(x->x["thetahat"], filter(x->x["name"] == "baseline", runs)))"
  @info "result: $(map(x->x["thetahat"], filter(x->x["name"] != "baseline", runs)))"

  return runs
end

function baseline_experiment(problem, wscales, n)
  tmp = []
  for wscale in wscales
    runs = []
    for i = 1:n
      @info "run $(i) of $(n)"
      p = problem(wscale, 1)
      push!(runs, run_est(p, p.yhat_baseline, "baseline"))
    end
    push!(tmp, runs)
  end
  save("baseline.jld", "wscales", tmp)
  return tmp
end
