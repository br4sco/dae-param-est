include("spectral_mc.jl")
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
  mk_mk_model::Function         # Function f(z), where z is a noise realization
                                # and reurns a function g(θ), where θ are the
                                # model paramaters and returns a system model.
  ZS::Array{Float64, 2}         # Matrix where columns are noise realiztions.
  tp::TimeParams                # Time parameters.
  ws::Array{Float64, 1}         # Disturbance of the true system.
  ys::Array{Float64, 1}         # Measurement of the true system.
  θ::Array{Float64, 1}          # Parameters of the true system.
  θ0::Array{Float64, 1}         # Initial guess of the parameters.
end

# Returns a function y(θ), the observations given θ averaged over the M noise
# realizations in ZS, on the time interval given by tp. The function
# mk_mk_model returns a function that constructs the model of interest given
# θ. Simulations are run in parallel, start julia as `Julia --threads n` where
# n is the number of threads you wish to use.
function mk_yhatM(mk_mk_model::Function,
                 tp::TimeParams,
                 ZS::Array{Float64, 2})::Function

  let N = tp.N
    function f(θ::Array{Float64, 1})::Array{Float64, 1}
      M = size(ZS, 2)
      yhats = [Threads.Atomic{Float64}(0.0) for i in 1:(N+1)]
      @info "θ set to $(θ)"
      p = Progress(M, 1, "Running $(M) simulations...", 50)
      @inbounds Threads.@threads for m = 1:M
        mk_model = mk_mk_model(ZS[:, m])
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

function mk_est_problem(tp, θ, θ0, σ, u, w, mk_mk_model, mk_ZS, M)
  z = reshape(mk_ZS(1), :)
  ys = simulate1(mk_mk_model(z), tp, θ)
  ys +=  σ*rand(Normal(), length(ys))
  ws = map(t -> w(z, t), collect(time_range(tp)))

  ZS = mk_ZS(M)

  EstProblem(mk_mk_model, ZS, tp, ws, ys, θ, θ0)
end

function problem0()
  let us = [0; rand(10)]
    θ = [1.]
    θ0 = [0.5]
    σ = 0.01

    N = 100
    T0 = 0.0
    ΔT = 0.05
    tp = TimeParams(T0, ΔT, N)

    function u(t)
      interpolation(tp, us)(t)
    end

    M = 1

    w = (z, t) -> 0.0
    mk_mk_model = z -> pendulumK(u, t -> w(z, t), T0)

    mk_est_problem(tp, θ, θ0, σ, u, w, mk_mk_model, M -> zeros(1, M), M)
  end
end

function problem1(wscale, M)
  let us = [0; rand(10)]
    θ = [1.]
    θ0 = [0.1]
    σ = 0.01

    N = 100
    T0 = 0.0
    ΔT = 0.05
    tp = TimeParams(T0, ΔT, N)

    function u(t)
      interpolation(tp, us)(t)
    end

    noise = spectral_mc_noise_model_1()
    mk_ZS = noise.mk_ZS
    w = (z, t) -> wscale * noise.w(z, t)
    mk_mk_model = z -> pendulumK(u, t -> w(z, t), T0)

    mk_est_problem(tp, θ, θ0, σ, u, w, mk_mk_model, mk_ZS, M)
  end
end

function mk_run(problem, Ms, wscales, θs)
  function run()
    runs = Any[]

    for wscale in wscales
      for M in Ms
        d = Dict()
        p = problem(wscale, M)

        @info "Parameters wscale: $(wscale), M: $(M)"
        d["theta"] = p.θ
        d["theta0"] = p.θ0

        d["wscale"] = wscale
        d["M"] = M

        @info "Computing cost function over θ"
        yhat = mk_yhatM(p.mk_mk_model, p.tp, p.ZS)
        yhats = map(θ -> yhat([θ]), θs)
        d["yhats"] = yhats
        costs = map(yhat -> mean((yhat - p.ys).^2), yhats)
        d["costs"] = costs

        @info "Fitting θ"
        fit = curve_fit((t, θ) -> yhat(θ), time_range(p.tp), p.ys, p.θ0)
        @info "converged: $(fit.converged)"
        @info "Found params: $(fit.param)"
        d["converged"] = fit.converged
        d["thetahat"] = fit.param

        push!(runs, d)
      end
    end

    p0 = problem0()
    yhat0 = mk_yhatM(p0.mk_mk_model, p0.tp, p0.ZS)
    yhats0 = map(θ -> yhat0([θ]), θs)
    costs0 = map(yhat -> mean((yhat - p0.ys).^2), yhats0)

    save("data.jld", "runs", runs, "thetas", θs, "yhats0", yhats0, "costs0", costs0)
  end
end
