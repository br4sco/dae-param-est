include("spectral_mc.jl")
include("simulation.jl")

using ProgressMeter
using LsqFit
using DelimitedFiles
using Statistics
using LaTeXStrings
using Distributions

Random.seed!(1234)              # set the seed

struct EstProblem
  mk_mk_model::Function         # Function f(z), where z is a noise realization
                                # and reurns a function g(θ), where θ are the
                                # model paramaters, and returns a system model.
  ZS::Array{Float64, 2}         # Matrix where columns are noise realiztions.
  tp::TimeParams                # Time parameters.
  ws::Array{Float64, 1}         # Disturbance of the true system.
  ys::Array{Float64, 1}         # Measurement of the true system.
  θ::Array{Float64, 1}          # Parameters of the true system.
  θ0::Array{Float64, 1}         # Initial guess of the parameters.
end

function mk_yhat(mk_mk_model::Function,
                 tp::TimeParams,
                 ZS::Array{Float64, 2})::Function

  let N = tp.N
    function f(θ::Array{Float64, 1})::Array{Float64, 1}
      M = size(ZS, 2)
      yhats = [Threads.Atomic{Float64}(0.0) for i in 1:(N+1)]
      @info "θ set to $(θ)"
      p = Progress(M, 1, "Running $(M) simulations...", 50)
      @inbounds Threads.@threads for m = 1:M
        z = ZS[:, m]
        mk_model = mk_mk_model(z)
        y = simulate1(mk_model, tp, θ)
        for i = 1:(N+1)
          Threads.atomic_add!(yhats[i], y[i])
        end
        next!(p)
      end
      map(y -> y[], yhats) / M
    end
  end
end

function plot_signal_to_noise_ratio(ys, ws)
  plot(ys.^2 ./ ws.^2, yaxis=:log, xlabel="time (steps)", ylabel=L"y^2/w^2")
end

function mk_est_problem(tp, θ, θ0, e, u, w, mk_mk_model, mk_ZS, M)


  z = reshape(mk_ZS(1), :)
  ys = simulate1(mk_mk_model(z), tp, θ)
  ys += e * rand(Normal(), length(ys))
  ws = map(t -> w(z, t), collect(time_range(tp)))

  ZS = mk_ZS(M)

  EstProblem(mk_mk_model, ZS, tp, ws, ys, θ, θ0)
end

function problem0()
  let us = [0; rand(10)]
    θ = [1.]
    θ0 = [0.5]
    e = 0.01

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

    mk_est_problem(tp, θ, θ0, e, u, w, mk_mk_model, M -> zeros(1, M), M)
  end
end

function problem1()
  let us = [0; rand(10)]
    θ = [1.]
    θ0 = [0.1]
    wscale = 0.02
    e = 0.01

    N = 100
    T0 = 0.0
    ΔT = 0.05
    tp = TimeParams(T0, ΔT, N)

    function u(t)
      interpolation(tp, us)(t)
    end

    M = 100

    noise = spectral_mc_noise_model_3()
    mk_ZS = noise.mk_ZS
    w = (z, t) -> wscale * noise.w(z, t)
    mk_mk_model = z -> pendulumK(u, t -> w(z, t), T0)

    mk_est_problem(tp, θ, θ0, e, u, w, mk_mk_model, mk_ZS, M)
  end
end

p = problem1()
# plot_signal_to_noise_ratio(p.ys, p.ws)

yhat = mk_yhat(p.mk_mk_model, p.tp, p.ZS)

# display(plot(0.001:0.05:1.5, θ -> mean((yhat([θ]) - p.ys).^2)))

fit = curve_fit((t, θ) -> yhat(θ), time_range(p.tp), p.ys, p.θ0)
@info "converged: $(fit.converged)"
@info "Found params: $(fit.param)"
writedlm("param.csv",  fit.param, ',')
