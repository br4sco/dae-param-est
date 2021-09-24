using DataFrames
using Dierckx
using Distributions
using CSV
using LaTeXStrings
using Plots
using StatsPlots

include("simulation.jl")

seed = 1234
Random.seed!(seed)

# ==================
# === PARAMETERS ===
# ==================

# === TIME ===
const δ = 0.01                  # noise sampling time
const Ts = 0.1                  # step-size

const noise_method_name = "Pre-generated unconditioned noise (δ = $(δ))"

# We do linear interpolation between exact values because it's fast
function interpw(WS::Array{Float64,2}, m::Int)
  function w(t::Float64)
    k = Int(floor(t / δ)) + 1
    w0 = WS[k, m]
    w1 = WS[k+1, m]
    w0 + (t - (k - 1) * δ) * (w1 - w0) / δ
  end
end

# === MODEL (AND DATA) PARAMETERS ===
const σ = 0.002                 # observation noise variance
const u_scale = 0.2             # input scale
const u_bias = 0.0              # input bias
const w_scale = 0.6             # noise scale

const m = 0.3                   # [kg]
const L = 6.25                  # [m], gives period T = 5s (T ≅ 2√L) not
# accounting for friction.
const g = 9.81                  # [m/s^2]
const k = 6.25                  # [1/s^2]

const φ0 = 0.0                   # Initial angle of pendulum from negative y-axis

# === OUTPUT FUNCTIONS ===
# The state vector x from the solver is organized as follows:
# x = [
#   x1              -- position in the x-direction
#   x1'             -- velocity in the x-direction
#   x2              -- position in the y-direction
#   x2'             -- velocity in the y-direction
#   int(x3)         -- integral of the tension per unit length (due to stabilized formulation)
#   int(dummy)      -- integral of dummy variable (due to stabilized formulation)
#   y               -- the output y = atan(x1/-x2) is computed by the solver
# ]
f(x::Array{Float64,1}) = x[7]            # applied on the state at each step
h(sol) = apply_outputfun(f, sol)          # for our model
h_baseline(sol) = apply_outputfun(f, sol) # for the baseline method

# === MODEL REALIZATION AND SIMULATION ===
const θ0 = k                    # true value of θ
mk_θs(θ::Float64) = [m, L, g, θ]
realize_model(u::Function, w::Function, θ::Float64, N::Int) = problem(
  pendulum(φ0, t -> u_scale * u(t) + u_bias, t -> w_scale * w(t), mk_θs(θ)),
  N,
  Ts,
)

# === SOLVER PARAMETERS ===
const abstol = 1e-8
const reltol = 1e-5
const maxiters = Int64(1e8)

solvew(u::Function, w::Function, θ::Float64, N::Int; kwargs...) = solve(
  realize_model(u, w, θ, N),
  saveat = 0:Ts:(N*Ts),
  abstol = abstol,
  reltol = reltol,
  maxiters = maxiters;
  kwargs...,
)

# data-set output function
h_data(sol) = apply_outputfun(x -> f(x) + σ * rand(Normal()), sol)

# === EXPERIMENT PARAMETERS ===
const lnθ = 30                  # number of steps in the left interval
const rnθ = 50                  # number of steps in the right interval
const δθ = 0.08
const θs = (θ0-lnθ*δθ):δθ:(θ0+rnθ*δθ) |> collect
const nθ = length(θs)

# ===============================================
# === HELPER FUNCTIONS TO READ AND WRITE DATA ===
# ===============================================

const data_dir = joinpath("data", "experiments")

# create directory for this experiment
exp_path(expid) = joinpath(data_dir, expid)

theta_path(expid) = joinpath(exp_path(expid), "theta.csv")
data_Y_path(expid, suffix) = joinpath(exp_path(expid), "Y_$(suffix).csv")
baseline_Y_path(expid) = joinpath(exp_path(expid), "Yb.csv")
mean_Y_path(expid) = joinpath(exp_path(expid), "Ym.csv")
meta_data_path(expid) = joinpath(exp_path(expid), "meta_data.csv")

function write_meta_data(expid)
  p = joinpath(exp_path(expid), "meta_data.csv")
  df = DataFrame(
    Ts = Ts,
    σ = σ,
    φ0 = φ0,
    θ0 = θ0,
    u_scale = u_scale,
    w_scale = w_scale,
    noise_method_name = noise_method_name,
    seed = seed,
    atol = abstol,
    rtol = reltol,
  )
  if !isfile(p)
    @info "Writing meta data"
    CSV.write(p, df)
  end
end

function read_meta_data(expid)
  p = joinpath(exp_path(expid), "meta_data.csv")
  CSV.File(p) |> DataFrame
end

# =======================
# === COMPUTE OUTPUTS ===
# =======================

"""
  `mk_exp_dir(expid)`

Creates an experiment folder named `expid`
"""
function mk_exp_dir(expid)
  joinpath(exp_path(expid), "tmp") |> mkpath
end

"""
A struct collecting the outputs from the experiment.

`Y` is the output of the true system
`Ym` is the output of the proposed method
`Yb` is the output of the baseline method
`θs` is the grid of the estimated parameter
`meta_data` holds meta data of the experiment
"""
struct Outputs
  Y::Array{Float64,2}
  Ym::Array{Float64,2}
  Yb::Array{Float64,2}
  θs::Array{Float64,1}
  meta_data::DataFrame
end

"""
  `get_outputs(expid)`

Computes and saves experiment outputs to the experiment folder `expid`. If this
folder already contains computed outputs, these are instead read from disk.
Finally, the outputs are returned.
"""
function get_outputs(expid)
  if !isfile(theta_path(expid))
    @info "Writing θs"
    writedlm(theta_path(expid), θs, ',')
  end

  mk_wm(WS::Array{Float64,2}) = (m::Int) -> interpw(WS, m)

  # a single realization of the noise serves as input
  WS_input = readdlm(
    joinpath(data_dir, "unconditioned_noise_input_001_500000_1234_alsvin.csv"),
    ',',
  )

  u(t::Float64) = interpw(WS_input, 1)(t)

  # compute the maximum number of steps we can take
  K = min(size(WS_input, 1)) - 2
  N = Int(floor(K * δ / Ts))

  # === We first generate the output of the true system in two batches ===
  function calc_Y(WS::Array{Float64,2})
    @assert (K < size(WS, 1)) "Noise data size mismatch"
    M = size(WS, 2)
    ms = collect(1:M)
    wm = mk_wm(WS)
    solve_in_parallel(m -> solvew(u, wm(m), θ0, N) |> h_data, ms)
  end

  if isfile(data_Y_path(expid, 1))
    @info "Reading output of true system batch 1"
    Y1 = readdlm(data_Y_path(expid, 1), ',')
  else
    @info "Generating output of true system batch 1"
    WS = readdlm(
      joinpath(
        data_dir,
        "unconditioned_noise_data_500_001_500000_1234_alsvin.csv",
      ),
      ',',
    )
    Y1 = calc_Y(WS)
    writedlm(data_Y_path(expid, 1), Y1, ',')
  end

  if isfile(data_Y_path(expid, 2))
    @info "Reading output of true system batch 2"
    Y2 = readdlm(data_Y_path(expid, 2), ',')
  else
    @info "Generating output of true system batch 2"
    WS = readdlm(
      joinpath(
        data_dir,
        "unconditioned_noise_data_500_001_500000_1234_alsvin_b.csv",
      ),
      ',',
    )
    Y2 = calc_Y(WS)
    writedlm(data_Y_path(expid, 2), Y2, ',')
  end


  # === We then generate the output of the baseline model ===
  if isfile(baseline_Y_path(expid))
    @info "Reading output of of baseline model"
    Yb = readdlm(baseline_Y_path(expid), ',')
  else
    @info "Generating output of baseline model"
    Yb = solve_in_parallel(θ -> solvew(u, t -> 0.0, θ, N) |> h_baseline, θs)
    writedlm(baseline_Y_path(expid), Yb, ',')
  end


  # === Finally we generate the output of the proposed model ==
  function calc_mean_Y()
    WS = readdlm(
      joinpath(
        data_dir,
        "unconditioned_noise_model_500_001_500000_1234_alsvin.csv",
      ),
      ',',
    )
    @assert (K < size(WS, 1)) "Noise data size mismatch"
    M = size(WS, 2)
    ms = collect(1:M)
    Ym = zeros(N + 1, nθ)
    wm = mk_wm(WS)

    calc_mean_y(θ::Float64, m::Int) = solvew(u, wm(m), θ, N) |> h
    for (i, θ) in enumerate(θs)
      p = joinpath(exp_path(expid), "tmp", "y_mean_$(i).csv")
      if isfile(p)
        @info "found saved solution for point ($(i)/$(nθ)) of θ"
      else
        @info "solving for point ($(i)/$(nθ)) of θ"
        Y = solve_in_parallel(m -> calc_mean_y(θ, m), ms)
        y = reshape(mean(Y, dims = 2), :)
        writedlm(p, y, ',')
        Ym[:, i] .+= y
      end
    end
    Ym
  end

  if isfile(mean_Y_path(expid))
    @info "Reading output of proposed model"
    Ym = readdlm(mean_Y_path(expid), ',')
  else
    @info "Generating output of proposed model"
    Ym = calc_mean_Y()
    writedlm(mean_Y_path(expid), Ym, ',')
  end

  write_meta_data(expid)

  Outputs(hcat(Y1, Y2), Ym, Yb, θs, read_meta_data(expid))
end


# ============================
# === POST PROCESS OUTPUTS ===
# ============================

"""
  `cost(y, yhat)`

Calculates the mean squared error of `y` and `yhat`.
"""
cost(y::Array{Float64,1}, yhat::Array{Float64,1}) = mean((y - yhat) .^ 2)

"""
  `calc_costs(Y, Yhat, N_trans, N)`

Calculates mean squared errors, for all columns of the estimated outputs Yhat,
over the columns of the true outputs Y, where `N` is the length to consider and
`N_tran` is the length of the transient.
"""
calc_costs(Y::Array{Float64,2}, Yhat::Array{Float64,2}, N_trans::Int, N::Int) =
  mapslices(
    y ->
      mapslices(yhat -> cost(y[N_trans:N], yhat[N_trans:N]), Yhat, dims = 1)[:],
    Y,
    dims = 1,
  )

"""
  `calc_theta_hats(θs, Y, Yhat, N_trans, N)`

Calculates the `θhats` that minimizes the costs (see `calc_costs`) given by the
true outputs `Y`, the estimated outputs `Yhat`, the transient `N_tran`, the
output length `N`, and the parameter grid `θs`. Each column `j` of `Yhat` is
assumed to be the output given θs[j].
"""
function calc_theta_hats(
  θs::Array{Float64,1},
  Y::Array{Float64,2},
  Yhat::Array{Float64},
  N_trans::Int,
  N::Int,
)
  cost = calc_costs(Y, Yhat, N_trans, N)
  argmincost = mapslices(argmin, cost, dims = 1)[:]
  θs[argmincost]
end

"""
    `thetahat_boxplots(outputs, Ns, N_trans)`

Produces two boxplots, comparing the baseline method to the proposed method from
`outputs` at the different lengths in `Ns` and with transient `N_trans`. The
first half of the boxlplots shows the baseline method, labeled bl, and the
second part shows the proposed method, labeled m.
"""
function thetahat_boxplots(outputs::Outputs, Ns::Array{Int,1}, N_trans::Int)
  θhatbs =
    map(N -> calc_theta_hats(outputs.θs, outputs.Y, outputs.Yb, N_trans, N), Ns)
  θhatms =
    map(N -> calc_theta_hats(outputs.θs, outputs.Y, outputs.Ym, N_trans, N), Ns)
  θhats = hcat(θhatbs..., θhatms...)
  labels = reshape([map(N -> "bl $(N)", Ns); map(N -> "m $(N)", Ns)], (1, :))
  idxs = 1:(2length(Ns))
  p = boxplot(
    θhats,
    xticks = (idxs, labels),
    label = "",
    ylabel = L"\hat{\theta}",
    notch = false,
  )

  hline!(p, [θ0], label = L"\theta_0", linestyle = :dot, linecolor = :gray)
end
