using DataFrames
using Dierckx
using Distributions

# include("noise_model.jl")
include("simulation.jl")
include("noise_generation.jl")
include("noise_interpolation.jl")

seed = 1234
Random.seed!(seed)

# ==================
# === PARAMETERS ===
# ==================

# === TIME ===
const δ = 0.01                  # noise sampling time
const Ts = 0.1                  # stepsize
# const Ts = 0.5                  # stepsize larger

# === NOISE ===
const Q = 100
const M = 500
const E = 500
const Nw = 10000
const W  = 100
const Nw_extra = 100   # Number of extra samples of noise trajectory to generate

# === PR-GENERATED ===
# const noise_method_name = "Pre-generated unconditioned noise (δ = $(δ))"
const noise_method_name = "Pre-generated conditioned noise (δ = $(δ), Q = $(Q))"

# We do linear interpolation between exact values because it's fast
function interpw(WS::Array{Float64, 2}, m::Int)
  function w(t::Float64)
    k = Int(floor(t / δ)) + 1
    w0 = WS[k, m]
    w1 = WS[k + 1, m]
    w0 + (t - (k - 1) * δ) * (w1 - w0) / δ
  end
end

# === PRE-GENERATED DATA ===
const WSd =
  readdlm(joinpath("data",
                   "experiments",
                   "unconditioned_noise_data_501_001_500000_1234_alsvin.csv"),
          ',')

const WSu =
  readdlm(joinpath("data",
                   "experiments",
                   "unconditioned_noise_input_001_500000_1234_alsvin.csv"),
          ',')

const WSm =
  readdlm(joinpath("data",
                   "experiments",
                   "unconditioned_noise_model_500_001_500000_1234_alsvin.csv"),
          ',')

# === NOISE INTERPOLATION ===
const nx = 2
const A = [0.0 1.0; -4^2 -0.8]
const B = reshape([0.0 1.0], (2,1))
const C = [1.0 0.0]
const x0 = zeros(nx)

# interpolation over w(tk)
wmd(e::Int) = interpw(WSd, e)
wmm(m::Int) = interpw(WSm, m)
u(t::Float64) = interpw(WSu, 1)(t)

# === MODEL ===
# we compute the maximum number of steps we can take
# const K = min(size(WSd, 1), size(WS, 1)) - 2
const N = Int(floor(Nw * δ / Ts))
# const N = 10000

# number of realizations in the model
# const M = size(WS, 2)

# === MODEL (AND DATA) PARAMETERS ===
const σ = 0.002                 # observation noise variance
const u_scale = 0.2             # input scale
const u_bias = 0.0              # input bias
const w_scale = 0.6             # noise scale

const m = 0.3                   # [kg]
const L = 6.25                  # [m], gives period T = 5s (T ≅ 2√L) not
                                # accounting for friction.
const g = 9.81                  # [m/s^2]
const k = 0.05                  # [1/s^2]

const φ0 = 0. / 8              # Initial angle of pendulum from negative y-axis

# === OUTPUT FUNCTIONS ===
f(x::Array{Float64, 1}) = x[7]            # applied on the state at each step
h(sol) = apply_outputfun(f, sol)          # for our model
h_baseline(sol) = apply_outputfun(f, sol) # for the baseline method

# === MODEL REALIZATION AND SIMULATION ===
const θ0 = L                    # true value of θ
mk_θs(θ::Float64) = [m, θ, g, k]
realize_model(w::Function, θ::Float64, N::Int) =
  problem(pendulum(φ0, t -> u_scale * u(t) + u_bias, w, mk_θs(θ)), N, Ts)

# === SOLVER PARAMETERS ===
const abstol = 1e-7
const reltol = 1e-4
const maxiters = Int64(1e8)

solvew(w::Function, θ::Float64, N::Int; kwargs...) =
  solve(realize_model(w, θ, N),
        saveat=0:Ts:(N*Ts),
        abstol = abstol,
        reltol = reltol,
        maxiters = maxiters;
        kwargs...)

# dataset output function
h_data(sol) = apply_outputfun(x -> f(x) + σ * rand(Normal()), sol)

# === EXPERIMENT PARAMETERS ===
const lnθ = 15                  # number of steps in the left interval
const rnθ = 15                  # number of steps in the right interval
const δθ = 0.08
const θs = (θ0 - lnθ * δθ):δθ:(θ0 + rnθ * δθ) |> collect
const nθ = length(θs)

# =======================
# === DATA GENERATION ===
# =======================

const data_dir = joinpath("data", "experiments")

exp_path(id) = joinpath(data_dir, id)
mk_exp_dir(id) =  id |> exp_path |> mkdir

calc_y(e::Int) = solvew(t -> w_scale * wmd(e)(t), θ0, N) |> h_data

function calc_Y()
  es = collect(1:E)
  solve_in_parallel(calc_y, es)
end

function write_Y(expid, Y)
  p = joinpath(exp_path(expid), "Yd1.csv")
  writedlm(p, Y, ",")
end

function read_Y(expid)
  p = joinpath(exp_path(expid), "Yd1.csv")
  readdlm(p, ",")
end

calc_baseline_y_N(N::Int, θ::Float64) = solvew(t -> 0., θ, N) |> h_baseline

calc_baseline_y(θ::Float64) = calc_baseline_y_N(N, θ)

calc_baseline_Y() = solve_in_parallel(calc_baseline_y, θs)

function write_theta(expid)
  p = joinpath(exp_path(expid), "theta.csv")
  writedlm(p, θs, ",")
end

function read_theta(expid)
  p = joinpath(exp_path(expid), "theta.csv")
  readdlm(p, ",")
end

function write_baseline_Y(expid, Yb)
  p = joinpath(exp_path(expid), "Yb.csv")
  writedlm(p, Yb, ",")
end

function read_baseline_Y(expid)
  p = joinpath(exp_path(expid), "Yb.csv")
  readdlm(p, ",")
end

calc_mean_y_N(N::Int, θ::Float64, m::Int) =
  solvew(t -> w_scale * wmm(m)(t), θ, N) |> h

calc_mean_y(θ::Float64, m::Int) = calc_mean_y_N(N, θ, m)

function calc_mean_Y()
  ms = collect(1:M)
  Ym = zeros(N + 1, nθ)

  for (i, θ) in enumerate(θs)
    @info "solving for point ($(i)/$(nθ)) of θ"
    Y = solve_in_parallel(m -> calc_mean_y(θ, m), ms)
    y = reshape(mean(Y, dims = 2), :)
    writedlm(joinpath(data_dir, "tmp", "y_mean_$(i).csv"), y, ',')
    Ym[:, i] .+= y
  end
  Ym
end

function write_mean_Y(expid, Ym)
  p = joinpath(exp_path(expid), "Ym.csv")
  writedlm(p, Ym, ",")
end

function read_mean_Y(expid)
  p = joinpath(exp_path(expid), "Ym.csv")
  readdlm(p, ",")
end

function write_meta_data(expid)
  p = joinpath(exp_path(expid), "meta_data.csv")
  df = DataFrame(Ts = Ts,
                 σ = σ,
                 φ0 = φ0,
                 θ0 = θ0,
                 u_scale = u_scale,
                 w_scale = w_scale,
                 noise_method_name = noise_method_name,
                 seed = seed,
                 atol = abstol,
                 rtol = reltol)
  CSV.write(p, df)
end

function read_meta_data(expid)
  p = joinpath(exp_path(expid), "meta_data.csv")
  CSV.File(p) |> DataFrame
end
