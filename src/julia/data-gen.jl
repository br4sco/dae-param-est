using DataFrames
using Dierckx

include("noise_model.jl")
include("simulation.jl")

seed = 1234
Random.seed!(seed)

# ==================
# === PARAMETERS ===
# ==================

# === TIME ===
const Ts = 0.1                  # stepsize
const N_trans = 250             # number of steps of the transient

# === NOISE ===
# load pre-generated realizations
const δ = 0.01                  # noise sampling time
const Tw = 550.0
const Mw = 1500
const noise_method_name = "Pre-generated unconditioned noise (δ = $(δ))"
const WS = read_unconditioned_noise_1(Mw, δ, Tw)

const K = size(WS, 1) - 1       # number of steps taken for the noise
const M_max = size(WS, 2)       # number of realizations avalible in the noise
                                # data

# We do linear interpolation between exact values because it's fast
function interpw(m::Int)
  function w(t::Float64)
    k = Int(floor(t / δ)) + 1
    w0 = WS[k, m]
    w1 = WS[k + 1, m]
    w0 + (t - (k - 1) * δ) * (w1 - w0) / δ
  end
end

# const tsw = collect(0:δ:(δ * K))
# interpw(w::Int) = Spline1D(tsw, WS[:, m]; k=2, bc="error", s=0.0)

# we compute the maximum number of steps we can take with this noise data
const N_max = Int(floor(K * δ / Ts))
const T_max = N_max * Ts

# === MODEL (AND DATA) PARAMETERS ===
const σ = 0.002                 # observation noise variance
const u_scale = 0.2             # input scale
const w_scale = 2.0             # noise scale

const m = 0.3                   # [kg]
const L = 6.25                  # [m], gives period T = 5s (T ≅ 2√L) not
                                # accounting for friction.
const g = 9.81                  # [m/s^2]
const k = 0.05                  # [1/s^2]

const φ0 = 0. / 8              # Initial angle of pendulum from negative y-axis

# === INPUT FUNCTION ===
# we choose the last realization of the noise as input
u(t::Float64) = u_scale * interpw(M_max)(t)

# === PROCESS NOISE FUNCTION ===
function wm(m::Int)             # gives the m'th realization
  let itp = interpw(m)
    t -> w_scale * itp(t)
  end
end

# === OUTPUT FUNCTIONS ===
f(x::Array{Float64, 1}) = atan(x[1] / -x[3]) # applied on the state at each step
h(sol) = apply_outputfun(f, sol)             # for our model
h_baseline(sol) = apply_outputfun(f, sol)    # for the baseline method

# === MODEL REALIZATION AND SIMULATION ===
mk_θs(θ) = [m, θ, g, k]
realize_model(w, θ, N) = problem(pendulum(φ0, u, w, mk_θs(θ)), N, Ts)

# === SOLVER PARAMETERS ===
const abstol = 1e-6
const reltol = 1e-3
# const abstols = [abstol, abstol, abstol, abstol, Inf, Inf, Inf, Inf]
const maxiters = Int64(1e6)

solvew(w, θ, N; kwargs...) = solve(realize_model(w, θ, N),
                                   saveat=0:Ts:(N*Ts),
                                   abstol = abstol,
                                   reltol = reltol,
                                   maxiters = maxiters;
                                   kwargs...)

# === DATASET ===
const θ0 = L                    # true value of θ
const ndata = 100               # the size of the dataset

# we pick the last realizations (excluding the input to form the dataset)
const ms_data = (M_max - ndata):(M_max - 1) |> collect

# data set output function
h_data(sol) = apply_outputfun(x -> f(x) + σ * rand(Normal()), sol)

# === EXPERIMENT PARAMETERS ===
const Δθ = 0.2θ0               # determines the interval around θ0
const hθ = 15                  # number of steps in the left/right interval
const δθ = round(Δθ / hθ; sigdigits = 1)
const θs = (θ0 - hθ * δθ):δθ:(θ0 + hθ * δθ) |> collect
const nθ = length(θs)

# =======================
# === DATA GENERATION ===
# =======================

const data_dir = "data"

exp_path(id) = joinpath(data_dir, id)
mk_exp_dir(id) =  id |> exp_path |> mkdir

function calc_Y()
  solve_in_parallel(m -> solvew(wm(m), θ0, N_max) |> h_data, ms_data)
end

function write_Y(expid, Y)
  p = joinpath(exp_path(expid), "Y.csv")
  writedlm(p, Y, ",")
end

function read_Y(expid)
  p = joinpath(exp_path(expid), "Y.csv")
  readdlm(p, ",")
end

function calc_baseline_Y()
  solve_in_parallel(θ -> solvew(t -> 0., θ, N_max) |> h_baseline, θs)
end

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

function calc_mean_Y()
  M = M_max - 1 - ndata
  ms = collect(1:M)
  Ym = zeros(N_max + 1, nθ)

  for (i, θ) in enumerate(θs)
    @info "solving for point ($(i)/$(nθ)) of θ"
    Y = solve_in_parallel(m -> solvew(wm(m), θ, N_max) |> h, ms)
    Ym[:, i] .+= reshape(mean(Y, dims = 2), :)
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
                 u_scale = u_scale,
                 w_scale = w_scale,
                 noise_method_name = noise_method_name,
                 seed = seed)
  CSV.write(p, df)
end

function read_meta_data(expid)
  p = joinpath(exp_path(expid), "meta_data.csv")
  CSV.File(p) |> DataFrame
end
