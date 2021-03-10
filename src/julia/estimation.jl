using Distributed
using LaTeXStrings
using Profile
using Interpolations

include("helpers.jl")
include("noise_model.jl")
include("simulation.jl")

seed = 1234
Random.seed!(seed)

# === experiment parameters ===
const Ts = 0.1                  # stepsize
const N_trans = 250             # number of steps of the transient
const N_max = 3000              # number if steps after the transient
const T = (N_trans + N_max)*Ts
const ts = 0:Ts:T
const M = 1000                  # number of noise realizations
const ms = collect(1:M)         # enumerate the realizations
const m_true_start = 1001       # the start of the true systems
const n_true = 100              # number of true systems
const ms_true =                 # true systems
  collect(m_true_start:(m_true_start + n_true - 1))
const σ = 0.002                 # observation noise variance

# === noise model ===
const δ = 0.01
const Tw = 550.0
const Mw = 1500
const noise_method_name = "Pre-generated unconditioned noise (δ = $(δ))"
const WS = read_unconditioned_noise_1(Mw, δ, Tw)
const K = size(WS, 1) - 1
const tsw = collect(0:δ:(δ * K))

function noise_fun(m::Int)
  function w(t::Float64)
    k = Int(floor(t / δ)) + 1
    w0 = WS[k, m]
    w1 = WS[k + 1, m]
    w0 + (t - (k - 1) * δ) * (w1 - w0) / δ
  end
end

wm(m::Int) = t -> w_scale * noise_fun(m)(t)

# === physical model parameters ===
const u_scale = 0.2             # input scale
const w_scale = 2.0             # noise scale

const m = 0.3                   # [kg]
const L = 6.25                  # [m], gives period T = 5s (T ≅ 2√L) not
                                # accounting for friction.
const g = 9.81                  # [m/s^2]
const k = 0.05                  # [1/s^2]

const φ0 = 0. / 8              # Initial angle of pendulum from negative y-axis

# u(t::Float64) = u_scale * noise_fun(m_u)(t)
u(t::Float64) = u_scale * noise_fun(1500)(t)

f(x) = atan(x[1] / -x[3])
# f(x) = x[1]

# h_bl(sol) = apply_outputfun(f, sol)
# h_mean(sol) = 2 * h(sol)
# h_true(sol) =
#   apply_outputfun(x -> f(x) * (1 + rand(Normal())^2) +  σ * rand(Normal()), sol)

h_bl(sol) = apply_outputfun(f, sol)
h_mean(sol) = h(sol)
h_true(sol) = apply_outputfun(x -> f(x) + σ * rand(Normal()), sol)

# === Parameter params ===
mk_θs(θ) = [m, θ, g, k]

const θ0 = L                    # We try to estimate the pendulum length
const Δθ = 0.2
const δθ = 0.08
const θs = (θ0 - Δθ * θ0):δθ:(θ0 + Δθ * θ0) |> collect
const nθ = length(θs)

mk_problem(w, θ, N) = problem(pendulum(φ0, u, w, mk_θs(θ)), N_trans + N, Ts)

# === cost function ===
cost(yhat::Array{Float64, 1}, y::Array{Float64, 1}) =
  mean((yhat[N_trans:end] - y[N_trans:end]).^2)

# === helper functions ===
solvew(w, θ, N) = solve(mk_problem(t -> w(t), θ, N), saveat=ts)
calc_yhat_bl(θ) = solvew(t -> 0., θ, N_max) |> h_bl
calc_y_true(m) = solvew(wm(m), θ0, N_max) |> h_true

calc_yhat(m, θ, N) = solvew(wm(m), θ, N) |> h_mean
calc_m(prob, ms) = solve_m(m -> prob(m), ms)

# === pre-computed values ===
const Y = calc_m(m -> calc_y_true(m), ms_true)
const ys_bl = map(calc_yhat_bl, θs)

function calc_baseline_cost(N)
  CS_b = zeros(nθ, n_true)

  for n = 1:n_true
    CS_b[:, n] .+= map(yhat -> cost(yhat[1:N_trans + N], Y[1:N_trans + N, n]), ys_bl)
  end

  CS_b
end

function plot_baseline_costs(N)
  CS_b = calc_baseline_cost(N)

  pl = plot(θs, mean(CS_b, dims = 2), ribbon = var(CS_b, dims = 2),
             fillalpha = .5, label="baseline", color=:red)

  vline!(pl, [θ0], linecolor=:green, lines = :dot, label="θ0")
end

function baseline_mins(N)
  CS_b = calc_baseline_cost(N)
  is = mapslices(argmin, CS_b, dims = 1)
  θs[is]
end

function print_baseline_mse(N)
  mse = mean((baseline_mins(N) .- θ0).^2)
  println("mse: $(mse)")
end

function print_baseline_bias(N)
  bias = mean(baseline_mins(N)) - θ0
  println("bias: $(bias)")
end


# Visualize the effect of the noise at the true θ
function plot_system_at_true_param(ms_true, ms, N)
  T = N*Ts
  solvem(m) = solve(mk_problem(wm(m), θ0, N); saveat = 0:Ts:T)

  # We simulate at the true θ to do get a feel for the problem
  sol_true = solvem(m_true)
  @info "true solution retcode: $(sol_true.retcode)"

  sols = pmap(solvem , ms)
  @info "failed simulations $(count(s -> s.retcode != :Success, sols))"

  vars = [(0,1), (0,3), (0,8)]

  ps = simulation_plots(T, sols, vars, label="",
                        linecolor=:gray, linealpha=0.5)

  map((p, v) ->
      plot!(p, sol_true, tspan=(0, T), vars=[v], linecolor=:red),
      ps, vars)

  plot(ps...)
end

function run1(dir, id, m_true, N)
  filename = "run_$(id)_$(M)_$(N)"

  y = Y[1:N, m_true]
  CS_b = calc_baseline_cost(N)
  cs_baseline = CS_b[:, m_true]

  function est()
    cs = zeros(nθ)
    cs_baseline = zeros(nθ)

    for (i, θ) in enumerate(θs)
      @info "θ point $(i) of $(nθ)"
      Yhat = calc_m(m -> calc_yhat(m, θ, N), ms)
      @info "mean(Yhat) = $(mean(Yhat)), var(Yhat) = $(var(Yhat))"
      cs[i] = cost(reshape(mean(Yhat, dims = 2), :), y)
    end

    cs_baseline, cs
  end

  cs_baseline, cs = est()

  yhat =
    reshape(mean(calc_m(m -> calc_yhat(m, θ0, N), ms), dims = 2), :)
  yhat_baseline = ys_bl[m_true]

  data = DataFrame(θ = θs, cost = cs, cost_baseline = cs_baseline)
  data_extra = DataFrame(y = y, yhat = yhat, yhat_baseline = yhat_baseline)
  meta_data = DataFrame(
    θ0 = θ0,
    N = N,
    N_trans = N_trans,
    Ts = Ts,
    M = M,
    σ = σ,
    u_scale = u_scale,
    w_scale = w_scale,
    noise_method_name = noise_method_name,
    seed = seed,
    m_true = m_true
  )

  CSV.write(joinpath("data", dir, "$(filename)_cost_data.csv"), data)
  CSV.write(joinpath("data", dir, "$(filename)_output_data.csv"), data_extra)
  CSV.write(joinpath("data", dir, "$(filename)_meta_data.csv"), meta_data)
end

function run(dir, N)
  mkdir(joinpath("data", dir))
  for (id, m_true) in enumerate(ms_true)
    @info "run $(id) of $(length(ms_true))"
    run1(dir, id, id, N)
  end
end
