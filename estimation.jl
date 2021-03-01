using Distributed
using LaTeXStrings

include("helpers.jl")
include("noise_model.jl")
include("simulation.jl")

seed = 1234
Random.seed!(seed)

# === experiment parameters ===
Ts = 0.05                       # stepsize

M = 2000                                 # number of noise realizations
m_true = 7                               # pick the true system
ms = filter(m -> m != m_true, 1:(M + 1)) # enumerate the realizations

noise_method_name = "Spectral Monte-Carlo"

u_scale = 0.2                   # input scale
w_scale = 0.8                   # noise scale

uu = mk_spectral_mc_noise_model_1(10.0, 0.05, 1, u_scale)(1)
u = t -> u_scale * uu(t)

mk_w = mk_spectral_mc_noise_model_1(50.0, 0.01, M + 1, w_scale)
# mk_w = exact_noise_interpolation_model_1(N, Ts, M + 1, w_scale)

σ = 0.02                        # observation noise variance

# === physical model ===
output_state = 1                # 1 = x, 3 = y
h = x -> x[output_state]

m = 0.3                         # [kg]
L = 6.25                        # [m], gives period T = 5s (T ≅ 2√L) not
                                # accounting for friction.
g = 9.81                        # [m/s^2]
k = 0.01                        # [1/s^2]

θ0 = L                          # We try to estimate the pendulum length
mk_θs = θ -> [m, θ, g, k]

φ0 = pi / 4                     # Initial angle of pendulum from negative
                                # y-axis

mk_model = (w, θ) -> pendulum(φ0, u, w, θ)

# === cost function ===
cost = (yhat, y) -> mean((yhat - y).^2)

Δθ = 0.2
δθ = 0.05

θs = (θ0 - Δθ * θ0):δθ:(θ0 + Δθ * θ0) |> collect
nθ = length(θs)

function mk_sim(N)
  (m, θ) -> simulate(mk_model(mk_w(m), mk_θs(θ)), N, Ts)
end

function mk_sim_h(N)
  (m, θ) -> simulate_h(mk_model(mk_w(m), mk_θs(θ)), N, Ts, h)
end

function mk_sim_h_baseline(N)
  θ -> simulate_h(mk_model(t -> 0., mk_θs(θ)), N, Ts, h)
end

function mk_sim_h_m(N)
  (ms, θ) -> simulate_h_m(m -> mk_model(mk_w(m), mk_θs(θ)), N, Ts, h, ms)
end

# Visualize the effect of the noise at the true θ
function plot_system_at_true_param(N)
  T = N*Ts

  sim = mk_sim(N)
  sol_true = sim(m_true, θ0)
  @info "true solution retcode: $(sol_true.retcode)"

  simm = θ -> pmap(m -> sim(m, θ), ms[1:min(10, end)])

  # We simulate at the true θ to do get a feel for the problem
    sols = simm(L)
  @info "failed simulations $(count(s -> s.retcode != :Success, sols))"

  vars = [(0,1), (0,3), (0,8)]

  ps = simulation_plots(T, sols, vars, label="",
                        linecolor=:gray, linealpha=0.5)

  map((p, v) ->
      plot!(p, sol_true, tspan=(0, T), vars=[v], linecolor=:red),
      ps, vars)

  plot(ps...)
end

function plot_baseline_costs(N)

  first_ms = ms[1:min(10, end)]
  nms = length(first_ms)

  sim_h_baseline = mk_sim_h_baseline(N)
  sim_h = mk_sim_h(N)
  sim_h_m = mk_sim_h_m(N)

  σs = σ * rand(Normal(), N + 1)

  cs = zeros(nms, nθ)

  y = sim_h(m_true, θ0)
  Y = sim_h_m(first_ms, θ0)

  for (i, θ) in enumerate(θs)
    @info "θ point $(i) of $(nθ)"
    yhat = sim_h_baseline(θ)
    for m = 1:nms
      cs[m, i] = cost(yhat, Y[:, m] + σs)
    end
  end

  pl = plot(xlabel=L"\theta",
            ylabel=L"\texttt{cost}(\theta)",
            title = "u_scale = $(u_scale), w_scale = $(w_scale), N = $(N)")

  plot!(pl,
        θ -> cost(sim_h_baseline(θ), y + σs),
        θs,
        label="m = $(m_true), true system", linecolor = :red, linewidth = 3)

  for (i, m) in enumerate(first_ms)
    plot!(pl, θs, cs[i, :], linealpha = 0.5, label = "m = $(m)")
  end

  vline!(pl, [θ0], linecolor = :gray, lines = :dot, label="θ0")
end

function run(id, N)
  T = N*Ts
  ts = 0:Ts:T

  filename = "run_$(id)_$(M)_$(N)"

  sim_h = mk_sim_h(N)
  sim_h_baseline = mk_sim_h_baseline(N)
  sim_h_m = mk_sim_h_m(N)

  y = sim_h(m_true, θ0) + σ * rand(Normal(), N + 1)
  plot(ts, y)

  function est()
    cs = zeros(nθ)
    cs_baseline = zeros(nθ)

    for (i, θ) in enumerate(θs)
      @info "θ point $(i) of $(nθ)"
      cs_baseline[i] = cost(sim_h_baseline(θ), y)
      cs[i] = cost(mean(sim_h_m(ms, θ), dims=2), y)
    end

    cs_baseline, cs
  end

  cs_baseline, cs = est()

  data = DataFrame(θ = θs, cost = cs, cost_baseline = cs_baseline)
  meta_data = DataFrame(
    θ0 = θ0,
    N = N,
    Ts = Ts,
    M = M,
    σ = σ,
    u_scale = u_scale,
    w_scale = w_scale,
    noise_method_name = noise_method_name,
    seed = seed,
    m_true = m_true,
    output_state = output_state
  )

  CSV.write("$(filename)_data.csv", data)
  CSV.write("$(filename)_meta_data.csv", meta_data)
end
