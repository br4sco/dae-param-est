using LaTeXStrings
using JLD
using Distributed

include("helpers.jl")
include("noise_model.jl")
include("simulation.jl")

seed = 1234
Random.seed!(seed)

M = 10                          # numer of noise realizations
N = 100                         # Number of steps
Ts = 0.05                       # Stepsize
T = N*Ts
ts = 0:Ts:T

filename = "run_$(M)_$(N)"

u_scale = 0.0
w_scale = 2.0

σ = 0.02                        # observation noise variance
output_state = 3                # y-position

noise_method_name = "Spectral Monte-Carlo"

uu = mk_spectral_mc_noise_model_1(10.0, 0.05, 1, u_scale)(1)
u = t -> u_scale * uu(t)

mk_w = mk_spectral_mc_noise_model_1(50.0, 0.01, M + 1, w_scale)

m = 0.3                         # [kg]
L = 6.25                        # [m], gives period T = 5s (T ≅ 2√L) not
                                # accounting for friction.
g = 9.81                        # [m/s^2]
k = 0.01                        # [1/s^2]

θ0 = L                          # We try to estimate the pendulum length
mk_θs = θ -> [m, θ, g, k]

φ0 = pi / 4                     # Initial angle of pendulum from negative
                                # y-axis

h = x -> x[output_state]
mk_model = (w, θ) -> pendulum(φ0, u, w, θ)
sim = (m, θ) -> simulate(mk_model(mk_w(m), mk_θs(θ)), N, Ts)
sim_h = (m, θ) -> simulate_h(mk_model(mk_w(m), mk_θs(θ)), N, Ts, h)
sim_h_baseline = θ -> simulate_h(mk_model(t -> 0., mk_θs(θ)), N, Ts, h)
sim_h_m = (ms, θ) -> simulate_h_m(m -> mk_model(mk_w(m), mk_θs(θ)), N, Ts, h, ms)

ms = shuffle(1:(M + 1))         # shuffle the realizations
m_true = pop!(ms)               # pick the true system

# Visualize the effect of the noise at the true θ
function plot_system_at_true_param()
  sol_true = sim(m_true, θ0)
  @info "true solution retcode: $(sol_true.retcode)"

  simm = θ -> pmap(m -> sim(m, θ), ms)

  # We simulate at the true θ to do get a feel for the problem
  sols = simm(L)
  @info "failed simulations $(count(s -> s.retcode != :Success, sols))"

  vars = [(0,1), (0,3), (0,8)]

  ps = simulation_plots(T, sols, vars, label="",
                        linecolor=:gray, linealpha=0.5)

  map((p, v) ->
      plot!(p, sol_true, tspan=(0, T), vars=[v], label="", linecolor=:red),
      ps, vars)

  plot(ps...)
end

y = sim_h(m_true, θ0) + σ * rand(Normal(), N + 1)
plot(ts, y)

cost = yhat -> mean((yhat - y).^2)

Δθ = 0.2
δθ = 0.05

θs = (θ0 - Δθ * θ0):δθ:(θ0 + Δθ * θ0) |> collect
nθ = length(θs)

function est()
  cs = zeros(nθ)
  cs_baseline = zeros(nθ)

  for (i, θ) in enumerate(θs)
    @info "θ point $(i) of $(nθ)"
    cs_baseline[i] = sim_h_baseline(θ) |> cost
    cs[i] = mean(sim_h_m(ms, θ), dims=2) |> cost
  end

  cs_baseline, cs
end

# cs_baseline, cs = est()

data = DataFrame(θ = θs, cost = cs, cost_baseline = cs_b)
meta_data = DataFrame(
  θ0 = θ0,
  N = N,
  Ts = Ts,
  M = M,
  σ = σ,
  u_scale = u_scale,
  w_scale = w_scale,
  noise_method_name = noise_method_name,
  seed = seed
)

CSV.write("$(filename)_data.csv", data)
CSV.write("$(filename)_meta_data.csv", meta_data)
