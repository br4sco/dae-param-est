using Distributed
using LaTeXStrings

include("helpers.jl")
include("noise_model.jl")
include("simulation.jl")

seed = 1234
Random.seed!(seed)

# === experiment parameters ===
const Ts = 0.05                                            # stepsize

M = 10                                                     # number of noise realizations
const m_true = 12                                           # pick the true system
const m_u = 1                                              # input realization
const ms = filter(m -> m != m_true && m != m_u, 1:(M + 2)) # enumerate the realizations

# noise_method_name = "Spectral Monte-Carlo"
# noise_fun = mk_spectral_mc_noise_model_1(50.0, 0.01, M + 2, 1.0)

const noise_method_name = "Pre-generated unconditioned noise"
const WS = read_unconditioned_noise_1(1100, 0.005, 100.0)
noise_fun(m::Int) = interpolation(100.0, WS[:, m])

const u_scale = 0.2                   # input scale
# const w_scale = 0.04                # noise scale
const w_scale = 0.02                  # noise scale

u(t::Float64) = u_scale * noise_fun(m_u)(t)
wm(m::Int) = t -> w_scale * noise_fun(m)(t)

const σ = 0.002                         # observation noise variance

# === physical model ===
# const output_state = 1                                       # 1 = x, 3 = y
# h(sol) = apply_outputfun(x -> x[output_state], sol)          # output function
h(sol) = apply_outputfun(x -> atan(x[1] / -x[3]), sol)          # output function

const m = 0.3                         # [kg]
const L = 6.25                        # [m], gives period T = 5s (T ≅ 2√L) not
                                # accounting for friction.
const g = 9.81                        # [m/s^2]
const k = 0.01                        # [1/s^2]

const θ0 = L                          # We try to estimate the pendulum length
mk_θs(θ) = [m, θ, g, k]

const φ0 = 0.                         # Initial angle of pendulum from negative
                                # y-axis

mk_problem(w, θ, N) = problem(pendulum(φ0, u, w, mk_θs(θ)), N, Ts)

# === cost function ===
cost(yhat, y) = mean((yhat - y).^2)

const Δθ = 0.2
const δθ = 0.05

const θs = (θ0 - Δθ * θ0):δθ:(θ0 + Δθ * θ0) |> collect
const nθ = length(θs)

# function mk_sim(N)
#   f(m, θ) = simulate(mk_model(mk_w(m), mk_θs(θ)), N, Ts)
# end
#
# function mk_sim_h(N)
#   f(m, θ) = simulate_h(mk_model(mk_w(m), mk_θs(θ)), N, Ts, h)
# end
#
# function mk_sim_h_baseline(N)
#   f(θ) = simulate_h(mk_model(t -> 0., mk_θs(θ)), N, Ts, h)
# end
#
# function mk_sim_h_m(N)
#   f(ms, θ) = simulate_h_m(m -> mk_model(mk_w(m), mk_θs(θ)), N, Ts, h, ms)
# end

# Visualize the effect of the noise at the true θ
function plot_system_at_true_param(N)
  T = N*Ts
  solvem(m) = solve(mk_problem(wm(m), θ0, N); saveat = 0:Ts:T)

  # We simulate at the true θ to do get a feel for the problem
  sol_true = solvem(m_true)
  @info "true solution retcode: $(sol_true.retcode)"

  sols = pmap(solvem , ms[1:min(10, end)])
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
  T = N*Ts
  ts = 0:Ts:T

  first_ms = ms[1:min(10, end)]
  nms = length(first_ms)

  solvew(w, θ) = solve(mk_problem(w, θ, N), saveat=ts) |> h

  σs = σ * rand(Normal(), N + 1)
  y = solvew(wm(m_true), θ0) + σs
  Y = solve_m(m -> solvew(wm(m), θ0), N, first_ms)

  cs = zeros(nms, nθ)
  for (i, θ) in enumerate(θs)
    @info "θ point $(i) of $(nθ)"
    yhat = solvew(t -> 0., θ)
    for m = 1:nms
      cs[m, i] = cost(yhat, Y[:, m] + σs)
    end
  end

  pl = plot(xlabel=L"\theta",
            ylabel=L"\texttt{cost}(\theta)",
            title = "u_scale = $(u_scale), w_scale = $(w_scale), N = $(N)")

  plot!(pl,
        θ -> cost(solvew(t -> 0., θ), y),
        θs,
        label="m = $(m_true), true system", linecolor = :red, linewidth = 3)

  for (i, m) in enumerate(first_ms)
    plot!(pl, θs, cs[i, :], linealpha = 0.5, label = "m = $(m)")
  end

  vline!(pl, [θ0], linecolor = :gray, lines = :dot, label="θ0")
end

# function plot_mean_vs_true_trajectory(N)
#   sim_h = mk_sim_h(N)
#   sim_h_m = mk_sim_h_m(N)
#   sim_h_baseline = mk_sim_h_baseline(N)
#
#   σs = σ * rand(Normal(), N + 1)
#   y = sim_h(m_true, θ0) + σs
#   yhat = mean(sim_h_m(ms, θ0), dims = 2)
#   yhat_baseline = sim_h_baseline(θ0)
#
#   pl = plot(xlabel="time [s]",
#             title = "u_scale = $(u_scale), w_scale = $(w_scale), N = $(N)")
#
#   plot!(pl, [yhat y], fillrange=[y yhat], fillalpha=0.2, c=:yellow, label="")
#   plot!(pl, [yhat_baseline y], fillrange=[y yhat_baseline], fillalpha=0.2, c=:blue, label="")
#   plot!(pl, yhat, linecolor = :green, linewidth = 1, label = "mean")
#   plot!(pl, yhat_baseline, linecolor = :blue, linewidth = 1, label = "baseline")
#   plot!(pl, y, linecolor = :red, linewidth = 1, label = "true trajectory")
# end
#
# function plot_baseline_vs_true_trajectory(N)
#   sim_h_baseline = mk_sim_h_baseline(N)
#   sim_h = mk_sim_h(N)
#
#   σs = σ * rand(Normal(), N + 1)
#   y = sim_h(m_true, θ0) + σs
#   yhat = sim_h_baseline(θ0)
#
#   pl = plot(xlabel="time [s]",
#             title = "u_scale = $(u_scale), w_scale = $(w_scale), N = $(N)")
#
#   plot!(pl, [yhat y], fillrange=[y yhat], fillalpha=0.2, c=:yellow, label="")
#   plot!(pl, yhat, linecolor = :green, linewidth = 1, label = "baseline")
#   plot!(pl, y, linecolor = :red, linewidth = 1, label = "true trajectory")
# end
#
function run(id, N)
  T = N*Ts
  ts = 0:Ts:T

  filename = "run_$(id)_$(M)_$(N)"

  solvew(w, θ) = solve(mk_problem(w, θ, N), saveat=ts) |> h

  y = solvew(wm(m_true), θ0) + σ * rand(Normal(), N + 1)

  function est()
    cs = zeros(nθ)
    cs_baseline = zeros(nθ)

    for (i, θ) in enumerate(θs)
      @info "θ point $(i) of $(nθ)"
      cs_baseline[i] = cost(solvew(t -> 0., θ), y)
      Y = solve_m(m -> solvew(wm(m), θ), N, ms)
      @info "mean(Y) = $(mean(Y)), var(Y) = $(var(Y))"
      cs[i] = cost(mean(Y, dims = 2), y)
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

  CSV.write(joinpath("data", "$(filename)_data.csv"), data)
  CSV.write(joinpath("data", "$(filename)_meta_data.csv"), meta_data)
end
