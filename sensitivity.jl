using Plots
gr()

include("noise_model.jl")
include("helpers.jl")
include("simulation.jl")

Random.seed!(1234)

N = 100
Ts = 0.05
T = N*Ts
ts = 0:Ts:T

u_scale = 0.2

u = mk_spectral_mc_noise_model_1(10.0, 0.05, 1, u_scale)(1)

plot(u, ts)

xidxs = [
  (1, 0),                            # x
  (3, 0),                            # y
  (5, 1)                             # λ
]

function filter_sol(sol)
  f = (i, n) -> map(t -> sol(t, Val{n})[i], sol.t)
  map(x -> f(x...), xidxs)
end

nx = length(xidxs)

mk_model = θ -> pendulum2(pi / 4, u, t -> 0., θ)
sim = θ -> simulate(mk_model(θ), N, Ts)

m = 0.3                         # [kg]
L = 6.25                        # [m], gives period T = 5s (T ≅ 2√L) not
                                # accounting for friction.
g = 9.81                        # [m/s^2]
k = 0.01                        # [1/s^2]
θ_nominal = [m, L, g, k]
nθ = length(θ_nominal)

pert = [-0.2, -0.1, 0.1, 0.2]
npert = length(pert)

sol_nominal = sim(θ_nominal)

tt = sol_nominal.t

xs_nominal = filter_sol(sol_nominal)

nominal_pls = map(x -> plot(tt, x, linecolor=:red, label=""), xs_nominal)
plot(nominal_pls..., layout=(1, nx))

function plot_sensitivity()
  plss = []
  for i=1:nθ
    pls = map(x -> plot(), nominal_pls)

    for j=1:npert
      θ = copy(θ_nominal)
      θ[i] += pert[j] * θ[i]
      xs = filter_sol(sim(θ))
      for k=1:nx
        plot!(pls[k], tt, xs[k], linecolor=:gray, linealpha=0.5, label="")
      end
    end

    for k=1:nx
      plot!(pls[k], tt, xs_nominal[k], linecolor=:red, label="")
    end

    push!(plss, pls)
  end

  plot(vcat(plss...)..., layout=(nθ, nx))
end
