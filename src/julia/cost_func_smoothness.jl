using Random, LaTeXStrings, DataFrames, CSV

include("noise_interpolation.jl")
include("noise_generation.jl")
include("simulation.jl")
Random.seed!(1234)

# --------------- SIMULATION PARAMETERS --------------
N = 100                         # number of steps of the simulation
const Ts = 0.1                  # stepsize
const N_trans = 1              # number of steps of the transient
const M = 20                   # number of noise realizations
const ms = collect(1:M)         # enumerate the realizations
const σ = 0.002                 # measurement noise standard deviation
Nw = 1000#Int(1e3)          # Number of noise samples, excluding the initial one, x_e(0)
Nw_extra = 100
δ = N*Ts/Nw                  # Sampling frequency of noise model
P = 0#10           # Number of inter-sample samples stored
Q = 0#100#Int(1e8)         # Number of inter-sample states stored
use_interpolation = true    # Use linear interpolation of noise instea dof
T = N*Ts
# conditional sampling when Q stored samples in an interval have been surpassed

# --------------- PHYSICAL MODEL PARAMETERS -------------
const u_scale = 0.0             # input scale
const w_scale = 3.0             # noise scale
const m = 0.3                   # [kg]
const L = 6.25  # [m], gives period T = 5s (T ≅ 2√L) not accounting for friction
const g = 9.81                  # [m/s^2]
const k = 0.05                  # [1/s^2]
const φ0 = pi / 8              # Initial angle of pendulum from negative y-axis

# u(t::Float64) = u_scale * noise_fun(m_u)(t)
u(t::Float64) = 0.0
h(sol) = apply_outputfun(x -> atan(x[1] / -x[3]), sol)         # output function
mk_problem(w, θ, N) = problem(pendulum(φ0, u, w, mk_θs(θ)), N, Ts)

# ------------------ PARAMETER PARAMETERS -------------------
mk_θs(θ) = [m, θ, g, k]

const θ0 = L                    # We try to estimate the pendulum length
const Δθ = 0.2
const δθ = 0.08
const θs = (θ0 - Δθ * θ0):δθ:(θ0 + Δθ * θ0) |> collect
const nθ = length(θs)

# ---------------- NOISE MODEL --------------
# 2-dimensional, pole excess 2, once differentiable
A = [0 -(4^2);
      1 -(2*4*0.1);];
B = reshape([0.5; 0.0], (2,1))
C = [0 1]
nx = size(A)[1]

# ----------------- DATA GENERATION ----------------
noise_model = discretize_ct_noise_model(A, B, C, δ, zeros(nx,))
# M+1 to generate one realization for true system as well
noise_uniform_dat, noise_inter_dat = generate_noise(Nw+Nw_extra, M+1, 0, nx)
# Computes all M realizations of filtered white noise
xw_mat = simulate_noise_process(noise_model, noise_uniform_dat)

# I THINK THIS CAN BE DELETED!!!!!!!!!!
# # z_all_uniform[m][i,j] is the j:th element of the i:th sample of
# # realization m
# uniform_data_true, inter_data_true = generate_noise(Nw+Nw_extra, 1, P, nx)
# # noise_model_true = discretize_ct_noise_model(A, B, C, Ts, zeros(nx,))
# xw_true = simulate_noise_process(noise_model, uniform_data_true)
# xw_true = xw_true[:]
# WS_true = [(C*xw_true[i])[1] for i=1:N+1]
# times_true = 0:Ts:N*Ts

# ---------------- FUNCTION DEFINITIONS ---------------------

# Cost function
cost(yhat::Array{Float64, 1}, y::Array{Float64, 1}) =
  mean((yhat[N_trans:end] - y[N_trans:end]).^2)

function plot_costs_R(θs, costs, costs_baseline, θ0)
    pl = plot(
      xlabel=L"\theta",
      ylabel=L"\texttt{cost}(\theta)")

    plot!(pl, θs, costs_baseline, label="proposed method", linecolor=:red, lines = :dashdot)
    n_min_b = argmin(costs_baseline)
    # plot!(pl, [θs[n_min_b]], [costs_baseline[n_min_b]],
    #   color=:red, seriestype = :scatter, label = "", alpha = 0.5)

    plot!(pl, θs, costs, label="re-sampling", linecolor=:black)
    n_min = argmin(costs)
    # plot!(pl, [θs[n_min]], [costs[n_min]],
    #   color=:black, seriestype = :scatter, label = "", alpha = 0.5)

    vline!(pl, [θ0], linecolor=:gray, lines = :dot, label="θ0")
    pl
end

# ----------------------------- MAIN BODY --------------------------

solvewθ(w, θ) = solve(mk_problem(w, θ, N), saveat=0:Ts:T) |> h
# TODO: THIS ISN'T RIGHT PLACE TO DEFINE, BUT DOESN'T MATTER WHEN Q IS 0!!!
Q = 0
@warn "Q=0 was used"
isd = initialize_isd(Q, Nw+Nw_extra, nx, use_interpolation)
isd_true = initialize_isd(Q, Nw+Nw_extra, nx, true)

function w(t::Float64, m::Int64)
    return (C*noise_inter(t, δ, A, B, xw_mat[:, m], isd))[1]
end
wm(m::Int64) = t -> w_scale*w(t, m)

function w_true(t::Float64)
    return (C*noise_inter(t, δ, A, B, xw_mat[:, M+1], isd_true))[1]
end
@time y = solve(mk_problem(w_true, θ0, N), saveat=0:Ts:T) |> h

# Proposed method cost function
cs = zeros(nθ)
for (i, θ) in enumerate(θs)
  Y = solve_in_parallel(m -> solvewθ(wm(m), θ), ms)
  cs[i] = cost(reshape(mean(Y, dims = 2), :), y)
end

# Naive method cost function
cs_jagged = zeros(nθ)
for (i, θ) in enumerate(θs)
    local noise_uniform_dat, noise_inter_dat = generate_noise(Nw, M, 0, nx)
    # Computes all M realizations of filtered white noise
    local xw_mat = simulate_noise_process(noise_model, noise_uniform_dat)

    function w_jagged(t::Float64, m::Int64)
        return (C*noise_inter(t, δ, A, B, xw_mat[:, m], isd))[1]
    end
    wm_jagged(m::Int64) = t -> w_scale*w_jagged(t, m)

    Y = solve_in_parallel(m -> solvewθ(wm_jagged(m), θ), ms)
    cs_jagged[i] = cost(reshape(mean(Y, dims = 2), :), y)
end

plot_costs_R(θs, cs_jagged, cs, θ0)
# df = DataFrame(Thetas = θs, Proposed=cs, Jagged=cs_jagged)
# CSV.write("smooth_cost_func.csv", df)
# savefig("./cost_smoothness.svg")
