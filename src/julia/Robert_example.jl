using Random, LaTeXStrings

include("new_noise_interpolation.jl")
include("noise_generation.jl")
include("simulation.jl")
Random.seed!(1234)

# This script demonstrates difference in smoothness of generate disturbance
# for different choices of Q as well as with and without linear interpolation

# --------------- SIMULATION PARAMETERS --------------
N = 100                         # number of steps of the simulation
const Ts = 0.1                  # stepsize
const N_trans = 1 #TODO: INCREASE              # number of steps of the transient
const M = 1#50                   # number of noise realizations
const ms = collect(1:M)         # enumerate the realizations
const σ = 0.002                 # measurement noise standard deviation
Nw = 200#Int(1e3)          # Number of noise samples, excluding the initial one, x_e(0)
Nw_extra = 10
δ = N*Ts/Nw                  # Sampling frequency of noise model
P = 0#10           # Number of inter-sample samples stored
Q = 0#100#Int(1e8)         # Number of inter-sample states stored
use_interpolation = true    # Use linear interpolation of noise instea dof
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
# # 3-dimensional, pole excess 3, twice differentiable
# A = [0 0 -(4^2);
#            1 0 -(4^2+2*4*0.1);
#            0 1 -(2*4*0.1+1)]
# B = reshape([0.5; 0.0; 0.0], (3,1))
# C = [0 0 1]
# # 4-dimensional, pole excess 4, three times differentiable
# A = [0 0 0 -24;
#            1 0 0 -33;
#            0 1 0 -19;
# B = reshape([0.5; 0.0; 0.0; 0.0], (4,1))
#            0 0 1 -2.8]
# C = [0 0 0 1]
nx = size(A)[1]

# ----------------- DATA GENERATION ----------------
noise_model = discretize_ct_noise_model(A, B, C, δ, zeros(nx,))
# M+1 to generate one realization for true system as well
noise_uniform_dat, noise_inter_dat = generate_noise_new(Nw+Nw_extra, M+1, P, nx)
# Computes all M realizations of filtered white noise
xw_mat = simulate_noise_process_new(noise_model, noise_uniform_dat)
WS = [ (C*xw_mat[i,m])[1] for i=1:Nw+1, m=1:M]

# z_all_uniform[m][i,j] is the j:th element of the i:th sample of
# realization m
uniform_data_true, inter_data_true = generate_noise_new(Nw+Nw_extra, 1, P, nx)
# noise_model_true = discretize_ct_noise_model(A, B, C, Ts, zeros(nx,))
xw_true = simulate_noise_process_new(noise_model, uniform_data_true)
xw_true = xw_true[:]
WS_true = [(C*xw_true[i])[1] for i=1:N+1]
times_true = 0:Ts:N*Ts

# ---------------- FUNCTION DEFINITIONS ---------------------

# Cost function
cost(yhat::Array{Float64, 1}, y::Array{Float64, 1}) =
  mean((yhat[N_trans:end] - y[N_trans:end]).^2)

function plot_costs_R(θs, costs, costs_baseline, θ0)
    pl = plot(
      xlabel=L"\theta",
      ylabel=L"\texttt{cost}(\theta)")

    plot!(pl, θs, costs_baseline, label="proposed method", linecolor=:red)
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

# ---------------- MAIN BODY ---------------------

T = N*Ts
# solve(problem, saveat)
# mk_problem(w_func, θ, N)
# w_func: t -> w(t)

# solve_in_parallel(wm_func, ms)
# wm_func: m -> solution
# where solution is the output of solve(), where the problem uses noise
# realization m and parameters θ


# ----------- USE THIS BLOCK OF CODE TO TEST NOISE FUNCITON SMOOTHNESS --------

isd_good = initialize_isd(100, Nw+Nw_extra, nx, true)
isd_line = initialize_isd(0, Nw+Nw_extra, nx, true)
isd_bad = initialize_isd(0, Nw+Nw_extra, nx, false)

function w_good(t::Float64)
    return (C*noise_inter(t, δ, A, B, xw_mat[:, 1], isd_good))[1]
end
function w_line(t::Float64)
    return (C*noise_inter(t, δ, A, B, xw_mat[:, 1], isd_line))[1]
end
function w_bad(t::Float64)
    return (C*noise_inter(t, δ, A, B, xw_mat[:, 1], isd_bad))[1]
end

t_vec = 0:0.001:0.1
w_vec_good = [w_good(t) for t=t_vec]
w_vec_line = [w_line(t) for t=t_vec]
w_vec_bad  = [w_bad(t) for t=t_vec]
# plot(t_vec, w_vec)

pl = plot(
  xlabel="t",
  ylabel="w(t)")

plot!(pl, t_vec, w_vec_good, label="Q=100", linecolor=:red)
plot!(pl, t_vec, w_vec_line, label="Q=0, interpolation", linecolor=:black)
plot!(pl, t_vec, w_vec_bad, label="Q=0, no interpolation", linecolor=:blue, lines = :dot)
# savefig(pl, "./noise_smoothness.svg")
