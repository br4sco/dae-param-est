using Plots, CSV, DataFrames
import Random
include("new_noise_interpolation.jl")
include("noise_generation.jl")
include("simulation.jl")
Random.seed!(1234)

# 2-dimensional, pole excess 1
const A = [0 -(4^2);
      1 -(2*4*0.1);];
const B = reshape([0.5; 0.0], (2,1))
const C = [0 1]
# # 3-dimensional, pole excess 2
# const A = [0 0 -(4^2);
#            1 0 -(4^2+2*4*0.1);
#            0 1 -(2*4*0.1+1)]
# const B = reshape([0.5; 0.0; 0.0], (3,1))
# const C = [0 0 1]
# # 4-dimensional, pole excess 4
# const A = [0 0 0 -24;
#            1 0 0 -33;
#            0 1 0 -19;
#            0 0 1 -2.8]
# const B = reshape([0.5; 0.0; 0.0; 0.0], (4,1))
# const C = [0 0 0 1]

Ts_model = 0.05
N_model = 100
T = N_model * Ts_model

Ts = 1e-6                       # Sampling frequency of noise model
N = Int(ceil(T / Ts))        # Noise samples, excluding the initial one, x_e(0)
@info "N = $(N)"
M = 1
P = 4       # Number of inter-sample samples stored
Q = 1000       # Number of inter-sample states stored. Should have Q >= P
nx = size(A)[1]
noise_model = discretize_ct_noise_model(A, B, C, Ts, zeros(nx,))
noise_uniform_dat, noise_inter_dat = generate_noise_new(N, M, P, nx)
# Computes all M realizations of filtered white noise
x_mat = simulate_noise_process_new(noise_model, noise_uniform_dat)

# isd is short for "inter-sample data"
isd = initialize_isd(Q, N, nx)

m = 1

# Using only the m:th realization right now. Each column of x_mat corresponds to
# one realization, and each row to one time instant. Each element is itself
# a state-vector (Array{Float64, 1}), not a scalar
function w(t::Float64)
    return (C*noise_inter(t, Ts, A, B, x_mat[:, m], noise_inter_dat[m],
            isd, noise_model.x0))[1]
end

# pendulum(Φ::Float64, u::Function, w::Function, θ::Array{Float64, 1})
# model = pendulum(pi / 4, t -> 0., t -> 0., [0.3, 6.25, 9.81, 0.01])
model = pendulum(pi / 4, t -> 0., w, [0.3, 6.25, 9.81, 0.01])
sol = simulate(model, N_model, Ts_model)
@info "min(h) = $(min(diff(sol.t)...))"
plot(sol, vars = [1, 3, 8], layout = (3, 1))

# plot(0:δ:N*δ, w)
# w(0.172)
# savefig("./plots.png")
