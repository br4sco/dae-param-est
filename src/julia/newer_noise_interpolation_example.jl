using Plots, CSV, DataFrames
import Random
include("new_noise_interpolation.jl")
include("noise_generation.jl")
include("newest_noise_interpolation.jl")
Random.seed!(1234)

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
#            0 0 1 -2.8]
# B = reshape([0.5; 0.0; 0.0; 0.0], (4,1))
# C = [0 0 0 1]

# const Ts = 0.05             # Sampling frequency of noise model
const Ts = 0.05             # Sampling frequency of noise model
save_data = false
# N = 100     # Noise samples, excluding the initial one, x_e(0)
N = 2     # Noise samples, excluding the initial one, x_e(0)
M = 100
P = 4       # Number of inter-sample samples stored
Q = 1000       # Number of inter-sample states stored
# N = 8     # Noise samples, excluding the initial one, x_e(0)
# M = 2
# P = 1       # Number of inter-sample samples stored
nx = size(A)[1]
noise_model = discretize_ct_noise_model(A, B, C, Ts, zeros(nx,))
rng_seed = 123  # RNG seed used for data-generation
noise_uniform_dat, noise_inter_dat = generate_noise_new(N, M, P, nx)
# Computes all M realizations of filtered white noise
x_mat = simulate_noise_process_new(noise_model, noise_uniform_dat)

# println(noise_inter_dat)# DEBUG

num_samples = zeros(Int64, N)

isd = initialize_isd(Q, N, nx)

m = 1

# Using only the first realization right now. Each column of x_mat corresponds to
# one realization, and each row to one time instant. Each element is itself
# a state-vector (Array{Float64, 1}), not a scalar
function w(t::Float64)
    return (C*noise_inter(t, Ts, A, B, x_mat[:, m], noise_inter_dat[m], num_samples, noise_model.x0))[1]
end


function w_new(t::Float64)
    return (C*noise_inter_new(t, Ts, A, B, x_mat[:, m], noise_inter_dat[m],
            isd, noise_model.x0))[1]
end

δ = 0.025

# plot(0:0.001:N*Ts, w)
# plot(0:0.001:N*Ts, w_new)

# t = 0.0172
t = 0.0368
# t = 0.172
# t = 0.242
plot(0:0.001:N*Ts, w)
# plot(0:0.0001:N*Ts, w_new)
# # # savefig("./plots.png")
