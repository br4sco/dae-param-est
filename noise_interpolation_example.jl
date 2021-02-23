using Plots, CSV, DataFrames
import Random
include("noise_interpolation.jl")
include("noise_generation.jl")
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

# # OLD DATA GENERATION
# nx = size(A)[1]
# x_dat = CSV.read("x_mat.csv", DataFrame)
# # x[k] contains the state vector from time t0 + Ts*k
# x  = [ [x_dat[row, 1]; x_dat[row, 2]] for row in 1:1:size(x_dat)[1]]
#
# const t0 = 0                # Initial time of noise model simulation
# const Ts = 0.05             # Sampling frequency of noise model
# const N  = size(x_dat)[1]   # Number of simulated time steps of noise model

# NEW DATA GENERATION
const Ts = 0.05             # Sampling frequency of noise model
save_data = false
N = 100     # Noise samples, excluding the initial one, x_e(0)
M = 10000
P = 2       # You can ignore this one for now, just keep it at 2
nx = size(A)[1]
noise_model = discretize_ct_model(A, B, C, Ts, zeros(nx, ))
metadata = load_metadata()
# If meta-paramters have changed, re-generate noise
if metadata != [N, M, P, nx]
    # Generates white noise realizations, NOT realizations of filtered white noise
    data_uniform, irrelevant_var = generate_noise(N, M, P, nx, save_data)
else
    # data_uniform, data_inter = load_data(N,M,P,nx)
    data_uniform = load_data(N,M,P,nx)
end
# Computes all M realizations of filtered white noise
x_mat = simulate_noise_process(noise_model, data_uniform)

# Using only the first realization right now. Each column of x corresponds to
# one realization, and each row to one time instant
function w(t::Float64)
    return (C*x_inter(t, Ts, A, B, x_mat[:, 1], noise_model.x0))[1]
end

δ = 0.025

plot(0:0.01:N*Ts, w)
# w(0.172)
# savefig("./plots.png")
