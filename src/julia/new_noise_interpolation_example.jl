using Plots, Future
import Random
include("new_noise_interpolation.jl")
include("noise_generation.jl")
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

const Ts = 0.05             # Sampling frequency of noise model
# N = 100     # Noise samples, excluding the initial one, x_e(0)
N = 2     # Noise samples, excluding the initial one, x_e(0)
M = 100
P = 0       # Number of inter-sample samples stored
Q = 10       # Number of inter-sample states stored
use_interpolation = true
# N = 8     # Noise samples, excluding the initial one, x_e(0)
# M = 2
# P = 1       # Number of inter-sample samples stored
nx = size(A)[1]
noise_model = discretize_ct_noise_model(A, B, C, Ts, zeros(nx,))
noise_uniform_dat, noise_inter_dat = generate_noise_new(N, M, P, nx)
# Computes all M realizations of filtered white noise
x_mat = simulate_noise_process_new(noise_model, noise_uniform_dat)

isd = initialize_isd(Q, N, nx, use_interpolation)

m = 1

# Using only the m:th realization right now. Each column of x_mat corresponds to
# one realization, and each row to one time instant. Each element is itself
# a state-vector (Array{Float64, 1}), not a scalar
function w(t::Float64)
    return (C*noise_inter(t, Ts, A, B, x_mat[:, m], noise_inter_dat[m],
            isd))[1]
end

δ = 0.025

# plot(0:0.001:N*Ts, w)

# t = 0.0172
t = 0.0368
# t = 0.172
# t = 0.242
t_vec = 0:0.001:N*Ts
w_vec = [w(t) for t=t_vec]
plot(t_vec, w_vec)
# plot(N*Ts-0.001:0.001:N*Ts, w)
# # # savefig("./plots.png")

# ---------- TESTING DETERMINISTIC COST FUNCTION --------------------
# template_rng = MersenneTwister(123)
# # Generates vector with 10 rng objects that give independent realizations
# # multiplies of big(10)^20 is comventional jump step size, so probably
# # shouldn't be changed
# rng_vec = [Future.randjump(template_rng, i*big(10)^20) for i=0:9]
#
# # θs = 0.1:0.01:0.2
# θs = [0.1, 0.11]
# for θ in θs
#     # Example for only one realization, m=3
#     m = 3
#     # Make sure to use new isd object for each θ and m
#     isd_local = initialize_isd(Q, N, nx, use_interpolation)
#     # IMPORTANT! copy rng object before defining function, so that every call
#     # to w(t) uses the same rng object, and not e.g. a fresh copy of the template
#     cp_rng = copy(rng_vec[m])
#     # In general, the matrices A and B should depend on θ here
#     function w(t::Float64)
#         return (C*noise_inter(t, Ts, A, B, x_mat[:, m], noise_inter_dat[m],
#                 isd_local, 10e-12, cp_rng))[1]
#     end
#     println(w(0.012))
#     println(w(0.025))
# end
