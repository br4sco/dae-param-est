using Plots, CSV, DataFrames
import Random
include("noise_interpolation.jl")
Random.seed!(1234)

const A = [-1. 0.; 0. -2.]
# const B = [0.3 0.; 0. 0.5]
const B = reshape([0.5; 0.0], (2,1))
const C = [1 1]
# Mohamed seems to have used this model, though cholesky decomposition fails then
# const A = [0 -16; 1 -0.8]
# const B = reshape([0.5; 0.0], (2,1))
# const C = [0 1]

x_dat = CSV.read("x_mat.csv", DataFrame)
# x[k] contains the state vector from time t0 + Ts*k
x  = [ [x_dat[row, 1]; x_dat[row, 2]] for row in 1:1:size(x_dat)[1]]

const t0 = 0                # Initial time of noise model simulation
const Ts = 0.05             # Sampling frequency of noise model
const N  = size(x_dat)[1]   # Number of simulated time steps of noise model

function w(t::Float64)
  return (C*x_inter(t, Ts, A, B, x))[1]
end

δ = 0.025
# time_steps = 10
# wδ = zeros(time_steps)     # state dimension hard-coded here
# w_k = zeros(time_steps)
#
#
#
# for t_k = 1:1:time_steps
#     w_k[t_k] = (C*x[t_k])[1]     # Sampled value of w
#     wδ[t_k] =  w(t_k*Ts+δ)     # Inter-sample value of w
# end
#
# sample_times = Ts*(1:1:time_steps)
# inter_times = sample_times + δ*ones(size(sample_times))
# # If I knew how to plot in Julia, I would plot w_k as a function of sample_times
# # (this is the sampled noise values), and wδ as a function of inter_times
# # (these are the interpolated noise values)

# to plot w you can write
plot(t0:0.01:N*Ts, w)
# I get errors like ERROR: PosDefException: matrix is not positive definite; Cholesky factorization failed, on this interval.
