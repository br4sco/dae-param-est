# Checks if probability distribution of noise realization is as predicted by theory

include("noise_interpolation.jl")
using Statistics

# #DEBUG
# using LinearAlgebra
# var_mat = [2 0; 0 3]
# col = cholesky(var_mat)
# StdDev = col.L
# # noise = StdDev*randn(Float64, (2, 1))
# M = 10000
# noise_vec = NaN*ones(M, 2)
# for m = 1:1:M
#     noise_vec[m, :] = ( StdDev*randn(Float64, (2,1)) )'
# end
#
# println(cov(noise_vec[:, 1]))
# println(cov(noise_vec[:, 2]))
# println(cov(noise_vec[:, 1], noise_vec[:, 2]))
# # YEP, SEEMS TO WORK AS EXPECTED, VARIANCE PRETTY MUCH THE SAME AS var_mat

# ------------------------------------------------------------------------------


Random.seed!(1234)

const A = [-1. 0.; 0. -2.]
const B = [0.3 0.; 0. 0.5]
const C = [1 1]

x_dat = CSV.read("x_mat.csv", DataFrame)
n = size(A)[1]
M = Int(size(x_dat)[2]÷2)
const t0 = 0                # Initial time of noise model simulation
const Ts = 0.05             # Sampling frequency of noise model
const N = size(x_dat)[1]

k = 20
δ = 0.02
t = k*Ts+δ

x1δ_vec = NaN*ones(M)
x2δ_vec = NaN*ones(M)

function x_inter(t::Float64, x::Array{Array{Float64, 1}, 1})
    return x_inter(t, Ts, A, B, x)
end

# DEBUG
m = 1
x  = [ [x_dat[row, 2*m-1]; x_dat[row, 2*m]] for row in 1:1:N]
xδ = x_inter(t, x)

# for m = 1:1:M
#     x  = [ [x_dat[row, 2*m-1]; x_dat[row, 2*m]] for row in 1:1:size(x_dat)[1]]
#     xδ = x(t, x)
#     x1δ_vec[m] = xδ[1]
#     x2δ_vec[m] = xδ[2]
# end

# # sδ1 = cov(x1δ_vec)
# # sδ2 = cov(x2δ_vec)
# # sδ12 = cov(x1δ_vec, x2δ_vec)
#
# x1_k   = [x_dat[k, 2*m-1] for m in 1:1:M]
# x2_k   = [x_dat[k, 2*m] for m in 1:1:M]
# x1_kp1  = [x_dat[k+1, 2*m-1] for m in 1:1:M]
# x2_kp1  = [x_dat[k+1, 2*m] for m in 1:1:M]
#
# xδ   = [x1δ_vec x2δ_vec]
# xk   = [x1_k x2_k]
# xkp1 = [x1_kp1 x2_kp1]
#
# # s1_k = cov(x1_k)
# # s2_k = cov(x2_k)
# # s12_k = cov(x1_k, x2_k)
# # s1_kp1 = cov(x1_kp1)
# # s2_kp1 = cov(x2_kp1)
# # s12_kp1 = cov(x1_kp1, x2_kp1)
# #
# # s_δ = [sδ1 sδ12; sδ12' sδ2]
# # s_k = [s1_k s12_k; s12_k' s2_k]
# # s_kp1 = [s1_kp1 s12_kp1; s12_kp1' s2_kp1]
#
# # dims=1 (in contrast to dims=2) means covariance along the columns, i.e. each
# #  row counts as a  realization and each column counts as a stochastic variable
# # dims=1 is the default argument, so we don't have to pass it
#
# s_δ = cov(xδ)
# s_k = cov(xk)
# s_kp1 = cov(xkp1)
# s_δ_k = cov(xδ, xk)
# s_kp1_k = cov(xkp1, xk)
# s_kp1_δ = cov(xkp1, xδ)
#
# Σ = [s_δ s_kp1_δ' s_δ_k; s_kp1_δ s_kp1 s_kp1_k; s_δ_k' s_kp1_k' s_k]
# println(Σ)
#
#
