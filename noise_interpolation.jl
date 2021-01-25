import Random

using LinearAlgebra
using CSV, DataFrames


Random.seed!(1234)
const A = [-1 0; 0 -2]
const B = [1 0; 0 0.5]
const C = [1 1]

const t0 = 0            # Initial time of noise model simulation
const Ts = 0.05         # Sampling frequency of noise model
const N  = 100          # Number of simulated time steps of noise model

x_dat = CSV.read("x_mat.csv", DataFrame)
# x[k] contains the state vector from time t0 + Ts*k
x  = [ [x_dat[row, 1]; x_dat[row, 2]] for row in 1:1:size(x_dat)[1]]

function w(t::Float64)
    k = Int((t-t0)÷Ts)        # t lies between t0 + k*Ts and t0 + (k+1)*Ts
    δ = t - (t0 + k*Ts)
    n = size(A)[1]

    Mexp    = [A B*B'; zeros(size(A)) A']
    Mδ      = exp(Mexp*δ)
    MTs_δ   = exp(Mexp*(Ts-δ))
    Adδ     = Mδ[1:n, 1:n]
    AdTs_δ  = MTs_δ[1:n, 1:n]
    Bd2δ    = Mδ[1:n, n+1:end]*Adδ
    Bd2Ts_δ = MTs_δ[1:n, n+1:end]*AdTs_δ

    Cδ      = cholesky(Bd2δ)        # Might need to wrap matrices in Hermitian()
    CTs_δ   = cholesky(Bd2Ts_δ)
    Bdδ     = Cδ.L
    BdTs_δ  = CTs_δ.L

    zkδ = randn(Float64, (n, 1)) # Samples noise contribution at time Ts*k + δ

    Bdz = AdTs_δ\( x[k+1] - BdTs_δ*zkδ) - Adδ*x[k]

    wkδ = (C*( Adδ*x[k] + Bdz))[1]      # The desired disturbance sample

    return wkδ
end

# DEBUG
t = 0.07
wt = w(t)
println(wt)
