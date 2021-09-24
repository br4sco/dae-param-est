using ControlSystems
using Plots
using Random
using Distributions
using DelimitedFiles
using ProgressMeter
using LinearAlgebra

struct LinearFilter
  a::Array{Float64}
  b::Array{Float64}
end

function gen_unconditioned_noise(
  A::Array{Float64,2},
  B::Array{Float64,2},
  C::Array{Float64,2},
  Ts::Float64,
  zs::Array{Float64,2},
)

  let
    nx = size(A, 1)
    N = size(zs, 2)
    G = [-A (B*B'); zeros(size(A)) A']
    ws = zeros(N + 1)
    x_prev = zeros(nx, 1)

    for n = 1:N
      F = G * Ts
      expF = exp(F)
      Ad = expF[nx+1:end, nx+1:end]'
      Σ = Ad * expF[1:nx, nx+1:end]
      Bd = cholesky(Hermitian(Σ)).L
      x = Ad * x_prev + Bd * zs[:, n]
      ws[n+1] += first(C * x)
      x_prev = x
    end
    ws
  end
end

function gen_noise_m(gen_noise, ZS)
  M = length(ZS)
  K = size(ZS[1], 2)
  WS = zeros(K + 1, M)
  p = Progress(M, 1, "Generating $(M) noise realizations...", 30)
  Threads.@threads for m = 1:M
    WS[:, m] .+= gen_noise(ZS[m])
    next!(p)
  end
  WS
end

"""
    `gen_unconditioned_noise_ZS(ZS, δ)`

As `gen_unconditioned_noise` but where the white noise is given by `ZS`.
"""
function gen_unconditioned_noise_ZS(
  filter::LinearFilter,
  ZS::Array{Float64,2},
  δ::Float64,
)::Array{Float64,2}
  A, B, C = ss_of_linear_filter(filter())
  nx = size(A, 1)
  WS = gen_noise_m(zs -> gen_unconditioned_noise(A, B, C, δ, zs), ZS)
  WS
end

"""
    `gen_unconditioned_noise(M, δ, K)`

Generates a noise matrix of `M` columns and `K+1` rows, where the noise is white
noise filtered through the linear filter defined by `filter`, exactly
discretized with sampling time `δ` and with inital states all set to zero.
"""
function gen_unconditioned_noise(
  filter::LinearFilter,
  M::Int,
  δ::Float64,
  K::Int,
)::Array{Float64,2}
  A, B, C = ss_of_linear_filter(filter())
  nx = size(A, 1)
  ZS = [rand(Normal(), nx, K) for m = 1:M]
  WS = gen_noise_m(zs -> gen_unconditioned_noise(A, B, C, δ, zs), ZS)
  WS
end

"""
    `gen_unconditioned_noise_1(ZS, δ)`

As `gen_unconditioned_noise_ZS` with linear filter `linear_filter_1`.
"""
gen_unconditioned_noise_1(ZS, δ) =
  gen_unconditioned_noise_ZS(linear_filter_1, ZS, δ)

"""
    `gen_unconditioned_noise_1(M, δ, K)`

As `gen_unconditioned_noise` with linear filter `linear_filter_1`
"""
gen_unconditioned_noise_1(M, δ, K) =
  gen_unconditioned_noise(linear_filter_1, M, δ, K)

function linear_filter_1()
  ω = 4           # natural freq. in rad/s (tunes freq. contents/fluctuations)
  ζ = 0.1          # damping coefficient (tunes damping)
  d1 = 1.0
  d2 = 2 * ω * ζ
  d3 = ω^2
  a = [1.0]
  b = [d1, d2, d3]
  LinearFilter(a, b)
end

function ss_of_linear_filter(f::LinearFilter)
  let s = ss(tf(f.a, f.b))
    s.A, s.B, s.C
  end
end
