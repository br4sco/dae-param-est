using ControlSystems
using Plots
using Random
using Distributions
using DelimitedFiles
using ProgressMeter

include("noise_interpolation.jl")
include("noise_generation.jl")

# struct NoiseModelParams
#   M::Int
#   N::Int
#   Ts::Float64
# end

struct LinearFilter
  a::Array{Float64}
  b::Array{Float64}
end

# mutable struct XW
#   x::Array{Float64, 1}
#   next_x::Array{Float64, 1}
#   t::Float64
#   k::Int
# end

function spectral_density(Gw)
  function sd(ω::Float64)::Float64
    Gw(ω*im) * Gw(-ω*im) |> real |> first
  end
end

function mk_spectral_mc_noise_model(Gw, ωmax, dω, M, scale)
  let
    ωs = collect(dω:dω:ωmax)
    K = length(ωs)
    ΦS = rand(K, M) * 2 * pi
    sd = map(spectral_density(Gw), ωs)

    function mk_w(m::Int)
      function w(t::Float64)::Float64
        scale * sum(2 * sqrt.(dω * sd) .* cos.(ωs * t + ΦS[:, m]))
      end
    end
  end
end

# function mk_discrete_unconditioned_noise_model(A, B, C, K, M, scale, ϵ=10e-25)
#   let
#     nx = size(A, 1)
#     ZS = [rand(Normal(), nx, K) for m in 1:M]
#     G = [-A (B * B'); zeros(size(A)) A']
#
#     function mk_w(xw, m)
#       let
#         zs = ZS[m]
#         function w(t::Float64)::Float64
#
#           δ = t - xw.t
#
#           if δ < ϵ
#             @info "δ = $(δ) < ϵ at t = $(t)"
#             return scale * first(C * xw.x)
#           end
#
#           F = G * δ
#           expF = exp(F)
#           Ad = expF[nx+1:end, nx+1:end]'
#           Σ = Ad * expF[1:nx, nx+1:end]
#           Bd = cholesky(Hermitian(Σ)).L
#           x = Ad * xw.x + Bd * zs[:, xw.k]
#
#           xw.next_x = x
#
#           return scale * first(C * x)
#         end
#       end
#     end
#   end
# end

function gen_unconditioned_noise(
  A::Array{Float64, 2},
  B::Array{Float64, 2},
  C::Array{Float64, 2},
  Ts::Float64,
  zs::Array{Float64, 2})

  let
    nx = size(A, 1)
    N = size(zs, 2)
    G = [-A (B * B'); zeros(size(A)) A']
    ws = zeros(N + 1)
    x_prev = zeros(nx, 1)

    for n = 1:N
      F = G * Ts
      expF = exp(F)
      Ad = expF[nx+1:end, nx+1:end]'
      Σ = Ad * expF[1:nx, nx+1:end]
      Bd = cholesky(Hermitian(Σ)).L
      x = Ad * x_prev + Bd * zs[:, n]
      ws[n + 1] += first(C * x)
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

function write_unconditioned_noise(filter, id, M, δ, T)
  A, B, C = ss_of_linear_filter(filter())
  nx = size(A, 1)
  K = length(0:δ:T)
  ZS = [rand(Normal(), nx, K) for m in 1:M]
  WS = gen_noise_m(zs -> gen_unconditioned_noise(A, B, C, δ, zs), ZS)
  path = joinpath("data", "unconditioned_noise_1_$(id)_$(M)_$(δ)_$(T).csv")
  writedlm(path, WS, ',')
end

function write_unconditioned_noise_1(id, M, δ, T)
  write_unconditioned_noise(linear_filter_1, id, M, δ, T)
end

function read_unconditioned_noise_1(id::Int, M::Int, δ::Float64, T::Float64)::Array{Float64, 2}
  path = joinpath("data", "unconditioned_noise_1_$(id)_$(M)_$(δ)_$(T).csv")
  readdlm(path, ',')
end

function read_unconditioned_noise_1(M, δ, T)
  read_unconditioned_noise(1, M, δ, T)
end

function mk_exact_noise_interpolation_model(A, B, C, N, x0, Ts, M, scale)
  let
    nx = size(A, 1)
    P = 2
    noise_model = discretize_ct_model(A, B, C, Ts, x0)

    data_uniform, irrelevant_var = generate_noise(N, M, P, nx, false)
    # Computes all M realizations of filtered white noise
    x_mat = simulate_noise_process(noise_model, data_uniform)

    function mk_w(m::Int)
      function w(t::Float64)::Float64
        scale * first(C*x_inter(t, Ts, A, B, x_mat[:, m], noise_model.x0))
      end
    end
  end
end

function linear_filter_1()
  ω = 4;           # natural freq. in rad/s (tunes freq. contents/fluctuations)
  ζ = 0.1          # damping coefficient (tunes damping)
  d1 = 1.0
  d2 = 2 * ω * ζ
  d3 = ω^2
  a = [1.0]
  b = [d1, d2, d3]
  LinearFilter(a, b)
end

function linear_filter_2()
  d1 = 4.2641
  d2 = 19.7713
  d3 = 56.2256
  d4 = 16.0                     # d1, ..., d4 are coeffs of denom.
  n3 = 1.0
  n4 = 1.0                      # one zero at -1
  a = [n3, n4]
  b = [d1, d2, d3, d4]
  LinearFilter(a, b)
end

function ss_of_linear_filter(f::LinearFilter)
  let s = ss(tf(f.a, f.b))
    s.A, s.B, s.C
  end
end

function mk_spectral_mc_noise_model_1(ωmax, dω, M, scale)
  f = linear_filter_1()
  Gw = tf(f.a, f.b)
  mk_spectral_mc_noise_model(Gw, ωmax, dω, M, scale)
end

function exact_noise_interpolation_model_1(N, Ts, M, scale)
  let
    f = linear_filter_1()
    sys = ss(tf(f.a, f.b))
    A = sys.A
    B = sys.B
    C = sys.C
    x0 = zeros(size(A, 1))
    mk_exact_noise_interpolation_model(A, B, C, N, x0, Ts, M, scale)
  end
end

function discrete_time_noise_model_1(K, M, scale)
  f = linear_filter_1()
  sys = ss(tf(f.a, f.b))
  A = sys.A
  B = sys.B
  C = sys.C
  x0 = zeros(size(A, 1), )
  mk_discrete_unconditioned_noise_model(A, B, C, K, M, scale)
end

# N = 100
# δ = 1e-9
# δ = 0.05
# w = mk_spectral_mc_noise_model_1(50, 0.01, 1, 1.0)(1)
# w = exact_noise_interpolation_model_1(N, 0.05, 1, 1.0)(1)
# plot(0:δ:N*δ, w, xlabel="time [s]", title = "spectral, δ = $(δ)")
