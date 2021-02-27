using ControlSystems
using Plots
using Random
using Distributions
using DelimitedFiles
include("noise_interpolation.jl")
include("noise_generation.jl")

struct NoiseModelParams
  M::Int
  N::Int
  Ts::Float64
end

struct LinearFilter
  a::Array{Float64}
  b::Array{Float64}
end

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

function mk_discrete_unconditioned_noise_model(A, B, C, x0, K, M, ϵ=10e-6)
  let
    nx = size(A, 1)
    ZS = [rand(Normal(), nx, K) for m in 1:M]
    G = [-A (B * B'); zeros(size(A)) A']

    function mk_w(m)
      let
        prev_x = x0
        prev_t = 0
        prev_k = 0
        zs = ZS[m]
        function w(t::Float64)::Float64

          δ = t - prev_t

          if δ < ϵ
            @info "δ = $(δ) < ϵ at t = $(t)"
            return first(C * x0)
          end

          F = G * δ
          expF = exp(F)
          Ad = expF[nx+1:end, nx+1:end]'
          Σ = Ad * expF[1:nx, nx+1:end]
          Bd = cholesky(Hermitian(Σ)).L
          k = prev_k + 1
          xw = Ad * prev_x + Bd * zs[:, k]

          prev_x = xw
          prev_t = t
          prev_k = k

          return first(C * xw)
        end
      end
    end
  end
end

function mk_exact_noise_interpolation_model(A, B, C, N, x0, Ts, M)
  let
    nx = size(A, 1)
    P = 2
    noise_model = discretize_ct_model(A, B, C, Ts, x0)

    data_uniform, irrelevant_var = generate_noise(N, M, P, nx, false)
    # Computes all M realizations of filtered white noise
    x_mat =
      simulate_noise_process(noise_model, data_uniform)

    function mk_w(m::Int)
      function w(t::Float64)::Float64
        first(C*x_inter(t, Ts, A, B, x_mat[:, m], noise_model.x0))
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

function mk_spectral_mc_noise_model_1(ωmax, dω, M, scale)
  f = linear_filter_1()
  Gw = tf(f.a, f.b)
  mk_spectral_mc_noise_model(Gw, ωmax, dω, M, scale)
end

function spectral_mc_noise_model_1(p::NoiseModelParams)
  ωmax = 50.0
  dω = 0.01
  mk_spectral_mc_noise_model_1(ωmax, dω, p.M, 1.0)
end

function exact_noise_interpolation_model_1(p::NoiseModelParams)
  let
    f = linear_filter_1()
    sys = ss(tf(f.a, f.b))
    A = sys.A
    B = sys.B
    C = sys.C
    x0 = zeros(size(A, 1))
    mk_exact_noise_interpolation_model(A, B, C, p.N, x0, p.Ts, p.M)
  end
end

function discrete_time_noise_model_1(p::NoiseModelParams)
  f = linear_filter_1()
  sys = ss(tf(f.a, f.b))
  A = sys.A
  B = sys.B
  C = sys.C
  x0 = zeros(size(A, 1))
  K = p.N * 1000
  mk_discrete_unconditioned_noise_model(A, B, C, x0, K, p.M)
end
