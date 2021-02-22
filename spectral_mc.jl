using ControlSystems
using Plots
using Random
using Distributions
using DelimitedFiles
include("noise_interpolation.jl")

struct LinearFilter
  a::Array{Float64}
  b::Array{Float64}
end

function spectral_density(Gw)
  function sd(ω::Float64)::Float64
    Gw(ω*im) * Gw(-ω*im) |> real |> first
  end
end

function mk_spectral_mc_noise_model(Gw, ωmax, dω, M)
  let
    ωs = collect(dω:dω:ωmax)
    K = length(ωs)
    ΦS = rand(K, M) * 2 * pi
    sd = map(spectral_density(Gw), ωs)

    function mk_w(m::Int)
      function w(t::Float64)::Float64
        sum(2 * sqrt.(dω * sd) .* cos.(ωs * t + ΦS[:, m]))
      end
    end
  end
end

function mk_approx_spectral_mc_noise_model(A, B, C, x0, t0, K, M)
  let
    nx = size(A, 1)
    ZS = [rand(Normal(), nx, K) for m in 1:M]
    G = [-A (B * B'); zeros(size(A)) A']

    function mk_w(m)
      let
        prev_x = x0
        prev_t = t0
        prev_k = 0
        zs = ZS[m]
        function w(t::Float64)::Float64
          if t == t0
            return first(C * x0)
          end

          F = G * (t - prev_t)
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

function mk_exact_noise_interpolation_model(A, B, C, Ts, file)
  let
    x_dat = readdlm(file, ',')
    nx = size(A, 1)

    function mk_w(m::Int)
      function w(t::Float64)::Float64
        x = [x_dat[row, (1 + m - 1):(nx + m - 1)] for row in 2:1:size(x_dat, 1)]
        first(C * x_inter(t, Ts, A, B, x))
      end
    end
  end
end

function linear_filter_1()
  ω = 4;       # natural freq. in rad/s (tunes freq. contents/fluctuations)
  ζ = 0.1      # damping coefficient (tunes damping)
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
  d4 = 16.0  # d1, ..., d4 are coeffs of denom.
  n3 = 1.0
  n4 = 1.0 # one zero at -1
  a = [n3, n4]
  b = [d1, d2, d3, d4]
  LinearFilter(a, b)
end

function spectral_mc_noise_model_1(M)
  f = linear_filter_1()
  Gw = tf(f.a, f.b)
  ωmax = 50.0
  dω = 0.01
  mk_spectral_mc_noise_model(Gw, ωmax, dω, M)
end

function spectral_mc_noise_model_2(M)
  f = linear_filter_2()
  Gw = tf(f.a, f.b)
  ωmax = 350.0
  dω = 0.01
  mk_spectral_mc_noise_model(Gw, ωmax, dω, M)
end

function exact_noise_interpolation_model_1()
  let
    Ts = 0.05
    A = [0 -(4^2); 1 -(2 * 4 * 0.1)]
    B = reshape([0.5; 0.0], (2,1))
    C = [1.0 1.0]
    mk_exact_noise_interpolation_model(A, B, C, Ts, "x_mat.csv")
  end
end

function approx_spectral_mc_noise_model_1(t0, K, M)
  f = linear_filter_1()
  sys = ss(tf(f.a, f.b))
  A = sys.A
  B = sys.B
  C = sys.C
  x0 = zeros(size(A, 1))
  mk_approx_spectral_mc_noise_model(A, B, C, x0, t0, K, M)
end

function approx_spectral_mc_noise_model_3(t0, K, M)
  A = [0 -(4^2); 1 -(2*4*0.1)]
  B = [1; 0;]  # replace by Bw = [c; 0;]; where c is the factor tuning the variance of w
  C  = [0 1]
  x0 = zeros(size(A, 1))
  mk_approx_spectral_mc_noise_model(A, B, C, x0, t0, K, M)
end


function plot_noise(w, ts, M)
  let w1 = w(1)
    p = plot(t -> w1(t), ts, legend=false)
    for m = 2:M
      wm = w(m)
      plot!(p, wm, ts)
    end
    p
  end
end

# w2 = exact_noise_interpolation_model_1()
# w1 = spectral_mc_noise_model_1(10)
# w3 = approx_spectral_mc_noise_model_3(0.,10000, 100)
# p1 = plot_noise(w1, 0:0.01:5, 10)
# p2 = plot_noise(w2, 0:0.01:5, 1)
# p3 = plot_noise(w3, 0.0:0.01:5, 100)
