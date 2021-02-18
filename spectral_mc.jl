using ControlSystems
using Plots
using Random
include("noise_interpolation.jl")

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

    function w(m::Int, t::Float64)::Float64
      sum(2 * sqrt.(dω * sd) .* cos.(ωs * t + ΦS[:, m]))
    end
  end
end

function mk_discrete_time_dist_model(Gw, K, M)
  let ZS = rand(K, M)
    work()
  end
end

function mk_exact_noise_interpolation_model(A, B, C, ΔT, file)
  let
    x_dat = readdlm(file, ',')
    nx = size(A, 1)

    function w(m::Int, t::Float64)::Float64
      x = [x_dat[row, (1 + m - 1):(nx + m - 1)] for row in 2:1:size(x_dat, 1)]
      first(C * x_inter(t, ΔT, A, B, x))
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
  tf(a, b)
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
  tf(a, b)
end

function spectral_mc_noise_model_1(M)
  Gw = linear_filter_1()
  ωmax = 50.0
  dω = 0.01
  mk_spectral_mc_noise_model(Gw, ωmax, dω, M)
end

function spectral_mc_noise_model_2(M)
  Gw = linear_filter_2()
  ωmax = 350.0
  dω = 0.01
  mk_spectral_mc_noise_model(Gw, ωmax, dω, M)
end

function exact_noise_interpolation_model_1(ΔT)
  let
    A = [-1.0 0.0; 0.0 -2.0]
    B = [0.3 0.0; 0.0 0.5]
    C = [1.0 1.0]
    mk_exact_noise_interpolation_model(A, B, C, ΔT, "x_mat.csv")
  end
end
