using ControlSystems
using Plots
using Random

struct SpectralMCNoiseModel
  mk_ZS::Function # Function f(M) which where M is an integer and returns a K×M
                  # matrix where the M columns are noise realizations.
  w::Function  # Function f(z, t) where z is a noise realization, t is the free
               # variable, and returns the disturbance at t.
end

function spectral_density(a::Array{Float64, 1},
                             b::Array{Float64, 1},
                             ωs::Array{Float64, 1})::Array{Float64, 1}
  let Gw = tf(a, b)
    map(ω -> Gw(ω*im) * Gw(-ω*im) |> real |> first, ωs)
  end
end

function spectral_mc_mk_mk_w(a::Array{Float64, 1},
                             b::Array{Float64, 1},
                             ωs::Array{Float64, 1},
                             dω::Float64)::Function

  let
    sd = spectral_density(a, b, ωs)
    function w(φs::Array{Float64, 1}, t::Float64)::Float64
      sum(2 * sqrt.(dω * sd) .* cos.(ωs * t + φs))
    end
  end
end

function mk_spectral_mc_noise_model(a, b, ωmax, dω)
  ωs = collect(dω:dω:ωmax)
  let K = length(ωs)
    function mk_ZS(M::Int)::Array{Float64, 2}
      rand(K, M) * 2 * pi
    end
    SpectralMCNoiseModel(mk_ZS, spectral_mc_mk_mk_w(a, b, ωs, dω))
  end
end

function spectral_mc_noise_model_1()
  ω = 4;       # natural freq. in rad/s (tunes freq. contents/fluctuations)
  ζ = 0.1      # damping coefficient (tunes damping)
  d1 = 1.0
  d2 = 2 * ω * ζ
  d3 = ω^2
  a = [1.0]
  b = [d1, d2, d3]
  ωmax = 50.0
  dω = 0.01
  mk_spectral_mc_noise_model(a, b, ωmax, dω)
end

function spectral_mc_noise_model_2()
  d1 = 4.2641
  d2 = 19.7713
  d3 = 56.2256
  d4 = 16.0  # d1, ..., d4 are coeffs of denom.
  n3 = 1.0
  n4 = 1.0 # one zero at -1
  a = [n3, n4]
  b = [d1, d2, d3, d4]
  ωmax = 350.0
  dω = 0.01
  mk_spectral_mc_noise_model(a, b, ωmax, dω)
end
