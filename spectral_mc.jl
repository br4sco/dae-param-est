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
  a = [1.0 , 1.0]
  b = [1., sqrt(12.0), 2.0]
  ωmax = 100.0
  dω = 0.01
  mk_spectral_mc_noise_model(a, b, ωmax, dω)
end

function spectral_mc_noise_model_2()
  a = [1.0, 0.0]
  b = [1.0, sqrt(12.0), 2.0, 0.1, 3.0]
  ωmax = 100.0
  dω = 0.01
  mk_spectral_mc_noise_model(a, b, ωmax, dω)
end

function spectral_mc_noise_model_3()
  a = [1.]
  b = [1.0, 2.0 * 4.0 * 0.1, 4^2]
  ωmax = 50.0
  dω = 0.02
  mk_spectral_mc_noise_model(a, b, ωmax, dω)
end
