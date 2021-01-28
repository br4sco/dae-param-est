using ControlSystems
using Plots
using Random

struct NoiseModel
  K::Int
  w::Function
end

function spectral_density(a::Array{Float64, 1},
                          b::Array{Float64, 1},
                          ωs::Array{Float64, 1})::Array{Float64, 1}
  let Gw = tf(a, b)
    map(ω -> Gw(ω*im) * Gw(-ω*im) |> real |> first, ωs)
  end
end

function spectral_mc_noise(a::Array{Float64, 1},
                           b::Array{Float64, 1},
                           ωs::Array{Float64, 1},
                           dω::Float64)::Function

  let
    sd = spectral_density(a, b, ωs)
    function w(zs::Array{Float64, 1}, t::Float64)::Float64
      φs = zs * 2 * pi
      sum(2 * sqrt.(dω * sd) .* cos.(ωs * t + φs))
    end
    w
  end
end

const SPECTRAL_MC_NOISE_1 = begin
  a = [1., 0.]
  b = [1., sqrt(12.), 2.]
  ωmax = 100.0
  dω = 0.01
  ωs = collect(dω:dω:ωmax)
  K = length(ωs)
  NoiseModel(K, spectral_mc_noise(a, b, ωs, dω))
end

const SPECTRAL_MC_NOISE_2 = begin
  a = [1., 0.]
  b = [1., sqrt(12.), 2., 0.1, 3.]
  ωmax = 100.0
  dω = 0.01
  ωs = collect(dω:dω:ωmax)
  K = length(ωs)
  NoiseModel(K, spectral_mc_noise(a, b, ωs, dω))
end

# z = rand(spectral_mc_noise1.K)
# plot(0:0.01:100, t -> spectral_mc_noise1.mkW(z, t))
