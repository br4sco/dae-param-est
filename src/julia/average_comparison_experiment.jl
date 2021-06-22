include("simulation.jl")
include("noise_generation.jl")
include("noise_interpolation.jl")

seed = 1234
Random.seed!(seed)

const Ts = 0.1
const δ1 = 0.05
const δ2 = 0.5
const factor = Int(δ2/δ1)
const T = 100
const Nw1 = Int(T/δ1)
const Nw2 = Int(T/δ2)
const Nw_extra = 100
const N = Int(T/Ts)
const Q = 500
const W = 100
const M = 500

const nx = 2
const A = [0.0 1.0; -4^2 -0.8]
const B = reshape([0.0 1.0], (2,1))
const C = [1.0 0.0]

const x0 = zeros(nx)
const dmdl1 = discretize_ct_noise_model(A, B, C, δ1, x0)
const dmdl2 = discretize_ct_noise_model(A, B, C, δ2, x0)

const Zu = [randn(Nw1+Nw_extra, nx)]
const Zm = [randn(Nw1+Nw_extra, nx) for m=1:M]
# const Z2 = [Z1[1][1:factor:end, :]]
# const Z2 = [randn(Nw2+Nw_extra, nx)]

mangle_XW(XW::Array{Array{Float64, 1}, 2}) =
  hcat([vcat(XW[:, m]...) for m = 1:size(XW,2)]...)

const XWup  = simulate_noise_process(dmdl1, Zu)
const XWd1p = simulate_noise_process(dmdl1, Zm)
# const XWd2p = simulate_noise_process(dmdl2, Z2)
const XWd2p = XWd1p[1:factor:end, :]
const XWu = mangle_XW(XWup)
const XWd1 = mangle_XW(XWd1p)
const XWd2 = mangle_XW(XWd2p)

function interpx(xl::AbstractArray{Float64, 1},
                 xu::AbstractArray{Float64, 1},
                 t::Float64,
                 δ::Float64,
                 n::Int)

  xl .+ (t - (n - 1) * δ) .* (xu .- xl) ./ δ
end

function mk_noise_interp1(A::Array{Float64, 2},
                             B::Array{Float64, 2},
                             C::Array{Float64, 2},
                             XW::Array{Float64, 2},
                             m::Int)

  let
    nx = size(A, 1)
    function w(t::Float64)
      n = Int(floor(t / δ1)) + 1

      k = (n - 1) * nx + 1

      # xl = view(XW, k:(k + nx - 1), m)
      # xu = view(XW, (k + nx):(k + 2nx - 1), m)

      xl = XW[k:(k + nx - 1), m]
      xu = XW[(k + nx):(k + 2nx - 1), m]

      first(C * interpx(xl, xu, t, δ1, n))
    end
  end
end

function mk_noise_interp2(A::Array{Float64, 2},
                             B::Array{Float64, 2},
                             C::Array{Float64, 2},
                             XW::Array{Float64, 2},
                             m::Int)

  let
    nx = size(A, 1)
    function w(t::Float64)
      n = Int(floor(t / δ2)) + 1

      k = (n - 1) * nx + 1

      # xl = view(XW, k:(k + nx - 1), m)
      # xu = view(XW, (k + nx):(k + 2nx - 1), m)

      xl = XW[k:(k + nx - 1), m]
      xu = XW[(k + nx):(k + 2nx - 1), m]

      first(C * interpx(xl, xu, t, δ2, n))
    end
  end
end

function mk_newer_noise_interp1(A::Array{Float64, 2},
                               B::Array{Float64, 2},
                               C::Array{Float64, 2},
                               isw::InterSampleWindow,
                               m::Int)

   let
       function w(t::Float64)
           xw_temp = noise_inter(t, δ1, A, B, view(XWd1p, :, m), isw)
           return first(C*xw_temp)
       end
   end
end

function mk_noise_interp2(A::Array{Float64, 2},
                             B::Array{Float64, 2},
                             C::Array{Float64, 2},
                             XW::Array{Float64, 2},
                             m::Int)

  let
    nx = size(A, 1)
    function w(t::Float64)
      n = Int(floor(t / δ2)) + 1

      k = (n - 1) * nx + 1

      # xl = view(XW, k:(k + nx - 1), m)
      # xu = view(XW, (k + nx):(k + 2nx - 1), m)

      xl = XW[k:(k + nx - 1), m]
      xu = XW[(k + nx):(k + 2nx - 1), m]
      first(C * interpx(xl, xu, t, δ2, n))
    end
  end
end

function mk_newer_noise_interp2(A::Array{Float64, 2},
                               B::Array{Float64, 2},
                               C::Array{Float64, 2},
                               isw::InterSampleWindow,
                               m::Int)

   let
       function w(t::Float64)
           xw_temp = noise_inter(t, δ2, A, B, view(XWd2p, :, m), isw)
           return first(C*xw_temp)
       end
   end
end

isws = [initialize_isw(Q, W, nx, true) for m=1:M]

wmd1(m::Int) = mk_noise_interp1(A, B, w_scale.*C, XWd1, m)
wmd2(m::Int) = mk_noise_interp2(A, B, w_scale.*C, XWd2, m)
wmn1(m::Int) = mk_newer_noise_interp1(A, B, w_scale.*C, isws[m], m)
wmn2(m::Int) = mk_newer_noise_interp2(A, B, w_scale.*C, isws[m], m)
u(t::Float64) = mk_noise_interp1(A, B, C, XWu, 1)(t)

# === MODEL (AND DATA) PARAMETERS ===
const σ = 0.002                 # observation noise variance
const u_scale = 0.2             # input scale
# const u_scale = 10.0            # input scale larger
const u_bias = 0.0              # input bias
const w_scale = 0.6             # noise scale
# const w_scale = 5.0             # noise scale larger

const m = 0.3                   # [kg]
const L = 6.25                  # [m], gives period T = 5s (T ≅ 2√L) not
                                # accounting for friction.
const g = 9.81                  # [m/s^2]
const k = 0.05                  # [1/s^2]

const φ0 = 0. / 8              # Initial angle of pendulum from negative y-axis

# === OUTPUT FUNCTIONS ===
f(x::Array{Float64, 1}) = x[7]            # applied on the state at each step
h(sol) = apply_outputfun(f, sol)          # for our model
h_baseline(sol) = apply_outputfun(f, sol) # for the baseline method

# === MODEL REALIZATION AND SIMULATION ===
h(sol) = apply_outputfun(f, sol)          # for our model
const θ0 = L                    # true value of θ
# mk_θs(θ::Float64) = [m, L, g, θ]
realize_model(w::Function, N::Int) =
  problem(pendulum(φ0, t -> u_scale * u(t) + u_bias, w, [m, L, g, k]), N, Ts)

  # === SOLVER PARAMETERS ===
  const abstol = 1e-8
  const reltol = 1e-5
  const maxiters = Int64(1e8)

solvew(w::Function, N::Int; kwargs...) =
    solve(realize_model(w, N),
          saveat=0:Ts:(N*Ts),
          abstol = abstol,
          reltol = reltol,
          maxiters = maxiters;
          kwargs...)

calc_y_2e(m::Int) = solvew(t -> wmn2(m)(t), N) |> h
calc_y_2l(m::Int) = solvew(t -> wmd2(m)(t), N) |> h

function get_ys_2le()
    reset_isws!(isws)
    ms = collect(1:M)
    Yme = solve_in_parallel(calc_y_2e, ms)
    Yml = solve_in_parallel(calc_y_2l, ms)
    return Yme, Yml
end

function plot_means_2le(to_show::Int = M)
    Yme, Yml = get_ys_2le()
    Ye = mean(Yme[:, 1:to_show], dims=2)
    Yl = mean(Yml[:, 1:to_show], dims=2)
    times = 0:Ts:(N*Ts)
    pl = plot()
    plot!(times, Ye, label="Exact sampling")
    plot!(times, Yl, label="Linear sampling")
end
