using DataFrames
using Dierckx
using Distributions

# include("noise_model.jl")
include("simulation.jl")
include("noise_generation.jl")
include("new_noise_interpolation.jl")

seed = 1234
Random.seed!(seed)

# ==================
# === PARAMETERS ===
# ==================

# === TIME ===
const δ = 0.01                  # noise sampling time
const Ts = 0.1                  # stepsize
# const Ts = 0.5                  # stepsize larger

# === NOISE ===
const Q = 100
const M = 500
const E = 500
const Nw = 10000
const Nw_extra = 100   # Number of extra samples of noise trajectory to generate

# === PR-GENERATED ===
# const noise_method_name = "Pre-generated unconditioned noise (δ = $(δ))"
const noise_method_name = "Pre-generated conditioned noise (δ = $(δ), Q = $(Q))"

# We do linear interpolation between exact values because it's fast
function interpw(WS::Array{Float64, 2}, m::Int)
  function w(t::Float64)
    k = Int(floor(t / δ)) + 1
    w0 = WS[k, m]
    w1 = WS[k + 1, m]
    w0 + (t - (k - 1) * δ) * (w1 - w0) / δ
  end
end

# const tsw = collect(0:δ:(δ * Nw))
# interpw(WS::Array{Float64, 2}, m::Int) =
#   Spline1D(tsw, view(WS, 1:(Nw+1), m); k=2, bc="error", s=0.0)

# === PRE-GENERATED DATA ===
# const WSd =
#   readdlm(joinpath("data",
#                    "experiments",
#                    "unconditioned_noise_data_501_001_250000_1234_alsvin.csv"),
#           ',')

# const WSu =
#   readdlm(joinpath("data",
#                    "experiments",
#                    "unconditioned_noise_input_001_250000_1234_alsvin.csv"),
#           ',')

# const WSm =
#   readdlm(joinpath("data",
#                    "experiments",
#                    "unconditioned_noise_model_500_001_250000_1234_alsvin.csv"),
#           ',')

# === NOISE INTERPOLATION ===
const nx = 2
const A = [0.0 1.0; -4^2 -0.8]
const B = reshape([0.0 1.0], (2,1))
const C = [1.0 0.0]
# const A = [0 -4^2; 1 -0.8]
# const B = reshape([1.0 0.0], (2,1))
# const C = [0 1]
const x0 = zeros(nx)
const dmdl = discretize_ct_noise_model(A, B, C, δ, x0)

to_data(Z::Array{Float64, 2}) =
  [Z[:, m:(m + nx - 1)] for m = 1:nx:(size(Z, 2) / nx)]

read_Z(f::String) = readdlm(joinpath("data", "experiments", f), ',') |>
  transpose |> copy |> to_data

# const Zd = read_Z("Zd_501_25_1234.csv")
# const Zm = read_Z("Zm_500_25_1234.csv")
# const Zu = read_Z("Zu_25_1234.csv")
# const Nw = min(size(Zd[1], 1), size(Zm[1], 1), size(Zu[1], 1))

const Zd = [randn(Nw + Nw_extra, nx) for e = 1:E]
const Zm = [randn(Nw + Nw_extra, nx) for m = 1:M]
const Zu = [randn(Nw + Nw_extra, nx)]

mangle_XW(XW::Array{Array{Float64, 1}, 2}) =
  hcat([vcat(XW[:, m]...) for m = 1:size(XW,2)]...)

const XWd = simulate_noise_process_new(dmdl, Zd) |> mangle_XW
const XWm = simulate_noise_process_new(dmdl, Zm) |> mangle_XW
const XWu = simulate_noise_process_new(dmdl, Zu) |> mangle_XW


# new noise interpolation optimization attempt
function interpx(xl::AbstractArray{Float64, 1},
                 xu::AbstractArray{Float64, 1},
                 t::Float64,
                 δ::Float64,
                 n::Int)

  xl .+ (t - (n - 1) * δ) .* (xu .- xl) ./ δ
end

function interpx_general(xl::AbstractArray{Float64, 1},
                 xu::AbstractArray{Float64, 1},
                 t::Float64,
                 tl::Float64,
                 tu::Float64)

  xl .+ (t - tl) .* (xu .- xl) ./ (tu-tl)
end

function interpw_general(wl::Float64,
                 wu::Float64,
                 t::Float64,
                 tl::Float64,
                 tu::Float64)

  wl + (t - tl) * (wu - wl) / (tu-tl)
end

function mk_new_noise_interp(A::Array{Float64, 2},
                             B::Array{Float64, 2},
                             C::Array{Float64, 2},
                             XW::Array{Float64, 2},
                             m::Int)

  let
    nx = size(A, 1)
    function w(t::Float64)
      n = Int(floor(t / δ)) + 1

      k = (n - 1) * nx + 1

      # xl = view(XW, k:(k + nx - 1), m)
      # xu = view(XW, (k + nx):(k + 2nx - 1), m)

      xl = XW[k:(k + nx - 1), m]
      xu = XW[(k + nx):(k + 2nx - 1), m]

      first(C * interpx(xl, xu, t, δ, n))
    end
  end
end

function mk_other_noise_interp(A::Array{Float64, 2},
                               B::Array{Float64, 2},
                               C::Array{Float64, 2},
                               XW::Array{Float64, 2},
                               m::Int,
                               isds::Array{InterSampleData, 1},
                               ϵ::Float64=10e-7,
                               rng::MersenneTwister=Random.default_rng())
    let
        nx = size(A, 1)
        function w(t::Float64)
            n = Int(floor(t / δ))
            isd = isds[m]
            Q = isd.Q
            # XW[j+nx*i, m] is element j of the state at time i of realization m
            # i=0,...,Nw
            # n = Int(t÷δ) + 1
            k = n * nx + 1
            # xl = XW[k:(k + nx - 1), m]
            # xu = XW[(k + nx):(k + 2nx - 1), m]
            xl = view(XW, k:(k + nx - 1), m)
            xu = view(XW, (k + nx):(k + 2nx - 1), m)
            tl = n*δ
            tu = (n+1)*δ
            il = 0      # for il>0,   tl = isd.sample_times[n][il]
            iu = Q+1    # for iu<Q+1, tu = isd.sample_times[n][iu]

            if Q == 0 && isd.use_interpolation
                # @warn "Used linear interpolation"
                # return first(C * interpx_general(xl, xu, t, tl, tu))
                return interpw_general(first(C*xl), first(C*xu), t, tl, tu)
            end

            # N = size(isd.states)[1]
            num_stored_samples = size(isd.states[n+1])[1]
            # setting il, tl, iu, tu
            if num_stored_samples > 0
                for q = 1:num_stored_samples
                    # interval index = n+1
                    t_inter = isd.sample_times[n+1][q]
                    if t_inter > tl && t_inter <= t
                        tl = t_inter
                        il = q
                    end
                    if t_inter < tu && t_inter >= t
                        tu = t_inter
                        iu = q
                    end
                end
            end
            δl = t-tl
            δu = tu-t

            # Setting xl and xu
            if il > 0
                # xl = isd.states[n+1][il,:]
                xl = view(isd.states[n+1], il,:)
            end
            if iu < Q+1
                # xu = isd.states[n+1][iu,:]
                xu = view(isd.states[n+1], iu,:)
            end
            if num_stored_samples >= Q && isd.use_interpolation
                # @warn "Used linear interpolation"   # DEBUG
                # return first(C * interpx_general(xl, xu, t, tl, tu))
                return interpw_general(first(C*xl), first(C*xu), t, tl, tu)
            end

            # Values of δ smaller than ϵ are treated as 0
            if δl < ϵ
                return first(C*xl)
            elseif δu < ϵ
                return first(C*xu)
            end

            Mexp    = [-A B*(B'); zeros(size(A)) A']
            Ml      = exp(Mexp*δl)
            Mu      = exp(Mexp*δu)
            Adl     = Ml[nx+1:end, nx+1:end]'
            Adu     = Mu[nx+1:end, nx+1:end]'
            AdΔ     = Adu*Adl
            B2dl    = Hermitian(Adl*Ml[1:nx, nx+1:end])
            B2du    = Hermitian(Adu*Mu[1:nx, nx+1:end])
            Cl      = cholesky(B2dl)
            Cu      = cholesky(B2du)
            Bdl     = Cl.L
            Bdu     = Cu.L

            σ_l = (Bdl*(Bdl'))
            σ_u = (Bdu*(Bdu'))
            σ_Δ = Adu*σ_l*(Adu') + σ_u
            σ_Δ_l = Adu*σ_l
            v_Δ = xu - AdΔ*xl
            μ = Adl*xl + (σ_Δ_l')*(σ_Δ\v_Δ) # TODO: Might want to double-check that this also covers n=0, but it seems to be the case
            # Hermitian()-call might not be necessary, but it probably depends on the
            # model, so I leave it in to ensure that cholesky decomposition will work
            Σ = Hermitian(σ_l - (σ_Δ_l')*(σ_Δ\(σ_Δ_l)))
            CΣ = Σ
            CΣ = cholesky(Σ)
            # try
            #     CΣ = cholesky(Σ)
            # catch err
            #     println(CΣ)
            #     println("δl: $δl, δu: $δu, n: $n, N: $N")
            #     throw(err)
            # end
            Σr = CΣ.L

            white_noise = randn(rng, Float64, (nx, 1))
            x_new = μ + Σr*white_noise
            if num_stored_samples < Q
                isd.states[n+1] = [isd.states[n+1]; x_new']
                isd.sample_times[n+1] = [isd.sample_times[n+1]; t]
            end
            return first(C * x_new)
        end
      end
end

# === CHOOSE NOISE INTERPOLATION METHOD ===

# isd = initialize_isd(100, Nw+1, nx, true)
isds = [initialize_isd(Q, Nw+1, nx, true) for e=1:E]
# new interpolation optimization attempt
# wmd(e::Int) = mk_new_noise_interp(A, B, C, XWd, e)
wmd(e::Int) = mk_other_noise_interp(A, B, C, XWd, e, isds)
wmm(m::Int) = mk_new_noise_interp(A, B, C, XWm, m)
u(t::Float64) = mk_new_noise_interp(A, B, C, XWu, 1)(t)

# interpolation over w(tk)
# wmd(e::Int) = interpw(WSd, e)
# wmm(m::Int) = interpw(WSm, m)
# u(t::Float64) = interpw(WSd, M + 1)(t)

# === MODEL ===
# we compute the maximum number of steps we can take
# const K = min(size(WSd, 1), size(WS, 1)) - 2
const N = Int(floor(Nw * δ / Ts))
# const N = 10000

# number of realizations in the model
# const M = size(WS, 2)

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
const θ0 = L                    # true value of θ
mk_θs(θ::Float64) = [m, L, g, θ]
realize_model(w::Function, θ::Float64, N::Int) =
  problem(pendulum(φ0, t -> u_scale * u(t) + u_bias, w, mk_θs(θ)), N, Ts)

# === SOLVER PARAMETERS ===
const abstol = 1e-8
const reltol = 1e-5
const maxiters = Int64(1e8)

solvew(w::Function, θ::Float64, N::Int; kwargs...) =
  solve(realize_model(w, θ, N),
        saveat=0:Ts:(N*Ts),
        abstol = abstol,
        reltol = reltol,
        maxiters = maxiters;
        kwargs...)

# dataset output function
h_data(sol) = apply_outputfun(x -> f(x) + σ * rand(Normal()), sol)

# === EXPERIMENT PARAMETERS ===
const lnθ = 3                  # number of steps in the left interval
const rnθ = 3                  # number of steps in the right interval
const δθ = 0.2
const θs = (θ0 - lnθ * δθ):δθ:(θ0 + rnθ * δθ) |> collect
const nθ = length(θs)

# =======================
# === DATA GENERATION ===
# =======================

const data_dir = joinpath("data", "experiments")

exp_path(id) = joinpath(data_dir, id)
mk_exp_dir(id) =  id |> exp_path |> mkdir

calc_y(e::Int) = solvew(t -> w_scale * wmd(e)(t), θ0, N) |> h_data

function calc_Y()
  es = collect(1:E)
  solve_in_parallel(calc_y, es)
end

function write_Y(expid, Y)
  p = joinpath(exp_path(expid), "Yd1.csv")
  writedlm(p, Y, ",")
end

function read_Y(expid)
  p = joinpath(exp_path(expid), "Yd1.csv")
  readdlm(p, ",")
end

calc_baseline_y_N(N::Int, θ::Float64) = solvew(t -> 0., θ, N) |> h_baseline

calc_baseline_y(θ::Float64) = calc_baseline_y_N(N, θ)

calc_baseline_Y() = solve_in_parallel(calc_baseline_y, θs)

function write_theta(expid)
  p = joinpath(exp_path(expid), "theta.csv")
  writedlm(p, θs, ",")
end

function read_theta(expid)
  p = joinpath(exp_path(expid), "theta.csv")
  readdlm(p, ",")
end

function write_baseline_Y(expid, Yb)
  p = joinpath(exp_path(expid), "Yb.csv")
  writedlm(p, Yb, ",")
end

function read_baseline_Y(expid)
  p = joinpath(exp_path(expid), "Yb.csv")
  readdlm(p, ",")
end

calc_mean_y_N(N::Int, θ::Float64, m::Int) =
  solvew(t -> w_scale * wmm(m)(t), θ, N) |> h

calc_mean_y(θ::Float64, m::Int) = calc_mean_y_N(N, θ, m)

function calc_mean_Y()
  ms = collect(1:M)
  Ym = zeros(N + 1, nθ)

  for (i, θ) in enumerate(θs)
    @info "solving for point ($(i)/$(nθ)) of θ"
    Y = solve_in_parallel(m -> calc_mean_y(θ, m), ms)
    y = reshape(mean(Y, dims = 2), :)
    writedlm(joinpath(data_dir, "tmp", "y_mean_$(i).csv"), y, ',')
    Ym[:, i] .+= y
  end
  Ym
end

function write_mean_Y(expid, Ym)
  p = joinpath(exp_path(expid), "Ym.csv")
  writedlm(p, Ym, ",")
end

function read_mean_Y(expid)
  p = joinpath(exp_path(expid), "Ym.csv")
  readdlm(p, ",")
end

function write_meta_data(expid)
  p = joinpath(exp_path(expid), "meta_data.csv")
  df = DataFrame(Ts = Ts,
                 σ = σ,
                 φ0 = φ0,
                 θ0 = θ0,
                 u_scale = u_scale,
                 w_scale = w_scale,
                 noise_method_name = noise_method_name,
                 seed = seed,
                 atol = abstol,
                 rtol = reltol)
  CSV.write(p, df)
end

function read_meta_data(expid)
  p = joinpath(exp_path(expid), "meta_data.csv")
  CSV.File(p) |> DataFrame
end
