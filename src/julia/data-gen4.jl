using DataFrames
using Dierckx
using Distributions
using LsqFit

# include("noise_model.jl")
include("simulation.jl")
include("noise_generation.jl")
include("noise_interpolation.jl")

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
# NOTE: Currently the array of isw:s is of length E. If E < M, then one needs to
# create a separate array of isw:s when running M simulations
const M = 500
const E = 500
const Nw = 100
const W  = 100
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
get_A(η::Array{Float64,1}) = [η[1] η[2]; 1.0 0.0]
# get_A(η::Array{Float64,1}) = [0.0 1.0; -4^2 η[1]]

function get_ct_noise_matrices(η)
    A = get_A(η)
    B = reshape([0.0 1.0], (2,1))
    C = [1.0 0.0]
    x0 = zeros(nx)
    return A, B, C, x0
end

const nx = 2
# System on controllable canonical form. Note that this is different from
# observable canonical form
const η0 = [-0.8, -4.0^2]                 # true value of η, should be a 1D-array
const A_true = get_A(η0)
const B_true = reshape([1.0 0.0], (2,1))
const C_true = [0.0 1.0]
const x0_true = zeros(nx)
const true_mdl = discretize_ct_noise_model(A_true, B_true, C_true, δ, x0_true)

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

# const XWdp = simulate_noise_process(dmdl, Zd)
# const XWmp = simulate_noise_process(dmdl, Zm)
const XWup = simulate_noise_process(true_mdl, Zu)
const XWdp = simulate_noise_process(true_mdl, Zd)
const XWd = mangle_XW(XWdp)
const XWu = mangle_XW(XWup)
# const XWm = mangle_XW(XWmp)


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

function mk_noise_interp(A::Array{Float64, 2},
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
            x = μ + Σr*white_noise
            if num_stored_samples < Q
                isd.states[n+1] = [isd.states[n+1]; x']
                isd.sample_times[n+1] = [isd.sample_times[n+1]; t]
            end
            return first(C * x)
        end
      end
end

function mk_newer_noise_interp(A::Array{Float64, 2},
                               B::Array{Float64, 2},
                               C::Array{Float64, 2},
                               XWp::Array{Array{Float64, 1}, 2},
                               m::Int,
                               isws::Array{InterSampleWindow, 1})

   let
       function w(t::Float64)
           xw_temp = noise_inter(t, δ, A, B, view(XWp, :, m), isws[m])
           return first(C*xw_temp)
       end
   end
end

function mk_newer_noise_interp_m(A::Array{Float64, 2},
                                 B::Array{Float64, 2},
                                 C::Array{Float64, 2},
                                 XWm::Array{Array{Float64, 1}, 2},
                                 m::Int,
                                 isws::Array{InterSampleWindow, 1})

   let
       function w(t::Float64)
           xw_temp = noise_inter(t, δ, A, B, view(XWm, :, m), isws[m])
           return first(C*xw_temp)
       end
   end
end

# === CHOOSE NOISE INTERPOLATION METHOD ===

# isd = initialize_isd(100, Nw+1, nx, true)
isds = [initialize_isd(Q, Nw+1, nx, true) for e=1:E]
isws = [initialize_isw(Q, W, nx, true) for e=1:E]
wmd(e::Int) = mk_newer_noise_interp(A_true, B_true, C_true, XWdp, e, isws)
# wmn(e::Int) = mk_newer_noise_interp(A, B, C, XWdp, e, isws)
# wmnm(m::Int) = mk_newer_noise_interp_m(A, B, C, XWmp, m, isws)
# wmm(m::Int) = mk_noise_interp(A, B, C, XWm, m)
u(t::Float64) = mk_noise_interp(A_true, B_true, C_true, XWu, 1)(t)

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
const θ0 = [L, L]                    # true value of θ
# const θ0 = [L]
mk_θs(θ::Array{Float64, 1}) = [m, θ[1], g, θ[2]]
# mk_θs(θ::Tuple{Vararg{Float64}}) = [m, θ[1], g, k]
realize_model(w::Function, θ::Array{Float64, 1}, N::Int) =
  problem(pendulum(φ0, t -> u_scale * u(t) + u_bias, w, mk_θs(θ)), N, Ts)

# === SOLVER PARAMETERS ===
const abstol = 1e-8
const reltol = 1e-5
const maxiters = Int64(1e8)

solvew(w::Function, θ::Array{Float64, 1}, N::Int; kwargs...) =
  solve(realize_model(w, θ, N),
        saveat=0:Ts:(N*Ts),
        abstol = abstol,
        reltol = reltol,
        maxiters = maxiters;
        kwargs...)

# dataset output function
h_data(sol) = apply_outputfun(x -> f(x) + σ * rand(Normal()), sol)

# === EXPERIMENT PARAMETERS ===
const lnθ = [0, 0]                  # number of steps in the left interval
const rnθ = [1, 1]                  # number of steps in the right interval
const δθ = [0.2, 0.01]
const dθ = length(θ0)
# See https://stackoverflow.com/questions/22873478/how-to-pass-tuple-as-function-arguments
# for explanation of ..., it basically turns elements of vector into sequence
# of arguments
const θs = Iterators.product(
    [(θ0[i] - lnθ[i]*δθ[i]):δθ[i]:(θ0[i] + rnθ[i]*δθ[i]) for i=1:dθ]...)
const nθ = length(θs)

const lnη = [0, 0]                  # number of steps in the left interval
const rnη = [1, 1]                  # number of steps in the right interval
const δη = [0.2, 0.2]
const dη = length(η0)
const ηs = Iterators.product(
    [(η0[i] - lnη[i]*δη[i]):δη[i]:(η0[i] + rnη[i]*δη[i]) for i=1:dη]...)
const nη = length(ηs)

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

function write_custom(expid, file_name, data)
    p = joinpath(exp_path(expid), file_name)
    writedlm(p, data, ",")
end

function read_custom(expid, file_name)
    p = joinpath(exp_path(expid), file_name)
    readdlm(p, ',')
end

function write_Y(expid, Y)
  p = joinpath(exp_path(expid), "Yd1.csv")
  writedlm(p, Y, ",")
end

function read_Y(expid)
  p = joinpath(exp_path(expid), "Yd1.csv")
  readdlm(p, ',')
end

calc_baseline_y_N(N::Int, θ::Array{Float64, 1}) = solvew(t -> 0., θ, N) |> h_baseline

calc_baseline_y(θ::Array{Float64, 1}) = calc_baseline_y_N(N, θ)

calc_baseline_Y() = solve_in_parallel(calc_baseline_y, θs)

function write_theta(expid)
  p = joinpath(exp_path(expid), "theta.csv")
  writedlm(p, θs, ",")
end

function read_theta(expid)
  p = joinpath(exp_path(expid), "theta.csv")
  readdlm(p, ',')
end

function write_baseline_Y(expid, Yb)
  p = joinpath(exp_path(expid), "Yb.csv")
  writedlm(p, Yb, ",")
end

function read_baseline_Y(expid)
  p = joinpath(exp_path(expid), "Yb.csv")
  readdlm(p, ',')
end

# model-function used by get_fit()
function model(dummy_input, p)
    # NOTE: The true input is encoded in the solvew()-function, but this function
    # still needs to to take two input arguments, so dummy_input could just be
    # anything, it's not used anyway
    θ = p[1:dθ]
    η = p[dθ+1: dθ+dη]

    # TODO: Surely we don't need to collect these, a range should work just as well?
    ms = collect(1:M)
    reset_isws!(isws)
    A, B, C, x0 = get_ct_noise_matrices(η)
    dmdl = discretize_ct_noise_model(A, B, C, δ, x0)
    XWmp = simulate_noise_process(dmdl, Zm)
    wmm(m::Int) = mk_newer_noise_interp_m(A, B, C, XWmp, m, isws)
    calc_mean_y_N(N::Int, θ::Array{Float64, 1}, m::Int) =
        solvew(t -> w_scale * wmm(m)(t), θ, N) |> h
    calc_mean_y(θ::Array{Float64, 1}, m::Int) = calc_mean_y_N(N, θ, m)
    Y = solve_in_parallel(m -> calc_mean_y(θ, m), ms)
    return reshape(mean(Y, dims = 2), :)
end

# NOTE:use coef(fit_result) to get optimal parameter values!!!!!!!!!!!!!
# TODO: This only gets optimal parameters for one out of E realizations, extend it
function get_fit(Y, θi, ηi)
    p = vcat(θi, ηi)
    # # Use this line if you are using the original LsqFit-package
    # return curve_fit(model, 1:2, Y[:,1], p, show_trace=true)
    # Use this line if you are using the modified LsqFit-package that also
    # returns trace
    fit_result, trace = curve_fit(model, 1:2, Y[:,1], p, show_trace=true)
    return fit_result, trace
end

function calc_mean_Y()
  ms = collect(1:M)
  Ym = zeros(N + 1, nθ*nη)

  for (j, η) in enumerate(ηs)
    @info "solving for point ($(j)/$(nη)) of η"

    A, B, C, x0 = get_ct_noise_matrices(collect(η))
    dmdl = discretize_ct_noise_model(A, B, C, δ, x0)
    XWmp = simulate_noise_process(dmdl, Zm)
    wmm(m::Int) = mk_newer_noise_interp_m(A, B, C, XWmp, m, isws)
    calc_mean_y_N(N::Int, θ::Array{Float64, 1}, m::Int) =
        solvew(t -> w_scale * wmm(m)(t), θ, N) |> h
    calc_mean_y(θ::Array{Float64, 1}, m::Int) = calc_mean_y_N(N, θ, m)

    for (i, θ) in enumerate(θs)
        @info "solving for point ($(i)/$(nθ)) of θ"
        println("θ: $θ, η: $η")
        reset_isws!(isws)
        Y = solve_in_parallel(m -> calc_mean_y(collect(θ), m), ms)
        y = reshape(mean(Y, dims = 2), :)
        writedlm(joinpath(data_dir, "tmp", "y_mean_$(i)_$(j).csv"), y, ',')
        Ym[:, (j-1)*nθ+i] .+= y
    end
  end
  Ym
end

# Returns row index (in e.g. Ym) correspodning to θs[i] and ηs[j]
params_to_row(i::Int64, j::Int64) = (j-1)*nθ + 1

# Returns (i,j) so that θs[i] and ηs[j] correspond to row index row (in e.g. Ym)
row_to_params(row::Int64) = ( (row-1)%nθ+1, (row-1)÷nθ+1 )

function write_mean_Y(expid, Ym)
  p = joinpath(exp_path(expid), "Ym.csv")
  writedlm(p, Ym, ",")
end

function read_mean_Y(expid)
  p = joinpath(exp_path(expid), "Ym.csv")
  readdlm(p, ',')
end

# NOTE: This finds optimal θs and ηs for E experiments for one fixed value of N
function get_opt_parameters(Y, Ym)
    costs = zeros(E, nθ*nη)
    for e = 1:E
        # for (j, η) in enumerate(ηs)
        #     for (i, θ) in enumerate(θs)
        for j in 1:length(ηs)
            for i in 1:length(θs)
                costs[e,(j-1)*nθ+i] = mean(sum( (Y[:,e] - Ym[:,(j-1)*nθ+i]).^2 ))
            end
        end
    end
    return get_opt_parameters_from_cost(costs)
end

function get_opt_parameters_from_cost(costs)
    len = size(costs, 1)
    opt_args = [argmin(costs[i,:]) for i=1:len]
    # opt_θs = Array{Tuple{Float64, dθ}}(undef, len)
    # opt_ηs = Array{Tuple{Float64, dη}}(undef, len)
    θs_collected = θs |> collect |> vec
    ηs_collected = ηs |> collect |> vec
    opt_θs = [opt_theta_helper(opt_args, θs_collected, row) for row=1:len]
    opt_ηs = [opt_eta_helper(opt_args, ηs_collected, row) for row=1:len]
    # for row = 1:len
    #     arg = opt_args[row]
    #     (i, j) = row_to_params(arg)
    #     # i = (arg-1)%nθ+1
    #     # j = (arg-1)÷nθ + 1
    #     # opt_params[row,:] = [θs_collected[i] ηs_collected[j]]
    #     opt_θs[row] = θs_collected[i]
    #     opt_ηs[row] = ηs_collected[j]
    # end
    return opt_θs, opt_ηs
end

function opt_theta_helper(opt_args, θs_collected, row::Int)
    arg = opt_args[row]
    (i, j) = row_to_params(arg)
    return θs_collected[i]
end

function opt_eta_helper(opt_args, ηs_collected, row::Int)
    arg = opt_args[row]
    (i, j) = row_to_params(arg)
    return ηs_collected[j]
end

function write_opt_parameters(expid, opt_params)
    p = joinpath(exp_path(expid), "opt_params.csv")
    writedlm(p, opt_params, ",")
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