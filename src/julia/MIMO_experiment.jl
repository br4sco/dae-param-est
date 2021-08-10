using DataFrames
using Dierckx
using Distributions
using LsqFit

# include("noise_model.jl")
include("simulation.jl")
include("noise_generation.jl")
include("noise_interpolation_multivar.jl")

seed = 1234
Random.seed!(seed)

# ==================
# === PARAMETERS ===
# ==================

# === TIME ===
# const δ = 0.01                  # noise sampling time
# Used to easier recall from which experiment temp-files were generated
const identifier = "040821"

const N = 10000
const Ts = 0.1                  # stepsize
const T = N*Ts
# const Ts = 0.5                  # stepsize larger

# === NOISE ===
const Q = 1000
# NOTE: Currently the array of isw:s is of length E. If E < M, then one needs to
# create a separate array of isw:s when running M simulations
const M = 500
const E = 500
# const Nws = [50000, 100000, 300000, 500000]
const Nws = [100, 200]
const Nw_max  = maximum(Nws)
const factors = [Int(Nw_max÷Nw) for Nw in Nws]
const δs = [T/Nw for Nw in Nws]
const δ_min = T/Nw_max
const W  = 100
const Nw_extra = 100   # Number of extra samples of noise trajectory to generate

# === PR-GENERATED ===
# const noise_method_name = "Pre-generated unconditioned noise (δ = $(δ))"
const noise_method_name = "Pre-generated conditioned noise (Q = $(Q))"

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
function get_ct_noise_matrices(η::Array{Float64,1}, nx::Int, n_out::Int)
    # First nx parameters of η are parameters for A-matrix. Next comes the noise
    # scale, and finally parameters for C-matrix
    A = diagm(-1 => ones(nx-1,))
    A[1,:] = -η[1:nx]
    B = zeros(nx,1)
    B[1] = 1.0
    # η[nx+1] is the noise scale, we simply multiply it by the C-matrix
    C = diagm(η[nx+1:nx+n_out])*reshape(η[nx+n_out+1:end], n_out, :)
    x0 = zeros(nx)
    return A, B, C, x0
end

const nx = 2        # model order
const n_out = 2     # number of outputs
const n_in = 2      # number of inputs
const n_u  = 1      # number of elements of control input
const w_scale = 0.6*ones(n_out)             # noise scale
# const w_scale = 5.0*ones(n_out)             # noise scale larger
# Denominator of every transfer function is given by p(s), where
# p(s) = s^n + a[1]*s^(n-1) + ... + a[n-1]*s + a[n]
a_true = [0.8, 4^2]
c_true_ = [zeros(nx) for i=1:n_out, j=1:n_in]
# Transfer function (i,j) has numerator c_true_[i,j][1]s^{nx-1} + ... + c_true_[i,j][nx]
c_true_[1,1][:] = vcat(zeros(nx-1), [1])
c_true_[2,2][:] = vcat(zeros(nx-1), [1])
# c_true_ = [zeros(n_out, n_in) for k=1:nx]
# # Numberator of transfter function corresponding to output i and input j is
# # given by b_ij(s) = c[1][i,j]*s^(n-1) + ... + c[n-1][i,j]*s + c[n][i,j]
# c_true_[1] = zeros(n_out,n_in)  # Coefficients of s^(n-1)
# c_true_[2] = diagm(ones(n_out))#ones(n_out,n_in)  # Coefficients of s^(n-2)
#
# c_true contains the same information as c_true_, but in a more suitable structure
c_true = zeros(nx*n_out*n_in)
for i = 1:n_out
    for j = 1:n_in
        for k = 1:nx
            c_true[(j-1)*n_out*nx + (k-1)*n_out + i] = c_true_[i,j][k]
        end
    end
end

# System on controllable canonical form. Note that this is different from
# observable canonical form
const η0 = vcat(a_true, w_scale, c_true)                 # true value of η, should be a 1D-array
const dη = length(η0)
const A_true, B_true, C_true, x0_true = get_ct_noise_matrices(η0, nx, n_out)
const true_mdl = discretize_ct_noise_model(A_true, B_true, C_true, δ_min, x0_true)

to_data(Z::Array{Float64, 2}) =
  [Z[:, m:(m + nx - 1)] for m = 1:nx:(size(Z, 2) / nx)]

read_Z(f::String) = readdlm(joinpath("data", "experiments", f), ',') |>
  transpose |> copy |> to_data

const Zd = [randn(Nw_max + Nw_extra, nx*n_in) for e = 1:E]
const Zm = [randn(Nw_max + Nw_extra, nx*n_in) for m = 1:M]
const Zu = [randn(Nw_max + Nw_extra, nx*n_in)]

mangle_XW(XW::Array{Array{Float64, 1}, 2}) =
  hcat([vcat(XW[:, m]...) for m = 1:size(XW,2)]...)

# const XWdp = simulate_noise_process(dmdl, Zd)
# const XWmp = simulate_noise_process(dmdl, Zm)
const XWup = simulate_multivar_noise_process(true_mdl, Zu, n_in)
const XWdp = simulate_multivar_noise_process(true_mdl, Zd, n_in)
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

function mk_noise_interp(a_vec::AbstractArray{Float64, 1},
                         C::Array{Float64, 2},
                         XW::Array{Float64, 2},
                         m::Int)

  let
    nx = length(a_vec)
    n_tot = size(C, 2)
    function w(t::Float64)
      n = Int(floor(t / δ_min)) + 1

      k = (n - 1) * nx + 1

      # xl = view(XW, k:(k + nx - 1), m)
      # xu = view(XW, (k + nx):(k + 2nx - 1), m)

      xl = XW[k:(k + n_tot - 1), m]
      xu = XW[(k + n_tot):(k + 2n_tot - 1), m]
      C * interpx(xl, xu, t, δ_min, n)
    end
  end
end

function mk_newer_noise_interp(a_vec::AbstractArray{Float64, 1},
                               C::Array{Float64, 2},
                               XWp::Array{Array{Float64, 1}, 2},
                               m::Int,
                               δ::Float64,
                               isws::Array{InterSampleWindow, 1})

   let
       function w(t::Float64)
           xw_temp = noise_inter(t, δ, a_vec, view(XWp, :, m), isws[m])
           return C*xw_temp
       end
   end
end

function mk_newer_noise_interp_m(a_vec::AbstractArray{Float64, 1},
                                 C::Array{Float64, 2},
                                 XWm::Array{Array{Float64, 1}, 2},
                                 m::Int,
                                 δ::Float64,
                                 isws::Array{InterSampleWindow, 1})

   let
       function w(t::Float64)
           xw_temp = noise_inter(t, δ, a_vec, view(XWm, :, m), isws[m])
           return C*xw_temp
       end
   end
end

# === CHOOSE NOISE INTERPOLATION METHOD ===

isws = [initialize_isw(Q, W, nx*n_in, true) for e=1:max(E,M)]
# wmd(e::Int) = mk_newer_noise_interp(a_true, C_true, XWdp, e, isws)
wmd(e::Int) = mk_noise_interp(a_true, C_true, XWd, e)
u(t::Float64) = mk_noise_interp(a_true, C_true[1:n_u,:]./w_scale[1:n_u], XWu, 1)(t) # ./w_scale removes dependence on w_scale

# interpolation over w(tk)
# wmd(e::Int) = interpw(WSd, e)
# wmm(m::Int) = interpw(WSm, m)
# u(t::Float64) = interpw(WSd, M + 1)(t)

# === MODEL (AND DATA) PARAMETERS ===
const σ = 0.002                 # observation noise variance
const u_scale = 0.2*ones(n_u)   # input scale
# const u_scale = 10.0            # input scale larger
const u_bias = 0.0*ones(n_u)    # input bias

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
# const θ0 = [L, L]                    # true value of θ
const θ0 = [L]
const dθ = length(θ0)
# mk_θs(θ::Array{Float64, 1}) = [m, θ[1], g, θ[2]]
mk_θs(θ::Array{Float64, 1}) = [m, θ[1], g, k]
realize_model(w::Function, θ::Array{Float64, 1}, N::Int) =
  problem(pendulum_multivar(φ0, t -> u_scale .* u(t) .+ u_bias, w, mk_θs(θ)), N, Ts)

# # Use this function to specify which parameters should be free and optimized over
# get_all_parameters(θ::Array{Float64,1}) = vcat(θ, η0) # Only θ is optimized over
# get_all_parameters(p::Array{Float64,1}) = p           # All parameters are optimized over
# Optimizes over one parameter in pendulum model and all w_scale parameters
# p should have the form [pendulum_parameter, w_scale_parameters]
get_all_parameters(p::Array{Float64,1}) = vcat(p[1], θ0[2:end], η0[1:nx], p[2:1+n_out], η0[nx+n_out+1:end])

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

# =======================
# === DATA GENERATION ===
# =======================

const data_dir = joinpath("data", "experiments")

exp_path(id) = joinpath(data_dir, id)
mk_exp_dir(id) =  id |> exp_path |> mkdir

calc_y(e::Int) = solvew(t -> wmd(e)(t), θ0, N) |> h_data

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
function model_parametrized(δ, Zm, dummy_input, pars)
    # NOTE: The true input is encoded in the solvew()-function, but this function
    # still needs to to take two input arguments, so dummy_input could just be
    # anything, it's not used anyway
    p = get_all_parameters(pars)
    θ = p[1:dθ]
    η = p[dθ+1: dθ+dη]

    # TODO: Surely we don't need to collect these, a range should work just as well?
    ms = collect(1:M)
    reset_isws!(isws)
    A, B, C, x0 = get_ct_noise_matrices(η, nx, n_out)
    dmdl = discretize_ct_noise_model(A, B, C, δ, x0)
    XWmp = simulate_multivar_noise_process(dmdl, Zm, n_in)
    wmm(m::Int) = mk_newer_noise_interp_m(view(η, 1:nx), C, XWmp, m, δ, isws)
    calc_mean_y_N(N::Int, θ::Array{Float64, 1}, m::Int) =
        solvew(t -> wmm(m)(t), θ, N) |> h
    calc_mean_y(θ::Array{Float64, 1}, m::Int) = calc_mean_y_N(N, θ, m)
    Y = solve_in_parallel(m -> calc_mean_y(θ, m), ms)
    return reshape(mean(Y, dims = 2), :)
end

function perform_experiments(Y, pars0)
    opt_pars = [zeros(length(pars0), E) for i=1:length(Nws)]
    # opt_pars = [zeros(2, E) for i=1:length(Nws)]   # DEBUG
    for k = 1:length(Nws)
        Zm_k = [Zm[j][1:factors[k]:end, :] for j=1:M]
        for e = 1:E
            fit_result, trace = get_fit(Y[:,e], pars0,
                (dummy_input, p) -> model_parametrized(δs[k], Zm_k, dummy_input, p), e)
            # fit_result, trace = get_fit_debug()       # DEBUG
            opt_pars[k][:,e] = coef(fit_result)
            writedlm(joinpath(data_dir, "tmp", "theta_opt_$(k)_$(e)_$(identifier).csv"), coef(fit_result), ',')
        end
    end
    write_opt_pars(opt_pars, "multipar")
    return opt_pars
end

function write_opt_pars(opt_pars, dir_name)
    for k = 1:length(Nws)
        writedlm(joinpath(data_dir, dir_name, "opt_pars_Nw_$(Nws[k]).csv"), opt_pars[k], ',')
    end
end

# NOTE:use coef(fit_result) to get optimal parameter values!!!!!!!!!!!!!
# TODO: This only gets optimal parameters for one out of E realizations, extend it
function get_fit(Ye, pars, model, e)
    # # Use this line if you are using the original LsqFit-package
    # return curve_fit(model, 1:2, Y[:,1], p, show_trace=true)
    # Use this line if you are using the modified LsqFit-package that also
    # returns trace
    fit_result, trace = curve_fit(model, 1:2, Ye, pars, show_trace=true)
    return fit_result, trace
end

# Solves a different optimization problem that is much easier to solve.
# Suitable for quick testing of other functions that use get_fit()
function get_fit_debug()
    @. model(x, p) = p[1]*exp(-x*p[2])
    xdata = range(0, stop=10, length=20)
    ydata = model(xdata, [1.0 2.0]) + 0.01*randn(length(xdata))
    p0 = [0.5, 0.5]
    fit_result, trace = curve_fit(model, xdata, ydata, p0, show_trace=true)
    return fit_result, trace
end

# function calc_mean_Y()
#   ms = collect(1:M)
#   Ym = zeros(N + 1, nθ*nη)
#
#   for (j, η) in enumerate(ηs)
#     @info "solving for point ($(j)/$(nη)) of η"
#
#     A, B, C, x0 = get_ct_noise_matrices(collect(η))
#     dmdl = discretize_ct_noise_model(A, B, C, δ, x0)
#     XWmp = simulate_noise_process(dmdl, Zm)
#     wmm(m::Int) = mk_newer_noise_interp_m(A, B, C, XWmp, m, isws)
#     calc_mean_y_N(N::Int, θ::Array{Float64, 1}, m::Int) =
#         solvew(t -> w_scale * wmm(m)(t), θ, N) |> h
#     calc_mean_y(θ::Array{Float64, 1}, m::Int) = calc_mean_y_N(N, θ, m)
#
#     for (i, θ) in enumerate(θs)
#         @info "solving for point ($(i)/$(nθ)) of θ"
#         println("θ: $θ, η: $η")
#         reset_isws!(isws)
#         Y = solve_in_parallel(m -> calc_mean_y(collect(θ), m), ms)
#         y = reshape(mean(Y, dims = 2), :)
#         writedlm(joinpath(data_dir, "tmp", "y_mean_$(i)_$(j).csv"), y, ',')
#         Ym[:, (j-1)*nθ+i] .+= y
#     end
#   end
#   Ym
# end

# Returns row index (in e.g. Ym) correspodning to θs[i] and ηs[j]
params_to_row(i::Int64, j::Int64) = (j-1)*nθ + 1

# Returns (i,j) so that θs[i] and ηs[j] correspond to row index row (in e.g. Ym)
row_to_params(row::Int64) = ( (row-1)%nθ+1, (row-1)÷nθ+1 )

# function write_mean_Y(expid, Ym)
#   p = joinpath(exp_path(expid), "Ym.csv")
#   writedlm(p, Ym, ",")
# end
#
# function read_mean_Y(expid)
#   p = joinpath(exp_path(expid), "Ym.csv")
#   readdlm(p, ',')
# end
#
# # NOTE: This finds optimal θs and ηs for E experiments for one fixed value of N
# function get_opt_parameters(Y, Ym)
#     costs = zeros(E, nθ*nη)
#     for e = 1:E
#         # for (j, η) in enumerate(ηs)
#         #     for (i, θ) in enumerate(θs)
#         for j in 1:length(ηs)
#             for i in 1:length(θs)
#                 costs[e,(j-1)*nθ+i] = mean(sum( (Y[:,e] - Ym[:,(j-1)*nθ+i]).^2 ))
#             end
#         end
#     end
#     return get_opt_parameters_from_cost(costs)
# end
#
# function get_opt_parameters_from_cost(costs)
#     len = size(costs, 1)
#     opt_args = [argmin(costs[i,:]) for i=1:len]
#     # opt_θs = Array{Tuple{Float64, dθ}}(undef, len)
#     # opt_ηs = Array{Tuple{Float64, dη}}(undef, len)
#     θs_collected = θs |> collect |> vec
#     ηs_collected = ηs |> collect |> vec
#     opt_θs = [opt_theta_helper(opt_args, θs_collected, row) for row=1:len]
#     opt_ηs = [opt_eta_helper(opt_args, ηs_collected, row) for row=1:len]
#     # for row = 1:len
#     #     arg = opt_args[row]
#     #     (i, j) = row_to_params(arg)
#     #     # i = (arg-1)%nθ+1
#     #     # j = (arg-1)÷nθ + 1
#     #     # opt_params[row,:] = [θs_collected[i] ηs_collected[j]]
#     #     opt_θs[row] = θs_collected[i]
#     #     opt_ηs[row] = ηs_collected[j]
#     # end
#     return opt_θs, opt_ηs
# end
#
# function opt_theta_helper(opt_args, θs_collected, row::Int)
#     arg = opt_args[row]
#     (i, j) = row_to_params(arg)
#     return θs_collected[i]
# end
#
# function opt_eta_helper(opt_args, ηs_collected, row::Int)
#     arg = opt_args[row]
#     (i, j) = row_to_params(arg)
#     return ηs_collected[j]
# end
#
# function write_opt_parameters(expid, opt_params)
#     p = joinpath(exp_path(expid), "opt_params.csv")
#     writedlm(p, opt_params, ",")
# end

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
