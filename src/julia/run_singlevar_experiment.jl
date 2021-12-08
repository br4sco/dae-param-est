using DataFrames, LsqFit, Statistics
include("simulation.jl")
include("noise_interpolation_multivar.jl")
include("noise_generation.jl")

seed = 1234
Random.seed!(seed)

# ==================
# === PARAMETERS ===
# ==================

# === TIME ===
# The relationship between number of data samples N and number of noise samples
# Nw is given by Nw >= (Ts/δ)*N
const δ = 0.01                  # noise sampling time
const Ts = 0.1                  # step-size
const M = 500       # Number of monte-carlo simulations used for estimating mean
# TODO: Surely we don't need to collect these, a range should work just as well?
const ms = collect(1:M)
const W = 100           # Number of intervals for which isw stores data
const Q = 1000          # Number of conditional samples stored per interval

const noise_method_name = "Pre-generated unconditioned noise (δ = $(δ))"

mangle_XW(XW::Array{Array{Float64, 1}, 2}) =
  hcat([vcat(XW[:, m]...) for m = 1:size(XW,2)]...)

demangle_XW(XW::Array{Float64, 2}, n_tot::Int) =
    [XW[(i-1)*n_tot+1:i*n_tot, m] for i=1:(size(XW,1)÷n_tot), m=1:size(XW,2)]

# We do linear interpolation between exact values because it's fast
# function interpw(WS::Array{Float64,2}, m::Int)
#   function w(t::Float64)
#     k = Int(floor(t / δ)) + 1
#     w0 = WS[k, m]
#     w1 = WS[k+1, m]
#     w0 + (t - (k - 1) * δ) * (w1 - w0) / δ
#   end
# end

# n is the dimension of one sample of the state
function interpw(W::Array{Float64, 2}, m::Int, n::Int)
    function w(t::Float64)
        # tₖ = kδ <= t <= (k+1)δ = tₖ₊₁
        # => The rows corresponding to tₖ start at index k*n+1
        # since t₀ = 0 corrsponds to block starting with row 1
        k = Int(t÷δ)
        w0 = W[k*n+1:(k+1)*n, m]
        w1 = W[(k+1)*n+1:(k+2)*n, m]
        return w0 + (t-k*δ)*(w1-w0)/δ
    end
end

# function interpx(XW::Array{Float64, 2}, C::Array{Float64, 2}, m::Int, n::Int)
#     function xw(t::Float64)
#         # k = Int(t÷δ)
#         # xw0 = XW[k*n+1:(k+1)*n, m]
#         # xw1 = XW[(k+1)*n+1:(k+2)*n, m]
#         # return C*(xw0 + (t-k*δ)*(xw1-xw0)/δ)
#         return C*interpw(XW, m, n)
#     end
# end

# This function relies on XW being mangled
function mk_noise_interp(C::Array{Float64, 2},
                         XW::Array{Float64, 2},
                         m::Int,
                         δ::Float64)

  let
    n_tot = size(C, 2)
    function w(t::Float64)
        # n*δ <= t <= (n+1)*δ
        n = Int(t÷δ)
        # row of x_1(t_n) in XW is given by k. Note that t=0 is given by row 1
        k = n * n_tot + 1

        # xl = view(XW, k:(k + nx - 1), m)
        # xu = view(XW, (k + nx):(k + 2nx - 1), m)

        xl = XW[k:(k + n_tot - 1), m]
        xu = XW[(k + n_tot):(k + 2n_tot - 1), m]
        return C*(xl + (t-n*δ)*(xu-xl)/δ)
    end
  end
end

# Function for using conditional interpolation
function mk_newer_noise_interp(a_vec::AbstractArray{Float64, 1},
                               C::Array{Float64, 2},
                               XW::Array{Array{Float64, 1}, 2},
                               m::Int,
                               n_in::Int,
                               δ::Float64,
                               isws::Array{InterSampleWindow, 1})
   # Conditional sampling depends on noise model, which is why a value of
   # a_vec has to be passed. a_vec contains parameters corresponding to the
   # A-matric of the noise model
   let
       function w(t::Float64)
           xw_temp = noise_inter(t, δ, a_vec, n_in, view(XW, :, m), isws[m])
           return C*xw_temp
       end
   end
end

# NOTE:use coef(fit_result) to get optimal parameter values
function get_fit(Ye, pars, model)
    # # Use this line if you are using the original LsqFit-package
    # return curve_fit(model, 1:2, Y[:,1], p, show_trace=true)
    # Use this line if you are using the modified LsqFit-package that also
    # returns trace
    fit_result, trace = curve_fit(model, 1:2, Ye, pars, show_trace=true)
    return fit_result, trace
end

# === MODEL (AND DATA) PARAMETERS ===
const σ = 0.002                 # measurement noise variance
const u_scale = 0.2             # input scale
const u_bias = 0.0              # input bias

const m = 0.3                   # [kg]
const L = 6.25                  # [m], gives period T = 5s (T ≅ 2√L) not
# accounting for friction.
const g = 9.81                  # [m/s^2]
const k = 6.25                  # [1/s^2]

const φ0 = 0.0                   # Initial angle of pendulum from negative y-axis

# === OUTPUT FUNCTIONS ===
# The state vector x from the solver is organized as follows:
# x = [
#   x1              -- position in the x-direction
#   x1'             -- velocity in the x-direction
#   x2              -- position in the y-direction
#   x2'             -- velocity in the y-direction
#   int(x3)         -- integral of the tension per unit length (due to stabilized formulation)
#   int(dummy)      -- integral of dummy variable (due to stabilized formulation)
#   y               -- the output y = atan(x1/-x2) is computed by the solver
# ]
f(x::Array{Float64,1}) = x[7]               # applied on the state at each step
h(sol) = apply_outputfun(f, sol)            # for our model
h_baseline(sol) = apply_outputfun(f, sol)   # for the baseline method

# === HELPER FUNCTIONS TO READ AND WRITE DATA
const data_dir = joinpath("data", "experiments")

# create directory for this experiment
exp_path(expid) = joinpath(data_dir, expid)

data_Y_path(expid) = joinpath(exp_path(expid), "Y.csv")
# data_input_path(expid) = joinpath(exp_path(expid), "U.csv")
# data_XW_path(expid) = joinpath(exp_path(expid), "XW.csv")
# metadata_W_path(expid) = joinpath(exp_path(expid), "W_meta.csv")
# metadata_input_path(expid) = joinpath(exp_path(expid), "U_meta.csv")

# === MODEL REALIZATION AND SIMULATION ===
const pars_true = [k]                    # true value of θ
get_all_θs(pars::Array{Float64,1}) = [m, L, g, pars[1]]
const dθ = length(get_all_θs(pars_true))
realize_model(u::Function, w::Function, pars::Array{Float64, 1}, N::Int) = problem(
  pendulum(φ0, t -> u_scale * u(t) .+ u_bias, w, get_all_θs(pars)),
  N,
  Ts,
)

# === SOLVER PARAMETERS ===
const abstol = 1e-8
const reltol = 1e-5
const maxiters = Int64(1e8)

solvew(u::Function, w::Function, pars::Array{Float64, 1}, N::Int; kwargs...) = solve(
  realize_model(u, w, pars, N),
  saveat = 0:Ts:(N*Ts),
  abstol = abstol,
  reltol = reltol,
  maxiters = maxiters;
  kwargs...,
)

# data-set output function
h_data(sol) = apply_outputfun(x -> f(x) + σ * randn(), sol)

function get_estimates(expid, pars0::Array{Float64,1}, N_trans::Int = 0)

    Y, N, Nw, W_meta, isws, u, get_all_ηs = get_Y_and_u(expid)
    nx = Int(W_meta.nx[1])
    n_in = Int(W_meta.n_in[1])
    n_out = Int(W_meta.n_out[1])
    n_tot = nx*n_in
    # Parameters of true system
    η_true = W_meta.η
    dη = length(η_true)
    a_vec = η_true[1:nx]
    C_true = reshape(η_true[nx+1:end], (n_out, n_tot))

    get_all_parameters(pars::Array{Float64, 1}) = vcat(get_all_θs(pars[1:1]), get_all_ηs(Float64[]))

    # === We then optimize parameters for the baseline model ===
    function baseline_model_parametrized(δ, dummy_input, pars)
        # NOTE: The true input is encoded in the solvew()-function, but this function
        # still needs to to take two input arguments, so dummy_input could just be
        # anything, it's not used anyway
        Y_base = solvew(u, t -> zeros(n_out), pars, N ) |> h

        # TODO: Currently assumes scalar output from DAE-system
        return reshape(Y_base[N_trans+1:end,:], :)   # Returns 1D-array
    end

    E = size(Y, 2)
    # #DEBUG
    # E = 1
    opt_pars_baseline = zeros(length(pars0), E)
    for e=1:E
        baseline_result, baseline_trace = get_fit(Y[:,e], pars0,
            (dummy_input, pars) -> baseline_model_parametrized(δ, dummy_input, pars))
        opt_pars_baseline[:, e] = coef(baseline_result)
    end

    @info "The mean optimal parameters for baseline are given by: $(mean(opt_pars_baseline, dims=2))"

    # === Finally we optimize parameters for the proposed model ==
    function proposed_model_parametrized(δ, Zm, dummy_input, pars, isws)
        # NOTE: The true input is encoded in the solvew()-function, but this function
        # still needs to to take two input arguments, so dummy_input could just be
        # anything, it's not used anyway
        p = get_all_parameters(pars)
        η = p[dθ+1: dθ+dη]
        C = reshape(η[nx+1:end], (n_out, n_tot))

        dmdl = discretize_ct_noise_model(get_ct_disturbance_model(η, nx, n_out), δ)
        # # NOTE: OPTION 1: Use the rows below here for linear interpolation
        XWm = simulate_noise_process_mangled(dmdl, Zm)
        wmm(m::Int) = mk_noise_interp(C, XWm, m, δ)
        # NOTE: OPTION 2: Use the rows below here for exact interpolation
        # reset_isws!(isws)
        # XWm = simulate_noise_process(dmdl, Zm)
        # wmm(m::Int) = mk_newer_noise_interp(view(η, 1:nx), C, XWm, m, n_in, δ, isws)

        calc_mean_y_N(N::Int, pars::Array{Float64, 1}, m::Int) =
            solvew(u, t -> wmm(m)(t), pars, N) |> h
        calc_mean_y(pars::Array{Float64, 1}, m::Int) = calc_mean_y_N(N, pars, m)
        Ym = solve_in_parallel(m -> calc_mean_y(pars, m), ms)
        return reshape(mean(Ym[N_trans+1:end,:], dims = 2), :) # Returns 1D-array
    end

    # TODO: Should we consider storing and loading white noise, to improve repeatability
    # of the results? Currently repeatability is based on fixed random seed
    Zm = [randn(Nw, n_tot) for m = 1:M]

    opt_pars_proposed = zeros(length(pars0), E)
    for e=1:E
        proposed_result, proposed_trace = get_fit(Y[:,e], pars0,
            (dummy_input, pars) -> proposed_model_parametrized(δ, Zm, dummy_input, pars, isws))
        opt_pars_proposed[:, e] = coef(proposed_result)
    end


    @info "The mean optimal parameters for proposed method are given by: $(mean(opt_pars_proposed, dims=2))"

    return (opt_pars_baseline, opt_pars_proposed)
end

function get_outputs(expid, pars0::Array{Float64,1})

    Y, N, Nw, W_meta, isws, u, get_all_ηs = get_Y_and_u(expid)
    nx = Int(W_meta.nx[1])
    n_in = Int(W_meta.n_in[1])
    n_out = Int(W_meta.n_out[1])
    n_tot = nx*n_in
    # Parameters of true system
    η_true = W_meta.η
    dη = length(η_true)
    a_vec = η_true[1:nx]
    C_true = reshape(η_true[nx+1:end], (n_out, n_tot))

    # === Computes output of the baseline model ===
    Y_base = solvew(u, t -> zeros(n_out), pars_true, N) |> h

    # === Computes outputs of the proposed model ===
    # TODO: Should we consider storing and loading white noise, to improve repeatability
    # of the results? Currently repeatability is based on fixed random seed
    Zm = [randn(Nw, n_tot) for m = 1:M]
    dmdl = discretize_ct_noise_model(get_ct_disturbance_model(η_true, nx, n_out), δ)
    # NOTE: OPTION 1: Use the rows below here for linear interpolation
    XWm = simulate_noise_process_mangled(dmdl, Zm)
    wmm(m::Int) = mk_noise_interp(C_true, XWm, m, δ)
    # NOTE: OPTION 2: Use the rows below here for exact interpolation
    # reset_isws!(isws)
    # XWm = simulate_noise_process(dmdl, Zm)
    # wmm(m::Int) = mk_newer_noise_interp(view(η_true, 1:nx), C, XWm, m, n_in, δ, isws)

    calc_mean_y_N_prop(N::Int, pars::Array{Float64, 1}, m::Int) =
        solvew(u, t -> wmm(m)(t), pars, N) |> h
    calc_mean_y_prop(θ::Array{Float64, 1}, m::Int) = calc_mean_y_N_prop(N, θ, m)
    Ym_prop = solve_in_parallel(m -> calc_mean_y_prop(pars_true, m), ms)
    Y_mean_prop = reshape(mean(Ym_prop, dims = 2), :)

    return Y, Y_base, Ym_prop, Y_mean_prop
end

# TODO: Surely there must be a nicer way to avoid code repetition...?!!
function get_Y_and_u(expid)::Tuple{Array{Float64,2}, Int, Int, DataFrame, Array{InterSampleWindow, 1}, Function, Function}
    # A single realization of the disturbance serves as input
    # input is assumed to contain the input signal, and not the state
    input  = readdlm(joinpath(data_dir, expid*"/U.csv"), ',')
    XW     = readdlm(joinpath(data_dir, expid*"/XW.csv"), ',')
    W_meta_raw, W_meta_names =
        readdlm(joinpath(data_dir, expid*"/meta_W.csv"), ',', header=true)
    W_meta = DataFrame(W_meta_raw, Symbol.(W_meta_names[:]))
    nx = Int(W_meta.nx[1])
    n_in = Int(W_meta.n_in[1])
    n_out = Int(W_meta.n_out[1])
    n_tot = nx*n_in
    # Parameters of true system
    η_true = W_meta.η
    dη = length(η_true)
    a_vec = η_true[1:nx]
    C_true = reshape(η_true[nx+1:end], (n_out, n_tot))

    # Use this function to specify which parameters should be free and optimized over
    get_all_ηs(pars::Array{Float64, 1}) = η_true

    mk_we(XW::Array{Array{Float64, 1},2}, isws::Array{InterSampleWindow, 1}) =
        (m::Int) -> mk_newer_noise_interp(
        a_vec::AbstractArray{Float64, 1}, C_true, XW, m, n_in, δ, isws)
    u(t::Float64) = interpw(input, 1, 1)(t)

    # compute the maximum number of steps we can take
    N_margin = 2    # Solver can request values of inputs after the time horizon
                    # ends, so we require a margin of a few samples of the noise
                    # to ensure that we can provide such values
    # Minimum of number of available disturbance or input samples
    Nw = min(size(XW, 1)÷n_tot, size(input, 1)÷n_in)
    N = Int((Nw - N_margin)*δ÷Ts)     # Number of steps we can take

    # === We first generate the output of the true system ===
    function calc_Y(XW::Array{Array{Float64, 1},2}, isws::Array{InterSampleWindow, 1})
        @assert (Nw <= size(XW, 1)) "Disturbance data size mismatch ($(Nw) > $(size(XW, 1)))"
        E = size(XW, 2)
        es = collect(1:E)
        we = mk_we(XW, isws)
        solve_in_parallel(e -> solvew(u, we(e), pars_true, N) |> h_data, es)
    end

    if isfile(data_Y_path(expid))
        @info "Reading output of true system"

        Y = readdlm(data_Y_path(expid), ',')
        isws = [initialize_isw(Q, W, n_tot, true) for e=1:M]
    else
        @info "Generating output of true system"
        isws = [initialize_isw(Q, W, n_tot, true) for e=1:max(size(XW,2), M)]
        Y = calc_Y(demangle_XW(XW, n_tot), isws)
        writedlm(data_Y_path(expid), Y, ',')
    end

    reset_isws!(isws)
    # TODO: Can we bake N and Nw into W_meta somehow?
    return Y, N, Nw, W_meta, isws, u, get_all_ηs
end

function plot_cost_function(expid, par_ind::Int, par_range::Tuple{Float64, Float64}, fixed_pars::Array{Float64}, step_size::Float64=0.1)
    Y, N, Nw, W_meta, isws, u, get_all_ηs = get_Y_and_u(expid)
    nx = Int(W_meta.nx[1])
    n_in = Int(W_meta.n_in[1])
    n_out = Int(W_meta.n_out[1])
    n_tot = nx*n_in
    dη = length(W_meta.η)
    E = size(Y, 2)
    get_all_parameters(pars::Array{Float64, 1}) = vcat(get_all_θs(pars), get_all_ηs(pars))
    par_values = par_range[1]:step_size:par_range[2]
    par_matrix = [get_all_parameters(vcat(fixed_pars[1:par_ind-1], par_values[i], fixed_pars[par_ind+1:end])) for i=1:length(par_values)]
    cost_vals = zeros(length(par_values), E)

    Zm = [randn(Nw, n_tot) for m = 1:M]
    for i=1:length(par_values)
        p = get_all_parameters(par_matrix[i])
        θ = p[1:dθ]
        η = p[dθ+1: dθ+dη]
        C = reshape(η[nx+1:end], (n_out, n_tot))

        dmdl = discretize_ct_noise_model(get_ct_disturbance_model(η, nx, n_out), δ)
        # NOTE: OPTION 1: Use the rows below here for linear interpolation
        XWm = simulate_noise_process_mangled(dmdl, Zm)
        wmm(m::Int) = mk_noise_interp(C, XWm, m, δ)
        # # NOTE: OPTION 2: Use the rows below here for exact interpolation
        # reset_isws!(isws)
        # XWm = simulate_noise_process(dmdl, Zm)
        # wmm(m::Int) = mk_newer_noise_interp(view(η_true, 1:nx), C, XWm, m, n_in, δ, isws)

        calc_mean_y_N_prop(N::Int, pars::Array{Float64, 1}, m::Int) =
            solvew_sens(u, t -> wmm(m)(t), pars, N) |> h_sens
        calc_mean_y_prop(pars::Array{Float64, 1}, m::Int) = calc_mean_y_N_prop(N, pars, m)
        Ym_prop, sens_m_prop = solve_in_parallel_sens(m -> calc_mean_y_prop(par_matrix[i], m), ms)
        Y_mean_prop = reshape(mean(Ym_prop, dims = 2), :)

        for e = 1:E
            cost_vals[i, e] = mean((Y[:,e]-Y_mean_prop).^2)
        end
    end

    p = plot(par_values, cost_vals[:,1])
    display(p)

    return cost_vals
end

function plot_outputs(Y_ref::Array{Float64, 1}, Y_base::Array{Float64, 1}, Y_prop::Array{Float64, 1}, Y_mean_prop::Array{Float64, 1})
    t = 1:size(Y_ref, 1)
    mse_base = round(mean((Y_ref-Y_base).^2), sigdigits=3)
    mse_prop= round(mean((Y_ref-Y_prop).^2), sigdigits=3)
    mse_mean = round(mean((Y_ref-Y_mean_prop).^2), sigdigits=3)
    p = plot(t, Y_ref, label="Reference")
    plot!(p, Y_base, label="Baseline ($(mse_base))")
    plot!(p, Y_prop, label="Proposed ($(mse_prop))")
    plot!(p, Y_mean_prop, label="Proposed Mean ($(mse_mean))")
end
