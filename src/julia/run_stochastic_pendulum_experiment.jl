using LsqFit, StatsPlots, LaTeXStrings
include("simulation.jl")
include("noise_interpolation_multivar.jl")
include("noise_generation.jl")

seed = 1234
Random.seed!(seed)

struct ExperimentData
    # Y is the measured output of system, contains N+1 rows and E columns
    # (+1 because of an additional sample at t=0)
    Y::Array{Float64,2}
    # u is the function specifying the input of the system
    u::Function
    # get_all_ηs encodes what information of the disturbance model is known
    # This function should always return all parameters of the disturbance model,
    # given only the free parameters
    get_all_ηs::Function
    # Array containing lower and upper bound of a disturbance parameter in each row
    dist_par_bounds::Array{Float64, 2}
    # W_meta is the metadata of the disturbance, containting e.g. dimensions
    W_meta::DisturbanceMetaData
    # Nw is a lower bound on the number of available samples of the disturbance and input
    # N, the number of data samples, have been computed based on this number
    Nw::Int
end

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

M_rate_max = 8
# max_allowed_step = 1.0  # Maximum magnitude of step that SGD is allowed to take
# M_rate(t) specifies over how many realizations the gradient estimate should be
# computed at iteration t
M_rate(t::Int) = M_rate_max
# This learning rate works very well with M_rate = 8
learning_rate(t::Int, grad_norm::Float64) = if (t < 50) 1/grad_norm else 1/(grad_norm*(1+1*(t-50) )) end

mangle_XW(XW::Array{Array{Float64, 1}, 2}) =
  hcat([vcat(XW[:, m]...) for m = 1:size(XW,2)]...)

demangle_XW(XW::Array{Float64, 2}, n_tot::Int) =
    [XW[(i-1)*n_tot+1:i*n_tot, m] for i=1:(size(XW,1)÷n_tot), m=1:size(XW,2)]

# NOTE: The realizaitons Ym and jacYm must be independent for this to return
# an unbiased estimate of the cost function gradient
function get_cost_jacobian(Y::Array{Float64,1}, Ym::Array{Float64,2}, jacYm::Array{Float64,2})
    (2/(size(Y,1)-1))*
        sum(
        mean(Y.-Ym, dims=2)
        .*mean(-jacYm, dims=2)
        , dims=1)
end

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

# function interpx(XW::Array{Float64, 2}, C::Array{Float64, 2}, m::Int)
#     function xw(t::Float64)
#         # k = Int(t÷δ)
#         # xw0 = XW[k*n+1:(k+1)*n, m]
#         # xw1 = XW[(k+1)*n+1:(k+2)*n, m]
#         # return C*(xw0 + (t-k*δ)*(xw1-xw0)/δ)
#         n = size(C,2)
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
    # return curve_fit(model, 1:2, Y[:,1], p, show_trace=true, inplace = false, x_tol = 1e-8)    # Default inplace = false, x_tol = 1e-8

    # HACK: Uses trace returned due to hacked LsqFit package
    # Use this line if you are using the modified LsqFit-package that also
    # returns trace
    fit_result, trace = curve_fit(model, Float64[], Ye, pars, show_trace=true, inplace=false, x_tol=1e-8)    # Default inplace = false, x_tol = 1e-8
    return fit_result, trace
end

# Uses a jacobian model instead of estimating jacobian from forward difference
function get_fit_sens(Ye, pars, model, jacobian_model)
    # HACK: Uses trace returned due to hacked LsqFit package
    # Use this line if you are using the modified LsqFit-package that also
    # returns trace
    fit_result, trace = curve_fit(model, jacobian_model, Float64[], Ye, pars, show_trace=true, inplace=false, x_tol=1e-8)    # Default inplace = false, x_tol = 1e-8
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
f_sens(x::Array{Float64,1}) = x[14]   # NOTE: Hard-coded right now
h(sol) = apply_outputfun(f, sol)                            # for our model
h_sens(sol) = apply_two_outputfun(f, f_sens, sol)           # for our model with sensitivity
h_baseline(sol) = apply_outputfun(f, sol)                   # for the baseline method
h_baseline_sens(sol) = apply_two_outputfun(f, f_sens, sol)  # for the baseline method with sensitivity

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
# We use the following naming conventions for parameters:
# θ: All parameters of the dynamical model
# η: All parameters of the disturbance model
# pars: All free parameters
# p: vcat(θ, η)
const pars_true = [k]                    # true value of all free parameters
get_all_θs(pars::Array{Float64,1}) = [m, L, g, pars[1]]
dyn_par_bounds = [0.1 Inf]    # Lower and upper bounds of each free dynamic parameter
const dθ = length(get_all_θs(pars_true))
realize_model_sens(u::Function, w::Function, pars::Array{Float64, 1}, N::Int) = problem(
  pendulum_sensitivity(φ0, t -> u_scale * u(t) .+ u_bias, w, get_all_θs(pars)),
  N,
  Ts,
)
realize_model(u::Function, w::Function, pars::Array{Float64, 1}, N::Int) = problem(
  pendulum_new(φ0, t -> u_scale * u(t) .+ u_bias, w, get_all_θs(pars)),
  N,
  Ts,
)

# === SOLVER PARAMETERS ===
const abstol = 1e-8
const reltol = 1e-5
const maxiters = Int64(1e8)

solvew_sens(u::Function, w::Function, pars::Array{Float64, 1}, N::Int; kwargs...) = solve(
  realize_model_sens(u, w, pars, N),
  saveat = 0:Ts:(N*Ts),
  abstol = abstol,
  reltol = reltol,
  maxiters = maxiters;
  kwargs...,
)
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

    exp_data, isws = get_experiment_data(expid)
    W_meta = exp_data.W_meta
    u = exp_data.u
    Y = exp_data.Y
    N = size(Y,1)-1
    Nw = exp_data.Nw
    nx = W_meta.nx
    n_in = W_meta.n_in
    n_out = W_meta.n_out
    n_tot = nx*n_in
    dη = length(W_meta.η)

    get_all_parameters(pars::Array{Float64, 1}) = vcat(get_all_θs(pars), exp_data.get_all_ηs(pars))
    par_bounds = vcat(dyn_par_bounds, exp_data.dist_par_bounds)

    # === We then optimize parameters for the baseline model ===
    function baseline_model_parametrized(δ, dummy_input, pars)
        # NOTE: The true input is encoded in the solvew_sens()-function, but this function
        # still needs to to take two input arguments, so dummy_input could just be
        # anything, it's not used anyway
        # Y_base = solvew(u, t -> zeros(n_out), pars, N ) |> h_baseline


        # temp, temp_sens = solvew_sens(u, t -> zeros(n_out), pars, N )
        Y_base = solvew(u, t -> zeros(n_out), pars, N ) |> h_baseline

        # TODO: Currently assumes scalar output from DAE-system
        return reshape(Y_base[N_trans+1:end,:], :)   # Returns 1D-array
    end

    function get_baseline_gradient(y, δ, pars)
        p = get_all_parameters(pars)
        η = p[dθ+1: dθ+dη]
        C = reshape(η[nx+1:end], (n_out, n_tot))

        _, gradYm = solvew_sens(u, t -> zeros(n_out), pars, N) |> h_sens
        gradYm
    end
    jacobian_model_b(x, p) = get_baseline_gradient(Y[:,1], δ, p)

    E = size(Y, 2)
    # # DEBUG
    # E = 1
    opt_pars_baseline = zeros(length(pars0), E)
    trace_base = [Float64[] for e=1:E]
    for e=1:E
        push!(trace_base[e], pars0[1]) # NOTE: Only works for scalar parameter
        # HACK: Uses trace returned due to hacked LsqFit package
        # baseline_result, baseline_trace = get_fit(Y[:,e], pars0,
        #     (dummy_input, pars) -> baseline_model_parametrized(δ, dummy_input, pars))
        baseline_result, baseline_trace = get_fit_sens(Y[:,e], pars0,
            (dummy_input, pars) -> baseline_model_parametrized(δ, dummy_input, pars), jacobian_model_b)
        opt_pars_baseline[:, e] = coef(baseline_result)
        if length(baseline_trace) > 1
            for j=2:length(baseline_trace)
                push!(trace_base[e], baseline_trace[j].metadata["x"][1])
            end
        end
    end

    @info "The mean optimal parameters for baseline are given by: $(mean(opt_pars_baseline, dims=2))"

    # === Finally we optimize parameters for the proposed model ==

    # function proposed_model_parametrized(δ, Zm, dummy_input, pars, isws)
    #     # NOTE: The true input is encoded in the solvew_sens()-function, but this function
    #     # still needs to to take two input arguments, so dummy_input could just be
    #     # anything, it's not used anyway
    #     p = get_all_parameters(pars)
    #     θ = p[1:dθ]
    #     η = p[dθ+1: dθ+dη]
    #     C = reshape(η[nx+1:end], (n_out, n_tot))
    #
    #     dmdl = discretize_ct_noise_model(get_ct_disturbance_model(η, nx, n_out), δ)
    #     # # NOTE: OPTION 1: Use the rows below here for linear interpolation
    #     XWm = simulate_noise_process_mangled(dmdl, Zm)
    #     wmm(m::Int) = mk_noise_interp(C, XWm, m, δ)
    #     # NOTE: OPTION 2: Use the rows below here for exact interpolation
    #     # reset_isws!(isws)
    #     # XWm = simulate_noise_process(dmdl, Zm)
    #     # wmm(m::Int) = mk_newer_noise_interp(view(η, 1:nx), C, XWm, m, n_in, δ, isws)
    #
    #     calc_mean_y_N(N::Int, pars::Array{Float64, 1}, m::Int) =
    #         solvew(u, t -> wmm(m)(t), pars, N) |> h
    #     calc_mean_y(pars::Array{Float64, 1}, m::Int) = calc_mean_y_N(N, pars, m)
    #     Ym = solve_in_parallel(m -> calc_mean_y(pars, m), ms)
    #     return reshape(mean(Ym[N_trans+1:end,:], dims = 2), :) # Returns 1D-array
    # end

    function get_outputs_sens(y, δ, pars, isws, M_mean)
        p = get_all_parameters(pars)
        η = p[dθ+1: dθ+dη]
        C = reshape(η[nx+1:end], (n_out, n_tot))
        ms = collect(1:2M_mean)

        dmdl = discretize_ct_noise_model(get_ct_disturbance_model(η, nx, n_out), δ)
        # NOTE: We use new realizations of white noise every time we estimate
        # the gradient, so that all samples of the gradient are independent
        # of each other
        # # NOTE: OPTION 1: Use the rows below here for linear interpolation
        XWm = simulate_noise_process_mangled(dmdl, [randn(Nw, n_tot) for m = ms])
        wmm(m::Int) = mk_noise_interp(C, XWm, m, δ)
        # NOTE: OPTION 2: Use the rows below here for exact interpolation
        # reset_isws!(isws)
        # XWm = simulate_noise_process(dmdl, [randn(Nw, n_tot) for m = ms])
        # wmm(m::Int) = mk_newer_noise_interp(view(η, 1:nx), C, XWm, m, n_in, δ, isws)

        calc_mean_y_N(N::Int, pars::Array{Float64, 1}, m::Int) =
            solvew_sens(u, t -> wmm(m)(t), pars, N) |> h_sens
        calc_mean_y(pars::Array{Float64, 1}, m::Int) = calc_mean_y_N(N, pars, m)
        Ym, gradYm = solve_in_parallel_sens(m -> calc_mean_y(pars, m), ms)
    end

    # Returns estimate of gradient of cost function
    # M_mean specified over how many realizations the gradient estimate is computed
    function get_gradient_estimate(y, δ, pars, isws, M_mean::Int=1)
        Ym, gradYm = get_outputs_sens(y, δ, pars, isws, 2M_mean)
        # Uses different noise realizations for estimate of output and estiamte of gradient
        grad_est = get_cost_jacobian(y[N_trans+1:end], Ym[N_trans+1:end,1:M_mean], gradYm[N_trans+1:end,M_mean+1:end])
        # grad_est will be 2D array with one dimenion equal to 1, we want to return a 1D array
        return grad_est[:]
    end

    # Returns estimate of gradient of output
    function get_proposed_gradient(y, δ, pars, isws, M_mean::Int=1)
        _, gradYm = get_outputs_sens(y, δ, pars, isws, M_mean)
        return mean(gradYm, dims=2)
    end

    # TODO: Should we consider storing and loading white noise, to improve repeatability
    # of the results? Currently repeatability is based on fixed random seed
    Zm = [randn(Nw, n_tot) for m = 1:M]

    get_gradient_estimate_p(pars, M_mean) = get_gradient_estimate(Y[:,1], δ, pars, isws, M_mean)

    opt_pars_proposed = zeros(length(pars0), E)
    trace_proposed = [Float64[] for e=1:E]
    for e=1:E
        # jacobian_model(x, p) = get_proposed_gradient(Y[:,e], δ, pars, isws, M)  # NOTE: This won't give a gradient estimate independent of Ym, but maybe we don't need that since this isn't SGD?
        opt_pars_proposed[:,e], trace_proposed[e] =
            perform_stochastic_gradient_descent(get_gradient_estimate_p, pars0, par_bounds, verbose=true)
        # proposed_result, proposed_trace = get_fit(Y[:,e], pars0,
        #     (dummy_input, pars) -> proposed_model_parametrized(δ, Zm, dummy_input, pars, isws))
        # proposed_result, proposed_trace = get_fit_sens(Y[:,e], pars0,
        #     (dummy_input, pars) -> proposed_model_parametrized(δ, Zm, dummy_input, pars, isws), jacobian_model)
        # opt_pars_proposed[:, e] = coef(proposed_result)
        println("Completed for dataset $e for parameters $(opt_pars_proposed[:,e])")
    end

    @info "The mean optimal parameters for proposed method are given by: $(mean(opt_pars_proposed, dims=2))"

    return (opt_pars_baseline, opt_pars_proposed, trace_base, trace_proposed)
end

function get_outputs(expid, pars0::Array{Float64,1})

    exp_data, isws = get_experiment_data(expid)
    W_meta = exp_data.W_meta
    Y = exp_data.Y
    N = size(Y,1)-1
    u = exp_data.u
    Nw = exp_data.Nw
    nx = W_meta.nx
    n_in = W_meta.n_in
    n_out = W_meta.n_out
    n_tot = nx*n_in
    dη = length(W_meta.η)

    get_all_parameters(pars::Array{Float64, 1}) = vcat(get_all_θs(pars), exp_data.get_all_ηs(pars))
    p = get_all_parameters(pars0)
    θ = p[1:dθ]
    η = p[dθ+1: dθ+dη]
    C = reshape(η[nx+1:end], (n_out, n_tot))

    # === Computes output of the baseline model ===
    Y_base, sens_base = solvew_sens(u, t -> zeros(n_out), pars0, N) |> h_baseline_sens


    # === Computes outputs of the proposed model ===
    # TODO: Should we consider storing and loading white noise, to improve repeatability
    # of the results? Currently repeatability is based on fixed random seed
    Zm = [randn(Nw, n_tot) for m = 1:M]
    dmdl = discretize_ct_noise_model(get_ct_disturbance_model(η, nx, n_out), δ)
    # NOTE: OPTION 1: Use the rows below here for linear interpolation
    XWm = simulate_noise_process_mangled(dmdl, Zm)
    wmm(m::Int) = mk_noise_interp(C, XWm, m, δ)
    # # NOTE: OPTION 2: Use the rows below here for exact interpolation
    # reset_isws!(isws)
    # XWm = simulate_noise_process(dmdl, Zm)
    # wmm(m::Int) = mk_newer_noise_interp(view(η, 1:nx), C, XWm, m, n_in, δ, isws)

    calc_mean_y_N_prop(N::Int, pars::Array{Float64, 1}, m::Int) =
        solvew_sens(u, t -> wmm(m)(t), pars, N) |> h_sens
    calc_mean_y_prop(pars::Array{Float64, 1}, m::Int) = calc_mean_y_N_prop(N, pars, m)
    Ym_prop, sens_m_prop = solve_in_parallel_sens(m -> calc_mean_y_prop(pars0, m), ms)
    Y_mean_prop = reshape(mean(Ym_prop, dims = 2), :)

    return Y, Y_base, sens_base, Ym_prop, Y_mean_prop, sens_m_prop
end

function get_experiment_data(expid)::Tuple{ExperimentData, Array{InterSampleWindow, 1}}
    # A single realization of the disturbance serves as input
    # input is assumed to contain the input signal, and not the state
    input  = readdlm(joinpath(data_dir, expid*"/U.csv"), ',')
    XW     = readdlm(joinpath(data_dir, expid*"/XW.csv"), ',')
    W_meta_raw, W_meta_names =
        readdlm(joinpath(data_dir, expid*"/meta_W.csv"), ',', header=true)

    # NOTE: These variable assignments are based on the names in W_meta_names,
    # but hard-coded, so if W_meta_names is changed then there is no guarantee
    # that they will match
    nx = Int(W_meta_raw[1,1])
    n_in = Int(W_meta_raw[1,2])
    n_out = Int(W_meta_raw[1,3])
    n_tot = nx*n_in
    # Parameters of true system
    η_true = W_meta_raw[:,4]
    dη = length(η_true)
    a_vec = η_true[1:nx]
    C_true = reshape(η_true[nx+1:end], (n_out, n_tot))

    # compute the maximum number of steps we can take
    N_margin = 2    # Solver can request values of inputs after the time horizon
                    # ends, so we require a margin of a few samples of the noise
                    # to ensure that we can provide such values
    # Minimum of number of available disturbance or input samples
    Nw = min(size(XW, 1)÷n_tot, size(input, 1)÷n_in)
    N = Int((Nw - N_margin)*δ÷Ts)     # Number of steps we can take
    W_meta = DisturbanceMetaData(nx, n_in, n_out, η_true)

    # Use this function to specify which parameters should be free and optimized over
    get_all_ηs(pars::Array{Float64, 1}) = η_true   # Known disturbance model
    # Array of tuples containing lower and upper bound for each free disturbance parameter
    dist_par_bounds = Array{Float64}(undef, 0, 2)

    mk_we(XW::Array{Array{Float64, 1},2}, isws::Array{InterSampleWindow, 1}) =
        (m::Int) -> mk_newer_noise_interp(
        a_vec::AbstractArray{Float64, 1}, C_true, XW, m, n_in, δ, isws)
    u(t::Float64) = interpw(input, 1, 1)(t)

    # === We first generate the output of the true system ===
    function calc_Y(XW::Array{Array{Float64, 1},2}, isws::Array{InterSampleWindow, 1})
        # NOTE: This XW should be non-mangled, which is why we don't divide by n_tot
        @assert (Nw <= size(XW, 1)) "Disturbance data size mismatch ($(Nw) > $(size(XW, 1)))"
        E = size(XW, 2)
        es = collect(1:E)
        we = mk_we(XW, isws)
        # solve_in_parallel(e -> solvew(u, we(e), pars_true, N) |> h_data, es)
        Y = solve_in_parallel(e -> solvew(u, we(e), pars_true, N) |> h_data, es)
        return Y
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
    return ExperimentData(Y, u, get_all_ηs, dist_par_bounds, W_meta, Nw), isws
end

function perform_stochastic_gradient_descent(
    get_grad_estimate::Function,
    pars0::Array{Float64,1},                        # Initial guess for parameters
    bounds::Array{Float64, 2},                      # Parameter bounds
    learning_rate::Function=learning_rate;
    tol::Float64=1e-4,# tol::Float64=1e-8,
    maxiters=200,
    verbose=false)

    # running_criterion(grad_est) = (norm(grad_est) > tol)
    last_10_params = zeros(10)
    last_10_params[end] = 10000     # Fills with some large number
    running_criterion() = (var(last_10_params) > tol)

    t = 1
    pars = pars0
    trace = Float64[]                  # NOTE: Only works for scalar parameter right now
    push!(trace, pars[1])
    while t <= maxiters
        grad_est = get_grad_estimate(pars, M_rate(t))
        # println("Iteration $t, gradient norm $(norm(grad_est)) and step sign $(sign(-first(grad_est))) with parameter estimate $pars")
        # running_criterion(grad_est) || break
        if verbose
            println("Iteration $t, variance $(var(last_10_params)) and step sign $(sign(-first(grad_est))) with parameter estimate $pars")
        end
        running_criterion() || break
        # pars = pars - learning_rate(t)*grad_est
        pars = pars - learning_rate(t, norm(grad_est))*grad_est
        project_on_bounds!(pars, bounds)
        push!(trace, pars[1])
        # (t-1)%10+1 maps t to a number between 1 and 10 (inclusive)
        last_10_params[(t-1)%10+1] = pars[1]
        t += 1
    end
    return pars, trace
end

# Row i of bounds should have two columns, where first element is lower bound
# for parameter i, and second element is upper bound for parameter i
function project_on_bounds!(vec::Array{Float64,1}, bounds::Array{Float64,2})
    low_inds = findall(vec .< bounds[:,1])
    high_inds = findall(vec .> bounds[:,2])
    vec[low_inds] = bounds[low_inds, 1]
    vec[high_inds] = bounds[high_inds, 2]
    nothing
end

# NOTE: Works only for scalar parameter!!!
# param_set_i[j,k] should contain optimal parameters corresponding to method i,
# time horizon Ns[j] and data-set k
function plot_parameter_boxplots(param_set_1::Array{Float64,2}, param_set_2::Array{Float64, 2}, Ns::Array{Int, 1})
    # θhatbs =
    # map(N -> calc_theta_hats(outputs.θs, outputs.Y, outputs.Yb, N_trans, N), Ns)
    # θhatms =
    # map(N -> calc_theta_hats(outputs.θs, outputs.Y, outputs.Ym, N_trans, N), Ns)
    # θhats = hcat(θhatbs..., θhatms...)

    labels = reshape([map(N -> "bl $(N)", Ns); map(N -> "m $(N)", Ns)], (1, :))
    idxs = 1:(2length(Ns))
    θhats = hcat(transpose(param_set_1), transpose(param_set_2))
    # Quantiles are computed over dimension 1, dimension 2 interpreted as
    # different box plots
    p = boxplot(
    θhats,
    xticks = (idxs, labels),
    label = "",
    ylabel = L"\hat{\theta}",
    notch = false,
    )

    hline!(p, pars_true, label = L"\theta_0", linestyle = :dot, linecolor = :gray)
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

# Simulates system with specified white noise
function simulate_system(
    exp_data::ExperimentData,
    par_vector::Array{Array{Float64,1},1},
    M::Int,
    isws::Array{InterSampleWindow,1},
    Zm::Array{Array{Float64,2},1})::Array{Array{Float64,2},1}

    W_meta = exp_data.W_meta
    nx = W_meta.nx
    n_in = W_meta.n_in
    n_out = W_meta.n_out
    n_tot = nx*n_in
    dη = length(W_meta.η)
    N = size(exp_data.Y, 1)-1
    get_all_parameters(pars::Array{Float64, 1}) = vcat(get_all_θs(pars), exp_data.get_all_ηs(pars))

    # Allocates space for system outputs
    Yms = [zeros(N+1, M) for j = 1:length(par_vector)]

    for (j, pars) in enumerate(par_vector)
        # NOTE: The true input is encoded in the solvew_sens()-function, but this function
        # still needs to to take two input arguments, so dummy_input could just be
        # anything, it's not used anyway
        p = get_all_parameters(pars)
        θ = p[1:dθ]
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
            solvew(exp_data.u, t -> wmm(m)(t), pars, N) |> h
        calc_mean_y(pars::Array{Float64, 1}, m::Int) = calc_mean_y_N(N, pars, m)
        Yms[j] = solve_in_parallel(m -> calc_mean_y(pars, m), collect(1:M))
    end
    return Yms
end

# Simulates system with newly generated white noise
function simulate_system(
    exp_data::ExperimentData,
    par_vector::Array{Array{Float64,1},1},
    M::Int,
    isws::Array{InterSampleWindow,1})::Array{Array{Float64,2},1}

    W_meta = exp_data.W_meta
    nx = W_meta.nx
    n_in = W_meta.n_in
    n_tot = nx*n_in
    Nw = exp_data.Nw
    Zm = [randn(Nw, n_tot) for m = 1:M]
    simulate_system(exp_data, par_vector, M, isws, Zm)
end

# Simulates system with specified white noise
function simulate_system_sens(
    exp_data::ExperimentData,
    par_vector::Array{Array{Float64,1},1},
    M::Int,
    isws::Array{InterSampleWindow,1},
    Zm::Array{Array{Float64,2},1})::Tuple{Array{Array{Float64,2},1}, Array{Array{Float64,2},1}}

    M_used = M÷2
    W_meta = exp_data.W_meta
    nx = W_meta.nx
    n_in = W_meta.n_in
    n_out = W_meta.n_out
    n_tot = nx*n_in
    dη = length(W_meta.η)
    N = size(exp_data.Y, 1)-1
    get_all_parameters(pars::Array{Float64, 1}) = vcat(get_all_θs(pars), exp_data.get_all_ηs(pars))

    # Allocates space for system outputs
    Yms = [zeros(N+1, M_used) for j = 1:length(par_vector)]
    Jacobians = [zeros(N+1, M_used) for j = 1:length(par_vector)]

    for (j, pars) in enumerate(par_vector)
        # NOTE: The true input is encoded in the solvew_sens()-function, but this function
        # still needs to to take two input arguments, so dummy_input could just be
        # anything, it's not used anyway
        p = get_all_parameters(pars)
        θ = p[1:dθ]
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
            solvew_sens(exp_data.u, t -> wmm(m)(t), pars, N) |> h_sens
        calc_mean_y(pars::Array{Float64, 1}, m::Int) = calc_mean_y_N(N, pars, m)
        @info "Solving in parallel for M=$M"
        @time Yms[j], Jacobians[j] = solve_in_parallel_sens(m -> calc_mean_y(pars, m), collect(1:M))
    end
    return Yms, Jacobians
end

# Simulates system with newly generated white noise
function simulate_system_sens(
    exp_data::ExperimentData,
    par_vector::Array{Array{Float64,1},1},
    M::Int,
    isws::Array{InterSampleWindow,1})::Tuple{Array{Array{Float64,2},1}, Array{Array{Float64,2},1}}

    W_meta = exp_data.W_meta
    nx = W_meta.nx
    n_in = W_meta.n_in
    n_tot = nx*n_in
    Nw = exp_data.Nw
    Zm = [randn(Nw, n_tot) for m = 1:M]
    simulate_system_sens(exp_data, par_vector, M, isws, Zm)
end

# ======================= DEBUGGING FUNCTIONS ========================
function compute_forward_difference_derivative(x_vals::Array{Float64,1}, y_vals::Array{Float64,1})
    @assert (length(x_vals) == length(y_vals)) "x_vals and y_vals must contain the same number of elements"
    diff = zeros(size(y_vals))
    for i=1:length(y_vals)-1
        diff[i] = (y_vals[i+1]-y_vals[i])/(x_vals[i+1]-x_vals[i])
    end
    diff[end] = diff[end-1]
    return diff
end

function debug_minimization(expid, pars0::Array{Float64,1}, par_l_range = 2.0, par_r_range = 2.0, N_trans::Int = 0, step_size::Float64=0.1)
    exp_data, isws = get_experiment_data(expid)
    W_meta = exp_data.W_meta
    Y = exp_data.Y
    N = size(Y,1)-1
    Nw = exp_data.Nw
    nx = W_meta.nx
    n_in = W_meta.n_in
    n_out = W_meta.n_out
    n_tot = nx*n_in
    dη = length(W_meta.η)
    u = exp_data.u
    par_bounds = vcat(dyn_par_bounds, exp_data.dist_par_bounds)

    get_all_parameters(pars::Array{Float64, 1}) = vcat(get_all_θs(pars), exp_data.get_all_ηs(pars))

    # === Optimizing parameters for the baseline model ===
    function baseline_model_parametrized(δ, dummy_input, pars)
        # NOTE: The true input is encoded in the solvew_sens()-function, but this function
        # still needs to to take two input arguments, so dummy_input could just be
        # anything, it's not used anyway
        # Y_base = solvew(u, t -> zeros(n_out), pars, N ) |> h_baseline


        # temp, temp_sens = solvew_sens(u, t -> zeros(n_out), pars, N )
        Y_base = solvew(u, t -> zeros(n_out), pars, N ) |> h_baseline

        # TODO: Currently assumes scalar output from DAE-system
        return reshape(Y_base[N_trans+1:end,:], :)   # Returns 1D-array
    end

    function get_baseline_gradient(y, δ, pars)
        p = get_all_parameters(pars)
        η = p[dθ+1: dθ+dη]
        C = reshape(η[nx+1:end], (n_out, n_tot))

        _, gradYm = solvew_sens(u, t -> 0.0, pars, N) |> h_sens
        gradYm
    end

    # E = size(Y, 2)
    # DEBUG
    E = 1
    opt_pars_baseline = zeros(length(pars0), E)
    trace_base = [Float64[] for e=1:E]
    jacobian_model_b(x, p) = get_baseline_gradient(Y[:,1], δ, p)
    for e=1:E
        push!(trace_base[e], pars0[1]) # NOTE: Only works for scalar parameter
        # HACK: Uses trace returned due to hacked LsqFit package
        baseline_result, baseline_trace = get_fit_sens(Y[:,e], pars0,
            (dummy_input, pars) -> baseline_model_parametrized(δ, dummy_input, pars), jacobian_model_b)
        opt_pars_baseline[:, e] = coef(baseline_result)
        if length(baseline_trace) > 1
            for j=2:length(baseline_trace)
                push!(trace_base[e], baseline_trace[j].metadata["x"][1])
            end
        end
    end

    @info "The mean optimal parameters for baseline are given by: $(mean(opt_pars_baseline, dims=2))"

    # === Computing cost function of baseline model
    base_par_vals = collect( (pars0[1]-par_l_range):step_size:(pars0[1]+par_r_range))
    par_vector = [[el] for el in base_par_vals]
    Ybs = [zeros(N+1) for j = 1:length(par_vector)]
    for (j, pars) in enumerate(par_vector)
        Ybs[j] = solvew(exp_data.u, t -> zeros(n_out), pars, N) |> h
    end
    base_cost_vals = zeros(length(base_par_vals), E)
    for ind = 1:length(base_par_vals)
        # Y has columns indexed with 1:E because in case E is changed for debugging purposes,
        # without the dimensions of Y changing, we can get a mismatch otherwise
        base_cost_vals[ind,:] = mean((Y[N_trans+1:end, 1:E].-Ybs[ind][N_trans+1:end]).^2, dims=1)
    end

    # === Computing cost function for proposed model ===
    @info "Plotting proposed cost function..."
    # prop_par_vals = collect(1.0:2step_size:35.0)
    prop_par_vals = collect(1.0:0.1:2.0)
    Yms = simulate_system(exp_data, [[el] for el in prop_par_vals], M, isws)
    prop_cost_vals = zeros(length(prop_par_vals), E)
    for ind = 1:length(prop_par_vals)
        Y_mean = mean(Yms[ind], dims=2)
        # Y has columns indexed with 1:E because in case E is changed for debugging purposes,
        # without the dimensions of Y changing, we can get a mismatch otherwise
        prop_cost_vals[ind,:] = mean((Y[N_trans+1:end, 1:E].-Y_mean[N_trans+1:end]).^2, dims=1)
    end

    # === Optimizing parameters for the proposed model or stochastic gradient descent ==
    function proposed_model_parametrized(δ, Zm, dummy_input, pars, isws)
        # NOTE: The true input is encoded in the solvew_sens()-function, but this function
        # still needs to to take two input arguments, so dummy_input could just be
        # anything, it's not used anyway
        p = get_all_parameters(pars)
        θ = p[1:dθ]
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

    function get_outputs_sens(y, δ, pars, isws, M_out)
        p = get_all_parameters(pars)
        η = p[dθ+1: dθ+dη]
        C = reshape(η[nx+1:end], (n_out, n_tot))
        ms = collect(1:M_out)

        dmdl = discretize_ct_noise_model(get_ct_disturbance_model(η, nx, n_out), δ)
        # NOTE: We use new realizations of white noise every time we estimate
        # the gradient, so that all samples of the gradient are independent
        # of each other
        # NOTE: OPTION 1: Use the rows below here for linear interpolation
        XWm = simulate_noise_process_mangled(dmdl, [randn(Nw, n_tot) for m = ms])
        wmm(m::Int) = mk_noise_interp(C, XWm, m, δ)
        # NOTE: OPTION 2: Use the rows below here for exact interpolation
        # reset_isws!(isws)
        # XWm = simulate_noise_process(dmdl, [randn(Nw, n_tot) for m = ms])
        # wmm(m::Int) = mk_newer_noise_interp(view(η, 1:nx), C, XWm, m, n_in, δ, isws)

        calc_mean_y_N(N::Int, pars::Array{Float64, 1}, m::Int) =
            solvew_sens(u, t -> wmm(m)(t), pars, N) |> h_sens
        calc_mean_y(pars::Array{Float64, 1}, m::Int) = calc_mean_y_N(N, pars, m)
        Ym, gradYm = solve_in_parallel_sens(m -> calc_mean_y(pars, m), ms)
    end

    # Returns estimate of gradient of cost function
    # M_mean specified over how many realizations the gradient estimate is computed
    function get_gradient_estimate(y, δ, pars, isws, M_mean::Int=1)
        Ym, gradYm = get_outputs_sens(y, δ, pars, isws, 2M_mean)
        # Uses different noise realizations for estimate of output and estiamte of gradient
        grad_est = get_cost_jacobian(y[N_trans+1:end],
            Ym[N_trans+1:end, 1:M_mean], gradYm[N_trans+1:end, M_mean+1:end])
        # grad_est will be 2D array with one dimenion equal to 1, we want to return a 1D array
        return grad_est[:]  # TODO: Generalize to multivariate case
    end

    # Returns estimate of gradient of cost function
    # M_mean specified over how many realizations the gradient estimate is computed
    function get_gradient_estimate_debug(y, δ, pars, isws, M_mean::Int=1)
        Ym, gradYm = get_outputs_sens(y, δ, pars, isws, 2M_mean)
        # Uses different noise realizations for estimate of output and estiamte of gradient
        grad_est = get_cost_jacobian(y[N_trans+1:end],
            Ym[N_trans+1:end, 1:M_mean], gradYm[N_trans+1:end, M_mean+1:end])
        # cost = mean((y-mean(Ym, dims=2).^2))
        cost = mean((y[N_trans+1:end]-Ym[N_trans+1:end,1]).^2)
        # grad_est will be 2D array with one dimenion equal to 1, we want to return a 1D array
        return grad_est[:], cost
    end

    # Returns estimate of gradient of output
    function get_proposed_gradient(y, δ, pars, isws, M_mean::Int=1)
        _, gradYm = get_outputs_sens(y, δ, pars, isws, M_mean)
        return mean(gradYm, dims=2)
    end

    # NOTE: This disturbance matrix is not used for stochastic gradient descent
    # TODO: Should we consider storing and loading white noise, to improve repeatability
    # of the results? Currently repeatability is based on fixed random seed
    Zm = [randn(Nw, n_tot) for m = 1:M]

    @info "Finding proposed minimum..."

    opt_pars_proposed = zeros(length(pars0), E)
    opt_pars_proposed_LSQ = zeros(length(pars0), E)
    trace_proposed = [Float64[] for e=1:E]
    trace_costs = [Float64[] for e=1:E]
    grad_norms  = [Float64[] for e=1:E]
    for e=1:E
        get_gradient_estimate_p(pars, M_mean) = get_gradient_estimate(Y[:,e], δ, pars, isws, M_mean)
        get_gradient_estimate_p_debug(pars, M_mean) = get_gradient_estimate_debug(Y[:,e], δ, pars, isws, M_mean)
        jacobian_model(x, p) = get_proposed_gradient(Y[:,e], δ, p, isws, M)  # NOTE: This won't give a gradient estimate independent of Ym, but maybe we don't need that since this isn't SGD?
        # opt_pars_proposed[:,e], trace_proposed[e] =
        #     perform_stochastic_gradient_descent(get_gradient_estimate_p, pars0, par_bounds, verbose=true)
        opt_pars_proposed[:,e], trace_proposed[e], trace_costs[e], grad_norms[e] =
            perform_stochastic_gradient_descent_debug(get_gradient_estimate_p_debug, pars0, par_bounds, verbose=true; maxiters=500, tol=1e-6)
        reset_isws!(isws)
        proposed_result, proposed_trace = get_fit_sens(Y[:,e], pars0,
            (dummy_input, pars) -> proposed_model_parametrized(δ, Zm, dummy_input, pars, isws), jacobian_model)
        # proposed_result, proposed_trace = get_fit_sens(Y[:,e], pars0,
        #     (dummy_input, pars) -> proposed_model_parametrized(δ, Zm, dummy_input, pars, isws), jacobian_model)
        # opt_pars_proposed[:, e] = coef(proposed_result)
        opt_pars_proposed_LSQ[:, e] = coef(proposed_result)

        println("Completed for dataset $e for parameters $(opt_pars_proposed[:,e])")
    end

    @info "The mean optimal parameters for proposed method are given by: $(mean(opt_pars_proposed, dims=2))"

    return opt_pars_baseline, opt_pars_proposed, trace_base, trace_proposed, base_cost_vals, prop_cost_vals, trace_costs, base_par_vals, prop_par_vals, opt_pars_proposed_LSQ
end

function debug_proposed_cost_func(expid, par_vector::Array{Array{Float64,1},1}, N_trans::Int = 0)
    exp_data, isws = get_experiment_data(expid)
    W_meta = exp_data.W_meta
    Y = exp_data.Y
    N = size(Y,1)-1
    Nw = exp_data.Nw
    nx = W_meta.nx
    n_in = W_meta.n_in
    n_out = W_meta.n_out
    n_tot = nx*n_in
    dη = length(W_meta.η)
    u = exp_data.u
    par_bounds = vcat(dyn_par_bounds, exp_data.dist_par_bounds)

    get_all_parameters(pars::Array{Float64, 1}) = vcat(get_all_θs(pars), exp_data.get_all_ηs(pars))

    Ms = [1, 10, 100, 500]
    Mmax = maximum(Ms)
    # E = size(Y, 2)
    # DEBUG
    E = 6
    e = 6

    costs = [zeros(length(par_vector)) for M in Ms]
    cost_jacobians = [zeros(length(par_vector)) for M in Ms]

    @info "Computing cost function"
    # ============== Computing Cost Functions =====================
    Yms = simulate_system(exp_data, par_vector, Mmax, isws)
    for (M_ind, M) in enumerate(Ms)
        for par_ind = 1:length(par_vector)
            Y_mean = mean(Yms[par_ind][:,1:M], dims=2)
            costs[M_ind][par_ind] =
                mean((Y[N_trans+1:end,e]-Y_mean[N_trans+1:end]).^2)
        end
    end

    @info "Computing cost function gradients"
    # ============== Computing Cost Gradients =======================
    Mjacs = [max(M÷2,1) for M in Ms]
    # max(Mmax, 2) because at least 2 independent realizations are needed to
    # estimate the gradient of the cost function
    Yms, Jacobians = simulate_system_sens(exp_data, par_vector, max(Mmax, 2), isws)
    for (M_ind, Mjac) in enumerate(Mjacs)
        for par_ind = 1:length(par_vector)
            cost_jacobians[M_ind][par_ind] = get_cost_jacobian(
                Y[N_trans+1:end, e],
                Yms[par_ind][N_trans+1:end, 1:Mjac],
                Jacobians[par_ind][N_trans+1:end, Mjac+1:2Mjac])[1]
        end
    end

    @info "Optimizing cost function using GD"
    # ================== Optimizing using SGD ======================
    function get_gradient_estimate(y, δ, pars, isws, M_mean::Int=1)
        Ym, gradYm = get_outputs_sens(y, δ, pars, isws, 2M_mean)
        # Uses different noise realizations for estimate of output and estiamte of gradient
        grad_est = get_cost_jacobian(y[N_trans+1:end], Ym[N_trans+1:end,1:M_mean], gradYm[N_trans+1:end, M_mean01:end])
        # grad_est will be 2D array with one dimenion equal to 1, we want to return a 1D array
        return grad_est[:]
    end

    opt_pars_proposed, trace_proposed =
        perform_stochastic_gradient_descent(get_gradient_estimate_p, pars0, par_bounds, verbose=true)

    return costs, cost_jacobians
end

# Similar to perform_stochastic_gradient_descent() but also returns
# the estimated costs at all the tested parameters as well as a log of
# the norms for the gradients used at each step
function perform_stochastic_gradient_descent_debug(
    get_grad_estimate_debug::Function,
    pars0::Array{Float64,1},                       # Initial guess for parameters
    bounds::Array{Float64, 2},                      # Parameter bounds
    learning_rate::Function=learning_rate;
    tol::Float64=1e-4,# tol::Float64=1e-8,
    maxiters=200,
    verbose=false)

    # running_criterion(grad_est) = (norm(grad_est) > tol)
    last_10_params = zeros(10)
    last_10_params[end] = 10000     # Fills with some large number
    running_criterion() = (var(last_10_params) > tol)


    t = 1
    pars = pars0
    cost_vals = Float64[]
    trace = Float64[]                  # NOTE: Only works for scalar parameter right now
    grad_norms = Float64[]
    push!(trace, pars[1])
    while t <= maxiters
        grad_est, cost_val = get_grad_estimate_debug(pars, M_rate(t))
        push!(cost_vals, cost_val)
        push!(grad_norms, norm(grad_est))
        # println("Iteration $t, gradient norm $(norm(grad_est)) and step sign $(sign(-first(grad_est))) with parameter estimate $pars")
        # running_criterion(grad_est) || break
        if verbose
            println("Iteration $t, variance $(var(last_10_params)) and step sign $(sign(-first(grad_est))) with parameter estimate $pars")
        end
        running_criterion() || break
        # pars = pars - learning_rate(t)*grad_est
        pars = pars - learning_rate(t, norm(grad_est))*grad_est
        project_on_bounds!(pars, bounds)
        push!(trace, pars[1])
        # (t-1)%10+1 maps t to a number between 1 and 10 (inclusive)
        last_10_params[(t-1)%10+1] = pars[1]
        t += 1
    end
    return pars, trace, cost_vals, grad_norms
end

function visualize_SGD_search(par_vals::Array{Float64,1}, cost_vals::Array{Float64, 1}, SGD_trace::Array{Float64, 1}, opt_par::Float64; file_name = "SGD.gif")
    anim = @animate for i = 1:length(SGD_trace)
        p = plot(par_vals, cost_vals, label="Cost Function")
        vline!(p, [opt_par], label="Final parameter")
        vline!(p, [SGD_trace[i]], label="Current parameter")
    end
    gif(anim, file_name, fps = 15)
    # # NOTE: Not sure if maybe this is faster?
    # @gif for i = 1:length(SGD_trace)
    #     p = plot(par_vals, cost_vals, label="Cost Function")
    #     vline!(p, [opt_par], label="Final parameter")
    #     vline!(p, [SGD_trace[i]], label="Current parameter")
    # end every 1
end

function Roberts_gif_generator(par_vals::Array{Float64,1}, cost_vals::Array{Float64, 2}, SGD_traces::Array{Array{Float64,1},1}, opt_pars::Array{Float64, 1})
    for i = 1:6
        visualize_SGD_search(par_vals, cost_vals[:,i], SGD_traces[i], opt_pars[i], file_name="M100_Mgrad8_newspecrate1e5_0p1_rel$(i).gif")
    end
end
