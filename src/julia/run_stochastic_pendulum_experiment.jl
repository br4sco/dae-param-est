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
const M = 100       # Number of monte-carlo simulations used for estimating mean
# TODO: Surely we don't need to collect these, a range should work just as well?
const ms = collect(1:M)
const W = 100           # Number of intervals for which isw stores data
const Q = 1000          # Number of conditional samples stored per interval

const noise_method_name = "Pre-generated unconditioned noise (δ = $(δ))"

M_rate_max = 4#16
# max_allowed_step = 1.0  # Maximum magnitude of step that SGD is allowed to take
# M_rate(t) specifies over how many realizations the output jacobian estimate
# should be computed at iteration t. NOTE: A total of 2*M_rate(t) iterations
# will be performed for estimating the gradient of the cost functions
M_rate(t::Int) = M_rate_max
# This learning rate works very well with M_rate = 8
lr_break = 80
# learning_rate(t::Int, grad_norm::Float64) = if (t < lr_break) 1/grad_norm else 1/(grad_norm*(1+1*(t-lr_break) )) end
learning_rate(t::Int, grad_norm::Float64) = if (t < lr_break) 1e6 else 1e6/(1+t) end
learning_rate_adam(t::Int, grad_norm::Float64) = 0.1#if (t < lr_break) 1e0 else 1e0/(1+t) end

mangle_XW(XW::Array{Array{Float64, 1}, 2}) =
  hcat([vcat(XW[:, m]...) for m = 1:size(XW,2)]...)

demangle_XW(XW::Array{Float64, 2}, n_tot::Int) =
    [XW[(i-1)*n_tot+1:i*n_tot, m] for i=1:(size(XW,1)÷n_tot), m=1:size(XW,2)]

# NOTE: SCALAR_OUTPUT is assumed
# NOTE: The realizaitons Ym and jacYm must be independent for this to return
# an unbiased estimate of the cost function gradient
function get_cost_gradient(Y::Array{Float64,1}, Ym::Array{Float64,2}, jacsYm::Array{Array{Float64,2},1}, N_trans::Int=0)
    (2/(size(Y[N_trans+1:end,:],1)-1))*
        sum(
        mean(Y[N_trans+1:end,:].-Ym[N_trans+1:end,:], dims=2)
        .*mean(-jacsYm)[N_trans+1:end,:]
        , dims=1)
end

# function get_cost_gradient(Y::Array{Float64,1}, Ym::Array{Float64,2}, jacYm::Array{Float64,2})
#     (2/(size(Y,1)-1))*
#         sum(
#         mean(Y.-Ym, dims=2)
#         .*mean(-jacYm, dims=2)
#         , dims=1)
# end

# We do linear interpolation between exact values because it's fast
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

function get_fit(Ye, pars, model)
    lb = fill(-Inf, length(pars))
    ub = fill(Inf,  length(pars))
    get_fit(Ye, pars, model, lb, ub)
end

# NOTE:use coef(fit_result) to get optimal parameter values
function get_fit(Ye, pars, model, lb, ub)
    # # Use this line if you are using the original LsqFit-package
    # return curve_fit(model, 1:2, Y[:,1], p, show_trace=true, inplace = false, x_tol = 1e-8)    # Default inplace = false, x_tol = 1e-8

    # HACK: Uses trace returned due to hacked LsqFit package
    # Use this line if you are using the modified LsqFit-package that also
    # returns trace
    fit_result, trace = curve_fit(model, Float64[], Ye, pars, lower=lb, upper=ub, show_trace=true, inplace=false, x_tol=1e-8)    # Default inplace = false, x_tol = 1e-8
    return fit_result, trace
end

function get_fit_sens(Ye, pars, model, jacobian_model)
    lb = fill(-Inf, length(pars))
    ub = fill(Inf,  length(pars))
    get_fit_sens(Ye, pars, model, jacobian_model, lb, ub)
end

# Uses a jacobian model (for system output) instead of estimating jacobian from forward difference
function get_fit_sens(Ye, pars, model, jacobian_model, lb, ub)
    # HACK: Uses trace returned due to hacked LsqFit package
    # Use this line if you are using the modified LsqFit-package that also
    # returns trace
    fit_result, trace = curve_fit(model, jacobian_model, Float64[], Ye, pars, lower=lb, upper=ub, show_trace=true, inplace=false, x_tol=1e-8)    # Default inplace = false, x_tol = 1e-8
    return fit_result, trace
end

# === MODEL (AND DATA) PARAMETERS ===
const σ = 0.002                 # measurement noise variance

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
# f_sens(x::Array{Float64,1}) = [x[14], x[21], x[28], x[35]]   # NOTE: Hard-coded right now
f_sens(x::Array{Float64, 1}) = [x[14], x[21], x[28]]
# NOTE: Has to be changed when number of free  parameters is changed.
# Specifically, f_sens() must return sensitivity wrt to all free parameters
# f_sens(x::Array{Float64,1}) = [x[14], x[21]]   # NOTE: Hard-coded right now
# f_sens(x::Array{Float64,1}) = [x[14], x[21], x[28]]   # NOTE: Hard-coded right now
h(sol) = apply_outputfun(f, sol)                            # for our model
h_comp(sol) = apply_two_outputfun_mvar(f, f_sens, sol)           # for complete model with dynamics sensitivity
h_sens(sol) = apply_outputfun_mvar(f_sens, sol)              # for only returning sensitivity

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
# NOTE: If number and location of free parameters change, the sensitivity TODO: There must be a nicer solution to this
# functions defined in the code must also be changed
# const pars_true = [k]                    # true value of all free parameters
# get_all_θs(pars::Array{Float64,1}) = [m, L, g, pars[1]]
# dyn_par_bounds = [0.1 Inf]    # Lower and upper bounds of each free dynamic parameter
# NOTE: Has to be changed when number of free dynamical parameters is changed.
# Specifically:
# 1. pars_true must contain the true values for all free dynamical parameters
# 2. get_all_θs() must return all variables, where the free variables
#       need to be replaced by the provided argument pars
# 3. dyn_par_bounds must include bounds for all free dynamical paramters
# 4. The first argument to problem() in the definition of realize_model_sens
#       must refere to DAE-problem that includes sensitvity equations for all
#       free dynamical parameters
# sensitivity for all free dynamical variables
# const pars_true = [m, L, g, k]                    # true value of all free parameters
const pars_true = [m, L, k]#[m, L, g, k]                    # true value of all free parameters
get_all_θs(pars::Array{Float64,1}) = [pars[1], pars[2], g, pars[3]]
# Each row corresponds to lower and upper bounds of a free dynamic parameter.
# dyn_par_bounds = [0.01 1e4; 0.1 1e4; 0.1 1e4; 0.1 1e4]
dyn_par_bounds = [0.01 1e4; 0.1 1e4; 0.1 1e4]
learning_rate_vec(t::Int, grad_norm::Float64) = [0.1, 1.0, 1.0]  #NOTE Dimensions must be equal to number of free parameters
model_to_use = pendulum_sensitivity_sans_g
const num_dyn_pars = size(dyn_par_bounds, 1)
realize_model_sens(u::Function, w::Function, pars::Array{Float64, 1}, N::Int) = problem(
  model_to_use(φ0, u, w, get_all_θs(pars)),
  N,
  Ts,
)
realize_model(u::Function, w::Function, pars::Array{Float64, 1}, N::Int) = problem(
  pendulum_new(φ0, u, w, get_all_θs(pars)),
  N,
  Ts,
)
const dθ = length(get_all_θs(pars_true))

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

function get_estimates(expid::String, pars0::Array{Float64,1}, N_trans::Int = 0)
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
    dist_par_inds = W_meta.free_par_inds

    @assert (length(pars0) == num_dyn_pars+length(dist_par_inds)) "Please pass exactly $(num_dyn_pars+length(W_meta.free_par_inds)) parameter values"

    if !isdir(joinpath(data_dir, "tmp/"))
        mkdir(joinpath(data_dir, "tmp/"))
    end

    get_all_parameters(pars::Array{Float64, 1}) = vcat(get_all_θs(pars), exp_data.get_all_ηs(pars))
    par_bounds = vcat(dyn_par_bounds, exp_data.dist_par_bounds)

    # === We then optimize parameters for the baseline model ===
    function baseline_model_parametrized(dummy_input, pars)
        # NOTE: The true input is encoded in the solvew_sens()-function, but this function
        # still needs to to take two input arguments, so dummy_input could just be
        # anything, it's not used anyway
        Y_base = solvew(u, t -> zeros(n_out+length(dist_par_inds)*n_out), pars, N ) |> h

        # NOTE: SCALAR_OUTPUT is assumed
        return reshape(Y_base[N_trans+1:end,:], :)   # Returns 1D-array
    end

    function jacobian_model_b(dummy_input, pars)
        jac = solvew_sens(u, t -> zeros(n_out+length(dist_par_inds)*n_out), pars, N) |> h_sens
        return jac[N_trans+1:end, :]
    end

    # jacobian_model_b(dummy_input, pars) =
    #     solvew_sens(u, t -> zeros(n_out), pars, N) |> h_sens

    E = size(Y, 2)
    # # DEBUG
    # E = 6
    opt_pars_baseline = zeros(length(pars0), E)
    # trace_base[e][t][j] contains the value of parameter j before iteration t
    # corresponding to dataset e
    trace_base = [[pars0] for e=1:E]
    # for e=1:E
    #     # HACK: Uses trace returned due to hacked LsqFit package
    #     baseline_result, baseline_trace = get_fit_sens(Y[N_trans+1:end,e], pars0,
    #         baseline_model_parametrized, jacobian_model_b,
    #         par_bounds[:,1], par_bounds[:,2])
    #     opt_pars_baseline[:, e] = coef(baseline_result)
    #
    #     writedlm(joinpath(data_dir, "tmp/backup_baseline_e$e.csv"), opt_pars_baseline[:,e], ',')
    #
    #     # Sometimes (the first returned value I think) the baseline_trace
    #     # has no elements, and therefore doesn't contain the metadata x
    #     if length(baseline_trace) > 1
    #         for j=2:length(baseline_trace)
    #             push!(trace_base[e], baseline_trace[j].metadata["x"])
    #         end
    #     end
    # end

    @info "The mean optimal parameters for baseline are given by: $(mean(opt_pars_baseline, dims=2))"

    # === Finally we optimize parameters for the proposed model ==

    # Returns estimate of gradient of cost function
    # M_mean specifies over how many realizations the gradient estimate is computed
    function get_gradient_estimate(y, pars, isws, M_mean::Int=1)
        Ym, jacsYm = simulate_system_sens(exp_data, pars, 2M_mean, dist_par_inds, isws)
        # Uses different noise realizations for estimate of output and estiamte of jacobian
        grad_est = get_cost_gradient(y, Ym[:,1:M_mean], jacsYm[M_mean+1:end], N_trans)
        # grad_est will be 2D array with one dimenion equal to 1, we want to return a 1D array
        return grad_est[:]
    end

    # Returns estimate of gradient of output
    function get_proposed_jacobian(pars, isws, M_mean::Int=1)
        jacYm = simulate_system_sens(exp_data, pars, M_mean, dist_par_inds, isws)[2]
        return mean(jacYm, dims=2)
    end

    # # NOTE: Not used for SGD
    # # TODO: Should we consider storing and loading white noise, to improve repeatability
    # # of the results? Currently repeatability is based on fixed random seed
    # Zm = [randn(Nw, n_tot) for m = 1:M]

    get_gradient_estimate_p(pars, M_mean) = get_gradient_estimate(Y[:,1], pars, isws, M_mean)

    opt_pars_proposed = zeros(length(pars0), E)
    trace_proposed = [ [Float64[]] for e=1:E]
    for e=1:E
        # jacobian_model(x, p) = get_proposed_jacobian(pars, isws, M)  # NOTE: This won't give a jacobian estimate independent of Ym, but maybe we don't need that since this isn't SGD?
        opt_pars_proposed[:,e], trace_proposed[e] =
            perform_SGD_adam(get_gradient_estimate_p, pars0, par_bounds, verbose=true, tol=1e-6)
            # perform_stochastic_gradient_descent(get_gradient_estimate_p, pars0, par_bounds, verbose=true)

        writedlm(joinpath(data_dir, "tmp/backup_proposed_e$e.csv"), opt_pars_proposed[:,e], ',')

        # proposed_result, proposed_trace = get_fit(Y[:,e], pars0,
        #     (dummy_input, pars) -> proposed_model_parametrized(δ, Zm, dummy_input, pars, isws),
        #     par_bounds[:,1], par_bounds[:,2])
        # proposed_result, proposed_trace = get_fit_sens(Y[:,e], pars0,
        #     (dummy_input, pars) -> proposed_model_parametrized(δ, Zm, dummy_input, pars, isws), jacobian_model)
        # opt_pars_proposed[:, e] = coef(proposed_result)
        println("Completed for dataset $e for parameters $(opt_pars_proposed[:,e])")
    end

    @info "The mean optimal parameters for proposed method are given by: $(mean(opt_pars_proposed, dims=2))"

    return opt_pars_baseline, opt_pars_proposed, trace_base, trace_proposed
end

function get_outputs(expid::String, pars0::Array{Float64,1})
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
    dist_par_inds = W_meta.free_par_inds

    @assert (length(pars0) == num_dyn_pars+length(W_meta.free_par_inds)) "Please pass exactly $(num_dyn_pars+length(W_meta.free_par_inds)) parameter values"

    get_all_parameters(pars::Array{Float64, 1}) = vcat(get_all_θs(pars), exp_data.get_all_ηs(pars))
    p = get_all_parameters(pars0)
    θ = p[1:dθ]
    η = p[dθ+1: dθ+dη]
    # C = reshape(η[nx+1:end], (n_out, n_tot))

    # === Computes output of the baseline model ===
    Y_base, sens_base = solvew_sens(u, t -> zeros(n_out+length(dist_par_inds)*n_out), pars0, N) |> h_comp

    # === Computes outputs of the proposed model ===
    # TODO: Should we consider storing and loading white noise, to improve repeatability
    # of the results? Currently repeatability is based on fixed random seed
    Zm = [randn(Nw, n_tot) for m = 1:M]
    q_a = length(dist_par_inds[findall(dist_par_inds .<= nx)])
    n_sens = nx*(1+q_a)
    # Creating Z_sens should ensure that the white noise that is fed into the
    # nominal (non-sensitivity) part of the disturbance system is the same as
    # the noise in Zm, so that the disturbance model state should always give
    # the same realization given the same Zm, regardless of the number of free
    # disturbance parameters corresponding to the "a-vector" in the disturbance model
    Z_sens = [zeros(Nw, n_tot*(1+q_a)) for m = 1:M]
    for m = 1:M
        for i = 1:n_out
            Z_sens[m][:, (i-1)*n_sens+1:(i-1)*n_sens+nx] = Zm[m][:, (i-1)*nx+1:i*nx]
            Z_sens[m][:, (i-1)*n_sens+nx+1:i*n_sens] = randn(Nw, q_a*nx)
        end
    end
    # dmdl = discretize_ct_noise_model(get_ct_disturbance_model(η, nx, n_out), δ)
    dmdl = discretize_ct_noise_model_with_sensitivities(get_ct_disturbance_model(η, nx, n_out), δ, dist_par_inds)
    # NOTE: OPTION 1: Use the rows below here for linear interpolation
    XWm = simulate_noise_process_mangled(dmdl, Z_sens)
    wmm(m::Int) = mk_noise_interp(dmdl.Cd, XWm, m, δ)
    # # NOTE: OPTION 2: Use the rows below here for exact interpolation
    # reset_isws!(isws)
    # XWm = simulate_noise_process(dmdl, Zm)
    # wmm(m::Int) = mk_newer_noise_interp(view(η, 1:nx), C, XWm, m, n_in, δ, isws)

    calc_mean_y_N_prop(N::Int, pars::Array{Float64, 1}, m::Int) =
        solvew_sens(u, t -> wmm(m)(t), pars, N) |> h_comp
    calc_mean_y_prop(pars::Array{Float64, 1}, m::Int) = calc_mean_y_N_prop(N, pars, m)
    Ym_prop, sens_m_prop = solve_in_parallel_sens(m -> calc_mean_y_prop(pars0, m), ms)
    Y_mean_prop = reshape(mean(Ym_prop, dims = 2), :)

    return Y, Y_base, sens_base, Ym_prop, Y_mean_prop, sens_m_prop
end

function get_experiment_data(expid::String)::Tuple{ExperimentData, Array{InterSampleWindow, 1}}
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

    # NOTE: Has to be changed when number of free disturbance parameters is changed.
    # Specifically: Both vector free_pars and the corresponding vector
    #dist_par_bounds need to be updated

    # Use this function to specify which parameters should be free and optimized over
    # Each element represent whether the corresponding element in η is a free parameter
    free_pars = fill(false, size(η_true))       # Known disturbance model
    # free_pars = vcat(fill(false, nx), true, fill(false, n_tot*n_out-1))       # First parameter of c-vector unknown
    # free_pars = vcat(true, fill(false, nx-1), false, fill(false, n_tot*n_out-1))       # First parameter of a-vector and first parameter of c-vector unknown
    free_par_inds = findall(free_pars)          # Indices of free variables in η. Assumed to be sorted in ascending order.
    # Array of tuples containing lower and upper bound for each free disturbance parameter
    dist_par_bounds = Array{Float64}(undef, 0, 2)
    # dist_par_bounds = [-Inf Inf]
    function get_all_ηs(pars::Array{Float64, 1})
        all_η = η_true
        all_η[free_par_inds] = pars[num_dyn_pars+1:end]
        return all_η
     end

    # compute the maximum number of steps we can take
    N_margin = 2    # Solver can request values of inputs after the time horizon
                    # ends, so we require a margin of a few samples of the noise
                    # to ensure that we can provide such values
    # Minimum of number of available disturbance or input samples
    Nw = min(size(XW, 1)÷n_tot, size(input, 1)÷n_in)
    N = Int((Nw - N_margin)*δ÷Ts)     # Number of steps we can take
    W_meta = DisturbanceMetaData(nx, n_in, n_out, η_true, free_par_inds)

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

    # # This block can be used to check whether different implementations result
    # # in the same Y
    # @warn "Debugging sim. First 5 XW: $(XW[1:5])"
    # wdebug = mk_noise_interp(C_true, XW, 1, δ)
    # my_y = solvew(u, wdebug, pars_true, N) |> h
    # writedlm("data/experiments/pendulum_sensitivity/my_y.csv", my_y, ',')

    reset_isws!(isws)
    return ExperimentData(Y, u, get_all_ηs, dist_par_bounds, W_meta, Nw), isws
end

function perform_stochastic_gradient_descent(
    get_grad_estimate::Function,
    pars0::Array{Float64,1},                        # Initial guess for parameters
    bounds::Array{Float64, 2},                      # Parameter bounds
    learning_rate::Function=learning_rate;
    tol::Float64=1e-4,
    maxiters=200,
    verbose=false)

    q = lr_break+50
    last_q_norms = fill(Inf, q)
    running_criterion() = mean(last_q_norms) > tol

    t = 1
    pars = pars0
    trace = [pars]
    while t <= maxiters
        grad_est = get_grad_estimate(pars, M_rate(t))
        # println("Iteration $t, gradient norm $(norm(grad_est)) and step sign $(sign(-first(grad_est))) with parameter estimate $pars")
        # running_criterion(grad_est) || break
        if verbose
            println("Iteration $t, average gradient norm $(mean(last_q_norms)) and step sign $(sign(-first(grad_est))) with parameter estimate $pars")
        end
        running_criterion() || break
        pars = pars - learning_rate(t, norm(grad_est))*grad_est
        project_on_bounds!(pars, bounds)
        push!(trace, pars)
        # (t-1)%10+1 maps t to a number between 1 and 10 (inclusive)
        last_q_norms[(t-1)%q+1] = norm(grad_est)
        t += 1
    end
    return pars, trace
end

# Uses alternative stopping criterion based on variance in the last 10
# parameter estimates, instead of average gradient norm
function perform_stochastic_gradient_descent_alt(
    get_grad_estimate::Function,
    pars0::Array{Float64,1},                        # Initial guess for parameters
    bounds::Array{Float64, 2},                      # Parameter bounds
    learning_rate::Function=learning_rate;
    tol::Float64=1e-6,
    maxiters=200,
    verbose=false)

    last_10_params = zeros(10, length(pars0))
    last_10_params[end,:] .= 10000  # Fills one parameter with some large number
    # mean((last_10_params - mean(last_10_params, dims=1)).^2) is the trace of
    # the covariance matrix of the last 10 parameters, divided by 10*length(pars0)
    running_criterion() = mean((last_10_params .- mean(last_10_params, dims=1)).^2) > tol

    t = 1
    pars = pars0
    trace = [pars]
    while t <= maxiters
        grad_est = get_grad_estimate(pars, M_rate(t))
        # println("Iteration $t, gradient norm $(norm(grad_est)) and step sign $(sign(-first(grad_est))) with parameter estimate $pars")
        # running_criterion(grad_est) || break
        if verbose
            println("Iteration $t, scaled trace of covariance $(mean((last_10_params .- mean(last_10_params, dims=1)).^2)) and normalized gradient $(grad_est./(norm(grad_est))) and non-normalized gradient $(grad_est), with parameter estimate $pars")
        end
        running_criterion() || break
        pars = pars - learning_rate(t, norm(grad_est))*grad_est
        project_on_bounds!(pars, bounds)
        push!(trace, pars)
        # (t-1)%10+1 maps t to a number between 1 and 10 (inclusive)
        last_10_params[(t-1)%10+1, :] = pars
        t += 1
    end
    return pars, trace
end

function perform_SGD_adam(
    get_grad_estimate::Function,
    pars0::Array{Float64,1},                        # Initial guess for parameters
    bounds::Array{Float64, 2},                      # Parameter bounds
    learning_rate::Function=learning_rate;
    tol::Float64=1e-6,
    maxiters=200,
    verbose=false,
    # betas::Array{Float64,1} = [0.9, 0.999])   # betas are the decay parameters of moment estimates
    betas::Array{Float64,1} = [0.5, 0.999])   # betas are the decay parameters of moment estimates

    # eps = 1e-8
    eps = 1e-13
    q = 20#lr_break+10# DEBUG
    last_q_norms = fill(Inf, q)
    running_criterion() = mean(last_q_norms) > tol
    s = zeros(size(pars0)) # First moment estimate
    r = zeros(size(pars0)) # Second moment estimate

    t = 1
    pars = pars0
    trace = [pars]
    while t <= maxiters
        grad_est = get_grad_estimate(pars, M_rate(t))

        # # DEBUG
        # if norm(grad_est) > 1e-5
        #     grad_est = (1e-5)*(grad_est/norm(grad_est))
        # end
        s = betas[1]*s + (1-betas[1])*grad_est
        r = betas[2]*r + (1-betas[2])*(grad_est.^2)
        shat = s/(1-betas[1]^t)
        rhat = r/(1-betas[2]^t)
        # step = -learning_rate_adam(t, norm(grad_est))*shat./(sqrt.(rhat).+eps)
        step = -learning_rate_vec(t, norm(grad_est)).*shat./(sqrt.(rhat).+eps)
        # println("Iteration $t, gradient norm $(norm(grad_est)) and step sign $(sign(-first(grad_est))) with parameter estimate $pars")
        # running_criterion(grad_est) || break
        if verbose
            println("Iteration $t, average gradient norm $(mean(last_q_norms)), -gradient $(-grad_est) and step $(step) with parameter estimate $pars")
            println("-s: $(-s), -shat: $(-shat), rhat: $(rhat)")
        end
        running_criterion() || break
        pars = pars + step
        project_on_bounds!(pars, bounds)
        push!(trace, pars)
        # (t-1)%10+1 maps t to a number between 1 and 10 (inclusive)
        last_q_norms[(t-1)%q+1] = norm(grad_est)
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
    pars::Array{Float64,1},
    M::Int,
    dist_sens_inds::Array{Int64, 1},
    isws::Array{InterSampleWindow,1},
    Zm::Array{Array{Float64,2},1})::Array{Float64,2}

    W_meta = exp_data.W_meta
    nx = W_meta.nx
    n_in = W_meta.n_in
    n_out = W_meta.n_out
    n_tot = nx*n_in
    dη = length(W_meta.η)
    N = size(exp_data.Y, 1)-1

    p = vcat(get_all_θs(pars), exp_data.get_all_ηs(pars))
    θ = p[1:dθ]
    η = p[dθ+1: dθ+dη]
    C = reshape(η[nx+1:end], (n_out, n_tot))

    # dmdl = discretize_ct_noise_model(get_ct_disturbance_model(η, nx, n_out), δ)
    dmdl = discretize_ct_noise_model_with_sensitivities(get_ct_disturbance_model(η, nx, n_out), δ, dist_sens_inds)
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
    return solve_in_parallel(m -> calc_mean_y(pars, m), collect(1:M))   # Returns Ym
end

# Simulates system with newly generated white noise
function simulate_system(
    exp_data::ExperimentData,
    pars::Array{Float64,1},
    M::Int,
    dist_sens_inds::Array{Int64, 1},
    isws::Array{InterSampleWindow,1})::Array{Float64,2}

    W_meta = exp_data.W_meta
    nx = W_meta.nx
    n_in = W_meta.n_in
    n_tot = nx*n_in
    Nw = exp_data.Nw
    Zm = [randn(Nw, n_tot) for m = 1:M]
    simulate_system(exp_data, pars, M, dist_sens_inds, isws, Zm)
end

# Simulates system with specified white noise
function simulate_system_sens(
    exp_data::ExperimentData,
    pars::Array{Float64,1},
    M::Int,
    dist_sens_inds::Array{Int64, 1},
    isws::Array{InterSampleWindow,1},
    Zm::Array{Array{Float64,2},1})::Tuple{Array{Float64,2}, Array{Array{Float64,2},1}}

    W_meta = exp_data.W_meta
    nx = W_meta.nx
    n_in = W_meta.n_in
    n_out = W_meta.n_out
    n_tot = nx*n_in
    dη = length(W_meta.η)
    N = size(exp_data.Y, 1)-1
    dist_par_inds = W_meta.free_par_inds

    p = vcat(get_all_θs(pars), exp_data.get_all_ηs(pars))
    θ = p[1:dθ]
    η = p[dθ+1: dθ+dη]
    # C = reshape(η[nx+1:end], (n_out, n_tot))

    dmdl = discretize_ct_noise_model_with_sensitivities(get_ct_disturbance_model(η, nx, n_out), δ, dist_sens_inds)
    # # NOTE: OPTION 1: Use the rows below here for linear interpolation
    XWm = simulate_noise_process_mangled(dmdl, Zm)
    wmm(m::Int) = mk_noise_interp(dmdl.Cd, XWm, m, δ)
    # NOTE: OPTION 2: Use the rows below here for exact interpolation
    # reset_isws!(isws)
    # XWm = simulate_noise_process(dmdl, Zm)
    # wmm(m::Int) = mk_newer_noise_interp(view(η, 1:nx), C, XWm, m, n_in, δ, isws)

    calc_mean_y_N(N::Int, pars::Array{Float64, 1}, m::Int) =
        solvew_sens(exp_data.u, t -> wmm(m)(t), pars, N) |> h_comp
    calc_mean_y(pars::Array{Float64, 1}, m::Int) = calc_mean_y_N(N, pars, m)
    return solve_in_parallel_sens(m -> calc_mean_y(pars, m), collect(1:M))  # Returns Ym and JacsYm
end

# Simulates system with newly generated white noise
function simulate_system_sens(
    exp_data::ExperimentData,
    pars::Array{Float64,1},
    M::Int,
    dist_sens_inds::Array{Int64, 1},
    isws::Array{InterSampleWindow,1})::Tuple{Array{Float64,2}, Array{Array{Float64,2},1}}

    W_meta = exp_data.W_meta
    nx = W_meta.nx
    n_in = W_meta.n_in
    n_tot = nx*n_in
    Nw = exp_data.Nw
    Zm = [randn(Nw, n_tot) for m = 1:M]
    simulate_system_sens(exp_data, pars, M, dist_sens_inds, isws, Zm)
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

function read_from_backup(dir::String, E::Int)
    sample = readdlm(joinpath(dir, "backup_baseline_e1.csv"), ',')
    k = length(sample)
    opt_pars_baseline = zeros(E,k)
    opt_pars_proposed = zeros(E,k)
    for e=1:E
        opt_pars_baseline[e,:] = readdlm(joinpath(dir, "backup_baseline_e$e.csv"), ',')
        opt_pars_proposed[e,:] = readdlm(joinpath(dir, "backup_proposed_e$e.csv"), ',')
    end
    return opt_pars_baseline, opt_pars_proposed
end

function debug_minimization(expid::String, pars0::Array{Float64,1}, N_trans::Int = 0)
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
    dist_sens_inds = W_meta.free_par_inds

    get_all_parameters(pars::Array{Float64, 1}) = vcat(get_all_θs(pars), exp_data.get_all_ηs(pars))

    # === Optimizing parameters for the baseline model ===
    function baseline_model_parametrized(δ, dummy_input, pars)
        # NOTE: The true input is encoded in the solvew()-function, but this function
        # still needs to to take two input arguments, so dummy_input could just be
        # anything, it's not used anyway
        Y_base = solvew(u, t -> zeros(n_out), pars, N ) |> h

        # NOTE: SCALAR_OUTPUT of system assumed here
        return reshape(Y_base[N_trans+1:end,:], :)   # Returns 1D-array
    end

    function jacobian_model_b(x, pars)
        jac = solvew_sens(u, t -> 0.0, pars, N) |> h_sens
        return jac[N_trans+1:end, :]
    end

    # E = size(Y, 2)
    # DEBUG
    E = 1
    opt_pars_baseline = zeros(length(pars0), E)
    trace_base = [[pars0] for e=1:E]
    for e=1:E
    # for e=4:4
        # HACK: Uses trace returned due to hacked LsqFit package
        baseline_result, baseline_trace = get_fit_sens(Y[N_trans+1:end,e], pars0,
            (dummy_input, pars) -> baseline_model_parametrized(δ, dummy_input, pars),
            jacobian_model_b, par_bounds[:,1], par_bounds[:,2])
        opt_pars_baseline[:, e] = coef(baseline_result)
        if length(baseline_trace) > 1
            for j=2:length(baseline_trace)
                push!(trace_base[e], baseline_trace[j].metadata["x"])
            end
        end
    end

    @info "The mean optimal parameters for baseline are given by: $(mean(opt_pars_baseline, dims=2))"

    # === Computing cost function of baseline model
    base_par_vals = collect( (pars0[1]-2.0):0.1:(pars0[1]+2.0))
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
    # TODO: Should we consider storing and loading white noise, to improve repeatability
    # of the results? Currently repeatability is based on fixed random seed
    Zm = [randn(Nw, n_tot) for m = 1:M]

    # NOTE: CURRENTLY ONLY TREATS SCALAR PARAMETERS
    @info "Plotting proposed cost function..."
    # vals1 = 1.0:0.25:8.0
    # vals2 = 9.0:2.0:35.0
    prop_par_vals = 1.0:2.0:30.0    # DEBUG
    # prop_par_vals = vcat(vals1, vals2)
    prop_cost_vals = zeros(length(prop_par_vals), E)
    for ind = 1:length(prop_par_vals)
        # NOTE: If we don't pass Zm here, we will see that the cost function
        # looks very irregular even with M = 500. That's a bit suprising,
        # I would expect it to average out over that many iterations
        Ym = simulate_system(exp_data, [prop_par_vals[ind]], M, dist_sens_inds, isws, Zm)
        Y_mean = mean(Ym, dims=2)
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

        dmdl = discretize_ct_noise_model(get_ct_disturbance_model(η, nx, n_out), δ) # TODO: Use new discretizations
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

    # Returns estimate of gradient of cost function
    # M_mean specified over how many realizations the gradient estimate is computed
    function get_gradient_estimate(y, δ, pars, isws, M_mean::Int=1)
        Ym, jacsYm = simulate_system_sens(exp_data, pars, 2M_mean, dist_sens_inds, isws)
        # Uses different noise realizations for estimate of output and estiamte of gradient
        grad_est = get_cost_gradient(y, Ym[:,1:M_mean], jacsYm[M_mean+1:end], N_trans)
        # grad_est will be 2D array with one dimenion equal to 1, we want to return a 1D array
        return grad_est[:]
    end

    # Returns estimate of gradient of cost function
    # M_mean specified over how many realizations the gradient estimate is computed
    function get_gradient_estimate_debug(y, δ, pars, isws, M_mean::Int=1)
        Ym, jacsYm = simulate_system_sens(exp_data, pars, 2M_mean, dist_sens_inds, isws)
        # Uses different noise realizations for estimate of output and estiamte of gradient
        grad_est = get_cost_gradient(y, Ym[:,1:M_mean], jacsYm[M_mean+1:end], N_trans)
        cost = mean((y[N_trans+1:end]-Ym[N_trans+1:end,1]).^2)
        # grad_est will be 2D array with one dimenion equal to 1, we want to return a 1D array
        return grad_est[:], cost
    end

    # Returns estimate of gradient of output
    function get_proposed_jacobian(pars, isws, M_mean::Int=1)
        jacsYm = simulate_system_sens(exp_data, pars, M_mean, dist_sens_inds, isws)[2]
        return mean(jacsYm)[N_trans+1:end, :]
    end

    @info "Finding proposed minimum..."

    opt_pars_proposed = zeros(length(pars0), E)
    opt_pars_proposed_LSQ = zeros(length(pars0), E)
    trace_proposed = [[Float64[]] for e=1:E]
    trace_costs = [Float64[] for e=1:E]
    grad_norms  = [Float64[] for e=1:E]
    for e=1:E
    # for e=4:4
        get_gradient_estimate_p(pars, M_mean) = get_gradient_estimate(Y[:,e], δ, pars, isws, M_mean)
        get_gradient_estimate_p_debug(pars, M_mean) = get_gradient_estimate_debug(Y[:,e], δ, pars, isws, M_mean)
        jacobian_model(x, p) = get_proposed_jacobian(p, isws, M)
        # opt_pars_proposed[:,e], trace_proposed[e] =
        #     perform_stochastic_gradient_descent(get_gradient_estimate_p, pars0, par_bounds, verbose=true)

        opt_pars_proposed[:,e], trace_proposed[e], trace_costs[e], grad_norms[e] =
            perform_SGD_adam_debug(get_gradient_estimate_p_debug, pars0, par_bounds, verbose=true; maxiters=300, tol=1e-8)
            # perform_stochastic_gradient_descent_debug(get_gradient_estimate_p_debug, pars0, par_bounds, verbose=true; maxiters=300, tol=1e-8)
        reset_isws!(isws)
        proposed_result, proposed_trace = get_fit_sens(Y[N_trans+1:end,e], pars0,
            (dummy_input, pars) -> proposed_model_parametrized(δ, Zm, dummy_input, pars, isws),
            jacobian_model, par_bounds[:,1], par_bounds[:,2])

        # proposed_result, proposed_trace = get_fit_sens(Y[:,e], pars0,
        #     (dummy_input, pars) -> proposed_model_parametrized(δ, Zm, dummy_input, pars, isws), jacobian_model)
        # opt_pars_proposed[:, e] = coef(proposed_result)
        opt_pars_proposed_LSQ[:, e] = coef(proposed_result)

        println("Completed for dataset $e for parameters $(opt_pars_proposed[:,e])")
    end

    @info "The mean optimal parameters for proposed method are given by: $(mean(opt_pars_proposed, dims=2))"

    # trace_proposed is an array with E elements, where each element is an array
    # of parameters that the SGD has gone through, and where every parameter
    # is an array of Float64-values
    return opt_pars_baseline, opt_pars_proposed, trace_base, trace_proposed, base_cost_vals, prop_cost_vals, trace_costs, base_par_vals, prop_par_vals, opt_pars_proposed_LSQ, grad_norms
end

function debug_minimization_2pars(expid::String, pars0::Array{Float64,1}, N_trans::Int = 0)
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
    dist_sens_inds = W_meta.free_par_inds

    get_all_parameters(pars::Array{Float64, 1}) = vcat(get_all_θs(pars), exp_data.get_all_ηs(pars))

    # === Optimizing parameters for the baseline model ===
    @info "Finding optimal parameters for baseline model"
    function baseline_model_parametrized(δ, dummy_input, pars)
        # NOTE: The true input is encoded in the solvew()-function, but this function
        # still needs to to take two input arguments, so dummy_input could just be
        # anything, it's not used anyway
        Y_base = solvew(u, t -> zeros(n_out), pars, N ) |> h

        # NOTE: SCALAR_OUTPUT of system assumed here
        return reshape(Y_base[N_trans+1:end,:], :)   # Returns 1D-array
    end

    function jacobian_model_b(x, pars)
        jac = solvew_sens(u, t -> 0.0, pars, N) |> h_sens
        return jac[N_trans+1:end, :]
    end

    # E = size(Y, 2)
    # DEBUG
    E = 1
    opt_pars_baseline = zeros(length(pars0), E)
    trace_base = [[pars0] for e=1:E]
    # for e=1:E
    #     # HACK: Uses trace returned due to hacked LsqFit package
    #     baseline_result, baseline_trace = get_fit_sens(Y[N_trans+1:end,e], pars0,
    #         (dummy_input, pars) -> baseline_model_parametrized(δ, dummy_input, pars),
    #         jacobian_model_b, par_bounds[:,1], par_bounds[:,2])
    #     opt_pars_baseline[:, e] = coef(baseline_result)
    #     if length(baseline_trace) > 1
    #         for j=2:length(baseline_trace)
    #             push!(trace_base[e], baseline_trace[j].metadata["x"])
    #         end
    #     end
    # end

    @info "The mean optimal parameters for baseline are given by: $(mean(opt_pars_baseline, dims=2))"
    @info "Plotting baseline cost function"

    # # === Computing cost function of baseline model

    # base_par_diffs1 = -0.6:0.2:0.6
    # base_par_diffs2 = -0.6:0.2:0.6
    # base_par_vals1 = opt_pars_baseline[1,1] .+ base_par_diffs1
    # base_par_vals2 = opt_pars_baseline[2,1] .+ base_par_diffs2
    # base_par_vals1 = 0.1:0.2:opt_pars_baseline[1,1]+0.4   # For m_true = 0.3
    # base_par_vals2 = (4.25-0.6):0.2:opt_pars_baseline[2,1]+0.4
    base_par_vals1 = 0.1:0.1:5.0
    base_par_vals2 = [4.25]

    base_cost_vals = [zeros(length(base_par_vals1), length(base_par_vals2)) for e=1:E]
    base_cost_true = zeros(E)

    # DEBUG!!!!
    # for ind1 = 1:length(base_par_vals1)
    #     for ind2 = 1:length(base_par_vals2)
    #         for e = 1:E
    #             pars = [base_par_vals1[ind1], base_par_vals2[ind2]]
    #             Yb = solvew(u, t -> zeros(n_out), pars, N) |> h
    #             base_cost_vals[e][ind1, ind2] = mean((Y[N_trans+1:end, e].-Yb[N_trans+1:end]).^2)
    #         end
    #     end
    # end
    # Yb_true = solvew(u, t -> zeros(n_out), pars_true, N) |> h
    # for e = 1:E
    #     base_cost_true[e] = mean((Y[N_trans+1:end, e].-Yb_true[N_trans+1:end]).^2)
    # end

    # === Computing cost function for proposed model ===
    # TODO: Should we consider storing and loading white noise, to improve repeatability
    # of the results? Currently repeatability is based on fixed random seed
    Zm = [randn(Nw, n_tot) for m = 1:M]

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

    # Returns estimate of gradient of cost function
    # M_mean specified over how many realizations the gradient estimate is computed
    function get_gradient_estimate(y, δ, pars, isws, M_mean::Int=1)
        Ym, jacsYm = simulate_system_sens(exp_data, pars, 2M_mean, dist_sens_inds, isws)
        # Uses different noise realizations for estimate of output and estiamte of gradient
        grad_est = get_cost_gradient(y, Ym[:,1:M_mean], jacsYm[M_mean+1:end], N_trans)
        # grad_est will be 2D array with one dimenion equal to 1, we want to return a 1D array
        return grad_est[:]
    end

    # Returns estimate of gradient of cost function
    # M_mean specified over how many realizations the gradient estimate is computed
    function get_gradient_estimate_debug(y, δ, pars, isws, M_mean::Int=1)
        Ym, jacsYm = simulate_system_sens(exp_data, pars, 2M_mean, dist_sens_inds, isws)
        # Uses different noise realizations for estimate of output and estiamte of gradient
        grad_est = get_cost_gradient(y, Ym[:,1:M_mean], jacsYm[M_mean+1:end], N_trans)
        cost = mean((y[N_trans+1:end]-Ym[N_trans+1:end,1]).^2)
        # grad_est will be 2D array with one dimenion equal to 1, we want to return a 1D array
        return grad_est[:], cost
    end

    # Returns estimate of gradient of output
    function get_proposed_jacobian(pars, isws, M_mean::Int=1)
        jacYm = simulate_system_sens(exp_data, pars, M_mean, dist_sens_inds, isws)[2]
        return mean(jacYm)
    end

    @info "Finding proposed minimum..."

    opt_pars_proposed = zeros(length(pars0), E)
    opt_pars_proposed_LSQ = zeros(length(pars0), E)
    trace_proposed = [[Float64[]] for e=1:E]
    trace_costs = [Float64[] for e=1:E]
    grad_norms  = [Float64[] for e=1:E]
    # for e=1:E
    #     get_gradient_estimate_p(pars, M_mean) = get_gradient_estimate(Y[:,e], δ, pars, isws, M_mean)
    #     get_gradient_estimate_p_debug(pars, M_mean) = get_gradient_estimate_debug(Y[:,e], δ, pars, isws, M_mean)
    #     jacobian_model(x, p) = get_proposed_jacobian(p, isws, M)
    #     # opt_pars_proposed[:,e], trace_proposed[e] =
    #     #     perform_stochastic_gradient_descent(get_gradient_estimate_p, pars0, par_bounds, verbose=true)
    #     opt_pars_proposed[:,e], trace_proposed[e], trace_costs[e], grad_norms[e] =
    #         perform_SGD_adam_debug(get_gradient_estimate_p_debug, pars0, par_bounds, verbose=true; maxiters=300, tol=1e-8)
    #         # perform_stochastic_gradient_descent_debug(get_gradient_estimate_p_debug, pars0, par_bounds, verbose=true; maxiters=300, tol=1e-4)
    #     reset_isws!(isws)
    #     # proposed_result, proposed_trace = get_fit_sens(Y[:,e], pars0,
    #     #     (dummy_input, pars) -> proposed_model_parametrized(δ, Zm, dummy_input, pars, isws),
    #     #     jacobian_model, par_bounds[:,1], par_bounds[:,2])
    #     # # proposed_result, proposed_trace = get_fit_sens(Y[:,e], pars0,
    #     # #     (dummy_input, pars) -> proposed_model_parametrized(δ, Zm, dummy_input, pars, isws), jacobian_model)
    #     # # opt_pars_proposed[:, e] = coef(proposed_result)
    #     # opt_pars_proposed_LSQ[:, e] = coef(proposed_result)
    #
    #     println("Completed for dataset $e for parameters $(opt_pars_proposed[:,e])")
    # end

    @info "The mean optimal parameters for proposed method are given by: $(mean(opt_pars_proposed, dims=2))"

    @info "Plotting proposed cost function..."
    # prop_par_diffs1 = -0.6:0.2:0.6
    # prop_par_diffs2 = -0.6:0.2:0.6
    # prop_par_vals1 = opt_pars_proposed[1,1] .+ prop_par_diffs1
    # prop_par_vals2 = opt_pars_proposed[2,1] .+ prop_par_diffs2
    prop_par_vals1 = 0.1:0.2:opt_pars_proposed[1,1]+0.4   # For m_true = 0.3
    prop_par_vals2 = (4.25-0.6):0.2:opt_pars_proposed[2,1]+0.4
    # prop_par_vals1 = 0.1:0.1:5.0
    # prop_par_vals2 = [4.25]
    prop_cost_vals = [zeros(length(prop_par_vals1), length(prop_par_vals2)) for e=1:E]
    prop_cost_true = zeros(E)
    for ind1 = 1:length(prop_par_vals1)
        for ind2 = 1:length(prop_par_vals2)
            for e=1:E
                pars = [prop_par_vals1[ind1], prop_par_vals2[ind2]]
                Ym = mean(simulate_system(exp_data, pars, M, dist_sens_inds, isws, Zm), dims=2)
                prop_cost_vals[e][ind1, ind2] = mean((Y[N_trans+1:end, e].-Ym[N_trans+1:end]).^2)
            end
        end
    end
    Ym_true = mean(simulate_system(exp_data, pars_true, M, dist_sens_inds, isws, Zm), dims=2)
    for e=1:E
        prop_cost_true[e] = mean((Y[N_trans+1:end, e].-Ym_true[N_trans+1:end]).^2)
    end


    # Yms, jacsYm = simulate_system_sens(exp_data, [1.796586589667771, 5.7244856197562175], 100, dist_sens_inds, isws)
    # my_grad_est = get_cost_gradient(Y[:,1], Yms[:,1:50], jacsYm[51:end], N_trans)

    #NOTE: To get interactive 3D-plot, do as follows:
    # > using Plots
    # > plotlyjs()
    # > plot(prop_par_vals[1][:,1], prop_par_vals[2][:,e], prop_cost_vals[e], st=:surface, xlabel="m values", ylabel="k values", zlabel="cost", size=(1200,1000))
    # > plot(prop_par_vals[1][:,1], prop_par_vals[2][:,1], prop_cost_vals[1], st=:surface, xlabel="m values", ylabel="k values", zlabel="cost", size=(1200,1000))
    # NOTE: Make sure x- and y-values are a 1D-vector, NOT a flat 2D-vector

    base_par_vals = (base_par_vals1, base_par_vals2)
    prop_par_vals = (prop_par_vals1, prop_par_vals2)
    # return Yms, jacsYm, my_grad_est
    return opt_pars_baseline, opt_pars_proposed, trace_base, trace_proposed, base_cost_vals, prop_cost_vals, base_par_vals, prop_par_vals, base_cost_true, prop_cost_true, temp_res
end

function debug_2par_simulation(expid::String, pars0::Array{Float64, 1}, N_trans::Int = 0)
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
    # E = size(Y, 2)
    # DEBUG
    E = 1

    # Overwrites solvew()-functions
    realize_model_sens1(u::Function, w::Function, pars::Array{Float64, 1}, N::Int) = problem(
      pendulum_sensitivity(φ0, u, w, get_all_θs(pars)),
      N,
      Ts,
    )
    realize_model_sens2(u::Function, w::Function, pars::Array{Float64, 1}, N::Int) = problem(
      pendulum_sensitivity2(φ0, u, w, get_all_θs(pars)),
      N,
      Ts,
    )

    solvew_sens1(u::Function, w::Function, pars::Array{Float64, 1}, N::Int; kwargs...) = solve(
      realize_model_sens1(u, w, pars, N),
      saveat = 0:Ts:(N*Ts),
      abstol = abstol,
      reltol = reltol,
      maxiters = maxiters;
      kwargs...,
    )
    solvew_sens2(u::Function, w::Function, pars::Array{Float64, 1}, N::Int; kwargs...) = solve(
      realize_model_sens1(u, w, pars, N),
      saveat = 0:Ts:(N*Ts),
      abstol = abstol,
      reltol = reltol,
      maxiters = maxiters;
      kwargs...,
    )

    get_all_parameters(pars::Array{Float64, 1}) = vcat(get_all_θs(pars), exp_data.get_all_ηs(pars))

    # === Computing cost function of baseline model
    # base_par_vals1 = collect( (pars0[1]-2.0):0.1:(pars0[1]+2.0))
    # base_par_vals2 = collect( (pars0[2]-2.0):0.1:(pars0[2]+2.0))
    #DEBUG
    base_par_vals = 1.0:0.5:10.0
    second_par = 6.25
    base_cost_vals = zeros(length(base_par_vals), E)
    base_cost_vals1 = zeros(length(base_par_vals), E)
    base_cost_vals2 = zeros(length(base_par_vals), E)
    for ind = 1:length(base_par_vals)
            pars = [base_par_vals[ind], second_par]
            Yb  = solvew(exp_data.u, t -> zeros(n_out), pars, N) |> h
            Yb1 = solvew_sens1(exp_data.u, t -> zeros(n_out), pars, N) |> h
            Yb2  = solvew_sens2(exp_data.u, t -> zeros(n_out), pars, N) |> h
            for e=1:E
                # NOTE: This part can be coded more efficiently
                base_cost_vals[ind, e]  = mean((Y[N_trans+1:end, e].-Yb[N_trans+1:end]).^2)
                base_cost_vals1[ind, e] = mean((Y[N_trans+1:end, e].-Yb1[N_trans+1:end]).^2)
                base_cost_vals2[ind, e] = mean((Y[N_trans+1:end, e].-Yb2[N_trans+1:end]).^2)
            end
    end

    return base_par_vals, second_par, base_cost_vals, base_cost_vals1, base_cost_vals2
end

function debug_grad_norm(expid::String, pars0::Array{Float64,1}, N_trans::Int = 0)
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
    dist_sens_inds = W_meta.free_par_inds

    get_all_parameters(pars::Array{Float64, 1}) = vcat(get_all_θs(pars), exp_data.get_all_ηs(pars))

    # E = size(Y, 2)
    # DEBUG
    E = 6

    # === Computing cost function for proposed model ===
    # TODO: Should we consider storing and loading white noise, to improve repeatability
    # of the results? Currently repeatability is based on fixed random seed
    Zm = [randn(Nw, n_tot) for m = 1:M]

    # NOTE: CURRENTLY ONLY TREATS SCALAR PARAMETERS
    @info "Plotting proposed cost function..."
    # prop_par_vals = 1.0:0.5:35.0
    vals1 = 1.0:0.25:8.0
    vals2 = 9.0:2.0:35.0
    # prop_par_vals = 1.0:2.0:35.0    # DEBUG
    prop_par_vals = vcat(vals1, vals2)
    prop_cost_vals = zeros(length(prop_par_vals), E)
    grad_vals = zeros(length(prop_par_vals), E)
    for ind = 1:length(prop_par_vals)
        # NOTE: If we don't pass Zm here, we will see that the cost function
        # looks very irregular even with M = 500. That's a bit suprising,
        # I would expect it to average out over that many iterations
        Yms = simulate_system(exp_data, [prop_par_vals[ind]], M, dist_sens_inds, isws, Zm)
        _, jacsYm = simulate_system_sens(exp_data, [prop_par_vals[ind]], M_rate_max, dist_sens_inds, isws)
        Y_mean = mean(Yms, dims=2)
        for e=1:E
            grad_vals[ind,e] = first(get_cost_gradient(Y[:,e], Y_mean, jacsYm))
        end

        # Y has columns indexed with 1:E because in case E is changed for debugging purposes,
        # without the dimensions of Y changing, we can get a mismatch otherwise
        prop_cost_vals[ind,:] = mean((Y[N_trans+1:end, 1:E].-Y_mean[N_trans+1:end]).^2, dims=1)
    end

    return prop_par_vals, prop_cost_vals, grad_vals
    # return opt_pars_baseline, opt_pars_proposed, trace_base, trace_proposed, base_cost_vals, prop_cost_vals, trace_costs, base_par_vals, prop_par_vals, opt_pars_proposed_LSQ, grad_norms
end

# function debug_1par_simulation(expid, pars0::Array{Float64, 1}, N_trans::Int = 0)
#     exp_data, isws = get_experiment_data(expid)
#     W_meta = exp_data.W_meta
#     Y = exp_data.Y
#     N = size(Y,1)-1
#     Nw = exp_data.Nw
#     nx = W_meta.nx
#     n_in = W_meta.n_in
#     n_out = W_meta.n_out
#     n_tot = nx*n_in
#     dη = length(W_meta.η)
#     u = exp_data.u
#     par_bounds = vcat(dyn_par_bounds, exp_data.dist_par_bounds)
#     # E = size(Y, 2)
#     # DEBUG
#     E = 1
#
#     get_all_parameters(pars::Array{Float64, 1}) = vcat(get_all_θs(pars), exp_data.get_all_ηs(pars))
#
#     # === Computing cost function of baseline model
#     # base_par_vals1 = collect( (pars0[1]-2.0):0.1:(pars0[1]+2.0))
#     # base_par_vals2 = collect( (pars0[2]-2.0):0.1:(pars0[2]+2.0))
#     #DEBUG
#     base_par_vals = 1.0:0.5:10.0
#     second_par = 6.25
#     base_cost_vals = zeros(length(base_par_vals), E)
#     for ind = 1:length(base_par_vals)
#             pars = [base_par_vals[ind], second_par]
#             Yb  = solvew(exp_data.u, t -> zeros(n_out), pars, N) |> h
#             for e=1:E
#                 # NOTE: This part can be coded more efficiently
#                 base_cost_vals[ind, e]  = mean((Y[N_trans+1:end, e].-Yb[N_trans+1:end]).^2)
#             end
#     end
#
#     # TODO: Okay, all these cost functions get a weird shape, try reverting all code
#     # back to one-parameter case and run some experiments to see that it still works
#     # and what shape of the cost function we expect there! TODO: TODO: TODO:
#     return base_par_vals, base_cost_vals
# end

function debug_proposed_cost_func(expid::String, par_vector::Array{Array{Float64,1},1}, N_trans::Int = 0)
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
    cost_gradients = [zeros(length(par_vector)) for M in Ms]

    @info "Computing cost function"
    # ============== Computing Cost Functions =====================
    for par_ind = 1:length(par_vector)
        Ym = simulate_system(exp_data, par_vector[par_ind], Mmax, isws)
        for (M_ind, M) in enumerate(Ms)
            Y_mean = mean(Ym[:,1:M], dims=2)
            costs[M_ind][par_ind] =
                mean((Y[N_trans+1:end,e]-Y_mean[N_trans+1:end]).^2)
        end
    end

    @info "Computing cost function gradients"
    # ============== Computing Cost Gradients =======================
    Mjacs = [max(M÷2,1) for M in Ms]
    # max(Mmax, 2) because at least 2 independent realizations are needed to
    # estimate the gradient of the cost function
    for par_ind = 1:length(par_vector)
        Ym, jacsYm = simulate_system_sens(exp_data, par_vector[par_ind], max(Mmax, 2), isws)
        for (M_ind, Mjac) in enumerate(Mjacs)
            cost_gradients[M_ind][par_ind] = get_cost_gradient(
                Y[:,e], Ym[:,1:Mjac], jacsYm[Mjac+1:2Mjac], N_trans)
        end
    end

    # @info "Optimizing cost function using SGD"
    # # ================== Optimizing using SGD ======================
    # function get_gradient_estimate(y, pars, isws, M_mean::Int=1)
    #     Ym, jacsYm = simulate_system_sens(exp_data, pars, 2M_mean, isws)
    #     # Uses different noise realizations for estimate of output and estiamte of gradient
    #     grad_est = get_cost_gradient(y, Ym[:,1:M_mean], jacYm[_, M_mean+1:end])
    #     # grad_est will be 2D array with one dimenion equal to 1, we want to return a 1D array
    #     return grad_est[:]
    # end
    #
    # # get_gradient_estimate_p(pars, M_mean) = get_gradient_estimate(Y[:,1], pars, isws, M_mean)
    # # opt_pars_proposed, trace_proposed =
    # #     perform_stochastic_gradient_descent(get_gradient_estimate_p, pars0, par_bounds, verbose=true)

    return costs, cost_gradients
end

function debug_output_jacobian_dynamics(expid::String, pars::Array{Float64, 1})
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

    get_all_parameters(pars::Array{Float64, 1}) = vcat(get_all_θs(pars), exp_data.get_all_ηs(pars))

    step = 0.01
    # Each rows represents the displacement of one variable used for estimating jacobian
    dvars = step*Matrix{Float64}(I, length(pars), length(pars))

    # NOTE: All simulations here are done without any disturbance

    Y, sens = solvew_sens(exp_data.u, t -> 0.0, pars, N) |> h_comp
    Jac_est = zeros(length(Y), length(pars))

    for ind in 1:length(pars)
        reset_isws!(isws)
        Ynew = solvew(exp_data.u, t -> 0.0, pars+dvars[ind,:], N) |> h
        Jac_est[:,ind] = (Ynew - Y)/step
    end

    p = plot(sens[:,2], label = "Output Gradient wrt L")
    plot!(p, Jac_est[:,2], label="Estimate")
    display(p)

    return sens, Jac_est
end

function debug_output_jacobian_disturbance(expid::String, pars::Array{Float64,1})
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

    W_meta = exp_data.W_meta
    nx = W_meta.nx
    n_in = W_meta.n_in
    n_out = W_meta.n_out
    n_tot = nx*n_in
    dη = length(W_meta.η)
    N = size(exp_data.Y, 1)-1
    Zm = [randn(Nw, n_tot)]

    step = 0.01
    # NOTE: Currently only treats case where disturbance parameters correspond
    # to parameters in the "c-vector"
    dist_par_inds = nx+1:nx+n_tot*n_out

    p = vcat(get_all_θs(pars), exp_data.get_all_ηs(pars))
    θ = p[1:dθ]
    η = p[dθ+1: dθ+dη]
    C = reshape(η[nx+1:end], (n_out, n_tot))

    dmdl = discretize_ct_noise_model(get_ct_disturbance_model(η, nx, n_out), δ)
    XWm = simulate_noise_process_mangled(dmdl, Zm)

    sens_estimates  = zeros(size(Y,1), length(dist_par_inds))
    euler_estimates = zeros(size(Y,1), length(dist_par_inds))

    # TODO: results "res" change depending on when they are generated, i.e.
    # if we change the order of parameters in dist_par_inds, we will not get the
    # same realization for same parameter. I'm not sure why, might be very
    # reasonable explanation for it, but I don't know of the top of my head
    # and I don't really want to spend more time on it
    for (i, dist_par_ind) in enumerate(dist_par_inds)
        η2 = η
        η2[dist_par_ind] += step
        C2 = reshape(η2[nx+1:end], (n_out, n_tot))

        # wmm(m::Int) = mk_noise_interp(C, XWm, m, δ, nx, [dist_par_ind])
        # wmm2(m::Int) = mk_noise_interp(C2, XWm, m, δ, nx, [dist_par_ind])
        wmm(m::Int) = mk_noise_interp(C, XWm, m, δ)
        wmm2(m::Int) = mk_noise_interp(C2, XWm, m, δ)

        # We must re-simulate "true" system to get sensitivity wrt the correct paramter
        res = solvew_sens(exp_data.u, t -> wmm(1)(t), pars, N) |> h_comp
        # @info "N: $N, y: $(size(Y)), res: $(size(res[2])), type: $(typeof(res[2]))"
        res2 = solvew_sens(exp_data.u, t -> wmm2(1)(t), pars, N) |> h_comp
        sens_estimates[:, i] = res[2][:,3]
        euler_estimates[:, i] = (res2[1]-res[1])/step
    end

    return sens_estimates, euler_estimates

    # p = plot(euler_estimate)
    # plot!(p, sens_estimate)
    # display(p)
end

function debug_input_effect(expid::String, N_trans::Int = 0)
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
    dist_sens_inds = W_meta.free_par_inds

    Zm = [randn(Nw, n_tot) for m = 1:M] # TODO: I'm not sure we need M, 1 enough?


    # NOTE: Assumes disturbance model isn't parametrized, so just uses true pars
    p = vcat(get_all_θs(pars_true), exp_data.get_all_ηs(pars_true))
    θ = p[1:dθ]
    η = p[dθ+1: dθ+dη]

    dmdl = discretize_ct_noise_model_with_sensitivities(get_ct_disturbance_model(η, nx, n_out), δ, dist_sens_inds)
    # # NOTE: OPTION 1: Use the rows below here for linear interpolation
    XWm = simulate_noise_process_mangled(dmdl, Zm)
    wmm(m::Int) = mk_noise_interp(dmdl.Cd, XWm, m, δ)
    # NOTE: OPTION 2: Use the rows below here for exact interpolation
    # reset_isws!(isws)
    # XWm = simulate_noise_process(dmdl, Zm)
    # wmm(m::Int) = mk_newer_noise_interp(view(η, 1:nx), C, XWm, m, n_in, δ, isws)

    # ks = 4.0:0.25:20.0
    ms = 0.1:0.1:3.0
    u_scales = 0.2:0.2:2.0
    ps = ms

    # NOTE: Assumes scalar output!!!
    Ys = [fill(NaN, (length(ps), N+1)) for u_scale in u_scales]
    # NOTE: Assumes that k is the only free parameter!
    for (i,u_scale) in enumerate(u_scales)
        for (j,p) in enumerate(ps)
            Ys[i][j,:] = solvew(t -> u_scale.*exp_data.u(t), t -> wmm(1)(t), [p], N) |> h
        end
    end

    return Ys
end

# Similar to perform_stochastic_gradient_descent() but also returns
# the estimated costs at all the tested parameters as well as a log of
# the norms for the gradients used at each step
function perform_stochastic_gradient_descent_debug(
    get_grad_estimate_debug::Function,
    pars0::Array{Float64,1},                       # Initial guess for parameters
    bounds::Array{Float64, 2},                      # Parameter bounds
    learning_rate::Function=learning_rate;
    tol::Float64=1e-4,
    maxiters=200,
    verbose=false)

    q = lr_break+10
    # q = lr_break+20 #DEBUG
    # q = 10
    last_q_norms = fill(Inf, q)
    running_criterion() = mean(last_q_norms) > tol

    t = 1
    pars = pars0
    cost_vals = Float64[]
    trace = [pars]
    grad_norms = Float64[]
    while t <= maxiters
        grad_est, cost_val = get_grad_estimate_debug(pars, M_rate(t))
        push!(cost_vals, cost_val)
        push!(grad_norms, norm(grad_est))
        step = -learning_rate(t, norm(grad_est))*grad_est
        # println("Iteration $t, gradient norm $(norm(grad_est)) and step sign $(sign(-first(grad_est))) with parameter estimate $pars")
        # running_criterion(grad_est) || break
        if verbose
            noninf_average = mean(last_q_norms[findall(last_q_norms .!= Inf)])
            println("Iteration $t, average gradient norm $(mean(last_q_norms)), gradient $(grad_est) and step $(step) with parameter estimate $pars")
        end
        running_criterion() || break
        # pars = pars - learning_rate(t)*grad_est
        pars = pars + step
        project_on_bounds!(pars, bounds)
        push!(trace, pars)
        # (t-1)%10+1 maps t to a number between 1 and 10 (inclusive)
        last_q_norms[(t-1)%q+1] = norm(grad_est)
        t += 1
    end
    return pars, trace, cost_vals, grad_norms
end

function perform_SGD_adam_debug(
    get_grad_estimate_debug::Function,
    pars0::Array{Float64,1},                        # Initial guess for parameters
    bounds::Array{Float64, 2},                      # Parameter bounds
    learning_rate::Function=learning_rate;
    tol::Float64=1e-4,
    maxiters=200,
    verbose=false,
    betas::Array{Float64,1} = [0.5, 0.999])   # betas are the decay parameters of moment estimates

    eps = 1e-13
    q = 20#lr_break+10
    last_q_norms = fill(Inf, q)
    running_criterion() = mean(last_q_norms) > tol
    s = zeros(size(pars0)) # First moment estimate
    r = zeros(size(pars0)) # Second moment estimate

    t = 1
    pars = pars0
    cost_vals = Float64[]
    trace = [pars]
    grad_norms = Float64[]
    while t <= maxiters
        grad_est, cost_val = get_grad_estimate_debug(pars, M_rate(t))
        push!(cost_vals, cost_val)
        push!(grad_norms, norm(grad_est))
        s = betas[1]*s + (1-betas[1])*grad_est
        r = betas[2]*r + (1-betas[2])*(grad_est.^2)
        shat = s/(1-betas[1]^t)
        rhat = r/(1-betas[2]^t)
        # step = -learning_rate_adam(t, norm(grad_est))*shat./(sqrt.(rhat).+eps)
        step = -learning_rate_vec(t, norm(grad_est)).*shat./(sqrt.(rhat).+eps)
        # println("Iteration $t, gradient norm $(norm(grad_est)) and step sign $(sign(-first(grad_est))) with parameter estimate $pars")
        # running_criterion(grad_est) || break
        if verbose
            println("Iteration $t, average gradient norm $(mean(last_q_norms)), gradient $(grad_est) and step $(step) with parameter estimate $pars")
        end
        running_criterion() || break
        pars = pars + step
        project_on_bounds!(pars, bounds)
        push!(trace, pars)
        # (t-1)%10+1 maps t to a number between 1 and 10 (inclusive)
        last_q_norms[(t-1)%q+1] = norm(grad_est)
        t += 1
    end
    return pars, trace, cost_vals, grad_norms
end

function test_SGD_on_known_cost()
    # Cost given by 0.5*(x^T)Hx+cx
    k = 1e-4
    a = 10000.0
    H = k.*[1 0; 0 a]
    c = k.*[1;1]
    σ = 1e-7
    # function get_gradient_estimate(pars::Array{Float64,1}, M::Int)
    #     x =
    # end

    get_gradient_estimate(x, M) = H*x+c+σ*randn(size(x))
    x0 = H\c+[20;20]
    bounds = [-Inf Inf; -Inf Inf]
    pars_opt, trace = perform_SGD_adam(get_gradient_estimate, x0, bounds, learning_rate_adam, verbose=true, maxiters=400, tol=1e-6)
    true_opt = -H\c
    true_opt_cost = 0.5*((true_opt')*H*true_opt) + (c')*true_opt
    opt_cost = 0.5*((pars_opt')*H*pars_opt) + (c')*pars_opt
    return pars_opt, true_opt, trace, opt_cost, true_opt_cost
end

function visualize_SGD_search(par_vals::Array{Float64,1}, cost_vals::Array{Float64, 1}, SGD_trace::Array{Array{Float64, 1},1}, opt_par::Float64; file_name = "SGD.gif")
    anim = @animate for i = 1:length(SGD_trace)
        p = plot(par_vals, cost_vals, label="Cost Function")
        vline!(p, [opt_par], label="Final parameter")
        vline!(p, SGD_trace[i], label="Current parameter")
    end
    gif(anim, file_name, fps = 15)
    # # NOTE: Not sure if maybe this is faster?
    # @gif for i = 1:length(SGD_trace)
    #     p = plot(par_vals, cost_vals, label="Cost Function")
    #     vline!(p, [opt_par], label="Final parameter")
    #     vline!(p, [SGD_trace[i]], label="Current parameter")
    # end every 1
end

# SGD_traces should be an array with E elements, where each element is an array
# of parameters that the optimization has gone through, and where every
# parameter is an array of Float64-values
function Roberts_gif_generator(par_vals::Array{Float64,1}, cost_vals::Array{Float64, 2}, SGD_traces::Array{Array{Array{Float64,1},1},1}, opt_pars::Array{Float64, 1})
    for i = 1:1
        visualize_SGD_search(par_vals, cost_vals[:,i], SGD_traces[i], opt_pars[i], file_name="newer_stopping_criterion_Mrate4_new$(i).gif")
    end
end
