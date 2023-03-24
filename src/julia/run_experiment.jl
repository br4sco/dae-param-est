using LsqFit, LaTeXStrings, Dates, Interpolations
using StatsPlots # Commented out since it breaks 3d-plotting in Julia 1.5.3
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

# DEBUG Only used for quicker testing of Adjoint Method
struct AdjointData
    wmm::Function
    y_func::Function
    dy_func::Function
    u::Function
    sols::Array{DAESolution,1}
    N::Int
    Ts::Float64 # TODO: No need to pass this I think, Ts already global?
    p::Array{Float64,1}
    xp0::Matrix{Float64}
end

const PENDULUM = 1
const MOH_MDL  = 2

# Selects which model to adapt code to
model_id = PENDULUM

# ==================
# === PARAMETERS ===
# ==================

# === TIME ===
# The relationship between number of data samples N and number of noise samples
# Nw is given by Nw >= (Ts/δ)*N
const δ = 0.01                  # noise sampling time
const Ts = 0.1                  # step-size
const M = 1#00       # Number of monte-carlo simulations used for estimating mean
# TODO: Surely we don't need to collect these, a range should work just as well?
const ms = collect(1:M)
const W = 100           # Number of intervals for which isw stores data
const Q = 1000          # Number of conditional samples stored per interval

M_rate_max = 4#100#8#4#16
# max_allowed_step = 1.0  # Maximum magnitude of step that SGD is allowed to take
# M_rate(t) specifies over how many realizations the output jacobian estimate
# should be computed at iteration t. NOTE: A total of 2*M_rate(t) iterations
# will be performed for estimating the gradient of the cost functions
# @warn "USING TIME VARYING MRATE NOW"
# M_rate(t::Int) = (t÷50+1)*M_rate_max
M_rate(t::Int) = M_rate_max

mangle_XW(XW::Array{Array{Float64, 1}, 2}) =
  hcat([vcat(XW[:, m]...) for m = 1:size(XW,2)]...)

demangle_XW(XW::Array{Float64, 2}, n_tot::Int) =
    [XW[(i-1)*n_tot+1:i*n_tot, m] for i=1:(size(XW,1)÷n_tot), m=1:size(XW,2)]

# NOTE: SCALAR_OUTPUT is assumed
# NOTE: The realizaitons Ym and jacYm must be independent for this to return
# an unbiased estimate of the cost function gradient
function get_cost_gradient(Y::Vector{Float64}, Ym::Matrix{Float64}, jacsYm::Vector{Matrix{Float64}}, N_trans::Int=0)
    # N = size(Y,1)-1, since Y also contains the zeroth sample.
    # While we sum over t0, t1, ..., tN, the error at t0 will always be zero
    # due to known initial conditions which is why we divide by N instead of N+1

    # Y[N_trans+1:end].-Ym[N_trans+1:end,:] is a matrix with as many columns as
    # Ym, where column i contains Y[N_trans+1:end]-Ym[N_trans+1:end,i]
    # Taking the mean of that gives us the average error as a function of time
    # over all realizations contained in Ym

    # mean(-jacsYm)[N_trans+1:end,:] is the average (over all m) jacobian of
    # Ym[N_trans+1:end].

    # Previously used (theoretically equivalent)
    # (2/(size(Y,1)-N_trans-1))*
    #     sum(
    #     mean(Y[N_trans+1:end].-Ym[N_trans+1:end,:], dims=2)
    #     .*mean(-jacsYm)[N_trans+1:end,:]
    #     , dims=1)

    (2/(size(Y,1)-N_trans-1))*(
        transpose(mean(-jacsYm)[N_trans+1:end,:])*
        mean(Y[N_trans+1:end].-Ym[N_trans+1:end,:], dims=2)
        )[:]
end

# NOTE: SCALAR_OUTPUT is assumed
function get_cost_value(Y::Array{Float64,1}, Ym::Array{Float64,2}, N_trans::Int=0)
    (1/(size(Y,1)-N_trans-1))*sum( ( Y[N_trans+1:end] - mean(Ym[N_trans+1:end,:], dims=2) ).^2 )
end

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
    # @warn "Using x_tol = 1e-12 instead of default 1e-8"
    # fit_result, trace = curve_fit(model, jacobian_model, Float64[], Ye, pars, lower=lb, upper=ub, show_trace=true, inplace=false, x_tol=1e-8)    # Default inplace = false, x_tol = 1e-8
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
# const free_dyn_pars_true = [k]                    # true value of all free parameters
# get_all_θs(pars::Array{Float64,1}) = [m, L, g, pars[1]]
# dyn_par_bounds = [0.1 Inf]    # Lower and upper bounds of each free dynamic parameter
# NOTE: Has to be changed when number of free dynamical parameters is changed.
# Specifically:
# 1. free_dyn_pars_true must contain the true values for all free dynamical parameters
# 2. get_all_θs() must return all variables, where the free variables
#       need to be replaced by the provided argument pars
# 3. dyn_par_bounds must include bounds for all free dynamical paramters
# 4. The first argument to problem() in the definition of realize_model_sens
#       must refere to DAE-problem that includes sensitvity equations for all
#       free dynamical parameters
# sensitivity for all free dynamical variables
# const free_dyn_pars_true = [m, L, g, k]                    # true value of all free parameters
# const free_dyn_pars_true = [m, L, g, k]                    # true value of all free parameters

if model_id == PENDULUM
    const free_dyn_pars_true = [m, L, k]#[m, L, g, k] # True values of free parameters #Array{Float64}(undef, 0)
    const num_dyn_vars = 7
    get_all_θs(pars::Array{Float64,1}) = [pars[1], pars[2], g, pars[3]]#[pars[1], pars[2], g, pars[3]]#[m, L, g, pars[1]]#[pars[1], L, pars[2], k]
    # Each row corresponds to lower and upper bounds of a free dynamic parameter.
    dyn_par_bounds = [0.01 1e4; 0.1 1e4; 0.1 1e4]#[0.01 1e4; 0.1 1e4; 0.1 1e4]#; 0.1 1e4] #Array{Float64}(undef, 0, 2)
    @warn "The learning rate dimensiond doesn't deal with disturbance parameters in any nice way, other info comes from W_meta, and this part is hard coded"
    const_learning_rate = [0.1, 1.0, 1.0, 0.01, 0.1, 0.01]
    model_sens_to_use = pendulum_sensitivity_sans_g_with_dist_sens_3#pendulum_sensitivity_k_with_dist_sens_1#pendulum_sensitivity_sans_g#_full
    model_to_use = pendulum_new
    model_adj_to_use = my_pendulum_adjoint_Lonly
    model_stepbystep = pendulum_adj_stepbystep_L
elseif model_id == MOH_MDL
    # For Mohamed's model:
    const free_dyn_pars_true = [0.8]
    get_all_θs(pars::Array{Float64,1}) = free_dyn_pars_true
    # Each row corresponds to lower and upper bounds of a free dynamic parameter.
    dyn_par_bounds = [0.01 1e4]
    @warn "The learning rate dimensiond doesn't deal with disturbance parameters in any nice way, other info comes from W_meta, and this part is hard coded"
    const_learning_rate = [0.1]
    model_sens_to_use = mohamed_sens
    model_to_use = model_mohamed
    model_adj_to_use = mohamed_adjoint_new
end

learning_rate_vec(t::Int, grad_norm::Float64) = const_learning_rate#if (t < 100) const_learning_rate else ([0.1/(t-99.0), 1.0/(t-99.0)]) end#, 1.0, 1.0]  #NOTE Dimensions must be equal to number of free parameters
learning_rate_vec_red(t::Int, grad_norm::Float64) = const_learning_rate./sqrt(t)

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
if model_id == PENDULUM
    f(x::Array{Float64,1}) = x[7]               # applied on the state at each step
    # f_sens(x::Array{Float64,1}) = [x[14], x[21], x[28]]#, x[35], x[42], x[49]]#, x[28]]##[x[14], x[21], x[28], x[35], x[42]]   # NOTE: Hard-coded right now
    f_sens(x::Array{Float64,1}) = [x[14]]
    f_int(x::Vector{Float64})  = [x[15], x[16], x[17], x[18], x[1], x[2], x[8], x[9], x[10], x[11], x[12], x[13], x[14], x[19], x[20], x[21], x[22]]
    f_sens_deb(x::Vector{Float64}) = x[8:14]
    f_deb(x::Vector{Float64}) = vcat(f_sens(x), f_int(x))
elseif model_id == MOH_MDL
    f(x::Array{Float64,1}) = x[2]
    f_sens(x::Array{Float64,1}) = [x[4]]
end
f_debug(x::Array{Float64,1}) = x
# f_sens(x::Array{Float64,1}) = [x[14], x[21], x[28]]
# f_sens(x::Array{Float64,1}) = [x[14], x[21]]
# NOTE: Has to be changed when number of free  parameters is changed.
# Specifically, f_sens() must return sensitivity wrt to all free parameters
h(sol) = apply_outputfun(f, sol)                            # for our model
h_comp(sol) = apply_two_outputfun_mvar(f, f_sens, sol)           # for complete model with dynamics sensitivity
h_sens(sol) = apply_outputfun_mvar(f_sens, sol)              # for only returning sensitivity
h_debug(sol) = apply_outputfun(f_debug, sol)
h_int(sol)  = apply_two_outputfun_mvar(f, f_deb, sol)            # for also computing integral cost
h_sens_deb(sol) = apply_two_outputfun_twomvar(f_debug, f_sens_deb, sol)

const num_dyn_pars = length(free_dyn_pars_true)#size(dyn_par_bounds, 1)
realize_model_sens(u::Function, w::Function, pars::Array{Float64, 1}, N::Int) = problem(
    model_sens_to_use(φ0, u, w, get_all_θs(pars)),
    N,
    Ts,
)
realize_model(u::Function, w::Function, free_dyn_pars::Array{Float64, 1}, N::Int) = problem(
    model_to_use(φ0, u, w, get_all_θs(free_dyn_pars)),
    N,
    Ts,
)
const dθ = length(get_all_θs(free_dyn_pars_true))

# === SOLVER PARAMETERS ===
const abstol = 1e-8#1e-9
const reltol = 1e-5#1e-6
const maxiters = Int64(1e8)

solvew_sens(u::Function, w::Function, free_dyn_pars::Array{Float64, 1}, N::Int; kwargs...) = solve(
  realize_model_sens(u, w, free_dyn_pars, N),
  saveat = 0:Ts:(N*Ts),
  abstol = abstol,
  reltol = reltol,
  maxiters = maxiters;
  kwargs...,
)
solvew(u::Function, w::Function, free_dyn_pars::Array{Float64, 1}, N::Int; kwargs...) = solve(
  realize_model(u, w, free_dyn_pars, N),
  saveat = 0:Ts:(N*Ts),
  abstol = abstol,
  reltol = reltol,
  maxiters = maxiters;
  kwargs...,
)

# data-set output function
h_data(sol) = apply_outputfun(x -> f(x) + σ * randn(), sol)

function get_estimates(expid::String, pars0::Array{Float64,1}, N_trans::Int = 0)
    start_datetime = now()
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
    @assert (size(dyn_par_bounds, 1) == num_dyn_pars) "Please provide bounds for exactly all free dynamic parameters"
    @assert (length(const_learning_rate) == length(pars0)) "The learning rate must have the same number of components as the number of parameters to be identified"

    if !isdir(joinpath(data_dir, "tmp/"))
        mkdir(joinpath(data_dir, "tmp/"))
    end

    get_all_parameters(free_pars::Array{Float64, 1}) = vcat(get_all_θs(free_pars), exp_data.get_all_ηs(free_pars))
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

    function jacobian_model_b(dummy_input, free_pars)
        jac = solvew_sens(u, t -> zeros(n_out+length(dist_par_inds)*n_out), free_pars, N) |> h_sens
        return jac[N_trans+1:end, :]
    end

    # jacobian_model_b(dummy_input, pars) =
    #     solvew_sens(u, t -> zeros(n_out), pars, N) |> h_sens

    # E = size(Y, 2)
    # DEBUG
    E = 10
    @warn "Using E = $E instead of default"
    opt_pars_baseline = zeros(length(pars0), E)
    # trace_base[e][t][j] contains the value of parameter j before iteration t
    # corresponding to dataset e
    trace_base = [[pars0] for e=1:E]
    setup_duration = now() - start_datetime
    baseline_durations = Array{Millisecond, 1}(undef, E)
    @warn "Not running baseline identification now"
    for e=[]#1:E
        time_start = now()
        # HACK: Uses trace returned due to hacked LsqFit package
        baseline_result, baseline_trace = get_fit_sens(Y[N_trans+1:end,e], pars0,
            baseline_model_parametrized, jacobian_model_b,
            par_bounds[:,1], par_bounds[:,2])
        opt_pars_baseline[:, e] = coef(baseline_result)

        println("Completed for dataset $e for parameters $(opt_pars_baseline[:,e])")
        writedlm(joinpath(data_dir, "tmp/backup_baseline_e$e.csv"), opt_pars_baseline[:,e], ',')

        # Sometimes (the first returned value I think) the baseline_trace
        # has no elements, and therefore doesn't contain the metadata x
        if length(baseline_trace) > 1
            for j=2:length(baseline_trace)
                push!(trace_base[e], baseline_trace[j].metadata["x"])
            end
        end
        baseline_durations[e] = now()-time_start
    end

    @info "The mean optimal parameters for baseline are given by: $(mean(opt_pars_baseline, dims=2))"

    # === Finally we optimize parameters for the proposed model ==

    # Returns estimate of gradient of cost function
    # M_mean specifies over how many realizations the gradient estimate is computed
    function get_gradient_estimate(y, free_pars, isws, M_mean::Int=1)
        Ym, jacsYm = simulate_system_sens(exp_data, free_pars, 2M_mean, dist_par_inds, isws)

        # Uses different noise realizations for estimate of output and estiamte of jacobian
        return get_cost_gradient(y, Ym[:,1:M_mean], jacsYm[M_mean+1:end], N_trans)
    end

    get_gradient_estimate_p(free_pars, M_mean) = get_gradient_estimate(Y[:,1], free_pars, isws, M_mean)
    get_adjoint_estimate_p(free_pars, M_mean) = get_adjoint_sensitivity(exp_data, free_pars, M_mean, dist_par_inds, isws)

    opt_pars_proposed = zeros(length(pars0), E)
    avg_pars_proposed = zeros(length(pars0), E)
    trace_proposed = [ [Float64[]] for e=1:E]
    trace_gradient = [ [Float64[]] for e=1:E]
    trace_step     = [ [Float64[]] for e=1:E]        ## DEBUG!!!!!
    trace_lrate     = [ [Float64[]] for e=1:E]        ## DEBUG!!!!!
    proposed_durations = Array{Millisecond, 1}(undef, E)
    # @warn "Not running proposed identification now"
    # for e=1:E
    @warn "Only running proposed identificaiton for e=3"
    for e=3:10
        time_start = now()
        # jacobian_model(x, p) = get_proposed_jacobian(pars, isws, M)  # NOTE: This won't give a jacobian estimate independent of Ym, but maybe we don't need that since this isn't SGD?
        @warn "Only using maxiters=100 right now"
        opt_pars_proposed[:,e], trace_proposed[e], trace_gradient[e] =
            perform_SGD_adam_new(get_gradient_estimate_p, pars0, par_bounds, verbose=true, tol=1e-8, maxiters=100)
        # # DEBUG
        # opt_pars_proposed[:,e], trace_proposed[e], trace_gradient[e], trace_step[e], trace_lrate[e] =
        #     perform_SGD_adam_debug(get_gradient_estimate_p, pars0, par_bounds, verbose=true, tol=1e-8, maxiters=300)#0)

        # avg_pars_proposed[:,e] = mean(trace_proposed[e][end-80:end])

        writedlm(joinpath(data_dir, "tmp/backup_proposed_e$e.csv"), opt_pars_proposed[:,e], ',')
        writedlm(joinpath(data_dir, "tmp/backup_average_e$e.csv"), avg_pars_proposed[:,e], ',')

        # proposed_result, proposed_trace = get_fit(Y[:,e], pars0,
        #     (dummy_input, pars) -> proposed_model_parametrized(δ, Zm, dummy_input, pars, isws),
        #     par_bounds[:,1], par_bounds[:,2])
        # proposed_result, proposed_trace = get_fit_sens(Y[:,e], pars0,
        #     (dummy_input, pars) -> proposed_model_parametrized(δ, Zm, dummy_input, pars, isws), jacobian_model)
        # opt_pars_proposed[:, e] = coef(proposed_result)
        println("Completed for dataset $e for parameters $(opt_pars_proposed[:,e])")
        proposed_durations[e] = now()-time_start
    end

    @info "The mean optimal parameters for proposed method are given by: $(mean(opt_pars_proposed, dims=2))"

    # Call Dates.value[setup_duration] or e.g. Dates.value.(baseline_durations) to convert Millisecond to Int
    durations = (setup_duration, baseline_durations, proposed_durations)
    # return opt_pars_baseline, opt_pars_proposed, avg_pars_proposed, trace_base, trace_proposed, trace_gradient, trace_step, trace_lrate, durations
    return opt_pars_baseline, opt_pars_proposed, avg_pars_proposed, trace_base, trace_proposed, trace_gradient, trace_step, durations
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

    get_all_parameters(free_pars::Array{Float64, 1}) = vcat(get_all_θs(free_pars), exp_data.get_all_ηs(free_pars))
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
    dmdl = discretize_ct_noise_model_with_sensitivities(get_ct_disturbance_model(η, nx, n_out), δ, dist_par_inds)
    # NOTE: OPTION 1: Use the rows below here for linear interpolation
    XWm = simulate_noise_process_mangled(dmdl, Z_sens)
    wmm(m::Int) = mk_noise_interp(dmdl.Cd, XWm, m, δ)
    # # NOTE: OPTION 2: Use the rows below here for exact interpolation
    # reset_isws!(isws)
    # XWm = simulate_noise_process(dmdl, Zm)
    # wmm(m::Int) = mk_newer_noise_interp(view(η, 1:nx), C, XWm, m, n_in, δ, isws)

    calc_mean_y_N_prop(N::Int, free_pars::Array{Float64, 1}, m::Int) =
        solvew_sens(u, t -> wmm(m)(t), free_pars, N) |> h_comp
    calc_mean_y_prop(free_pars::Array{Float64, 1}, m::Int) = calc_mean_y_N_prop(N, free_pars, m)
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
    # Structure: η = vcat(ηa, ηc), where ηa is nx large, and ηc is n_tot*n_out large
    # free_dist_pars = fill(false, size(η_true))                                         # Known disturbance model
    free_dist_pars = vcat(fill(true, nx), false, fill(true, n_tot*n_out-1))            # Whole a-vector and all but first element of c-vector unknown (MAXIMUM UNKNOWN PARAMETERS) # TODO: Why not one more C-parameter?
    # free_dist_pars = vcat(fill(false, nx), false, fill(true, n_tot*n_out-1))           # All but first element (last elements?) of c-vector unknown
    # free_dist_pars = vcat(true, fill(false, nx-1), false, fill(true, n_tot*n_out-1))   # First element of a-vector and all but first (usually just one) element of c-vector unknown
    # free_dist_pars = vcat(fill(true, nx), false, fill(false, n_tot*n_out-1))           # Whole a-vector unknown
    # free_dist_pars = vcat(fill(false, nx), true, fill(false, n_tot*n_out-1))           # First parameter of c-vector unknown
    # free_dist_pars = vcat(true, fill(false, nx-1), fill(false, n_tot*n_out))           # First parameter of a-vector unknown
    # free_dist_pars = vcat(true, fill(false, nx-1), true, fill(false, n_tot*n_out-1))   # First parameter of a-vector and first parameter of c-vector unknown
    free_par_inds = findall(free_dist_pars)          # Indices of free variables in η. Assumed to be sorted in ascending order.
    # Array of tuples containing lower and upper bound for each free disturbance parameter
    # dist_par_bounds = Array{Float64}(undef, 0, 2)
    dist_par_bounds = [-Inf Inf; -Inf Inf; -Inf Inf]#[0 Inf; 0 Inf; -Inf Inf]
    function get_all_ηs(free_pars::Array{Float64, 1})
        # If copy() is not used here, some funky stuff that I don't fully understand happens.
        # I think essentially η_true stops being defined after function returns, so
        # setting all_η to its value doesn't behave quite as I expected
        all_η = copy(η_true)
        # Fetches user-provided values for free disturbance parameters only
        all_η[free_par_inds] = free_pars[num_dyn_pars+1:end]
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

    # Exact interpolation
    mk_we(XW::Array{Array{Float64, 1},2}, isws::Array{InterSampleWindow, 1}) =
        (m::Int) -> mk_newer_noise_interp(
        a_vec::AbstractArray{Float64, 1}, C_true, XW, m, n_in, δ, isws)
    # Linear interpolation. Not recommended DEBUG
    # mk_we(XW::Array{Array{Float64,1},2}, isws::Array{InterSampleWindow, 1}) =
    #     (m::Int) -> mk_noise_interp(C_true, mangle_XW(XW), m, δ)

    u(t::Float64) = interpw(input, 1, 1)(t)

    # === We first generate the output of the true system ===
    function calc_Y(XW::Array{Array{Float64, 1},2}, isws::Array{InterSampleWindow, 1})
        # NOTE: This XW should be non-mangled, which is why we don't divide by n_tot
        @assert (Nw <= size(XW, 1)) "Disturbance data size mismatch ($(Nw) > $(size(XW, 1)))"
        # @warn "Using E=1 instead of size of XW when generating Y!"
        E = size(XW, 2)
        # E = 1
        es = collect(1:E)
        we = mk_we(XW, isws)
        # solve_in_parallel(e -> solvew(u, we(e), free_dyn_pars_true, N) |> h_data, es)
        Y = solve_in_parallel(e -> solvew(u, we(e), free_dyn_pars_true, N) |> h_data, es)
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
    # my_y = solvew(u, wdebug, free_dyn_pars_true, N) |> h
    # writedlm("data/experiments/pendulum_sensitivity/my_y.csv", my_y, ',')

    reset_isws!(isws)
    return ExperimentData(Y, u, get_all_ηs, dist_par_bounds[1:length(free_par_inds),:], W_meta, Nw), isws
end

function perform_SGD_adam(
    get_grad_estimate::Function,
    pars0::Array{Float64,1},                        # Initial guess for parameters
    bounds::Array{Float64, 2},                      # Parameter bounds
    learning_rate::Function=learning_rate_vec;
    tol::Float64=1e-6,
    maxiters=200,
    verbose=false,
    betas::Array{Float64,1} = [0.9, 0.999])   # betas are the decay parameters of moment estimates
    # betas::Array{Float64,1} = [0.5, 0.999])   # betas are the decay parameters of moment estimates

    eps = 0.#1e-8
    q = 20  # TODO: This is a little arbitrary, but because of low tolerance, the stopping criterion is never reached anyway
    last_q_norms = fill(Inf, q)
    running_criterion() = mean(last_q_norms) > tol
    s = zeros(size(pars0)) # First moment estimate
    r = zeros(size(pars0)) # Second moment estimate

    t = 1
    pars = pars0
    trace = [pars]
    grad_trace = []
    while t <= maxiters
        grad_est = get_grad_estimate(pars, M_rate(t))

        s = betas[1]*s + (1-betas[1])*grad_est
        r = betas[2]*r + (1-betas[2])*(grad_est.^2)
        shat = s/(1-betas[1]^t)
        rhat = r/(1-betas[2]^t)
        step = -learning_rate(t, norm(grad_est)).*shat./(sqrt.(rhat).+eps)
        # println("Iteration $t, gradient norm $(norm(grad_est)) and step sign $(sign(-first(grad_est))) with parameter estimate $pars")
        # running_criterion(grad_est) || break
        if verbose
            println("Iteration $t, average gradient norm $(mean(last_q_norms)), -gradient $(-grad_est) and step $(step) with parameter estimate $pars")
        end
        running_criterion() || break
        pars = pars + step
        project_on_bounds!(pars, bounds)
        push!(trace, pars)
        push!(grad_trace, grad_est)
        # (t-1)%10+1 maps t to a number between 1 and 10 (inclusive)
        last_q_norms[(t-1)%q+1] = norm(grad_est)
        t += 1
    end
    return pars, trace, grad_trace
end

# Also adds time-varying beta_1, dependent on the new hyper-parameter λ
function perform_SGD_adam_new(
    get_grad_estimate::Function,
    pars0::Array{Float64,1},                        # Initial guess for parameters
    bounds::Array{Float64, 2},                      # Parameter bounds
    learning_rate::Function=learning_rate_vec_red;
    tol::Float64=1e-6,
    maxiters=200,
    verbose=false,
    betas::Array{Float64,1} = [0.9, 0.999],
    λ = 0.5)   # betas are the decay parameters of moment estimates
    # betas::Array{Float64,1} = [0.5, 0.999])   # betas are the decay parameters of moment estimates

    eps = 0.#1e-8
    q = 20# TODO: This is a little arbitrary, but because of low tolerance, the stopping criterion is never reached anyway
    last_q_norms = fill(Inf, q)
    running_criterion() = mean(last_q_norms) > tol
    s = zeros(size(pars0)) # First moment estimate
    r = zeros(size(pars0)) # Second moment estimate

    t = 1
    pars = pars0
    trace = [pars]
    grad_trace = []
    while t <= maxiters
        grad_est = get_grad_estimate(pars, M_rate(t))

        beta1t = betas[1]*(λ^(t-1))
        s = beta1t*s + (1-beta1t)*grad_est
        r = betas[2]*r + (1-betas[2])*(grad_est.^2)
        shat = s/(1-betas[1]^t) # Seems like betas[1] should be used instead of beta1t here
        rhat = r/(1-betas[2]^t)
        unscaled_step = -shat./(sqrt.(rhat).+eps)
        step = learning_rate(t, norm(grad_est)).*unscaled_step
        # step = -learning_rate_vec_red(t, norm(grad_est)).*shat./(sqrt.(rhat).+eps)
        # println("Iteration $t, gradient norm $(norm(grad_est)) and step sign $(sign(-first(grad_est))) with parameter estimate $pars")
        # running_criterion(grad_est) || break
        if verbose
            println("Iteration $t, average gradient norm $(mean(last_q_norms)), -gradient $(-grad_est) and step $(step) with parameter estimate $pars")
        end
        running_criterion() || break
        pars = pars + step
        project_on_bounds!(pars, bounds)
        push!(trace, pars)
        push!(grad_trace, grad_est)
        # (t-1)%10+1 maps t to a number between 1 and 10 (inclusive)
        last_q_norms[(t-1)%q+1] = norm(grad_est)
        t += 1
    end
    return pars, trace, grad_trace
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
    free_pars::Array{Float64,1},
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

    p = vcat(get_all_θs(free_pars), exp_data.get_all_ηs(free_pars))
    θ = p[1:dθ]
    η = p[dθ+1: dθ+dη]
    # C = reshape(η[nx+1:end], (n_out, n_tot))  # Not correct when disturbance model is parametrized, use dmdl.Cd instead

    # dmdl = discretize_ct_noise_model(get_ct_disturbance_model(η, nx, n_out), δ)
    dmdl = discretize_ct_noise_model_with_sensitivities(get_ct_disturbance_model(η, nx, n_out), δ, dist_sens_inds)
    # # NOTE: OPTION 1: Use the rows below here for linear interpolation
    XWm = simulate_noise_process_mangled(dmdl, Zm)
    wmm(m::Int) = mk_noise_interp(dmdl.Cd, XWm, m, δ)
    # NOTE: OPTION 2: Use the rows below here for exact interpolation
    # reset_isws!(isws)
    # XWm = simulate_noise_process(dmdl, Zm)
    # wmm(m::Int) = mk_newer_noise_interp(view(η, 1:nx), C, XWm, m, n_in, δ, isws)

    calc_mean_y_N(N::Int, free_pars::Array{Float64, 1}, m::Int) =
        solvew(exp_data.u, t -> wmm(m)(t), free_pars, N) |> h
    calc_mean_y(free_pars::Array{Float64, 1}, m::Int) = calc_mean_y_N(N, free_pars, m)
    return solve_in_parallel(m -> calc_mean_y(free_pars, m), collect(1:M))   # Returns Ym
end

# Simulates system with newly generated white noise
function simulate_system(
    exp_data::ExperimentData,
    free_pars::Array{Float64,1},
    M::Int,
    dist_sens_inds::Array{Int64, 1},
    isws::Array{InterSampleWindow,1})::Array{Float64,2}

    W_meta = exp_data.W_meta
    nx = W_meta.nx
    n_in = W_meta.n_in
    n_tot = nx*n_in
    Nw = exp_data.Nw
    Zm = [randn(Nw, n_tot) for m = 1:M]
    simulate_system(exp_data, free_pars, M, dist_sens_inds, isws, Zm)
end

# Computes adjoint sensitivity with specified white noise
function get_adjoint_sensitivity(
    exp_data::ExperimentData,
    free_pars::Array{Float64,1},
    M::Int,
    dist_sens_inds::Array{Int64, 1},
    isws::Array{InterSampleWindow,1},
    Zm::Array{Array{Float64,2},1})::Array{Float64,1}

    W_meta = exp_data.W_meta
    nx = W_meta.nx
    n_out = W_meta.n_out
    N = size(exp_data.Y, 1)-1
    u = exp_data.u
    η = exp_data.get_all_ηs(free_pars)
    Y = exp_data.Y
    p = vcat(vcat(get_all_θs(free_dyn_pars_true), W_meta.η))

    dmdl = discretize_ct_noise_model_with_sensitivities(get_ct_disturbance_model(η, nx, n_out), δ, dist_sens_inds)
    # # NOTE: OPTION 1: Use the rows below here for linear interpolation
    XWm = simulate_noise_process_mangled(dmdl, Zm)
    wmm(m::Int) = mk_noise_interp(dmdl.Cd, XWm, m, δ)
    # NOTE: OPTION 2: Use the rows below here for exact interpolation
    # reset_isws!(isws)
    # XWm = simulate_noise_process(dmdl, Zm)
    # wmm(m::Int) = mk_newer_noise_interp(view(η, 1:nx), dmdl.Cd, XWm, m, n_in, δ, isws)

    # exp_data, sol1, sol2, sol_true = get_M_solutions("5k_u2w6_from_Alsvin", pars, 2, u, w1, w2, wtrue)
    # NOTE: OKAY, BUILT-IN INTERPOLATION IS ZEROTH ORDER (AT LEAST FOR DERIVATIVE)
    # AND ABSOLUTELY SUCKS, BUT LET'S JUST STICK WITH IT FOR NOW, EASY TO USE AT LEAST
    # But, TODO: You should probably not use 0th order interpolation in simulation.jl
    y_func = interpolated_signal(Y[:,1], 0:Ts:(size(Y,1)-1)*Ts)

    forward_solve(m) = solvew(u, wmm(m), free_pars, N)
    sols = get_sol_in_parallel(forward_solve, 1:2)#M)

    # Computing xp0, initial conditions of derivative of x wrt to p
    mdl = model_to_use(φ0, u, wmm(1), p)
    mdl_sens = model_sens_to_use(φ0, u, wmm(1), p)
    n_mdl = length(mdl.x0)
    xp0 = reshape(mdl_sens.x0[n_mdl+1:end], n_mdl, :)
    mdl_adj, get_Gp = model_adj_to_use(u, wmm(1), p, N*Ts, sols[1], sols[2], y_func, xp0)
    adj_prob = problem(mdl_adj, N, Ts)

    adj_sol = solve(adj_prob, saveat = 0:Ts:N*Ts, abstol = abstol, reltol = reltol,
        maxiters = maxiters)

    return get_Gp(adj_sol)
end

# Computes adjoint sensitivity with newly generated white noise
function get_adjoint_sensitivity(
    exp_data::ExperimentData,
    free_pars::Array{Float64,1},
    M::Int,
    dist_sens_inds::Array{Int64, 1},
    isws::Array{InterSampleWindow,1})::Array{Float64,1}

    W_meta = exp_data.W_meta
    nx = W_meta.nx
    n_in = W_meta.n_in
    n_tot = nx*n_in
    Nw = exp_data.Nw
    Zm = [randn(Nw, n_tot) for m = 1:M]
    get_adjoint_sensitivity(exp_data, free_pars, M, dist_sens_inds, isws, Zm)
end

# Simulates system with specified white noise
function simulate_system_sens(
    exp_data::ExperimentData,
    free_pars::Array{Float64,1},
    M::Int,
    dist_sens_inds::Array{Int64, 1},    # TODO: Aren't these included in exp_data? Why pass them separately then? Just seems random
    isws::Array{InterSampleWindow,1},
    Zm::Array{Array{Float64,2},1})::Tuple{Array{Float64,2}, Array{Array{Float64,2},1}}

    W_meta = exp_data.W_meta
    nx = W_meta.nx
    # n_in = W_meta.n_in
    n_out = W_meta.n_out
    # n_tot = nx*n_in
    # dη = length(W_meta.η)
    N = size(exp_data.Y, 1)-1
    # dist_par_inds = W_meta.free_par_inds

    η = exp_data.get_all_ηs(free_pars)
    # p = vcat(get_all_θs(free_pars), exp_data.get_all_ηs(free_pars))
    # θ = p[1:dθ]
    # η = p[dθ+1: dθ+dη]
    # # C = reshape(η[nx+1:end], (n_out, n_tot))

    dmdl = discretize_ct_noise_model_with_sensitivities(get_ct_disturbance_model(η, nx, n_out), δ, dist_sens_inds)
    # # NOTE: OPTION 1: Use the rows below here for linear interpolation
    XWm = simulate_noise_process_mangled(dmdl, Zm)
    wmm(m::Int) = mk_noise_interp(dmdl.Cd, XWm, m, δ)
    # NOTE: OPTION 2: Use the rows below here for exact interpolation
    # reset_isws!(isws)
    # XWm = simulate_noise_process(dmdl, Zm)
    # wmm(m::Int) = mk_newer_noise_interp(view(η, 1:nx), dmdl.Cd, XWm, m, n_in, δ, isws)

    calc_mean_y_N(N::Int, free_pars::Array{Float64, 1}, m::Int) =
        solvew_sens(exp_data.u, t -> wmm(m)(t), free_pars, N) |> h_comp
    calc_mean_y(free_pars::Array{Float64, 1}, m::Int) = calc_mean_y_N(N, free_pars, m)
    return solve_in_parallel_sens(m -> calc_mean_y(free_pars, m), collect(1:M))  # Returns Ym and JacsYm
end

# Simulates system with newly generated white noise
function simulate_system_sens(
    exp_data::ExperimentData,
    free_pars::Array{Float64,1},
    M::Int,
    dist_sens_inds::Array{Int64, 1},
    isws::Array{InterSampleWindow,1})::Tuple{Array{Float64,2}, Array{Array{Float64,2},1}}

    W_meta = exp_data.W_meta
    nx = W_meta.nx
    n_in = W_meta.n_in
    n_tot = nx*n_in
    Nw = exp_data.Nw
    Zm = [randn(Nw, n_tot) for m = 1:M]
    simulate_system_sens(exp_data, free_pars, M, dist_sens_inds, isws, Zm)
end

function write_results_to_file(path::String, opt_pars_baseline, opt_pars_proposed, avg_pars_proposed, trace_base, trace_proposed, trace_gradient, durations)
    # To make a trace-file with E = 1 readable by readdlm, replace all
    # "]," with "\n" and all "[" by "", and lastly "]" by ""
    writedlm(path*"opt_pars_baseline.csv", opt_pars_baseline, ',')
    writedlm(path*"opt_pars_proposed.csv", opt_pars_proposed, ',')
    writedlm(path*"avg_pars_proposed.csv", avg_pars_proposed, ',')
    writedlm(path*"trace_base.csv", trace_base, ',')
    writedlm(path*"trace_proposed.csv", trace_proposed, ',')
    writedlm(path*"trace_gradient.csv", trace_gradient, ',')
    writedlm(path*"setup_duration.csv", Dates.value(durations[1]), ',')
    writedlm(path*"baseline_durations.csv", Dates.value.(durations[2]), ',')
    writedlm(path*"proposed_durations.csv", Dates.value.(durations[3]), ',')
end

# NOTE: Only generate baseline cost function
function generate_cost_func(expid::String, pars0::Array{Float64,1}, step_sizes::Array{Float64,1}, step_num::Int, N_trans::Int = 0)
    exp_data, isws = get_experiment_data(expid)
    W_meta = exp_data.W_meta
    Y = exp_data.Y
    # N = 200
    # @warn "Using N=200 instead of default!!!!!!!!!!"
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

    debug_store = fill(NaN, N+1, 2*step_num+1)

    # === Computing cost function of baseline model
    nθ = length(pars0)
    @assert (nθ > 0 && nθ <= 2) "Current code only supports number of free parameters up 2 (and more than 0)"
    base_par_vals = zeros(2*step_num+1, nθ)
    for (i, par) in enumerate(pars0)
        h = step_sizes[i]
        base_par_vals[:,i] = par-step_num*h:h:par+step_num*h+0.5*h
    end

    if nθ == 1
        base_cost_vals = zeros(2*step_num+1,1)
    elseif nθ == 2
        base_cost_vals = zeros(2*step_num+1,2*step_num+1)
    end

    if nθ == 1
        for (i,par) in enumerate(base_par_vals)
            Ys = solvew(exp_data.u, t -> zeros(n_out), [par], N) |> h
            base_cost_vals[i] = first(mean((Y[N_trans+1:N+1, 1].-Ys[N_trans+1:end]).^2, dims=1))
            debug_store[:,i] = Ys
            println("par value: $par, i: $i")
            # p = plot(Ys)
            # display(p)
        end
    elseif nθ == 2
        for (i1, par1) in enumerate(base_par_vals[:,1])
            for (i2, par2) in enumerate(base_par_vals[:,2])

                # DEBUG
                if mod(i1+i2,50) == 0
                    @info "Running iteration iwth i1=$i1, i2=$i2"
                end

                Ys = solvew(exp_data.u, t -> zeros(n_out), [par1, par2], N) |> h
                # NOTE: Different rows of base_cost_vals should correspond to
                # different values of parameter 2, while different columns should
                # correspond to different values of parameter 1. This makes it
                # so that if we make a surface plot by calling
                # plot(params1, params2, base_cost_vals, st=:surface)
                # then we will get the correct surface plot
                # One can view this as: x-param1, y-param2
                base_cost_vals[i2,i1] = first(mean((Y[N_trans+1:N+1, 1].-Ys[N_trans+1:end]).^2, dims=1))
            end
        end
    end

    # In base_cost_vals, columns (x-axis) correspond to different values of the second parameter
    return base_par_vals, base_cost_vals, debug_store
end

# NOTE: Only genererates proposed cost function
function sample_cost_func(expid::String, par_vec::Array{Array{Float64,1},1}, N_trans::Int = 0)
    exp_data, isws = get_experiment_data(expid)
    W_meta = exp_data.W_meta
    Y = exp_data.Y
    # N = 200
    # @warn "Using N=200 instead of default!!!!!!!!!!"
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

    Zm = [randn(Nw, n_tot) for m = 1:M]

    # === Computing cost function of baseline model
    base_cost_vals = zeros(length(par_vec))
    prop_cost_vals = zeros(length(par_vec))
    for (i,pars) in enumerate(par_vec)
        Ys = solvew(exp_data.u, t -> zeros(n_out), pars, N) |> h
        base_cost_vals[i] = first(mean((Y[N_trans+1:N+1, 1].-Ys[N_trans+1:end]).^2, dims=1))
        Yms = simulate_system(exp_data, pars, M, dist_sens_inds, isws, Zm)
        prop_cost_vals[i] = first(mean((Y[N_trans+1:N+1, 1].-mean(Yms, dims=2)[N_trans+1:end]).^2, dims=1))
    end

    # In base_cost_vals, columns (x-axis) correspond to different values of the second parameter
    return par_vec, base_cost_vals, prop_cost_vals
end

# NOTE: Only genererates proposed cost function gradient (using forward sensitivity). TODO: Also get baseline?
function sample_cost_func_grad(expid::String, par_vec::Array{Array{Float64,1},1}, N_trans::Int = 0)
    exp_data, isws = get_experiment_data(expid)
    W_meta = exp_data.W_meta
    Y = exp_data.Y
    # N = 200
    # @warn "Using N=200 instead of default!!!!!!!!!!"
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

    Zm = [randn(Nw, n_tot) for m = 1:2M]

    # === Computing cost function of baseline model
    # base_cost_vals = zeros(length(par_vec))
    prop_grad_vals = zeros(length(par_vec),length(par_vec[1]))
    for (i,pars) in enumerate(par_vec)
        # Ys = solvew(exp_data.u, t -> zeros(n_out), pars, N) |> h
        # base_cost_vals[i] = first(mean((Y[N_trans+1:N+1, 1].-Ys[N_trans+1:end]).^2, dims=1))
        Ym, jacsYm = simulate_system_sens(exp_data, pars, 2M, dist_sens_inds, isws, Zm)
        prop_grad_vals[i,:] = get_cost_gradient(Y[1:N+1, 1], Ym[:,1:M], jacsYm[M+1:end], N_trans)   # TODO: This one crashes, figure out what get_cost_gradient actually returns. Or maybe it crashes inside?
    end

    # In base_cost_vals, columns (x-axis) correspond to different values of the second parameter
    return par_vec, prop_grad_vals
end

# ======================= DEBUGGING FUNCTIONS ========================

function debug_dist_sens(expid::String, pars0::Array{Float64,1}, Δ::Float64=0.01)
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

    get_all_parameters(free_pars::Array{Float64, 1}) = vcat(get_all_θs(free_pars), exp_data.get_all_ηs(free_pars))
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
    dmdl = discretize_ct_noise_model_with_sensitivities_alt(get_ct_disturbance_model(η, nx, n_out), δ, dist_par_inds)
    # NOTE: OPTION 1: Use the rows below here for linear interpolation
    XWm = simulate_noise_process_mangled(dmdl, Z_sens)
    wmm(m::Int) = mk_noise_interp(dmdl.Cd, XWm, m, δ)

    calc_mean_y_N_prop(N::Int, free_pars::Array{Float64, 1}, m::Int) =
        solvew_sens(u, t -> wmm(m)(t), free_pars, N) |> h_comp
    calc_mean_y_prop(free_pars::Array{Float64, 1}, m::Int) = calc_mean_y_N_prop(N, free_pars, m)
    Ym_prop, sens_m_prop = solve_in_parallel_sens(m -> calc_mean_y_prop(pars0, m), ms)
    Y_mean_prop = reshape(mean(Ym_prop, dims = 2), :)

    val = wmm(1)(0.07)
    println("dist + sens at true val: $(val)")

    # RESULTS SEEM TO MATCH NUMERICAL APPROXIMATIONS VERY WELL!!!! :D

    # Ym_prop1 = Ym_prop
    # sens_m_prop1 = sens_m_prop
    # Y_mean_prop1 = Y_mean_prop
    # Ym_prop2 = Ym_prop
    # sens_m_prop2 = sens_m_prop
    # Y_mean_prop2 = Y_mean_prop
    # Ym_prop3 = Ym_prop
    # sens_m_prop3 = sens_m_prop
    # Y_mean_prop3 = Y_mean_prop

    # ################## a1 ################
    η = [0.8+Δ, 16.0, 0.0, 0.6]
    dmdl = discretize_ct_noise_model_with_sensitivities_alt(get_ct_disturbance_model(η, nx, n_out), δ, dist_par_inds)
    # NOTE: OPTION 1: Use the rows below here for linear interpolation
    XWm = simulate_noise_process_mangled(dmdl, Z_sens)
    wmm1(m::Int) = mk_noise_interp(dmdl.Cd, XWm, m, δ)
    println("Approx a1 sens: $((wmm1(1)(0.07)[1]-val[1])/Δ)")

    calc_mean_y_N_prop1(N::Int, free_pars::Array{Float64, 1}, m::Int) =
        solvew_sens(u, t -> wmm1(m)(t), free_pars, N) |> h_comp
    calc_mean_y_prop1(free_pars::Array{Float64, 1}, m::Int) = calc_mean_y_N_prop1(N, free_pars, m)
    Ym_prop1, sens_m_prop1 = solve_in_parallel_sens(m -> calc_mean_y_prop1(pars0, m), ms)
    Y_mean_prop1 = reshape(mean(Ym_prop1, dims = 2), :)

    # ################## a2 ################
    η = [0.80, 16+Δ, 0.0, 0.6]
    dmdl = discretize_ct_noise_model_with_sensitivities_alt(get_ct_disturbance_model(η, nx, n_out), δ, dist_par_inds)
    # NOTE: OPTION 1: Use the rows below here for linear interpolation
    XWm = simulate_noise_process_mangled(dmdl, Z_sens)
    wmm2(m::Int) = mk_noise_interp(dmdl.Cd, XWm, m, δ)
    println("Approx a2 sens: $((wmm2(1)(0.07)[1]-val[1])/Δ)")

    calc_mean_y_N_prop2(N::Int, free_pars::Array{Float64, 1}, m::Int) =
        solvew_sens(u, t -> wmm2(m)(t), free_pars, N) |> h_comp
    calc_mean_y_prop2(free_pars::Array{Float64, 1}, m::Int) = calc_mean_y_N_prop2(N, free_pars, m)
    Ym_prop2, sens_m_prop2 = solve_in_parallel_sens(m -> calc_mean_y_prop2(pars0, m), ms)
    Y_mean_prop2 = reshape(mean(Ym_prop1, dims = 2), :)

    # ################## c ################
    η = [0.80, 16.0, 0.0, 0.6+Δ]
    dmdl = discretize_ct_noise_model_with_sensitivities_alt(get_ct_disturbance_model(η, nx, n_out), δ, dist_par_inds)
    # NOTE: OPTION 1: Use the rows below here for linear interpolation
    XWm = simulate_noise_process_mangled(dmdl, Z_sens)
    wmm3(m::Int) = mk_noise_interp(dmdl.Cd, XWm, m, δ)
    println("Approx c sens: $((wmm3(1)(0.07)[1]-val[1])/Δ)")

    calc_mean_y_N_prop3(N::Int, free_pars::Array{Float64, 1}, m::Int) =
        solvew_sens(u, t -> wmm3(m)(t), free_pars, N) |> h_comp
    calc_mean_y_prop3(free_pars::Array{Float64, 1}, m::Int) = calc_mean_y_N_prop3(N, free_pars, m)
    Ym_prop3, sens_m_prop3 = solve_in_parallel_sens(m -> calc_mean_y_prop3(pars0, m), ms)
    Y_mean_prop3 = reshape(mean(Ym_prop1, dims = 2), :)

    # Comparing output sensitivities to numerical estimates
    numrate1 = (Ym_prop1-Ym_prop)./Δ;
    numrate2 = (Ym_prop2-Ym_prop)./Δ;
    numrate3 = (Ym_prop3-Ym_prop)./Δ;
    sens1    = zeros(length(sens_m_prop[1][:,1]), length(sens_m_prop))
    sens2    = zeros(length(sens_m_prop[1][:,1]), length(sens_m_prop))
    sens3    = zeros(length(sens_m_prop[1][:,1]), length(sens_m_prop))
    for i = eachindex(sens_m_prop)
        sens1[:,i] = sens_m_prop[i][:,1]
        sens2[:,i] = sens_m_prop[i][:,2]
        sens3[:,i] = sens_m_prop[i][:,3]
    end

    # Comparing cost function sensitivity to numerical estimate
    N = size(Y,2)-1
    costsens = zeros(3,N+1)
    costests1 = zeros(N+1)
    costests2 = zeros(N+1)
    costests3 = zeros(N+1)
    costests1NOTRANS = zeros(N+1)
    costests2NOTRANS = zeros(N+1)
    costests3NOTRANS = zeros(N+1)
    for e=eachindex(Y[1,:])
        costsens[:,e]  = get_cost_gradient(Y[:,e], Ym_prop[:,1:M÷2], sens_m_prop[M÷2+1:end], 500)
        costests1[e]   = (get_cost_value(Y[:,e], Ym_prop1, 500)-get_cost_value(Y[:,e], Ym_prop, 500))/Δ
        costests2[e]   = (get_cost_value(Y[:,e], Ym_prop2, 500)-get_cost_value(Y[:,e], Ym_prop, 500))/Δ
        costests3[e]   = (get_cost_value(Y[:,e], Ym_prop3, 500)-get_cost_value(Y[:,e], Ym_prop, 500))/Δ
    end
    # costests[i,e]/costsens[i,e] gives estimate/sensitivity wrt to parameter i for data-set e
    costests = vcat(costests1', costests2', costests3')
    costestsNOTRANS = vcat(costests1NOTRANS', costests2NOTRANS', costests3NOTRANS')

    # To compare nuermical estimates with sensitivity estimates for parameter i,
    # realizaiton e, call
    # p = plot(numratei[:,e]); plot(p, sensi[:,e])
    res_tup = (Ym_prop, sens_m_prop, Ym_prop1, sens_m_prop1, Ym_prop2, sens_m_prop2, Ym_prop3, sens_m_prop3)
    detcostres = debug_det_costs(Y, Ym_prop, Ym_prop1, Ym_prop2, Ym_prop3, sens1, sens2, sens3, Δ, 500)

    return numrate1, numrate2, numrate3, sens1, sens2, sens3, costests, costsens, res_tup, detcostres
end

function debug_det_costs(Y::Array{Float64,2}, Ym_prop::Array{Float64,2},
        Ym_prop1::Array{Float64,2}, Ym_prop2::Array{Float64,2},
        Ym_prop3::Array{Float64,2}, sens1::Array{Float64,2},
        sens2::Array{Float64,2}, sens3::Array{Float64,2}, Δ::Float64, N_trans::Int = 0)
    N = size(Y,1)-1

    E = size(Y,2)
    derivs1 = Matrix{Float64}(undef, M, E)
    derivs2 = Matrix{Float64}(undef, M, E)
    derivs3 = Matrix{Float64}(undef, M, E)
    ests1 = Matrix{Float64}(undef, M, E)
    ests2 = Matrix{Float64}(undef, M, E)
    ests3 = Matrix{Float64}(undef, M, E)
    derivs1E = Vector{Float64}(undef, E)
    derivs2E = Vector{Float64}(undef, E)
    derivs3E = Vector{Float64}(undef, E)
    ests1E   = Vector{Float64}(undef, E)
    ests2E   = Vector{Float64}(undef, E)
    ests3E   = Vector{Float64}(undef, E)
    Mlim     = M  # Mlim=1 makes E-version match non-E versions for m=1
    # For analysing bias/variance of sensitivity as a function of M
    num_blocks = 10
    maxM = M÷(2num_blocks)
    derivs1M = Matrix{Float64}(undef, num_blocks, E)
    derivs2M = Matrix{Float64}(undef, num_blocks, E)
    derivs3M = Matrix{Float64}(undef, num_blocks, E)
    for e=1:E

        ########### THIS PART CONSIDERS MSE OF TRUE OUTPUT AND ONE SINGLE REALIZATION OF THE SIMULATED OUTPUT

        # diffs[k,m] is the difference at time instant k using realization m
        diffs   = Y[:,e].-Ym_prop
        diffs1  = Y[:,e].-Ym_prop1
        diffs2  = Y[:,e].-Ym_prop2
        diffs3  = Y[:,e].-Ym_prop3
        # costs (row-vector), costs[m] is the cost using realization m
        costs   = (1/(N+1-N_trans))*sum(diffs[N_trans+1:end,:].^2, dims=1)
        costs1  = (1/(N+1-N_trans))*sum(diffs1[N_trans+1:end,:].^2, dims=1)
        costs2  = (1/(N+1-N_trans))*sum(diffs2[N_trans+1:end,:].^2, dims=1)
        costs3  = (1/(N+1-N_trans))*sum(diffs3[N_trans+1:end,:].^2, dims=1)
        # derivs[m, e] is the derivative for cost of data-set e using simulated realization m
        derivs1[:,e] = transpose((2/(N+1-N_trans))*sum(-diffs[N_trans+1:end,:].*sens1[N_trans+1:end,:], dims=1))
        derivs2[:,e] = transpose((2/(N+1-N_trans))*sum(-diffs[N_trans+1:end,:].*sens2[N_trans+1:end,:], dims=1))
        derivs3[:,e] = transpose((2/(N+1-N_trans))*sum(-diffs[N_trans+1:end,:].*sens3[N_trans+1:end,:], dims=1))
        # ests[m,e] is the estimated derivative for cost of realization e using simulated realization m
        ests1[:,e]   = transpose(costs1-costs)/Δ
        ests2[:,e]   = transpose(costs2-costs)/Δ
        ests3[:,e]   = transpose(costs3-costs)/Δ

        ######## THIS PART CONSIDERED MSE OF TRUE OUTPUT AND MEAN SIMULATED OUTPUT OVER M REALIZATIONS #############

        # diffsE[k] difference data and mean realization at timestep k
        diffsE   = Y[:,e].-mean(Ym_prop[:,1:Mlim], dims=2)
        diffs1E  = Y[:,e].-mean(Ym_prop1[:,1:Mlim], dims=2)
        diffs2E  = Y[:,e].-mean(Ym_prop2[:,1:Mlim], dims=2)
        diffs3E  = Y[:,e].-mean(Ym_prop3[:,1:Mlim], dims=2)
        # costsE cost for data-set e
        costsE   = first((1/(N+1-N_trans))*sum(diffsE[N_trans+1:end].^2, dims=1))
        costsE1  = first((1/(N+1-N_trans))*sum(diffs1E[N_trans+1:end].^2, dims=1))
        costsE2  = first((1/(N+1-N_trans))*sum(diffs2E[N_trans+1:end].^2, dims=1))
        costsE3  = first((1/(N+1-N_trans))*sum(diffs3E[N_trans+1:end].^2, dims=1))
        # derivsE[e] derivative of cost for data-set e
        derivs1E[e] = first((2/(N+1-N_trans))*sum(-diffsE[N_trans+1:end].*mean(sens1[N_trans+1:end,1:Mlim], dims=2)))
        derivs2E[e] = first((2/(N+1-N_trans))*sum(-diffsE[N_trans+1:end].*mean(sens2[N_trans+1:end,1:Mlim], dims=2)))
        derivs3E[e] = first((2/(N+1-N_trans))*sum(-diffsE[N_trans+1:end].*mean(sens3[N_trans+1:end,1:Mlim], dims=2)))
        # estsE[e] estimated derivative of cost for data-set e
        ests1E[e]   = (costsE1-costsE)/Δ
        ests2E[e]   = (costsE2-costsE)/Δ
        ests3E[e]   = (costsE3-costsE)/Δ

        ### ANALYZING MEAN AND VARIANCE OF GRADIENT ESTIMATES AS A FUNCTION OF M ###
        for limM = 1:maxM
            for iblock = 1:num_blocks
                range  = (iblock-1)*limM+1:iblock*limM
                range2 = M÷2+(iblock-1)*limM+1:M÷2+iblock*limM
                # diffsE[k] difference data and mean realization at timestep k
                diffsM   = Y[:,e].-mean(Ym_prop[:,range], dims=2)
                diffs1M  = Y[:,e].-mean(Ym_prop1[:,range], dims=2)
                diffs2M  = Y[:,e].-mean(Ym_prop2[:,range], dims=2)
                diffs3M  = Y[:,e].-mean(Ym_prop3[:,range], dims=2)
                # costsE cost for data-set e
                costsM   = first((1/(N+1-N_trans))*sum(diffsM[N_trans+1:end].^2, dims=1))
                costsM1  = first((1/(N+1-N_trans))*sum(diffs1M[N_trans+1:end].^2, dims=1))
                costsM2  = first((1/(N+1-N_trans))*sum(diffs2M[N_trans+1:end].^2, dims=1))
                costsM3  = first((1/(N+1-N_trans))*sum(diffs3M[N_trans+1:end].^2, dims=1))
                # derivsE[e] derivative of cost for data-set e
                derivs1M[iblock, e] = first((2/(N+1-N_trans))*sum(-diffsM[N_trans+1:end].*mean(sens1[N_trans+1:end,range2], dims=2)))
                derivs2M[iblock, e] = first((2/(N+1-N_trans))*sum(-diffsM[N_trans+1:end].*mean(sens2[N_trans+1:end,range2], dims=2)))
                derivs3M[iblock, e] = first((2/(N+1-N_trans))*sum(-diffsM[N_trans+1:end].*mean(sens3[N_trans+1:end,range2], dims=2)))
            end
        end
    end




    ##############################################
    ########## ALSO COMPUTE MSE?????? ############
    ##############################################

    # diffs   = Y[:,1].-Ym_prop
    # diffs1  = Y[:,1].-Ym_prop1
    # diffs2  = Y[:,1].-Ym_prop2
    # diffs3  = Y[:,1].-Ym_prop3
    # costs   = (1/(N+1-N_trans))*sum(diffs[N_trans+1:end,:].^2, dims=1)
    # costs1  = (1/(N+1-N_trans))*sum(diffs1[N_trans+1:end,:].^2, dims=1)
    # costs2  = (1/(N+1-N_trans))*sum(diffs2[N_trans+1:end,:].^2, dims=1)
    # costs3  = (1/(N+1-N_trans))*sum(diffs3[N_trans+1:end,:].^2, dims=1)
    # derivs1 = (2/(N+1-N_trans))*sum(-diffs[N_trans+1:end,:].*sens1[N_trans+1:end,:], dims=1)
    # derivs2 = (2/(N+1-N_trans))*sum(-diffs[N_trans+1:end,:].*sens2[N_trans+1:end,:], dims=1)
    # derivs3 = (2/(N+1-N_trans))*sum(-diffs[N_trans+1:end,:].*sens3[N_trans+1:end,:], dims=1)
    # ests1   = (costs1-costs)/Δ
    # ests2   = (costs2-costs)/Δ
    # ests3   = (costs3-costs)/Δ
    return derivs1, derivs2, derivs3, ests1, ests2, ests3, derivs1E, derivs2E, derivs3E, ests1E, ests2E, ests3E, derivs1M, derivs2M, derivs3M
end

function estimate_gradient_directions(expid::String, N_trans::Int = 0)
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

    pars_true = [0.3, 6.25, 6.25, 0.8, 16, 1]
    pars_to_try = [0.7*pars_true, 0.8*pars_true, 0.9*pars_true, pars_true, 1.1*pars_true, 1.2*pars_true, 1.3*pars_true]
    num_tests = length(pars_to_try)

    if !isdir(joinpath(data_dir, "tmp/"))
        mkdir(joinpath(data_dir, "tmp/"))
    end

    get_all_parameters(free_pars::Array{Float64, 1}) = vcat(get_all_θs(free_pars), exp_data.get_all_ηs(free_pars))

    function get_gradient_estimate_base(y, dummy_input, free_pars)
        Yb, jacB = solvew_sens(u, t -> zeros(n_out+length(dist_par_inds)*n_out), free_pars, N) |> h_comp
        # Doesn't need to use different noise realizations for estimate of output and estiamte of jacobian since output is deterministic

        return get_cost_gradient(y, Yb[:,1:1], [jacB], N_trans)
    end
    # Returns estimate of gradient of cost function
    # M_mean specifies over how many realizations the gradient estimate is computed
    function get_gradient_estimate(y, free_pars, isws, M_mean::Int=1)
        Ym, jacsYm = simulate_system_sens(exp_data, free_pars, 2M_mean, dist_par_inds, isws)

        # Uses different noise realizations for estimate of output and estiamte of jacobian
        return get_cost_gradient(y, Ym[:,1:M_mean], jacsYm[M_mean+1:end], N_trans)
    end
    get_gradient_estimate_p(free_pars, M_mean) = get_gradient_estimate(Y[:,1], free_pars, isws, M_mean)

    writedlm(joinpath("data/results/gradient_dirs/pars_tried.csv"), transpose(hcat(pars_to_try...)), ',')
    writedlm(joinpath("data/results/gradient_dirs/$(expid).csv"), ["Name of this file is the experiment ID that was used"], ',')

    gradPs = zeros(length(pars_to_try), length(pars_true))
    gradBs = zeros(length(pars_to_try), length(pars_true))
    # gradB = Array{Float64}(undef, 3,1)
    # gradP = Array{Float64}(undef, 3,1)

    for (ind, pars) in enumerate(pars_to_try)
        # gradB = zeros(3,1)
        gradBs[ind,:] = get_gradient_estimate_base(Y[:,1], zeros(3,1), pars)
        gradPs[ind,:] = get_gradient_estimate_p(pars, 100)
        # writedlm(joinpath(data_dir, "tmp/grad_dir_backup_b_$(Dates.format(now(), "yymdHMS")).csv"), gradB, ',')
        # writedlm(joinpath(data_dir, "tmp/grad_dir_backup_p_$(Dates.format(now(), "yymdHMS")).csv"), gradP, ',')
        writedlm(joinpath(data_dir, "tmp/grad_dir_backup_b_$(ind).csv"), gradBs[ind,:], ',')
        writedlm(joinpath(data_dir, "tmp/grad_dir_backup_p_$(ind).csv"), gradPs[ind,:], ',')
    end

    writedlm(joinpath("data/results/gradient_dirs/gradBs.csv"), gradBs, ',')
    writedlm(joinpath("data/results/gradient_dirs/gradPs.csv"), gradPs, ',')

    return pars_to_try, gradBs, gradPs
end

function compute_forward_difference_derivative(x_vals::Array{Float64,1}, y_vals::Array{Float64,1})
    @assert (length(x_vals) == length(y_vals)) "x_vals and y_vals must contain the same number of elements"
    diff = zeros(size(y_vals))
    for i=1:length(y_vals)-1
        diff[i] = (y_vals[i+1]-y_vals[i])/(x_vals[i+1]-x_vals[i])
    end
    diff[end] = diff[end-1]
    return diff
end

# Checks if analytical sensitivity of w wrt to disturbance parameter matches
# the numerically obtained approximations
function test_disturbance_sensitivities(expid::String)
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

    # For full pendulum model with one unknown disturbance parameter
    my_δ = 0.0001
    free_pars1 = [0.3, 6.25, 6.25, 1.0]
    free_pars2 = [0.3, 6.25, 6.25, 1.0+my_δ]

    Zm = [randn(Nw, n_tot) for m = 1:M]
    η1 = exp_data.get_all_ηs(free_pars1)
    η2 = exp_data.get_all_ηs(free_pars2)

    dmdl1 = discretize_ct_noise_model_with_sensitivities(get_ct_disturbance_model(η1, nx, n_out), δ, dist_par_inds)
    dmdl2 = discretize_ct_noise_model_with_sensitivities(get_ct_disturbance_model(η2, nx, n_out), δ, dist_par_inds)
    # # NOTE: OPTION 1: Use the rows below here for linear interpolation
    XWm1 = simulate_noise_process_mangled(dmdl1, Zm)
    XWm2 = simulate_noise_process_mangled(dmdl2, Zm)
    wmm1(m::Int) = mk_noise_interp(dmdl1.Cd, XWm1, m, δ)
    wmm2(m::Int) = mk_noise_interp(dmdl2.Cd, XWm2, m, δ)

    times = 0.0:0.1:N*δ
    w1s = [wmm1(1)(t)[1] for t=times]
    w2s = [wmm2(1)(t)[1] for t=times]
    wdiffs_analytical = [wmm1(1)(t)[2] for t=times]
    wdiffs = (w2s-w1s)./my_δ

    p = plot(wdiffs)
    plot!(p, wdiffs_analytical)
end

# Ultimate stochastic adjoint debugging function
function debug_adjoint_stochastic(expid::String, N_trans::Int=0)
    Random.seed!(123)
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

    get_all_parameters(free_pars::Array{Float64, 1}) = vcat(get_all_θs(free_pars), exp_data.get_all_ηs(free_pars))
    p = get_all_parameters(free_dyn_pars_true) # All true parameters
    θ = p[1:dθ]                                # All true dynamical parameters
    η = p[dθ+1: dθ+dη]                         # All true disturbance parameters
    free_pars_true = vcat(free_dyn_pars_true, W_meta.η[dist_par_inds])  # All true free parameters

    Zm = [randn(Nw, n_tot) for m = 1:2M]

    # --------------------------------------------------------------------------
    # --------------- Forward solution of nominal system -----------------------
    # --------------------------------------------------------------------------
    dmdl_true = discretize_ct_noise_model(get_ct_disturbance_model(η, nx, n_out), δ)
    # NOTE: OPTION 1: Use the rows below here for linear interpolation
    XWm = simulate_noise_process_mangled(dmdl_true, Zm)
    wmm(m::Int) = mk_noise_interp(dmdl_true.Cd, XWm, m, δ)

    # forward_solve(m) = solvew(u, wmm(m), free_pars_true, N)                 # Without sensitivity analysis
    forward_solve(m) = solvew_sens(u, wmm(m), free_pars_true, N)              # With sensitivity analysis
    @info "Starting forward solving..."
    # @warn "Used to simulate with 2M realizations here, now only with M, to get determinstic cost function"
    sols_for = get_sol_in_parallel(forward_solve, 1:2M)
    @info "Finished forward solving!"

    # --------------------------------------------------------------------------
    # ------------- Cost gradient using forward sensitivty ---------------------
    # --------------------------------------------------------------------------

    # NOTE: Assumes forward_solve included sensitivities

    my_np = 1   # Only identifying one parameter, namely k
    Ys = zeros(N+1, length(sols_for))
    senss = [zeros(N+1, my_np) for j=1:length(sols_for)]
    cost_grads = zeros(M)
    for (i, sol) = enumerate(sols_for)
        Yi, sensi = apply_two_outputfun_mvar(f, f_sens, sol)
        Ys[:,i] = Yi
        senss[i] = sensi
    end

    # TODO: Delete
    # for m=1:M
    #     cost_grads[m] = first(get_cost_gradient(Y[:, 1], Ys[:,m:m], [senss[M+m]], N_trans))
    # end

    for_dif_est = first(get_cost_gradient(Y[:, 1], Ys[:,1:M], senss[M+1:2M], N_trans))
    # for_dif_est = first(get_cost_gradient(Y[:, 1], Ys[:,1:M], senss[1:M], N_trans))

    for_dif_est2 = 0.0
    for m=1:M
        # for_dif_est2 += first(get_cost_gradient(Y[:,1], Ys[:,m:m], senss[m:m]))
        for_dif_est2 += first(get_cost_gradient(Y[:,1], Ys[:,m:m], senss[M+m:M+m]))
    end
    for_dif_est2 = for_dif_est2/M;

    # For extracting Ys if not using forward sensitivity
    Ys = zeros(N+1, length(sols_for))
    for (i, sol) = enumerate(sols_for)
        Yi = apply_outputfun(f, sol)
        Ys[:,i] = Yi
    end

    # --------------------------------------------------------------------------
    # ------------ Numerical approximation of cost gradient --------------------
    # --------------------------------------------------------------------------

    my_δ = 0.01
    forward_solve2(m) = solvew(u, wmm(m), free_pars_true.+my_δ, N) |> h # NOTE: Not solving for sensitivities here
    # @warn "Used to simulate with 2M realizations here, now only with M, to get determinstic cost function"
    Ym2 = solve_in_parallel(forward_solve2, collect(1:2M))

    numcost1 = get_cost_value(Y[:,1], Ys,  N_trans)
    numcost2 = get_cost_value(Y[:,1], Ym2, N_trans)
    num_est = (numcost2-numcost1)/my_δ

    num_est2 = 0.0
    for m = 1:M
        cost  = get_cost_value(Y[:,1], Ys[:,m:m], N_trans)
        cost2 = get_cost_value(Y[:,1], Ym2[:,m:m], N_trans)
        num_est2 += (cost2-cost)/my_δ
    end
    num_est2 = num_est2/M

    # # NOTE: TODO: Manual way gives sliightly different result, I rly don't know why!!!!!!!!!!!
    # # ------------------ MANUAL WAY ----------------------
    # sol_for = solvew_sens(exp_data.u, wmm(1), free_dyn_pars_true, N)
    # for_sol_data = h_debug(sol_for)
    # for_sol_mat = zeros(length(sol_for.u),length(sol_for.u[1]))
    # for i=eachindex(sol_for.u)
    #     for j=eachindex(sol_for.u[1])
    #         for_sol_mat[i,j] = sol_for.u[i][j]
    #     end
    # end
    # Y1 = for_sol_mat[:,7]
    # sens1 = for_sol_mat[:,14]
    #
    # sol_for2 = solvew_sens(exp_data.u, wmm(2), free_dyn_pars_true, N)
    # for_sol_data2 = h_debug(sol_for2)
    # for_sol_mat2 = zeros(length(sol_for2.u),length(sol_for2.u[1]))
    # for i=eachindex(sol_for2.u)
    #     for j=eachindex(sol_for2.u[1])
    #         for_sol_mat2[i,j] = sol_for2.u[i][j]
    #     end
    # end
    # Y2 = for_sol_mat[:,7]
    # sens2 = for_sol_mat[:,14]
    #
    # # Obtaining cost derivative (deterministic cost from one realization)
    # for_dif_est = first(get_cost_gradient(Y[1:N+1, 1], reshape(Y1, length(Y1), 1), [reshape(sens1, length(sens1),1)], N_trans))
    # ----------------------------------------------------

    # --------------------------------------------------------------------------
    # ------------- Solution of adjoint system (backwards) ---------------------
    # --------------------------------------------------------------------------
    # Computing dx
    function get_der_est(sol)
        der_est = (sol.u[2:end]-sol.u[1:end-1])/Ts
        # ts = sol.t[1:end-1]
        ts = 0:Ts:(N-1)*Ts
        return ts, der_est
    end
    function get_der_est(vals::Matrix{Float64})
        # Rows of matrix are assumed to be different values of t, columns of
        # matrix are assumed to be different elements of the vector-valued process
        der_est = (vals[2:end,:]-vals[1:end-1,:])/Ts
        ts = 0:Ts:(N-1)*Ts
        return ts, der_est
        # Returns range of times, and matrix der_est with same structure as vals, just one row fewer
    end
    function get_mvar_cubic(ts, der_est::Vector{Vector{Float64}})
        temp = [cubic_spline_interpolation(ts, [der_est[i][j] for i=1:length(der_est)], extrapolation_bc=Line()) for j=1:length(der_est[1])]
        t -> [temp[i](t) for i=1:length(temp)]
        # return [cubic_spline_interpolation(ts, [der_est[i][j] for i=1:length(der_est)]) for j=1:length(der_est[1])]
    end
    function get_mvar_cubic(ts, der_est::AbstractMatrix{Float64})
        # Rows of der_est are assumed to be different values of t, columns of
        # matrix are assumed to be different elements of the vector-valued process
        temp = [cubic_spline_interpolation(ts, der_est[:,i], extrapolation_bc=Line()) for i=1:size(der_est,2)]
        # temp = [t->t for i=1:size(der_est,2)]
        return t -> [temp[i](t) for i=eachindex(temp)]
        # Returns a function mapping time to the vector-value of the function at that time
    end

    y_func  = interpolated_signal(Y[N_trans+1:end,1], 0:Ts:(size(Y,1)-1)*Ts)
    dY_est  = (Y[2:end,1]-Y[1:end-1,1])/Ts
    dy_func = interpolated_signal(dY_est, 0:Ts:(size(dY_est,1)-1)*Ts)

    # Computing xp0, initial conditions of derivative of x wrt to p
    mdl = model_to_use(φ0, u, wmm(1), p)
    mdl_sens = model_sens_to_use(φ0, u, wmm(1), p)
    n_mdl = length(mdl.x0)
    xp0 = reshape(mdl_sens.x0[n_mdl+1:end], n_mdl, :)


    # # -------- MANUAL WAY -------------
    # ts, der_est = get_der_est(sol_for)
    # dx = get_mvar_cubic(ts, der_est)
    # _, der_est2 = get_der_est(sol_for2)
    # dx2 = get_mvar_cubic(ts, der_est2)
    # # ts, der_est = get_der_est(sols_for[1])
    # # dx = get_mvar_cubic(ts, der_est)
    # # _, der_est2 = get_der_est(sols_for[2])
    # # dx2 = get_mvar_cubic(ts, der_est2)
    # mdl_adj, get_Gp = my_pendulum_adjoint_konly(u, wmm(1), p, N*Ts, sol_for, sol_for2, y_func, dy_func, xp0, dx, dx2)
    # adj_prob = problem(mdl_adj, N, Ts)
    # adj_sol = solve(adj_prob, saveat = 0:Ts:N*Ts, abstol = abstol, reltol = reltol,
    #     maxiters = maxiters)
    # Gp = first(get_Gp(adj_sol))
    # # ---------------------------------

    function calc_adj_m(m::Int)
        ts, der_est = get_der_est(sols_for[m])
        dx = get_mvar_cubic(ts, der_est)
        # Stochastic cost
        _, der_est2 = get_der_est(sols_for[M+m])
        dx2 = get_mvar_cubic(ts, der_est2)
        mdl_adj, get_Gp = model_adj_to_use(u, wmm(m), p, N*Ts, sols_for[m], sols_for[M+m], y_func, dy_func, xp0, dx, dx2)
        # mdl_adj, get_Gp = model_adj_to_use(u, wmm(m), p, N*Ts, sols_for[m], sols_for[m], y_func, dy_func, xp0, dx, dx)
        adj_prob = problem(mdl_adj, N, Ts)
        adj_sol = solve(adj_prob, saveat = 0:Ts:N*Ts, abstol = abstol, reltol = reltol,
            maxiters = maxiters)
        return get_Gp(adj_sol)
    end

    adj_ms = collect(1:M)
    Gps = solve_adj_in_parallel(calc_adj_m, adj_ms)
    Gp = mean(Gps)

    # # DEBUG: Alternative way to compute Gp, just to compare
    # x_for   = for_sol_mat[:, 1:7]
    # xθ_for  = for_sol_mat[:, 8:14]
    # adj_sol_mat = zeros(N+1,8)
    # for i=eachindex(adj_sol.u)
    #     for j=eachindex(adj_sol.u[1])
    #         adj_sol_mat[i,j] = adj_sol.u[i][j]
    #     end
    # end
    # λs = adj_sol_mat[end:-1:1,1:7]
    # βs = adj_sol_mat[end:-1:1,8]
    # ts, λ_der = get_der_est(λs)
    # λ_func  = get_mvar_cubic(0:Ts:N*Ts, λs)
    # dλ_func = get_mvar_cubic(ts, λ_der)
    #
    # Fdx = (x1, x2) -> vcat([1   0   0          0   0   2x1    0
    #                         0   1   0          0   0   2x2    0
    #                         0   0   -x1      0.3   0   0      0
    #                         0   0   -x2      0   0.3   0      0], zeros(3,7))
    # # TODO: I should be able to obtain these from the forward problem!
    # # NOTE: One of the terms here should be zero, if we initialized correctly
    # term = (λ_func(N*Ts)')*Fdx(x_for[end,1], x_for[end,2])*xθ_for[end,:] - (λ_func(0.)')*Fdx(x_for[1,1], x_for[1,2])*xθ_for[1,:]
    # Gp_alt = βs[1] - term
    # # End of DEBUG: Alternative way to compute Gp

    return num_est, for_dif_est, Gp, num_est2, for_dif_est2
    # return num_est, for_dif_est, Gp, Gps, num_est2, for_dif_est2
end

function solve_adjoint_deterministic(expid::String, N_trans::Int=0)
    Random.seed!(123)
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

    get_all_parameters(free_pars::Array{Float64, 1}) = vcat(get_all_θs(free_pars), exp_data.get_all_ηs(free_pars))
    p = get_all_parameters(free_dyn_pars_true) # All true parameters
    θ = p[1:dθ]                                # All true dynamical parameters
    η = p[dθ+1: dθ+dη]                         # All true disturbance parameters

    Zm = [randn(Nw, n_tot) for m = 1:M]

    # --------------------------------------------------------------------------
    # --------------- Forward solution of nominal system -----------------------
    # --------------------------------------------------------------------------
    dmdl_true = discretize_ct_noise_model(get_ct_disturbance_model(η, nx, n_out), δ)

    @info "HERE COMES THE DISTURBANCE MODEL: "
    println(get_ct_disturbance_model(η, nx, n_out))
    @info "and η: $η"
    # NOTE: OPTION 1: Use the rows below here for linear interpolation
    XWm = simulate_noise_process_mangled(dmdl_true, Zm)
    wmm(m::Int) = mk_noise_interp(dmdl_true.Cd, XWm, m, δ)

    sol_for = solvew_sens(u, wmm(1), free_dyn_pars_true, N)

    # --------------------------------------------------------------------------
    # ------------- Solution of adjoint system (backwards) ---------------------
    # --------------------------------------------------------------------------
    # Computing dx
    function get_der_est(sol)
        der_est = (sol.u[2:end]-sol.u[1:end-1])/Ts
        # ts = sol.t[1:end-1]
        ts = 0:Ts:(N-1)*Ts
        return ts, der_est
    end
    function get_der_est(vals::Matrix{Float64})
        # Rows of matrix are assumed to be different values of t, columns of
        # matrix are assumed to be different elements of the vector-valued process
        der_est = (vals[2:end,:]-vals[1:end-1,:])/Ts
        ts = 0:Ts:(N-1)*Ts
        return ts, der_est
        # Returns range of times, and matrix der_est with same structure as vals, just one row fewer
    end
    function get_mvar_cubic(ts, der_est::Vector{Vector{Float64}})
        temp = [cubic_spline_interpolation(ts, [der_est[i][j] for i=1:length(der_est)], extrapolation_bc=Line()) for j=1:length(der_est[1])]
        t -> [temp[i](t) for i=1:length(temp)]
        # return [cubic_spline_interpolation(ts, [der_est[i][j] for i=1:length(der_est)]) for j=1:length(der_est[1])]
    end
    function get_mvar_cubic(ts, der_est::AbstractMatrix{Float64})
        # Rows of der_est are assumed to be different values of t, columns of
        # matrix are assumed to be different elements of the vector-valued process
        temp = [cubic_spline_interpolation(ts, der_est[:,i], extrapolation_bc=Line()) for i=1:size(der_est,2)]
        # temp = [t->t for i=1:size(der_est,2)]
        return t -> [temp[i](t) for i=eachindex(temp)]
        # Returns a function mapping time to the vector-value of the function at that time
    end

    y_func  = interpolated_signal(Y[:,1], 0:Ts:(size(Y,1)-1)*Ts)
    dY_est  = (Y[2:end,1]-Y[1:end-1,1])/Ts
    dy_func = interpolated_signal(dY_est, 0:Ts:(size(dY_est,1)-1)*Ts)
    ts, der_est = get_der_est(sol_for)
    dx = get_mvar_cubic(ts, der_est)

    # Computing xp0, initial conditions of derivative of x wrt to p
    mdl = model_to_use(φ0, u, wmm(1), p)
    mdl_sens = model_sens_to_use(φ0, u, wmm(1), p)
    n_mdl = length(mdl.x0)
    # In case model_sens_to_use computes multuple sensitivities, we only pick the first one
    xp0 = f_sens_deb(mdl_sens.x0) # NOTE: NEW

    # mdl_adj, get_Gp = my_pendulum_adjoint_konly(u, wmm(1), p, N*Ts, sol_for, sol_for, y_func, dy_func, xp0, dx, dx)
    mdl_adj, get_Gp = model_adj_to_use(u, wmm(1), p, N*Ts, sol_for, sol_for, y_func, dy_func, xp0, dx, dx)
    adj_prob = problem(mdl_adj, N, Ts)
    adj_sol = solve(adj_prob, saveat = 0:Ts:N*Ts, abstol = abstol, reltol = reltol,
        maxiters = maxiters)

    Gp = first(get_Gp(adj_sol))

    solmat = zeros(N+1,8)
    for i=eachindex(adj_sol.u)
        for j=eachindex(adj_sol.u[1])
            solmat[i,j] = adj_sol.u[i][j]
        end
    end
    # Reverses the obtained vector, since adjoint problem was solved in reverse
    λs = solmat[end:-1:1,1:7]
    βs = solmat[end:-1:1,8]

    # DEBUG: Alternative way to compute Gp, just to compare
    x_for, xθ_for = h_sens_deb(sol_for)
    ts, λ_der = get_der_est(λs)
    λ_func  = get_mvar_cubic(0:Ts:N*Ts, λs)
    dλ_func = get_mvar_cubic(ts, λ_der)

    term = 0.0
    if model_id == PENDULUM
        Fdx = (x1, x2) -> vcat([1   0   0          0   0   2x1    0
                                0   1   0          0   0   2x2    0
                                0   0   -x1      0.3   0   0      0
                                0   0   -x2      0   0.3   0      0], zeros(3,7))
    elseif model_id == MOH_MDL
        Fdx = (x1, x2) -> [1.0 0.0; 0.0 0.0]
    end
    # TODO: I should be able to obtain these from the forward problem!
    # NOTE: One of the terms here should be zero, if we initialized correctly
    term = (λ_func(N*Ts)')*Fdx(x_for[end,1], x_for[end,2])*xθ_for[end,:] - (λ_func(0.)')*Fdx(x_for[1,1], x_for[1,2])*xθ_for[1,:]
    Gp_alt = βs[1] - term
    # End of DEBUG: Alternative way to compute Gp

    println("Let's also print things in here")
    println((λ_func(N*Ts)'))
    println(Fdx(x_for[end,1], x_for[end,2]))
    println(xθ_for[end,:])
    println(λ_func(0.))
    println(Fdx(x_for[1,1], x_for[1,2]))
    println(xθ_for[1,:])
    println("Finished also printing things in here")

    return Gp, λs, βs, sol_for, wmm, Gp_alt, term
end

# Ultimate deterministic adjoint debugging function
function debug_adjoint_deterministic(expid::String, N_trans::Int=0)
    Random.seed!(12453)
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

    get_all_parameters(free_pars::Array{Float64, 1}) = vcat(get_all_θs(free_pars), exp_data.get_all_ηs(free_pars))
    p = get_all_parameters(free_dyn_pars_true) # All true parameters
    θ = p[1:dθ]                                # All true dynamical parameters
    η = p[dθ+1: dθ+dη]                         # All true disturbance parameters

    Zm = [randn(Nw, n_tot) for m = 1:M]

    # --------------------------------------------------------------------------
    # --------------- Forward solution of nominal system -----------------------
    # --------------------------------------------------------------------------
    dmdl_true = discretize_ct_noise_model(get_ct_disturbance_model(η, nx, n_out), δ)
    # NOTE: OPTION 1: Use the rows below here for linear interpolation
    XWm = simulate_noise_process_mangled(dmdl_true, Zm)
    wmm(m::Int) = mk_noise_interp(dmdl_true.Cd, XWm, m, δ)

    sol_for = solvew_sens(u, wmm(1), free_dyn_pars_true, N)
    # for_sol_data = h_debug(sol_for)

    # Y1, sens1 = sol_for |> h_comp
    Y1, sens1 = h_comp(sol_for)

    # for_sol_mat = zeros(length(sol_for.u),length(sol_for.u[1]))
    # for i=eachindex(sol_for.u)
    #     for j=eachindex(sol_for.u[1])
    #         for_sol_mat[i,j] = sol_for.u[i][j]
    #     end
    # end
    # Y1 = for_sol_mat[:,num_dyn_vars]
    # sens1 = for_sol_mat[:,2num_dyn_vars]

    # Obtaining cost derivative (deterministic cost from one realization)
    for_dif_est = first(get_cost_gradient(Y[1:N+1, 1], reshape(Y1, length(Y1), 1), [reshape(sens1, length(sens1),1)], N_trans))

    my_δ = 0.01
    Y2 = solvew(exp_data.u, wmm(1), free_dyn_pars_true.+my_δ, N) |> h
    cost  = get_cost_value(Y[:,1], reshape(Y1, length(Y1),1), N_trans)
    cost2 = get_cost_value(Y[:,1], reshape(Y2, length(Y2),1), N_trans)
    num_est = (cost2-cost)/my_δ

    # --------------------------------------------------------------------------
    # ------------- Solution of adjoint system (backwards) ---------------------
    # --------------------------------------------------------------------------
    # Computing dx
    function get_der_est(sol)
        der_est = (sol.u[2:end]-sol.u[1:end-1])/Ts
        # ts = sol.t[1:end-1]
        ts = 0:Ts:(N-1)*Ts
        return ts, der_est
    end
    function get_der_est(vals::Matrix{Float64})
        # Rows of matrix are assumed to be different values of t, columns of
        # matrix are assumed to be different elements of the vector-valued process
        der_est = (vals[2:end,:]-vals[1:end-1,:])/Ts
        ts = 0:Ts:(N-1)*Ts
        return ts, der_est
        # Returns range of times, and matrix der_est with same structure as vals, just one row fewer
    end
    function get_mvar_cubic(ts, der_est::Vector{Vector{Float64}})
        temp = [cubic_spline_interpolation(ts, [der_est[i][j] for i=1:length(der_est)], extrapolation_bc=Line()) for j=1:length(der_est[1])]
        t -> [temp[i](t) for i=1:length(temp)]
        # return [cubic_spline_interpolation(ts, [der_est[i][j] for i=1:length(der_est)]) for j=1:length(der_est[1])]
    end
    function get_mvar_cubic(ts, der_est::AbstractMatrix{Float64})
        # Rows of der_est are assumed to be different values of t, columns of
        # matrix are assumed to be different elements of the vector-valued process
        temp = [cubic_spline_interpolation(ts, der_est[:,i], extrapolation_bc=Line()) for i=1:size(der_est,2)]
        # temp = [t->t for i=1:size(der_est,2)]
        return t -> [temp[i](t) for i=eachindex(temp)]
        # Returns a function mapping time to the vector-value of the function at that time
    end

    y_func  = interpolated_signal(Y[:,1], 0:Ts:(size(Y,1)-1)*Ts)
    dY_est  = (Y[2:end,1]-Y[1:end-1,1])/Ts
    dy_func = interpolated_signal(dY_est, 0:Ts:(size(dY_est,1)-1)*Ts)
    ts, der_est = get_der_est(sol_for)
    dx = get_mvar_cubic(ts, der_est)

    # Computing xp0, initial conditions of derivative of x wrt to p
    mdl = model_to_use(φ0, u, wmm(1), p)
    mdl_sens = model_sens_to_use(φ0, u, wmm(1), p)
    n_mdl = length(mdl.x0)
    # In case model_sens_to_use computes multuple sensitivities, we only pick the first one
    xp0 = reshape(mdl_sens.x0[n_mdl+1:end], n_mdl, :)[:,1]
    # xp0 = f_sens_deb(mdl_sens.x0) # NOTE: NEW

    # mdl_adj, get_Gp = my_pendulum_adjoint_konly(u, wmm(1), p, N*Ts, sol_for, sol_for, y_func, dy_func, xp0, dx, dx)
    mdl_adj, get_Gp = model_adj_to_use(u, wmm(1), p, N*Ts, sol_for, sol_for, y_func, dy_func, xp0, dx, dx)
    adj_prob = problem(mdl_adj, N, Ts)
    adj_sol = solve(adj_prob, saveat = 0:Ts:N*Ts, abstol = abstol, reltol = reltol,
        maxiters = maxiters)

    Gp = first(get_Gp(adj_sol))
    # DEBUG: Alternative way to compute Gp, just to compare
    x_for, xθ_for = h_sens_deb(sol_for)
    @assert (xθ_for[N-10, end] == sens1[N-10]) "There is a mismatch between values returned from h_sens_deb and h_comp, make sure both functions have been updated correctly for the current choice of the sensitivity variable"
    # x_for   = for_sol_mat[:, 1:num_dyn_vars]
    # xθ_for  = for_sol_mat[:, num_dyn_vars+1:2num_dyn_vars]
    adj_sol_mat = zeros(N+1,num_dyn_vars+1)
    for i=eachindex(adj_sol.u)
        for j=eachindex(adj_sol.u[1])
            adj_sol_mat[i,j] = adj_sol.u[i][j]
        end
    end
    λs = adj_sol_mat[end:-1:1,1:num_dyn_vars]
    βs = adj_sol_mat[end:-1:1,num_dyn_vars+1]
    ts, λ_der = get_der_est(λs)
    λ_func  = get_mvar_cubic(0:Ts:N*Ts, λs)
    dλ_func = get_mvar_cubic(ts, λ_der)

    term = 0.0
    if model_id == PENDULUM
        Fdx = (x1, x2) -> vcat([1   0   0          0   0   2x1    0
                                0   1   0          0   0   2x2    0
                                0   0   -x1      0.3   0   0      0
                                0   0   -x2      0   0.3   0      0], zeros(3,7))
    elseif model_id == MOH_MDL
        Fdx = (x1, x2) -> [1.0 0.0; 0.0 0.0]
    end
    # TODO: I should be able to obtain these from the forward problem!
    # NOTE: One of the terms here should be zero, if we initialized correctly
    term = (λ_func(N*Ts)')*Fdx(x_for[end,1], x_for[end,2])*xθ_for[end,:] - (λ_func(0.)')*Fdx(x_for[1,1], x_for[1,2])*xθ_for[1,:]
    Gp_alt = βs[1] - term
    # End of DEBUG: Alternative way to compute Gp

    return num_est, for_dif_est, Gp, Gp_alt#, debug_thing, term#, λs
end

# Ultimate deterministic adjoint debugging function
function debug_adjoint_semistochastic(expid::String, N_trans::Int=0)
    Random.seed!(123)
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

    get_all_parameters(free_pars::Array{Float64, 1}) = vcat(get_all_θs(free_pars), exp_data.get_all_ηs(free_pars))
    p = get_all_parameters(free_dyn_pars_true) # All true parameters
    θ = p[1:dθ]                                # All true dynamical parameters
    η = p[dθ+1: dθ+dη]                         # All true disturbance parameters
    free_pars_true = vcat(free_dyn_pars_true, W_meta.η[dist_par_inds])  # All true free parameters

    # Zm = [randn(Nw, n_tot) for m = 1:2M]
    Zm = [randn(Nw, n_tot) for m = 1:M]

    # --------------------------------------------------------------------------
    # --------------- Forward solution of nominal system -----------------------
    # --------------------------------------------------------------------------
    dmdl_true = discretize_ct_noise_model(get_ct_disturbance_model(η, nx, n_out), δ)
    # NOTE: OPTION 1: Use the rows below here for linear interpolation
    XWm = simulate_noise_process_mangled(dmdl_true, Zm)
    wmm(m::Int) = mk_noise_interp(dmdl_true.Cd, XWm, m, δ)

    forward_solve(m) = solvew_sens(u, wmm(m), free_pars_true, N)
    @info "Starting forward solving..."
    # sols_for = get_sol_in_parallel(forward_solve, 1:2M)
    sols_for = get_sol_in_parallel(forward_solve, 1:M)
    @info "Finished forward solving!"

    # --------------------------------------------------------------------------
    # ------------- Cost gradient using forward sensitivty ---------------------
    # --------------------------------------------------------------------------

    my_np = 1   # Only identifying one parameter, namely k
    Ys = zeros(N+1, length(sols_for))
    senss = [zeros(N+1, my_np) for j=1:length(sols_for)]
    for_dif_ests = zeros(M)
    for (i, sol) = enumerate(sols_for)
        Yi, sensi = apply_two_outputfun_mvar(f, f_sens, sol)
        Ys[:,i] = Yi
        senss[i] = sensi
    end

    for_dif_est = first(get_cost_gradient(Y[:,1], Ys[:,1:M], senss[1:M], N_trans))
    # for_dif_est = first(get_cost_gradient(Y[:,1], Ys[:,1:M], senss[M+1:2M], N_trans))

    for_dif_est2 = 0.0
    for m=1:M
        for_dif_est2 += first(get_cost_gradient(Y[:,1], Ys[:,m:m], senss[m:m], N_trans))
        # for_dif_est2 += first(get_cost_gradient(Y[:,1], Ys[:,m:m], senss[M+m:M+m], N_trans))
    end
    for_dif_est2 = for_dif_est2/M;

    # --------------------------------------------------------------------------
    # ------------ Numerical approximation of cost gradient --------------------
    # --------------------------------------------------------------------------

    # my_δ = 0.001 # Only for m-parameter
    my_δ = 0.01
    forward_solve2(m) = solvew(u, wmm(m), free_pars_true.+my_δ, N) |> h # NOTE: Not solving for sensitivities here
    # Ym2 = solve_in_parallel(forward_solve2, collect(1:2M))
    Ym2 = solve_in_parallel(forward_solve2, collect(1:M))
    cost  = get_cost_value(Y[:,1], Ys, N_trans)
    cost2 = get_cost_value(Y[:,1], Ym2, N_trans)
    num_est = (cost2-cost)/my_δ

    num_est2 = 0.0
    for m = 1:M
        cost  = get_cost_value(Y[:,1], Ys[:,m:m], N_trans)
        cost2 = get_cost_value(Y[:,1], Ym2[:,m:m], N_trans)
        num_est2 += (cost2-cost)/my_δ
    end
    num_est2 = num_est2/M

    ######################
    ########################
    # ACTUALLY USE NTRANS!!!!!!!!!!
    # AND FIX STOCHSATIC TOOO!!!

    ##################################################################################
    # NOTE NOTE: STOCHASTIC CODE TALKS ABOUT SOME MANUAL WAY, IS THAT OF ANY RELEVANCE HERE??
    ##################################################################################


    # --------------------------------------------------------------------------
    # ------------- Solution of adjoint system (backwards) ---------------------
    # --------------------------------------------------------------------------
    # Computing dx
    function get_der_est(sol)
        der_est = (sol.u[2:end]-sol.u[1:end-1])/Ts
        # ts = sol.t[1:end-1]
        ts = 0:Ts:(N-1)*Ts
        return ts, der_est
    end
    function get_der_est(vals::Matrix{Float64})
        # Rows of matrix are assumed to be different values of t, columns of
        # matrix are assumed to be different elements of the vector-valued process
        der_est = (vals[2:end,:]-vals[1:end-1,:])/Ts
        ts = 0:Ts:(N-1)*Ts
        return ts, der_est
        # Returns range of times, and matrix der_est with same structure as vals, just one row fewer
    end
    function get_mvar_cubic(ts, der_est::Vector{Vector{Float64}})
        temp = [cubic_spline_interpolation(ts, [der_est[i][j] for i=1:length(der_est)], extrapolation_bc=Line()) for j=1:length(der_est[1])]
        t -> [temp[i](t) for i=1:length(temp)]
        # return [cubic_spline_interpolation(ts, [der_est[i][j] for i=1:length(der_est)]) for j=1:length(der_est[1])]
    end
    function get_mvar_cubic(ts, der_est::AbstractMatrix{Float64})
        # Rows of der_est are assumed to be different values of t, columns of
        # matrix are assumed to be different elements of the vector-valued process
        temp = [cubic_spline_interpolation(ts, der_est[:,i], extrapolation_bc=Line()) for i=1:size(der_est,2)]
        # temp = [t->t for i=1:size(der_est,2)]
        return t -> [temp[i](t) for i=eachindex(temp)]
        # Returns a function mapping time to the vector-value of the function at that time
    end

    y_func  = interpolated_signal(Y[:,1], 0:Ts:(size(Y,1)-1)*Ts)
    dY_est  = (Y[2:end,1]-Y[1:end-1,1])/Ts
    dy_func = interpolated_signal(dY_est, 0:Ts:(size(dY_est,1)-1)*Ts)
    # ts, der_est = get_der_est(sol_for)
    # dx = get_mvar_cubic(ts, der_est)

    # Computing xp0, initial conditions of derivative of x wrt to p
    mdl = model_to_use(φ0, u, wmm(1), p)
    mdl_sens = model_sens_to_use(φ0, u, wmm(1), p)
    n_mdl = length(mdl.x0)
    # In case model_sens_to_use computes multuple sensitivities, we only pick the first one
    xp0 = reshape(mdl_sens.x0[n_mdl+1:end], n_mdl, :)[:,1:1]

    if N_trans > 0
        @warn "There really doesn't seem to be a good reason to use N_trans > 0 with adjoint, it just doesn't work"
    end

    function calc_adj_m(m::Int)
        ts, der_est = get_der_est(sols_for[m])
        dx = get_mvar_cubic(ts, der_est)
        # _, der_est2 = get_der_est(sols_for[M+m])
        # dx2 = get_mvar_cubic(ts, der_est2)
        der_est2 = der_est
        dx2 = dx
        mdl_adj, get_Gp = model_adj_to_use(u, wmm(m), p, N*Ts, sols_for[m], sols_for[m], y_func, dy_func, xp0, dx, dx2)
        # NOTE: Not using version with N_trans, it doesn't seem to do any good anyway
        # mdl_adj, get_Gp = model_adj_to_use(u, wmm(m), p, N*Ts, sols_for[m], sols_for[m], y_func, dy_func, xp0, dx, dx2, N_trans)
        adj_prob = problem(mdl_adj, N, Ts)
        adj_sol = solve(adj_prob, saveat = 0:Ts:N*Ts, abstol = abstol, reltol = reltol,
            maxiters = maxiters)
        return get_Gp(adj_sol)
    end

    adj_ms = collect(1:M)
    Gps    = solve_adj_in_parallel(calc_adj_m, adj_ms)
    Gp     = mean(Gps)

    # # num_est and for_dif_est estimate cost function gradient in a different
    # # way and will therefore not match Gp
    # return num_est, for_dif_est, Gp, Gps, num_est2, for_dif_est2
    return num_est2, for_dif_est2, Gp
end

# Ultimate deterministic adjoint debugging function
function debug_adjoint_semistochastic_mvar(expid::String, N_trans::Int=0)
    Random.seed!(123)
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

    get_all_parameters(free_pars::Array{Float64, 1}) = vcat(get_all_θs(free_pars), exp_data.get_all_ηs(free_pars))
    p = get_all_parameters(free_dyn_pars_true) # All true parameters
    θ = p[1:dθ]                                # All true dynamical parameters
    η = p[dθ+1: dθ+dη]                         # All true disturbance parameters
    free_pars_true = vcat(free_dyn_pars_true, W_meta.η[dist_par_inds])  # All true free parameters

    # Zm = [randn(Nw, n_tot) for m = 1:2M]
    Zm = [randn(Nw, n_tot) for m = 1:M]

    # --------------------------------------------------------------------------
    # --------------- Forward solution of nominal system -----------------------
    # --------------------------------------------------------------------------
    dmdl_true = discretize_ct_noise_model(get_ct_disturbance_model(η, nx, n_out), δ)
    # NOTE: OPTION 1: Use the rows below here for linear interpolation
    XWm = simulate_noise_process_mangled(dmdl_true, Zm)
    wmm(m::Int) = mk_noise_interp(dmdl_true.Cd, XWm, m, δ)

    forward_solve(m) = solvew_sens(u, wmm(m), free_pars_true, N)
    @info "Starting forward solving..."
    # sols_for = get_sol_in_parallel(forward_solve, 1:2M)
    sols_for = get_sol_in_parallel(forward_solve, 1:M)
    @info "Finished forward solving!"

    # --------------------------------------------------------------------------
    # ------------- Cost gradient using forward sensitivty ---------------------
    # --------------------------------------------------------------------------

    my_np = 1   # Only identifying one parameter, namely k
    Ys = zeros(N+1, length(sols_for))
    senss = [zeros(N+1, my_np) for j=1:length(sols_for)]
    for_dif_ests = zeros(M)
    for (i, sol) = enumerate(sols_for)
        Yi, sensi = apply_two_outputfun_mvar(f, f_sens, sol)
        Ys[:,i] = Yi
        senss[i] = sensi
    end

    for_dif_est = get_cost_gradient(Y[:,1], Ys[:,1:M], senss[1:M], N_trans)
    # for_dif_est = first(get_cost_gradient(Y[:,1], Ys[:,1:M], senss[M+1:2M], N_trans))

    for_dif_est2 = zeros(length(free_dyn_pars_true))
    for m=1:M
        for_dif_est2 += get_cost_gradient(Y[:,1], Ys[:,m:m], senss[m:m], N_trans)
        # for_dif_est2 += first(get_cost_gradient(Y[:,1], Ys[:,m:m], senss[M+m:M+m], N_trans))
    end
    for_dif_est2 = for_dif_est2./M;

    # --------------------------------------------------------------------------
    # ------------ Numerical approximation of cost gradient --------------------
    # --------------------------------------------------------------------------

    # NOTE: This doesn't compute the entire gradient! Just in one direction, same in every dimension
    my_δ = 0.01
    pars2 = free_pars_true+(0.1my_δ)*[1.,0.,0.]
    pars3 = free_pars_true+my_δ*[0.,1.,0.]
    pars4 = free_pars_true+my_δ*[0.,0.,1.]
    forward_solve2(m) = solvew(u, wmm(m), pars2, N) |> h
    forward_solve3(m) = solvew(u, wmm(m), pars3, N) |> h
    forward_solve4(m) = solvew(u, wmm(m), pars4, N) |> h
    # Ym2 = solve_in_parallel(forward_solve2, collect(1:2M))
    Ym2 = solve_in_parallel(forward_solve2, collect(1:M))
    Ym3 = solve_in_parallel(forward_solve3, collect(1:M))
    Ym4 = solve_in_parallel(forward_solve4, collect(1:M))
    cost  = get_cost_value(Y[:,1], Ys, N_trans)
    cost2 = get_cost_value(Y[:,1], Ym2, N_trans)
    cost3 = get_cost_value(Y[:,1], Ym3, N_trans)
    cost4 = get_cost_value(Y[:,1], Ym4, N_trans)
    num_est = [(cost2-cost)/(0.1my_δ), (cost3-cost)/my_δ, (cost4-cost)/my_δ]

    num_est2 = zeros(length(free_dyn_pars_true))
    for m = 1:M
        cost  = get_cost_value(Y[:,1], Ys[:,m:m], N_trans)
        cost2 = get_cost_value(Y[:,1], Ym2[:,m:m], N_trans)
        cost3 = get_cost_value(Y[:,1], Ym3[:,m:m], N_trans)
        cost4 = get_cost_value(Y[:,1], Ym4[:,m:m], N_trans)
        num_est2 += [(cost2-cost)/(0.1my_δ), (cost3-cost)/my_δ, (cost4-cost)/my_δ]
    end
    num_est2 = num_est2./M

    ##################################################################################
    # NOTE NOTE: STOCHASTIC CODE TALKS ABOUT SOME MANUAL WAY, IS THAT OF ANY RELEVANCE HERE??
    ##################################################################################


    # --------------------------------------------------------------------------
    # ------------- Solution of adjoint system (backwards) ---------------------
    # --------------------------------------------------------------------------
    # Computing dx
    function get_der_est(sol)
        der_est = (sol.u[2:end]-sol.u[1:end-1])/Ts
        # ts = sol.t[1:end-1]
        ts = 0:Ts:(N-1)*Ts
        return ts, der_est
    end
    function get_der_est(vals::Matrix{Float64})
        # Rows of matrix are assumed to be different values of t, columns of
        # matrix are assumed to be different elements of the vector-valued process
        der_est = (vals[2:end,:]-vals[1:end-1,:])/Ts
        ts = 0:Ts:(N-1)*Ts
        return ts, der_est
        # Returns range of times, and matrix der_est with same structure as vals, just one row fewer
    end
    function get_mvar_cubic(ts, der_est::Vector{Vector{Float64}})
        temp = [cubic_spline_interpolation(ts, [der_est[i][j] for i=1:length(der_est)], extrapolation_bc=Line()) for j=1:length(der_est[1])]
        t -> [temp[i](t) for i=1:length(temp)]
        # return [cubic_spline_interpolation(ts, [der_est[i][j] for i=1:length(der_est)]) for j=1:length(der_est[1])]
    end
    function get_mvar_cubic(ts, der_est::AbstractMatrix{Float64})
        # Rows of der_est are assumed to be different values of t, columns of
        # matrix are assumed to be different elements of the vector-valued process
        temp = [cubic_spline_interpolation(ts, der_est[:,i], extrapolation_bc=Line()) for i=1:size(der_est,2)]
        # temp = [t->t for i=1:size(der_est,2)]
        return t -> [temp[i](t) for i=eachindex(temp)]
        # Returns a function mapping time to the vector-value of the function at that time
    end

    y_func  = interpolated_signal(Y[:,1], 0:Ts:(size(Y,1)-1)*Ts)
    dY_est  = (Y[2:end,1]-Y[1:end-1,1])/Ts
    dy_func = interpolated_signal(dY_est, 0:Ts:(size(dY_est,1)-1)*Ts)
    # ts, der_est = get_der_est(sol_for)
    # dx = get_mvar_cubic(ts, der_est)

    # Computing xp0, initial conditions of derivative of x wrt to p
    mdl = model_to_use(φ0, u, wmm(1), p)
    mdl_sens = model_sens_to_use(φ0, u, wmm(1), p)
    n_mdl = length(mdl.x0)
    xp0 = reshape(mdl_sens.x0[n_mdl+1:end], n_mdl, :)

    if N_trans > 0
        @warn "There really doesn't seem to be a good reason to use N_trans > 0 with adjoint, it just doesn't work"
    end

    function calc_adj_m(m::Int)
        ts, der_est = get_der_est(sols_for[m])
        dx = get_mvar_cubic(ts, der_est)
        # _, der_est2 = get_der_est(sols_for[M+m])
        # dx2 = get_mvar_cubic(ts, der_est2)
        der_est2 = der_est
        dx2 = dx
        mdl_adj, get_Gp = model_adj_to_use(u, wmm(m), p, N*Ts, sols_for[m], sols_for[m], y_func, dy_func, xp0, dx, dx2)
        # NOTE: Not using version with N_trans, it doesn't seem to do any good anyway
        # mdl_adj, get_Gp = model_adj_to_use(u, wmm(m), p, N*Ts, sols_for[m], sols_for[m], y_func, dy_func, xp0, dx, dx2, N_trans)
        adj_prob = problem(mdl_adj, N, Ts)
        adj_sol = solve(adj_prob, saveat = 0:Ts:N*Ts, abstol = abstol, reltol = reltol,
            maxiters = maxiters)
        return get_Gp(adj_sol)
    end

    adj_ms = collect(1:M)
    Gps    = solve_adj_in_parallel(calc_adj_m, adj_ms)
    Gp     = mean(Gps, dims=2)

    # # num_est and for_dif_est estimate cost function gradient in a different
    # # way and will therefore not match Gp
    # return num_est, for_dif_est, Gp, Gps, num_est2, for_dif_est2
    return num_est2, for_dif_est2, Gp
end

function debug_adjoint_stepbystep(expid::String, δp::Float64=0.01, N_trans::Int64=0)
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

    get_all_parameters(free_pars::Array{Float64, 1}) = vcat(get_all_θs(free_pars), exp_data.get_all_ηs(free_pars))
    p = get_all_parameters(free_dyn_pars_true)
    θ = p[1:dθ]
    η = p[dθ+1: dθ+dη]
    # C = reshape(η[nx+1:end], (n_out, n_tot))

    Zm = [randn(Nw, n_tot) for m = 1:M]

    # ####### Obtaining cost derivative using forward sensitivity analysis #######
    # dmdl = discretize_ct_noise_model(get_ct_disturbance_model(η, nx, n_out), δ)
    # XWm = simulate_noise_process_mangled(dmdl, Zm)
    # wmm(m::Int) = mk_noise_interp(dmdl.Cd, XWm, m, δ)

    # sol = solvew_sens(exp_data.u, wmm(1), free_dyn_pars_true, N)
    Gp, λs, βs, sol_for, wmm, Gp_alt, adj_term = solve_adjoint_deterministic("5k_u2w6_from_Alsvin", 0)
    Y1, sens1 = h_comp(sol_for)
    for_dif_est = get_cost_gradient(Y[1:N+1, 1], reshape(Y1, length(Y1), 1), [sens1], N_trans)
    @info "Gp: $Gp, Gp_alt: $Gp_alt, term: $adj_term"

    ######################### Adjoint part #####################################

    y_func  = interpolated_signal(Y[:,1], 0:Ts:(size(Y,1)-1)*Ts)
    dY_est  = (Y[2:end,1]-Y[1:end-1,1])/Ts
    dy_func = interpolated_signal(dY_est, 0:Ts:(size(dY_est,1)-1)*Ts)

    # ------------- Trying to get hold of our desired λ ------------------------

    # # @warn "Using lambdas2 instead of lambdas"
    # if isfile("data/results/adjoint_test/lambdas.csv")
    #     λs = readdlm("data/results/adjoint_test/lambdas.csv", ',')
    #     @info "Loaded λs from file"
    # else
    #     @info "Generating new λs"
    #     # TODO: Surely we can have a better function to call here? Or not?
    #     _, _, Gp, _, λs, _, _ = debug_adj_and_forward(expid, 0.01, N_trans)
    #     writedlm("data/results/adjoint_test/lambdas.csv", λs, ',')
    #     @info "Finished λs generation"
    # end
    # # λs need to be reversed, since they're solved backwards
    # λs = λs[end:-1:1, :]

    function get_der_est(vals::Matrix{Float64})
        # Rows of matrix are assumed to be different values of t, columns of
        # matrix are assumed to be different elements of the vector-valued process
        der_est = (vals[2:end,:]-vals[1:end-1,:])/Ts
        ts = 0:Ts:(N-1)*Ts
        return ts, der_est
        # Returns range of times, and matrix der_est with same structure as vals, just one row fewer
    end
    function get_mvar_cubic(ts, der_est::Matrix{Float64})
        # Rows of der_est are assumed to be different values of t, columns of
        # matrix are assumed to be different elements of the vector-valued process
        temp = [cubic_spline_interpolation(ts, der_est[:,i], extrapolation_bc=Line()) for i=1:size(der_est,2)]
        # temp = [t->t for i=1:size(der_est,2)]
        return t -> [temp[i](t) for i=eachindex(temp)]
        # Returns a function mapping time to the vector-value of the function at that time
    end

    ts, λ_der = get_der_est(λs)
    λ_func  = get_mvar_cubic(0:Ts:N*Ts, λs)
    dλ_func = get_mvar_cubic(ts, λ_der)
    # λ_func  = t -> [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0]
    # dλ_func = t -> [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]

    # # ---------------- For step 3 -------------------
    # dmdl_adj = pendulum_adj_step3(φ0, exp_data.u, wmm(1), p, y_func, λ_func, dλ_func)
    # prob_adj = problem(dmdl_adj, N, Ts)
    # sol_adj  = solve(prob_adj, saveat = 0:Ts:N*Ts, abstol = abstol, reltol = reltol,
    #     maxiters = maxiters)
    # Y_adj, sens_and_cost_adj = h_int_2(sol_adj)  #h_int_2 is the same as h_int_3 would be
    # adj_cost_sens = sens_and_cost_adj[:,5]/(N*Ts)

    # ----------------- For step 4 --------------------
    dmdl_adj = model_stepbystep(φ0, exp_data.u, wmm(1), p, y_func, λ_func, dλ_func, N*Ts)
    prob_adj = problem(dmdl_adj, N, Ts)
    sol_adj  = solve(prob_adj, saveat = 0:Ts:N*Ts, abstol = abstol, reltol = reltol,
        maxiters = maxiters)
    Y_adj, sens_and_cost_adj = h_int(sol_adj)
    int_sens      = sens_and_cost_adj[:,3]/(N*Ts)   # col 3  -> x[16] (h_int) (point 2)
    adj_sens_1    = sens_and_cost_adj[:,4]/(N*Ts)   # col 4  -> x[17] (h_int) (point 3)
    adj_sens_2_miss = sens_and_cost_adj[:,5]        # col 5  -> x[18] (h_int) # Must have additional term added to it (point 4)
    adj_sens_miss = sens_and_cost_adj[:,15]         # col 15 -> x[19] (h_int) # Must have additional term added to it (point 5)
    remainder     = sens_and_cost_adj[:,16]         # col 16 -> x[20] (h_int) (bonus point)
    adj_sens_extra_miss = sens_and_cost_adj[:,17]   # col 17 -> x[21] (h_int) (point 3.5)
    adj_sens_6_miss = sens_and_cost_adj[:,18]       # col 18 -> x[22] (h_int) (point 6)

    # NOTE: PERHAPS TOO MANY DIVISIONS BY T????

    sens1_adj = sens_and_cost_adj[:,1:1]
    int_cost_adj = sens_and_cost_adj[:,2]/(N*Ts)
    cost_sens_adj = sens_and_cost_adj[:,3]/(N*Ts)
    xs_adj = sens_and_cost_adj[:, 6:7]
    xθs_adj = sens_and_cost_adj[:, 8:14]
    for_dif_est_adj = get_cost_gradient(Y[1:N+1, 1], reshape(Y_adj, length(Y_adj), 1), [sens1_adj], N_trans)

    Fdx = (x1, x2) -> vcat([1   0   0          0   0   2x1    0
                            0   1   0          0   0   2x2    0
                            0   0   -x1      0.3   0   0      0
                            0   0   -x2      0   0.3   0      0], zeros(3,7))
    term = (λ_func(N*Ts)')*Fdx(xs_adj[end,1], xs_adj[end,2])*xθs_adj[end,:] - (λ_func(0.)')*Fdx(xs_adj[1,1], xs_adj[1,2])*xθs_adj[1,:]
    @info "This term: $term"
    println("Let's print some stuff:")
    println(λ_func(N*Ts))
    println(Fdx(xs_adj[end,1], xs_adj[end,2]))
    println(xθs_adj[end,:])
    println(λ_func(0.))
    println(Fdx(xs_adj[1,1], xs_adj[1,2]))
    println(xθs_adj[1,:])
    println("Finished printing some stuff")

    return for_dif_est[1], int_sens[end], adj_sens_1[end], adj_sens_2_miss[end]-term, adj_sens_miss[end]-term, Gp, adj_sens_extra_miss[end]-term, adj_sens_6_miss[end]-term
    # return for_dif_est[1], adj_sens_miss[end]-term, remainder
    # # return num_dif_est, for_dif_est[1], adj_sens_2[end]-term/(N*Ts), adj_sens_3[end]-term3/(N*Ts)
    # # # return num_dif_est, for_dif_est[1], for_dif_est2[1], cost_sens[end], adj_sens_1[end], adj_sens_2[end]-term/(N*Ts), adj_sens_3[end]-term3/(N*Ts)
end

# Ultimate benchmarkin problem generating function
function get_benchmarking_problems(expid::String, N_trans::Int=0)
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

    get_all_parameters(free_pars::Array{Float64, 1}) = vcat(get_all_θs(free_pars), exp_data.get_all_ηs(free_pars))
    p = get_all_parameters(free_dyn_pars_true) # All true parameters
    θ = p[1:dθ]                                # All true dynamical parameters
    η = p[dθ+1: dθ+dη]                         # All true disturbance parameters

    Zm = [randn(Nw, n_tot) for m = 1:M]

    # --------------------------------------------------------------------------
    # --------------- Forward solution of nominal system -----------------------
    # --------------------------------------------------------------------------
    dmdl_true = discretize_ct_noise_model(get_ct_disturbance_model(η, nx, n_out), δ)
    # NOTE: OPTION 1: Use the rows below here for linear interpolation
    XWm = simulate_noise_process_mangled(dmdl_true, Zm)
    wmm(m::Int) = mk_noise_interp(dmdl_true.Cd, XWm, m, δ)
    sol_for = solvew_sens(u, wmm(1), free_dyn_pars_true, N)

    for_prob = problem(
        model_sens_to_use(φ0, u, wmm(1), p), N, Ts)

    # --------------------------------------------------------------------------
    # ------------- Solution of adjoint system (backwards) ---------------------
    # --------------------------------------------------------------------------
    # Computing dx
    function get_der_est(sol)
        der_est = (sol.u[2:end]-sol.u[1:end-1])/Ts
        # ts = sol.t[1:end-1]
        ts = 0:Ts:(N-1)*Ts
        return ts, der_est
    end
    function get_der_est(vals::Matrix{Float64})
        # Rows of matrix are assumed to be different values of t, columns of
        # matrix are assumed to be different elements of the vector-valued process
        der_est = (vals[2:end,:]-vals[1:end-1,:])/Ts
        ts = 0:Ts:(N-1)*Ts
        return ts, der_est
        # Returns range of times, and matrix der_est with same structure as vals, just one row fewer
    end
    function get_mvar_cubic(ts, der_est::Vector{Vector{Float64}})
        temp = [cubic_spline_interpolation(ts, [der_est[i][j] for i=1:length(der_est)], extrapolation_bc=Line()) for j=1:length(der_est[1])]
        t -> [temp[i](t) for i=1:length(temp)]
        # return [cubic_spline_interpolation(ts, [der_est[i][j] for i=1:length(der_est)]) for j=1:length(der_est[1])]
    end
    function get_mvar_cubic(ts, der_est::AbstractMatrix{Float64})
        # Rows of der_est are assumed to be different values of t, columns of
        # matrix are assumed to be different elements of the vector-valued process
        temp = [cubic_spline_interpolation(ts, der_est[:,i], extrapolation_bc=Line()) for i=1:size(der_est,2)]
        # temp = [t->t for i=1:size(der_est,2)]
        return t -> [temp[i](t) for i=eachindex(temp)]
        # Returns a function mapping time to the vector-value of the function at that time
    end

    y_func  = interpolated_signal(Y[:,1], 0:Ts:(size(Y,1)-1)*Ts)
    dY_est  = (Y[2:end,1]-Y[1:end-1,1])/Ts
    dy_func = interpolated_signal(dY_est, 0:Ts:(size(dY_est,1)-1)*Ts)
    ts, der_est = get_der_est(sol_for)
    dx = get_mvar_cubic(ts, der_est)

    # Computing xp0, initial conditions of derivative of x wrt to p
    mdl = model_to_use(φ0, u, wmm(1), p)
    mdl_sens = model_sens_to_use(φ0, u, wmm(1), p)
    n_mdl = length(mdl.x0)
    xp0 = reshape(mdl_sens.x0[n_mdl+1:end], n_mdl, :)

    mdl_adj, get_Gp = my_pendulum_adjoint_konly(u, wmm(1), p, N*Ts, sol_for, sol_for, y_func, dy_func, xp0, dx, dx)
    adj_prob = problem(mdl_adj, N, Ts)

    trange = 0:Ts:N*Ts

    #= To compare benchmarks, run
    @benchmark for_sol = solve(for_prob, saveat = trange, abstol = abstol, reltol = reltol, maxiters = 1000)
    and
    @benchmark adj_sol = solve(adj_prob, saveat = trange, abstol = abstol, reltol = reltol, maxiters = 1000)
    =#

    return for_prob, adj_prob, trange
end

function interpolated_signal(out, times)
    @assert (length(out) == length(times)) "out and times signals must have the same length (currently $(length(out)) vs $(length(times)))"
    function func_to_return(t::Float64)::Float64
        if t <= minimum(times)
            return out[1]
        elseif t >= maximum(times)
            return out[end]
        else
            ind = 1
            while !(times[ind] < t && times[ind+1] >= t)
                ind += 1
            end
            return out[ind] + (t-times[ind])*(out[ind+1]-out[ind])/(times[ind+1]-times[ind])
        end
    end

    return func_to_return
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

    hline!(p, free_dyn_pars_true, label = L"\theta_0", linestyle = :dot, linecolor = :gray)
end

function simulate_experiment(expid::String, pars::Array{Float64, 1}, Zm::Array{Array{Float64,2},1})
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

    Ym = simulate_system(exp_data, pars, M, dist_par_inds, isws, Zm)
    return Ym, Zm
end

function simulate_experiment(expid::String, pars::Array{Float64, 1})
    exp_data, isws = get_experiment_data(expid)
    W_meta = exp_data.W_meta
    Nw = exp_data.Nw
    nx = W_meta.nx
    n_in = W_meta.n_in
    n_out = W_meta.n_out
    n_tot = nx*n_in

    Zm = [randn(Nw, n_tot) for m=1:M]
    simulate_experiment(expid, pars, Zm)
end

function simulate_experiment_debug(expid::String, pars::Array{Float64, 1})
    input  = readdlm(joinpath(data_dir, expid*"/U.csv"), ',')
    u(t::Float64) = interpw(input, 1, 1)(t)
    # Ym = solvew(u, t -> 0.0, pars, 4999 ) |> h

    # function u_debug(t::Float64)
    #     if t <= -0.1
    #         return u(t)
    #     else
    #         return 0.0
    #     end
    # end

    Ym_temp = solvew(u, t -> 0.0, pars, 4999 ) |> h_debug
    Ym = zeros(length(Ym_temp), length(Ym_temp[1]))
    for i = 1:length(Ym_temp)
        Ym[i,:] = Ym_temp[i]
    end

    return Ym
end

function read_from_backup(dir::String, E::Int)
    sample = readdlm(joinpath(dir, "backup_baseline_e1.csv"), ',')
    # sample = readdlm(joinpath(dir, "backup_proposed_e1.csv"), ',')
    k = length(sample)
    opt_pars_baseline = zeros(k, E)
    opt_pars_proposed = zeros(k, E)
    avg_pars_proposed = zeros(k, E)
    for e=1:E
        opt_pars_baseline[:,e] = readdlm(joinpath(dir, "backup_baseline_e$e.csv"), ',')
        # opt_pars_proposed[:,e] = readdlm(joinpath(dir, "backup_proposed_e$e.csv"), ',')
        # avg_pars_proposed[:,e] = readdlm(joinpath(dir, "backup_average_e$e.csv"), ',')
    end
    return opt_pars_baseline, opt_pars_proposed, avg_pars_proposed
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
    E = 100
    @warn "Using E = $E instead of default"
    opt_pars_baseline = zeros(length(pars0), E)
    trace_base = [[pars0] for e=1:E]
    @warn "Not running baseline parameter estimation"
    # for e=1:E
    # # for e=4:4
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
    #
    # @info "The mean optimal parameters for baseline are given by: $(mean(opt_pars_baseline, dims=2))"

    # === Computing cost function of baseline model
    base_par_vals = 5.3:0.1:7.8
    par_vector = [[el] for el in base_par_vals]
    Ybs = [zeros(N+1) for j = 1:length(par_vector)]
    base_cost_vals = zeros(length(base_par_vals), E)
    @warn "Not plotting baseline cost function"
    # for (j, pars) in enumerate(par_vector)
    #     Ybs[j] = solvew(exp_data.u, t -> zeros(n_out), pars, N) |> h
    # end
    # for ind = 1:length(base_par_vals)
    #     # Y has columns indexed with 1:E because in case E is changed for debugging purposes,
    #     # without the dimensions of Y changing, we can get a mismatch otherwise
    #     base_cost_vals[ind,:] = mean((Y[N_trans+1:end, 1:E].-Ybs[ind][N_trans+1:end]).^2, dims=1)
    # end

    # === Computing cost function for proposed model ===
    # TODO: Should we consider storing and loading white noise, to improve repeatability
    # of the results? Currently repeatability is based on fixed random seed
    Zm = [randn(Nw, n_tot) for m = 1:M]

    # NOTE: CURRENTLY ONLY TREATS SCALAR PARAMETERS
    @info "Plotting proposed cost function..."
    # vals1 = 1.0:0.25:8.0
    # vals2 = 9.0:2.0:35.0
    prop_par_vals = 7.5:0.1:7.5    # DEBUG
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
        # C = reshape(η[nx+1:end], (n_out, n_tot))  # Not correct when disturbance model is parametrized, use dmdl.Cd instead

        dmdl = discretize_ct_noise_model(get_ct_disturbance_model(η, nx, n_out), δ) # TODO: Use new discretizations
        # # NOTE: OPTION 1: Use the rows below here for linear interpolation
        XWm = simulate_noise_process_mangled(dmdl, Zm)
        wmm(m::Int) = mk_noise_interp(dmdl.Cd, XWm, m, δ)
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
        return get_cost_gradient(y, Ym[:,1:M_mean], jacsYm[M_mean+1:end], N_trans)
    end

    # Returns estimate of gradient of cost function
    # M_mean specified over how many realizations the gradient estimate is computed
    function get_gradient_estimate_debug(y, δ, pars, isws, M_mean::Int=1)
        Ym, jacsYm = simulate_system_sens(exp_data, pars, 2M_mean, dist_sens_inds, isws)
        # Uses different noise realizations for estimate of output and estiamte of gradient
        grad_est = get_cost_gradient(y, Ym[:,1:M_mean], jacsYm[M_mean+1:end], N_trans)
        cost = mean((y[N_trans+1:end]-Ym[N_trans+1:end,1]).^2)
        # grad_est will be 2D array with one dimenion equal to 1, we want to return a 1D array
        return grad_est, cost
    end

    # Returns estimate of gradient of output
    function get_proposed_jacobian(pars, isws, M_mean::Int=1)
        jacsYm = simulate_system_sens(exp_data, pars, M_mean, dist_sens_inds, isws)[2]
        return mean(jacsYm)[N_trans+1:end, :]
    end

    @info "Finding proposed minimum..."

    avg_pars_proposed = zeros(length(pars0), E)
    opt_pars_proposed = zeros(length(pars0), E)
    opt_pars_proposed_LSQ = zeros(length(pars0), E)
    trace_proposed = [[Float64[]] for e=1:E]
    trace_costs = [Float64[] for e=1:E]
    grad_trace  = [[Float64[]] for e=1:E]

    @warn "Not running proposed identification at the moment"
    for e=[]#1:E
    # for e=4:4
        get_gradient_estimate_p(pars, M_mean) = get_gradient_estimate(Y[:,e], δ, pars, isws, M_mean)
        get_gradient_estimate_p_debug(pars, M_mean) = get_gradient_estimate_debug(Y[:,e], δ, pars, isws, M_mean)
        jacobian_model(x, p) = get_proposed_jacobian(p, isws, M)

        opt_pars_proposed[:,e], trace_proposed[e], trace_costs[e], grad_trace[e] =
            perform_SGD_adam_debug(get_gradient_estimate_p_debug, pars0, par_bounds, verbose=true; maxiters=100, tol=1e-8)
        avg_pars_proposed[:,e] = mean(trace_proposed[e][end-80:end])
        reset_isws!(isws)
        # proposed_result, proposed_trace = get_fit_sens(Y[N_trans+1:end,e], pars0,
        #     (dummy_input, pars) -> proposed_model_parametrized(δ, Zm, dummy_input, pars, isws),
        #     jacobian_model, par_bounds[:,1], par_bounds[:,2])

        # proposed_result, proposed_trace = get_fit_sens(Y[:,e], pars0,
        #     (dummy_input, pars) -> proposed_model_parametrized(δ, Zm, dummy_input, pars, isws), jacobian_model)
        # opt_pars_proposed[:, e] = coef(proposed_result)
        # opt_pars_proposed_LSQ[:, e] = coef(proposed_result)

        println("Completed for dataset $e for parameters $(opt_pars_proposed[:,e])")
    end

    @info "The mean optimal parameters for proposed method are given by: $(mean(opt_pars_proposed, dims=2))"

    # trace_proposed is an array with E elements, where each element is an array
    # of parameters that the SGD has gone through, and where every parameter
    # is an array of Float64-values
    return opt_pars_baseline, opt_pars_proposed, avg_pars_proposed, trace_base, trace_proposed, base_cost_vals, prop_cost_vals, trace_costs, base_par_vals, prop_par_vals, grad_trace
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
    base_par_vals1 = 0.25:0.025:0.45
    base_par_vals2 = 8.41:0.2:10.21

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
    # Yb_true = solvew(u, t -> zeros(n_out), free_dyn_pars_true, N) |> h
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
        # C = reshape(η[nx+1:end], (n_out, n_tot))  # Not correct when disturbance model is parametrized, use dmdl.Cd instead

        dmdl = discretize_ct_noise_model(get_ct_disturbance_model(η, nx, n_out), δ)
        # # NOTE: OPTION 1: Use the rows below here for linear interpolation
        XWm = simulate_noise_process_mangled(dmdl, Zm)
        wmm(m::Int) = mk_noise_interp(dmdl.Cd, XWm, m, δ)
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
        return get_cost_gradient(y, Ym[:,1:M_mean], jacsYm[M_mean+1:end], N_trans)
    end

    # Returns estimate of gradient of cost function
    # M_mean specified over how many realizations the gradient estimate is computed
    function get_gradient_estimate_debug(y, δ, pars, isws, M_mean::Int=1)
        Ym, jacsYm = simulate_system_sens(exp_data, pars, 2M_mean, dist_sens_inds, isws)
        # Uses different noise realizations for estimate of output and estiamte of gradient
        grad_est = get_cost_gradient(y, Ym[:,1:M_mean], jacsYm[M_mean+1:end], N_trans)
        cost = mean((y[N_trans+1:end]-Ym[N_trans+1:end,1]).^2)
        # grad_est will be 2D array with one dimenion equal to 1, we want to return a 1D array
        return grad_est, cost
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
    grad_trace  = [[Float64[]] for e=1:E]
    for e=1:E
        get_gradient_estimate_p(pars, M_mean) = get_gradient_estimate(Y[:,e], δ, pars, isws, M_mean)
        get_gradient_estimate_p_debug(pars, M_mean) = get_gradient_estimate_debug(Y[:,e], δ, pars, isws, M_mean)
        jacobian_model(x, p) = get_proposed_jacobian(p, isws, M)
        opt_pars_proposed[:,e], trace_proposed[e], trace_costs[e], grad_trace[e] =
            perform_SGD_adam_debug(get_gradient_estimate_p_debug, pars0, par_bounds, verbose=true; maxiters=150, tol=1e-8)
        reset_isws!(isws)
        # proposed_result, proposed_trace = get_fit_sens(Y[:,e], pars0,
        #     (dummy_input, pars) -> proposed_model_parametrized(δ, Zm, dummy_input, pars, isws),
        #     jacobian_model, par_bounds[:,1], par_bounds[:,2])
        # # proposed_result, proposed_trace = get_fit_sens(Y[:,e], pars0,
        # #     (dummy_input, pars) -> proposed_model_parametrized(δ, Zm, dummy_input, pars, isws), jacobian_model)
        # # opt_pars_proposed[:, e] = coef(proposed_result)
        # opt_pars_proposed_LSQ[:, e] = coef(proposed_result)

        println("Completed for dataset $e for parameters $(opt_pars_proposed[:,e])")
    end

    @info "The mean optimal parameters for proposed method are given by: $(mean(opt_pars_proposed, dims=2))"

    @info "Plotting proposed cost function..."
    # prop_par_diffs1 = -0.6:0.2:0.6
    # prop_par_diffs2 = -0.6:0.2:0.6
    # prop_par_vals1 = opt_pars_proposed[1,1] .+ prop_par_diffs1
    # prop_par_vals2 = opt_pars_proposed[2,1] .+ prop_par_diffs2
    # prop_par_vals1 = 0.1:0.2:opt_pars_proposed[1,1]+0.4   # For m_true = 0.3
    # prop_par_vals2 = (4.25-0.6):0.2:opt_pars_proposed[2,1]+0.4
    prop_par_vals1 = 0.2:0.05:0.6   # For m_true = 0.3
    # prop_par_vals2 = (4.25-0.6):0.2:(9.81+0.4)
    prop_par_vals2 = 6.0:0.05:6.40
    # prop_par_vals1 = 0.1:0.1:5.0
    # prop_par_vals2 = [4.25]
    prop_cost_vals = [zeros(length(prop_par_vals1), length(prop_par_vals2)) for e=1:E]
    prop_cost_true = zeros(E)
    for ind1 = 1:length(prop_par_vals1)
        for ind2 = 1:length(prop_par_vals2)
            pars = [prop_par_vals1[ind1], prop_par_vals2[ind2]]
            Ym = mean(simulate_system(exp_data, pars, M, dist_sens_inds, isws, Zm), dims=2)
            for e=1:E
                prop_cost_vals[e][ind1, ind2] = mean((Y[N_trans+1:end, e].-Ym[N_trans+1:end]).^2)
            end
        end
    end
    Ym_true = mean(simulate_system(exp_data, free_dyn_pars_true, M, dist_sens_inds, isws, Zm), dims=2)
    for e=1:E
        prop_cost_true[e] = mean((Y[N_trans+1:end, e].-Ym_true[N_trans+1:end]).^2)
    end


    # Yms, jacsYm = simulate_system_sens(exp_data, [1.796586589667771, 5.7244856197562175], 100, dist_sens_inds, isws)
    # my_grad_est = get_cost_gradient(Y[:,1], Yms[:,1:50], jacsYm[51:end], N_trans)

    #NOTE: To get interactive 3D-plot, do as follows:
    # > using Plots
    # > plotlyjs()
    # > plot(prop_par_vals[1][:,1], prop_par_vals[2][:,e], transpose(prop_cost_vals[e]), st=:surface, xlabel="m values", ylabel="k values", zlabel="cost", size=(1200,1000))
    # > plot(prop_par_vals[1][:,1], prop_par_vals[2][:,1], transpose(prop_cost_vals[1]), st=:surface, xlabel="m values", ylabel="g values", zlabel="cost", size=(1200,1000))
    # NOTE: Make sure x- and y-values are a 1D-vector, NOT a flat 2D-vector

    base_par_vals = (base_par_vals1, base_par_vals2)
    prop_par_vals = (prop_par_vals1, prop_par_vals2)
    # return Yms, jacsYm, my_grad_est
    return opt_pars_baseline, opt_pars_proposed, trace_base, trace_proposed, base_cost_vals, prop_cost_vals, base_par_vals, prop_par_vals, base_cost_true, prop_cost_true
end

function debug_minimization_full(expid::String, N_trans::Int = 0; pars0::Array{Float64,1}=[0.5, 4.25, 11.0, 4.25])
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
    E = 1
    Zm = [randn(Nw, n_tot) for m = 1:M]


    # === Optimizing parameters for the proposed model or stochastic gradient descent ==
    function proposed_model_parametrized(δ, Zm, dummy_input, pars, isws)
        # NOTE: The true input is encoded in the solvew_sens()-function, but this function
        # still needs to to take two input arguments, so dummy_input could just be
        # anything, it's not used anyway
        p = get_all_parameters(pars)
        θ = p[1:dθ]
        η = p[dθ+1: dθ+dη]
        # C = reshape(η[nx+1:end], (n_out, n_tot))  # Not correct when disturbance model is parametrized, use dmdl.Cd instead

        dmdl = discretize_ct_noise_model(get_ct_disturbance_model(η, nx, n_out), δ)
        # # NOTE: OPTION 1: Use the rows below here for linear interpolation
        XWm = simulate_noise_process_mangled(dmdl, Zm)
        wmm(m::Int) = mk_noise_interp(dmdl.C, XWm, m, δ)
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
        return get_cost_gradient(y, Ym[:,1:M_mean], jacsYm[M_mean+1:end], N_trans)
    end

    # Returns estimate of gradient of cost function
    # M_mean specified over how many realizations the gradient estimate is computed
    function get_gradient_estimate_debug(y, δ, pars, isws, M_mean::Int=1)
        Ym, jacsYm = simulate_system_sens(exp_data, pars, 2M_mean, dist_sens_inds, isws)
        # Uses different noise realizations for estimate of output and estiamte of gradient
        grad_est = get_cost_gradient(y, Ym[:,1:M_mean], jacsYm[M_mean+1:end], N_trans)
        cost = mean((y[N_trans+1:end]-Ym[N_trans+1:end,1]).^2)
        # grad_est will be 2D array with one dimenion equal to 1, we want to return a 1D array
        return grad_est, cost
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
    grad_trace  = [[Float64[]] for e=1:E]
    for e=1:E
        get_gradient_estimate_p(pars, M_mean) = get_gradient_estimate(Y[:,e], δ, pars, isws, M_mean)
        get_gradient_estimate_p_debug(pars, M_mean) = get_gradient_estimate_debug(Y[:,e], δ, pars, isws, M_mean)
        jacobian_model(x, p) = get_proposed_jacobian(p, isws, M)
        opt_pars_proposed[:,e], trace_proposed[e], trace_costs[e], grad_trace[e] =
            perform_SGD_adam_debug(get_gradient_estimate_p_debug, pars0, par_bounds, verbose=true; maxiters=500, tol=1e-8)
        reset_isws!(isws)
        # proposed_result, proposed_trace = get_fit_sens(Y[:,e], pars0,
        #     (dummy_input, pars) -> proposed_model_parametrized(δ, Zm, dummy_input, pars, isws),
        #     jacobian_model, par_bounds[:,1], par_bounds[:,2])
        # # proposed_result, proposed_trace = get_fit_sens(Y[:,e], pars0,
        # #     (dummy_input, pars) -> proposed_model_parametrized(δ, Zm, dummy_input, pars, isws), jacobian_model)
        # # opt_pars_proposed[:, e] = coef(proposed_result)
        # opt_pars_proposed_LSQ[:, e] = coef(proposed_result)

        println("Completed for dataset $e for parameters $(opt_pars_proposed[:,e])")
    end

    @info "The mean optimal parameters for proposed method are given by: $(mean(opt_pars_proposed, dims=2))"

    m_true = 0.3
    L_true = 6.25
    g_true = 9.81
    k_true = 6.25

    Δm = norm(m_true-opt_pars_proposed[1,1])
    ΔL = norm(L_true-opt_pars_proposed[2,1])
    Δg = norm(g_true-opt_pars_proposed[3,1])
    Δk = norm(k_true-opt_pars_proposed[4,1])
    δm = Δm/6.
    δL = ΔL/6.
    δg = Δg/6.
    δk = Δk/6.

    # Ensures 2401 possible combinations (7^4)
    mvals = min(m_true, opt_pars_proposed[1,1]):δm:max(m_true, opt_pars_proposed[1,1])+δm
    Lvals = min(L_true, opt_pars_proposed[2,1]):δL:max(L_true, opt_pars_proposed[2,1])+δL
    gvals = min(g_true, opt_pars_proposed[3,1]):δg:max(g_true, opt_pars_proposed[3,1])+δg
    kvals = min(k_true, opt_pars_proposed[4,1]):δk:max(k_true, opt_pars_proposed[4,1])+δk
    par_vecs = [collect(mvals), collect(Lvals), collect(gvals), collect(kvals)]

    # mvals = min(m_true, opt_pars_proposed[1,1]):0.01:max(m_true, opt_pars_proposed[1,1])+0.01
    # Lvals = min(L_true, opt_pars_proposed[2,1]):0.1:max(L_true, opt_pars_proposed[2,1])
    # gvals = min(g_true, opt_pars_proposed[3,1]):0.1:max(g_true, opt_pars_proposed[3,1])
    # kvals = min(k_true, opt_pars_proposed[4,1]):0.1:max(k_true, opt_pars_proposed[4,1])
    mdiffs = mvals.-opt_pars_proposed[1,1]
    Ldiffs = Lvals.-opt_pars_proposed[2,1]
    gdiffs = gvals.-opt_pars_proposed[3,1]
    kdiffs = kvals.-opt_pars_proposed[4,1]

    # mdiffs = [-0.03, -0.02, -0.01, 0.0, 0.01, 0.02, 0.03]
    # Ldiffs = [-0.3, -0.2, -0.1, 0.0, 0.1, 0.2, 0.3]
    # gdiffs = [-0.3, -0.2, -0.1, 0.0, 0.1, 0.2, 0.3]
    # kdiffs = [-0.3, -0.2, -0.1, 0.0, 0.1, 0.2, 0.3]

    cost_vals = [zeros(length(mdiffs)*length(Ldiffs)*length(gdiffs)*length(kdiffs)) for e=1:E]
    all_pars = zeros(4, length(mdiffs)*length(Ldiffs)*length(gdiffs)*length(kdiffs))
    min_ind = fill(-1, (E,))
    min_cost = fill(Inf, (E,))
    ind = 1
    for mdiff in mdiffs
        for Ldiff in Ldiffs
            for gdiff in gdiffs
                for kdiff in kdiffs
                    for e = 1:E
                        opt_pars = opt_pars_proposed[:,e]
                        pars = [opt_pars[1]+mdiff, opt_pars[2]+Ldiff, opt_pars[3]+gdiff, opt_pars[4]+kdiff]
                        all_pars[:,ind] = pars
                        Ym = mean(simulate_system(exp_data, pars, M, dist_sens_inds, isws, Zm), dims=2)
                        cost_vals[e][ind] = mean((Y[N_trans+1:end, e].-Ym[N_trans+1:end]).^2)
                        if cost_vals[e][ind] < min_cost[e]
                            min_ind[e] = ind
                            min_cost[e] = cost_vals[e][ind]
                        end
                    end
                    ind += 1
                end
            end
        end
    end

    return all_pars, cost_vals, par_vecs, min_ind, opt_pars_proposed, trace_proposed, grad_trace
end

function simplified_debug_minimization_full(expid::String, N_trans::Int = 0; pars0::Array{Float64,1}=[0.5, 4.25, 11.0, 4.25])
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
    E = 1
    Zm = [randn(Nw, n_tot) for m = 1:M]

    # m_true = 0.6
    # L_true = 6.25
    # g_true = 9.81
    # k_true = 6.25

    # # Supposed baseline minimum CONFIRMED
    # m_true = 0.31347614234659554
    # L_true = 6.5977461076540065
    # g_true = 9.187837718538397
    # k_true = 5.745295250122789

    # # Supposed proposed minimum DECONFIRMED, taken from opt_pars_proposed
    # m_true = 0.314900750524853
    # L_true = 6.538145444830321
    # g_true = 8.467384658475243
    # k_true = 6.196423857924822

    # Supposed proposed minimum PRETTY GOOD, k A BIT OFF
    # m_true = 0.33446203077924525
    # L_true = 6.199579981059762
    # g_true = 8.861699081959312
    # k_true = 5.72609788642009

    # # Supposed proposed minimum. Not perfect but pretty good
    # m_true = 0.33446203077924525
    # L_true = 6.199579981059762
    # g_true = 8.861699081959312
    # k_true = 6.0

    # Yet another supposed minimum, obtained with higher M
    m_true = 0.3739397136543269
    L_true = 5.42606245309193
    g_true = 7.969227340494199
    k_true = 7.580752417309324

    @warn "Not plotting around true parameter values, but around suspected minimum"
    true_pars_all = [m_true, L_true, g_true, k_true]

    # # DEBUG
    # # Ym1 = mean(simulate_system(exp_data, true_pars_all, M, dist_sens_inds, isws, Zm), dims=2)
    # # Ym2 = mean(simulate_system(exp_data, [0.3, 6.25, 9.81, 6.25], M, dist_sens_inds, isws, Zm), dims=2)
    # Ym3 = mean(simulate_system(exp_data, [0.314900750524853, 6.538145444830321, 8.467384658475243, 6.196423857924822], M, dist_sens_inds, isws, Zm), dims=2)
    # # cost1 = mean((Y[N_trans+1:end, 1].-Ym1[N_trans+1:end]).^2)
    # # cost2 = mean((Y[N_trans+1:end, 1].-Ym2[N_trans+1:end]).^2)
    # cost3 = mean((Y[N_trans+1:end, 1].-Ym3[N_trans+1:end]).^2)
    # @info "opt cost: $cost3"
    # # @info "Local cost: $(cost1), true cost: $(cost2)"

    # mdiffs = [-0.03, -0.02, -0.01, 0.0, 0.01, 0.02, 0.03]
    # Ldiffs = [-0.3, -0.2, -0.1, 0.0, 0.1, 0.2, 0.3]
    # gdiffs = [-0.3, -0.2, -0.1, 0.0, 0.1, 0.2, 0.3]
    # kdiffs = [-0.3, -0.2, -0.1, 0.0, 0.1, 0.2, 0.3]
    mdiffs = [-0.03, -0.025, -0.02, -0.015, -0.01, -0.005, 0.0, 0.005, 0.01, 0.015, 0.02, 0.025, 0.03]
    Ldiffs = [-0.3, -0.25, -0.2, -0.15, -0.1, -0.05, 0.0, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3]
    gdiffs = [-0.3, -0.25, -0.2, -0.15, -0.1, -0.05, 0.0, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3]
    # kdiffs = [-0.3, -0.25, -0.2, -0.15, -0.1, -0.05, 0.0, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3]
    kdiffs = [-0.3, -0.25, -0.2, -0.15, -0.1, -0.05, 0.0, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3]
    mpars = m_true.+mdiffs
    Lpars = L_true.+Ldiffs
    gpars = g_true.+gdiffs
    kpars = k_true.+kdiffs

    # @warn "Using the baseline method instead of the proposed method now!!!"
    e = 1
    cost_vals = [zeros(length(mdiffs)), zeros(length(Ldiffs)), zeros(length(gdiffs)), zeros(length(kdiffs))]
    for (im, mpar) in enumerate(mpars)
        pars = vcat(mpar, true_pars_all[2:end])
        Ym = mean(simulate_system(exp_data, pars, M, dist_sens_inds, isws, Zm), dims=2)
        # Ym = solvew(u, t -> 0.0, pars, N ) |> h
        cost_vals[1][im] = mean((Y[N_trans+1:end, e].-Ym[N_trans+1:end]).^2)
    end
    for (iL, Lpar) in enumerate(Lpars)
        pars = vcat(m_true, Lpar, true_pars_all[3:end])
        Ym = mean(simulate_system(exp_data, pars, M, dist_sens_inds, isws, Zm), dims=2)
        # Ym = solvew(u, t -> 0.0, pars, N ) |> h
        cost_vals[2][iL] = mean((Y[N_trans+1:end, e].-Ym[N_trans+1:end]).^2)
    end
    for (ig, gpar) in enumerate(gpars)
        pars = vcat(true_pars_all[1:2], gpar, k_true)
        Ym = mean(simulate_system(exp_data, pars, M, dist_sens_inds, isws, Zm), dims=2)
        # Ym = solvew(u, t -> 0.0, pars, N ) |> h
        cost_vals[3][ig] = mean((Y[N_trans+1:end, e].-Ym[N_trans+1:end]).^2)
    end
    for (ik, kpar) in enumerate(kpars)
        pars = vcat(true_pars_all[1:3], kpar)
        Ym = mean(simulate_system(exp_data, pars, M, dist_sens_inds, isws, Zm), dims=2)
        # Ym = solvew(u, t -> 0.0, pars, N ) |> h
        cost_vals[4][ik] = mean((Y[N_trans+1:end, e].-Ym[N_trans+1:end]).^2)
    end


    return mpars, Lpars, gpars, kpars, cost_vals
end

function debug_minimization_full_along_curve(expid::String, N_trans::Int = 0; pars0::Array{Float64,1}=[0.5, 4.25, 11.0, 4.25])
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

    @warn "Not using real true parameters, but substitutes"
    # true_pars_all = [0.31, 6.23, 9.725, 6.2]
    true_pars_all = [0.3072846387299368, 6.2553202969939505, 9.763055279858479, 6.159898907321155]
    # true_pars_all = [0.3, 6.25, 9.81, 6.25]
    get_all_parameters(pars::Array{Float64, 1}) = vcat(get_all_θs(pars), exp_data.get_all_ηs(pars))

    # E = size(Y, 2)
    # DEBUG
    E = 1
    Zm = [randn(Nw, n_tot) for m = 1:M]

    # ====== Optimizing paramters for baseline model ======
    function baseline_model_parametrized(dummy_input, pars)
        # NOTE: The true input is encoded in the solvew_sens()-function, but this function
        # still needs to to take two input arguments, so dummy_input could just be
        # anything, it's not used anyway
        Y_base = solvew(u, t -> zeros(n_out+length(dist_sens_inds)*n_out), pars, N ) |> h

        # NOTE: SCALAR_OUTPUT is assumed
        return reshape(Y_base[N_trans+1:end,:], :)   # Returns 1D-array
    end

    function jacobian_model_b(dummy_input, pars)
        jac = solvew_sens(u, t -> zeros(n_out+length(dist_sens_inds)*n_out), pars, N) |> h_sens
        return jac[N_trans+1:end, :]
    end

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
    #     # Sometimes (the first returned value I think) the baseline_trace
    #     # has no elements, and therefore doesn't contain the metadata x
    #     if length(baseline_trace) > 1
    #         for j=2:length(baseline_trace)
    #             push!(trace_base[e], baseline_trace[j].metadata["x"])
    #         end
    #     end
    # end

    @warn "Using custom optimal baseline instead of optimizing"
    # Optimum obtained from NEW_medium_2022_u2w6
    opt_pars_baseline[:,1] = [0.3130713086787108, 6.350474211582251, 9.766499125255166, 6.106793073907525]

    @info "The mean optimal parameters for baseline are given by: $(mean(opt_pars_baseline, dims=2))"

    # === Optimizing parameters for the proposed model or stochastic gradient descent ==
    function proposed_model_parametrized(δ, Zm, dummy_input, pars, isws)
        # NOTE: The true input is encoded in the solvew_sens()-function, but this function
        # still needs to to take two input arguments, so dummy_input could just be
        # anything, it's not used anyway
        p = get_all_parameters(pars)
        θ = p[1:dθ]
        η = p[dθ+1: dθ+dη]
        # C = reshape(η[nx+1:end], (n_out, n_tot))  # Not correct when disturbance model is parametrized, use dmdl.Cd instead

        dmdl = discretize_ct_noise_model(get_ct_disturbance_model(η, nx, n_out), δ)
        # # NOTE: OPTION 1: Use the rows below here for linear interpolation
        XWm = simulate_noise_process_mangled(dmdl, Zm)
        wmm(m::Int) = mk_noise_interp(dmdl.C, XWm, m, δ)
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
        return get_cost_gradient(y, Ym[:,1:M_mean], jacsYm[M_mean+1:end], N_trans)
    end

    # Returns estimate of gradient of cost function
    # M_mean specified over how many realizations the gradient estimate is computed
    function get_gradient_estimate_debug(y, δ, pars, isws, M_mean::Int=1)
        Ym, jacsYm = simulate_system_sens(exp_data, pars, 2M_mean, dist_sens_inds, isws)
        # Uses different noise realizations for estimate of output and estiamte of gradient
        grad_est = get_cost_gradient(y, Ym[:,1:M_mean], jacsYm[M_mean+1:end], N_trans)
        cost = mean((y[N_trans+1:end]-Ym[N_trans+1:end,1]).^2)
        # grad_est will be 2D array with one dimenion equal to 1, we want to return a 1D array
        return grad_est, cost
    end

    # Returns estimate of gradient of output
    function get_proposed_jacobian(pars, isws, M_mean::Int=1)
        jacYm = simulate_system_sens(exp_data, pars, M_mean, dist_sens_inds, isws)[2]
        return mean(jacYm)
    end

    opt_pars_proposed = zeros(length(pars0), E)
    trace_proposed = [[Float64[]] for e=1:E]
    trace_costs = [Float64[] for e=1:E]
    grad_trace  = [[Float64[]] for e=1:E]
    # for e=1:E
    #     get_gradient_estimate_p(pars, M_mean) = get_gradient_estimate(Y[:,e], δ, pars, isws, M_mean)
    #     get_gradient_estimate_p_debug(pars, M_mean) = get_gradient_estimate_debug(Y[:,e], δ, pars, isws, M_mean)
    #     jacobian_model(x, p) = get_proposed_jacobian(p, isws, M)
    #     opt_pars_proposed[:,e], trace_proposed[e], trace_costs[e], grad_trace[e] =
    #         perform_SGD_adam_debug(get_gradient_estimate_p_debug, pars0, par_bounds, verbose=true; maxiters=200, tol=1e-8)
    #     reset_isws!(isws)
    #
    #     println("Completed for dataset $e for parameters $(opt_pars_proposed[:,e])")
    # end

    @warn "Using custom optimal proposed instead of optimizing"
    # opt_pars_proposed[:,1] = [0.344339654705459, 5.563828034842836, 8.603799024532677, 7.924118847221097]   # Optimal value found previously after 500 iterations
    # opt_pars_proposed[:,1] = [0.344339654705459, 5.568828034842836, 8.65, 7.834118847221096]
    # Optimum obtained from NEW_medium_2022_u2w6
    opt_pars_proposed[:,1] = [0.3652565971753042, 5.419086209611716, 8.383448389450797, 8.048928288044202]

    @info "The mean optimal parameters for proposed method are given by: $(mean(opt_pars_proposed, dims=2))"

    # ======= Plotting costs =======
    diff_vec_base = opt_pars_baseline[:,1] - true_pars_all
    base_steps = -0.1:0.05:1.1
    base_costs = zeros(length(base_steps))
    base_pars = zeros(length(true_pars_all), length(base_steps))

    for (iα, α) in enumerate(base_steps)
        pars = true_pars_all + α*diff_vec_base
        base_pars[:,iα] = pars
        Ym = solvew(exp_data.u, t -> 0.0, pars, N) |> h
        base_costs[iα] = mean((Y[N_trans+1:end, 1].-Ym[N_trans+1:end]).^2)
    end

    # Recommended plotting:
    # p = plot(base_costs)
    # vline!(p, [3])
    # vline!(p, [length(base_costs)-2])

    diff_vec_prop = opt_pars_proposed[:,1] - true_pars_all
    prop_steps = -0.1:0.05:1.1
    # prop_steps = -0.1:0.05:2.1
    # prop_steps = -0.2:0.025:1.2
    prop_costs = zeros(length(prop_steps))
    prop_pars = zeros(length(true_pars_all), length(prop_steps))


    for (iα, α) in enumerate(prop_steps)
        # @warn "Only simulating for α greater than 1.1 now!!!"
        # if α > 1.1
        pars = true_pars_all + α*diff_vec_prop
        prop_pars[:,iα] = pars
        Ym = mean(simulate_system(exp_data, pars, M, dist_sens_inds, isws, Zm), dims=2)
        prop_costs[iα] = mean((Y[N_trans+1:end, 1].-Ym[N_trans+1:end]).^2)
        # end
    end

    return base_pars, base_costs, prop_pars, prop_costs
end

function gridsearch_sans_g(expid::String, N_trans::Int = 0)
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
    E = 1
    # @warn "Using E = 1 right now, instead of something larger"
    Zm = [randn(Nw, n_tot) for m = 1:M]

    mref = free_dyn_pars_true[1]
    Lref = free_dyn_pars_true[2]
    kref = free_dyn_pars_true[3]
    δm = 0.01
    δL = 0.1
    δk = 0.1

    mvals = [mref]
    Lvals = [Lref]
    kvals = kref-4*δk:δk:kref+5δk

    cost_vals = [zeros(length(mvals)*length(Lvals)*length(kvals)) for e=1:E]
    all_pars = zeros(length(free_dyn_pars_true), length(mvals)*length(Lvals)*length(kvals))
    min_ind = fill(-1, (E,))
    min_cost = fill(Inf, (E,))
    ind = 1
    time_start = now()
    for (im, my_m) in enumerate(mvals)
        for (iL, my_L) in enumerate(Lvals)
            for (ik, my_k) in enumerate(kvals)
                for e = 1:E
                    pars = [my_m, my_L, my_k]
                    all_pars[:,ind] = pars
                    Ym = mean(simulate_system(exp_data, pars, M, dist_sens_inds, isws, Zm), dims=2)
                    cost_vals[e][ind] = mean((Y[N_trans+1:end, e].-Ym[N_trans+1:end]).^2)
                    if cost_vals[e][ind] < min_cost[e]
                        min_ind[e] = ind
                        min_cost[e] = cost_vals[e][ind]
                    end
                    @info "Completed computing cost for e = $e, im=$im, iL=$iL, ik = $ik"
                end
                ind += 1
            end
        end
    end
    duration = now()-time_start

    return all_pars, cost_vals, min_ind, duration
end

function gridsearch_3distsens(expid::String, N_trans::Int = 0)
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
    E = 1
    # @warn "Using E = 1 right now, instead of something larger"
    Zm = [randn(Nw, n_tot) for m = 1:M]

    a1ref = 0.8
    a2ref = 16
    cref  = 0.6
    δa1 = 0.02
    δa2 = 0.2
    δc  = 0.02

    a1vals = a1ref-2δa1:δa1:a1ref+2δa1
    a2vals = a2ref-2δa2:δa2:a2ref+2δa2
    cvals  =   cref-2δc:δc:cref+2δc
    # a1vals = a1ref
    # a2vals = a2ref
    # cvals  = cref

    a1fixed = [[zeros(length(a2vals), length(cvals)) for i=1:length(a1vals)] for e=1:E]
    a2fixed = [[zeros(length(a1vals), length(cvals)) for i=1:length(a2vals)] for e=1:E]
    cfixed  = [[zeros(length(a1vals), length(a2vals)) for i=1:length(cvals)]  for e=1:E]
    cost_vals = [zeros(length(a1vals)*length(a2vals)*length(cvals)) for e=1:E]
    all_pars = zeros(3, length(a1vals)*length(a2vals)*length(cvals))
    min_ind = fill(-1, (E,))
    min_cost = fill(Inf, (E,))
    ind = 1
    time_start = now()
    try
        for (ia1, my_a1) in enumerate(a1vals)
            for (ia2, my_a2) in enumerate(a2vals)
                for (ic, my_c) in enumerate(cvals)
                    for e = 1:E
                        pars = [my_a1, my_a2, my_c]
                        all_pars[:,ind] = pars
                        Ym = mean(simulate_system(exp_data, pars, M, dist_sens_inds, isws, Zm), dims=2)
                        cost_vals[e][ind] = mean((Y[N_trans+1:end, e].-Ym[N_trans+1:end]).^2)
                        if cost_vals[e][ind] < min_cost[e]
                            min_ind[e] = ind
                            min_cost[e] = cost_vals[e][ind]
                        end
                        @info "Completed computing cost for e = $e, ia1=$ia2, ia2=$ia1, ic = $ic"

                        # Elements arranged in such a way that e.g.
                        # plot(a1vals, a2vals, cfixed[1][3], st=:surface)
                        # will have correctly labeled axes. Same for other plots
                        a1fixed[e][ia1][ic, ia2] = cost_vals[e][ind]
                        a2fixed[e][ia2][ic, ia1] = cost_vals[e][ind]
                        cfixed[e][ic][ia2, ia2]  = cost_vals[e][ind]
                    end
                    ind += 1
                end
            end
        end
    catch e
        bt = catch_backtrace()
        msg = sprint(showerror, e, bt)
        @warn "Terminating prematurely due to error"
        println(msg)
    end
    duration = now()-time_start

    # Can stack up plots nicely as follows:
    # p1 = plot(a1vals, a2vals, cfixed[1][3], legend=false, st=:surface)
    # p2...p5
    # l = @layout [a b; c d e]
    # plot(p1, p2, p3, p4, p5, layout=l)
    plot_tools = (a1vals, a2vals, cvals, a1fixed, a2fixed, cfixed)

    # # The following code can be used to plot a 3D scatter plot where cost is
    # # represented by color. cost_vals probably have to be re-scaled
    # using PlotlyJS
    #
    # plot(scatter(
    #     x=allpars[1,:],
    #     y=allpars[2,:],
    #     z=allpars[3,:],
    #     mode="markers",
    #     marker=attr(
    #         size=12,
    #         color=costvals2,                # set color to an array/list of desired values
    #         colorscale="Viridis",   # choose a colorscale
    #         opacity=0.8
    #     ),
    #     type="scatter3d"
    # ), Layout(margin=attr(l=0, r=0, b=0, t=0)))

    return all_pars, cost_vals, min_ind, duration, plot_tools
end

function newdebug_alongcurve(expid::String, N_trans::Int = 0)
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
    E = 10
    # @warn "Using E = 1 right now, instead of something larger"
    Zm = [randn(Nw, n_tot) for m = 1:M]
    # Zm5000 = [randn(Nw, n_tot) for m = 1:5000]

    opt_pars_proposed = readdlm("data/results/CDC23_20k/opt_pars_proposed.csv", ',')

    # ref1 = [0.28985242672688427, 6.423540049745506, 5.997677634129923, 0.5515604690820335, 17.301181823064447, 0.49488112002045737]
    ref2 = [0.3, 6.25, 6.25, 0.8, 16, 0.6]

    # vec = ref2-ref1;
    midsteps = 10
    sidesteps = 3
    frac = 1/midsteps;

    range = -sidesteps*frac:frac:1+sidesteps*frac

    range = 1-frac:frac/10:1+frac

    all_pars = [zeros(length(ref2), length(range)) for e=1:E]
    cost_vals = [zeros(length(range)) for e=1:E]
    # Ym5000 = [[zeros(N+1, 5000) for i=eachindex(range)] for e=1:E]

    ind = 1
    for e = 1:E
        ref1 = opt_pars_proposed[:,e]
        vec = ref2-ref1
        for diff = range
            pars = ref1+diff*vec
            all_pars[e][:,ind] = pars
            Ym = mean(simulate_system(exp_data, pars, M, dist_sens_inds, isws, Zm), dims=2)
            # Ym1000[e][ind] = simulate_system(exp_data, pars, 5000, dist_sens_inds, isws, Zm5000)
            # Ym = mean(Ym5000[e][ind][:,1:M], dims=2)
            cost_vals[e][ind] = mean((Y[N_trans+1:end, e].-Ym[N_trans+1:end]).^2)
            @info "Completed computing cost for e = $e,ind=$ind out of $(length(range)). Cost=$(cost_vals[e][ind])"
            ind += 1
        end
        writedlm("data/results/20k_alongcurve/cost_vals$e.csv", cost_vals[e], ',')
        writedlm("data/results/20k_alongcurve/pars$e.csv", all_pars[e], ',')
        ind = 1
    end

    return all_pars, cost_vals#, Ym5000
end

function Ym1000_alongcurve(expid::String, Ym1000::Vector{Vector{Matrix{Float64}}}, N_trans::Int = 0)
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
    E = 1#0
    # @warn "Using E = 1 right now, instead of something larger"
    # Zm = [randn(Nw, n_tot) for m = 1:M]
    # Zm1000 = [randn(Nw, n_tot) for m = 1:1000]

    opt_pars_proposed = readdlm("data/results/CDC23_20k/opt_pars_proposed.csv", ',')

    # ref1 = [0.28985242672688427, 6.423540049745506, 5.997677634129923, 0.5515604690820335, 17.301181823064447, 0.49488112002045737]
    ref2 = [0.3, 6.25, 6.25, 0.8, 16, 0.6]

    # vec = ref2-ref1;
    midsteps = 1#10
    sidesteps = 1#3
    frac = 1/midsteps;

    range = -sidesteps*frac:frac:1+sidesteps*frac

    all_pars = [zeros(length(ref2), length(range)) for e=1:E]
    cost_vals = [zeros(length(range)) for e=1:E]
    cost_vals_300 = [zeros(length(range)) for e=1:E]
    cost_vals_500 = [zeros(length(range)) for e=1:E]
    cost_vals_750 = [zeros(length(range)) for e=1:E]
    cost_vals_1000 = [zeros(length(range)) for e=1:E]

    ind = 1
    for e = 1:E
        ref1 = opt_pars_proposed[:,e]
        vec = ref2-ref1
        for diff = range
            pars = ref1+diff*vec
            all_pars[e][:,ind] = pars
            # Ym = mean(simulate_system(exp_data, pars, M, dist_sens_inds, isws, Zm), dims=2)
            # Ym1000[e][ind] = simulate_system(exp_data, pars, 1000, dist_sens_inds, isws, Zm1000)
            Ym = mean(Ym1000[e][ind][:,1:M], dims=2)
            myYm1000 = mean(Ym1000[e][ind], dims=2)
            myYm300 = mean(Ym1000[e][ind][:,1:300], dims=2)
            myYm500 = mean(Ym1000[e][ind][:,1:500], dims=2)
            myYm750 = mean(Ym1000[e][ind][:,1:750], dims=2)
            cost_vals[e][ind] = mean((Y[N_trans+1:end, e].-Ym[N_trans+1:end]).^2)
            cost_vals_1000[e][ind] = mean((Y[N_trans+1:end, e].-myYm1000[N_trans+1:end]).^2)
            cost_vals_300[e][ind]  = mean((Y[N_trans+1:end, e].-myYm300[N_trans+1:end]).^2)
            cost_vals_500[e][ind]  = mean((Y[N_trans+1:end, e].-myYm500[N_trans+1:end]).^2)
            cost_vals_750[e][ind]  = mean((Y[N_trans+1:end, e].-myYm750[N_trans+1:end]).^2)
            @info "Completed computing cost for e = $e,ind=$ind out of $(length(range))"
            ind += 1
        end
        # writedlm("data/results/20k_alongcurve/cost_vals$e.csv", cost_vals[e], ',')
        # writedlm("data/results/20k_alongcurve/pars$e.csv", all_pars[e], ',')
        ind = 1
    end

    return all_pars, cost_vals, cost_vals_300, cost_vals_500, cost_vals_750, cost_vals_1000
end

function newdebug_separatedim(expid::String, N_trans::Int = 0)
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
    E = 10
    # @warn "Using E = 1 right now, instead of something larger"
    Zm = [randn(Nw, n_tot) for m = 1:M]

    # opt_pars_proposed = readdlm("data/results/CDC23_20k/opt_pars_proposed.csv", ',')

    # ref1 = [0.28985242672688427, 6.423540049745506, 5.997677634129923, 0.5515604690820335, 17.301181823064447, 0.49488112002045737]
    ref = [0.3, 6.25, 6.25, 0.8, 16, 0.6]
    step_sizes = 0.01*ref#[0.01, 0.1, 0.1, 0.01, 0.1, 0.01]
    num_steps  = 5  # Steps taken on each side of the reference
    Emat = I(length(ref))

    @warn "Overwriting original ref"
    # Row 26 is trace_prop3 for results CDC23_20k (i.e. row 26 of trace for e=3 basically)
    # ref = [0.2642295187977945, 6.4804506374222175, 5.784126863615928, 0.5646819510243084, 17.7808097323123, 0.48348040209345633]

    # I lowered reference for k, since obtained results seemed to indicate that k was too high, despite other values being fine
    ref = [0.2642295187977945, 6.4804506374222175, 5.784126863615928, 0.5646819510243084, 17.7808097323123, 0.48348040209345633]

    # par_vals[e][j,i] Contains value for parameter j during iteration i, for data-set e
    # All parameters other than j are assumed to be fixed to the value given in ref
    par_vals  = [zeros(length(ref), 2num_steps+1) for e=1:E]
    cost_vals = [zeros(length(ref), 2*num_steps+1) for e=1:E]

    @warn "Only simulating for e=3"
    for e = 3:3
    # for e = 1:E
        # ref1 = opt_pars_proposed[:,e]
        # vec = ref2-ref1
        for j=eachindex(ref)    # Considers one parameter at a time
            ind = 1
            for i = -num_steps:num_steps
                pars = ref+i*step_sizes[j]*Emat[:,j]
                par_vals[e][j,ind] = pars[j]
                Ym = mean(simulate_system(exp_data, pars, M, dist_sens_inds, isws, Zm), dims=2)
                cost_vals[e][j, ind] = mean((Y[N_trans+1:end, e].-Ym[N_trans+1:end]).^2)
                @info "Completed computing cost for e = $e, j=$j, ind=$ind out of $(2num_steps+1). Cost=$(cost_vals[e][j, ind])"
                ind += 1
            end
        end
        writedlm("data/results/20k_separatedim/cost_vals$e.csv", cost_vals[e], ',')
        writedlm("data/results/20k_separatedim/pars$e.csv", par_vals[e], ',')
    end

    #= Can plot results nicely like this:
    p1 = plot(par_vals[1,:], cost_vals[1,:]);
    p2 = plot(par_vals[2,:], cost_vals[2,:]);
    p3 = plot(par_vals[3,:], cost_vals[3,:]);
    p4 = plot(par_vals[4,:], cost_vals[4,:]);
    p5 = plot(par_vals[5,:], cost_vals[5,:]);
    p6 = plot(par_vals[6,:], cost_vals[6,:]);
    l  = @layout [a b c; d e f];
    plot(p1,p2,p3,p4,p5,p6, layout=l)

    or without ticks

    p1 = plot(par_vals[1,:], cost_vals[1,:], yticks=false);
    p2 = plot(par_vals[2,:], cost_vals[2,:], yticks=false);
    p3 = plot(par_vals[3,:], cost_vals[3,:], yticks=false);
    p4 = plot(par_vals[4,:], cost_vals[4,:], yticks=false);
    p5 = plot(par_vals[5,:], cost_vals[5,:], yticks=false);
    p6 = plot(par_vals[6,:], cost_vals[6,:], yticks=false);
    l  = @layout [a b c; d e f];
    plot(p1,p2,p3,p4,p5,p6, layout=l)
    =#

    return par_vals, cost_vals
end

function gridsearch_debug(expid::String, N_trans::Int = 0)
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

    num_free_pars = length(free_dyn_pars_true) + length(dist_sens_inds)

    # get_all_parameters(pars::Array{Float64, 1}) = vcat(get_all_θs(pars), exp_data.get_all_ηs(pars))

    # E = size(Y, 2)
    # DEBUG
    E = 1
    # @warn "Using E = 1 right now, instead of something larger"
    Zm = [randn(Nw, n_tot) for m = 1:M]

    num = 10

    cref = 0.6
    δc   = 0.05
    cvals = cref-num*δc:δc:cref+num*δc
    # cvals = 0.0:0.1:cref-(num-1)*δc
    # cvals = 0.0:0.2:1.0

    # @warn "Not using default M here"
    # my_M = 100

    Ym_log = [zeros(5000, M) for i=1:length(cvals)]
    cost_vals = [zeros(length(cvals)) for e=1:E]
    alt_cost_vals = [zeros(length(cvals)) for e=1:E]
    min_ind = fill(-1, (E,))
    min_cost = fill(Inf, (E,))
    time_start = now()
    # for (cind, my_c) in enumerate(cvals)
    #     # Ym = mean(simulate_system(exp_data, [my_c], M, dist_sens_inds, isws, Zm), dims=2)
    #     Ym_log[cind] = simulate_system(exp_data, [my_c], M, dist_sens_inds, isws, Zm)
    #     Ym = mean(Ym_log[cind], dims=2)
    #     for e = 1:E
    #         cost_vals[e][cind] = mean((Y[N_trans+1:end, e].-Ym[N_trans+1:end]).^2)
    #         if cost_vals[e][cind] < min_cost[e]
    #             min_ind[e] = cind
    #             min_cost[e] = cost_vals[e][cind]
    #         end
    #     end
    #     @info "Completed computing cost for cind=$cind, c=$my_c "
    # end

    new_Y = simulate_system(exp_data, [1.0], 1, dist_sens_inds, isws, Zm)
    for (cind, my_c) in enumerate(cvals)
        # Ym = mean(simulate_system(exp_data, [my_c], M, dist_sens_inds, isws, Zm), dims=2)
        Ym_log[cind] = simulate_system(exp_data, [my_c], M, dist_sens_inds, isws, Zm)
        Ym = mean(Ym_log[cind], dims=2)
        for e = 1:E
            cost_vals[e][cind] = mean((Y[N_trans+1:end, e].-Ym[N_trans+1:end]).^2)
            alt_cost_vals[e][cind] = mean((new_Y[N_trans+1:end].-Ym[N_trans+1:end]).^2)
            if cost_vals[e][cind] < min_cost[e]
                min_ind[e] = cind
                min_cost[e] = cost_vals[e][cind]
            end
        end
        @info "Completed computing cost for cind=$cind, c=$my_c "
    end

    duration = now()-time_start

    return cvals, cost_vals, min_ind, duration, Ym_log, alt_cost_vals
end

function gridsearch_k_1distsens(expid::String, N_trans::Int = 0)
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

    num_free_pars = length(free_dyn_pars_true) + length(dist_sens_inds)

    # get_all_parameters(pars::Array{Float64, 1}) = vcat(get_all_θs(pars), exp_data.get_all_ηs(pars))

    # E = size(Y, 2)
    # DEBUG
    E = 1
    # @warn "Using E = 1 right now, instead of something larger"
    Zm = [randn(Nw, n_tot) for m = 1:M]

    ida = true

    if ida
        # For identifying k and a
        kref = free_dyn_pars_true[1]
        aref = 0.8
        δk = 0.1
        δa = 0.05
        # kvals = kref-10*δk:δk:kref+10δk
        # avals = aref-10*δa:δa:aref+10δa
        kvals = kref-10*δk:δk:kref+10δk
        avals = aref-10*δa:δa:aref+10δa
        # kvals = kref
        # avals = aref
    else
        # For identifying k and L
        Lref = free_dyn_pars_true[1]
        kref = free_dyn_pars_true[2]
        δk = 0.1
        δL = 0.1
        kvals = kref-3*δk:δk:kref+3δk
        avals = Lref-3*δL:δL:Lref+3δL
        # kvals = kref
        # avals = Lref
    end

    # cost_vals = [zeros(length(avals), length(kvals)) for e=1:E]
    cost_vals = [fill(NaN, (length(avals), length(kvals))) for e=1:E]
    ind = 1
    time_start = now()
    try
        for (ia, my_a) in enumerate(avals)
            for (ik, my_k) in enumerate(kvals)
                for e = 1:E
                    if ida
                        # For identifying k and a
                        pars = [my_k, my_a]
                    else
                        # For identifying k and L
                        pars = [my_a, my_k]
                    end

                    Ym = mean(simulate_system(exp_data, pars, M, dist_sens_inds, isws, Zm), dims=2)
                    cost_vals[e][ia, ik] = mean((Y[N_trans+1:end, e].-Ym[N_trans+1:end]).^2)
                    @info "Completed computing cost for e = $e, ia=$ia, ik=$ik "
                end
                ind += 1
            end
        end
    finally
        writedlm(joinpath(data_dir, "tmp/backup_avals.csv"), avals, ',')
        writedlm(joinpath(data_dir, "tmp/backup_kvals.csv"), avals, ',')
        for e=1:E
            writedlm(joinpath(data_dir, "tmp/cost_vals_$e.csv"), cost_vals[e], ',')
        end
    end
    duration = now()-time_start

    return avals, kvals, cost_vals, duration
end

function gridsearch_2distsens(expid::String, N_trans::Int = 0)
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

    num_free_pars = length(free_dyn_pars_true) + length(dist_sens_inds)

    # get_all_parameters(pars::Array{Float64, 1}) = vcat(get_all_θs(pars), exp_data.get_all_ηs(pars))

    # E = size(Y, 2)
    # DEBUG
    E = 1
    @warn "Using E = 1 right now, instead of something larger"
    Zm = [randn(Nw, n_tot) for m = 1:M]

    # NOTE: All parameters have to be set as free for this functions,
    # so that we can set m, L, k, and c to their fixed values
    # Taken from from_Alsvin/20k_hugest_differentadam/2000its_biasstart
    # mean starting from iteration 500 forward, of all parameters that were
    # relatively constant (so should be close to minimum)
    mfix = 0.2980673022654188
    Lfix = 6.288208419671489
    kfix = 6.264040559373865
    cfix = 0.5258645076185884

    a1ref = 0.8
    a2ref = 16.0

    δ1 = a1ref/50
    δ2 = a2ref/50
    num_steps = 10
    a1vals = a1ref-num_steps*δ1:δ1:a1ref+num_steps*δ1
    a2vals = a2ref-num_steps*δ2:δ2:a2ref+num_steps*δ2

    # cost_vals = [zeros(length(avals), length(kvals)) for e=1:E]
    cost_vals = [fill(NaN, (length(a1vals), length(a2vals))) for e=1:E]
    ind = 1
    time_start = now()
    try
        for (i1, my_a1) in enumerate(a1vals)
            for (i2, my_a2) in enumerate(a2vals)
                for e = 1:E
                    pars = [mfix, Lfix, kfix, my_a1, my_a2, cfix]
                    Ym = mean(simulate_system(exp_data, pars, M, dist_sens_inds, isws, Zm), dims=2)
                    # cost_vals[e][i1, i2] = mean((Y[N_trans+1:end, e].-Ym[N_trans+1:end]).^2)
                    cost_vals[e][i1, i2] = mean((Y[N_trans+1:end, 3].-Ym[N_trans+1:end]).^2)
                    @info "Completed computing cost for e = $e, ia=$i1, ik=$i2. WARN: Using e=3 instead of default"
                end
                ind += 1
            end
        end
    finally
        writedlm(joinpath(data_dir, "tmp/backup_a1vals.csv"), a1vals, ',')
        writedlm(joinpath(data_dir, "tmp/backup_a2vals.csv"), a2vals, ',')
        for e=1:E
            writedlm(joinpath(data_dir, "tmp/cost_vals_$e.csv"), cost_vals[e], ',')
        end
    end
    duration = now()-time_start

    return a1vals, a2vals, cost_vals, duration
end

function gridsearch_2distsens_directional(expid::String, N_trans::Int = 0)
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

    num_free_pars = length(free_dyn_pars_true) + length(dist_sens_inds)

    # get_all_parameters(pars::Array{Float64, 1}) = vcat(get_all_θs(pars), exp_data.get_all_ηs(pars))

    # E = size(Y, 2)
    # DEBUG
    E = 1
    @warn "Using E = 1 right now, instead of something larger"
    Zm = [randn(Nw, n_tot) for m = 1:M]

    # NOTE: All parameters have to be set as free for this functions,
    # so that we can set m, L, k, and c to their fixed values
    # Taken from from_Alsvin/20k_hugest_differentadam/2000its_biasstart
    # mean starting from iteration 500 forward, of all parameters that were
    # relatively constant (so should be close to minimum)
    mfix = 0.2980673022654188
    Lfix = 6.288208419671489
    kfix = 6.264040559373865
    cfix = 0.5258645076185884

    a1ref = 0.8
    a2ref = 16.0
    ref = [a1ref, a2ref]
    dir = [a1ref, -a2ref]
    ort_dir = [a2ref, a1ref]

    scale = 0.5
    num_steps = 10
    scales = -scale:(1/2num_steps):scale
    ort_scales = -0.02:0.01:0.02
    lenx = length(scales)
    leny = length(ort_scales)
    # Creates matrices, where each column is a vector pointing in the direction
    # of dir (ort_dir), with a scale given by scales (ort_scales), each row
    # corresponding to one scale
    xdiffs = dir*(collect(scales)')
    ydiffs = dir*(collect(ort_scales)')
    par_vals = Matrix{Vector{Float64}}(undef, (length(scales),length(ort_scales)))

    # cost_vals = [zeros(length(avals), length(kvals)) for e=1:E]
    cost_vals = [fill(NaN, (length(scales), length(ort_scales))) for e=1:E]
    ind = 1
    time_start = now()
    try
        for ix in 1:length(scales)
            for iy in 1:length(ort_scales)
                for e = 1:E
                    dist_pars = ref + xdiffs[:,ix] + ydiffs[:,iy]
                    par_vals[ix, iy] = dist_pars
                    pars = [mfix, Lfix, kfix, dist_pars[1], dist_pars[2], cfix]
                    Ym = mean(simulate_system(exp_data, pars, M, dist_sens_inds, isws, Zm), dims=2)
                    # cost_vals[e][i1, i2] = mean((Y[N_trans+1:end, e].-Ym[N_trans+1:end]).^2)
                    cost_vals[e][ix, iy] = mean((Y[N_trans+1:end, 3].-Ym[N_trans+1:end]).^2)
                    @info "Completed computing cost for e = $e, ix=$ix, iy=$iy out of ($lenx, $leny). WARN: Using e=3 instead of default"
                end
                # ind += 1
            end
        end
    finally
        writedlm(joinpath(data_dir, "tmp/backup_parvals.csv"), par_vals, ',')
        for e=1:E
            writedlm(joinpath(data_dir, "tmp/cost_vals_$e.csv"), cost_vals[e], ',')
        end
    end
    duration = now()-time_start

    return par_vals, cost_vals, duration
end

function gridsearch_with_grad_2distsens(expid::String, N_trans::Int = 0)
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

    num_free_pars = length(free_dyn_pars_true) + length(dist_sens_inds)

    # get_all_parameters(pars::Array{Float64, 1}) = vcat(get_all_θs(pars), exp_data.get_all_ηs(pars))

    # E = size(Y, 2)
    # DEBUG
    E = 1
    @warn "Using E = 1 right now, instead of something larger"
    Zm = [randn(Nw, n_tot) for m = 1:M]

    # NOTE: All parameters have to be set as free for this functions,
    # so that we can set m, L, k, and c to their fixed values
    # Taken from from_Alsvin/20k_hugest_differentadam/2000its_biasstart
    # mean starting from iteration 500 forward, of all parameters that were
    # relatively constant (so should be close to minimum)
    mfix = 0.2980673022654188
    Lfix = 6.288208419671489
    kfix = 6.264040559373865
    cfix = 0.5258645076185884

    a1ref = 0.8
    a2ref = 16.0

    δ1 = a1ref/50
    δ2 = a2ref/50
    num_steps = 10
    a1vals = a1ref-num_steps*δ1:δ1:a1ref+num_steps*δ1
    a2vals = a2ref-num_steps*δ2:δ2:a2ref+num_steps*δ2

    # cost_vals = [zeros(length(avals), length(kvals)) for e=1:E]
    cost_vals   = [fill(NaN, (length(a1vals), length(a2vals))) for e=1:E]
    cost_grads1 = [fill(NaN, (length(a1vals), length(a2vals))) for e=1:E]
    cost_grads2 = [fill(NaN, (length(a1vals), length(a2vals))) for e=1:E]
    ind = 1
    time_start = now()
    try
        for (i1, my_a1) in enumerate(a1vals)
            for (i2, my_a2) in enumerate(a2vals)
                for e = 1:E
                    pars = [mfix, Lfix, kfix, my_a1, my_a2, cfix]
                    Ym, jacsYm = simulate_system_sens(exp_data, pars, M, dist_sens_inds, isws, Zm)
                    cost_grad = get_cost_gradient(Y[:,3], Ym[:,1:M÷2], jacsYm[M÷2+1:end], N_trans)
                    # Ym = mean(simulate_system(exp_data, pars, M, dist_sens_inds, isws, Zm), dims=2)
                    # cost_vals[e][i1, i2] = mean((Y[N_trans+1:end, 3].-Ym[N_trans+1:end]).^2)
                    cost_vals[e][i1, i2] = get_cost_value(Y[:,3], Ym, N_trans)
                    cost_grads1[e][i1, i2] = cost_grad[1]
                    cost_grads2[e][i1, i2] = cost_grad[2]
                    @info "Completed computing cost and gradient for e = $e, i1=$i1, i2=$i2. WARN: Using e=3 instead of default"
                end
                ind += 1
            end
        end
    finally
        writedlm(joinpath(data_dir, "tmp/backup_a1vals.csv"), a1vals, ',')
        writedlm(joinpath(data_dir, "tmp/backup_a2vals.csv"), a2vals, ',')
        for e=1:E
            writedlm(joinpath(data_dir, "tmp/cost_vals_$e.csv"), cost_vals[e], ',')
            writedlm(joinpath(data_dir, "tmp/cost_grads1_$e.csv"), cost_grads1[e], ',')
            writedlm(joinpath(data_dir, "tmp/cost_grads2_$e.csv"), cost_grads2[e], ',')
        end
    end
    duration = now()-time_start

    return a1vals, a2vals, cost_vals, cost_grads1, cost_grads2, duration
end

function get_3par_plottable(all_pars, cost_vals)
    firstfixed = Array{Tuple()}
end

function allpar_viz_1par(all_pars::Array{Float64,2}, cost_vals::Array{Float64,1}, par_vecs::Array{Array{Float64,1},1}, var_par_ind::Int, fixed_pars::Array{Float64,1})
    num_pars = size(par_vecs, 1)
    @assert length(fixed_pars) == num_pars-1 "All parameters except one have to be fixed"

    perm = sortperm(all_pars[var_par_ind, :])
    all_pars_sorted = all_pars[:, perm]
    cost_vals_sorted = cost_vals[perm]
    # all_pars[:,perm] is sorted according to var_par_ind

    # Find elements corresponding to indices in fixed pars
    # fixed_pars is assumed to contain indices of desired fixed values of parameters (indices in par_vecs)
    fixed_inds = setdiff(1:num_pars, var_par_ind)
    indices_to_keep = collect(1:size(all_pars,2))
    for i=1:length(fixed_inds)
        ind = fixed_inds[i]
        indices_to_remove = findall(all_pars_sorted[ind,:].!=fixed_pars[i])
        setdiff!(indices_to_keep, indices_to_remove)
    end
    return perm, all_pars_sorted, cost_vals_sorted, indices_to_keep
end

function allpar_viz_2par(all_pars::Array{Float64,2}, cost_vals::Array{Float64,1}, par_vecs::Array{Array{Float64,1},1}, var_par_inds::Array{Int,1}, fixed_pars::Array{Float64,1})
    num_pars = size(par_vecs, 1)
    @assert length(fixed_pars) == num_pars-2 "All parameters except two have to be fixed"

    perm = sortperm(all_pars[var_par_inds[1], :])
    all_pars_sorted = all_pars[:, perm]
    cost_vals_sorted = cost_vals[perm]
    # all_pars[:,perm] is sorted according to var_par_ind

    # Find elements corresponding to indices in fixed pars
    # fixed_pars is assumed to contain indices of desired fixed values of parameters (indices in par_vecs)
    fixed_inds = setdiff(1:num_pars, var_par_inds)
    indices_to_keep = collect(1:size(all_pars,2))
    for i=1:length(fixed_inds)
        ind = fixed_inds[i]
        indices_to_remove = findall(all_pars_sorted[ind,:].!=fixed_pars[i])
        setdiff!(indices_to_keep, indices_to_remove)
    end

    # Generating cost matrix
    num_cols = length(par_vecs[var_par_inds[1]])
    num_rows = length(par_vecs[var_par_inds[2]])
    cost_mat = zeros(num_rows, num_cols)
    pars_to_keep = all_pars[:,indices_to_keep]
    costs_to_keep = cost_vals_sorted[indices_to_keep]
    for col_ind = 1:length(indices_to_keep)÷num_rows
        cost_mat[:, col_ind] = costs_to_keep[ (col_ind-1)*num_rows+1:col_ind*num_cols]
    end
    return perm, all_pars_sorted, cost_vals_sorted, indices_to_keep, cost_mat
end

function alt_viz_2par(all_pars::Array{Float64,2}, cost_vals::Array{Float64,1}, par_vecs::Array{Array{Float64,1},1}, plot_par_inds::Array{Int,1})
    @assert length(plot_par_inds) == 2 "Can only plot with respect to two parameters at a time (length(plot_par_inds) must be 2)"
    num_pars = size(par_vecs, 1)
    non_plot_inds = setdiff(1:num_pars, plot_par_inds)

    par1_vec = par_vecs[plot_par_inds[1]]
    par2_vec = par_vecs[plot_par_inds[2]]
    opt_pars_mat = fill(zeros(num_pars-2), (length(par1_vec), length(par2_vec)))
    for par1_ind in 1:length(par1_vec)
        for par2_ind in 1:length(par2_vec)
            par1 = par1_vec[par1_ind]
            par2 = par2_vec[par2_ind]
            temp = setdiff(1:length(cost_vals), findall(all_pars[plot_par_inds[1],:].==par1))
            inds_to_search_among = setdiff(setdiff(1:length(cost_vals), findall(all_pars[plot_par_inds[1],:].!=par1)), findall(all_pars[plot_par_inds[2],:].!=par2))
            cost_min = Inf
            min_ind = 0
            for ind in inds_to_search_among
                if cost_min > cost_vals[ind]
                    cost_min = cost_vals[ind]
                    min_ind = ind
                end
            end
            opt_pars_mat[par1_ind, par2_ind] = all_pars[non_plot_inds,min_ind]
        end
    end
    plot_par_vecs = [par1_vec, par2_vec]
    return plot_par_vecs, opt_pars_mat
end

function mock_delete()
    pars1 = [1.,2.0]
    pars2 = [1.,2.0]
    pars3 = [1.,2.0]
    pars4 = [1.,2.0]
    all_pars = zeros(4, length(pars1)*length(pars2)*length(pars3)*length(pars4))
    i=1
    for par1 in pars1
        for par2 in pars2
            for par3 in pars3
                for par4 in pars4
                    all_pars[:,i] = [par1, par2, par3, par4]
                    i += 1
                end
            end
        end
    end
    par_vecs = [pars1, pars2, pars3, pars4]
    cost_vals = ones(size(all_pars,2))
    cost_vals[1] = 0.5
    cost_vals[6] = 0.5
    cost_vals[11] = 0.5
    cost_vals[16] = 0.5
    return all_pars, par_vecs, cost_vals
end

function find_index_allpar(all_pars, desired_pars)
    minds = findall(all_pars[1,:].==desired_pars[1])
    Linds = findall(all_pars[2,:].==desired_pars[2])
    ginds = findall(all_pars[3,:].==desired_pars[3])
    kinds = findall(all_pars[4,:].==desired_pars[4])
    index = intersect(intersect(intersect(minds, Linds), ginds), kinds)
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
    @warn "WARNING: This way of defining C is incorrect whenever disturbance model is parametrized. Should use dmdl.Cd instead, but not obvisous how to do that in this code, since no dmdl2 is defined"

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
    p = vcat(get_all_θs(free_dyn_pars_true), exp_data.get_all_ηs(free_dyn_pars_true))
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

# Also adds time-varying beta_1, dependent on the new hyper-parameter λ
#!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
# TODO: IT WAS A BAD IDEA TO DELETE PREVIOUS SGD DEBUD, MANY DEBUG FUNCTIONS USE IT!!!! ADD IT BACK FROM OLD COMMIT!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
#!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
function perform_SGD_adam_debug(
    get_grad_estimate::Function,
    pars0::Array{Float64,1},                        # Initial guess for parameters
    bounds::Array{Float64, 2},                      # Parameter bounds
    learning_rate::Function=learning_rate_vec_red;
    tol::Float64=1e-6,
    maxiters=200,
    verbose=false,
    betas::Array{Float64,1} = [0.9, 0.999],
    λ = 0.5)   # betas are the decay parameters of moment estimates
    # betas::Array{Float64,1} = [0.5, 0.999])   # betas are the decay parameters of moment estimates

    eps = 0.#1e-8
    q = 20# TODO: This is a little arbitrary, but because of low tolerance, the stopping criterion is never reached anyway
    last_q_norms = fill(Inf, q)
    running_criterion() = mean(last_q_norms) > tol
    s = zeros(size(pars0)) # First moment estimate
    r = zeros(size(pars0)) # Second moment estimate

    t = 1
    pars = pars0
    trace = [pars]
    grad_trace = []
    step_trace = []
    lrate_trace = []
    margin = 0#10
    while t <= maxiters+margin
        grad_est = get_grad_estimate(pars, M_rate(t))

        beta1t = betas[1]*(λ^(t-1))
        s = beta1t*s + (1-beta1t)*grad_est
        r = betas[2]*r + (1-betas[2])*(grad_est.^2)
        shat = s/(1-betas[1]^t) # Seems like betas[1] should be used instead of beta1t here
        rhat = r/(1-betas[2]^t)
        unscaled_step = -shat./(sqrt.(rhat).+eps)
        step = learning_rate(t, norm(grad_est)).*unscaled_step
        # step = -learning_rate_vec_red(t, norm(grad_est)).*shat./(sqrt.(rhat).+eps)
        # println("Iteration $t, gradient norm $(norm(grad_est)) and step sign $(sign(-first(grad_est))) with parameter estimate $pars")
        # running_criterion(grad_est) || break
        if verbose
            println("Iteration $t, average gradient norm $(mean(last_q_norms)), -gradient $(-grad_est) and step $(step) with parameter estimate $pars")
        end
        running_criterion() || break
        if t > margin
            pars = pars + step
            project_on_bounds!(pars, bounds)
            push!(trace, pars)
        end
        push!(grad_trace, grad_est)
        push!(step_trace, step)
        push!(lrate_trace, learning_rate(t, norm(grad_est)).*unscaled_step)
        # (t-1)%10+1 maps t to a number between 1 and 10 (inclusive)
        last_q_norms[(t-1)%q+1] = norm(grad_est)
        t += 1
    end
    return pars, trace, grad_trace, step_trace, lrate_trace
end

# function perform_SGD_adam_debug(
#     get_grad_estimate_debug::Function,
#     pars0::Array{Float64,1},                        # Initial guess for parameters
#     bounds::Array{Float64, 2},                      # Parameter bounds
#     learning_rate::Function=learning_rate_vec;
#     tol::Float64=1e-4,
#     maxiters=200,
#     verbose=false,
#     betas::Array{Float64,1} = [0.9, 0.999])   # betas are the decay parameters of moment estimates
#     # betas::Array{Float64,1} = [0.5, 0.999])   # betas are the decay parameters of moment estimates
#
#     eps = 0.#1e-8
#     q = 20
#     last_q_norms = fill(Inf, q)
#     running_criterion() = mean(last_q_norms) > tol
#     s = zeros(size(pars0)) # First moment estimate
#     r = zeros(size(pars0)) # Second moment estimate
#
#     t = 1
#     pars = pars0
#     cost_vals = Float64[]
#     trace = [pars]
#     grad_norms = Float64[]
#     grad_trace = []
#     while t <= maxiters
#         grad_est, cost_val = get_grad_estimate_debug(pars, M_rate(t))
#         push!(cost_vals, cost_val)
#         push!(grad_norms, norm(grad_est))
#         s = betas[1]*s + (1-betas[1])*grad_est
#         r = betas[2]*r + (1-betas[2])*(grad_est.^2)
#         shat = s/(1-betas[1]^t)
#         rhat = r/(1-betas[2]^t)
#         step = -learning_rate(t, norm(grad_est)).*shat./(sqrt.(rhat).+eps)
#         # println("Iteration $t, gradient norm $(norm(grad_est)) and step sign $(sign(-first(grad_est))) with parameter estimate $pars")
#         # running_criterion(grad_est) || break
#         if verbose
#             println("Iteration $t, average gradient norm $(mean(last_q_norms)), gradient $(grad_est) and step $(step) with parameter estimate $pars")
#         end
#         running_criterion() || break
#         pars = pars + step
#         project_on_bounds!(pars, bounds)
#         push!(trace, pars)
#         push!(grad_trace, grad_est)
#         # (t-1)%10+1 maps t to a number between 1 and 10 (inclusive)
#         last_q_norms[(t-1)%q+1] = norm(grad_est)
#         t += 1
#     end
#     return pars, trace, cost_vals, grad_trace
# end

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
    pars_opt, trace, grad_trace = perform_SGD_adam(get_gradient_estimate, x0, bounds, learning_rate_vec, verbose=true, maxiters=400, tol=1e-6)
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

function plot_boxplots(θs, θ0)
    # θs should be a matrix, each column corresponds to one box, containing
    # all outcomes for that box. θ0 should correspond to the true parameter
    # value, or some other value where one wants a horizontal line drawn

    p = boxplot(
    θs,
    # xticks = (idxs, labels),
    # label = "",
    # ylabel = L"\hat{\theta}",
    notch = false,
    )

    hline!(p, [θ0], label = L"\theta_0", linestyle = :dot, linecolor = :gray)
end

function plot_boxplots(θs, θ0, labels)
    # θs should be a matrix, each column corresponds to one box, containing
    # all outcomes for that box. θ0 should correspond to the true parameter
    # value, or some other value where one wants a horizontal line drawn

    idxs = collect(1:length(labels))
    p = boxplot(
    θs,
    xticks = (idxs, labels),
    # label = "",
    # ylabel = L"\hat{\theta}",
    notch = false,
    )

    hline!(p, [θ0], label = L"\theta_0", linestyle = :dot, linecolor = :gray)
    savefig("C:\\Programming\\dae-param-est\\src\\julia\\data\\results\\50k_hugest\\boxplot.png")
end
