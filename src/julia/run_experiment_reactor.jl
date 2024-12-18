using LsqFit, StatsPlots, LaTeXStrings, Dates
include("simulation.jl")
include("noise_interpolation_multivar.jl")
include("noise_generation.jl")

seed = 1234
Random.seed!(seed)

struct ExperimentData
    # Y is the measured output of system, contains N+1 rows and E columns
    # (+1 because of an additional sample at t=0)
    Y::Array{Float64,2}
    # ny is the dimension of the output. Necessary to since all outputs
    # for all different times are stacked on top of each other in Y
    ny::Int
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

# @warn "Running with M = 100 instead of default"
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
function get_cost_gradient(Y::Array{Float64,1}, Ym::Array{Float64,2}, jacsYm::Array{Array{Float64,2},1}, N_trans::Int=0)
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
    fit_result, trace = curve_fit(model, jacobian_model, Float64[], Ye, pars, lower=lb, upper=ub, show_trace=true, inplace=false, x_tol=1e-8)    # Default inplace = false, x_tol = 1e-8
    return fit_result, trace
end

# === MODEL (AND DATA) PARAMETERS ===
const σ = 0.002                 # measurement noise variance

const k0 = 1.0                  # [?], pre-exponential factor, see https://en.wikipedia.org/wiki/Arrhenius_equation
const k1 = 1.0                  # [?], Eₐ/R, activation energy divided by universal gas constant, see Arrhenius equation
const k2 = 0.02                 # [?], ΔHᵣ/ρcₚ, ΔHᵣ might be change in enthalpy, see https://www.vedantu.com/chemistry/heat-of-reaction
const k3 = 1.0                  # [?], 1/ρcₚ, ρ is likely density of fluid,
# cₚ seems to be specific heat capacity, see https://en.wikipedia.org/wiki/Specific_heat_capacity
# or https://en.wikipedia.org/wiki/Thermal_conductivity for more context
const k4 = 1.0                  # [dm³] V_h, volume of heating fluid

const V0 = 10.0                 # [dm³] Initial volume of liquid in tank
const T0 = 293.15               # [K] Initial temperature of liquid in tank, Celsius = Kelvin - 273.15

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
# const pars_true = [m, L, g, k]                    # true value of all free parameters
const pars_true = [k4]#[k0, k1, k2, k3, k4] # True values of free parameters
get_all_θs(pars::Array{Float64,1}) = [k0, k1, k2, k3, pars[1]]
# Each row corresponds to lower and upper bounds of a free dynamic parameter.
# dyn_par_bounds = [0.01 1e4; 0.1 1e4; 0.1 1e4; 0.1 1e4]
# dyn_par_bounds = [0.01 1e4; 0.1 1e4]#; 0.1 1e4; 0.1 1e4]
# dyn_par_bounds = [0.01 1e4; 0.1 1e4; 0.1 1e4; 0.1 1e4]
dyn_par_bounds = [0.01 1e4]
const_learning_rate = [1.0]
learning_rate_vec(t::Int, grad_norm::Float64) = const_learning_rate#if (t < 100) const_learning_rate else ([0.1/(t-99.0), 1.0/(t-99.0)]) end#, 1.0, 1.0]  #NOTE Dimensions must be equal to number of free parameters
learning_rate_vec_red(t::Int, grad_norm::Float64) = const_learning_rate./sqrt(t)
model_to_use = fast_heat_transfer_reactor
# TODO: Finish from here down! solvew(exp_data.u, t -> wmm(m)(t), pars, N) |> h
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
f(x::Array{Float64,1}) = x#x[2]               # applied on the state at each step
f_debug(x::Array{Float64,1}) = x
f_sens(x::Array{Float64,1}) = []
@warn "f_sens not defined right now!!! So SGD will, of course, not work"
# NOTE: Has to be changed when number of free  parameters is changed.
# Specifically, f_sens() must return sensitivity wrt to all free parameters
h(sol) = apply_outputfun(f, sol)                            # for our model
h_comp(sol) = apply_two_outputfun_mvar(f, f_sens, sol)           # for complete model with dynamics sensitivity
h_sens(sol) = apply_outputfun_mvar(f_sens, sol)              # for only returning sensitivity
h_debug(sol) = apply_outputfun(f_debug, sol)

const num_dyn_pars = size(dyn_par_bounds, 1)
realize_model_sens(u::Function, w::Function, pars::Array{Float64, 1}, N::Int) = problem(
    model_to_use(V0, T0, u, w, get_all_θs(pars)),
    N,
    Ts,
)
realize_model(u::Function, w::Function, pars::Array{Float64, 1}, N::Int) = problem(
    model_to_use(V0, T0, u, w, get_all_θs(pars)),
    N,
    Ts,
)
const dθ = length(get_all_θs(pars_true))

# === SOLVER PARAMETERS ===
const abstol = 1e-9#1e-8
const reltol = 1e-6#1e-5
const maxiters = Int64(1e8)
# const maxiters = Int64(1e8)

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

# NOTE: Adds same measurement noise to every component
# data-set output function
h_data(sol) = apply_outputfun(x -> f(x) + σ * randn(size(f(x))), sol)

# DEBUG INPUT
u1(t) = 0.3 + 0.05*sin(t);        # FA
u2(t) = 3.2;                      # CA0
u3(t) = 293.15;                   # TA
u4(t) = 0.3 + 0.02*sin(0.5*t);    # F
u5(t) = 0.1 + 0.01*sin(2*t);      # Fh
u6(t) = 313.30;                   # Th
u(t) = [u1(t); u2(t); u3(t); u4(t); u5(t); u6(t)]
# END DEBUG

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

    # E = size(Y, 2)
    # DEBUG
    E = 1
    @warn "Using E = $E instead of default"
    opt_pars_baseline = zeros(length(pars0), E)
    # trace_base[e][t][j] contains the value of parameter j before iteration t
    # corresponding to dataset e
    trace_base = [[pars0] for e=1:E]
    setup_duration = now() - start_datetime
    baseline_durations = Array{Millisecond, 1}(undef, E)
    # @warn "Not running baseline identification"
    for e=1:E
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
    function get_gradient_estimate(y, pars, isws, M_mean::Int=1)
        Ym, jacsYm = simulate_system_sens(exp_data, pars, 2M_mean, dist_par_inds, isws)

        # Uses different noise realizations for estimate of output and estiamte of jacobian
        return get_cost_gradient(y, Ym[:,1:M_mean], jacsYm[M_mean+1:end], N_trans)
    end

    # Returns estimate of gradient of output
    function get_proposed_jacobian(pars, isws, M_mean::Int=1)
        jacYm = simulate_system_sens(exp_data, pars, M_mean, dist_par_inds, isws)[2]
        return mean(jacYm, dims=2)
    end

    get_gradient_estimate_p(pars, M_mean) = get_gradient_estimate(Y[:,1], pars, isws, M_mean)

    opt_pars_proposed = zeros(length(pars0), E)
    avg_pars_proposed = zeros(length(pars0), E)
    trace_proposed = [ [Float64[]] for e=1:E]
    trace_gradient = [ [Float64[]] for e=1:E]
    proposed_durations = Array{Millisecond, 1}(undef, E)
    # @warn "Not running proposed identification now" # DEBUG
    for e=1:E
        time_start = now()
        # jacobian_model(x, p) = get_proposed_jacobian(pars, isws, M)  # NOTE: This won't give a jacobian estimate independent of Ym, but maybe we don't need that since this isn't SGD?
        @warn "Only using maxiters=100 right now"
        opt_pars_proposed[:,e], trace_proposed[e], trace_gradient[e] =
            perform_SGD_adam_new(get_gradient_estimate_p, pars0, par_bounds, verbose=true, tol=1e-8, maxiters=100)
            # perform_SGD_adam(get_gradient_estimate_p, pars0, par_bounds, verbose=true, tol=1e-8, maxiters=100)
        avg_pars_proposed[:,e] = mean(trace_proposed[e][end-80:end])

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
    return opt_pars_baseline, opt_pars_proposed, avg_pars_proposed, trace_base, trace_proposed, trace_gradient, durations
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
    input  = readdlm(exp_path(expid)*"/U.csv", ',')
    XW     = readdlm(exp_path(expid)*"/XW.csv", ',')
    W_meta_raw, W_meta_names =
        readdlm(exp_path(expid)*"/meta_W.csv", ',', header=true)

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
    # free_pars = vcat(true, fill(false, nx-1), true, fill(false, n_tot*n_out-1))       # First parameter of a-vector and first parameter of c-vector unknown
    free_par_inds = findall(free_pars)          # Indices of free variables in η. Assumed to be sorted in ascending order.
    # Array of tuples containing lower and upper bound for each free disturbance parameter
    dist_par_bounds = Array{Float64}(undef, 0, 2)
    # dist_par_bounds = [-Inf Inf]#; -Inf Inf]
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

    # # Exact interpolation
    # mk_we(XW::Array{Array{Float64, 1},2}, isws::Array{InterSampleWindow, 1}) =
    #     (m::Int) -> mk_newer_noise_interp(
    #     a_vec::AbstractArray{Float64, 1}, C_true, XW, m, n_in, δ, isws)
    # Linear interpolation. Not recommended DEBUG
    @warn "Using linear interpolation to compute true output (default is exact interpolation)"
    mk_we(XW::Array{Array{Float64,1},2}, isws::Array{InterSampleWindow, 1}) =
        (m::Int) -> mk_noise_interp(C_true, mangle_XW(XW), m, δ)

    #TODO: WARN: This function (interpw) is not adapted to multidimensional input!
    # u(t::Float64) = interpw(input, 1, 1)(t)
    @warn "Input function u(t) is hard-coded instead of being loaded from file" #DEBUG
    u1(t) = 0.3 + 0.05*sin(t);        # FA
    u2(t) = 3.2;                      # CA0
    u3(t) = 293.15;                   # TA
    u4(t) = 0.3 + 0.02*sin(0.5*t);    # F
    u5(t) = 0.1 + 0.01*sin(2*t);      # Fh
    u6(t) = 313.30;                   # Th
    u(t) = [u1(t); u2(t); u3(t); u4(t); u5(t); u6(t)]

    # === We first generate the output of the true system ===
    function calc_Y(XW::Array{Array{Float64, 1},2}, isws::Array{InterSampleWindow, 1})
        # NOTE: This XW should be non-mangled, which is why we don't divide by n_tot
        @assert (Nw <= size(XW, 1)) "Disturbance data size mismatch ($(Nw) > $(size(XW, 1)))"
        # @warn "Using E=1 instead of size of XW when generating Y!"
        E = size(XW, 2)
        # E = 1
        es = collect(1:E)
        we = mk_we(XW, isws)
        return solve_in_parallel_multivar(e -> solvew(u, we(e), pars_true, N) |> h_data, es)
    end

    if isfile(exp_path(expid)*"/Y.csv")
        @info "Reading output of true system"
        Y = readdlm(exp_path(expid)*"/Y.csv", ',')
        ny = readdlm(exp_path(expid)*"/ny.csv", ',')[1]
        isws = [initialize_isw(Q, W, n_tot, true) for e=1:M]
    else
        @info "Generating output of true system"
        isws = [initialize_isw(Q, W, n_tot, true) for e=1:max(size(XW,2), M)]
        Y, ny = calc_Y(demangle_XW(XW, n_tot), isws)
        writedlm(exp_path(expid)*"/Y.csv", Y, ',')
        writedlm(exp_path(expid)*"/ny.csv", ny, ',')
    end

    # # This block can be used to check whether different implementations result
    # # in the same Y
    # @warn "Debugging sim. First 5 XW: $(XW[1:5])"
    # wdebug = mk_noise_interp(C_true, XW, 1, δ)
    # my_y = solvew(u, wdebug, pars_true, N) |> h
    # writedlm("data/experiments/pendulum_sensitivity/my_y.csv", my_y, ',')

    reset_isws!(isws)
    return ExperimentData(Y, ny, u, get_all_ηs, dist_par_bounds, W_meta, Nw), isws
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
    N = size(exp_data.Y, 1)÷exp_data.ny-1

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
    return solve_in_parallel_multivar(m -> calc_mean_y(pars, m), collect(1:M))[1]   # Returns Ym
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
    N = size(exp_data.Y, 1)÷exp_data.ny-1
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
    # wmm(m::Int) = mk_newer_noise_interp(view(η, 1:nx), dmdl.Cd, XWm, m, n_in, δ, isws)

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
    # sample = readdlm(joinpath(dir, "backup_baseline_e1.csv"), ',')
    sample = readdlm(joinpath(dir, "backup_proposed_e1.csv"), ',')
    k = length(sample)
    opt_pars_baseline = zeros(k, E)
    opt_pars_proposed = zeros(k, E)
    avg_pars_proposed = zeros(k, E)
    for e=1:E
        # opt_pars_baseline[:,e] = readdlm(joinpath(dir, "backup_baseline_e$e.csv"), ',')
        opt_pars_proposed[:,e] = readdlm(joinpath(dir, "backup_proposed_e$e.csv"), ',')
        avg_pars_proposed[:,e] = readdlm(joinpath(dir, "backup_average_e$e.csv"), ',')
    end
    return opt_pars_baseline, opt_pars_proposed, avg_pars_proposed
end

function debug_minimization(expid::String, pars0::Array{Float64,1}, N_trans::Int = 0)
    exp_data, isws = get_experiment_data(expid)
    W_meta = exp_data.W_meta
    Y = exp_data.Y
    ny = exp_data.ny
    N = size(Y,1)÷ny-1
    Nw = exp_data.Nw
    nx = W_meta.nx
    n_in = W_meta.n_in
    n_out = W_meta.n_out
    n_tot = nx*n_in
    dη = length(W_meta.η)
    u = exp_data.u
    par_bounds = vcat(dyn_par_bounds, exp_data.dist_par_bounds)
    dist_sens_inds = W_meta.free_par_inds

    ########################################################################
    # TODO: CHECK THE README-FILE CORRESPONDING TO THIS FILE THAT DESCRIBES WHAT YOU HAVE TO DO
    ########################################################################

    get_all_parameters(pars::Array{Float64, 1}) = vcat(get_all_θs(pars), exp_data.get_all_ηs(pars))

    # === Optimizing parameters for the baseline model ===
    function baseline_model_parametrized(δ, dummy_input, pars)
        # NOTE: The true input is encoded in the solvew()-function, but this function
        # still needs to to take two input arguments, so dummy_input could just be
        # anything, it's not used anyway
        # Y_base = solvew(u, t -> zeros(n_out), pars, N ) |> h
        # return reshape(Y_base[ny*N_trans+1:end,:], :)   # Returns 1D-array
        temp = solvew(u, t -> zeros(n_out), pars, N ) |> h
        Y_base = vcat(temp...)
        return Y_base[ny*N_trans+1:end]
    end

    function jacobian_model_b(x, pars)
        # jac = solvew_sens(u, t -> 0.0, pars, N) |> h_sens
        # return jac[N_trans+1:end, :]
        temp = solvew_sens(u, t -> 0.0, pars, N) |> h_sens
        @info "typeof: $(typeof(temp)), size: $(size(temp))"
        jac = vcat(temp...) # TODO: This won't give us the correct Jacobian when we have several parameters
        return jac[ny*N_trans+1:end]
    end

    # E = size(Y, 2)
    # DEBUG
    E = 1
    @warn "Using E = $E instead of default"
    opt_pars_baseline = zeros(length(pars0), E)
    trace_base = [[pars0] for e=1:E]
    @warn "Not running baseline identification"
    for e=[]#1:E
    # for e=4:4
        # HACK: Uses trace returned due to hacked LsqFit package
        # TODO: Okay, so here convergence fails, while in proposed method below
        # there are no convergence issues, despite disturbance??? Yeah, something
        # is weird, fix that :)
        baseline_result, baseline_trace = get_fit_sens(Y[ny*N_trans+1:end,e], pars0,
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
    base_par_vals = 0.7:0.05:1.3
    par_vector = [[el] for el in base_par_vals]
    Ybs = [zeros(ny*(N+1)) for j = 1:length(par_vector)]
    base_cost_vals = zeros(length(base_par_vals), E)
    # @warn "Not plotting baseline cost function"
    for (j, pars) in enumerate(par_vector)
        ########################################################################
        # TODO: DO WE REALLY NEED THE APPLY_OUTPUTFUN_MVAR DEFINED IN SIMULATION.JL??
        # TODO: ISN'T THE MVAR APPROACH WE USE HERE ENOUGH??
        ########################################################################
        temp = solvew(exp_data.u, t -> zeros(n_out), pars, N) |> h
        # vcat(temp...) flattens temp from array of arrays into a 1D-array
        # Inner array elements vary inner-most in resulting array
        Ybs[j] = vcat(temp...)
    end
    for ind = 1:length(base_par_vals)
        # Y has columns indexed with 1:E because in case E is changed for debugging purposes,
        # without the dimensions of Y changing, we can get a mismatch otherwise
        # base_cost_vals[ind,:] = mean((Y[N_trans+1:end, 1:E].-Ybs[ind][:,N_trans+1:end]).^2, dims=1)
        base_cost_vals[ind,:] = ny*mean((Y[ny*N_trans+1:end, 1:E].-Ybs[ind][ny*N_trans+1:end]).^2, dims=1)
    end

    @info "Finished with baseline plots???"

    # === Computing cost function for proposed model ===
    # TODO: Should we consider storing and loading white noise, to improve repeatability
    # of the results? Currently repeatability is based on fixed random seed
    Zm = [randn(Nw, n_tot) for m = 1:M]

    # NOTE: CURRENTLY ONLY TREATS SCALAR PARAMETERS
    @info "Plotting proposed cost function..."
    prop_par_vals = 0.7:0.05:1.3
    prop_par_vals = 0.8:0.01:1.2
    prop_cost_vals = zeros(length(prop_par_vals), E)
    @warn "Actually, not plotting proposed cost function"
    for ind = []#1:length(prop_par_vals)
        # NOTE: If we don't pass Zm here, we will see that the cost function
        # looks very irregular even with M = 500. That's a bit suprising,
        # I would expect it to average out over that many iterations
        # Ym = simulate_system(exp_data, [prop_par_vals[ind]], M, dist_sens_inds, isws, Zm)
        Ym = simulate_system(exp_data, [prop_par_vals[ind]], M, dist_sens_inds, isws, Zm) #DEBUG
        Y_mean = mean(Ym, dims=2)
        # Y has columns indexed with 1:E because in case E is changed for debugging purposes,
        # at which point Y will have more than E columns and we'll get a dimension mismatch
        prop_cost_vals[ind,:] = ny*mean((Y[ny*N_trans+1:end, 1:E].-Y_mean[ny*N_trans+1:end]).^2, dims=1)
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
        Ym, ny = solve_in_parallel_multivar(m -> calc_mean_y(pars, m), ms)
        return reshape(mean(Ym[ny*N_trans+1:end,:], dims = 2), :) # Returns 1D-array
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
        Ym, ny = solve_in_parallel_multivar(m -> calc_mean_y(pars, m), ms)
        debug += 1 # FIX THIS, SHOULDN'T RETURN MEAN LIKE THIS, USE ny
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
    Ym_true = mean(simulate_system(exp_data, pars_true, M, dist_sens_inds, isws, Zm), dims=2)
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
        Ym, ny = solve_in_parallel_multivar(m -> calc_mean_y(pars, m), ms)
        debug += 1 # FIX THIS, SHOULDN'T RETURN MEAN LIKE THIS, USE ny
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
        Ym, ny = solve_in_parallel_multivar(m -> calc_mean_y(pars, m), ms)
        debug += 1 # FIX THIS, SHOULDN'T RETURN MEAN LIKE THIS, USE ny
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

# function plot_baseline_cost(expid::String, N_trans::Int=0)
#     exp_data, isws = get_experiment_data(expid)
#     Y = exp_data.Y
#     N = size(Y,1)-1
#     u = exp_data.u
#
#     k_diffs = -0.3:0.025:0.3
#     kpars = 6.25.+k_diffs
#     kcosts = zeros(length(k_diffs))
#     for (ik, kpar) in enumerate(kpars)
#         Y_base = solvew(u, t -> 0.0, [kpar], N ) |> h
#         kcosts[ik] = mean((Y[N_trans+1:end, 2].-Y_base[N_trans+1:end]).^2)
#     end
#     return kpars, kcosts
# end

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

function perform_SGD_adam_debug(
    get_grad_estimate_debug::Function,
    pars0::Array{Float64,1},                        # Initial guess for parameters
    bounds::Array{Float64, 2},                      # Parameter bounds
    learning_rate::Function=learning_rate_vec;
    tol::Float64=1e-4,
    maxiters=200,
    verbose=false,
    betas::Array{Float64,1} = [0.9, 0.999])   # betas are the decay parameters of moment estimates
    # betas::Array{Float64,1} = [0.5, 0.999])   # betas are the decay parameters of moment estimates

    eps = 0.#1e-8
    q = 20
    last_q_norms = fill(Inf, q)
    running_criterion() = mean(last_q_norms) > tol
    s = zeros(size(pars0)) # First moment estimate
    r = zeros(size(pars0)) # Second moment estimate

    t = 1
    pars = pars0
    cost_vals = Float64[]
    trace = [pars]
    grad_norms = Float64[]
    grad_trace = []
    while t <= maxiters
        grad_est, cost_val = get_grad_estimate_debug(pars, M_rate(t))
        push!(cost_vals, cost_val)
        push!(grad_norms, norm(grad_est))
        s = betas[1]*s + (1-betas[1])*grad_est
        r = betas[2]*r + (1-betas[2])*(grad_est.^2)
        shat = s/(1-betas[1]^t)
        rhat = r/(1-betas[2]^t)
        step = -learning_rate(t, norm(grad_est)).*shat./(sqrt.(rhat).+eps)
        # println("Iteration $t, gradient norm $(norm(grad_est)) and step sign $(sign(-first(grad_est))) with parameter estimate $pars")
        # running_criterion(grad_est) || break
        if verbose
            println("Iteration $t, average gradient norm $(mean(last_q_norms)), gradient $(grad_est) and step $(step) with parameter estimate $pars")
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
    return pars, trace, cost_vals, grad_trace
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
