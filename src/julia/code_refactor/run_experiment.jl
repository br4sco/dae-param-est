include("noise_generation.jl")
include("noise_interpolation_multivar.jl")
# include("minimizers.jl")
include("models.jl")
using .NoiseGeneration: DisturbanceMetaData, demangle_XW, get_ct_disturbance_model, discretize_ct_noise_model_disc_then_diff, simulate_noise_process, simulate_noise_process_mangled, discretize_ct_noise_model_with_adj_SDEApprox_mats
using .NoiseGeneration: discretize_ct_noise_model, discretize_ct_noise_model_with_adj_SDEApprox_mats_Ainvertible, get_multisines, discretize_ct_noise_model_diff_then_disc
using .NoiseInterpolation: InterSampleWindow, initialize_isw, reset_isws!, noise_inter, mk_newer_noise_interp, mk_noise_interp, linear_interpolation_multivar
using .DynamicalModels: AdjointSDEApproxData
using Interpolations: Cubic, BSpline, NoInterp, Line, extrapolate, scale, interpolate, Extrapolation
using DelimitedFiles: readdlm, writedlm
using LsqFit: curve_fit, coef
using LinearAlgebra: I
# For pendulum.jl file.
using .DynamicalModels: pendulum, pendulum_forward_m, pendulum_forward_k, get_pendulum_initial, get_pendulum_initial_msens, get_pendulum_initial_ksens, get_pendulum_initial_distsens
using .DynamicalModels: pendulum_adjoint_m, pendulum_adjoint_k_1dist_ODEdist, pendulum_adjoint_k_1dist, pendulum_forward_k_1dist, Model_ode, Model
# For delta_robot.jl file
using .DynamicalModels: delta_robot, delta_forward_γ, delta_adjoint_γ, delta_forward_allpar_alldist, delta_adjoint_allpar_alldist, delta_adjoint_allpar_alldist_ODEdist
using .DynamicalModels: get_delta_initial_with_mats, get_delta_initial_L0sens, get_delta_initial_L1sens, get_delta_initial_L2sens, get_delta_initial_L3sens, get_delta_initial_LC1sens
using .DynamicalModels: get_delta_initial_LC2sens, get_delta_initial_M1sens, get_delta_initial_M2sens, get_delta_initial_M3sens, get_delta_initial_J1sens, get_delta_initial_J2sens, get_delta_initial_γsens
using .DynamicalModels: delta_forward_1dist, delta_adjoint_1dist, delta_adjoint_1dist_ODEdist, delta_adjoint_allpar, delta_adjoint_alldist, delta_adjoint_M3

import CSV, Statistics

include("simulation.jl")
# From the above include, we get
# import DifferentialEquations as DE
# import Sundials
# import ProgressMeter

# === DATA TYPES AND CONSTANTS ===
const PENDULUM::Int = 1
const MOH_MDL::Int  = 2
const DELTA::Int    = 3

const FOR_SENS::Int = 1
const ADJ_SENS::Int = 2
const ADJ_ODEDIST::Int = 3

struct ExperimentData
    # Y is the measured output of system, contains N+1 rows and E columns
    # (+1 because of an additional sample at t=0)
    Y::Matrix{Float64}
    # u is the function specifying the input of the system
    u::Function
    # W_meta is the metadata of the disturbance, containting e.g. dimensions
    W_meta::DisturbanceMetaData
    # Sampling time at which the data is sampled
    Ts::Float64
end

# ===================================================================================================
# ============================== Values that can be set by user =====================================
# ===================================================================================================
# Selects which model to use/simulate
model_id::Int = DELTA
# Number of monte-carlo simulations used for estimation, e.g. samples of the gradient used to estimate the gradient
# Note that, when two independent estimates are needed, a total of 2M samples is generated, M for each estimate
const M::Int = Threads.nthreads()÷2

# Settings for the InterSampleWindow, only relevant when user_exact_interp in get_experiment_data() is set to true
const W::Int = 100           # Number of intervals for which isw stores data
const Q::Int = 1000          # Number of conditional samples stored per interval

# The initial learning rate for each component for each component of the disturbance parameters ρ
# The components corresponding to the free disturbance parameters η are picked out later
# using a call to get_disturbance_free_pars
# dist_init_learning_rate = [0.1, 0.2, 0.2, 0.05, 0.05, 0.05, 0.05, 0.05, 0.05, 0.05, 0.05, 0.05] # Small disturbance model
dist_init_learning_rate = vcat([0.1, 0.2, 0.2, 0.05, 0.05, 0.05], zeros(9), [0.05, 0.05, 0.05], zeros(9), [0.05, 0.05, 0.05]) # Large disturbance model
# Similarly for disturbance parameter bounds
dist_bounds = repeat([-Inf Inf], 30)

function get_disturbance_free_pars(nx::Int, nw::Int, n_tot::Int)::Vector{Bool}
    # Use this function to specify which parameters should be free and optimized over
    # Each element represent whether the corresponding element in η is a free parameter
    # Structure: η = vcat(ηa, ηc), where ηa is nx large, and ηc is n_tot*nw large
    # free_dist_pars = fill(false, nx + n_tot*nw)                                             # Known disturbance model
    # free_dist_pars = vcat(fill(true, nx), fill(false, nw), fill(true, (n_tot-1)*nw))     # Whole a-vector and all but first nw elements of c-vector unknown (MAXIMUM UNKNOWN PARAMETERS FOR SINGLE DIFFERENTIABILITY (PENDULUM))
    # free_dist_pars = vcat(fill(true, nx), fill(true, n_tot*nw))                     # All parameters unknown (MAXIMUM UNKNOWN PARAMETERS, NO DIFFERENTIABILITY (DELTA))
    free_dist_pars = begin
        tmp = fill(false, nx + n_tot*nw)
        tmp[1:nx] .= true   # a-parameters
        tmp[[1,4,7,11,14,17,21,24,27].+nx] .= true  # c-parameters
        tmp
    end                 # New disturbance model but corresponding same free parameter as old disturbance model all parameters, delta case
    # free_dist_pars = vcat(fill(true, nx), fill(false, n_tot*nw))                    # Whole a-vector unknown
    # free_dist_pars = vcat(true, fill(false, nx-1), fill(false, n_tot*nw))           # First parameter of a-vector unknown
    # free_dist_pars = vcat(false, true, fill(false, nx-2), fill(false, n_tot*nw))    # Second parameter of a-vector unknown
end
# ===================================================================================================
# ========================= End of values that should be set by user ================================
# ===================================================================================================

# === LOADING MODEL AND GENERATING RELATED FUNCTIONS ===
if model_id == PENDULUM
    include("model_metadata/pendulum.jl")
    md = pend_model_data # md is short for 'model data'
elseif model_id == DELTA
    include("model_metadata/delta_robot.jl") # md is short for 'model data'
    md = delta_model_data
end

h_data(sol,θ) = apply_outputfun(x -> md.f(x,θ) .+ md.σ * randn(size(md.f(x,θ))), sol)       # Output of "true" system, including measurement noise
h(sol,θ) = apply_outputfun(x->md.f(x,θ), sol)                                               # Output of model, no measurement noise 
h_comp(sol,θ) = apply_two_outputfun(x->md.f(x,θ), x->md.f_sens(x,θ), sol)                   # For complete model with dynamics sensitivity
h_sens_base(sol,θ) = apply_outputfun(x->md.f_sens_baseline(x,θ), sol)                       # For baseline method, which does not include sensitivity wrt disturbance parameters
h_all_adj(sol,θ) = apply_outputfun(x->md.f_all_adj(x,θ), sol)                               # For adjoint method, needs to get all nominal states, including model output

# === SOLVER PARAMETERS ===
const abstol::Float64 = 1e-8#1e-9
const reltol::Float64 = 1e-5#1e-6
const maxiters_sol::Int = Int(1e8)
const maxiters_opt::Int = 100

# === HELPER FUNCTIONS TO READ AND WRITE DATA ===
const data_dir::String = joinpath("../data", "experiments")
exp_path(expid) = joinpath(data_dir, expid)
data_Y_path(expid) = joinpath(exp_path(expid), "Y.csv")

# The type of interpolation used on forward solution in adjoint method
const interp_type = Cubic()

# ===================== HELPER FUNCTIONS =====================
# data should have time along rows (dim 1) and dimension along columns (dim 2)
function get_interpolation(data::AbstractMatrix{Float64}, T::Float64, Ts::Float64)
    extrapolate(scale(interpolate(data, (BSpline(interp_type), NoInterp())), 0.0:Ts:T, 1:size(data,2)), Line())
end

# Approximates the derivative of func(⋅,1:dim) using backward differences
function get_der_est(ts, func, dim)
    der_est = zeros(length(ts)-1, dim)
    for (i,t) = enumerate(ts)
        if i > 1
            der_est[i-1,:] = (func(t,1:dim)-func(ts[i-1],1:dim))./(t-ts[i-1])
        end
    end
    return der_est
end

# ====================== MAIN FUNCTIONS ======================

solve_wrapper(mdl_func::Function, u::Function, w::Function, pars::Vector{Float64}, T::Float64, Ts::Float64; kwargs...) = DE.solve(
    begin
        m = mdl_func(u, w, md.get_all_pars(pars), md)
        DE.DAEProblem(m.f!, m.dx0, m.x0, (0.0, T), (), differential_vars=m.dvars)
    end,
    saveat = 0:Ts:T,
    abstol = abstol,
    reltol = reltol,
    maxiters = maxiters_sol;
    kwargs...,
)

# Similar to solve_wrapper(), except 1. adjoint models require more input arguments, and 2. the solution is made backwards in time instead of forward, and 3. returns the get_Gθ() function too
solve_adj_wrapper(mdl_adj_func::Function, 
    u::Function, w::Function, pars::Vector{Float64}, T::Float64, Ts::Float64,
    x_func::Extrapolation,
    x2_func::Extrapolation,
    y_func::Extrapolation,
    dy_func::Extrapolation,
    xp0::AbstractMatrix{Float64},
    dx::Extrapolation,
    dx2::Extrapolation; ad::Union{Nothing,AdjointSDEApproxData}=nothing, kwargs...) = 
begin
    m, get_Gθ = begin
        if isnothing(ad)
            mdl_adj_func(w, md.get_all_pars(pars), T, x_func, x2_func, y_func, dy_func, xp0, dx, dx2)
        else
            mdl_adj_func(w, md.get_all_pars(pars), T, x_func, x2_func, y_func, dy_func, xp0, dx, dx2, ad)
        end
    end
    DE.solve(
        DE.DAEProblem(m.f!, m.dx0, m.x0, (T, 0.0), (), differential_vars=m.dvars),
        saveat = 0:Ts:T,
        abstol = abstol,
        reltol = reltol,
        maxiters = maxiters_sol;
        kwargs...,
    ), get_Gθ
end


function get_experiment_data(expid::String; use_exact_interp::Bool = false, E_gen::Integer = 100)::Tuple{ExperimentData, Array{InterSampleWindow, 1}}

    U_meta_raw, U_meta_names = 
        readdlm(joinpath(data_dir, expid*"/meta_U.csv"), ',', header=true)
    Y_meta_raw, Y_meta_names =
        readdlm(joinpath(data_dir, expid*"/meta_Y.csv"), ',', header=true)
    n_u_out = Int(U_meta_raw[1,3])
    Ts = Y_meta_raw[1,1]
    N = Int(Y_meta_raw[1,2])
    W_meta_raw, W_meta_names =
        readdlm(joinpath(data_dir, expid*"/meta_W.csv"), ',', header=true)
    W_meta = get_disturbance_metadata(W_meta_raw)

    u = if size(U_meta_raw, 2) == 3
        # Input is multisine
        dim = Int(U_meta_raw[1,3])
        # The first column of U_meta_raw is the amplitudes of the multisines while the other is the frequencies
        get_multisines(reshape(U_meta_raw[:,1], :, dim), reshape(U_meta_raw[:,2], :, dim))
    else
        # Input is from the same model as the disturbance
        input  = readdlm(joinpath(data_dir, expid*"/U.csv"), ',')[:]
        linear_interpolation_multivar(input, U_meta_raw[1,8], n_u_out)(t)   # U_meta_raw[1,8] is δ used for input signal. Usually same as W_meta.δ, but doesn't need to be.
    end

    # Makes sure there is a tmp-directory, for future use
    if !isdir(joinpath(data_dir, "tmp/"))
        mkdir(joinpath(data_dir, "tmp/"))
    end

    isws::Vector{InterSampleWindow} = if use_exact_interp
        [initialize_isw(Q, W, W_meta.nx*W_meta.nv, true) for _=1:max(2M,E_gen)]
    else
        Array{InterSampleWindow}(undef, 0)
    end

    # We only want to load the disturbance corresponding to the currently simulated realization of Y,
    # because loading all disturbance data in one go uses needlessly much memory. Therefore, we create
    # a wrapper for loading the needed disturbances and simulating the corresponding Y
    function sim_Y_wrapper(e::Int)::Vector{Float64}
        let nx = W_meta.nx, nw = W_meta.nw, nv = W_meta.nv, δ = W_meta.δ
            XW = transpose(CSV.read(exp_path(expid)*"/XW_T.csv", CSV.Tables.matrix; header=false, skipto=e, limit=1))
            C_true = reshape(W_meta.η[nx+1:end], (nw, nx*nv))
            w = if use_exact_interp
                a_vec = W_meta.η[1:nx]
                mk_newer_noise_interp(a_vec, C_true, demangle_XW(XW, nx*nv), 1, nv, δ, isws[e:e])
            else
                mk_noise_interp(C_true, XW, 1, δ)
            end
            # This function call returns an Array of Arrays, outer index is sample index, and each sample is also an array, since the output can be multidimensional
            nested_y = solve_wrapper(md.model_nominal, u, w, md.free_dyn_pars_true, N*Ts, Ts) |> sol -> h_data(sol,md.get_all_pars(md.free_dyn_pars_true))
            # The output is collapsed into a single long Array before returning
            return vcat(nested_y...)
        end
    end

    # ------------- Loading/generate "true" system output ---------------
    if isfile(data_Y_path(expid))
        @info "Loading output data from file"
        Y = readdlm(data_Y_path(expid), ',')
    else
        @info "Generating output data"
        Y = solve_in_parallel(e -> sim_Y_wrapper(e), 1:E_gen, md.ny, N+1)  # N+1 because we also include extra sample for t=0 in Y
        writedlm(data_Y_path(expid), Y, ',')
    end

    return ExperimentData(Y, u, W_meta, Ts), isws

end

function get_disturbance_metadata(W_meta_raw::Matrix{Float64})::DisturbanceMetaData
    nx = Int(W_meta_raw[1,1])
    nv = Int(W_meta_raw[1,2])
    nw = Int(W_meta_raw[1,3])
    η_true = W_meta_raw[:,4]
    num_rel = Int(W_meta_raw[1,6])
    Nw = Int(W_meta_raw[1,7])
    δ = Float64(W_meta_raw[1,8])
    n_tot = nx*nv

    free_dist_pars = get_disturbance_free_pars(nx, nw, n_tot)
    free_par_inds = findall(free_dist_pars) 
    # Array of tuples containing lower and upper bound for each free disturbance parameter
    # dist_par_bounds = hcat(fill(-Inf, nx+n_tot*nw, 1), fill(Inf, nx+n_tot*nw, 1))
    free_par_bounds = hcat(fill(-Inf, length(free_par_inds), 1), fill(Inf, length(free_par_inds), 1))
    function get_all_ηs(free_dist_pars::Vector{Float64})
        # If copy() is not used here, some funky stuff that I don't fully understand happens.
        # I think essentially η_true stops being defined after function returns, so
        # setting all_η to its value doesn't behave quite as I expected
        all_η = copy(η_true)
        # Fetches user-provided values for free disturbance parameters only
        all_η[free_par_inds] = free_dist_pars       # TODO: I think this gives a very uninformative error message if wrong number of parameters is passed, fix error messages?
        return all_η
    end

    DisturbanceMetaData(nx, nv, nw, η_true, free_par_inds, free_par_bounds, get_all_ηs, num_rel, Nw, δ)
end

function get_baseline_estimates(pars0::Vector{Float64}, exp_data::ExperimentData; verbose::Bool = true, E_in::Int=typemax(Int))
    let N = size(exp_data.Y, 1)÷md.ny-1, E = min(size(exp_data.Y, 2), E_in), dη = length(exp_data.W_meta.η), W_meta = exp_data.W_meta

        opt_pars_baseline = zeros(md.dθ, E)
        trace_baseline = [[pars0] for e=1:E]

        for e=1:E

            # === Setting up functions used by the optimizer ===
            function get_baseline_output(_, free_pars)
                sol = solve_wrapper(md.model_nominal, exp_data.u, t->zeros(W_meta.nw*(1+dη)), free_pars, N*Ts, Ts)
                Y_base = h(sol, md.get_all_pars(free_pars))
                # Each output sample can be multi-dimensional, vcat(Y_base...) splats all samples into one long vector
                return vcat(Y_base...)
            end

            function get_baseline_sens(_, free_pars)
                sol = solve_wrapper(md.model_sens, exp_data.u, t->zeros(W_meta.nw*(1+dη)), free_pars, N*Ts, Ts)
                jac = h_sens_base(sol, md.get_all_pars(free_pars))
                # Each jacobian sample can be multi-dimensional, vcat(jac...) splats all samples into one long vector
                return vcat(jac...)
            end

            # === Computing result ===
            baseline_result = curve_fit(
                get_baseline_output, 
                get_baseline_sens, 
                Float64[], 
                exp_data.Y[:,e], 
                pars0, 
                lower=md.par_bounds[:,1], 
                upper=md.par_bounds[:,2], 
                show_trace=verbose, 
                inplace=false, 
                x_tol=1e-8)    # Default: inplace = false, x_tol = 1e-8

            # === Saving result and trace ===
            opt_pars_baseline[:, e] = coef(baseline_result)

            # Sometimes (the first returned value I think) the trace_baseline has no elements, and therefore doesn't contain the metadata dx
            if length(baseline_result.trace) > 1
                for j=2:length(baseline_result.trace)
                    push!(trace_baseline[e], trace_baseline[e][end]+baseline_result.trace[j].metadata["dx"])
                end
            end

            @info "Completed for dataset $e for parameters $(opt_pars_baseline[:,e])"

            writedlm(joinpath(data_dir, "tmp/backup_baseline_e$e.csv"), opt_pars_baseline[:,e], ',')
            writedlm(joinpath(data_dir, "tmp/backup_trace_baseline_e$e.csv"), trace_baseline[e], ',')

        end

        return opt_pars_baseline, trace_baseline
    end
end

function get_proposed_estimates(pars0::Vector{Float64}, exp_data::ExperimentData, isws::Vector{InterSampleWindow}; 
    use_exact_interp::Bool = false, maxiters::Int = maxiters_opt, verbose::Bool = true, E_in::Int=typemax(Int), method_type::Int = FOR_SENS, Ainvertible::Bool = false)
    
    # Tsλ is the sampling period of the forward solution that is then used for the backwards computation for the adjoint method, good to have smaller for better interpolation
    let N = size(exp_data.Y, 1)÷md.ny-1, E = min(size(exp_data.Y, 2), E_in), W_meta = exp_data.W_meta, δ = W_meta.δ, Ts = exp_data.Ts, Tsλ = exp_data.Ts/10

        # Makes sure there is a "tmp" directory to store intermediate results in during simulation
        if !isdir(joinpath(data_dir, "tmp/"))
            mkdir(joinpath(data_dir, "tmp/"))
        end

        # Depending on method, the function used for computing cost function gradient estimate differs a lot
        get_gradient_func = if method_type == FOR_SENS

            # Helper function to improve readability. Instead of calling solve_wrapper with lots of arguments and then transforming
            # the solution to samples of the output, this function can be called instead
            function get_one_out_sens_rel(w_func::Function, pars::Vector{Float64})
                sol = solve_wrapper(md.model_sens, exp_data.u, w_func, pars, N*Ts, Ts)
                h_comp(sol, md.get_all_pars(pars))
            end

            # Returns estimate of gradient of cost function
            # M_mean specifies over how many realizations the gradient estimate is computed
            function get_gradient_estimate_for(y::Vector{Float64}, free_pars::Vector{Float64}, isws::Vector{InterSampleWindow}, M_mean::Int=1)
                # --- Generates disturbance signal for the provided parameters ---
                # (Assumes that disturbance model is parametrized, in theory doing this multiple times could be avoided if the disturbance model is known)
                Zm = [randn(W_meta.nx*W_meta.nv, W_meta.Nw+1) for _ = 1:2M_mean]
                η = W_meta.get_all_ηs(free_pars[md.dθ+1:end])  # NOTE: Assumes that the disturbance parameters always come after the dynamical parameters.
                dmdl = discretize_ct_noise_model_diff_then_disc(get_ct_disturbance_model(η, W_meta.nx, W_meta.nv), δ, W_meta.free_par_inds)
                wmm(m::Int) = if use_exact_interp
                        reset_isws!(isws)
                        XWm = simulate_noise_process(dmdl, Zm)
                        mk_newer_noise_interp(view(η, 1:W_meta.nx), dmdl.Cd, XWm, m, W_meta.nv, δ, isws)
                    else
                        XWm = simulate_noise_process_mangled(dmdl, Zm)
                        mk_noise_interp(dmdl.Cd, XWm, m, δ)
                    end

                # 2M_mean solutions computed because the used realizations of Ym and jacsYm have to be independent
                # (For computing the Ym, a smaller DAE could be solved instead, but this is not currently implemented)
                Ym, jacsYm = solve_in_parallel_sens(m->get_one_out_sens_rel(wmm(m),free_pars), 1:2M_mean, md.ny, md.dθ+length(W_meta.free_par_inds), N)

                # --- Computes cost function gradient ---
                # Y.-Ym is a matrix with as many columns as Ym, where column i contains Y-Ym[:,i]
                # Taking the mean of that gives us the average error as a function of time over all realizations contained in Ym.
                # mean(-jacsYm) is the average (over all m) jacobian of Ym.
                # Different rows correspond to different t, while different columns correspond to different parameters
                (2/N) * (transpose(Statistics.mean(-jacsYm))*Statistics.mean(y.-Ym, dims=2))[:]
            end

        elseif method_type == ADJ_SENS || (method_type == ADJ_ODEDIST && length(W_meta.free_par_inds)==0)
            
            function get_one_grad_rel_adj(y_func, dy_func, xvec1::Matrix{Float64}, xvec2::Matrix{Float64}, free_pars::Vector{Float64}, w::Function)
                x_func  = get_interpolation(xvec1, N*Ts, Tsλ)
                x2_func = get_interpolation(xvec2, N*Ts, Tsλ)
    
                der_est  = get_der_est(0.0:Tsλ:N*Ts, x_func, size(xvec1,2))
                der_est2 = get_der_est(0.0:Tsλ:N*Ts, x2_func, size(xvec1,2))
                # Subtracting Tsλ/2 because sometimes we don't get the right number of elements due to numerical inaccuracies otherwise
                dx =  get_interpolation(der_est,  N*Ts-Tsλ/2, Tsλ)
                dx2 = get_interpolation(der_est2, N*Ts-Tsλ/2, Tsλ)
        
                # ----------------- Actually solving adjoint system ------------------------
                xp0 = md.get_sens_init(md.get_all_pars(free_pars), exp_data.u(0.0), w(0.0)) # NOTE: In case initial conditions are independent of m (independent of w in this case), we could do this outside
    
                # To choose between dist and non-dist cases, the only change is the model md.model_adjoint it seems! So it should have a better name :))))))
                adj_sol, get_Gθ = solve_adj_wrapper(md.model_adjoint, exp_data.u, w, md.get_all_pars(free_pars), N*Ts, Ts, x_func, x2_func, y_func, dy_func, xp0, dx, dx2) # TODO: Add back option to have different Tso????
                get_Gθ(adj_sol)
            end

            function get_gradient_estimate_adj(y::Vector{Float64}, free_pars::Vector{Float64}, isws::Vector{InterSampleWindow}, M_mean::Int=1)
                # --- Generates disturbance signal for the provided parameters ---
                # (Assumes that disturbance model is parametrized, in theory doing this multiple times could be avoided if the disturbance model is known)
                # Zm = [randn(W_meta.nx*W_meta.nv, W_meta.Nw+1) for _ = 1:2M_mean]    # Size of input when disturbance model is discretized before differentiation
                na = length(findall(W_meta.free_par_inds .<= W_meta.nx))   # Number of the disturbance parameters that corresponds to A-matrix. Rest will correspond to C-matrix
                Zm = [randn(W_meta.nx*W_meta.nv*(1+na), W_meta.Nw+1) for _ = 1:2M_mean]   # Size of input when disturbance model is differentiated before discretization
                η = W_meta.get_all_ηs(free_pars[md.dθ+1:end])  # NOTE: Assumes that the disturbance parameters always come after the dynamical parameters.
                dmdl = discretize_ct_noise_model_diff_then_disc(get_ct_disturbance_model(η, W_meta.nx, W_meta.nv), δ, W_meta.free_par_inds)
                wm(m::Int) = if use_exact_interp
                        reset_isws!(isws)
                        XWm = simulate_noise_process(dmdl, Zm)
                        mk_newer_noise_interp(view(η, 1:W_meta.nx), dmdl.Cd, XWm, m, W_meta.nv, δ, isws)
                    else
                        XWm = simulate_noise_process_mangled(dmdl, Zm)
                        mk_noise_interp(dmdl.Cd, XWm, m, δ)
                    end
    
                # --- Runs forward pass and generates output function needed for backward pass ---
                solve_func(m) = solve_wrapper(md.model_nominal, exp_data.u, wm(m), free_pars, N*Ts, Tsλ) |> sol -> h_all_adj(sol,md.get_all_pars(free_pars))
                Xcomp_m = solve_in_parallel_stateout(m -> solve_func(m), 1:2M_mean, md.ny, N)
    
                y_func = get_interpolation(transpose(reshape(y, md.ny, :)), N*Ts, Ts)
                dy_est  = (y[md.ny+1:end,1]-y[1:end-md.ny,1])/Ts
                dy_func = get_interpolation(transpose(reshape(dy_est, md.ny, :)), (N-1)*Ts, Ts)
    
                Statistics.mean(solve_adj_in_parallel(m -> get_one_grad_rel_adj(y_func, dy_func, Xcomp_m[m], Xcomp_m[M_mean+m], free_pars, wm(m)), 1:M_mean, length(free_pars)), dims=2)[:]
            end

        elseif method_type == ADJ_ODEDIST

            function get_one_grad_rel_adjodedist(y_func, dy_func, xvec1, xvec2, free_pars, w, ad::AdjointSDEApproxData)
                x_func  = get_interpolation(xvec1, N*Ts, Tsλ)
                x2_func = get_interpolation(xvec2, N*Ts, Tsλ)
    
                der_est  = get_der_est(0.0:Tsλ:N*Ts, x_func, size(xvec1,2))
                der_est2 = get_der_est(0.0:Tsλ:N*Ts, x2_func, size(xvec1,2))
                # Subtracting Tsλ/2 because sometimes we don't get the right number of elements due to numerical inaccuracies otherwise
                dx =  get_interpolation(der_est,  N*Ts-Tsλ/2, Tsλ)
                dx2 = get_interpolation(der_est2, N*Ts-Tsλ/2, Tsλ)
        
                # ----------------- Actually solving adjoint system ------------------------
                xp0 = md.get_sens_init(md.get_all_pars(free_pars), exp_data.u(0.0), w(0.0)) # NOTE: In case initial conditions are independent of m (independent of w in this case), we could do this outside
    
                adj_sol, get_Gθ = solve_adj_wrapper(md.model_adjoint_odedist, exp_data.u, w, md.get_all_pars(free_pars), N*Ts, Ts, x_func, x2_func, y_func, dy_func, xp0, dx, dx2; ad=ad) # TODO: Add back option to have different Tso????
                get_Gθ(adj_sol)
            end

            function get_gradient_estimate_adjodedist(y::Vector{Float64}, free_pars::Vector{Float64}, isws::Vector{InterSampleWindow}, M_mean::Int=1)
                # --- Generates disturbance signal for the provided parameters ---
                # (Assumes that disturbance model is parametrized, in theory doing this multiple times could be avoided if the disturbance model is known)
                # For the discrete-time model, the input is the same size as the state, so we have nx input dimensions for nv subsystems
                Zm = [randn(W_meta.nx*W_meta.nv, W_meta.Nw+1) for _ = 1:2M_mean]
                η = W_meta.get_all_ηs(free_pars[md.dθ+1:end])  # NOTE: Assumes that the disturbance parameters always come after the dynamical parameters.
                dmdl, Ǎηa, B̌ηa, Čηc, Ǎ = if Ainvertible
                    discretize_ct_noise_model_with_adj_SDEApprox_mats_Ainvertible(get_ct_disturbance_model(η, W_meta.nx, W_meta.nv), δ, W_meta.free_par_inds)
                else
                    discretize_ct_noise_model_with_adj_SDEApprox_mats(get_ct_disturbance_model(η, W_meta.nx, W_meta.nv), δ, W_meta.free_par_inds)
                end
                
                wm(m::Int) = if use_exact_interp
                    reset_isws!(isws)
                    XWm = simulate_noise_process(dmdl, Zm)
                    mk_newer_noise_interp(view(η, 1:W_meta.nx), dmdl.Cd, XWm, m, W_meta.nv, δ, isws)
                else
                    XWm = simulate_noise_process_mangled(dmdl, Zm)
                    mk_noise_interp(dmdl.Cd, XWm, m, δ)
                end
    
                xwm(m::Int) = if use_exact_interp
                    reset_isws!(isws)
                    XWm = simulate_noise_process(dmdl, Zm)
                    mk_newer_noise_interp(view(η, 1:W_meta.nx), Matrix(1.0*I(W_meta.nx*W_meta.nv)), XWm, m, W_meta.nv, δ, isws)
                else
                    XWm = simulate_noise_process_mangled(dmdl, Zm)
                    mk_noise_interp(Matrix(1.0*I(W_meta.nx*W_meta.nv)), XWm, m, δ)
                end

                # vm returns a function that's the zero-order-hold version of the white noise signal
                vm(m::Int) = t -> begin
                    # n*δ <= t <= (n+1)*δ
                    n = Int(t÷δ)
                    Zm[m][:, n+1]  # +1 because t = 0 (and thus n=0) corresponds to index 1
                end
    
                # --- Runs forward pass and generates output function needed for backward pass ---
                solve_func(m) = solve_wrapper(md.model_nominal, exp_data.u, wm(m), free_pars, N*Ts, Tsλ) |> sol -> h_all_adj(sol,md.get_all_pars(free_pars))
                Xcomp_m = solve_in_parallel_stateout(m -> solve_func(m), 1:2M_mean, md.ny, N)
    
                y_func = get_interpolation(transpose(reshape(y, md.ny, :)), N*Ts, Ts)
                dy_est  = (y[md.ny+1:end,1]-y[1:end-md.ny,1])/Ts
                dy_func = get_interpolation(transpose(reshape(dy_est, md.ny, :)), (N-1)*Ts, Ts)
    
                na = length(findall(W_meta.free_par_inds .<= W_meta.nx))   # Number of the disturbance parameters that corresponds to A-matrix. Rest will correspond to C-matrix
                ad(m) = AdjointSDEApproxData(xwm(m), vm(m), Ǎηa, B̌ηa, Čηc, Ǎ, dmdl.Cd, na, W_meta.nx*W_meta.nv, W_meta.nw, length(W_meta.free_par_inds))   # TODO: Copying these matrices is a waste of memory, a reference of some sort would be better

                Statistics.mean(solve_adj_in_parallel(m -> get_one_grad_rel_adjodedist(y_func, dy_func, Xcomp_m[m], Xcomp_m[M_mean+m], free_pars, wm(m), ad(m)), 1:M_mean, length(free_pars)), dims=2)[:]
            end

        else
            throw(DomainError(method_type, "Given method_type does not correspond to any implemented method."))
        end

        opt_pars_proposed = zeros(md.dθ+length(W_meta.free_par_inds), E)
        trace_proposed = [ [Float64[]] for _=1:E]
        trace_gradient = [ [Float64[]] for _=1:E]

        free_dist_inds = findall(get_disturbance_free_pars(W_meta.nx, W_meta.nw, W_meta.nx*W_meta.nv))
        # Concatenates learning rates of parameters of dynamical model and of disturbance model
        init_learning_rate = vcat(md.init_learning_rate, dist_init_learning_rate[free_dist_inds])
        # Concatenates bounds of parameters of dynamical model and of disturbance model
        par_bounds = vcat(md.par_bounds, dist_bounds[free_dist_inds,:])

        for e=1:E
            opt_pars_proposed[:,e], trace_proposed[e], trace_gradient[e] =
                md.minimizer(
                    # (free_pars, M_mean) -> get_gradient_estimate(exp_data.Y[:,e], free_pars, isws, M_mean),       # Forward sens
                    # (free_pars, M_mean) -> get_gradient_adjoint_distsens_new(exp_data.Y[:,e], free_pars, isws, M_mean), # adjoint, sansdist or new dist
                    # (free_pars, M_mean) -> get_gradient_adjoint_distsens_old(exp_data.Y[:,e], free_pars, M_mean),           # adjoint old dist
                    (free_pars, M_mean) -> get_gradient_func(exp_data.Y[:,e], free_pars, isws, M_mean),
                    (t,_)->init_learning_rate./sqrt(t),
                    t->M,
                    pars0,
                    par_bounds;
                    maxiters=maxiters,
                    verbose=verbose)

            @info "Completed for dataset $e for parameters $(opt_pars_proposed[:,e])"
            # === Writing backups ===
            writedlm(joinpath(data_dir, "tmp/backup_proposed_e$e.csv"), opt_pars_proposed[:,e], ',')
            writedlm(joinpath(data_dir, "tmp/backup_trace_e$e.csv"), trace_proposed[e], ',')
            writedlm(joinpath(data_dir, "tmp/backup_gradient_e$e.csv"), trace_gradient[e], ',')
        end

        opt_pars_proposed, trace_proposed, trace_gradient
    end
end

# ===================== DEBUGGING FUNCTIONS =====================
function get_disturbance_from_file(expid::String; M::Int=1, use_exact_interp::Bool=false)::Function
    W_meta_raw, _ =
        readdlm(joinpath(data_dir, expid*"/meta_W.csv"), ',', header=true)
    W_meta = get_disturbance_metadata(W_meta_raw)

    Zm = [randn(W_meta.nx*W_meta.nv, W_meta.Nw+1) for _ = 1:M]
    η = W_meta.η
    dmdl = discretize_ct_noise_model(get_ct_disturbance_model(η, W_meta.nx, W_meta.nv), W_meta.δ)
    wm(m::Int) = if use_exact_interp
            reset_isws!(isws)
            XWm = simulate_noise_process(dmdl, Zm)
            mk_newer_noise_interp(view(η, 1:W_meta.nx), dmdl.Cd, XWm, m, W_meta.nv, W_meta.δ, isws)
        else
            XWm = simulate_noise_process_mangled(dmdl, Zm)
            mk_noise_interp(dmdl.Cd, XWm, m, W_meta.δ)
        end

    wm
end

function get_input_from_file(expid::String)::Function
    U_meta_raw, _ = 
        readdlm(joinpath(data_dir, expid*"/meta_U.csv"), ',', header=true)

    u = if size(U_meta_raw, 2) == 3
        # Input is multisine
        dim = Int(U_meta_raw[1,3])
        # The first column of U_meta_raw is the amplitudes of the multisines while the other is the frequencies
        get_multisines(reshape(U_meta_raw[:,1], :, dim), reshape(U_meta_raw[:,2], :, dim))
    else
        n_u_out = Int(U_meta_raw[1,3])
        δ = U_meta_raw[1,8]
        # Needs to load disturbance metadata, since 
        # Input is from the same model as the disturbance
        input  = readdlm(joinpath(data_dir, expid*"/U.csv"), ',')[:]
        linear_interpolation_multivar(input, δ, n_u_out)(t)
    end
    u
end
