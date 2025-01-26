include("noise_generation.jl")
include("noise_interpolation_multivar.jl")
# include("minimizers.jl")
include("models.jl")
using .NoiseGeneration: DisturbanceMetaData, demangle_XW, get_ct_disturbance_model, discretize_ct_noise_model_with_sensitivities, simulate_noise_process, simulate_noise_process_mangled, discretize_ct_noise_model_with_adj_SDEApprox_mats
using .NoiseInterpolation: InterSampleWindow, initialize_isw, reset_isws!, noise_inter, mk_newer_noise_interp, mk_noise_interp, linear_interpolation_multivar
using .DynamicalModels: AdjointSDEApproxData
using Interpolations: Cubic, BSpline, NoInterp, Line, extrapolate, scale, interpolate, Extrapolation
# import .NoiseGeneration, .NoiseInterpolation
using DelimitedFiles: readdlm, writedlm
using LsqFit: curve_fit, coef
using LinearAlgebra: I
import CSV, Statistics
# For pendulum.jl file. Okay figure out a way to make this all niucer
using .DynamicalModels: pendulum, pendulum_forward_m, pendulum_forward_k, get_pendulum_initial, get_pendulum_initial_msens, get_pendulum_initial_ksens, get_pendulum_initial_distsens
using .DynamicalModels: pendulum_adjoint_m, pendulum_adjoint_k_1dist_ODEdist, pendulum_adjoint_k_1dist, pendulum_forward_k_1dist, Model_ode, Model
include("simulation.jl")

# === DATA TYPES AND CONSTANTS ===
const PENDULUM = 1
const MOH_MDL  = 2
const DELTA    = 3

const FOR_SENS = 1
const ADJ_SENS = 2
const ADJ_ODEDIST = 3

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
model_id::Int = PENDULUM
# Number of monte-carlo simulations used for estimation, e.g. samples of the gradient used to estimate the gradient
# Note that, when two independent estimates are needed, a total of 2M samples is generated, M for each estimate
const M = Threads.nthreads()÷2

# Settings for the InterSampleWindow, only relevant when user_exact_interp in get_experiment_data() is set to true
const W = 100           # Number of intervals for which isw stores data
const Q = 1000          # Number of conditional samples stored per interval

function get_disturbance_free_pars(nx::Int, n_out::Int, n_tot::Int)::Vector{Bool}
    # Use this function to specify which parameters should be free and optimized over
    # Each element represent whether the corresponding element in η is a free parameter
    # Structure: η = vcat(ηa, ηc), where ηa is nx large, and ηc is n_tot*n_out large
    # free_dist_pars = fill(false, nx + n_tot*n_out)                                             # Known disturbance model
    # free_dist_pars = vcat(fill(true, nx), fill(false, n_out), fill(true, (n_tot-1)*n_out))     # Whole a-vector and all but first n_out elements of c-vector unknown (MAXIMUM UNKNOWN PARAMETERS FOR SINGLE DIFFERENTIABILITY (PENDULUM))
    # free_dist_pars = vcat(fill(true, nx), fill(true, n_tot*n_out))                     # All parameters unknown (MAXIMUM UNKNOWN PARAMETERS, NO DIFFERENTIABILITY (DELTA))
    # free_dist_pars = vcat(fill(true, nx), fill(false, n_tot*n_out))                    # Whole a-vector unknown
    free_dist_pars = vcat(true, fill(false, nx-1), fill(false, n_tot*n_out))           # First parameter of a-vector unknown
    # free_dist_pars = vcat(false, true, fill(false, nx-2), fill(false, n_tot*n_out))    # Second parameter of a-vector unknown
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
const abstol = 1e-8#1e-9
const reltol = 1e-5#1e-6
const maxiters_sol = Int64(1e8)
const maxiters_opt = 100

# === HELPER FUNCTIONS TO READ AND WRITE DATA ===
const data_dir = joinpath("../data", "experiments")
exp_path(expid) = joinpath(data_dir, expid)
data_Y_path(expid) = joinpath(exp_path(expid), "Y.csv")

# The type of interpolation used on forward solution in adjoint method
const interp_type = Cubic()

# ====================== MAIN FUNCTIONS ======================

solve_wrapper(mdl_func::Function, u::Function, w::Function, pars::Vector{Float64}, T::Float64, Ts::Float64; kwargs...) = DifferentialEquations.solve(
    begin
        m = mdl_func(u, w, md.get_all_θs(pars), md)
        DifferentialEquations.DAEProblem(m.f!, m.dx0, m.x0, (0.0, T), (), differential_vars=m.dvars)  # TODO: Trying to splat into tuples right now, see if that fixes warning and adress if it does
    end,
    saveat = 0:Ts:T,
    abstol = abstol,
    reltol = reltol,
    maxiters = maxiters_sol;
    kwargs...,
)

# TODO: Are dx and dx2 also interpolated functions? Rename them to match the convention then!!
# Similar to solve_wrapper(), except 1. adjoint models require more input arguments, and 2. the solution is made backwards in time instead of forward, and 3. returns the get_Gp() function too
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
    m, get_Gp = begin
        if isnothing(ad)
            mdl_adj_func(w, md.get_all_θs(pars), T, x_func, x2_func, y_func, dy_func, xp0, dx, dx2)
        else
            mdl_adj_func(w, md.get_all_θs(pars), T, x_func, x2_func, y_func, dy_func, xp0, dx, dx2, ad)
        end
    end
    DifferentialEquations.solve(
        DifferentialEquations.DAEProblem(m.f!, m.dx0, m.x0, (T, 0.0), (), differential_vars=m.dvars),
        saveat = 0:Ts:T,
        abstol = abstol,
        reltol = reltol,
        maxiters = maxiters_sol;
        kwargs...,
    ), get_Gp
end


function get_experiment_data(expid::String; use_exact_interp::Bool = false, E_gen::Integer = 100)::Tuple{ExperimentData, Array{InterSampleWindow, 1}}

    input  = readdlm(joinpath(data_dir, expid*"/U.csv"), ',')[:]
    U_meta_raw, U_meta_names = 
        readdlm(joinpath(data_dir, expid*"/meta_U.csv"), ',', header=true)
    Y_meta_raw, Y_meta_names =
        readdlm(joinpath(data_dir, expid*"/meta_Y.csv"), ',', header=true)
    n_u_out = Int(U_meta_raw[1,3])
    Ts = Y_meta_raw[1,1]
    N = Int(Y_meta_raw[1,2])
    W_meta_raw, W_meta_names =
        readdlm(joinpath(data_dir, expid*"/meta_W.csv"), ',', header=true)  # Used to open meta_W_new.csv, but it's time for new to become the default
    W_meta = get_disturbance_metadata(W_meta_raw)

    u(t::Float64) = linear_interpolation_multivar(input, W_meta.δ, n_u_out)(t)

    # Makes sure there is a tmp-directory, for future use
    if !isdir(joinpath(data_dir, "tmp/"))
        mkdir(joinpath(data_dir, "tmp/"))
    end

    isws::Vector{InterSampleWindow} = if use_exact_interp
        [initialize_isw(Q, W, W_meta.nx*W_meta.n_in, true) for _=1:max(2M,E_gen)]
    else
        Array{InterSampleWindow}(undef, 0)
    end

    # We only want to load the disturbance corresponding to the currently simulated realization of Y,
    # because loading all disturbance data in one go uses needlessly much memory. Therefore, we create
    # a wrapper for loading the needed disturbances and simulating the corresponding Y
    function sim_Y_wrapper(e::Int)::Vector{Float64}
        let nx = W_meta.nx, n_out = W_meta.n_out, n_in = W_meta.n_in, δ = W_meta.δ
            XW = transpose(CSV.read(exp_path(expid)*"/XW_T.csv", CSV.Tables.matrix; header=false, skipto=e, limit=1))
            C_true = reshape(W_meta.η[nx+1:end], (n_out, nx*n_in))
            w = if use_exact_interp
                a_vec = W_meta.η[1:nx]
                mk_newer_noise_interp(a_vec, C_true, demangle_XW(XW, nx*n_in), 1, n_in, δ, isws[e:e])
            else
                mk_noise_interp(C_true, XW, 1, δ)
            end
            # This function call returns an Array of Arrays, outer index is sample index, and each sample is also an array, since the output can be multidimensional
            nested_y = solve_wrapper(md.model_nominal, u, w, md.free_dyn_pars_true, N*Ts, Ts) |> sol -> h_data(sol,md.get_all_θs(md.free_dyn_pars_true))
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
        Y = solve_in_parallel(e -> sim_Y_wrapper(e), 1:E_gen)
        writedlm(data_Y_path(expid), Y, ',')
    end

    return ExperimentData(Y, u, W_meta, Ts), isws

end

function get_disturbance_metadata(W_meta_raw::Matrix{Float64})::DisturbanceMetaData
    nx = Int(W_meta_raw[1,1])
    n_in = Int(W_meta_raw[1,2])
    n_out = Int(W_meta_raw[1,3])
    η_true = W_meta_raw[:,4]
    num_rel = Int(W_meta_raw[1,6])
    Nw = Int(W_meta_raw[1,7])
    δ = Float64(W_meta_raw[1,8])
    n_tot = nx*n_in

    free_dist_pars = get_disturbance_free_pars(nx, n_out, n_tot)
    free_par_inds = findall(free_dist_pars) 
    # Array of tuples containing lower and upper bound for each free disturbance parameter
    # dist_par_bounds = hcat(fill(-Inf, nx+n_tot*n_out, 1), fill(Inf, nx+n_tot*n_out, 1))
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

    DisturbanceMetaData(nx, n_in, n_out, η_true, free_par_inds, free_par_bounds, get_all_ηs, num_rel, Nw, δ)
end

function get_baseline_estimates(pars0::Vector{Float64}, exp_data::ExperimentData; verbose::Bool = true, E_in::Int64=typemax(Int64))

    let N = size(exp_data.Y, 1)÷md.ny-1, E = min(size(exp_data.Y, 2), E_in), dη = length(exp_data.W_meta.η), W_meta = exp_data.W_meta

        opt_pars_baseline = zeros(md.dθ, E)
        trace_baseline = [[pars0] for e=1:E]

        for e=1:E

            # === Setting up functions used by the optimizer ===
            function get_baseline_output(_, free_pars)
                sol = solve_wrapper(md.model_nominal, exp_data.u, t->zeros(W_meta.n_out*(1+dη)), free_pars, N*Ts, Ts)
                Y_base = h(sol, md.get_all_θs(free_pars))
                # Each output sample can be multi-dimensional, vcat(Y_base...) splats all samples into one long vector
                return vcat(Y_base...)
            end

            function get_baseline_sens(_, free_pars)
                sol = solve_wrapper(md.model_sens, exp_data.u, t->zeros(W_meta.n_out*(1+dη)), free_pars, N*Ts, Ts)
                jac = h_sens_base(sol, md.get_all_θs(free_pars))
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

            println("Completed for dataset $e for parameters $(opt_pars_baseline[:,e])")

            writedlm(joinpath(data_dir, "tmp/backup_baseline_e$e.csv"), opt_pars_baseline[:,e], ',')
            writedlm(joinpath(data_dir, "tmp/backup_trace_baseline_e$e.csv"), trace_baseline[e], ',')

        end

        return opt_pars_baseline, trace_baseline
    end
end

# TEMP HELPER FUNCTION, MOVE IT!
# data should have time along rows (dim 1) and dimension along columns (dim 2)
function get_interpolation(data::AbstractMatrix{Float64}, T::Float64, Ts::Float64)
    extrapolate(scale(interpolate(data, (BSpline(interp_type), NoInterp())), 0.0:Ts:T, 1:size(data,2)), Line())
end

# TODO: Figure out what to do with this function!!!
function get_der_est2(ts, func, dim)
    der_est = zeros(length(ts)-1, dim)
    for (i,t) = enumerate(ts)
        if i > 1
            der_est[i-1,:] = (func(t,1:dim)-func(ts[i-1],1:dim))./(t-ts[i-1])
        end
    end
    return der_est
end

# TODO: RENAME ALL THE FUNCTIONS INSIDE FOR MORE INTUITIVE NAMES!!!!!

function get_proposed_estimates(pars0::Vector{Float64}, exp_data::ExperimentData, isws::Vector{InterSampleWindow}; 
    use_exact_interp::Bool = false, maxiters::Int64 = maxiters_opt, verbose::Bool = true, E_in::Int64=typemax(Int64), method_type::Int64 = FOR_SENS)
    
    # Tsλ is the sampling period of the forward solution that is then used for the backwards computation for the adjoint method, good to have smaller for better interpolation
    let N = size(exp_data.Y, 1)÷md.ny-1, E = min(size(exp_data.Y, 2), E_in), W_meta = exp_data.W_meta, δ = W_meta.δ, Ts = exp_data.Ts, Tsλ = exp_data.Ts/10

        # Depending on method, the function used for computing cost function gradient estimate differs a lot
        get_gradient_func = if method_type == FOR_SENS

            # Helper function to improve readability. Instead of calling solve_wrapper with lots of arguments and then transforming
            # the solution to samples of the output, this function can be called instead
            function solve_with_sens_func(w_func::Function, pars::Vector{Float64})
                sol = solve_wrapper(md.model_sens, exp_data.u, w_func, pars, N*Ts, Ts)
                h_comp(sol, md.get_all_θs(pars))
            end

            # Returns estimate of gradient of cost function
            # M_mean specifies over how many realizations the gradient estimate is computed
            function get_gradient_estimate(y::Vector{Float64}, free_pars::Vector{Float64}, isws::Vector{InterSampleWindow}, M_mean::Int=1)
                # --- Generates disturbance signal for the provided parameters ---
                # (Assumes that disturbance model is parametrized, in theory doing this multiple times could be avoided if the disturbance model is known)
                Zm = [randn(W_meta.Nw+1, W_meta.nx*W_meta.n_in) for _ = 1:2M_mean]
                η = W_meta.get_all_ηs(free_pars[md.dθ+1:end])  # NOTE: Assumes that the disturbance parameters always come after the dynamical parameters.
                dmdl = discretize_ct_noise_model_with_sensitivities(get_ct_disturbance_model(η, W_meta.nx, W_meta.n_out), δ, W_meta.free_par_inds)
                wmm(m::Int) = if use_exact_interp
                        reset_isws!(isws)
                        XWm = simulate_noise_process(dmdl, Zm)
                        mk_newer_noise_interp(view(η, 1:W_meta.nx), dmdl.Cd, XWm, m, W_meta.n_in, δ, isws)
                    else
                        XWm = simulate_noise_process_mangled(dmdl, Zm)
                        mk_noise_interp(dmdl.Cd, XWm, m, δ)
                    end

                # 2M_mean solutions computed because the used realizations of Ym and jacsYm have to be independent
                # (For computing the Ym, a smaller DAE could be solved instead, but this is not currently implemented)
                Ym, jacsYm = solve_in_parallel_sens(m->solve_with_sens_func(wmm(m),free_pars), 1:2M_mean)

                # --- Computes cost function gradient ---
                # Y.-Ym is a matrix with as many columns as Ym, where column i contains Y-Ym[:,i]
                # Taking the mean of that gives us the average error as a function of time over all realizations contained in Ym.
                # mean(-jacsYm) is the average (over all m) jacobian of Ym.
                # Different rows correspond to different t, while different columns correspond to different parameters
                tmp = (2/N) * (transpose(Statistics.mean(-jacsYm))*Statistics.mean(y.-Ym, dims=2))[:]
                tmp
            end

        elseif method_type == ADJ_SENS || (method_type == ADJ_ODEDIST && length(W_meta.free_par_inds)==0)
            
            function compute_Gp_adj_dist_sens_new(y_func, dy_func, xvec1::Matrix{Float64}, xvec2::Matrix{Float64}, free_pars::Vector{Float64}, w::Function)
                x_func  = get_interpolation(xvec1, N*Ts, Tsλ)
                x2_func = get_interpolation(xvec2, N*Ts, Tsλ)
    
                der_est  = get_der_est2(0.0:Tsλ:N*Ts, x_func, size(xvec1,2)) # TODO: Rename this if I never end up using get_der_est() without the 2
                der_est2 = get_der_est2(0.0:Tsλ:N*Ts, x2_func, size(xvec1,2))
                # Subtracting Tsλ/2 because sometimes we don't get the right number of elements due to numerical inaccuracies otherwise
                dx =  get_interpolation(der_est,  N*Ts-Tsλ/2, Tsλ)
                dx2 = get_interpolation(der_est2, N*Ts-Tsλ/2, Tsλ)
        
        
                # ----------------- Actually solving adjoint system ------------------------
                xp0 = md.get_sens_init(md.get_all_θs(free_pars), exp_data.u(0.0), w(0.0)) # NOTE: In case initial conditions are independent of m (independent of w in this case), we could do this outside
    
                # To choose between dist and non-dist cases, the only change is the model md.model_adjoint it seems! So it should have a better name :))))))
                adj_sol, get_Gp = solve_adj_wrapper(md.model_adjoint, exp_data.u, w, md.get_all_θs(free_pars), N*Ts, Ts, x_func, x2_func, y_func, dy_func, xp0, dx, dx2) # TODO: Add back option to have different Tso????
                get_Gp(adj_sol)
            end

            function get_gradient_adjoint_distsens_new(y::Vector{Float64}, free_pars::Vector{Float64}, isws::Vector{InterSampleWindow}, M_mean::Int=1)
                compute_Gp = compute_Gp_adj_dist_sens_new       # DEBUG: I'm not sure if we'll need several of these, probably yes. I can just define other function first, then it will exist here.
                # --- Generates disturbance signal for the provided parameters ---
                # (Assumes that disturbance model is parametrized, in theory doing this multiple times could be avoided if the disturbance model is known)
                Zm = [randn(W_meta.Nw+1, W_meta.nx*W_meta.n_in) for _ = 1:2M_mean]
                η = W_meta.get_all_ηs(free_pars[md.dθ+1:end])  # NOTE: Assumes that the disturbance parameters always come after the dynamical parameters.
                dmdl = discretize_ct_noise_model_with_sensitivities(get_ct_disturbance_model(η, W_meta.nx, W_meta.n_out), δ, W_meta.free_par_inds)
                wm(m::Int) = if use_exact_interp
                        reset_isws!(isws)
                        XWm = simulate_noise_process(dmdl, Zm)
                        mk_newer_noise_interp(view(η, 1:W_meta.nx), dmdl.Cd, XWm, m, W_meta.n_in, δ, isws)
                    else
                        XWm = simulate_noise_process_mangled(dmdl, Zm)
                        mk_noise_interp(dmdl.Cd, XWm, m, δ)
                    end
    
                # --- Runs forward pass and generates output function needed for backward pass ---
                solve_func(m) = solve_wrapper(md.model_nominal, exp_data.u, wm(m), free_pars, N*Ts, Tsλ) |> sol -> h_all_adj(sol,md.get_all_θs(free_pars))
                Xcomp_m = solve_in_parallel_debug_new(m -> solve_func(m), 1:2M_mean, md.ny, N)  # TODO: Get rid of unused simulation functions, potentially rename this one to remove _new
    
                y_func = get_interpolation(transpose(reshape(y, md.ny, :)), N*Ts, Ts)
                dy_est  = (y[md.ny+1:end,1]-y[1:end-md.ny,1])/Ts
                dy_func = get_interpolation(transpose(reshape(dy_est, md.ny, :)), (N-1)*Ts, Ts)
    
                Statistics.mean(solve_adj_in_parallel(m -> compute_Gp(y_func, dy_func, Xcomp_m[m], Xcomp_m[M_mean+m], free_pars, wm(m)), 1:M_mean, length(free_pars)), dims=2)[:]
            end

        elseif method_type == ADJ_ODEDIST

            function compute_Gp_adj_dist_sens_old(y_func, dy_func, xvec1, xvec2, free_pars, w, ad::AdjointSDEApproxData)
                x_func  = get_interpolation(xvec1, N*Ts, Tsλ)
                x2_func = get_interpolation(xvec2, N*Ts, Tsλ)
    
                der_est  = get_der_est2(0.0:Tsλ:N*Ts, x_func, size(xvec1,2)) # TODO: Rename this if I never end up using get_der_est() without the 2
                der_est2 = get_der_est2(0.0:Tsλ:N*Ts, x2_func, size(xvec1,2))
                # Subtracting Tsλ/2 because sometimes we don't get the right number of elements due to numerical inaccuracies otherwise
                dx =  get_interpolation(der_est,  N*Ts-Tsλ/2, Tsλ)
                dx2 = get_interpolation(der_est2, N*Ts-Tsλ/2, Tsλ)
        
        
                # ----------------- Actually solving adjoint system ------------------------
                xp0 = md.get_sens_init(md.get_all_θs(free_pars), exp_data.u(0.0), w(0.0)) # NOTE: In case initial conditions are independent of m (independent of w in this case), we could do this outside
    
                adj_sol, get_Gp = solve_adj_wrapper(md.model_adjoint_odedist, exp_data.u, w, md.get_all_θs(free_pars), N*Ts, Ts, x_func, x2_func, y_func, dy_func, xp0, dx, dx2; ad=ad) # TODO: Add back option to have different Tso????
                get_Gp(adj_sol)
            end

            function get_gradient_adjoint_distsens_old(y::Vector{Float64}, free_pars::Vector{Float64}, isws::Vector{InterSampleWindow}, M_mean::Int=1)
                compute_Gp = compute_Gp_adj_dist_sens_old       # DEBUG: I'm not sure if we'll need several of these, probably yes. I can just define other function first, then it will exist here.
                # --- Generates disturbance signal for the provided parameters ---
                # (Assumes that disturbance model is parametrized, in theory doing this multiple times could be avoided if the disturbance model is known)
                Zm = [randn(W_meta.Nw+1, W_meta.nx*W_meta.n_in) for _ = 1:2M_mean]
                η = W_meta.get_all_ηs(free_pars[md.dθ+1:end])  # NOTE: Assumes that the disturbance parameters always come after the dynamical parameters.
                dmdl, Ǎη, B̌η, Čη = discretize_ct_noise_model_with_adj_SDEApprox_mats(get_ct_disturbance_model(η, W_meta.nx, W_meta.n_out), δ, W_meta.free_par_inds)
                # dmdl, B̌, B̌ηa = discretize_ct_noise_model_with_sensitivities_for_adj(get_ct_disturbance_model(η, W_meta.nx, W_meta.n_out), δ, W_meta.free_par_inds)
                wm(m::Int) = if use_exact_interp
                    reset_isws!(isws)
                    XWm = simulate_noise_process(dmdl, Zm)
                    mk_newer_noise_interp(view(η, 1:W_meta.nx), dmdl.Cd, XWm, m, W_meta.n_in, δ, isws)
                else
                    XWm = simulate_noise_process_mangled(dmdl, Zm)
                    mk_noise_interp(dmdl.Cd, XWm, m, δ)
                end
    
                xwm(m::Int) = if use_exact_interp
                    reset_isws!(isws)
                    XWm = simulate_noise_process(dmdl, Zm)
                    mk_newer_noise_interp(view(η, 1:W_meta.nx), Matrix(1.0*I(W_meta.nx*W_meta.n_in)), XWm, m, W_meta.n_in, δ, isws)
                else
                    XWm = simulate_noise_process_mangled(dmdl, Zm)
                    mk_noise_interp(Matrix(1.0*I(W_meta.nx*W_meta.n_in)), XWm, m, δ)
                end
    
                # vm returns a function that's the zero-order-hold version of the white noise signal
                vm(m::Int) = t -> begin
                    # n*δ <= t <= (n+1)*δ
                    n = Int(t÷δ)
                    Zm[m][n+1, :]  # +1 because t = 0 (and thus n=0) corresponds to index 1
                end
    
                # --- Runs forward pass and generates output function needed for backward pass ---
                solve_func(m) = solve_wrapper(md.model_nominal, exp_data.u, wm(m), free_pars, N*Ts, Tsλ) |> sol -> h_all_adj(sol,md.get_all_θs(free_pars))
                Xcomp_m = solve_in_parallel_debug_new(m -> solve_func(m), 1:2M_mean, md.ny, N)  # TODO: Get rid of unused simulation functions, potentially rename this one to remove _new
    
                y_func = get_interpolation(transpose(reshape(y, md.ny, :)), N*Ts, Ts)
                dy_est  = (y[md.ny+1:end,1]-y[1:end-md.ny,1])/Ts
                dy_func = get_interpolation(transpose(reshape(dy_est, md.ny, :)), (N-1)*Ts, Ts)
    
                na = length(findall(W_meta.free_par_inds .<= W_meta.nx))   # Number of the disturbance parameters that corresponds to A-matrix. Rest will correspond to C-matrix
                ad(m) = AdjointSDEApproxData(xwm(m), vm(m), Ǎη, B̌η, Čη, dmdl.Cd, η, na, W_meta.nx*W_meta.n_in, length(W_meta.free_par_inds))   # TODO: Copying these matrices is a waste of space, a reference of some sort would be better
    
                Statistics.mean(solve_adj_in_parallel(m -> compute_Gp(y_func, dy_func, Xcomp_m[m], Xcomp_m[M_mean+m], free_pars, wm(m), ad(m)), 1:M_mean, length(free_pars)), dims=2)[:]
            end

        else
            throw(DomainError(method_type, "Given method_type does not correspond to any implemented method."))
        end

        opt_pars_proposed = zeros(md.dθ+length(W_meta.free_par_inds), E)
        trace_proposed = [ [Float64[]] for _=1:E]
        trace_gradient = [ [Float64[]] for _=1:E]

        for e=1:E
            opt_pars_proposed[:,e], trace_proposed[e], trace_gradient[e] =
                md.minimizer(
                    # (free_pars, M_mean) -> get_gradient_estimate(exp_data.Y[:,e], free_pars, isws, M_mean),       # Forward sens
                    # (free_pars, M_mean) -> get_gradient_adjoint_distsens_new(exp_data.Y[:,e], free_pars, isws, M_mean), # adjoint, sansdist or new dist
                    # (free_pars, M_mean) -> get_gradient_adjoint_distsens_old(exp_data.Y[:,e], free_pars, M_mean),           # adjoint old dist
                    (free_pars, M_mean) -> get_gradient_func(exp_data.Y[:,e], free_pars, isws, M_mean),
                    (t,_)->md.init_learning_rate./sqrt(t),
                    t->M,
                    pars0,
                    md.par_bounds;
                    maxiters=maxiters,
                    verbose=verbose)
            println("Completed for dataset $e for parameters $(opt_pars_proposed[:,e])")
        end

        opt_pars_proposed, trace_proposed, trace_gradient
    end
end


