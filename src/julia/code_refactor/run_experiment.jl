include("noise_generation.jl")
include("noise_interpolation_multivar.jl")
include("simulation.jl")
include("minimizers.jl")
using .NoiseGeneration: DisturbanceMetaData, demangle_XW, get_ct_disturbance_model, discretize_ct_noise_model_with_sensitivities, simulate_noise_process, simulate_noise_process_mangled
using .NoiseInterpolation: InterSampleWindow, initialize_isw, reset_isws!, noise_inter, mk_newer_noise_interp, mk_noise_interp, linear_interpolation_multivar
# import .NoiseGeneration, .NoiseInterpolation
using DelimitedFiles: readdlm, writedlm
using LsqFit: curve_fit, coef
import CSV, Statistics

# === DATA TYPES AND CONSTANTS ===
const PENDULUM = 1
const MOH_MDL  = 2
const DELTA    = 3

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

function get_disturbance_free_pars(η_true::Vector{Float64}, nx::Int, n_out::Int, n_tot::Int)::Vector{Bool}
    # Use this function to specify which parameters should be free and optimized over
    # Each element represent whether the corresponding element in η is a free parameter
    # Structure: η = vcat(ηa, ηc), where ηa is nx large, and ηc is n_tot*n_out large
    free_dist_pars = fill(false, size(η_true))                                                  # Known disturbance model
    # free_dist_pars = vcat(fill(true, nx), fill(false, n_out), fill(true, (n_tot-1)*n_out))    # Whole a-vector and all but first n_out elements of c-vector unknown (MAXIMUM UNKNOWN PARAMETERS FOR SINGLE DIFFERENTIABILITY (PENDULUM))
    # free_dist_pars = vcat(fill(true, nx), fill(true, n_tot*n_out))                     # All parameters unknown (MAXIMUM UNKNOWN PARAMETERS, NO DIFFERENTIABILITY (DELTA))
    # free_dist_pars = vcat(fill(true, nx), fill(false, n_tot*n_out))                    # Whole a-vector unknown
    # free_dist_pars = vcat(true, fill(false, nx-1), fill(false, n_tot*n_out))           # First parameter of a-vector unknown
    # free_dist_pars = vcat(false, true, fill(false, nx-2), fill(false, n_tot*n_out))    # Second parameter of a-vector unknown
end
# ===================================================================================================
# ========================= End of values that should be set by user ================================
# ===================================================================================================

# === LOADING MODEL AND GENERATING RELATED FUNCTIONS ===
if model_id == PENDULUM
    include("model_metadata/pendulum.jl")
    mdl = pend_model
end

h_data(sol,θ) = apply_outputfun(x -> mdl.f(x,θ) .+ mdl.σ * randn(size(mdl.f(x,θ))), sol) # Output of "true" system, including measurement noise
h(sol,θ) = apply_outputfun(x->mdl.f(x,θ), sol)                                               # Output of model, no measurement noise 
h_comp(sol,θ) = apply_two_outputfun(x->mdl.f(x,θ), x->mdl.f_sens(x,θ), sol)           # for complete model with dynamics sensitivity
h_sens_base(sol,θ) = apply_outputfun(x->mdl.f_sens_baseline(x,θ), sol)

# === SOLVER PARAMETERS ===
const abstol = 1e-8#1e-9
const reltol = 1e-5#1e-6
const maxiters_sol = Int64(1e8)
const maxiters_opt = 100

# === HELPER FUNCTIONS TO READ AND WRITE DATA ===
const data_dir = joinpath("../data", "experiments")
exp_path(expid) = joinpath(data_dir, expid)
data_Y_path(expid) = joinpath(exp_path(expid), "Y.csv")

# ====================== MAIN FUNCTIONS ======================

# NOTE: It is this call to DifferentialEquations.solve() that causes the performance warning 
# "Using arrays or dicts to store parameters of different types can hurt performance. │ Consider using tuples instead."
# It's not obvious to me exactly what causes it or how to solve it, so I'm gonna let it be for now.
solve_wrapper(mdl_func::Function, u::Function, w::Function, pars::Vector{Float64}, N::Int, Ts::Float64; kwargs...) = DifferentialEquations.solve(
    begin
        m = mdl_func(u, w, mdl.get_all_θs(pars), mdl)
        DifferentialEquations.DAEProblem(m.f!, m.dx0, m.x0, (0, N*Ts), [], differential_vars=m.dvars)
    end,
    # problem(
    #     mdl_func(φ0, u, w, get_all_θs(pars)),
    #     N,
    #     Ts,
    # ),
    saveat = 0:Ts:N*Ts,
    abstol = abstol,
    reltol = reltol,
    maxiters = maxiters_sol;
    kwargs...,
)

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
                mk_newer_noise_interp(a_vec, C_true, demangle_XW(XW, nx*W.n_in), 1, n_in, δ, isws[e:e])
            else
                mk_noise_interp(C_true, XW, 1, δ)
            end
            # This function call returns an Array of Arrays, outer index is sample index, and each sample is also an array, since the output can be multidimensional
            nested_y = solve_wrapper(mdl.model_nominal, u, w, mdl.free_dyn_pars_true, N, Ts) |> sol -> h_data(sol,mdl.get_all_θs(mdl.free_dyn_pars_true))
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

    free_dist_pars = get_disturbance_free_pars(η_true, nx, n_out, n_tot)
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
        all_η[free_par_inds] = free_dist_pars
        return all_η
    end

    DisturbanceMetaData(nx, n_in, n_out, η_true, free_par_inds, free_par_bounds, get_all_ηs, num_rel, Nw, δ)
end

function get_baseline_estimates(pars0::Vector{Float64}, exp_data::ExperimentData; verbose::Bool = true)

    let N = size(exp_data.Y, 1)-1, E = size(exp_data.Y, 2), dη = length(exp_data.W_meta.η), W_meta = exp_data.W_meta

        opt_pars_baseline = zeros(mdl.dθ, E)
        trace_baseline = [[pars0] for e=1:E]

        for e=1:E

            # === Setting up functions used by the optimizer ===
            function get_baseline_output(_, free_pars)
                sol = solve_wrapper(mdl.model_nominal, exp_data.u, t->zeros(W_meta.n_out*(1+dη)), free_pars, N, exp_data.Ts)
                Y_base = h(sol, mdl.get_all_θs(free_pars))
                # Each output sample can be multi-dimensional, vcat(Y_base...) splats all samples into one long vector
                return vcat(Y_base...)
            end

            function get_baseline_sens(_, free_pars)
                sol = solve_wrapper(mdl.model_sens, exp_data.u, t->zeros(W_meta.n_out*(1+dη)), free_pars, N, exp_data.Ts)
                jac = h_sens_base(sol, mdl.get_all_θs(free_pars))
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
                lower=mdl.par_bounds[:,1], 
                upper=mdl.par_bounds[:,2], 
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

function get_proposed_estimates(pars0::Vector{Float64}, exp_data::ExperimentData, isws::Vector{InterSampleWindow}; use_exact_interp::Bool = false, maxiters::Int64 = maxiters_opt, verbose::Bool = true, E_in::Int64=typemax(Int64))
    
    let N = size(exp_data.Y, 1)-1, E = min(size(exp_data.Y, 2), E_in), W_meta = exp_data.W_meta, δ = W_meta.δ

        # Generates white noise that will be used for generating disturbances
        Zm = [randn(W_meta.Nw+1, W_meta.nx*W_meta.n_in) for _ = 1:2M]

        # Helper function to improve readability. Instead of calling solve_wrapper with lots of arguments and then transforming
        # the solution to samples of the output, this function can be called instead
        function solve_with_sens_func(w_func::Function, pars::Vector{Float64})
            sol = solve_wrapper(mdl.model_sens, exp_data.u, w_func, pars, N, exp_data.Ts)
            h_comp(sol, mdl.get_all_θs(pars))
        end

        # Returns estimate of gradient of cost function
        # M_mean specifies over how many realizations the gradient estimate is computed
        function get_gradient_estimate(y, free_pars, isws, M_mean::Int=1)
            # --- Generates disturbance signal for the provided parameters ---
            # (Assumes that disturbance model is parametrized, in theory doing this multiple times could be avoided if the disturbance model is known)
            η = W_meta.get_all_ηs(free_pars[mdl.dθ+1:end])  # NOTE: Assumes that the disturbance parameters always come after the dynamical parameters.
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
            (2/N) * (transpose(Statistics.mean(-jacsYm))*Statistics.mean(y.-Ym, dims=2))[:]
        end

        opt_pars_proposed = zeros(mdl.dθ, E)
        trace_proposed = [ [Float64[]] for _=1:E]
        trace_gradient = [ [Float64[]] for _=1:E]

        for e=1:E
            opt_pars_proposed[:,e], trace_proposed[e], trace_gradient[e] =
                mdl.minimizer(
                    (free_pars, M_mean) -> get_gradient_estimate(exp_data.Y[:,e], free_pars, isws, M_mean),
                    (t,_)->mdl.init_learning_rate./sqrt(t),
                    t->M,
                    pars0,
                    mdl.par_bounds;
                    maxiters=maxiters,
                    verbose=verbose)
            println("Completed for dataset $e for parameters $(opt_pars_proposed[:,e])")
        end

        opt_pars_proposed, trace_proposed, trace_gradient
    end
end


