include("noise_generation.jl")
include("noise_interpolation_multivar.jl")
include("simulation.jl")
using .NoiseGeneration: DisturbanceMetaData
using .NoiseInterpolation: InterSampleWindow, initialize_isw, noise_inter
# import .NoiseGeneration, .NoiseInterpolation
using DelimitedFiles: readdlm, writedlm
import CSV

# === USER SETTINGS AND RELATED ===
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
end

# --------------- To be changed by user --------------------        # THESE COMMENTS ARE ODD AND UGLY, FIX THAT!
# Selects which model to use/simulate
model_id = PENDULUM
const M = Threads.nthreads()÷2          # Number of monte-carlo simulations used for estimating mean

# SOME OPTIONS, MAYBE MOVE TO BETTER PLACE???
const W = 100           # Number of intervals for which isw stores data
const Q = 1000          # Number of conditional samples stored per interval
# ----------------------------------------------------------

if model_id == PENDULUM
    include("model_metadata/pendulum.jl")
    mdl = pend_model
end

h_data(sol,θ) = apply_outputfun(x -> mdl.f(x,θ) .+ mdl.σ * randn(size(mdl.f(x,θ))), sol)

demangle_XW(XW::AbstractMatrix{Float64}, n_tot::Int) = [XW[(i-1)*n_tot+1:i*n_tot, m] for i=1:(size(XW,1)÷n_tot), m=1:size(XW,2)]

# === SOLVER PARAMETERS ===
const abstol = 1e-8#1e-9
const reltol = 1e-5#1e-6
const maxiters = Int64(1e8)

# === HELPER FUNCTIONS TO READ AND WRITE DATA ===
const data_dir = joinpath("../data", "experiments")
exp_path(expid) = joinpath(data_dir, expid)
data_Y_path(expid) = joinpath(exp_path(expid), "Y.csv")

function linear_interpolation_multivar(y::AbstractVector, Ts::Float64, ny::Int)
    max_n = length(y)÷ny-2
    function y_func(t::Float64)
        n = min(Int(t÷Ts), max_n)
        return ( ((n+1)*Ts-t)*y[n*ny+1:(n+1)*ny] .+ (t-n*Ts)*y[(n+1)*ny+1:(n+2)*ny])./Ts
    end
end

# TO BE EDITED BY USER
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
    maxiters = maxiters;
    kwargs...,
)

# Function for using conditional interpolation
function mk_newer_noise_interp(a_vec::AbstractVector{Float64},
                                C::Matrix{Float64},
                                XW::Matrix{Vector{Float64}},
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

function mk_noise_interp(C::Matrix{Float64},
    XW::AbstractMatrix{Float64},
    m::Int,
    δ::Float64)
    let
        n_tot = size(C, 2)
        n_max = size(XW,1)÷n_tot-1
        function xw(t::Float64)
            # n*δ <= t <= (n+1)*δ
            n = Int(t÷δ)
            if n >= n_max
                # The disturbance only has samples from t=0.0 to t=N*Ts-δ. 
                # The requested t was large enough to t=N*Ts that we must return last sample of disturbance instead of interpolating
                return C*XW[end-n_tot+1:end, m]
            else
                # row of x_1(t_n) in XW is given by k. Note that t=0 is given by row 1
                k = n * n_tot + 1

                xl = XW[k:(k + n_tot - 1), m]
                xu = XW[(k + n_tot):(k + 2n_tot - 1), m]
                return C*(xl + (t-n*δ)*(xu-xl)/δ)
            end
        end
    end
end

function get_experiment_data(expid::String; use_exact_interp::Bool = false, E_gen::Integer = 100)::Tuple{ExperimentData, Array{InterSampleWindow, 1}}

    input  = readdlm(joinpath(data_dir, expid*"/U.csv"), ',')[:]
    U_meta_raw, U_meta_names = 
        readdlm(joinpath(data_dir, expid*"/meta_U.csv"), ',', header=true)
    Y_meta_raw, Y_meta_names =
        readdlm(joinpath(data_dir, expid*"/meta_Y.csv"), ',', header=true)
    n_u_out = Int(U_meta_raw[1,3])
    u(t::Float64) = linear_interpolation_multivar(input, δ, n_u_out)(t)             # GOTTA GET δ FROM SOMEWHERE, PROBABLY READ IT FROM FILE!!!!!
    Ts = Y_meta_raw[1,1]
    δ = 0.1Ts
    N = Int(Y_meta_raw[1,2])
    W_meta_raw, W_meta_names =
        readdlm(joinpath(data_dir, expid*"/meta_W_new.csv"), ',', header=true)
    W_meta = get_disturbance_metadata(W_meta_raw)

    isws::Vector{InterSampleWindow} = if use_exact_interp
        [initialize_isw(Q, W, W_meta.nx*W_meta.n_in, true) for e=1:M]       # TODO: Does M still make sense? I think so right? Or should this be 2M???
    else
        Array{InterSampleWindow}(undef, 0)
    end

    # We only want to load the disturbance corresponding to the currently simulated realization of Y,
    # because loading all disturbance data in one go uses needlessly much memory. Therefore, we create
    # a wrapper for loading the needed disturbances and simulating the corresponding Y
    function sim_Y_wrapper(e::Int)::Vector{Float64}                 # TODO: Add let-statement for nice readability!let nx and so on
        XW = transpose(CSV.read(exp_path(expid)*"/XW_T.csv", CSV.Tables.matrix; header=false, skipto=e, limit=1))
        C_true = reshape(W_meta.η[W_meta.nx+1:end], (W_meta.n_out, W_meta.nx*W_meta.n_in))
        w = if use_exact_interp
            a_vec = W_meta.η[1:W_meta.nx]
            mk_newer_noise_interp(a_vec, C_true, demangle_XW(XW, W_meta.nx*W_meta.n_in), 1, W_meta.n_in, δ, isws[e:e])
        else
            mk_noise_interp(C_true, XW, 1, δ)
        end
        # This function call returns an Array of Arrays, outer index is sample index, and each sample is also an array, since the output can be multidimensional
        nested_y = solve_wrapper(mdl.model_nominal, u, w, mdl.free_dyn_pars_true, N, Ts) |> sol -> h_data(sol,mdl.get_all_θs(mdl.free_dyn_pars_true))
        # The output is collapsed into a single long Array before returning
        return vcat(nested_y...)
    end

    # ------------- Loading/generate "true" system output ---------------
    if isfile(data_Y_path(expid))
        Y = readdlm(data_Y_path(expid), ',')
    else
        Y = solve_in_parallel(e -> sim_Y_wrapper(e), 1:E_gen)
        writedlm(data_Y_path(expid), Y, ',')
    end

    return ExperimentData(Y, u, W_meta), isws

end

function get_disturbance_metadata(W_meta_raw::Matrix{Float64})::DisturbanceMetaData
    nx = Int(W_meta_raw[1,1])
    n_in = Int(W_meta_raw[1,2])
    n_out = Int(W_meta_raw[1,3])
    η_true = W_meta_raw[:,4]
    num_rel = Int(W_meta_raw[1,6])
    Nw = Int(W_meta_raw[1,7])
    n_tot = nx*n_in

    free_dist_pars = get_disturbance_free_pars(η_true, nx, n_out, n_tot)
    free_par_inds = findall(free_dist_pars) 
    # Array of tuples containing lower and upper bound for each free disturbance parameter
    # dist_par_bounds = hcat(fill(-Inf, nx+n_tot*n_out, 1), fill(Inf, nx+n_tot*n_out, 1))
    free_par_bounds = hcat(fill(-Inf, length(free_par_inds), 1), fill(Inf, length(free_par_inds), 1))
    function get_all_ηs(free_pars::Vector{Float64})
        # If copy() is not used here, some funky stuff that I don't fully understand happens.
        # I think essentially η_true stops being defined after function returns, so
        # setting all_η to its value doesn't behave quite as I expected
        all_η = copy(η_true)
        # Fetches user-provided values for free disturbance parameters only
        all_η[free_par_inds] = free_pars[num_dyn_pars+1:end]
        return all_η
    end

    DisturbanceMetaData(nx, n_in, n_out, η_true, free_par_inds, free_par_bounds, get_all_ηs, num_rel, Nw)
end