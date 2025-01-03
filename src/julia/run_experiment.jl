using LsqFit, LaTeXStrings, Dates, Interpolations, DelimitedFiles
using StatsPlots # Commented out since it breaks 3d-plotting in Julia 1.5.3
using QuadGK
include("simulation.jl")
include("models.jl")
include("noise_interpolation_multivar.jl")
include("noise_generation.jl")
include("minimizers.jl")

seed = 1234
Random.seed!(seed)

struct ExperimentData
    # Y is the measured output of system, contains N+1 rows and E columns
    # (+1 because of an additional sample at t=0)
    Y::Matrix{Float64}
    # u is the function specifying the input of the system
    u::Function
    # get_all_ηs encodes what information of the disturbance model is known
    # This function should always return all parameters of the disturbance model,
    # given only the free parameters
    get_all_ηs::Function
    # Array containing lower and upper bound of a disturbance parameter in each row
    dist_par_bounds::Matrix{Float64}
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
    λ_func::Function
    dλ_func::Function
    u::Function
    sols::Array{DAESolution,1}
    N::Int
    Ts::Float64 # TODO: No need to pass this I think, Ts already global?
    p::Vector{Float64}
    xp0::Matrix{Float64}
end

const PENDULUM = 1
const MOH_MDL  = 2
const DELTA    = 3

# Selects which model to adapt code to
model_id = PENDULUM     # Remember that δ and T_s might depend on the model, I used different ones for Pendulum and for Delta robot

# ==================
# === PARAMETERS ===
# ==================

# === TIME ===
# The relationship between number of data samples N and number of noise samples
# Nw is given by Nw >= (Ts/δ)*N
const δ = 0.01#2e-5#0.01                  # noise sampling time
const Ts = 0.1#2e-4#0.1                  # step-size
const Tsλ = 0.01#2e-5#0.01
const Tso = 0.01#2e-5#0.01
# const δ = 0.001                  # noise sampling time
# const Ts = 0.001                  # step-size
const M = Threads.nthreads()÷2#4#00       # Number of monte-carlo simulations used for estimating mean
# TODO: Surely we don't need to collect these, a range should work just as well?
const ms = collect(1:M)
const W = 100           # Number of intervals for which isw stores data
const Q = 1000          # Number of conditional samples stored per interval
const interp_type = Cubic()


M_rate_max = M#min(4, M)#100#8#4#16   
# max_allowed_step = 1.0  # Maximum magnitude of step that SGD is allowed to take
# M_rate(t) specifies over how many realizations the output jacobian estimate
# should be computed at iteration t. NOTE: A total of 2*M_rate(t) iterations
# will be performed for estimating the gradient of the cost functions
# @warn "USING TIME VARYING MRATE NOW"
# M_rate(t::Int) = (t÷50+1)*M_rate_max
M_rate(t::Int) = M_rate_max

mangle_XWs(XWs::Matrix{Vector{Float64}}) = hcat([vcat(XWs[:, m]...) for m = 1:size(XWs,2)]...)
demangle_XW(XW::AbstractMatrix{Float64}, n_tot::Int) = [XW[(i-1)*n_tot+1:i*n_tot, m] for i=1:(size(XW,1)÷n_tot), m=1:size(XW,2)]

function deb_info(obj)
    @info "type: $(typeof(obj)), size: $(size(obj)), inner size: $(size(obj[1]))"
end

# === MODEL (AND DATA) PARAMETERS ===
const σ = 0.002                 # measurement noise variance

const m = 0.3                   # [kg]
const L = 6.25                  # [m], gives period T = 5s (T ≅ 2√L) not
# accounting for friction.
const g = 9.81                  # [m/s^2]
const k = 6.25                  # [1/s^2]

const φ0 = 0.0                   # Initial angle of pendulum from negative y-axis

# ------------------ For Delta-robot ---------------------
const L0 = 1.0
const L1 = 1.5
const L2 = 2.0
const L3 = 0.5
const LC1 = 0.75
const LC2 = 1.0
const M1 = 0.1
const M2 = 0.1
const M3 = 0.3
const J1 = 0.4
const J2 = 0.4
const γ = 1.0

# === HELPER FUNCTIONS TO READ AND WRITE DATA
const data_dir = joinpath("data", "experiments")

# create directory for this experiment
exp_path(expid) = joinpath(data_dir, expid)
data_Y_path(expid) = joinpath(exp_path(expid), "Y.csv")


# === MODEL REALIZATION AND SIMULATION ===
# We use the following naming conventions for parameters:
# θ: All parameters of the dynamical model
# η: All parameters of the disturbance model
# pars: All free parameters
# p: vcat(θ, η)
# NOTE: If number and location of free parameters change, the sensitivity TODO: There must be a nicer solution to this
# functions defined in the code must also be changed
# const free_dyn_pars_true = [k]                    # true value of all free parameters
# get_all_θs(pars::Vector{Float64}) = [m, L, g, pars[1]]
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
    const free_dyn_pars_true = [k]#Array{Float64}(undef, 0)#[k]# True values of free parameters #Array{Float64}(undef, 0)
    const num_dyn_vars = 7
    const num_dyn_vars_adj = 7 # For adjoint method, there might be additional state variables, since outputs need to be baked into the state. Though outputs are already baked in for pendulum
    use_adjoint = false
    use_new_adj = true
    get_all_θs(pars::Vector{Float64}) = [m, L, g, pars[1]]#[pars[1], L, pars[2], k]
    # Each row corresponds to lower and upper bounds of a free dynamic parameter.
    dyn_par_bounds = [0.1 1e4]#[0.01 1e4; 0.1 1e4; 0.1 1e4]#; 0.1 1e4] #Array{Float64}(undef, 0, 2)
    @warn "The learning rate dimensiond doesn't deal with disturbance parameters in any nice way, other info comes from W_meta, and this part is hard coded"
    const_learning_rate = [1.0]#[0.1, 1.0, 0.1]
    model_sens_to_use = pendulum_sensitivity_k#_with_dist_sens_1#_sans_g_with_dist_sens_1#_with_dist_sens_3#pendulum_sensitivity_k_with_dist_sens_1#pendulum_sensitivity_sans_g#_full
    model_to_use = pendulum_new
    model_adj_to_use = my_pendulum_adjoint_konly_new
    model_adj_to_use_dist_sens = my_pendulum_adjoint_distsensa1_new#_with_dist_sens_3
    model_adj_to_use_dist_sens_new = my_pendulum_foradj_distsensa1
    sgd_version_to_use = perform_SGD_adam_new
    # Models for debug:
    model_stepbystep = pendulum_adj_stepbystep_NEW#pendulum_adj_stepbystep_k#pendulum_adj_stepbystep_deb
    model_stepbystep_dist = pendulum_adj_stepbystep_dist_new

    Fpk = (x, dx) -> [0.; 0.; abs(x[4])*x[4]; abs(x[5])*x[5]; 0.; 0.; 0.;;]
    Fpm = (x, dx) -> [.0; .0; dx[4]; dx[5]+g; .0; .0; .0;;]
    FpL = (x, dx) -> [.0; .0; .0; .0; -2L; .0; .0;;]
    deb_Fp = Fpk

    f(x::Vector{Float64}, θ::Vector{Float64}) = x[7]               # applied on the state at each step
    # f_sens should return a matrix/column vector with each row corresponding to a different output component and each column corresponding to a different parameter
    f_sens(x::Vector{Float64}, θ::Vector{Float64})::Matrix{Float64} = [x[14];;]# x[21] x[28] x[35]]# x[42] x[49]]# x[28]]##[x[14] x[21] x[28] x[35] x[42]]   # NOTE: Hard-coded right now
    # f_sens(x::Vector{Float64}, θ::Vector{Float64}) = [x[14], x[21], x[28]]                                                                                           #tuesday debug starting here
    f_sens_deb(x::Vector{Float64}, θ::Vector{Float64}) = x[8:end]
    f_debug(x::Vector{Float64}, θ::Vector{Float64}) = x[1:7]
    # The purpose of a separate baseline function is only relevant for delta robot, because of parameter-dependent output function
    f_sens_baseline(x::Vector{Float64}, θ::Vector{Float64})::Matrix{Float64} = f_sens(x::Vector{Float64}, θ::Vector{Float64})
elseif model_id == MOH_MDL
    # For Mohamed's model:
    const free_dyn_pars_true = [0.8]
    const num_dyn_vars = 2
    use_adjoint = true
    get_all_θs(pars::Vector{Float64}) = pars#free_dyn_pars_true
    # Each row corresponds to lower and upper bounds of a free dynamic parameter.
    dyn_par_bounds = [0.01 1e4]
    @warn "The learning rate dimensiond doesn't deal with disturbance parameters in any nice way, other info comes from W_meta, and this part is hard coded"
    const_learning_rate = [0.1]
    model_sens_to_use = mohamed_sens
    model_to_use = model_mohamed
    model_adj_to_use = mohamed_adjoint_new
    model_stepbystep = mohamed_stepbystep
    sgd_version_to_use = perform_SGD_adam_new

    f(x::Vector{Float64}) = x[1]#x[2]
    # f_sens should return a matrix/column vector with each row corresponding to a different output component and each column corresponding to a different parameter
    f_sens(x::Vector{Float64}, θ::Vector{Float64})::Matrix{Float64} = [x[3];;]#[x[4]]
    f_sens_deb(x::Vector{Float64}) = x[3:4]
elseif model_id == DELTA
    const free_dyn_pars_true = [L0, L1, L2, L3, LC1, LC2, M1, M2, M3, J1, J2, γ]#Array{Float64}(undef, 0) # TODO: Change dyn_par_bounds if changing parameter
    const num_dyn_vars = 30#24#30
    const num_dyn_vars_adj = 33#27#33 # For adjoint method, there might be additional state variables, since outputs need to be baked into the state
    use_adjoint = true
    use_new_adj = true
    get_all_θs(pars::Vector{Float64}) = vcat(pars[1:11], [g], pars[12])#[L0, L1, L2, L3, LC1, LC2, M1, M2, M3, J1, J2, g, γ]
    # dyn_par_bounds = Array{Float64}(undef, 0, 2)
    dyn_par_bounds = hcat(fill(0.01, 12, 1), fill(1e4, 12, 1))#[0.01 1e4]#[2*(L3-L0-L2)/sqrt(3)+0.01 2*(L2+L3-L0)/sqrt(3)-0.01; 0.01 1e4; 0.01 1e4]#[0.01 1e4]
    dyn_par_bounds[3,1] = 1.0 # Setting lower bound for L2
    @warn "The learning rate dimension doesn't deal with disturbance parameters in any nice way, other info comes from W_meta, and this part is hard coded" # Oooh, what if we define what function of nx, n_in etc to use here, and in get_experiment_data that function is simply used? Instead of having to define stuff there since only then are nx and n_in defined
    # const_learning_rate = [0.1]#[0.1, 0.1, 0.1, 0.05, 0.05, 0.1, 0.02, 0.02, 0.05, 0.05, 0.05, 0.2]
    const_learning_rate = [0.1, 0.1, 0.1, 0.05, 0.05, 0.1, 0.02, 0.02, 0.05, 0.05, 0.05, 0.2]#, 0.1, 0.2, 0.2, 0.05, 0.05, 0.05, 0.05, 0.05, 0.05, 0.05, 0.05, 0.05] #[0.1, 0.2, 0.2, 0.05, 0.05, 0.05, 0.05, 0.05, 0.05, 0.05, 0.05, 0.05] # For disturbance model
    model_sens_to_use = delta_robot_gc_allparsens#_alldist_FAKE#delta_robot_gc_γsens    # Ah, when using _FAKE version, baseline fails because of computation of Jacobian
    # TODO: Add length assertions here in file instead of in functions? So they crash during include? Or maybe that's worse
    model_to_use = delta_robot_gc
    model_adj_to_use = delta_robot_gc_adjoint_allpar_new
    model_adj_to_use_dist_sens = delta_robot_gc_adjoint_allpar_alldist  # Old adjoint approach, i.e. not using foradj
    model_adj_to_use_dist_sens_new = delta_robot_gc_foradj_allpar_alldist
    sgd_version_to_use = perform_SGD_adam_new_deltaversion  # Needs to update bounds of L3 dynamically based on L0
    # Models for debug:
    model_stepbystep = delta_adj_stepbystep_NEW
    
    # Only used for adjoint debugging purposes
    FpL1 = (x, dx) -> [cos(x[1])*dx[27]+cos(x[1])*dx[30]-sin(x[1])*dx[26]-sin(x[1])*dx[29]; 0.0; 0.0; -cos(x[4])*dx[27]-(sin(x[4])*dx[26])*0.5-(sqrt(3)*sin(x[4])*dx[25])*0.5; 0.0; 0.0; (sqrt(3)*sin(x[7])*dx[28])*0.5-(sin(x[7])*dx[29])*0.5-cos(x[7])*dx[30]; 0.0; 0.0; sin(x[1])*dx[20]-cos(x[1])*dx[24]-cos(x[1])*dx[21]+sin(x[1])*dx[23]-0.0*cos(x[1])*(M2+M3)+dx[11]*(L2*M3+LC2*M2)*(sin(x[1])*sin(x[2])+cos(x[1])*cos(x[2])*cos(x[3]))+2*L1*dx[10]*(M2+M3)+x[11]^2*(L2*M3+LC2*M2)*(cos(x[2])*sin(x[1])-cos(x[1])*cos(x[3])*sin(x[2]))-cos(x[1])*sin(x[2])*sin(x[3])*dx[12]*(L2*M3+LC2*M2)-cos(x[1])*cos(x[3])*sin(x[2])*x[12]^2*(L2*M3+LC2*M2)-2*cos(x[1])*cos(x[2])*sin(x[3])*x[11]*x[12]*(L2*M3+LC2*M2); dx[10]*(L2*M3+LC2*M2)*(sin(x[1])*sin(x[2])+cos(x[1])*cos(x[2])*cos(x[3]))+x[10]^2*(L2*M3+LC2*M2)*(cos(x[1])*sin(x[2])-cos(x[2])*cos(x[3])*sin(x[1])); sin(x[1])*sin(x[2])*sin(x[3])*x[10]^2*(L2*M3+LC2*M2)-cos(x[1])*sin(x[2])*sin(x[3])*dx[10]*(L2*M3+LC2*M2); cos(x[4])*dx[21]+(sin(x[4])*dx[20])*0.5-0.0*cos(x[4])*(M2+M3)+dx[14]*(L2*M3+LC2*M2)*(sin(x[4])*sin(x[5])+cos(x[4])*cos(x[5])*cos(x[6]))+2*L1*dx[13]*(M2+M3)+x[14]^2*(L2*M3+LC2*M2)*(cos(x[5])*sin(x[4])-cos(x[4])*cos(x[6])*sin(x[5]))+(sqrt(3)*sin(x[4])*dx[19])*0.5-cos(x[4])*sin(x[5])*sin(x[6])*dx[15]*(L2*M3+LC2*M2)-cos(x[4])*cos(x[6])*sin(x[5])*x[15]^2*(L2*M3+LC2*M2)-2*cos(x[4])*cos(x[5])*sin(x[6])*x[14]*x[15]*(L2*M3+LC2*M2); dx[13]*(L2*M3+LC2*M2)*(sin(x[4])*sin(x[5])+cos(x[4])*cos(x[5])*cos(x[6]))+x[13]^2*(L2*M3+LC2*M2)*(cos(x[4])*sin(x[5])-cos(x[5])*cos(x[6])*sin(x[4])); sin(x[4])*sin(x[5])*sin(x[6])*x[13]^2*(L2*M3+LC2*M2)-cos(x[4])*sin(x[5])*sin(x[6])*dx[13]*(L2*M3+LC2*M2); cos(x[7])*dx[24]+(sin(x[7])*dx[23])*0.5-0.0*cos(x[7])*(M2+M3)+dx[17]*(L2*M3+LC2*M2)*(sin(x[7])*sin(x[8])+cos(x[7])*cos(x[8])*cos(x[9]))+2*L1*dx[16]*(M2+M3)+x[17]^2*(L2*M3+LC2*M2)*(cos(x[8])*sin(x[7])-cos(x[7])*cos(x[9])*sin(x[8]))-(sqrt(3)*sin(x[7])*dx[22])*0.5-cos(x[7])*sin(x[8])*sin(x[9])*dx[18]*(L2*M3+LC2*M2)-cos(x[7])*cos(x[9])*sin(x[8])*x[18]^2*(L2*M3+LC2*M2)-2*cos(x[7])*cos(x[8])*sin(x[9])*x[17]*x[18]*(L2*M3+LC2*M2); dx[16]*(L2*M3+LC2*M2)*(sin(x[7])*sin(x[8])+cos(x[7])*cos(x[8])*cos(x[9]))+x[16]^2*(L2*M3+LC2*M2)*(cos(x[7])*sin(x[8])-cos(x[8])*cos(x[9])*sin(x[7])); sin(x[7])*sin(x[8])*sin(x[9])*x[16]^2*(L2*M3+LC2*M2)-cos(x[7])*sin(x[8])*sin(x[9])*dx[16]*(L2*M3+LC2*M2); (sqrt(3)*cos(x[4]))*0.5; cos(x[1])+cos(x[4])*0.5; sin(x[1])-sin(x[4]); -(sqrt(3)*cos(x[7]))*0.5; cos(x[1])+cos(x[7])*0.5; sin(x[1])-sin(x[7]); -(sqrt(3)*sin(x[4])*x[13])*0.5; -sin(x[1])*x[10]-(sin(x[4])*x[13])*0.5; cos(x[1])*x[10]-cos(x[4])*x[13]; (sqrt(3)*sin(x[7])*x[16])*0.5; -sin(x[1])*x[10]-(sin(x[7])*x[16])*0.5; cos(x[1])*x[10]-cos(x[7])*x[16]; 0.0; -cos(x[1]); -sin(x[1]);;]
    FpL2 = (x, dx) -> [0.0; cos(x[2])*cos(x[3])*dx[27]-sin(x[2])*dx[29]-sin(x[2])*dx[26]+cos(x[2])*cos(x[3])*dx[30]+cos(x[2])*sin(x[3])*dx[25]+cos(x[2])*sin(x[3])*dx[28]; cos(x[3])*sin(x[2])*dx[25]+cos(x[3])*sin(x[2])*dx[28]-sin(x[2])*sin(x[3])*dx[27]-sin(x[2])*sin(x[3])*dx[30]; 0.0; dx[25]*((cos(x[5])*sin(x[6]))*0.5-(sqrt(3)*sin(x[5]))*0.5)-dx[26]*(sin(x[5])*0.5+(sqrt(3)*cos(x[5])*sin(x[6]))*0.5)-cos(x[5])*cos(x[6])*dx[27]; (cos(x[6])*sin(x[5])*dx[25])*0.5+sin(x[5])*sin(x[6])*dx[27]-(sqrt(3)*cos(x[6])*sin(x[5])*dx[26])*0.5; 0.0; dx[28]*((cos(x[8])*sin(x[9]))*0.5+(sqrt(3)*sin(x[8]))*0.5)-dx[29]*(sin(x[8])*0.5-(sqrt(3)*cos(x[8])*sin(x[9]))*0.5)-cos(x[8])*cos(x[9])*dx[30]; (cos(x[9])*sin(x[8])*dx[28])*0.5+sin(x[8])*sin(x[9])*dx[30]+(sqrt(3)*cos(x[9])*sin(x[8])*dx[29])*0.5; L1*M3*dx[11]*(sin(x[1])*sin(x[2])+cos(x[1])*cos(x[2])*cos(x[3]))+L1*M3*x[11]^2*(cos(x[2])*sin(x[1])-cos(x[1])*cos(x[3])*sin(x[2]))-L1*M3*cos(x[1])*cos(x[3])*sin(x[2])*x[12]^2-L1*M3*cos(x[1])*sin(x[2])*sin(x[3])*dx[12]-2*L1*M3*cos(x[1])*cos(x[2])*sin(x[3])*x[11]*x[12]; sin(x[2])*dx[20]+sin(x[2])*dx[23]-cos(x[2])*cos(x[3])*dx[21]-cos(x[2])*cos(x[3])*dx[24]+2*L2*M3*dx[11]-cos(x[2])*sin(x[3])*dx[19]-cos(x[2])*sin(x[3])*dx[22]+L1*M3*dx[10]*(sin(x[1])*sin(x[2])+cos(x[1])*cos(x[2])*cos(x[3]))-M3*g*cos(x[2])*cos(x[3])+L1*M3*x[10]^2*(cos(x[1])*sin(x[2])-cos(x[2])*cos(x[3])*sin(x[1]))-2*L2*M3*cos(x[2])*sin(x[2])*x[12]^2; sin(x[2])*sin(x[3])*dx[21]-cos(x[3])*sin(x[2])*dx[22]-cos(x[3])*sin(x[2])*dx[19]+sin(x[2])*sin(x[3])*dx[24]+2*L2*M3*sin(x[2])^2*dx[12]+M3*g*sin(x[2])*sin(x[3])+2*L2*M3*sin(2*x[2])*x[11]*x[12]+L1*M3*sin(x[1])*sin(x[2])*sin(x[3])*x[10]^2-L1*M3*cos(x[1])*sin(x[2])*sin(x[3])*dx[10]; L1*M3*dx[14]*(sin(x[4])*sin(x[5])+cos(x[4])*cos(x[5])*cos(x[6]))+L1*M3*x[14]^2*(cos(x[5])*sin(x[4])-cos(x[4])*cos(x[6])*sin(x[5]))-L1*M3*cos(x[4])*cos(x[6])*sin(x[5])*x[15]^2-L1*M3*cos(x[4])*sin(x[5])*sin(x[6])*dx[15]-2*L1*M3*cos(x[4])*cos(x[5])*sin(x[6])*x[14]*x[15]; dx[20]*(sin(x[5])*0.5+(sqrt(3)*cos(x[5])*sin(x[6]))*0.5)-dx[19]*((cos(x[5])*sin(x[6]))*0.5-(sqrt(3)*sin(x[5]))*0.5)+cos(x[5])*cos(x[6])*dx[21]+2*L2*M3*dx[14]+L1*M3*dx[13]*(sin(x[4])*sin(x[5])+cos(x[4])*cos(x[5])*cos(x[6]))-M3*g*cos(x[5])*cos(x[6])+L1*M3*x[13]^2*(cos(x[4])*sin(x[5])-cos(x[5])*cos(x[6])*sin(x[4]))-2*L2*M3*cos(x[5])*sin(x[5])*x[15]^2; (sqrt(3)*cos(x[6])*sin(x[5])*dx[20])*0.5-sin(x[5])*sin(x[6])*dx[21]-(cos(x[6])*sin(x[5])*dx[19])*0.5+2*L2*M3*sin(x[5])^2*dx[15]+M3*g*sin(x[5])*sin(x[6])+2*L2*M3*sin(2*x[5])*x[14]*x[15]+L1*M3*sin(x[4])*sin(x[5])*sin(x[6])*x[13]^2-L1*M3*cos(x[4])*sin(x[5])*sin(x[6])*dx[13]; L1*M3*dx[17]*(sin(x[7])*sin(x[8])+cos(x[7])*cos(x[8])*cos(x[9]))+L1*M3*x[17]^2*(cos(x[8])*sin(x[7])-cos(x[7])*cos(x[9])*sin(x[8]))-L1*M3*cos(x[7])*cos(x[9])*sin(x[8])*x[18]^2-L1*M3*cos(x[7])*sin(x[8])*sin(x[9])*dx[18]-2*L1*M3*cos(x[7])*cos(x[8])*sin(x[9])*x[17]*x[18]; dx[23]*(sin(x[8])*0.5-(sqrt(3)*cos(x[8])*sin(x[9]))*0.5)-dx[22]*((cos(x[8])*sin(x[9]))*0.5+(sqrt(3)*sin(x[8]))*0.5)+cos(x[8])*cos(x[9])*dx[24]+2*L2*M3*dx[17]+L1*M3*dx[16]*(sin(x[7])*sin(x[8])+cos(x[7])*cos(x[8])*cos(x[9]))-M3*g*cos(x[8])*cos(x[9])+L1*M3*x[16]^2*(cos(x[7])*sin(x[8])-cos(x[8])*cos(x[9])*sin(x[7]))-2*L2*M3*cos(x[8])*sin(x[8])*x[18]^2; 2*L2*M3*sin(x[8])^2*dx[18]-sin(x[8])*sin(x[9])*dx[24]-(sqrt(3)*cos(x[9])*sin(x[8])*dx[23])*0.5-(cos(x[9])*sin(x[8])*dx[22])*0.5+M3*g*sin(x[8])*sin(x[9])+2*L2*M3*sin(2*x[8])*x[17]*x[18]+L1*M3*sin(x[7])*sin(x[8])*sin(x[9])*x[16]^2-L1*M3*cos(x[7])*sin(x[8])*sin(x[9])*dx[16]; (sqrt(3)*cos(x[5]))*0.5+sin(x[2])*sin(x[3])+(sin(x[5])*sin(x[6]))*0.5; cos(x[2])+cos(x[5])*0.5-(sqrt(3)*sin(x[5])*sin(x[6]))*0.5; cos(x[3])*sin(x[2])-cos(x[6])*sin(x[5]); sin(x[2])*sin(x[3])-(sqrt(3)*cos(x[8]))*0.5+(sin(x[8])*sin(x[9]))*0.5; cos(x[2])+cos(x[8])*0.5+(sqrt(3)*sin(x[8])*sin(x[9]))*0.5; cos(x[3])*sin(x[2])-cos(x[9])*sin(x[8]); cos(x[2])*sin(x[3])*x[11]-(sqrt(3)*sin(x[5])*x[14])*0.5+cos(x[3])*sin(x[2])*x[12]+(cos(x[5])*sin(x[6])*x[14])*0.5+(cos(x[6])*sin(x[5])*x[15])*0.5; -sin(x[2])*x[11]-(sin(x[5])*x[14])*0.5-(sqrt(3)*cos(x[5])*sin(x[6])*x[14])*0.5-(sqrt(3)*cos(x[6])*sin(x[5])*x[15])*0.5; cos(x[2])*cos(x[3])*x[11]-cos(x[5])*cos(x[6])*x[14]-sin(x[2])*sin(x[3])*x[12]+sin(x[5])*sin(x[6])*x[15]; (sqrt(3)*sin(x[8])*x[17])*0.5+cos(x[2])*sin(x[3])*x[11]+cos(x[3])*sin(x[2])*x[12]+(cos(x[8])*sin(x[9])*x[17])*0.5+(cos(x[9])*sin(x[8])*x[18])*0.5; (sqrt(3)*cos(x[8])*sin(x[9])*x[17])*0.5-(sin(x[8])*x[17])*0.5-sin(x[2])*x[11]+(sqrt(3)*cos(x[9])*sin(x[8])*x[18])*0.5; cos(x[2])*cos(x[3])*x[11]-cos(x[8])*cos(x[9])*x[17]-sin(x[2])*sin(x[3])*x[12]+sin(x[8])*sin(x[9])*x[18]; -sin(x[2])*sin(x[3]); -cos(x[2]); -cos(x[3])*sin(x[2]) ;;]
    Fpγ  = (x, dx) -> [0.0; 0.0; 0.0; 0.0; 0.0; 0.0; 0.0; 0.0; 0.0; x[10]; x[11]; x[12]; x[13]; x[14]; x[15]; x[16]; x[17]; x[18]; 0.0; 0.0; 0.0; 0.0; 0.0; 0.0; 0.0; 0.0; 0.0; 0.0; 0.0; 0.0; 0.0; 0.0; 0.0]
    deb_Fp = Fpγ

    # # If output is all three servo angles
    # f(x::Vector{Float64}) = [x[1],x[4],x[7]]    # All three servo angles
    # f_sens(x::Vector{Float64}, θ::Vector{Float64}) = [x[25],x[28],x[31]]
    # # If output is position of end effector, expressed in angles of first arm
    f(x::Vector{Float64}, θ::Vector{Float64}) = [θ[3]*sin(x[2])*sin(x[3]) #L2*sin(x[2])*sin(x[3])
        θ[2]*cos(x[1]) + θ[3]*cos(x[2]) + θ[1] - θ[4] #L1*cos(x[1]) + L2*cos(x[2]) + L0 - L3
        θ[2]*sin(x[1]) + θ[3]*sin(x[2])*cos(x[3])] #L1*sin(x[1]) + L2*sin(x[2])*cos(x[3])]
    # f(x::Vector{Float64}) = x[1:30]   # DEBUG
    
    ##################################################################################################################################################
    # f_sens should return a matrix with each row corresponding to a different output component and each column corresponding to a different parameter
    ##################################################################################################################################################

    # sans_p-part
    f_sens_base(x::Vector{Float64}, θ::Vector{Float64}, par_ind::Int)::Matrix{Float64} = 
        [θ[3]*cos(x[2])*sin(x[3])*x[30*par_ind+2]+θ[3]*cos(x[3])*sin(x[2])*x[30*par_ind+3] #L2*cos(x[2])*sin(x[3])*x[30*par_ind+2]+L2*cos(x[3])*sin(x[2])*x[30*par_ind+3]
        -θ[2]*sin(x[1])*x[30*par_ind+1]-θ[3]*sin(x[2])*x[30*par_ind+2] #-L1*sin(x[1])*x[30*par_ind+1]-L2*sin(x[2])*x[30*par_ind+2]
        θ[2]*cos(x[1])*x[30*par_ind+1]+θ[3]*cos(x[2])*cos(x[3])*x[30*par_ind+2]-θ[3]*sin(x[2])*sin(x[3])*x[30*par_ind+3];;] #L1*cos(x[1])*x[30*par_ind+1]+L2*cos(x[2])*cos(x[3])*x[30*par_ind+2]-L2*sin(x[2])*sin(x[3])*x[30*par_ind+3];;]
    # p-parts
    f_sens_L0(x::Vector{Float64})::Matrix{Float64} = [0.0; 1.0; 0.0;;]
    f_sens_L1(x::Vector{Float64})::Matrix{Float64} = [0.0; cos(x[1]); sin(x[1]);;]
    f_sens_L2(x::Vector{Float64})::Matrix{Float64} = [sin(x[2])*sin(x[3]); cos(x[2]); cos(x[3])*sin(x[2]);;]
    f_sens_L3(x::Vector{Float64})::Matrix{Float64} = [0.0; -1.0; 0.0;;]
    f_sens_other(x::Vector{Float64})::Matrix{Float64} = zeros(3,1)

    # # Sensitivity wrt to L1 (currently for stabilised model). To create a column-matrix, make sure to use ;; at the end, e.g. [...;;]
    # f_sens(x::Vector{Float64}, θ::Vector{Float64})::Matrix{Float64} = f_sens_base(x, θ, 1)+f_sens_L1(x)

    # # Sensitivity wrt to whichever individual parameter except L0, L1, L2, L3, all others are the same
    # f_sens(x::Vector{Float64}, θ::Vector{Float64})::Matrix{Float64} = f_sens_base(x, θ, 1)+f_sens_other(x)

    # # Sensitivity wrt to [L1, M1, J1]
    # f_sens(x::Vector{Float64}, θ::Vector{Float64})::Matrix{Float64} = hcat(f_sens_L1(x)+f_sens_base(x,θ,1), f_sens_other(x)+f_sens_base(x,θ,2), f_sens_other(x)+f_sens_base(x,θ,3))

    # # Sensitivity wrt to γ and one disturbance parameter
    # f_sens(x::Vector{Float64}, θ::Vector{Float64})::Matrix{Float64} = [f_sens_base(x, θ, 1)+f_sens_other(x)    f_sens_base(x, θ, 2)+f_sens_other(x)]

    # Sensitivity wrt to debug2-case parameters
    # f_sens(x::Vector{Float64}, θ::Vector{Float64})::Matrix{Float64} = hcat(f_sens_base(x, θ, 1)+f_sens_other(x), f_sens_base(x, θ, 2)+f_sens_other(x))#, f_sens_base(x, θ, 3)+f_sens_L2(x))#, 
    #     # f_sens_base(x, θ, 4)+f_sens_L3(x), f_sens_base(x, θ, 5)+f_sens_other(x), f_sens_base(x, θ, 6)+f_sens_other(x), f_sens_base(x, θ, 7)+f_sens_other(x),
    #     # f_sens_base(x, θ, 8)+f_sens_other(x), f_sens_base(x, θ, 9)+f_sens_other(x), f_sens_base(x, θ, 10)+f_sens_other(x), f_sens_base(x, θ, 11)+f_sens_other(x),
    #     # f_sens_base(x, θ, 12)+f_sens_other(x))
    # f_sens(x::Vector{Float64}, θ::Vector{Float64})::Matrix{Float64} = hcat(f_sens_base(x, θ, 1)+f_sens_L1(x), f_sens_base(x, θ, 2)+f_sens_other(x))

    # # Sensitivity for deb1 tests
    # f_sens(x::Vector{Float64}, θ::Vector{Float64})::Matrix{Float64} = f_sens_base(x, θ, 1)+f_sens_L2(x)

    # # Sensitivity wrt to one disturbance parameter
    # f_sens(x::Vector{Float64}, θ::Vector{Float64})::Matrix{Float64} = f_sens_base(x, θ, 1)+f_sens_other(x)

    # Sensitivity wrt to all parameters
    f_sens(x::Vector{Float64}, θ::Vector{Float64})::Matrix{Float64} = hcat(f_sens_base(x, θ, 1)+f_sens_L0(x), f_sens_base(x, θ, 2)+f_sens_L1(x), f_sens_base(x, θ, 3)+f_sens_L2(x), 
        f_sens_base(x, θ, 4)+f_sens_L3(x), f_sens_base(x, θ, 5)+f_sens_other(x), f_sens_base(x, θ, 6)+f_sens_other(x), f_sens_base(x, θ, 7)+f_sens_other(x),
        f_sens_base(x, θ, 8)+f_sens_other(x), f_sens_base(x, θ, 9)+f_sens_other(x), f_sens_base(x, θ, 10)+f_sens_other(x), f_sens_base(x, θ, 11)+f_sens_other(x),
        f_sens_base(x, θ, 12)+f_sens_other(x))

    # # Sensitivity wrt to all disturbance parameters
    # f_sens(x::Vector{Float64}, θ::Vector{Float64})::Matrix{Float64} = hcat(f_sens_base(x, θ, 1)+f_sens_other(x), f_sens_base(x, θ, 2)+f_sens_other(x), f_sens_base(x, θ, 3)+f_sens_other(x), 
    #     f_sens_base(x, θ, 4)+f_sens_other(x), f_sens_base(x, θ, 5)+f_sens_other(x), f_sens_base(x, θ, 6)+f_sens_other(x), f_sens_base(x, θ, 7)+f_sens_other(x),
    #     f_sens_base(x, θ, 8)+f_sens_other(x), f_sens_base(x, θ, 9)+f_sens_other(x), f_sens_base(x, θ, 10)+f_sens_other(x), f_sens_base(x, θ, 11)+f_sens_other(x),
    #     f_sens_base(x, θ, 12)+f_sens_other(x))

    # # Sensitivity wrt to all dynamical parameters AND all disturbance parameters
    # f_sens(x::Vector{Float64}, θ::Vector{Float64})::Matrix{Float64} = hcat(f_sens_base(x, θ, 1)+f_sens_L0(x), f_sens_base(x, θ, 2)+f_sens_L1(x), f_sens_base(x, θ, 3)+f_sens_L2(x), 
    #     f_sens_base(x, θ, 4)+f_sens_L3(x), f_sens_base(x, θ, 5)+f_sens_other(x), f_sens_base(x, θ, 6)+f_sens_other(x), f_sens_base(x, θ, 7)+f_sens_other(x),
    #     f_sens_base(x, θ, 8)+f_sens_other(x), f_sens_base(x, θ, 9)+f_sens_other(x), f_sens_base(x, θ, 10)+f_sens_other(x), f_sens_base(x, θ, 11)+f_sens_other(x),
    #     f_sens_base(x, θ, 12)+f_sens_other(x), f_sens_base(x, θ, 13)+f_sens_other(x), f_sens_base(x, θ, 14)+f_sens_other(x), f_sens_base(x, θ, 15)+f_sens_other(x), 
    #     f_sens_base(x, θ, 16)+f_sens_other(x), f_sens_base(x, θ, 17)+f_sens_other(x), f_sens_base(x, θ, 18)+f_sens_other(x), f_sens_base(x, θ, 19)+f_sens_other(x),
    #     f_sens_base(x, θ, 20)+f_sens_other(x), f_sens_base(x, θ, 21)+f_sens_other(x), f_sens_base(x, θ, 22)+f_sens_other(x), f_sens_base(x, θ, 23)+f_sens_other(x),
    #     f_sens_base(x, θ, 24)+f_sens_other(x))

    # BASELINE: SHOULD NOT INCLUDE DISTURBANCE PARAMETERS, SINCE BASELINE METHOD CANNOT IDENTIFY THEM ANYWAY
    # Sensitivity wrt to all dynamical parameters
    f_sens_baseline(x::Vector{Float64}, θ::Vector{Float64})::Matrix{Float64} = hcat(f_sens_base(x, θ, 1)+f_sens_L0(x), f_sens_base(x, θ, 2)+f_sens_L1(x), f_sens_base(x, θ, 3)+f_sens_L2(x), 
    f_sens_base(x, θ, 4)+f_sens_L3(x), f_sens_base(x, θ, 5)+f_sens_other(x), f_sens_base(x, θ, 6)+f_sens_other(x), f_sens_base(x, θ, 7)+f_sens_other(x),
    f_sens_base(x, θ, 8)+f_sens_other(x), f_sens_base(x, θ, 9)+f_sens_other(x), f_sens_base(x, θ, 10)+f_sens_other(x), f_sens_base(x, θ, 11)+f_sens_other(x),
    f_sens_base(x, θ, 12)+f_sens_other(x))
    # # Sensitivity wrt to whichever individual parameter except L0, L1, L2, L3, all others are the same
    # f_sens_baseline(x::Vector{Float64}, θ::Vector{Float64})::Matrix{Float64} = f_sens_base(x, θ, 1)+f_sens_other(x)
    # # Sensitivity wrt whichever parameters I felt like while debugging (γ)
    # f_sens_baseline(x::Vector{Float64}, θ::Vector{Float64})::Matrix{Float64} = hcat(f_sens_base(x, θ, 1)+f_sens_other(x))


    # # Just getting all states
    # f(x::Vector{Float64}) = x[1:24]
    # f_sens(x::Vector{Float64}, θ::Vector{Float64}) = x[1:48]
    # Since none of the state variables are the outputs, we add output sensitivites at the end. Those three extra states are e.g. needed for adjoint method.
    f_sens_deb(x::Vector{Float64}, θ::Vector{Float64}) = inject_adj_sens(x, f_sens(x, θ))
    f_debug(x::Vector{Float64}, θ::Vector{Float64}) = vcat(x[1:num_dyn_vars], f(x, θ))
end

y_len = length(f(ones(num_dyn_vars), get_all_θs(free_dyn_pars_true)))
h(sol,θ) = apply_outputfun(x->f(x,θ), sol)                            # for our model
h_comp(sol,θ) = apply_two_outputfun(x->f(x,θ), x->f_sens(x,θ), sol)           # for complete model with dynamics sensitivity
h_sens(sol,θ) = apply_outputfun(x->f_sens(x,θ), sol)                  # for only returning sensitivity 
h_sens_base(sol,θ) = apply_outputfun(x->f_sens_baseline(x,θ), sol)                  # for only sensitivity, minus sens of disturbance parameters which baseline doesn't use 
h_comp_base(sol,θ) = apply_two_outputfun(x->f(x,θ), x->f_sens_baseline(x,θ), sol)
h_debug(sol,θ) = apply_outputfun(x->f_debug(x,θ), sol)
h_debug_with_sens(sol,θ) = apply_outputfun(x->vcat(f_debug(x,θ), f_sens_deb(x,θ)), sol)
h_sens_deb(sol,θ) = apply_two_outputfun(x->f_debug(x,θ), x->f_sens_deb(x,θ), sol)
# data-set output function
h_data(sol,θ) = apply_outputfun(x -> f(x,θ) .+ σ * randn(size(f(x,θ))), sol)

learning_rate_vec(t::Int, grad_norm::Float64) = const_learning_rate#if (t < 100) const_learning_rate else ([0.1/(t-99.0), 1.0/(t-99.0)]) end#, 1.0, 1.0]  #NOTE Dimensions must be equal to number of free parameters
learning_rate_vec_red(t::Int, grad_norm::Float64) = const_learning_rate./sqrt(t)# t>50 ? const_learning_rate./sqrt(100*t) : const_learning_rate./sqrt(t)

const num_dyn_pars = length(free_dyn_pars_true)#size(dyn_par_bounds, 1)
realize_model_sens(u::Function, w::Function, pars::Vector{Float64}, N::Int) = problem(
    model_sens_to_use(φ0, u, w, get_all_θs(pars)),
    N,
    Ts,
)
realize_model(u::Function, w::Function, free_dyn_pars::Vector{Float64}, N::Int) = problem(
    model_to_use(φ0, u, w, get_all_θs(free_dyn_pars)),
    N,
    Ts,
)

const dθ = length(get_all_θs(free_dyn_pars_true))

# === SOLVER PARAMETERS ===
const abstol = 1e-8#1e-9
const reltol = 1e-5#1e-6
const maxiters = Int64(1e8)


solvew_sens(u::Function, w::Function, free_dyn_pars::Vector{Float64}, N::Int; kwargs...) = solve(
  realize_model_sens(u, w, free_dyn_pars, N),
  # In e.g. solve_customstep we subtract half the sampling interval from the final time. This seems enough to make sure the length of sol.t is actually
  # N samples. However, here we seem to have to subtract a whole sampling-interval extra for the same effect.
  saveat = 0:Ts:N*Ts,
  abstol = abstol,
  reltol = reltol,
  maxiters = maxiters;
  kwargs...,
)
solvew(u::Function, w::Function, free_dyn_pars::Vector{Float64}, N::Int; kwargs...) = solve(
  realize_model(u, w, free_dyn_pars, N),
  # In e.g. solve_customstep we subtract half the sampling interval from the final time. This seems enough to make sure the length of sol.t is actually
  # N samples. However, here we seem to have to subtract a whole sampling-interval extra for the same effect.
  saveat = 0:Ts:N*Ts,
  abstol = abstol,
  reltol = reltol,
  maxiters = maxiters;
  kwargs...,
)
solve_customstep(u::Function, w::Function, free_dyn_pars::Vector{Float64}, N::Int, myTs::Float64; kwargs...) = solve(
    realize_model(u, w, free_dyn_pars, N),
    # Because of numerical inaccuracies, solve() often returns one sample more than the length of 0:myTs:N*Ts, just past time N*Ts. To avoid this, we subtract 0.0001
    saveat = 0:myTs:N*Ts-myTs/2,
    abstol = abstol,
    reltol = reltol,
    maxiters = maxiters;
    kwargs...,
)

function get_estimates(expid::String, pars0::Vector{Float64}, N_trans::Int = 0, num_stacks::Int=1)
    start_datetime = now()
    exp_data, isws = get_experiment_data(expid)
    W_meta = exp_data.W_meta
    u = exp_data.u
    Y = exp_data.Y
    # TODO: Is this stacked way of saving multidimensional Y really the best? Maybe
    N = size(Y,1)÷y_len-1
    Nw = exp_data.Nw
    nx = W_meta.nx
    n_in = W_meta.n_in
    n_out = W_meta.n_out
    n_tot = nx*n_in
    dη = length(W_meta.η)
    dist_par_inds = W_meta.free_par_inds

    @assert (length(pars0) == num_dyn_pars+length(dist_par_inds)) "Please pass exactly $(num_dyn_pars+length(W_meta.free_par_inds)) parameter values"
    @assert (size(dyn_par_bounds, 1) == num_dyn_pars) "Please provide bounds for exactly all free dynamic parameters. Now $(size(dyn_par_bounds, 1)) are provided for $num_dyn_pars parameters"
    @assert (length(const_learning_rate) == length(pars0)) "The learning rate must have the same number of components as the number of parameters to be identified, currently is has $(length(const_learning_rate)) instead of $(length(pars0))"

    if !isdir(joinpath(data_dir, "tmp/"))
        mkdir(joinpath(data_dir, "tmp/"))
    end

    get_all_parameters(free_pars::Vector{Float64}) = vcat(get_all_θs(free_pars), exp_data.get_all_ηs(free_pars))
    par_bounds = vcat(dyn_par_bounds, exp_data.dist_par_bounds)

    # === We then optimize parameters for the baseline model ===
    function baseline_model_parametrized(dummy_input, pars)
        # NOTE: The true input is encoded in the solvew_sens()-function, but this function
        # still needs to to take two input arguments, so dummy_input could just be
        # anything, it's not used anyway
        Y_base = solvew(u, t -> zeros(n_out+length(dist_par_inds)*n_out), pars, N ) |> sol -> h(sol,get_all_θs(pars))

        # NOTE: SCALAR_OUTPUT is assumed. Edit: I think I generalized to multivariate output in a way that makes this still work
        # Y_base is either a vector of vectors (inner vector is a multi-dimensional output) or a vector of scalars.
        return vcat(Y_base[N_trans+1:end]...)   # New, should be equivalent with old but with a far more straightforward expression
        # return reshape(vcat(Y_base[N_trans+1:end,:]...), :)   # Returns 1D-array # Old, I do not know why it had this expression, seems just overkill
    end

    function jacobian_model_b(dummy_input, free_pars)
        jac = solvew_sens(u, t -> zeros(n_out+length(dist_par_inds)*n_out), free_pars, N) |> sol -> h_sens_base(sol,get_all_θs(free_pars))
        return vcat(jac[N_trans+1:end, :]...)
    end

    function baseline_model_parametrized_stacked(dummy_input, pars)
        # NOTE: The true input is encoded in the solvew_sens()-function, but this function
        # still needs to to take two input arguments, so dummy_input could just be
        # anything, it's not used anyway
        Y_base_deep = solvew(u, t -> zeros(n_out+length(dist_par_inds)*n_out), pars, N ) |> sol -> h(sol,get_all_θs(pars))
        # Y_base_deep is either a vector of vectors (inner vector is a multi-dimensional output) or a vector of scalars.
        Y_base_flat = vcat(Y_base_deep[N_trans+1:end]...)
        Y_base_stacked = zeros(num_stacks*length(Y_base_flat))
        len = length(Y_base_flat)
        for ind = 1:num_stacks
            Y_base_stacked[(ind-1)*len+1:ind*len] = Y_base_flat
        end
        return Y_base_stacked
    end

    function jacobian_model_b_stacked(dummy_input, free_pars)
        jac_deep = solvew_sens(u, t -> zeros(n_out+length(dist_par_inds)*n_out), free_pars, N) |> sol -> h_sens_base(sol,get_all_θs(free_pars))
        jac_flat = vcat(jac_deep[N_trans+1:end, :]...)
        len = size(jac_flat, 1)
        jac_stacked = zeros(num_stacks*size(jac_flat,1), size(jac_flat,2))
        for ind = 1:num_stacks
            jac_stacked[(ind-1)*len+1:ind*len,:] = jac_flat
        end
        return jac_stacked
    end

    # Returns estimate of gradient of cost function
    # M_mean specifies over how many realizations the gradient estimate is computed
    function get_gradient_base(y, free_pars, isws)
        Yb, jacYb = solvew_sens(u, t -> zeros(n_out+length(dist_par_inds)*n_out), free_pars, N) |> sol -> h_comp_base(sol,get_all_θs(free_pars))
        return get_cost_gradient(y, vcat(Yb...)[:,1:1], [vcat(jacYb...)], N_trans)
    end

    # So this is for also using SGD for baseline, since LM didn't work
    # NOTE: THIS FUNCTION HAS NOT BEEN TESTED, I GAVE IT A THOUGHT BUT DIDN'T TRY IT
    function get_gradient_base_stacked(ystacked, free_pars, isws, num_stacks::Int=1)
        # Ym, jacsYm = simulate_system_sens(exp_data, free_pars, 2M_mean*num_stacks, dist_par_inds, isws)
        Yb, jacYb = solvew_sens(u, t -> zeros(n_out+length(dist_par_inds)*n_out), free_pars, N) |> sol -> h_comp_base(sol,get_all_θs(free_pars))
        if num_stacks > 1
            len = size(Yb,1)
            Ybstacked = zeros(num_stacks*len)
            jacstacked = [zeros(num_stacks*len, length(free_pars))]
            for ind2=1:num_stacks
                Ybstacked[(ind2-1)*len+1:ind2*len] = vcat(Yb...)
                jacstacked[1][(ind2-1)*len+1:ind2*len,:] = vcat(jacYb...)
            end
            # NOTE: Non-zero N_trans not supported!!
            return get_cost_gradient(ystacked, Ybstacked[:,1:M_mean], jacstacked[M_mean+1:end])
        end
        # Uses different noise realizations for estimate of output and estiamte of jacobian
        return get_cost_gradient(ystacked, Yb, jacsYm, N_trans)
    end

    # E = size(Y, 2)
    # DEBUG
    E = 1
    @warn "Using E = $E instead of default"
    opt_pars_baseline = zeros(length(free_dyn_pars_true), E)
    # trace_base[e][t][j] contains the value of parameter j before iteration t
    # corresponding to dataset e
    trace_base = [[pars0] for e=1:E]
    setup_duration = now() - start_datetime
    baseline_durations = Array{Millisecond, 1}(undef, E)
    # @warn "Not running baseline identification now"
    for e=1:E
        time_start = now()
        use_sgd_instead = false
        if use_sgd_instead
            # Since baseline doesn't identify disturbace parameters, it only needs the learning rate corresponding to the dynamical parameters
            base_lr = (t, gn) -> learning_rate_vec_red(t, gn)[1:length(free_dyn_pars_true)]
            # ------------------------------------------------>
            if num_stacks > 1
                # Y_true_stacked = vcat([Y[N_trans+1:end,(ind-1)*E+e] for ind=1:num_stacks]...)
                # baseline_result = get_fit_sens(Y_true_stacked, pars0,
                #     baseline_model_parametrized_stacked, jacobian_model_b_stacked,
                #     par_bounds[:,1], par_bounds[:,2])
                Y_true_stacked = vcat([Y[N_trans+1:end,(ind-1)*E+e] for ind=1:num_stacks]...)
                opt_pars_baseline[:, e], trace_base[e], _ = sgd_version_to_use((free_pars, M_mean)->get_gradient_base_stacked(Y_true_stacked, free_pars, isws, num_stacks), pars0[1:length(free_dyn_pars_true)], dyn_par_bounds, base_lr, maxiters=100, verbose=true, tol=1e-8)
            else
                # baseline_result = get_fit_sens(Y[N_trans+1:end,e], pars0,
                #     baseline_model_parametrized, jacobian_model_b,
                #     par_bounds[:,1], par_bounds[:,2])
                opt_pars_baseline[:, e], trace_base[e], _ = sgd_version_to_use((free_pars, M_mean)->get_gradient_base(Y[:,e], free_pars, isws), pars0[1:length(free_dyn_pars_true)], dyn_par_bounds, base_lr, maxiters=100, verbose=true, tol=1e-8)
            end    
            println("Completed for dataset $e for parameters $(opt_pars_baseline[:,e])")
            # <---------------------------------------------------
        else
            # ------------------------------------------------>
            if num_stacks > 1
                Y_true_stacked = vcat([Y[N_trans+1:end,(ind-1)*E+e] for ind=1:num_stacks]...)
                baseline_result = get_fit_sens(Y_true_stacked, pars0,
                    baseline_model_parametrized_stacked, jacobian_model_b_stacked,
                    par_bounds[:,1], par_bounds[:,2])
            else
                baseline_result = get_fit_sens(Y[N_trans+1:end,e], pars0,
                    baseline_model_parametrized, jacobian_model_b,
                    par_bounds[:,1], par_bounds[:,2])
            end
            opt_pars_baseline[:, e] = coef(baseline_result)
    
            println("Completed for dataset $e for parameters $(opt_pars_baseline[:,e])")
    
            # Sometimes (the first returned value I think) the baseline_trace
            # has no elements, and therefore doesn't contain the metadata x
            if length(baseline_result.trace) > 1
                for j=2:length(baseline_result.trace)
                    push!(trace_base[e], trace_base[e][end]+baseline_result.trace[j].metadata["dx"])
                end
            end
            # <---------------------------------------------------
        end
        writedlm(joinpath(data_dir, "tmp/backup_baseline_e$e.csv"), opt_pars_baseline[:,e], ',')
        writedlm(joinpath(data_dir, "tmp/backup_baseline_trace_e$e.csv"), trace_base[e], ',')
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

    function get_gradient_estimate_stacked(ystacked, free_pars, isws, M_mean::Int=1, num_stacks::Int=1)
        Ym, jacsYm = simulate_system_sens(exp_data, free_pars, 2M_mean*num_stacks, dist_par_inds, isws)
        if num_stacks > 1
            len = size(Ym,1)
            Ymstacked = zeros(num_stacks*len, 2M_mean)
            jacsstacked = [zeros(num_stacks*len, length(free_pars)) for i=1:2M_mean]
            for ind1=1:2M_mean
                for ind2=1:num_stacks
                    Ymstacked[(ind2-1)*len+1:ind2*len,ind1] = Ym[:,(ind2-1)*2M_mean+ind1]
                    jacsstacked[ind1][(ind2-1)*len+1:ind2*len,:] = jacsYm[(ind2-1)*2M_mean+ind1]
                end
            end
            # NOTE: Non-zero N_trans not supported!!
            return get_cost_gradient(ystacked, Ymstacked[:,1:M_mean], jacsstacked[M_mean+1:end])
        end
        # Uses different noise realizations for estimate of output and estiamte of jacobian
        return get_cost_gradient(ystacked, Ym[:,1:M_mean], jacsYm[M_mean+1:end], N_trans)
    end

    # ------------------------------- For using adjoint sensitivity ---------------------------------

    function compute_Gp_acc(y_func, dy_func, xvec1, xvec2, free_pars, wmm)
        # NOTE: m shouldn't be larger than M÷2
        x_func  = get_mvar_cubic(0.0:Tsλ:N*Ts, xvec1) # x_func  = get_mvar_cubic(0.0:Tsλ:N*Ts, Xcomp_m[m])
        x2_func = get_mvar_cubic(0.0:Tsλ:N*Ts, xvec2) # x2_func = get_mvar_cubic(0.0:Tsλ:N*Ts, Xcomp_m[M÷2+m])
        der_est  = get_der_est(0.0:Tsλ:N*Ts, x_func)
        # Subtracting Tsλ/2 because sometimes we don't get the right number of elements due to numerical inaccuracies otherwise (edit: more than N+1 elements or what?)
        dx = get_mvar_cubic(0.0:Tsλ:N*Ts-Tsλ/2, der_est)
        

        # m, L, g, k
        θ = get_all_θs(free_pars)
        m = θ[1]
        L = θ[2]
        g = θ[3]
        k = θ[4]

        # NOTE: ODE equation needs to be hard-coded for appropriate choice of parameters here
        λs_ODE, λsint_ODE = solve_accurate_adjoint(N, Ts, x_func, dx, x2_func, y_func, 1)   # 1 because my_ind=1, not that it matters at all here, I just picked any value
        # int_m(t) = dx(t)[4]*λs_ODE(t)[3] + (dx(t)[5]+g)*λs_ODE(t)[4]
        # int_L(t) = -2L*λs_ODE(t)[5]
        # int_g(t) = m*λs_ODE(t)[4]     # NOTE: g-estimation doesn't seem to work at all, not for default adjoint method either
        # int_k(t) = abs(x_func(t)[4])*x_func(t)[4]*λs_ODE(t)[3] + abs(x_func(t)[5])*x_func(t)[5]*λs_ODE(t)[4]

        # int_func(t) = dx(t)[4]*λs_ODE(t)[3] + (dx(t)[5]+g)*λs_ODE(t)[4] # For only m
        int_func(t) = [dx(t)[4]*λs_ODE(t)[3] + (dx(t)[5]+g)*λs_ODE(t)[4]
                        -2L*λs_ODE(t)[5]
                        abs(x_func(t)[4])*x_func(t)[4]*λs_ODE(t)[3] + abs(x_func(t)[5])*x_func(t)[5]*λs_ODE(t)[4]]  # For m, L, k
        return -quadgk(int_func, 0.0, N*Ts, rtol=1e-10)[1]#/(N*Ts)
    end

    function compute_Gp_adj(y_func, dy_func, xvec1, xvec2, free_pars, wmm_m)
        x_func  = extrapolate(scale(interpolate(xvec1, (BSpline(interp_type), NoInterp())), 0.0:Tsλ:N*Ts, 1:size(xvec1,2)), Line())
        x2_func = extrapolate(scale(interpolate(xvec2, (BSpline(interp_type), NoInterp())), 0.0:Tsλ:N*Ts, 1:size(xvec2,2)), Line())
        der_est  = get_der_est2(0.0:Tsλ:N*Ts, x_func, size(xvec1,2))
        der_est2 = get_der_est2(0.0:Tsλ:N*Ts, x2_func, size(xvec1,2))
        # Subtracting Tsλ/2 because sometimes we don't get the right number of elements due to numerical inaccuracies otherwise
        dx = extrapolate(scale(interpolate(der_est, (BSpline(interp_type), NoInterp())), 0.0:Tsλ:N*Ts-Tsλ/2, 1:size(xvec1,2)), Line())
        dx2 = extrapolate(scale(interpolate(der_est2, (BSpline(interp_type), NoInterp())), 0.0:Tsλ:N*Ts-Tsλ/2, 1:size(xvec2,2)), Line())

        # NOTE: In case initial conditions are independent of m (independent of wmm in this case), we could do this outside
        # ---------------- Computing xp0, initial conditions of derivative of x wrt to p ----------------------
        mdl_sens = model_sens_to_use(φ0, u, wmm_m, get_all_θs(free_pars))
        xp0 = reshape(f_sens_deb(mdl_sens.x0,get_all_θs(free_pars)), num_dyn_vars_adj, length(f_sens_deb(mdl_sens.x0,get_all_θs(free_pars)))÷num_dyn_vars_adj)

        # ----------------- Actually solving adjoint system ------------------------
        mdl_adj, get_Gp = model_adj_to_use(u, wmm_m, get_all_θs(free_pars), N*Ts, x_func, x2_func, y_func, dy_func, xp0, dx, dx2)
        adj_prob = problem_reverse(mdl_adj, N, Ts)
        adj_sol = solve(adj_prob, saveat = 0:Tso:(N*Ts-Tso/2), abstol =  abstol, reltol = reltol,
            maxiters = maxiters)

        return get_Gp(adj_sol)
    end

    function compute_Gp_adj_dist_sens(y_func, dy_func, xvec1, xvec2, free_pars, wmm_m, xwmm_m, vmm_m, B̃, B̃ηa, η, ndist, na)
        # ndist should be the number of free disturbance parameters
        # na should be the number of the free disturbance parameters that correspond to the A-matrix
        # NOTE: m shouldn't be larger than M÷2
        x_func  = extrapolate(scale(interpolate(xvec1, (BSpline(interp_type), NoInterp())), 0.0:Tsλ:N*Ts, 1:size(xvec1,2)), Line())
        x2_func = extrapolate(scale(interpolate(xvec2, (BSpline(interp_type), NoInterp())), 0.0:Tsλ:N*Ts, 1:size(xvec2,2)), Line())
        der_est  = get_der_est2(0.0:Tsλ:N*Ts, x_func, size(xvec1,2))
        der_est2 = get_der_est2(0.0:Tsλ:N*Ts, x2_func, size(xvec1,2))
        # Subtracting Tsλ/2 because sometimes we don't get the right number of elements due to numerical inaccuracies otherwise
        dx = extrapolate(scale(interpolate(der_est, (BSpline(interp_type), NoInterp())), 0.0:Tsλ:N*Ts-Tsλ/2, 1:size(xvec1,2)), Line())
        dx2 = extrapolate(scale(interpolate(der_est2, (BSpline(interp_type), NoInterp())), 0.0:Tsλ:N*Ts-Tsλ/2, 1:size(xvec2,2)), Line())

        # NOTE: In case initial conditions are independent of m (independent of wmm in this case), we could do this outside
        # ---------------- Computing xp0, initial conditions of derivative of x wrt to p ----------------------
        mdl_sens = model_sens_to_use(φ0, u, t->vcat(wmm_m(t), zeros(ndist*length(wmm_m(t)),1)), get_all_θs(free_pars))   # The model expects w plus its sensitivities, which we haven't computed since we don't need them for xp0. So we just pad the wmm_m function
        xp0 = reshape(f_sens_deb(mdl_sens.x0,get_all_θs(free_pars)), num_dyn_vars_adj, length(f_sens_deb(mdl_sens.x0,get_all_θs(free_pars)))÷num_dyn_vars_adj)

        # u, w, xw, v, θ, T, x, x2, y, dy, xp0, dx, dx2, B̃, B̃θ, η, N_trans
        # TODO: Define vmm somewhere and apss it here!

        # ----------------- Actually solving adjoint system ------------------------
        mdl_adj, get_Gp = model_adj_to_use_dist_sens(u, wmm_m, xwmm_m, vmm_m, get_all_θs(free_pars), N*Ts, x_func, x2_func, y_func, dy_func, xp0, dx, dx2, B̃, B̃ηa, η, na)
        adj_prob = problem_reverse(mdl_adj, N, Ts)
        adj_sol = solve(adj_prob, saveat = 0:Tso:(N*Ts-Tso/2), abstol =  abstol, reltol = reltol,
            maxiters = maxiters)

        return get_Gp(adj_sol)
    end

    # function compute_Gp_adj_dist_sens_new(y_func, dy_func, xvec1, xvec2, free_pars, wmm_m, xwmm_m, η, na)
    #     x_func  = extrapolate(scale(interpolate(xvec1, (BSpline(interp_type), NoInterp())), 0.0:Tsλ:N*Ts, 1:size(xvec1,2)), Line())
    #     x2_func = extrapolate(scale(interpolate(xvec2, (BSpline(interp_type), NoInterp())), 0.0:Tsλ:N*Ts, 1:size(xvec2,2)), Line())
    #     der_est  = get_der_est2(0.0:Tsλ:N*Ts, x_func, size(xvec1,2))
    #     der_est2 = get_der_est2(0.0:Tsλ:N*Ts, x2_func, size(xvec1,2))
    #     # Subtracting Tsλ/2 because sometimes we don't get the right number of elements due to numerical inaccuracies otherwise
    #     dx = extrapolate(scale(interpolate(der_est, (BSpline(interp_type), NoInterp())), 0.0:Tsλ:N*Ts-Tsλ/2, 1:size(xvec1,2)), Line())
    #     dx2 = extrapolate(scale(interpolate(der_est2, (BSpline(interp_type), NoInterp())), 0.0:Tsλ:N*Ts-Tsλ/2, 1:size(xvec2,2)), Line())

    #     # NOTE: In case initial conditions are independent of m (independent of wmm in this case), we could do this outside
    #     # ---------------- Computing xp0, initial conditions of derivative of x wrt to p ----------------------
    #     mdl_sens = model_sens_to_use(φ0, u, wmm_m, get_all_θs(free_pars))
    #     xp0 = reshape(f_sens_deb(mdl_sens.x0,get_all_θs(free_pars)), num_dyn_vars_adj, length(f_sens_deb(mdl_sens.x0,get_all_θs(free_pars)))÷num_dyn_vars_adj)

    #     # ----------------- Actually solving adjoint system ------------------------
    #     # mdl_adj, get_Gp = model_adj_to_use(u, wmm_m, get_all_θs(free_pars), N*Ts, x_func, x2_func, y_func, dy_func, xp0, dx, dx2)
    #     mdl_adj, get_Gp = model_adj_to_use_dist_sens_new(u, wmm_m, xwmm_m, get_all_θs(free_pars), N*Ts, x_func, x2_func, y_func, dy_func, xp0, dx, dx2, η, na)
    #     adj_prob = problem_reverse(mdl_adj, N, Ts)
    #     adj_sol = solve(adj_prob, saveat = 0:Tso:(N*Ts-Tso/2), abstol =  abstol, reltol = reltol,
    #         maxiters = maxiters)

    #     return get_Gp(adj_sol)
    # end

    function compute_Gp_adj_dist_sens_new(y_func, dy_func, xvec1, xvec2, free_pars, wmm_m)
        x_func  = extrapolate(scale(interpolate(xvec1, (BSpline(interp_type), NoInterp())), 0.0:Tsλ:N*Ts, 1:size(xvec1,2)), Line())
        x2_func = extrapolate(scale(interpolate(xvec2, (BSpline(interp_type), NoInterp())), 0.0:Tsλ:N*Ts, 1:size(xvec2,2)), Line())
        der_est  = get_der_est2(0.0:Tsλ:N*Ts, x_func, size(xvec1,2))
        der_est2 = get_der_est2(0.0:Tsλ:N*Ts, x2_func, size(xvec1,2))
        # Subtracting Tsλ/2 because sometimes we don't get the right number of elements due to numerical inaccuracies otherwise
        dx = extrapolate(scale(interpolate(der_est, (BSpline(interp_type), NoInterp())), 0.0:Tsλ:N*Ts-Tsλ/2, 1:size(xvec1,2)), Line())
        dx2 = extrapolate(scale(interpolate(der_est2, (BSpline(interp_type), NoInterp())), 0.0:Tsλ:N*Ts-Tsλ/2, 1:size(xvec2,2)), Line())

        # NOTE: In case initial conditions are independent of m (independent of wmm in this case), we could do this outside
        # ---------------- Computing xp0, initial conditions of derivative of x wrt to p ----------------------
        mdl_sens = model_sens_to_use(φ0, u, wmm_m, get_all_θs(free_pars))
        xp0 = reshape(f_sens_deb(mdl_sens.x0,get_all_θs(free_pars)), num_dyn_vars_adj, length(f_sens_deb(mdl_sens.x0,get_all_θs(free_pars)))÷num_dyn_vars_adj)

        # ----------------- Actually solving adjoint system ------------------------
        # mdl_adj, get_Gp = model_adj_to_use(u, wmm_m, get_all_θs(free_pars), N*Ts, x_func, x2_func, y_func, dy_func, xp0, dx, dx2)
        mdl_adj, get_Gp = model_adj_to_use_dist_sens_new(u, wmm_m, get_all_θs(free_pars), N*Ts, x_func, x2_func, y_func, dy_func, xp0, dx, dx2)
        adj_prob = problem_reverse(mdl_adj, N, Ts)
        adj_sol = solve(adj_prob, saveat = 0:Tso:(N*Ts-Tso/2), abstol =  abstol, reltol = reltol,
            maxiters = maxiters)

        return get_Gp(adj_sol)
    end

    function get_gradient_adjoint(y, free_pars, compute_Gp, M_mean::Int=1)
        Zm = [randn(Nw, n_tot) for m = 1:2M_mean]
        W_meta = exp_data.W_meta
        nx = W_meta.nx
        n_out = W_meta.n_out
        N = size(exp_data.Y, 1)÷y_len-1

        η = exp_data.get_all_ηs(free_pars)

        dmdl = discretize_ct_noise_model_with_sensitivities_alt(get_ct_disturbance_model(η, nx, n_out), δ, dist_par_inds)
        # # NOTE: OPTION 1: Use the rows below here for linear interpolation
        XWm = simulate_noise_process_mangled(dmdl, Zm)
        wmm(m::Int) = mk_noise_interp(dmdl.Cd, XWm, m, δ)

        # NOTE: No option of using transient here, that might be confusing for future reference! Or should that option even be here, or maybe outside?
        # y_func  = linear_interpolation_multivar(y[:,1], Ts, y_len)        # Old, custom interpolation
        y_func  = extrapolate(scale(interpolate(transpose(reshape(y[:,1], y_len, :)), (BSpline(interp_type), NoInterp())), 0.0:Ts:N*Ts, 1:y_len), Line())   # New, better interpolation
        dy_est  = (y[y_len+1:end,1]-y[1:end-y_len,1])/Ts
        # dy_func = linear_interpolation_multivar(dy_est, Ts, y_len)        # Old, custom interpolation
        dy_func = extrapolate(scale(interpolate(transpose(reshape(dy_est, y_len, :)), (BSpline(interp_type), NoInterp())), 0.0:Ts:(N-1)*Ts, 1:y_len), Line())   # New, better interpolation
        sampling_ratio = Int(Ts/Tsλ)
        solve_func(m) = solve_customstep(u, wmm(m), free_pars, N, Tsλ) |> sol -> h_debug(sol,get_all_θs(free_pars))
        Xcomp_m, _ = solve_in_parallel_debug(m -> solve_func(m), 1:2M_mean, 7, sampling_ratio)
        mean(solve_adj_in_parallel2(m -> compute_Gp(y_func, dy_func, Xcomp_m[m], Xcomp_m[M_mean+m], free_pars, wmm(m)), 1:M_mean, length(free_pars)), dims=2)[:]
    end

    function get_gradient_adjoint_distsens(y, free_pars, compute_Gp, M_mean::Int=1)
        Zm = [randn(Nw, n_tot) for _ = 1:2M_mean]
        W_meta = exp_data.W_meta
        nx = W_meta.nx
        n_out = W_meta.n_out
        N = size(exp_data.Y, 1)÷y_len-1

        η = exp_data.get_all_ηs(free_pars)

        vmm(m::Int) = mk_v_ZOH(Zm[m], δ)

        dmdl, B̃, B̃ηa = discretize_ct_noise_model_with_sensitivities_for_adj(get_ct_disturbance_model(η, nx, n_out), δ, dist_par_inds)
        # # NOTE: OPTION 1: Use the rows below here for linear interpolation
        XWm = simulate_noise_process_mangled(dmdl, Zm)
        wmm(m::Int)  = mk_noise_interp(dmdl.Cd, XWm, m, δ)
        xwmm(m::Int) = mk_xw_interp(dmdl.Cd, XWm, m, δ)

        # NOTE: No optoin of using transient here, that might be confusing for future reference! Or should that option even be here, or maybe outside?
        # y_func  = linear_interpolation_multivar(y[:,1], Ts, y_len)        # Old, custom interpolation
        y_func  = extrapolate(scale(interpolate(transpose(reshape(y[:,1], y_len, :)), (BSpline(interp_type), NoInterp())), 0.0:Ts:N*Ts, 1:y_len), Line())   # New, better interpolation
        dy_est  = (y[y_len+1:end,1]-y[1:end-y_len,1])/Ts
        # dy_func = linear_interpolation_multivar(dy_est, Ts, y_len)        # Old, custom interpolation
        dy_func = extrapolate(scale(interpolate(transpose(reshape(dy_est, y_len, :)), (BSpline(interp_type), NoInterp())), 0.0:Ts:(N-1)*Ts, 1:y_len), Line())   # New, better interpolation
        sampling_ratio = Int(Ts/Tsλ)
        solve_func(m) = solve_customstep(u, wmm(m), free_pars, N, Tsλ) |> sol -> h_debug(sol,get_all_θs(free_pars))
        Xcomp_m, _ = solve_in_parallel_debug(m -> solve_func(m), 1:2M_mean, 7, sampling_ratio)    # NOTE: Have to make sure not to solve problem with forward sensitivities, that might not work and also just defeats purpose of adjoint method
        # temp = solve_adj_in_parallel(m -> compute_Gp(y_func, dy_func, Xcomp_m[m], Xcomp_m[M_mean+m], free_pars, wmm(m)), 1:M_mean)
        # mean(temp, dims=2)[:]
        na = length(findall(W_meta.free_par_inds .<= nx))   # Number of the disturbance parameters that corresponds to A-matrix. Rest will correspond to C-matrix
        # mean(solve_adj_in_parallel(m -> compute_Gp(y_func, dy_func, Xcomp_m[m], Xcomp_m[M_mean+m], free_pars, wmm(m), xwmm(m), vmm(m), B̃, B̃ηa, η, length(W_meta.free_par_inds), na), 1:M_mean), dims=2)[:]
        mean(solve_adj_in_parallel2(m -> compute_Gp(y_func, dy_func, Xcomp_m[m], Xcomp_m[M_mean+m], free_pars, wmm(m), xwmm(m), vmm(m), B̃, B̃ηa, η, length(W_meta.free_par_inds), na), 1:M_mean, length(free_pars)), dims=2)[:]
    end

    function get_gradient_adjoint_distsens_new(y, free_pars, compute_Gp, M_mean::Int=1)
        Zm = [randn(Nw, n_tot) for m = 1:2M_mean]
        W_meta = exp_data.W_meta
        nx = W_meta.nx
        n_out = W_meta.n_out
        N = size(exp_data.Y, 1)÷y_len-1

        η = exp_data.get_all_ηs(free_pars)

        dmdl = discretize_ct_noise_model_with_sensitivities_alt(get_ct_disturbance_model(η, nx, n_out), δ, dist_par_inds)
        # # NOTE: OPTION 1: Use the rows below here for linear interpolation
        XWm = simulate_noise_process_mangled(dmdl, Zm)
        wmm(m::Int) = mk_noise_interp(dmdl.Cd, XWm, m, δ)

        # NOTE: No option of using transient here, that might be confusing for future reference! Or should that option even be here, or maybe outside?
        # y_func  = linear_interpolation_multivar(y[:,1], Ts, y_len)        # Old, custom interpolation
        y_func  = extrapolate(scale(interpolate(transpose(reshape(y[:,1], y_len, :)), (BSpline(interp_type), NoInterp())), 0.0:Ts:N*Ts, 1:y_len), Line())   # New, better interpolation
        dy_est  = (y[y_len+1:end,1]-y[1:end-y_len,1])/Ts
        # dy_func = linear_interpolation_multivar(dy_est, Ts, y_len)        # Old, custom interpolation
        dy_func = extrapolate(scale(interpolate(transpose(reshape(dy_est, y_len, :)), (BSpline(interp_type), NoInterp())), 0.0:Ts:(N-1)*Ts, 1:y_len), Line())   # New, better interpolation
        sampling_ratio = Int(Ts/Tsλ)
        solve_func(m) = solve_customstep(u, wmm(m), free_pars, N, Tsλ) |> sol -> h_debug(sol,get_all_θs(free_pars))
        Xcomp_m, _ = solve_in_parallel_debug(m -> solve_func(m), 1:2M_mean, 7, sampling_ratio)
        mean(solve_adj_in_parallel2(m -> compute_Gp(y_func, dy_func, Xcomp_m[m], Xcomp_m[M_mean+m], free_pars, wmm(m)), 1:M_mean, length(free_pars)), dims=2)[:]
    end

    # -------------------------------- end of adjoint sensitivity specifics ----------------------------------------

    # NOTE: About adjoint method and num_stacks: We can simply multiply M by num_stacks to achieve the same effect as if we obtained Gp by averaging the
    # gradient obtained from every data-set in the stack. This is because the final Gp is anyway obtained by averaging over all M, so it doesn't matter if we
    # average over M averages, or if we just average once over M*num_stacks gradients

    # Picking correct way of computing gradients
    if use_adjoint
        if length(dist_par_inds) > 0
            if use_new_adj
                get_gradient_estimate_p = (free_pars, M_mean, e) -> get_gradient_adjoint_distsens_new(Y[:,e], free_pars, compute_Gp_adj_dist_sens_new, M_mean*num_stacks)
            else
                get_gradient_estimate_p = (free_pars, M_mean, e) -> get_gradient_adjoint_distsens(Y[:,e], free_pars, compute_Gp_adj_dist_sens, M_mean*num_stacks)
            end
        else
            get_gradient_estimate_p = (free_pars, M_mean, e) -> get_gradient_adjoint(Y[:,e], free_pars, compute_Gp_adj, M_mean*num_stacks)
        end
    elseif num_stacks > 1
        get_gradient_estimate_p = (free_pars, M_mean, e) -> get_gradient_estimate_stacked(vcat([Y[:,(ind-1)*E+e] for ind=1:num_stacks]...), free_pars, isws, M_mean, num_stacks)
    else
        get_gradient_estimate_p = (free_pars, M_mean, e) -> get_gradient_estimate(Y[:,e], free_pars, isws, M_mean)
    end


    opt_pars_proposed = zeros(length(pars0), E)
    avg_pars_proposed = zeros(length(pars0), E)
    trace_proposed = [ [Float64[]] for e=1:E]
    trace_gradient = [ [Float64[]] for e=1:E]
    trace_step     = [ [Float64[]] for e=1:E]        ## DEBUG!!!!!
    trace_lrate     = [ [Float64[]] for e=1:E]        ## DEBUG!!!!!
    proposed_durations = Array{Millisecond, 1}(undef, E)

    # # Use this block of code to move random number generator forward exactly as if experiments 1:e_skip had been performed
    # # This allows for exact reproducibility of any experiment, regardless of which experiment we start with
    # e_skip = 2
    # for t=1:5*e_skip# 1:maxiters*e_skip is what it should say
    #     for m=1:2M_rate(t)
    #         randn(Nw, n_tot)
    #     end
    # end

    # @warn "Not running proposed identification now"
    for e=1:E
        time_start = now()
        # jacobian_model(x, p) = get_proposed_jacobian(pars, isws, M)  # NOTE: This won't give a jacobian estimate independent of Ym, but maybe we don't need that since this isn't SGD?
        @warn "Only using maxiters=100 right now"
        opt_pars_proposed[:,e], trace_proposed[e], trace_gradient[e] =
                sgd_version_to_use((free_pars, M_mean) -> get_gradient_estimate_p(free_pars, M_mean, e), pars0, par_bounds, verbose=true, tol=1e-8, maxiters=100)

        # @warn "NOT WRITING BACKUPS RIGHT NOW!"
        writedlm(joinpath(data_dir, "tmp/backup_proposed_e$e.csv"), opt_pars_proposed[:,e], ',')
        writedlm(joinpath(data_dir, "tmp/backup_average_e$e.csv"), avg_pars_proposed[:,e], ',')
        writedlm(joinpath(data_dir, "tmp/backup_trace_e$e.csv"), trace_proposed[e], ',')

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

# It's quite a lot depracated actually, maybe worth getting rid of now that we have simulate_system()?
# NOTE: This function is a little depracated now that we have introduced num_stacks into the other functions
function get_outputs(expid::String, pars0::Vector{Float64})
    exp_data, isws = get_experiment_data(expid)
    W_meta = exp_data.W_meta
    Y = exp_data.Y
    N = size(Y,1)÷y_len-1
    u = exp_data.u
    Nw = exp_data.Nw
    nx = W_meta.nx
    n_in = W_meta.n_in
    n_out = W_meta.n_out
    n_tot = nx*n_in
    dη = length(W_meta.η)
    dist_par_inds = W_meta.free_par_inds

    @assert (length(pars0) == num_dyn_pars+length(W_meta.free_par_inds)) "Please pass exactly $(num_dyn_pars+length(W_meta.free_par_inds)) parameter values"

    get_all_parameters(free_pars::Vector{Float64}) = vcat(get_all_θs(free_pars), exp_data.get_all_ηs(free_pars))
    p = get_all_parameters(pars0)
    θ = p[1:dθ]
    η = p[dθ+1: dθ+dη]
    # C = reshape(η[nx+1:end], (n_out, n_tot))

    # === Computes output of the baseline model ===
    Y_base, sens_base = solvew_sens(u, t -> zeros(n_out+length(dist_par_inds)*n_out), pars0, N) |> sol -> h_comp(sol,get_all_θs(pars0))

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
    # # NOTE: OPTION 2: Use the rows below here for exact interpolation
    # reset_isws!(isws)
    # XWm = simulate_noise_process(dmdl, Zm)
    # wmm(m::Int) = mk_newer_noise_interp(view(η, 1:nx), C, XWm, m, n_in, δ, isws)

    calc_mean_y_N_prop(N::Int, free_pars::Vector{Float64}, m::Int) =
        solvew_sens(u, t -> wmm(m)(t), free_pars, N) |> sol -> h_comp(sol,get_all_θs(free_pars))
    calc_mean_y_prop(free_pars::Vector{Float64}, m::Int) = calc_mean_y_N_prop(N, free_pars, m)
    Ym_prop, sens_m_prop = solve_in_parallel_sens(m -> calc_mean_y_prop(pars0, m), ms)
    Y_mean_prop = reshape(mean(Ym_prop, dims = 2), :)

    return Y, Y_base, sens_base, Ym_prop, Y_mean_prop, sens_m_prop
end

function get_experiment_data(expid::String)::Tuple{ExperimentData, Array{InterSampleWindow, 1}}
    # A single realization of the disturbance serves as input
    # input is assumed to contain the input signal, and not the state
    input  = readdlm(joinpath(data_dir, expid*"/U.csv"), ',')[:]
    W_meta_raw, W_meta_names =
        readdlm(joinpath(data_dir, expid*"/meta_W_new.csv"), ',', header=true)

    U_meta_raw, U_meta_names = 
        readdlm(joinpath(data_dir, expid*"/meta_U.csv"), ',', header=true)
    n_u_out = Int(U_meta_raw[1,3])

    # NOTE: These variable assignments are based on the names in W_meta_names,
    # but hard-coded, so if W_meta_names is changed then there is no guarantee
    # that they will match
    nx = Int(W_meta_raw[1,1])
    n_in = Int(W_meta_raw[1,2])
    n_out = Int(W_meta_raw[1,3])
    num_rel = Int(W_meta_raw[1,6])
    E = num_rel
    n_tot = nx*n_in
    # Parameters of true system. η = [a_pars c_pars], where the first nx elements are paramters of the A-matrix, and the remaining are parameters of the C-matrix
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
    free_dist_pars = fill(false, size(η_true))                                                  # Known disturbance model
    # free_dist_pars = vcat(fill(true, nx), fill(false, n_out), fill(true, (n_tot-1)*n_out))    # Whole a-vector and all but first n_out elements of c-vector unknown (MAXIMUM UNKNOWN PARAMETERS FOR SINGLE DIFFERENTIABILITY (PENDULUM))
    # free_dist_pars = vcat(fill(true, nx), fill(true, n_tot*n_out))                     # All parameters unknown (MAXIMUM UNKNOWN PARAMETERS, NO DIFFERENTIABILITY (DELTA))
    # free_dist_pars = vcat(fill(true, nx), fill(false, n_tot*n_out))                    # Whole a-vector unknown
    # free_dist_pars = vcat(true, fill(false, nx-1), fill(false, n_tot*n_out))           # First parameter of a-vector unknown
    # free_dist_pars = vcat(false, true, fill(false, nx-2), fill(false, n_tot*n_out))    # Second parameter of a-vector unknown
    free_par_inds = findall(free_dist_pars)          # Indices of free variables in η. Assumed to be sorted in ascending order.
    # Array of tuples containing lower and upper bound for each free disturbance parameter
    # dist_par_bounds = Array{Float64}(undef, 0, 2)
    dist_par_bounds = [-Inf Inf; -Inf Inf; -Inf Inf; -Inf Inf; -Inf Inf; -Inf Inf; -Inf Inf; -Inf Inf; -Inf Inf; -Inf Inf; -Inf Inf; -Inf Inf]#[0 Inf; 0 Inf; -Inf Inf]
    function get_all_ηs(free_pars::Vector{Float64})
        # If copy() is not used here, some funky stuff that I don't fully understand happens.
        # I think essentially η_true stops being defined after function returns, so
        # setting all_η to its value doesn't behave quite as I expected
        all_η = copy(η_true)
        # Fetches user-provided values for free disturbance parameters only
        all_η[free_par_inds] = free_pars[num_dyn_pars+1:end]
        return all_η
     end

    # compute the maximum number of steps we can take
    N_margin = 0#2    # Solver can request values of inputs after the time horizon
                    # ends, so we require a margin of a few samples of the noise
                    # to ensure that we can provide such values. EDIT: TODO: Do we really? Or was this just because of your poorly implemented interpolation? Let's try having margin 0
    Nw = size(input, 1)÷n_in
    N = (Nw-N_margin)÷(Int(Ts/δ)*y_len)     # Number of steps we can take. Computed this convoluted way to minimize chance of numerical inaccuracy by maximizing amount of integers
    W_meta = DisturbanceMetaData(nx, n_in, n_out, η_true, free_par_inds, num_rel)

    # u(t::Float64) = sin(t)*ones(3)
    # u(t::Float64) = [sin(t); cos(t); 0.5*sin(t)+0.5*cos(t)]
    # @warn "USING SINUSOID AS INPUT INSTEAD"
    u(t::Float64) = linear_interpolation_multivar(input, δ, n_u_out)(t)

    function solve_wrapper(e::Int, isws::Vector{InterSampleWindow})::Vector{Float64}
        XW = transpose(CSV.read("data/experiments/$(expid)/XW_T.csv", CSV.Tables.matrix; header=false, skipto=e, limit=1))
        # # Exact interpolation
        # w = mk_newer_noise_interp(a_vec, C_true, demangle_XW(XW, n_tot), 1, n_in, δ, isws[e:e])
        # Linear interpolation
        w = mk_noise_interp(C_true, XW, 1, δ)
        nested_y = solvew(u, w, free_dyn_pars_true, N) |> sol -> h_data(sol,get_all_θs(free_dyn_pars_true))
        return vcat(nested_y...)
    end

    # === We first generate the output of the true system ===
    function calc_Y(isws::Vector{InterSampleWindow})
        # NOTE: XWs should be non-mangled, which is why we don't divide by n_tot
        # @assert (Nw <= size(XWs, 1)) "Disturbance data size mismatch ($(Nw) > $(size(XWs, 1)))"
        es = 1:E
        # we = mk_we(XWs, isws)
        solve_in_parallel(e -> solve_wrapper(e, isws), es)  # returns Y
    end

    if isfile(data_Y_path(expid))
        @info "Reading output of true system"
        Y = readdlm(data_Y_path(expid), ',')
        isws = [initialize_isw(Q, W, n_tot, true) for e=1:M]
    else
        @info "Generating output of true system"
        # XW = readdlm(joinpath(data_dir, expid*"/XW.csv"), ',')
        isws = [initialize_isw(Q, W, n_tot, true) for e=1:max(num_rel, M)]
        Y = calc_Y(isws)
        writedlm(data_Y_path(expid), Y, ',')
    end

    # # This block can be used to check whether different implementations result
    # # in the same Y
    # @warn "Debugging sim. First 5 XW: $(XW[1:5])"
    # wdebug = mk_noise_interp(C_true, XW, 1, δ)
    # my_y = solvew(u, wdebug, free_dyn_pars_true, N) |> sol -> h(sol,get_all_θs(free_dyn_pars_true))
    # writedlm("data/experiments/pendulum_sensitivity/my_y.csv", my_y, ',')

    reset_isws!(isws)
    return ExperimentData(Y, u, get_all_ηs, dist_par_bounds[1:length(free_par_inds),:], W_meta, Nw), isws
end

# ======================= HELPER FUNCTIONS ========================

# NOTE: The realizations Ym and jacYm must be independent for this to return
# an unbiased estimate of the cost function gradient
function get_cost_gradient(Y::Vector{Float64}, Ym::Matrix{Float64}, jacsYm::Vector{Matrix{Float64}}, N_trans::Int=0)
    # N = size(Y,1)÷y_len-1, since Y also contains the zeroth sample.
    # While we sum over t0, t1, ..., tN, the error at t0 will always be zero
    # due to known initial conditions which is why we divide by N instead of N+1

    # Y[N_trans+1:end].-Ym[N_trans+1:end,:] is a matrix with as many columns as
    # Ym, where column i contains Y[N_trans+1:end]-Ym[N_trans+1:end,i]
    # Taking the mean of that gives us the average error as a function of time
    # over all realizations contained in Ym

    # mean(-jacsYm)[N_trans+1:end,:] is the average (over all m) jacobian of Ym[N_trans+1:end].
    # Different rows correspond to different t, while different columns correspond to different parameters

    # Previously used (theoretically equivalent)
    # (2/(size(Y,1)-N_trans-1))*
    #     sum(
    #     mean(Y[N_trans+1:end].-Ym[N_trans+1:end,:], dims=2)
    #     .*mean(-jacsYm)[N_trans+1:end,:]
    #     , dims=1)

    (2/(size(Y,1)÷y_len-N_trans))*(
        transpose(mean(-jacsYm)[N_trans+1:end,:])*
        mean(Y[N_trans+1:end].-Ym[N_trans+1:end,:], dims=2)
        )[:]
end

function get_cost_value(Y::Vector{Float64}, Ym::Matrix{Float64}, N_trans::Int=0)
    (1/(size(Y,1)÷y_len-N_trans))*sum( ( Y[N_trans+1:end] - mean(Ym[N_trans+1:end,:], dims=2) ).^2 )
end

function get_der_est(ts, func::Function)
    dim = length(func(0.0))
    der_est = zeros(length(ts)-1, dim)
    for (i,t) = enumerate(ts)
        if i > 1
            der_est[i-1,:] = (func(t)-func(ts[i-1]))./(t-ts[i-1])
        end
    end
    return der_est
end

# For new and improved interpolation handling
function get_der_est2(ts, func, dim)
    der_est = zeros(length(ts)-1, dim)
    for (i,t) = enumerate(ts)
        if i > 1
            der_est[i-1,:] = (func(t,1:dim)-func(ts[i-1],1:dim))./(t-ts[i-1])
        end
    end
    return der_est
end

function get_mvar_cubic(ts, der_est::AbstractMatrix{Float64})
    # Rows of der_est are assumed to be different values of t, columns of
    # matrix are assumed to be different elements of the vector-valued process
    temp = [cubic_spline_interpolation(ts, der_est[:,i], extrapolation_bc=Line()) for i=1:size(der_est,2)]
    # temp = [t->t for i=1:size(der_est,2)]
    return t -> [temp[i](t) for i=eachindex(temp)]
    # Returns a function mapping time to the vector-value of the function at that time
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

function mk_xw_interp(C::Matrix{Float64},
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
                return XW[end-n_tot+1:end, m]
            else
                # row of x_1(t_n) in XW is given by k. Note that t=0 is given by row 1
                k = n * n_tot + 1

                xl = XW[k:(k + n_tot - 1), m]
                xu = XW[(k + n_tot):(k + 2n_tot - 1), m]
                return (xl + (t-n*δ)*(xu-xl)/δ)
            end
        end
    end
end

function mk_v_ZOH(Zmm::Matrix{Float64}, δ::Float64)
    let
        function z(t::Float64)
            # n*δ <= t <= (n+1)*δ
            n = Int(t÷δ)
            return Zmm[n+1, :]  # +1 because t = 0 (and thus n=0) corresponds to index 1
        end
    end
end

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
    fit_result = curve_fit(model, jacobian_model, Float64[], Ye, pars, lower=lb, upper=ub, show_trace=true, inplace=false, x_tol=1e-8)    # Default: inplace = false, x_tol = 1e-8
    # return fit_result, trace
end

######################################################
############# TODO: WHY DO WE PASS dist_sens_inds IF THEY ALREADY ARE IN WMETA???????????????
######################################################

# Simulates system with specified white noise
function simulate_system(
    exp_data::ExperimentData,
    free_pars::Vector{Float64},
    M::Int,
    dist_sens_inds::Array{Int64, 1},
    isws::Array{InterSampleWindow,1},
    Zm::Array{Matrix{Float64},1})::Matrix{Float64}

    W_meta = exp_data.W_meta
    nx = W_meta.nx
    n_in = W_meta.n_in
    n_out = W_meta.n_out
    n_tot = nx*n_in
    dη = length(W_meta.η)
    N = size(exp_data.Y, 1)÷y_len-1

    p = vcat(get_all_θs(free_pars), exp_data.get_all_ηs(free_pars))
    θ = p[1:dθ]
    η = p[dθ+1: dθ+dη]
    # C = reshape(η[nx+1:end], (n_out, n_tot))  # Not correct when disturbance model is parametrized, use dmdl.Cd instead

    # dmdl = discretize_ct_noise_model(get_ct_disturbance_model(η, nx, n_out), δ)
    dmdl = discretize_ct_noise_model_with_sensitivities_alt(get_ct_disturbance_model(η, nx, n_out), δ, dist_sens_inds)
    # # NOTE: OPTION 1: Use the rows below here for linear interpolation
    XWm = simulate_noise_process_mangled(dmdl, Zm)
    wmm(m::Int) = mk_noise_interp(dmdl.Cd, XWm, m, δ)
    # NOTE: OPTION 2: Use the rows below here for exact interpolation
    # reset_isws!(isws)
    # XWm = simulate_noise_process(dmdl, Zm)
    # wmm(m::Int) = mk_newer_noise_interp(view(η, 1:nx), C, XWm, m, n_in, δ, isws)

    calc_mean_y_N(N::Int, free_pars::Vector{Float64}, m::Int) =
        solvew(exp_data.u, t -> wmm(m)(t), free_pars, N) |> sol -> h(sol,get_all_θs(free_pars))
        # solvew(exp_data.u, t -> wmm(m)(t), free_pars, N) |> sol -> h_debug(sol,get_all_θs(free_pars)) # DEBUG, for getting entire state instead of only measured output
    calc_mean_y(free_pars::Vector{Float64}, m::Int) = calc_mean_y_N(N, free_pars, m)
    return solve_in_parallel(m -> calc_mean_y(free_pars, m), collect(1:M))   # Returns Ym
end

# Simulates system with newly generated white noise
function simulate_system(
    exp_data::ExperimentData,
    free_pars::Vector{Float64},
    M::Int,
    dist_sens_inds::Array{Int64, 1},
    isws::Array{InterSampleWindow,1})::Matrix{Float64}

    W_meta = exp_data.W_meta
    nx = W_meta.nx
    n_in = W_meta.n_in
    n_tot = nx*n_in
    Nw = exp_data.Nw
    Zm = [randn(Nw, n_tot) for m = 1:M]
    simulate_system(exp_data, free_pars, M, dist_sens_inds, isws, Zm)
end

# Simulates system with specified white noise
function simulate_system_sens(
    exp_data::ExperimentData,
    free_pars::Vector{Float64},
    M::Int,
    dist_sens_inds::Array{Int64, 1},    # TODO: Aren't these included in exp_data? Why pass them separately then? Just seems random
    isws::Array{InterSampleWindow,1},
    Zm::Array{Matrix{Float64},1})::Tuple{Matrix{Float64}, Array{Matrix{Float64},1}}

    W_meta = exp_data.W_meta
    nx = W_meta.nx
    # n_in = W_meta.n_in
    n_out = W_meta.n_out
    # n_tot = nx*n_in
    # dη = length(W_meta.η)
    N = size(exp_data.Y, 1)÷y_len-1
    # dist_par_inds = W_meta.free_par_inds

    η = exp_data.get_all_ηs(free_pars)
    # p = vcat(get_all_θs(free_pars), exp_data.get_all_ηs(free_pars))
    # θ = p[1:dθ]
    # η = p[dθ+1: dθ+dη]
    # # C = reshape(η[nx+1:end], (n_out, n_tot))

    dmdl = discretize_ct_noise_model_with_sensitivities_alt(get_ct_disturbance_model(η, nx, n_out), δ, dist_sens_inds)
    # # NOTE: OPTION 1: Use the rows below here for linear interpolation
    XWm = simulate_noise_process_mangled(dmdl, Zm)
    wmm(m::Int) = mk_noise_interp(dmdl.Cd, XWm, m, δ)
    # NOTE: OPTION 2: Use the rows below here for exact interpolation
    # reset_isws!(isws)
    # XWm = simulate_noise_process(dmdl, Zm)
    # wmm(m::Int) = mk_newer_noise_interp(view(η, 1:nx), dmdl.Cd, XWm, m, n_in, δ, isws)

    calc_mean_y_N(N::Int, free_pars::Vector{Float64}, m::Int) =
        solvew_sens(exp_data.u, t -> wmm(m)(t), free_pars, N) |> sol -> h_comp(sol,get_all_θs(free_pars))
    calc_mean_y(free_pars::Vector{Float64}, m::Int) = calc_mean_y_N(N, free_pars, m)
    return solve_in_parallel_sens2(m -> calc_mean_y(free_pars, m), collect(1:M), y_len, length(free_pars), N)  # Returns Ym and JacsYm
    # return solve_in_parallel_sens(m -> calc_mean_y(free_pars, m), collect(1:M))  # Returns Ym and JacsYm
end

# Simulates system with newly generated white noise
function simulate_system_sens(
    exp_data::ExperimentData,
    free_pars::Vector{Float64},
    M::Int,
    dist_sens_inds::Array{Int64, 1},
    isws::Array{InterSampleWindow,1})::Tuple{Matrix{Float64}, Array{Matrix{Float64},1}}

    W_meta = exp_data.W_meta
    nx = W_meta.nx
    n_in = W_meta.n_in
    n_tot = nx*n_in
    Nw = exp_data.Nw
    # Zm = [randn(Nw, n_tot*(length(free_pars)+1)) for m = 1:M]   # Differentiation first, discretisation second
    Zm = [randn(Nw, n_tot) for m = 1:M]   # Discretisation first, differentiation second
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

# For adjoint method, the outputs of the system need to be the last state-variables. If x is the state of the system with all sensitivities, but without the outputs,
# then this function takes the sensitivities from the ny x np matrix adj_sens and inserts them correctly, so that the function returns the state with all sensitivities, 
# where the outputs are included in the state variables
function inject_adj_sens(x::Vector{Float64}, adj_sens::Matrix{Float64})
    np = size(adj_sens,2)
    out = zeros(num_dyn_vars_adj*np)
    for ind = 1:np
        # The first num_dyn_vars elements of x are the nominal states, only after that do we get the state sensitivities.
        out[(ind-1)*num_dyn_vars_adj+1:(ind-1)*num_dyn_vars_adj+num_dyn_vars] = x[ind*num_dyn_vars+1:(ind+1)*num_dyn_vars]
        out[(ind-1)*num_dyn_vars_adj+num_dyn_vars+1:ind*num_dyn_vars_adj] = adj_sens[:,ind]
    end
    return out
end

# ======================= DEBUGGING FUNCTIONS ========================

# -------------- Some visualisation for delta robot ----------------------
function animate_delta_gif(outmat::Matrix{Float64}, file_name::String)
    l = @layout [a b c]
    # L0 = θ[1]
    # L1 = θ[2]
    # L2 = θ[3]
    # L3 = θ[4]

    # other xlim actually, oddly enough, L1+L2+L0-L3 in magnitude, which seems weird...
    # but ylim should be L1 + L2
    p1 = scatter(-outmat[2,1:1], -outmat[3,1:1], shape=:star8, color=:blue, xlims=(-L2,L2), ylims=(-L2-L3,-0.5*L1-0.5*L3), legend=false)
    p2 = scatter(outmat[1,1:1], -outmat[3,1:1], shape=:star8, color=:blue, xlims=(-L2,L2), ylims=(-L2-L3,-0.5*L1-0.5*L3), legend=false)
    p3 = scatter(outmat[1,1:1], outmat[2,1:1], shape=:star8, color=:blue, xlims=(-L2,L2), ylims=(-L2,L2), legend=false)
    plot(p1, p2, p3, layout=l)

    anim = @animate for i = eachindex(outmat[1,:])
        p1 = scatter(-outmat[2,1:1], -outmat[3,1:1], shape=:star8, color=:blue, xlims=(-L2,L2), ylims=(-L2-L3,-0.5*L1-0.5*L3), legend=false)
        p2 = scatter(outmat[1,1:1], -outmat[3,1:1], shape=:star8, color=:blue, xlims=(-L2,L2), ylims=(-L2-L3,-0.5*L1-0.5*L3), legend=false)
        p3 = scatter(outmat[1,1:1], outmat[2,1:1], shape=:star8, color=:blue, xlims=(-L2,L2), ylims=(-L2,L2), legend=false)
        plot!(p1, -outmat[2,1:i], -outmat[3,1:i])
        plot!(p2, outmat[1,1:i], -outmat[3,1:i])
        plot!(p3, outmat[1,1:i], outmat[2,1:i])
        plot(p1, p2, p3, layout=l)
    end
    # gif(anim, file_name, fps = 15)
    gif(anim, file_name, fps = 30)
end

function solve_delta(N::Int)
    θ = [1.0, 1.5, 2.0, 0.5, 0.75, 1.0, 0.1, 0.1, 0.3, 0.4, 0.4, 9.81, 1.0]
    # θ[10] = 1.0210850451039977
    # L0 = θ[1]
    # L1 = θ[2]
    # L2 = θ[3]
    # L3 = θ[4]
    
    mdl_to_use = delta_robot_gc_J1sens#delta_robot_gc_stab #delta_robot_gc_L1sens

    # # du0 = [2.0; 2*cos(2*π/3); 2*cos(-2*π/3)]   # TODO: Replace cos-values by known root-expressions
    # delta_mdl = mdl_to_use(0.0, t->[sin(2*t); sin(2*(t+0.2*π/3)); sin(2*(t-0.2*π/3))], t->zeros(3), du0, θ) #t->5*[sin(10*t); sin(10*(t+0.2*π/3)); sin(10*(t-0.2*π/3))]
    # ------------- Two different input alternatives -------------
    # input  = readdlm("data/experiments/100_delta/U.csv", ',')
    # u_tmp(t::Float64) = interpw(input, 1, 1)(t)
    # u(t::Float64) = 5*u_tmp(t)[1]*[1.;1.;1.]
    
    # exp_data, _ = get_experiment_data("delta_crazy_2k_100")
    # u = exp_data.u
    u = t->[sin(2*t); sin(2*(t+0.2*π/3)); sin(2*(t-0.2*π/3))]
    delta_mdl = mdl_to_use(0.0, u, t->zeros(3), θ)

    delta_prob = problem(delta_mdl, N, Ts)
    sol = solve(delta_prob, saveat = 0:Ts:(N*Ts), abstol = abstol, reltol = reltol, maxiters = maxiters)
    outmat = get_delta_output(sol, θ)

    # DEBUG, to return entire state trajectory
    debug_outmat = zeros(60, size(outmat, 2))
    for (i,x) = enumerate(sol.u)
        debug_outmat[:,i] = x
    end

    # p1 = plot(-outmat[2,:], -outmat[3,:], xlims=(-L2,L2), ylims=(-L1-L3,-0.5*L1-0.5*L3), color=:blue, legend=false)
    # p2 = plot(outmat[1,:], -outmat[3,:], xlims=(-L2,L2), ylims=(-L1-L3,-0.5*L1-0.5*L3), color=:blue, legend=false)
    # p3 = plot(outmat[1,:], outmat[2,:], xlims=(-L2,L2), ylims=(-L2,L2), color=:blue, legend=false)
    # scatter!(p1, -outmat[2,1:1], -outmat[3,1:1], shape=:star8, color=:blue)
    # scatter!(p2, outmat[1,1:1], -outmat[3,1:1], shape=:star8, color=:blue)
    # scatter!(p3, outmat[1,1:1], outmat[2,1:1], shape=:star8, color=:blue)
    # l = @layout [a b c]
    # plot(p1, p2, p3, layout=l)
    
    # animate_delta_gif(outmat, θ, "data/results/delta_gif.gif")
    animate_delta_gif(outmat, "data/results/delta_gif.gif")
    return outmat, debug_outmat
end

function get_delta_output(sol, θ)
    T = length(sol)
    outmat = zeros(3,T)
    # TODO: sol.u is not the recommended way to access the solution
    for (i,x) = enumerate(sol.u)
        # out = 
        # [L2*sin(ϕ2)*sin(ϕ3)
        #  L1*cos(ϕ1) + L2*cos(ϕ2) + L0 - L3
        #  L1*sin(ϕ1) + L2*sin(ϕ2)*cos(ϕ3)]
        # where
        # L0 = θ[1], L1 = θ[2], L2 = θ[3], L4 = θ[4]
        outmat[:,i] = [θ[3]*sin(x[2])*sin(x[3])
                       θ[2]*cos(x[1]) + θ[3]*cos(x[2]) + θ[1] - θ[4]
                       θ[2]*sin(x[1]) + θ[3]*sin(x[2])*cos(x[3])]
    end
    return outmat
end

# ----------------- End of above mentioned visualisation of delta robot -------------------

# ONLY NOMINAL ALGEBRAIC CONSTRAINTS
function delta_constraint_residual(Xmat)
    N = size(Xmat,1)÷num_dyn_vars_adj
    residuals = zeros(6,N)
    residual_norms = zeros(N)

    for ind=1:N
        z = Xmat[(ind-1)*num_dyn_vars_adj+1:ind*num_dyn_vars_adj]
        residuals[1,ind] = (sqrt(3)*(L0-L3+L1*cos(z[4])+L2*cos(z[5])))*0.5+L2*sin(z[2])*sin(z[3])+(L2*sin(z[5])*sin(z[6]))*0.5
        residuals[2,ind] = (3*L0)*0.5-(3*L3)*0.5+L1*cos(z[1])+L2*cos(z[2])+(L1*cos(z[4]))*0.5+(L2*cos(z[5]))*0.5-(sqrt(3)*L2*sin(z[5])*sin(z[6]))*0.5
        residuals[3,ind] = L1*sin(z[1])-L1*sin(z[4])+L2*cos(z[3])*sin(z[2])-L2*cos(z[6])*sin(z[5])
        residuals[4,ind] = L2*sin(z[2])*sin(z[3])-(sqrt(3)*(L0-L3+L1*cos(z[7])+L2*cos(z[8])))*0.5+(L2*sin(z[8])*sin(z[9]))*0.5
        residuals[5,ind] = (3*L0)*0.5-(3*L3)*0.5+L1*cos(z[1])+L2*cos(z[2])+(L1*cos(z[7]))*0.5+(L2*cos(z[8]))*0.5+(sqrt(3)*L2*sin(z[8])*sin(z[9]))*0.5
        residuals[6,ind] = L1*sin(z[1])-L1*sin(z[7])+L2*cos(z[3])*sin(z[2])-L2*cos(z[9])*sin(z[8])
        residual_norms[ind] = norm(residuals[:,ind])
    end
    return residuals, residual_norms
end

# ALL ALGEBRAIC CONSTRAINTS
function delta_constraint_residual2(Xmat)
    N = size(Xmat,1)÷num_dyn_vars_adj
    residuals = zeros(12,N)
    residual_norms = zeros(N)

    for ind=1:N
        z = Xmat[(ind-1)*num_dyn_vars_adj+1:ind*num_dyn_vars_adj]

        residuals[1,ind] = (sqrt(3)*(L0-L3+L1*cos(z[4])+L2*cos(z[5])))*0.5+L2*sin(z[2])*sin(z[3])+(L2*sin(z[5])*sin(z[6]))*0.5
        residuals[2,ind] = (3*L0)*0.5-(3*L3)*0.5+L1*cos(z[1])+L2*cos(z[2])+(L1*cos(z[4]))*0.5+(L2*cos(z[5]))*0.5-(sqrt(3)*L2*sin(z[5])*sin(z[6]))*0.5
        residuals[3,ind] = L1*sin(z[1])-L1*sin(z[4])+L2*cos(z[3])*sin(z[2])-L2*cos(z[6])*sin(z[5])
        residuals[4,ind] = L2*sin(z[2])*sin(z[3])-(sqrt(3)*(L0-L3+L1*cos(z[7])+L2*cos(z[8])))*0.5+(L2*sin(z[8])*sin(z[9]))*0.5
        residuals[5,ind] = (3*L0)*0.5-(3*L3)*0.5+L1*cos(z[1])+L2*cos(z[2])+(L1*cos(z[7]))*0.5+(L2*cos(z[8]))*0.5+(sqrt(3)*L2*sin(z[8])*sin(z[9]))*0.5
        residuals[6,ind] = L1*sin(z[1])-L1*sin(z[7])+L2*cos(z[3])*sin(z[2])-L2*cos(z[9])*sin(z[8])
        residuals[7,ind] = L2*cos(z[2])*sin(z[3])*z[11]-(sqrt(3)*(L1*sin(z[4])*z[13]+L2*sin(z[5])*z[14]))*0.5+L2*cos(z[3])*sin(z[2])*z[12]+(L2*cos(z[5])*sin(z[6])*z[14])*0.5+(L2*cos(z[6])*sin(z[5])*z[15])*0.5
        residuals[8,ind] = -L1*sin(z[1])*z[10]-L2*sin(z[2])*z[11]-(L1*sin(z[4])*z[13])*0.5-(L2*sin(z[5])*z[14])*0.5-(sqrt(3)*L2*cos(z[5])*sin(z[6])*z[14])*0.5-(sqrt(3)*L2*cos(z[6])*sin(z[5])*z[15])*0.5
        residuals[9,ind] = L1*cos(z[1])*z[10]-L1*cos(z[4])*z[13]+L2*cos(z[2])*cos(z[3])*z[11]-L2*cos(z[5])*cos(z[6])*z[14]-L2*sin(z[2])*sin(z[3])*z[12]+L2*sin(z[5])*sin(z[6])*z[15]
        residuals[10,ind] = (sqrt(3)*(L1*sin(z[7])*z[16]+L2*sin(z[8])*z[17]))*0.5+L2*cos(z[2])*sin(z[3])*z[11]+L2*cos(z[3])*sin(z[2])*z[12]+(L2*cos(z[8])*sin(z[9])*z[17])*0.5+(L2*cos(z[9])*sin(z[8])*z[18])*0.5
        residuals[11,ind] = (sqrt(3)*L2*cos(z[8])*sin(z[9])*z[17])*0.5-L2*sin(z[2])*z[11]-(L1*sin(z[7])*z[16])*0.5-(L2*sin(z[8])*z[17])*0.5-L1*sin(z[1])*z[10]+(sqrt(3)*L2*cos(z[9])*sin(z[8])*z[18])*0.5
        residuals[12,ind] = L1*cos(z[1])*z[10]-L1*cos(z[7])*z[16]+L2*cos(z[2])*cos(z[3])*z[11]-L2*cos(z[8])*cos(z[9])*z[17]-L2*sin(z[2])*sin(z[3])*z[12]+L2*sin(z[8])*sin(z[9])*z[18]
        residual_norms[ind] = norm(residuals[:,ind])
    end
    return residuals, residual_norms
end

function solve_accurate_adjoint(N::Int, Ts::Float64, x::Function, dx::Function, x2::Function, y_func::Function, my_ind::Int)
    l2(x,dx,z,t) = (-x(t)[1]/x(t)[2])*z[1]
    l4(x,dx,z,t) = (-x(t)[1]/x(t)[2])*z[2]

    # These are just nicer structured versions of l6 and l5
    l6(x,dx,z,t) = (-(2k*x(t)[1]*abs(x(t)[4]) + m*dx(t)[1])*z[2] - (2k*x(t)[2]*abs(x(t)[5]) + m*dx(t)[2])*l4(x,dx,z,t))/(L^2)
    l5(x,dx,z,t) = -(dx(t)[1]*z[1] + dx(t)[2]*l2(x,dx,z,t) + (x(t)[1]*x(t)[4] + x(t)[2]*x(t)[5])*l6(x,dx,z,t))/(2*L^2)

    # # OLD, THEORETICALLY INCORRECT BUT SOMEHOW WORKING BETTER (Edit: I'm pretty sure it's not working better now that bugs have been fixed)
    # ode_eq(z,p,t) = [-dx(t)[3]*z[2] + (2*x(t)[2]/(T*L^2))*(x2(t)[7]-y_func(t));
    #                  (-z[1] + 2*k*abs(x(t)[4])*z[2])/m]

    # THEORETICALLY CORRECT VERSION
    ode_eq(z,p,t) = [-dx(t)[3]*z[2] + 2*x(t)[1]*l5(x,dx,z,t) + x(t)[4]*l6(x,dx,z,t)     + (2*x(t)[2]/(T*L^2))*(x2(t)[7]-first(y_func(t)));
                     (-z[1] + 2*k*abs(x(t)[4])*z[2] + x(t)[1]*l6(x,dx,z,t))/m]

    T = N*Ts
    span = (T, 0.0)
    zT = zeros(2)
    prob = ODEProblem(ode_eq, zT, span, [])
    sol = DifferentialEquations.solve(prob, Tsit5(), reltol=reltol, abstol=abstol, saveat=0.0:Ts:N*Ts)
    λ1 = [sol.u[end-i+1][1] for i=eachindex(sol.u)]
    λ3 = [sol.u[end-i+1][2] for i=eachindex(sol.u)]
    λ1_func = cubic_spline_interpolation(0.0:Ts:N*Ts, λ1, extrapolation_bc=Line())
    λ3_func = cubic_spline_interpolation(0.0:Ts:N*Ts, λ3, extrapolation_bc=Line())
    λ_func(t)  = [  λ1_func(t);
                    l2(x,dx,[λ1_func(t);λ3_func(t)],t);
                    λ3_func(t);
                    l4(x,dx,[λ1_func(t);λ3_func(t)],t);
                    l5(x,dx,[λ1_func(t);λ3_func(t)],t);
                    l6(x,dx,[λ1_func(t);λ3_func(t)],t);
                    (2/T)*(x2(t)[7]-first(y_func(t)))]

    # TODO: Is this really the best place to do this integration?
    # Wouldn't it be better to do it in ultimate_adjoint_debug???
    # Then we can get rid of the my_int-argument too!
    # -------- Let's also compute λsint -----------
    λsint_func = integrate_lambdas(λ_func, N, N*Ts)
    # i = my_ind
    # λint_eq(x,p,t) = -λ_func(t)[i]
    # λint_prob = ODEProblem(λint_eq, 0.0, (0.0, T), [])
    # λint_sol  = DifferentialEquations.solve(λint_prob, Tsit5(), reltol=reltol, abstol=abstol, saveat=0.0:Ts:N*Ts)
    # λint_vec  = [λint_sol.u[i][1] for i=eachindex(λint_sol)]
    # λint_func = cubic_spline_interpolation(0.0:Ts:N*Ts, λint_vec, extrapolation_bc=Line())

    return λ_func, λsint_func
end

function solve_accurate_adjoint_distsensa(N::Int, Ts::Float64, x::Function, dx::Function, x2::Function, y_func::Function, w::Function, η::Vector{Float64}, my_ind::Int)
    l2(x,dx,z,t) = (-x(t)[1]/x(t)[2])*z[1]
    l4(x,dx,z,t) = (-x(t)[1]/x(t)[2])*z[2]

    # These are just nicer structured versions of l6 and l5
    l6(x,dx,z,t) = (-(2k*x(t)[1]*abs(x(t)[4]) + m*dx(t)[1])*z[2] - (2k*x(t)[2]*abs(x(t)[5]) + m*dx(t)[2])*l4(x,dx,z,t))/(L^2)
    l5(x,dx,z,t) = -(dx(t)[1]*z[1] + dx(t)[2]*l2(x,dx,z,t) + (x(t)[1]*x(t)[4] + x(t)[2]*x(t)[5])*l6(x,dx,z,t))/(2*L^2)

    # # OLD, THEORETICALLY INCORRECT BUT SOMEHOW WORKING BETTER (Edit: I'm pretty sure it's not working better now that bugs have been fixed)
    # ode_eq(z,p,t) = [-dx(t)[3]*z[2] + (2*x(t)[2]/(T*L^2))*(x2(t)[7]-y_func(t));
    #                  (-z[1] + 2*k*abs(x(t)[4])*z[2])/m]

    # THEORETICALLY CORRECT VERSION
    ode_eq(z,p,t) = [-dx(t)[3]*z[2] + 2*x(t)[1]*l5(x,dx,z,t) + x(t)[4]*l6(x,dx,z,t)     + (2*x(t)[2]/(T*L^2))*(x2(t)[7]-y_func(t));
                     (-z[1] + 2*k*abs(x(t)[4])*z[2] + x(t)[1]*l6(x,dx,z,t))/m]

    T = N*Ts
    span = (T, 0.0)
    zT = zeros(2)
    prob = ODEProblem(ode_eq, zT, span, [])
    sol = DifferentialEquations.solve(prob, Tsit5(), reltol=reltol, abstol=abstol, saveat=0.0:Ts:N*Ts)
    λ1 = [sol.u[end-i+1][1] for i=eachindex(sol.u)]
    λ3 = [sol.u[end-i+1][2] for i=eachindex(sol.u)]
    λ1_func = cubic_spline_interpolation(0.0:Ts:N*Ts, λ1, extrapolation_bc=Line())
    λ3_func = cubic_spline_interpolation(0.0:Ts:N*Ts, λ3, extrapolation_bc=Line())

    λ10_func(t) = 2*w(t)[1]*λ3_func(t)
    ode_eq_dist(z,p,t) = [η[1]*z[1] - z[2] - η[3]*λ10_func(t)
                          η[2]*z[1] - η[4]*λ10_func(t)]
    prob_dist = ODEProblem(ode_eq_dist, zeros(2), span, [])
    sol_dist = DifferentialEquations.solve(prob_dist, Tsit5(), reltol=reltol, abstol=abstol, saveat=0.0:Ts:N*Ts)
    λ8 = [sol_dist.u[end-i+1][1] for i=eachindex(sol_dist.u)]
    λ9 = [sol_dist.u[end-i+1][2] for i=eachindex(sol_dist.u)]
    λ8_func = cubic_spline_interpolation(0.0:Ts:N*Ts, λ8, extrapolation_bc=Line())
    λ9_func = cubic_spline_interpolation(0.0:Ts:N*Ts, λ9, extrapolation_bc=Line())

    λ_func(t)  = [  λ1_func(t);
                    l2(x,dx,[λ1_func(t);λ3_func(t)],t);
                    λ3_func(t);
                    l4(x,dx,[λ1_func(t);λ3_func(t)],t);
                    l5(x,dx,[λ1_func(t);λ3_func(t)],t);
                    l6(x,dx,[λ1_func(t);λ3_func(t)],t);
                    (2/T)*(x2(t)[7]-y_func(t));
                    λ8_func(t);
                    λ9_func(t);
                    λ10_func(t)]

    # TODO: Is this really the best place to do this integration?
    # Wouldn't it be better to do it in ultimate_adjoint_debug???
    # Then we can get rid of the my_int-argument too!
    # -------- Let's also compute λsint -----------
    λsint_func = integrate_lambdas_distsens(λ_func, N, N*Ts)
    # i = my_ind
    # λint_eq(x,p,t) = -λ_func(t)[i]
    # λint_prob = ODEProblem(λint_eq, 0.0, (0.0, T), [])
    # λint_sol  = DifferentialEquations.solve(λint_prob, Tsit5(), reltol=reltol, abstol=abstol, saveat=0.0:Ts:N*Ts)
    # λint_vec  = [λint_sol.u[i][1] for i=eachindex(λint_sol)]
    # λint_func = cubic_spline_interpolation(0.0:Ts:N*Ts, λint_vec, extrapolation_bc=Line())

    return λ_func, λsint_func
end

function integrate_lambdas(λs_func, N, T)
    λint_eq(x,p,t) = -λs_func(t)
    λint_prob = ODEProblem(λint_eq, zeros(7), (0.0, T), [])
    λint_sol  = DifferentialEquations.solve(λint_prob, Tsit5(), reltol=reltol, abstol=abstol, saveat=0.0:Ts:N*Ts)
    λint_vec1 = [λint_sol.u[i][1] for i=eachindex(λint_sol)]
    λint_vec2 = [λint_sol.u[i][2] for i=eachindex(λint_sol)]
    λint_vec3 = [λint_sol.u[i][3] for i=eachindex(λint_sol)]
    λint_vec4 = [λint_sol.u[i][4] for i=eachindex(λint_sol)]
    λint_vec5 = [λint_sol.u[i][5] for i=eachindex(λint_sol)]
    λint_vec6 = [λint_sol.u[i][6] for i=eachindex(λint_sol)]
    λint_vec7 = [λint_sol.u[i][7] for i=eachindex(λint_sol)]
    λint_func1 = cubic_spline_interpolation(0.0:Ts:N*Ts, λint_vec1, extrapolation_bc=Line())
    λint_func2 = cubic_spline_interpolation(0.0:Ts:N*Ts, λint_vec2, extrapolation_bc=Line())
    λint_func3 = cubic_spline_interpolation(0.0:Ts:N*Ts, λint_vec3, extrapolation_bc=Line())
    λint_func4 = cubic_spline_interpolation(0.0:Ts:N*Ts, λint_vec4, extrapolation_bc=Line())
    λint_func5 = cubic_spline_interpolation(0.0:Ts:N*Ts, λint_vec5, extrapolation_bc=Line())
    λint_func6 = cubic_spline_interpolation(0.0:Ts:N*Ts, λint_vec6, extrapolation_bc=Line())
    λint_func7 = cubic_spline_interpolation(0.0:Ts:N*Ts, λint_vec7, extrapolation_bc=Line())
    λsint_func(t) = [λint_func1(t); λint_func2(t); λint_func3(t); λint_func4(t); λint_func5(t); λint_func6(t); λint_func7(t)]
    return λsint_func
end

function integrate_lambdas_distsens(λs_func, N, T)
    λint_eq(x,p,t) = -λs_func(t)
    λint_prob = ODEProblem(λint_eq, zeros(10), (0.0, T), [])
    λint_sol  = DifferentialEquations.solve(λint_prob, Tsit5(), reltol=reltol, abstol=abstol, saveat=0.0:Ts:N*Ts)
    λint_vec1 = [λint_sol.u[i][1] for i=eachindex(λint_sol)]
    λint_vec2 = [λint_sol.u[i][2] for i=eachindex(λint_sol)]
    λint_vec3 = [λint_sol.u[i][3] for i=eachindex(λint_sol)]
    λint_vec4 = [λint_sol.u[i][4] for i=eachindex(λint_sol)]
    λint_vec5 = [λint_sol.u[i][5] for i=eachindex(λint_sol)]
    λint_vec6 = [λint_sol.u[i][6] for i=eachindex(λint_sol)]
    λint_vec7 = [λint_sol.u[i][7] for i=eachindex(λint_sol)]
    λint_vec8 = [λint_sol.u[i][8] for i=eachindex(λint_sol)]
    λint_vec9 = [λint_sol.u[i][9] for i=eachindex(λint_sol)]
    λint_vec10 = [λint_sol.u[i][10] for i=eachindex(λint_sol)]
    λint_func1 = cubic_spline_interpolation(0.0:Ts:N*Ts, λint_vec1, extrapolation_bc=Line())
    λint_func2 = cubic_spline_interpolation(0.0:Ts:N*Ts, λint_vec2, extrapolation_bc=Line())
    λint_func3 = cubic_spline_interpolation(0.0:Ts:N*Ts, λint_vec3, extrapolation_bc=Line())
    λint_func4 = cubic_spline_interpolation(0.0:Ts:N*Ts, λint_vec4, extrapolation_bc=Line())
    λint_func5 = cubic_spline_interpolation(0.0:Ts:N*Ts, λint_vec5, extrapolation_bc=Line())
    λint_func6 = cubic_spline_interpolation(0.0:Ts:N*Ts, λint_vec6, extrapolation_bc=Line())
    λint_func7 = cubic_spline_interpolation(0.0:Ts:N*Ts, λint_vec7, extrapolation_bc=Line())
    λint_func8 = cubic_spline_interpolation(0.0:Ts:N*Ts, λint_vec8, extrapolation_bc=Line())
    λint_func9 = cubic_spline_interpolation(0.0:Ts:N*Ts, λint_vec9, extrapolation_bc=Line())
    λint_func10 = cubic_spline_interpolation(0.0:Ts:N*Ts, λint_vec10, extrapolation_bc=Line())
    λsint_func(t) = [λint_func1(t); λint_func2(t); λint_func3(t); λint_func4(t); λint_func5(t); λint_func6(t); λint_func7(t); λint_func8(t); λint_func9(t); λint_func10(t)]
    return λsint_func
end

function linear_interpolation(y::AbstractVector, Ts::Float64)
    max_n = length(y)-2
    function y_func(t::Float64)
        n = min(Int(t÷Ts), max_n)
        return ( ((n+1)*Ts-t)*y[n+1] .+ (t-n*Ts)*y[n+2])./Ts
    end
end

function linear_interpolation_multivar(y::AbstractVector, Ts::Float64, ny::Int)
    max_n = length(y)÷ny-2
    function y_func(t::Float64)
        n = min(Int(t÷Ts), max_n)
        return ( ((n+1)*Ts-t)*y[n*ny+1:(n+1)*ny] .+ (t-n*Ts)*y[(n+1)*ny+1:(n+2)*ny])./Ts
    end
end

function simulate_experiment(expid::String, pars::Vector{Float64}, Zm::Array{Matrix{Float64},1})
    exp_data, isws = get_experiment_data(expid)
    W_meta = exp_data.W_meta
    u = exp_data.u
    Y = exp_data.Y
    N = size(Y,1)÷y_len-1
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

function simulate_experiment(expid::String, pars::Vector{Float64})
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

function read_from_backup(dir::String, E::Int)
    sample = readdlm(joinpath(dir, "backup_trace_e1.csv"), ',')
    (its,k) = size(sample)
    opt_pars_proposed = zeros(k, E)
    avg_pars_proposed = zeros(k, E)
    trace_proposed = [zeros(its, k) for e=1:E]
    for e=1:E
        opt_pars_proposed[:,e] = readdlm(joinpath(dir, "backup_proposed_e$e.csv"), ',')
        avg_pars_proposed[:,e] = readdlm(joinpath(dir, "backup_average_e$e.csv"), ',')
        trace_proposed[e] = readdlm(joinpath(dir, "backup_trace_e$e.csv"), ',')
    end
    return opt_pars_proposed, avg_pars_proposed, trace_proposed
end

# NOTE: Only works for baeline using SGD, otherwise its is not constant over the different experiments
function read_baseline_from_backup(dir::String, E::Int)
    sample = readdlm(joinpath(dir, "backup_baseline_trace_e1.csv"), ',')
    # sample = readdlm(joinpath(dir, "backup_proposed_e1.csv"), ',')
    # sample = readdlm(joinpath(dir, "backup_trace_e1.csv"), ',')
    (its,k) = size(sample)
    # its = 1
    # k = length(sample)
    opt_pars_baseline = zeros(k, E)
    # opt_pars_proposed = zeros(k, E)
    # avg_pars_proposed = zeros(k, E)
    trace_baseline = [zeros(its, k) for e=1:E]
    for e=1:E
        opt_pars_baseline[:,e] = readdlm(joinpath(dir, "backup_baseline_e$e.csv"), ',')
        trace_baseline[e] = readdlm(joinpath(dir, "backup_baseline_trace_e$e.csv"), ',')
    end
    return opt_pars_baseline, trace_baseline
end

function extract_part_results_from_trace(dir::String, E::Int, checkpoint::Int)
    sample = readdlm(joinpath(dir, "backup_trace_e1.csv"), ',')
    (its,k) = size(sample)
    # trace_proposed = [zeros(its, k) for e=1:E]
    check_val = zeros(k,E)
    end_val = zeros(k,E)
    for e=1:E
        trace_proposed = readdlm(joinpath(dir, "backup_trace_e$e.csv"), ',')
        check_val[:,e] = trace_proposed[checkpoint,:]
        end_val[:,e] = trace_proposed[end,:]
    end
    return check_val, end_val
end

function parameter_separated_results(opt_pars::Matrix{Float64}, names::Vector{String}, dir::String)
    for parind=eachindex(opt_pars[:,1])
        # writedlm(joinpath(dir, "$(names[parind])_base.csv"), opt_pars[parind,1:end .!= 28], ",")   # Getting rid of an index that baseline happened to fail for
        # writedlm(joinpath(dir, "$(names[parind])_vals2.csv"), opt_pars[parind,:], ",")
        writedlm(joinpath(dir, "$(names[parind])_vals3.csv"), hcat(opt_pars[parind,:],opt_pars[parind,:],opt_pars[parind,:],opt_pars[parind,:]), ",")
    end
end

function gridsearch_delta_wrapper(expid::String, savedir::String)
    krange = [1]#, 2, 4, 10, 40]
    pars, prop_vals, base_vals = gridsearch_delta2(expid, krange[1], 0)
    # pars, prop_vals, base_vals = gridsearch_delta(expid, krange[1], 0)
    # all_pars = [zeros(size(pars)) for k = krange]
    # all_pars[1] = pars
    all_vals = [[zeros(size(prop_vals[1])) for ind = eachindex(prop_vals)] for k=eachindex(krange)]
    all_base = [[zeros(size(base_vals[1])) for ind = eachindex(base_vals)] for k=eachindex(krange)]
    for ind = eachindex(prop_vals)
        all_vals[1][ind] = prop_vals[ind]
        all_base[1][ind] = base_vals[ind]
    end
    try
        for it = 2:length(krange)
            @info "EXPERIMENT FOR k=$(krange[it])"
            # pars, prop_vals, base_vals = gridsearch_delta(expid, krange[it], 0)   # Same parameters investigated for every e
            pars, prop_vals, base_vals = gridsearch_delta2(expid, krange[it], 0)    # Different parameters investigated for every e
            # all_pars[it] = pars
            for ind = 1:length(prop_vals)
                all_vals[it][ind] = prop_vals[ind]
                all_base[it][ind] = base_vals[ind]
            end
        end
    catch ex
        writedlm("data/experiments/tmp/delta_all_pars.csv", pars, ',')
        writedlm("data/experiments/tmp/delta_all_vals.csv", all_vals, ',')
        writedlm("data/experiments/tmp/delta_all_base.csv", all_base, ',')
        throw(ex)
    end

    save_wrapped_gridsearch(savedir, krange, pars, all_vals, all_base)

    # return all_pars, all_vals, all_base
    return pars, all_vals, all_base
end

function save_wrapped_gridsearch(dirname::String, krange, pars, all_vals, all_base)
    nk = length(all_vals)
    E  = length(all_vals[1])

    writedlm("data/results/$(dirname)/all_pars.csv", pars, ',')
    for (ik, k) = enumerate(krange)
        for (ie, e) = enumerate(1:E)
            writedlm("data/results/$(dirname)/all_vals_N$(100k)_e$(e).csv",  all_vals[ik][e],  ',')
            writedlm("data/results/$(dirname)/all_base_N$(100k)_e$(e).csv", all_base[ik][e], ',')
        end
    end
end

function gridsearch_delta(expid::String, K::Int = 1, N_trans::Int = 0)
    exp_data, isws = get_experiment_data(expid)
    W_meta = exp_data.W_meta
    Y = exp_data.Y
    N = size(Y,1)÷y_len-1
    Nw = exp_data.Nw
    nx = W_meta.nx
    n_in = W_meta.n_in
    n_out = W_meta.n_out
    n_tot = nx*n_in
    dη = length(W_meta.η)
    u = exp_data.u
    par_bounds = vcat(dyn_par_bounds, exp_data.dist_par_bounds)
    dist_sens_inds = W_meta.free_par_inds

    get_all_parameters(pars::Vector{Float64}) = vcat(get_all_θs(pars), exp_data.get_all_ηs(pars))

    # Total data-set will be of length K*N, so K number of smaller data-sets will be used to compute cost
    # E = size(Y, 2)
    # DEBUG
    E = 3
    myM = 24#100#8
    Zm = [randn(Nw, n_tot) for m = 1:myM*K]
    # myZeros = [zeros(Nw, n_tot) for k=1:K]
    # Zm = myZeros
    # @warn "USING ZERO DISTURBANCE! And thus M=1 instead of M=100"

    # ref = 0.0#free_dyn_pars_true[1]
    # myδ = 0.1
    # # # vals = ref-10myδ:myδ:ref+10myδ
    # # vals = 0.5:0.1:1.6
    # # # # vals = ref-myδ:myδ:ref+myδ
    # # vals = 0.5:0.25:1.5
    # vals = 6.25-2*1.0:1.0:6.25+2*1.0
    # vals = 0.9:0.5:10.0
    vals = 0.5:0.2:5.0

    cost_vals = [zeros(length(vals)) for e=1:E]
    cost_base = [zeros(length(vals)) for e=1:E]
    grad_vals = [zeros(length(vals)) for e=1:E]
    all_pars = zeros(length(free_dyn_pars_true), length(vals))
    Ym = zeros(size(Y[N_trans+1:end,1], 1), K)
    time_start = now()
    # TODO: YOUR N_TRANS IMPLEMENTATION EVERYWHERE DOESN'T WORK FOR ny > 1!!!!!!!!! YOU NEED TO FIX THAT, OR REMOVE IT!
    for (ind, my_par) in enumerate(vals)
        # for e = 1:E
            e=3
            @info "Computing cost for p=$my_par, or after clamping, p=$(clamp(my_par, dyn_par_bounds[1], dyn_par_bounds[2]))"
            # pars = [clamp(my_par, dyn_par_bounds[1], dyn_par_bounds[2])]
            pars = [my_par]

            all_pars[:,ind] = pars
            Ysim = simulate_system(exp_data, pars, myM*K, dist_sens_inds, isws, Zm)
            # TODO: Since each realization of the baseline should be the same, since it's deterministic, it really should be enough to simulate only 1
            # Ybase = simulate_system(exp_data, pars, K, dist_sens_inds, isws, myZeros)
            for k = 1:K
                Ym[:,k] = mean(Ysim[:,(k-1)*myM+1:k*myM], dims=2)
                cost_vals[e][ind] += mean((Y[N_trans+1:end, (e-1)*K+k].-Ym[N_trans+1:end,k]).^2)
                # cost_base[e][ind] += mean((Y[N_trans+1:end, (e-1)*K+k].-Ybase[N_trans+1:end,k]).^2)
            end
            cost_vals[e][ind] = cost_vals[e][ind]/K
            
            # cost_base[e][ind] = cost_base[e][ind]/K
            # Ym = mean(simulate_system(exp_data, pars, myM, dist_sens_inds, isws, Zm), dims=2)

            # # BONUS: Computing cost function gradient
            # Ymsens, jacsYm = simulate_system_sens(exp_data, pars, 1, dist_sens_inds, isws)
            # grad_vals[e][ind] = first(get_cost_gradient(Y[N_trans+1:end, e], Ymsens[:,1:1], jacsYm[1:1], N_trans))
            @info "Completed computing cost for e = $e out of $E, ind =  $ind out of $(length(vals))"
        # end
        writedlm("data/experiments/tmp/backup_cost_vals_par$ind.csv", [cost_vals[e][ind] for e=1:E], ',')
    end
    duration = now()-time_start

    return all_pars, cost_vals, cost_base, duration#, grad_vals
end

function gridsearch_baseline(expid::String, N_trans::Int = 0)
    exp_data, isws = get_experiment_data(expid)
    W_meta = exp_data.W_meta
    Y = exp_data.Y
    N = size(Y,1)÷y_len-1
    Nw = exp_data.Nw
    nx = W_meta.nx
    n_in = W_meta.n_in
    n_out = W_meta.n_out
    n_tot = nx*n_in
    dη = length(W_meta.η)
    u = exp_data.u
    par_bounds = vcat(dyn_par_bounds, exp_data.dist_par_bounds)
    dist_sens_inds = W_meta.free_par_inds

    get_all_parameters(pars::Vector{Float64}) = vcat(get_all_θs(pars), exp_data.get_all_ηs(pars))

    E = 3

    ref = 0.0#free_dyn_pars_true[1]
    myδ = 0.1
    # vals = ref-10myδ:myδ:ref+10myδ
    # vals = 0.5:0.05:3.5 # RAN ALREADY
    # vals = 3.6:0.05:10.0
    vals = 0.5:0.2:5.0
    # # vals = ref-myδ:myδ:ref+myδ

    # cost_vals = [zeros(length(vals)) for e=1:E]
    cost_base = [zeros(length(vals)) for e=1:E]
    # grad_vals = [zeros(length(vals)) for e=1:E]
    all_pars = zeros(length(free_dyn_pars_true), length(vals))
    time_start = now()
    # TODO: YOUR N_TRANS IMPLEMENTATION EVERYWHERE DOESN'T WORK FOR ny > 1!!!!!!!!! YOU NEED TO FIX THAT, OR REMOVE IT!
    for (ind, my_par) in enumerate(vals)
        # for e = 1:E
            e=3
            @info "Computing cost for p=$my_par, or after clamping, p=$(clamp(my_par, dyn_par_bounds[1], dyn_par_bounds[2]))"
            pars = [clamp(my_par, dyn_par_bounds[1], dyn_par_bounds[2])]

            all_pars[:,ind] = pars
            Ybase = solvew(exp_data.u, t -> zeros(3), pars, N) |> sol -> vcat(h(sol,get_all_θs(pars))...)
            cost_base[e][ind] += mean((Y[N_trans+1:end, e].-Ybase[N_trans+1:end]).^2)
            
            # cost_base[e][ind] = cost_base[e][ind]/K
            # Ym = mean(simulate_system(exp_data, pars, myM, dist_sens_inds, isws, Zm), dims=2)

            # # BONUS: Computing cost function gradient
            # Ymsens, jacsYm = simulate_system_sens(exp_data, pars, 1, dist_sens_inds, isws)
            # grad_vals[e][ind] = first(get_cost_gradient(Y[N_trans+1:end, e], Ymsens[:,1:1], jacsYm[1:1], N_trans))
            @info "Completed computing cost for e = $e out of $E, ind =  $ind out of $(length(vals))"
        # end
        writedlm("data/experiments/tmp/backup_cost_base_par$ind.csv", [cost_base[e][ind] for e=1:E], ',')
    end
    duration = now()-time_start

    return all_pars, cost_base, duration#, grad_vals
end

function delta_traj_debug(expid::String, pars::Vector{Float64})
    exp_data, isws = get_experiment_data(expid)
    W_meta = exp_data.W_meta
    Y = exp_data.Y
    N = size(Y,1)÷y_len-1
    Nw = exp_data.Nw
    nx = W_meta.nx
    n_in = W_meta.n_in
    n_out = W_meta.n_out
    n_tot = nx*n_in
    dη = length(W_meta.η)
    u = exp_data.u
    par_bounds = vcat(dyn_par_bounds, exp_data.dist_par_bounds)
    dist_sens_inds = W_meta.free_par_inds

    get_all_parameters(pars::Vector{Float64}) = vcat(get_all_θs(pars), exp_data.get_all_ηs(pars))


    Ybase = solvew(u, t -> zeros(3), pars, N) |> sol -> vcat(h(sol,get_all_θs(pars))...)
    myM = 1
    Zm = [randn(Nw, n_tot) for m = 1:myM]
    Ysim = simulate_system(exp_data, pars, myM, dist_sens_inds, isws, Zm)

    return Ybase, Ysim
end

function gridsearch_2d_baseline(expid::String, N_trans::Int = 0)
    exp_data, isws = get_experiment_data(expid)
    W_meta = exp_data.W_meta
    Y = exp_data.Y
    N = size(Y,1)÷y_len-1
    Nw = exp_data.Nw
    nx = W_meta.nx
    n_in = W_meta.n_in
    n_out = W_meta.n_out
    n_tot = nx*n_in
    dη = length(W_meta.η)
    u = exp_data.u
    par_bounds = vcat(dyn_par_bounds, exp_data.dist_par_bounds)
    dist_sens_inds = W_meta.free_par_inds

    get_all_parameters(pars::Vector{Float64}) = vcat(get_all_θs(pars), exp_data.get_all_ηs(pars))

    # E = 25

    vals1 = 0.9:0.01:1.1
    # vals2 = 0.9:0.05:2.0
    vals2 = 2.05:0.05:5.0

    cost_base = zeros(length(vals2), length(vals1))

    e = 3
    for (ind1, p1) in enumerate(vals1)
        for (ind2, p2) in enumerate(vals2)
            pars = [p1, p2]
            Ybase = solvew(exp_data.u, t -> zeros(3), pars, N) |> sol -> vcat(h(sol,get_all_θs(pars))...)
            cost_base[ind2, ind1] += mean((Y[N_trans+1:end, e].-Ybase[N_trans+1:end]).^2)
            @info "Completed computing cost for ind1=$ind1/$(length(vals1)), ind2=$ind2/$(length(vals2))"
        end
    end
    return vals1, vals2, cost_base

    # # cost_vals = [zeros(length(vals)) for e=1:E]
    # cost_base = [zeros(length(vals)) for e=1:E]
    # # grad_vals = [zeros(length(vals)) for e=1:E]
    # all_pars = zeros(length(free_dyn_pars_true), length(vals))
    # time_start = now()
    # # TODO: YOUR N_TRANS IMPLEMENTATION EVERYWHERE DOESN'T WORK FOR ny > 1!!!!!!!!! YOU NEED TO FIX THAT, OR REMOVE IT!
    # for (ind, my_par) in enumerate(vals)
    #     for e = 1:E
    #         @info "Computing cost for p=$my_par, or after clamping, p=$(clamp(my_par, dyn_par_bounds[1], dyn_par_bounds[2]))"
    #         pars = [clamp(my_par, dyn_par_bounds[1], dyn_par_bounds[2])]

    #         all_pars[:,ind] = pars
    #         Ybase = solvew(exp_data.u, t -> zeros(3), pars, N) |> sol -> vcat(h(sol,get_all_θs(pars))...)
    #         cost_base[e][ind] += mean((Y[N_trans+1:end, e].-Ybase[N_trans+1:end]).^2)
            
    #         # cost_base[e][ind] = cost_base[e][ind]/K
    #         # Ym = mean(simulate_system(exp_data, pars, myM, dist_sens_inds, isws, Zm), dims=2)

    #         # # BONUS: Computing cost function gradient
    #         # Ymsens, jacsYm = simulate_system_sens(exp_data, pars, 1, dist_sens_inds, isws)
    #         # grad_vals[e][ind] = first(get_cost_gradient(Y[N_trans+1:end, e], Ymsens[:,1:1], jacsYm[1:1], N_trans))
    #         @info "Completed computing cost for e = $e out of $E, ind =  $ind out of $(length(vals))"
    #     end
    #     writedlm("data/experiments/tmp/backup_cost_base_par$ind.csv", [cost_base[e][ind] for e=1:E], ',')
    # end
    # duration = now()-time_start

    # return all_pars, cost_base, duration#, grad_vals
end

function gridsearch_baseline_with_traj(expid::String, N_trans::Int = 0)
    exp_data, isws = get_experiment_data(expid)
    W_meta = exp_data.W_meta
    Y = exp_data.Y
    N = size(Y,1)÷y_len-1
    Nw = exp_data.Nw
    nx = W_meta.nx
    n_in = W_meta.n_in
    n_out = W_meta.n_out
    n_tot = nx*n_in
    dη = length(W_meta.η)
    u = exp_data.u
    par_bounds = vcat(dyn_par_bounds, exp_data.dist_par_bounds)
    dist_sens_inds = W_meta.free_par_inds

    get_all_parameters(pars::Vector{Float64}) = vcat(get_all_θs(pars), exp_data.get_all_ηs(pars))

    E = 5#25

    ref = 0.0#free_dyn_pars_true[1]
    myδ = 0.1
    # vals = ref-10myδ:myδ:ref+10myδ
    # vals = 0.5:0.05:3.5 # RAN ALREADY
    # vals = 3.6:0.05:10.0
    vals = 0.5:0.5:12.0
    # # vals = ref-myδ:myδ:ref+myδ

    # cost_vals = [zeros(length(vals)) for e=1:E]
    cost_base = [zeros(length(vals)) for e=1:E]
    # grad_vals = [zeros(length(vals)) for e=1:E]
    all_pars = zeros(length(free_dyn_pars_true), length(vals))
    time_start = now()
    Ybase_log = [zeros(y_len*(N+1), length(vals)) for e=1:E]
    # TODO: YOUR N_TRANS IMPLEMENTATION EVERYWHERE DOESN'T WORK FOR ny > 1!!!!!!!!! YOU NEED TO FIX THAT, OR REMOVE IT!
    for (ind, my_par) in enumerate(vals)
        for e = 1:E
            @info "Computing cost for p=$my_par, or after clamping, p=$(clamp(my_par, dyn_par_bounds[1], dyn_par_bounds[2]))"
            pars = [clamp(my_par, dyn_par_bounds[1], dyn_par_bounds[2])]

            all_pars[:,ind] = pars
            Ybase = solvew(exp_data.u, t -> zeros(3), pars, N) |> sol -> vcat(h(sol,get_all_θs(pars))...)
            Ybase_log[e][:,ind] = Ybase
            cost_base[e][ind] += mean((Y[N_trans+1:end, e].-Ybase[N_trans+1:end]).^2)
            
            # cost_base[e][ind] = cost_base[e][ind]/K
            # Ym = mean(simulate_system(exp_data, pars, myM, dist_sens_inds, isws, Zm), dims=2)

            # # BONUS: Computing cost function gradient
            # Ymsens, jacsYm = simulate_system_sens(exp_data, pars, 1, dist_sens_inds, isws)
            # grad_vals[e][ind] = first(get_cost_gradient(Y[N_trans+1:end, e], Ymsens[:,1:1], jacsYm[1:1], N_trans))
            @info "Completed computing cost for e = $e out of $E, ind =  $ind out of $(length(vals))"
        end
        writedlm("data/experiments/tmp/backup_cost_base_par$ind.csv", [cost_base[e][ind] for e=1:E], ',')
    end
    duration = now()-time_start

    return all_pars, cost_base, duration, Ybase_log, Y
end

# Different parameter values used for each e
function gridsearch_delta2(expid::String, K::Int = 1, N_trans::Int = 0)
    exp_data, isws = get_experiment_data(expid)
    W_meta = exp_data.W_meta
    Y = exp_data.Y
    N = size(Y,1)÷y_len-1
    Nw = exp_data.Nw
    nx = W_meta.nx
    n_in = W_meta.n_in
    n_out = W_meta.n_out
    n_tot = nx*n_in
    dη = length(W_meta.η)
    u = exp_data.u
    par_bounds = vcat(dyn_par_bounds, exp_data.dist_par_bounds)
    dist_sens_inds = W_meta.free_par_inds

    get_all_parameters(pars::Vector{Float64}) = vcat(get_all_θs(pars), exp_data.get_all_ηs(pars))

    # Total data-set will be of length K*N, so K number of smaller data-sets will be used to compute cost
    # E = size(Y, 2)
    # DEBUG
    E = 1#0#0
    myM = 1#000#8
    Zm = [randn(Nw, n_tot) for m = 1:myM*K]
    myZeros = [zeros(Nw, n_tot) for k=1:K]

    # NEW
    # step = 0.01
    # num_steps = 5
    # middles = [1.5294175755234023,1.5697144775005665,1.3005878253282788,1.5739717409753844,1.5533817724346919,1.455981221009796,1.502854138838781,1.5368594086139387,1.5226786385094824,1.4884874418509237]
    # vals = [mid-num_steps*step:step:mid+num_steps*step for mid=middles]
    vals = [1.1:0.1:2.0]

    cost_vals = [zeros(length(vals[1])) for e=1:E]
    cost_base = [zeros(length(vals[1])) for e=1:E]
    grad_vals = [zeros(length(vals[1])) for e=1:E]
    all_pars = zeros(length(vals[1]), E)
    Ym = zeros(size(Y[N_trans+1:end,1], 1), K)
    # -------------------------- DEBUG ALSO COMPUTING GRADIENTS
    jacs = zeros(size(Y[N_trans+1:end,1], 1), K)
    # -------------------------- END OF DEBUG
    ind = 1
    time_start = now()
    # TODO: YOUR N_TRANS IMPLEMENTATION EVERYWHERE DOESN'T WORK FOR ny > 1!!!!!!!!! YOU NEED TO FIX THAT, OR REMOVE IT!
    for e = 1:E
        writedlm("data/experiments/tmp/backup_pars_e$e.csv", collect(vals[e]), ',')
        for (ind, my_par) in enumerate(vals[e])
            @info "Computing cost for p=$my_par, or after clamping, p=$(clamp(my_par, dyn_par_bounds[1], dyn_par_bounds[2]))"
            pars = [clamp(my_par, dyn_par_bounds[1], dyn_par_bounds[2])]

            all_pars[ind,e] = pars[1]
            # -------------------------- DEBUG ALSO COMPUTING GRADIENTS
            Ysim, jacsim = simulate_system_sens(exp_data, pars, myM*K, dist_sens_inds, isws, Zm)
            # -------------------------- END OF DEBUG
            # Ysim = simulate_system(exp_data, pars, myM*K, dist_sens_inds, isws, Zm)
            # TODO: Since each realization of the baseline should be the same, since it's deterministic, it really should be enough to simulate only 1
            # Ybase = simulate_system(exp_data, pars, K, dist_sens_inds, isws, myZeros)
            for k = 1:K
                Ym[:,k] = mean(Ysim[:,(k-1)*myM+1:k*myM], dims=2)
                cost_vals[e][ind] += mean((Y[N_trans+1:end, (e-1)*K+k].-Ym[N_trans+1:end,k]).^2)
                # cost_base[e][ind] += mean((Y[N_trans+1:end, (e-1)*K+k].-Ybase[N_trans+1:end,k]).^2)
                # -------------------------- DEBUG ALSO COMPUTING GRADIENTS
                # jacs[:,k] = mean(jacsim[1][:,(k-1)*myM+1:k*myM], dims=2)    # DELETE: Old and wrong
                jacs[:,k] = mean(jacsim[(k-1)*myM+1:k*myM][:])
                # -------------------------- END OF DEBUG
            end
            cost_vals[e][ind] = cost_vals[e][ind]/K
            # cost_base[e][ind] = cost_base[e][ind]/K
            # Ym = mean(simulate_system(exp_data, pars, myM, dist_sens_inds, isws, Zm), dims=2)
            # -------------------------- DEBUG ALSO COMPUTING GRADIENTS
            grad_vals[e][ind] = get_cost_gradient(Y[N_trans+1:end, (e-1)*K+1:e*K][:], reshape(Ym[:], length(Ym), 1), [reshape(jacs[:], length(jacs), 1)])[1]
            # -------------------------- END OF DEBUG

            # # BONUS: Computing cost function gradient
            # Ymsens, jacsYm = simulate_system_sens(exp_data, pars, 1, dist_sens_inds, isws)
            # grad_vals[e][ind] = first(get_cost_gradient(Y[N_trans+1:end, e], Ymsens[:,1:1], jacsYm[1:1], N_trans))
            @info "Completed computing cost for e = $e out of $E, ind =  $ind out of $(length(vals[1]))"
        end
        writedlm("data/experiments/tmp/backup_cost_vals_e$e.csv", cost_vals[e], ',')
        # -------------------------- DEBUG ALSO COMPUTING GRADIENTS
        writedlm("data/experiments/tmp/backup_grad_vals_e$e.csv", grad_vals[e], ',')
        # -------------------------- END OF DEBUG
    end
    duration = now()-time_start

    return all_pars, cost_vals, cost_base, duration#, grad_vals
end

function gridsearch_delta_2d(expid::String, K::Int = 1, N_trans::Int = 0)
    exp_data, isws = get_experiment_data(expid)
    W_meta = exp_data.W_meta
    Y = exp_data.Y
    N = size(Y,1)÷y_len-1
    Nw = exp_data.Nw
    nx = W_meta.nx
    n_in = W_meta.n_in
    n_out = W_meta.n_out
    n_tot = nx*n_in
    dη = length(W_meta.η)
    u = exp_data.u
    par_bounds = vcat(dyn_par_bounds, exp_data.dist_par_bounds)
    dist_sens_inds = W_meta.free_par_inds

    get_all_parameters(pars::Vector{Float64}) = vcat(get_all_θs(pars), exp_data.get_all_ηs(pars))

    # Total data-set will be of length K*N, so K number of smaller data-sets will be used to compute cost
    # E = size(Y, 2)
    # DEBUG
    E = 1#00
    myM = 100#8
    Zm = [randn(Nw, n_tot) for m = 1:myM*K]
    myZeros = [zeros(Nw, n_tot) for k=1:K]

    ref1 = 1.5#free_dyn_pars_true[1]
    ref2 = 0.75
    myδ1 = 0.1
    myδ2 = 0.1
    # vals1 = ref1-1myδ1:myδ1:ref1+10myδ1
    # vals2 = ref2-1myδ2:myδ2:ref2+10myδ2
    vals1 = ref1-10myδ1:myδ1:ref1+10myδ1
    vals2 = ref2-5myδ2:myδ2:ref2+5myδ2

    cost_vals = [zeros(length(vals2), length(vals1)) for e=1:E]
    cost_base = [zeros(length(vals2), length(vals1)) for e=1:E]
    grad_vals = [zeros(length(vals2), length(vals1)) for e=1:E]
    # all_pars = zeros(length(free_dyn_pars_true), length(vals))
    Ym = zeros(size(Y[N_trans+1:end,1], 1), K)
    time_start = now()
    # TODO: YOUR N_TRANS IMPLEMENTATION EVERYWHERE DOESN'T WORK FOR ny > 1!!!!!!!!! YOU NEED TO FIX THAT, OR REMOVE IT!
    for (ind1, my_par1) in enumerate(vals1)
        for (ind2, my_par2) in enumerate(vals2)
            for e = 1:E
                pars = [my_par1, my_par2]
                Ysim = simulate_system(exp_data, pars, myM*K, dist_sens_inds, isws, Zm)
                # TODO: Since each realization of the baseline should be the same, since it's deterministic, it really should be enough to simulate only 1
                # Ybase = simulate_system(exp_data, pars, K, dist_sens_inds, isws, myZeros)
                for k = 1:K
                    Ym[:,k] = mean(Ysim[:,(k-1)*myM+1:k*myM], dims=2)
                    cost_vals[e][ind2,ind1] += mean((Y[N_trans+1:end, (e-1)*K+k].-Ym[N_trans+1:end,k]).^2)  # Need to have ind2 first, since then we can use surface(vals1,vals2,cost_vals[e])
                    # cost_base[e][ind] += mean((Y[N_trans+1:end, (e-1)*K+k].-Ybase[N_trans+1:end,k]).^2)
                end
                cost_vals[e][ind2,ind1] = cost_vals[e][ind2,ind1]/K
                
                # cost_base[e][ind] = cost_base[e][ind]/K
                # Ym = mean(simulate_system(exp_data, pars, myM, dist_sens_inds, isws, Zm), dims=2)

                # BONUS: Computing cost function gradient
                Ymsens, jacsYm = simulate_system_sens(exp_data, pars, 1, dist_sens_inds, isws)
                grad_vals[e][ind2,ind1] = first(get_cost_gradient(Y[N_trans+1:end, e], Ymsens[:,1:1], jacsYm[1:1], N_trans))
                @info "Completed computing cost for e = $e out of $E, ind =  $ind1, $ind2 out of $(length(vals1)), $(length(vals2))"
            end
            writedlm("data/experiments/tmp/backup_cost_vals_par$(ind1)_$(ind2).csv", [cost_vals[e][ind2,ind1] for e=1:E], ',')
        end
    end
    duration = now()-time_start

    return vals1, vals2, cost_vals, cost_base, duration, grad_vals
end

# Assumes scalar parameter
function find_delta_minima(E::Int, krange::Vector{Int}, dir::String)
    myN = 500
    base_mins = zeros(E, length(krange))
    prop_mins = zeros(E, length(krange))
    par_vals = readdlm("$(dir)/all_pars.csv", ',')
    for e = 1:E
        for (ik, k) = enumerate(krange)
            base_vals = readdlm("$(dir)/all_base_N$(k*myN)_e$(e).csv", ',')[:]
            prop_vals = readdlm("$(dir)/all_vals_N$(k*myN)_e$(e).csv", ',')[:]
            base_mins[e,ik] = par_vals[findall(base_vals.==minimum(base_vals))[1]]
            prop_mins[e,ik] = par_vals[findall(prop_vals.==minimum(prop_vals))[1]]
        end
    end
    return par_vals, base_mins, prop_mins
end

function visualize_SGD_search(par_vals::Vector{Float64}, cost_vals::Vector{Float64}, SGD_trace::Array{Vector{Float64},1}, opt_par::Float64; file_name = "SGD.gif")
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
function Roberts_gif_generator(par_vals::Vector{Float64}, cost_vals::Matrix{Float64}, SGD_traces::Array{Array{Vector{Float64},1},1}, opt_pars::Vector{Float64})
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

    for col = eachindex(θs[1,:])
        scatter!([col], [mean(θs[:,col])], markershape=:diamond)
    end

    hline!(p, [θ0], label = L"\theta_0", linestyle = :dot, linecolor = :gray)
    savefig("C:\\Programming\\dae-param-est\\src\\julia\\data\\results\\delta_L1_k100_100\\boxplot.png")
end

# -------------------------------- Ultimate super-clean debugging functions --------------------------------------

# Not including disturbance parameters (perhaps easy generalization, need to generate η instead of using η_true then)
function adjoint_dyn_debug(expid::String)
    # ------------------------------------------------------------------------
    # ------------------ Setup and disturbance generation --------------------
    # ------------------------------------------------------------------------
    exp_data, isws = get_experiment_data(expid)
    W_meta = exp_data.W_meta
    Y = exp_data.Y
    N = size(Y,1)÷y_len-1
    Nw = exp_data.Nw
    u = exp_data.u
    nx = W_meta.nx
    n_in = W_meta.n_in
    n_out = W_meta.n_out
    n_tot = nx*n_in
    η_true = W_meta.η


    @assert (num_dyn_pars == 1) "Only debugging with respect to 1 dynamical parameter is supported"

    Zm = [randn(Nw, n_tot) for m = 1:1]
    dmdl_true = discretize_ct_noise_model(get_ct_disturbance_model(η_true, nx, n_out), δ)
    # NOTE: OPTION 1: Use the rows below here for linear interpolation
    XWm = simulate_noise_process_mangled(dmdl_true, Zm)
    wmm(m::Int) = mk_noise_interp(dmdl_true.Cd, XWm, m, δ)
    # @warn "Not using any disturbance right now!!!"
    # wmm(m::Int) = t -> zeros(size(dmdl_true.Cd,1))
    # ---------------------------------------------------------------------------------------------------
    # ------------------ Forward solution of nominal system with forward sensitivity --------------------
    # ---------------------------------------------------------------------------------------------------
    
    # Simulates a second solution without sensitivity to use for numerical estimate of gradient
    my_δ = 0.001
    Ts_exact = 0.0001 # Reducing below 0.0001 didn't seem to provide any additional benefit. Just using Ts also seems to work fine actually
    # # Original
    # @time sol_for1 = solvew_sens(u, wmm(1), free_dyn_pars_true, N)
    # Replacing solvew_sens with parts to carefully compute time impact of every one of them
    @time my_mdl = model_sens_to_use(φ0, u, wmm(1), get_all_θs(free_dyn_pars_true))
    @time my_prob = problem(my_mdl, N, Ts)
    @time sol_for1 = solve(my_prob, saveat = 0:Ts:N*Ts, abstol = abstol, reltol = reltol, maxiters = maxiters)

    ts_exact = 0:Ts_exact:N*Ts+Ts_exact/2
    sol_for_exact = solve(my_prob, saveat = ts_exact, abstol = abstol, reltol = reltol, maxiters = maxiters)

    sol_for2 = solvew(u, wmm(1), free_dyn_pars_true.+my_δ, N)

    Y1, sens1 = h_comp(sol_for1, get_all_θs(free_dyn_pars_true))
    Y2 = h(sol_for2, get_all_θs(free_dyn_pars_true.+my_δ))
    Y1 = vcat(Y1...)
    Y2 = vcat(Y2...)
    sens1 = vcat(sens1...)

    cost1 = get_cost_value(Y[:,1], Y1[:,1:1])   # Reshaping vector Y1 to Matrix (I think they are vectors to begin with)
    cost2 = get_cost_value(Y[:,1], Y2[:,1:1])   # Reshaping vector Y2 to Matrix (I think they are vectors to begin with)

    for_est = first(get_cost_gradient(Y[:, 1], Y1[:,1:1], [sens1]))
    num_est = (cost2-cost1)/my_δ

    # ---------------------------------------------------------------------------------------------------
    # ------------------------------ Adjoint setup and gradient estimate --------------------------------
    # ---------------------------------------------------------------------------------------------------

    # !!!!!!!!!!!!!!!!!!!!!!!!!!!
    #  Unitfy notation for linear_interpolation and mvar_cubic???
    # They are defined in very different places, and use very different notation
    # Is there even a good point to using both? Maybe useful for pendulum...
    # !!!!!!!!!!!!!!!!!!!!!!!!!!!
    # Also figure out if you wanna keep Tsλ or just use Ts??? In get_estimates that is, here we already only use Ts

    # Extracts entire state trajecotry from first forward solution
    xvec1 = vcat(transpose(h_debug_with_sens(sol_for_exact, get_all_θs(free_dyn_pars_true)))...)   # transpose so that result is a matrix, with each row being one of the inner vectors (h_sens_deb returns a vector of vectors)

    # Creates functions for output, states, and their derivatives using interpolation.
    # y_func  = linear_interpolation_multivar(Y[:,1], Ts, y_len)
    y_func = extrapolate(scale(interpolate(Y[:,1], BSpline(interp_type)), 0.0:Ts:N*Ts), Line())
    dy_est  = (Y[y_len+1:end,1]-Y[1:end-y_len,1])/Ts
    # dy_func = linear_interpolation_multivar(dy_est, Ts, y_len)
    dy_func = extrapolate(scale(interpolate(dy_est, BSpline(interp_type)), 0.0:Ts:(N-1)*Ts), Line())
    # x_func  = get_mvar_cubic(ts_exact, xvec1)
    x_func = extrapolate(scale(interpolate(xvec1, (BSpline(interp_type), NoInterp())), ts_exact, 1:size(xvec1,2)), Line())
    # der_est  = get_der_est(ts_exact, x_func)
    der_est  = get_der_est2(ts_exact, x_func, size(xvec1, 2))
    dx_func = extrapolate(scale(interpolate(der_est, (BSpline(interp_type), NoInterp())), ts_exact[1:end-1], 1:size(der_est,2)), Line())
    # dx_func = get_mvar_cubic(ts_exact[1:end-1], der_est)

    # Computing xp0, initial conditions of derivative of x wrt to p
    mdl = model_to_use(φ0, u, wmm(1), get_all_θs(free_dyn_pars_true))
    mdl_sens = model_sens_to_use(φ0, u, wmm(1), get_all_θs(free_dyn_pars_true))
    n_mdl = length(mdl.x0)
    xp0 = reshape(f_sens_deb(mdl_sens.x0, get_all_θs(free_dyn_pars_true)), num_dyn_vars_adj, length(f_sens_deb(mdl_sens.x0, get_all_θs(free_dyn_pars_true)))÷num_dyn_vars_adj)
    (nx,np) = size(xp0)

    mdl_adj, get_Gp, debugs = model_adj_to_use(u, wmm(1), get_all_θs(free_dyn_pars_true), N*Ts, x_func, x_func, y_func, dy_func, xp0, dx_func, dx_func)
    adj_prob = problem_reverse(mdl_adj, N, Ts) # Adjoint problem must be solved backwards, problem_reverse ensures it is
    # NOTE: The solution is oriented backwards in time, i.e. first element
    # is t=T and last is t=0
    adj_sol = solve(adj_prob, saveat = 0:Ts:N*Ts, abstol = abstol, reltol = reltol, maxiters = maxiters)
    adj_sol_exact = solve(adj_prob, saveat = ts_exact, abstol = abstol, reltol = reltol, maxiters = maxiters)
    Gp = first(get_Gp(adj_sol))
    println("Num: $num_est, for: $for_est, adj: $Gp")

    # # --------- Obtaining lambda functions  - VER 1 ---------
    λs = [adj_sol_exact.u[end-ind+1][1:nx] for ind=eachindex(adj_sol_exact.u)]
    λ_func = linear_interpolation_multivar(vcat(λs...), Ts_exact, length(λs[1]))
    # --------- More accurate lambda functions - VER 2 - ONLY FOR PENDULUM ---------
    # if model_id == PENDULUM
    #     λ_func, _ = solve_accurate_adjoint(N, Ts, x_func, dx_func, x_func, y_func, 0)
    # end

    # --------------------------------------------------------------------------------------------------------------
    # ------------------ BONUS: COMPARING ADJOINT SOLUTION TRAJECTORY TO FORWARD SENSITIVITY -----------------------
    # --------------------------------------------------------------------------------------------------------------

    (get_Gp_debug, get_term_debug) = debugs # Only get_term_debug seems used
    _, integral, term = get_Gp_debug(adj_sol)

    # Extracts beta and term trajectories from adjoint solution
    _, sens_deb = h_sens_deb(sol_for1, get_all_θs(free_dyn_pars_true))
    xps = hcat(sens_deb...)
    term_vec = get_term_debug(adj_sol, xps, 0:Ts:(N*Ts))    # ONLY USED AT VERY END. BUT VERY NICE TO HAVE :D
    beta = [adj_sol.u[end-ind+1][nx+1] for ind=1:length(adj_sol.u)]     # ALSO VERY NICE TO HAVE FOR DEBUGGING AND COMPARING TO INTEGRAL COST

    # if model_id == PENDULUM
    #     function get_pend_betas(λs_func, x, dx_func, xps, N, Ts)
    #         times = 0.0:Ts:N*Ts

    #         mint(z,p,t) = [-λs_func(t)[3]*dx_func(t)[4] - λs_func(t)[4]*(dx_func(t)[5]+g)]
    #         Lint(z,p,t) = [2*λs_func(t)[2]*L]
    #         kint(z,p,t) = [-λs_func(t)[3]*abs(x(t)[4])*x(t)[4] - λs_func(t)[4]*abs(x(t)[5])*x(t)[5]]
    #         mprob = ODEProblem(mint, [0.0], (0.0, N*Ts), [])
    #         Lprob = ODEProblem(Lint, [0.0], (0.0, N*Ts), [])
    #         kprob = ODEProblem(kint, [0.0], (0.0, N*Ts), [])
    #         msol = DifferentialEquations.solve(mprob, Tsit5(), reltol=reltol, abstol=abstol, saveat=times)
    #         Lsol = DifferentialEquations.solve(Lprob, Tsit5(), reltol=reltol, abstol=abstol, saveat=times)
    #         ksol = DifferentialEquations.solve(kprob, Tsit5(), reltol=reltol, abstol=abstol, saveat=times)
    #         mvec = [msol.u[end][1]-msol.u[i][1] for i=eachindex(msol)]
    #         Lvec = [Lsol.u[end][1]-Lsol.u[i][1] for i=eachindex(Lsol)]
    #         kvec = [ksol.u[end][1]-ksol.u[i][1] for i=eachindex(ksol)]

    #         # ---------------- Getting term ----------------
    #         Fdx = t -> vcat([1   0   0          0   0   2x(t)[1]    0
    #                         0   1   0          0   0   2x(t)[2]    0
    #                         0   0   -x(t)[1]   m   0   0           0
    #                         0   0   -x(t)[2]   0   m   0           0], zeros(3,7))

    #         term = zeros(length(times))
    #         for ind=eachindex(adj_sol.u)
    #             term[ind] = ((λs_func(times[ind])')*Fdx(times[ind]))*xps[:,ind]
    #         end

    #         return mvec, Lvec, kvec, term
    #     end

    #     # NOTE: term only valid for parameter corresponding to xps
    #     βm, βL, βk, term_acc = get_pend_betas(λ_func, x_func, dx_func, xps, N, Ts)
    # end
    
    cost_grad_plot = [first(get_cost_gradient(Y[1:y_len*ind,1], Y1[1:y_len*ind,1:1], [sens1[1:y_len*ind,1:1]])) for ind=1:size(Y,1)÷y_len]  
    plot(beta[1].-beta .- term_vec.+term_vec[1], label="adj_est")
    # if model_id == PENDULUM
    #     # Interestingly enough this plot always seems to be the most off, less so with accurate version, but still
    #     plot!(βk[1].-βk .- term_acc.+term_acc[1], label="other_est(acc/other)")
    # end
    plot!(cost_grad_plot, label="for_est")

    # -----------------------------------------------------------------------------------------------------------------------
    # ------------------ BONUS: STEP-BY-STEP ANALYSIS OF ADJOINT SOLUTION, TO SEE WHERE ERRORS APPEAR -----------------------
    # -----------------------------------------------------------------------------------------------------------------------

    # mydelta = 0.01
    # dλ_func(t) = (λ_func(t+mydelta)-λ_func(t))/mydelta
    # stepbystep_mdl = model_stepbystep(φ0, u, wmm(1), get_all_θs(free_dyn_pars_true), y_func, x_func, dx_func, λ_func, dλ_func, deb_Fp, N*Ts)

    # stepstep_prob = problem(stepbystep_mdl, N, Ts)
    # stepstep_sol = solve(stepstep_prob, saveat = 0:Ts:N*Ts, abstol = abstol, reltol = reltol, maxiters = maxiters)
    # integral_sens_1 = [stepstep_sol.u[ind][1] for ind=eachindex(stepstep_sol.u)]
    # int_with_lambda_2 = [stepstep_sol.u[ind][2] for ind=eachindex(stepstep_sol.u)]
    # post_partial_3 = [stepstep_sol.u[ind][3] for ind=eachindex(stepstep_sol.u)]
    # final_expression_4 = [stepstep_sol.u[ind][4] for ind=eachindex(stepstep_sol.u)]

    # function Fdx(dx,x)
    #     if model_id == PENDULUM
    #         return vcat([1   0   0    0   0   2x[1]    0
    #             0   1   0    0   0   2x[2]    0
    #             0   0   -x[1]   m   0   0     0
    #             0   0   -x[2]   0   m   0     0], zeros(3,7))
    #     elseif model_id == DELTA
    #         return [   # x and dx have to be adjoint versions of states, i.e. include model output as a state
    #             1.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   -L1*sin(x[1])   L1*cos(x[1])   0.0   -L1*sin(x[1])   L1*cos(x[1])   0.0   0.0   0.0
    #             0.0   1.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   L2*cos(x[2])*sin(x[3])   -L2*sin(x[2])   L2*cos(x[2])*cos(x[3])   L2*cos(x[2])*sin(x[3])   -L2*sin(x[2])   L2*cos(x[2])*cos(x[3])   0.0   0.0   0.0
    #             0.0   0.0   1.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   L2*cos(x[3])*sin(x[2])   0.0   -L2*sin(x[2])*sin(x[3])   L2*cos(x[3])*sin(x[2])   0.0   -L2*sin(x[2])*sin(x[3])   0.0   0.0   0.0
    #             0.0   0.0   0.0   1.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   -(sqrt(3)*L1*sin(x[4]))*0.5   -(L1*sin(x[4]))*0.5   -L1*cos(x[4])   0.0   0.0   0.0   0.0   0.0   0.0
    #             0.0   0.0   0.0   0.0   1.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   (L2*cos(x[5])*sin(x[6]))*0.5-(sqrt(3)*L2*sin(x[5]))*0.5   -(L2*sin(x[5]))*0.5-(sqrt(3)*L2*cos(x[5])*sin(x[6]))*0.5   -L2*cos(x[5])*cos(x[6])   0.0   0.0   0.0   0.0   0.0   0.0
    #             0.0   0.0   0.0   0.0   0.0   1.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   (L2*cos(x[6])*sin(x[5]))*0.5   -(sqrt(3)*L2*cos(x[6])*sin(x[5]))*0.5   L2*sin(x[5])*sin(x[6])   0.0   0.0   0.0   0.0   0.0   0.0
    #             0.0   0.0   0.0   0.0   0.0   0.0   1.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   (sqrt(3)*L1*sin(x[7]))*0.5   -(L1*sin(x[7]))*0.5   -L1*cos(x[7])   0.0   0.0   0.0
    #             0.0   0.0   0.0   0.0   0.0   0.0   0.0   1.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   (L2*cos(x[8])*sin(x[9]))*0.5+(sqrt(3)*L2*sin(x[8]))*0.5   (sqrt(3)*L2*cos(x[8])*sin(x[9]))*0.5-(L2*sin(x[8]))*0.5   -L2*cos(x[8])*cos(x[9])   0.0   0.0   0.0
    #             0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   1.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   (L2*cos(x[9])*sin(x[8]))*0.5   (sqrt(3)*L2*cos(x[9])*sin(x[8]))*0.5   L2*sin(x[8])*sin(x[9])   0.0   0.0   0.0
    #             0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   J1+L1^2*(M2+M3)+LC1^2*M1   L1*(L2*M3+LC2*M2)*(sin(x[1])*sin(x[2])+cos(x[1])*cos(x[2])*cos(x[3]))   -L1*cos(x[1])*sin(x[2])*sin(x[3])*(L2*M3+LC2*M2)   0.0   0.0   0.0   0.0   0.0   0.0   0.0   L1*sin(x[1])   -L1*cos(x[1])   0.0   L1*sin(x[1])   -L1*cos(x[1])   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0
    #             0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   L1*(L2*M3+LC2*M2)*(sin(x[1])*sin(x[2])+cos(x[1])*cos(x[2])*cos(x[3]))   J2+L2^2*M3+LC2^2*M2   0.0   0.0   0.0   0.0   0.0   0.0   0.0   -L2*cos(x[2])*sin(x[3])   L2*sin(x[2])   -L2*cos(x[2])*cos(x[3])   -L2*cos(x[2])*sin(x[3])   L2*sin(x[2])   -L2*cos(x[2])*cos(x[3])   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0
    #             0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   -L1*cos(x[1])*sin(x[2])*sin(x[3])*(L2*M3+LC2*M2)   0.0   sin(x[2])^2*(J2+L2^2*M3+LC2^2*M2)   0.0   0.0   0.0   0.0   0.0   0.0   -L2*cos(x[3])*sin(x[2])   0.0   L2*sin(x[2])*sin(x[3])   -L2*cos(x[3])*sin(x[2])   0.0   L2*sin(x[2])*sin(x[3])   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0
    #             0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   J1+L1^2*(M2+M3)+LC1^2*M1   L1*(L2*M3+LC2*M2)*(sin(x[4])*sin(x[5])+cos(x[4])*cos(x[5])*cos(x[6]))   -L1*cos(x[4])*sin(x[5])*sin(x[6])*(L2*M3+LC2*M2)   0.0   0.0   0.0   (sqrt(3)*L1*sin(x[4]))*0.5   (L1*sin(x[4]))*0.5   L1*cos(x[4])   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0
    #             0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   L1*(L2*M3+LC2*M2)*(sin(x[4])*sin(x[5])+cos(x[4])*cos(x[5])*cos(x[6]))   J2+L2^2*M3+LC2^2*M2   0.0   0.0   0.0   0.0   (sqrt(3)*L2*sin(x[5]))*0.5-(L2*cos(x[5])*sin(x[6]))*0.5   (L2*sin(x[5]))*0.5+(sqrt(3)*L2*cos(x[5])*sin(x[6]))*0.5   L2*cos(x[5])*cos(x[6])   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0
    #             0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   -L1*cos(x[4])*sin(x[5])*sin(x[6])*(L2*M3+LC2*M2)   0.0   sin(x[5])^2*(J2+L2^2*M3+LC2^2*M2)   0.0   0.0   0.0   -(L2*cos(x[6])*sin(x[5]))*0.5   (sqrt(3)*L2*cos(x[6])*sin(x[5]))*0.5   -L2*sin(x[5])*sin(x[6])   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0
    #             0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   J1+L1^2*(M2+M3)+LC1^2*M1   L1*(L2*M3+LC2*M2)*(sin(x[7])*sin(x[8])+cos(x[7])*cos(x[8])*cos(x[9]))   -L1*cos(x[7])*sin(x[8])*sin(x[9])*(L2*M3+LC2*M2)   0.0   0.0   0.0   -(sqrt(3)*L1*sin(x[7]))*0.5   (L1*sin(x[7]))*0.5   L1*cos(x[7])   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0
    #             0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   L1*(L2*M3+LC2*M2)*(sin(x[7])*sin(x[8])+cos(x[7])*cos(x[8])*cos(x[9]))   J2+L2^2*M3+LC2^2*M2   0.0   0.0   0.0   0.0   -(L2*cos(x[8])*sin(x[9]))*0.5-(sqrt(3)*L2*sin(x[8]))*0.5   (L2*sin(x[8]))*0.5-(sqrt(3)*L2*cos(x[8])*sin(x[9]))*0.5   L2*cos(x[8])*cos(x[9])   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0
    #             0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   -L1*cos(x[7])*sin(x[8])*sin(x[9])*(L2*M3+LC2*M2)   0.0   sin(x[8])^2*(J2+L2^2*M3+LC2^2*M2)   0.0   0.0   0.0   -(L2*cos(x[9])*sin(x[8]))*0.5   -(sqrt(3)*L2*cos(x[9])*sin(x[8]))*0.5   -L2*sin(x[8])*sin(x[9])   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0
    #             0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0
    #             0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0
    #             0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0
    #             0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0
    #             0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0
    #             0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0
    #             0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0
    #             0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0
    #             0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0
    #             0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0
    #             0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0
    #             0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0
    #             0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0
    #             0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0
    #             0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0
    #         ]
    #     end
    # end
    # term_func(t) = (λ_func(t)')*Fdx(dx_func(t),x_func(t))*x_func(t)[num_dyn_vars_adj+1:end]

    # println("End values of trajectories: ", integral_sens_1[end], "; ", int_with_lambda_2[end], "; ", post_partial_3[end], "+", -term_func.(N*Ts)+term_func(0.0), "=", post_partial_3[end]-term_func.(N*Ts)+term_func(0.0), "; ", final_expression_4[end], "+", -term_func.(N*Ts)+term_func(0.0), "=", final_expression_4[end]-term_func.(N*Ts)+term_func(0.0))

    # deb_rem_6 = [stepstep_sol.u[ind][6] for ind=eachindex(stepstep_sol.u)]

    # # PREMISES: integral_sens_1 should match gradient on original sum-cost well. 
    # # int_with_lambda_2 should be similar, with only a little bit of drift, otherwise something is wrong with our forward solution
    # # DEBUGGING: post_partial_3 can look a little different, since partial integration seems to have an unintuittive effect on numerical errors.
    # # However, the final value should match integral_sens_1 as well as possible, otherwise we will have issues.
    # # If adjoint solution is correct, post_partial_3 and final_expression_4 should match well, since that's where we replace lambda-expressions.
    # # NOTE: If you seem to get large numerical mismatch, try to disable disturbances and use a smoother input signal, e.g. a sinusoid. If this reduces the numerical
    # # mismatch, then perhaps the earlier mismatch is just inherent to the very quickly changing input signals
    # # I think it also should be expected that sum-cost and integral-cost will differ somewhat when inputs are so irregular

    # plot(integral_sens_1, label="int_cost_sens")
    # plot!(int_with_lambda_2, label="int_sens_with_lam")
    # plot!(post_partial_3-term_func.(0.0:Ts:N*Ts).+term_func(0.0), label="post_partial")
    # plot!(final_expression_4-term_func.(0.0:Ts:N*Ts).+term_func(0.0), label="adj_final")

    # # ------------------ EXTRA: COMPARING SUM AND INTEGRAL COST -----------------------

    # # Comparing sum and integral costs:
    # intcosttraj = [stepstep_sol.u[ind][5] for ind=eachindex(stepstep_sol.u)]
    # sumcosttraj = [(1/(N+1))*sum( ( Y[1:y_len*ind] - Y1[1:y_len*ind,1] ).^2 ) for ind=1:N+1]
    # plot(intcosttraj)
    # plot!(sumcosttraj)


    # # Comparing numerical cost gradient with integral cost gradient
    # sumcosttraj1 = [(1/N)*sum( ( Y[1:y_len*ind] - Y1[1:y_len*ind,1] ).^2 ) for ind=1:N+1]
    # sumcosttraj2 = [(1/N)*sum( ( Y[1:y_len*ind] - Y2[1:y_len*ind,1] ).^2 ) for ind=1:N+1]
    # sumcostgrad = (sumcosttraj2-sumcosttraj1)/my_δ
    # plot(integral_sens_1, label="int_cost_sens")
    # plot!(sumcostgrad, label="sum_cost_numsens")
end

function adjoint_dist_debug(expid::String)
    # ------------------------------------------------------------------------
    # ------------------------------ Setup -----------------------------------
    # ------------------------------------------------------------------------
    exp_data, isws = get_experiment_data(expid)
    W_meta = exp_data.W_meta
    Y = exp_data.Y
    N = size(Y,1)÷y_len-1
    Nw = exp_data.Nw
    u = exp_data.u
    nxw = W_meta.nx
    n_in = W_meta.n_in
    n_out = W_meta.n_out
    n_tot = nxw*n_in
    η_true = W_meta.η
    dist_par_inds = W_meta.free_par_inds
    free_pars_true = vcat(free_dyn_pars_true, W_meta.η[dist_par_inds])

    @assert (num_dyn_pars == 0 && length(dist_par_inds) == 1) "Only debugging with respect to zero dynamical parameters and 1 disturbance parameter is supported"

    Zm = [randn(Nw, n_tot) for m = 1:1]
    # ---------------------------------------------------------------------------------------------------
    # ------------------ Forward solution of nominal system with forward sensitivity --------------------
    # ---------------------------------------------------------------------------------------------------
    
    η_true = exp_data.get_all_ηs(free_pars_true)
    dmdl_sens = discretize_ct_noise_model_with_sensitivities_alt(get_ct_disturbance_model(η_true, nxw, n_out), δ, dist_par_inds)
    # NOTE: OPTION 1: Use the rows below here for linear interpolation
    @time XWm_sens = simulate_noise_process_mangled(dmdl_sens, Zm)
    wmm_sens(m::Int) = mk_noise_interp(dmdl_sens.Cd, XWm_sens, m, δ)
    xwm_sens(m::Int) = mk_xw_interp(dmdl_sens.Cd, XWm_sens, m, δ)

    # Simulates a second solution without sensitivity to use for numerical estimate of gradient
    my_δ = 0.01
    Ts_exact = 0.0001 # Reducing below 0.0001 didn't seem to provide any additional benefit. Just using Ts also seems to work fine actually
    # # Original
    # @time sol_for1 = solvew_sens(u, wmm(1), free_dyn_pars_true, N)
    # Replacing solvew_sens with parts to carefully compute time impact of every one of them
    @time my_mdl = model_sens_to_use(φ0, u, wmm_sens(1), get_all_θs(free_dyn_pars_true))
    @time my_prob = problem(my_mdl, N, Ts)
    @time sol_for1 = solve(my_prob, saveat = 0:Ts:N*Ts, abstol = abstol, reltol = reltol, maxiters = maxiters)

    # TODO Solving once and then downsampling would be smarter rather than solving twice, but oh well
    ts_exact = 0:Ts_exact:N*Ts+Ts_exact/2
    sol_for_exact = solve(my_prob, saveat = ts_exact, abstol = abstol, reltol = reltol, maxiters = maxiters)

    η_2 = exp_data.get_all_ηs(free_pars_true.+my_δ)
    dmdl_2 = discretize_ct_noise_model(get_ct_disturbance_model(η_2, nxw, n_out), δ)
    # NOTE: OPTION 1: Use the rows below here for linear interpolation
    @time XWm_2 = simulate_noise_process_mangled(dmdl_2, Zm)
    wmm_2(m::Int) = mk_noise_interp(dmdl_2.Cd, XWm_2, m, δ)
    sol_for2 = solvew(u, wmm_2(1), free_dyn_pars_true, N)

    Y1, sens1 = h_comp(sol_for1, get_all_θs(free_dyn_pars_true))
    Y2 = h(sol_for2, get_all_θs(free_dyn_pars_true))
    Y1 = vcat(Y1...)
    Y2 = vcat(Y2...)
    sens1 = vcat(sens1...)

    cost1 = get_cost_value(Y[:,1], Y1[:,1:1])   # Reshaping vector Y1 to Matrix (I think they are vectors to begin with)
    cost2 = get_cost_value(Y[:,1], Y2[:,1:1])   # Reshaping vector Y2 to Matrix (I think they are vectors to begin with)

    for_est = first(get_cost_gradient(Y[:, 1], Y1[:,1:1], [sens1]))
    num_est = (cost2-cost1)/my_δ


    # ---------------------------------------------------------------------------------------------------
    # ------------------------------ Adjoint setup and gradient estimate --------------------------------
    # ---------------------------------------------------------------------------------------------------

    # Extracts entire state trajecotry from first forward solution. Adjoint version of state
    xvec1 = vcat(transpose(h_debug_with_sens(sol_for_exact, get_all_θs(free_dyn_pars_true)))...)   # transpose so that result is a matrix, with each row being one of the inner vectors (h_sens_deb returns a vector of vectors)

    # Creates functions for output, states, and their derivatives using interpolation.
    # y_func  = linear_interpolation_multivar(Y[:,1], Ts, y_len)
    y_func = extrapolate(scale(interpolate(Y[:,1], BSpline(interp_type)), 0.0:Ts:N*Ts), Line())
    dy_est  = (Y[y_len+1:end,1]-Y[1:end-y_len,1])/Ts
    # dy_func = linear_interpolation_multivar(dy_est, Ts, y_len)
    dy_func = extrapolate(scale(interpolate(dy_est, BSpline(interp_type)), 0.0:Ts:(N-1)*Ts), Line())
    # x_func  = get_mvar_cubic(ts_exact, xvec1)
    x_func = extrapolate(scale(interpolate(xvec1, (BSpline(interp_type), NoInterp())), ts_exact, 1:size(xvec1,2)), Line())
    # der_est  = get_der_est(ts_exact, x_func)
    der_est  = get_der_est2(ts_exact, x_func, size(xvec1, 2))
    dx_func = extrapolate(scale(interpolate(der_est, (BSpline(interp_type), NoInterp())), ts_exact[1:end-1], 1:size(der_est,2)), Line())
    # dx_func = get_mvar_cubic(ts_exact[1:end-1], der_est)

    # Computing xp0, initial conditions of derivative of x wrt to p
    mdl = model_to_use(φ0, u, wmm_sens(1), get_all_θs(free_dyn_pars_true))
    mdl_sens = model_sens_to_use(φ0, u, wmm_sens(1), get_all_θs(free_dyn_pars_true))
    n_mdl = length(mdl.x0)
    xp0 = reshape(f_sens_deb(mdl_sens.x0,get_all_θs(free_pars_true)), num_dyn_vars_adj, length(f_sens_deb(mdl_sens.x0,get_all_θs(free_pars_true)))÷num_dyn_vars_adj)
    (nx,np) = size(xp0)

    dmdl, B̃, B̃ηa = discretize_ct_noise_model_with_sensitivities_for_adj(get_ct_disturbance_model(η_true, nxw, n_out), δ, dist_par_inds)
    vm(m::Int) = mk_v_ZOH(Zm[m], δ)
    XWm_adj = simulate_noise_process_mangled(dmdl, Zm)
    xwm_adj(m::Int) = mk_xw_interp(dmdl.Cd, XWm_adj, m, δ)
    nw = size(dmdl.Cd, 1)
    function z_func(t::Float64)
        xwt = xwm_sens(1)(t)
        wt =  wmm_sens(1)(t)
        return vcat(x_func(t,1:num_dyn_vars_adj), xwt[1:nxw], wt[1:nw], x_func(t,num_dyn_vars_adj+1:2num_dyn_vars_adj), xwt[nxw+1:end], wt[nw+1:end])
    end
    
    # wmm_sens and xwm_adj actually come from the same white noise realization Zm. So they are not independent, just to emphasize that.
    mdl_adj, get_Gp, debugs = model_adj_to_use_dist_sens(u, wmm_sens(1), xwm_adj(1), vm(1), get_all_θs(free_pars_true), N*Ts, x_func, x_func, y_func, dy_func, xp0, dx_func, dx_func, B̃, B̃ηa, η_true, 1)
    adj_prob = problem_reverse(mdl_adj, N, Ts)
    adj_sol = solve(adj_prob, saveat = 0:Ts:N*Ts, abstol =  abstol, reltol = reltol,
        maxiters = maxiters)
    adj_sol_exact = solve(adj_prob, saveat = ts_exact, abstol =  abstol, reltol = reltol,
        maxiters = maxiters)

    Gp = first(get_Gp(adj_sol))

    println("Num: $num_est, for: $for_est, adj: $Gp")

    # # --------- Obtaining lambda functions  - VER 1 ---------
    λs = [adj_sol_exact.u[end-ind+1][1:nx+nxw+nw] for ind=eachindex(adj_sol_exact.u)]
    λ_func = linear_interpolation_multivar(vcat(λs...), Ts_exact, length(λs[1]))
    # --------- More accurate lambda functions - VER 2 - ONLY FOR PENDULUM ---------
    # if model_id == PENDULUM
    #     λ_func, _ = solve_accurate_adjoint(N, Ts, x_func, dx_func, x_func, y_func, 0)
    # end

    # --------------------------------------------------------------------------------------------------------------
    # ------------------ BONUS: COMPARING ADJOINT SOLUTION TRAJECTORY TO FORWARD SENSITIVITY -----------------------
    # --------------------------------------------------------------------------------------------------------------

    (get_Gp_debug, get_term_debug) = debugs # Only get_term_debug seems used
    _, integral, term = get_Gp_debug(adj_sol)


    # Extracts beta and term trajectories from adjoint solution
    _, sens_deb = h_sens_deb(sol_for1, get_all_θs(free_dyn_pars_true))
    xps = hcat(sens_deb...)
    term_vec = get_term_debug(adj_sol, xps, 0:Ts:(N*Ts))    # ONLY USED AT VERY END. BUT VERY NICE TO HAVE :D
    beta = [adj_sol.u[end-ind+1][nx+nxw+nw+1] for ind=1:length(adj_sol.u)]     # ALSO VERY NICE TO HAVE FOR DEBUGGING AND COMPARING TO INTEGRAL COST

    # TODO: Consider generalizing the exact pendulum-part for disturbance parameters
    # it currently only works for m, L, and k!!!!!!! <-------------------------------
    ##################################################################################
    # if model_id == PENDULUM
    #     function get_pend_betas(λs_func, x, dx_func, xps, N, Ts)
    #         times = 0.0:Ts:N*Ts

    #         mint(z,p,t) = [-λs_func(t)[3]*dx_func(t)[4] - λs_func(t)[4]*(dx_func(t)[5]+g)]
    #         Lint(z,p,t) = [2*λs_func(t)[2]*L]
    #         kint(z,p,t) = [-λs_func(t)[3]*abs(x(t)[4])*x(t)[4] - λs_func(t)[4]*abs(x(t)[5])*x(t)[5]]
    #         mprob = ODEProblem(mint, [0.0], (0.0, N*Ts), [])
    #         Lprob = ODEProblem(Lint, [0.0], (0.0, N*Ts), [])
    #         kprob = ODEProblem(kint, [0.0], (0.0, N*Ts), [])
    #         msol = DifferentialEquations.solve(mprob, Tsit5(), reltol=reltol, abstol=abstol, saveat=times)
    #         Lsol = DifferentialEquations.solve(Lprob, Tsit5(), reltol=reltol, abstol=abstol, saveat=times)
    #         ksol = DifferentialEquations.solve(kprob, Tsit5(), reltol=reltol, abstol=abstol, saveat=times)
    #         mvec = [msol.u[end][1]-msol.u[i][1] for i=eachindex(msol)]
    #         Lvec = [Lsol.u[end][1]-Lsol.u[i][1] for i=eachindex(Lsol)]
    #         kvec = [ksol.u[end][1]-ksol.u[i][1] for i=eachindex(ksol)]

    #         # ---------------- Getting term ----------------
    #         Fdx = t -> vcat([1   0   0          0   0   2x(t)[1]    0
    #                         0   1   0          0   0   2x(t)[2]    0
    #                         0   0   -x(t)[1]   m   0   0           0
    #                         0   0   -x(t)[2]   0   m   0           0], zeros(3,7))

    #         term = zeros(length(times))
    #         for ind=eachindex(adj_sol.u)
    #             term[ind] = ((λs_func(times[ind])')*Fdx(times[ind]))*xps[:,ind]
    #         end

    #         return mvec, Lvec, kvec, term
    #     end

    #     # NOTE: term only valid for parameter corresponding to xps
    #     βm, βL, βk, term_acc = get_pend_betas(λ_func, x_func, dx_func, xps, N, Ts)
    # end
    
    cost_grad_plot = [first(get_cost_gradient(Y[1:y_len*ind,1], Y1[1:y_len*ind,1:1], [sens1[1:y_len*ind,1:1]])) for ind=1:size(Y,1)÷y_len]  
    plot(beta[1].-beta .- term_vec.+term_vec[1], label="adj_est")
    # if model_id == PENDULUM
    #     # Interestingly enough this plot always seems to be the most off, less so with accurate version, but still
    #     plot!(βk[1].-βk .- term_acc.+term_acc[1], label="other_est(acc/other)")
    # end
    plot!(cost_grad_plot, label="for_est")

    # -----------------------------------------------------------------------------------------------------------------------
    # ------------------ BONUS: STEP-BY-STEP ANALYSIS OF ADJOINT SOLUTION, TO SEE WHERE ERRORS APPEAR -----------------------
    # -----------------------------------------------------------------------------------------------------------------------

    mydelta = 0.01
    dλ_func(t) = (λ_func(t+mydelta)-λ_func(t))/mydelta
    stepbystep_mdl = model_stepbystep_dist(u, wmm_sens(1), xwm_sens(1), vm(1), get_all_θs(free_dyn_pars_true), y_func, dy_func, x_func, dx_func, λ_func, dλ_func, deb_Fp, B̃, B̃ηa, η_true, N*Ts)

    stepstep_prob = problem(stepbystep_mdl, N, Ts)
    stepstep_sol = solve(stepstep_prob, saveat = 0:Ts:N*Ts, abstol = abstol, reltol = reltol, maxiters = maxiters)
    integral_sens_1 = [stepstep_sol.u[ind][1] for ind=eachindex(stepstep_sol.u)]
    int_with_lambda_2 = [stepstep_sol.u[ind][2] for ind=eachindex(stepstep_sol.u)]
    post_partial_3 = [stepstep_sol.u[ind][3] for ind=eachindex(stepstep_sol.u)]
    final_expression_4 = [stepstep_sol.u[ind][4] for ind=eachindex(stepstep_sol.u)]

    function Fdx(dx,x)
        if model_id == PENDULUM
            Fdx_tmp = vcat([1   0   0    0   0   2x[1]    0
                            0   1   0    0   0   2x[2]    0
                            0   0   -x[1]   m   0   0     0
                            0   0   -x[2]   0   m   0     0], zeros(3,7))
            return [Fdx_tmp         zeros(7,2)   zeros(7,1)
                    zeros(2,7)  [1.0 0.0; 0.0 1.0] zeros(2,1)
                    zeros(1,7)       zeros(1,2)    zeros(1,1)]
        elseif model_id == DELTA
            Fdx_tmp =  [   # x and dx have to be adjoint versions of states, i.e. include model output as a state
                1.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   -L1*sin(x[1])   L1*cos(x[1])   0.0   -L1*sin(x[1])   L1*cos(x[1])   0.0   0.0   0.0
                0.0   1.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   L2*cos(x[2])*sin(x[3])   -L2*sin(x[2])   L2*cos(x[2])*cos(x[3])   L2*cos(x[2])*sin(x[3])   -L2*sin(x[2])   L2*cos(x[2])*cos(x[3])   0.0   0.0   0.0
                0.0   0.0   1.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   L2*cos(x[3])*sin(x[2])   0.0   -L2*sin(x[2])*sin(x[3])   L2*cos(x[3])*sin(x[2])   0.0   -L2*sin(x[2])*sin(x[3])   0.0   0.0   0.0
                0.0   0.0   0.0   1.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   -(sqrt(3)*L1*sin(x[4]))*0.5   -(L1*sin(x[4]))*0.5   -L1*cos(x[4])   0.0   0.0   0.0   0.0   0.0   0.0
                0.0   0.0   0.0   0.0   1.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   (L2*cos(x[5])*sin(x[6]))*0.5-(sqrt(3)*L2*sin(x[5]))*0.5   -(L2*sin(x[5]))*0.5-(sqrt(3)*L2*cos(x[5])*sin(x[6]))*0.5   -L2*cos(x[5])*cos(x[6])   0.0   0.0   0.0   0.0   0.0   0.0
                0.0   0.0   0.0   0.0   0.0   1.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   (L2*cos(x[6])*sin(x[5]))*0.5   -(sqrt(3)*L2*cos(x[6])*sin(x[5]))*0.5   L2*sin(x[5])*sin(x[6])   0.0   0.0   0.0   0.0   0.0   0.0
                0.0   0.0   0.0   0.0   0.0   0.0   1.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   (sqrt(3)*L1*sin(x[7]))*0.5   -(L1*sin(x[7]))*0.5   -L1*cos(x[7])   0.0   0.0   0.0
                0.0   0.0   0.0   0.0   0.0   0.0   0.0   1.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   (L2*cos(x[8])*sin(x[9]))*0.5+(sqrt(3)*L2*sin(x[8]))*0.5   (sqrt(3)*L2*cos(x[8])*sin(x[9]))*0.5-(L2*sin(x[8]))*0.5   -L2*cos(x[8])*cos(x[9])   0.0   0.0   0.0
                0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   1.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   (L2*cos(x[9])*sin(x[8]))*0.5   (sqrt(3)*L2*cos(x[9])*sin(x[8]))*0.5   L2*sin(x[8])*sin(x[9])   0.0   0.0   0.0
                0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   J1+L1^2*(M2+M3)+LC1^2*M1   L1*(L2*M3+LC2*M2)*(sin(x[1])*sin(x[2])+cos(x[1])*cos(x[2])*cos(x[3]))   -L1*cos(x[1])*sin(x[2])*sin(x[3])*(L2*M3+LC2*M2)   0.0   0.0   0.0   0.0   0.0   0.0   0.0   L1*sin(x[1])   -L1*cos(x[1])   0.0   L1*sin(x[1])   -L1*cos(x[1])   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0
                0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   L1*(L2*M3+LC2*M2)*(sin(x[1])*sin(x[2])+cos(x[1])*cos(x[2])*cos(x[3]))   J2+L2^2*M3+LC2^2*M2   0.0   0.0   0.0   0.0   0.0   0.0   0.0   -L2*cos(x[2])*sin(x[3])   L2*sin(x[2])   -L2*cos(x[2])*cos(x[3])   -L2*cos(x[2])*sin(x[3])   L2*sin(x[2])   -L2*cos(x[2])*cos(x[3])   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0
                0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   -L1*cos(x[1])*sin(x[2])*sin(x[3])*(L2*M3+LC2*M2)   0.0   sin(x[2])^2*(J2+L2^2*M3+LC2^2*M2)   0.0   0.0   0.0   0.0   0.0   0.0   -L2*cos(x[3])*sin(x[2])   0.0   L2*sin(x[2])*sin(x[3])   -L2*cos(x[3])*sin(x[2])   0.0   L2*sin(x[2])*sin(x[3])   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0
                0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   J1+L1^2*(M2+M3)+LC1^2*M1   L1*(L2*M3+LC2*M2)*(sin(x[4])*sin(x[5])+cos(x[4])*cos(x[5])*cos(x[6]))   -L1*cos(x[4])*sin(x[5])*sin(x[6])*(L2*M3+LC2*M2)   0.0   0.0   0.0   (sqrt(3)*L1*sin(x[4]))*0.5   (L1*sin(x[4]))*0.5   L1*cos(x[4])   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0
                0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   L1*(L2*M3+LC2*M2)*(sin(x[4])*sin(x[5])+cos(x[4])*cos(x[5])*cos(x[6]))   J2+L2^2*M3+LC2^2*M2   0.0   0.0   0.0   0.0   (sqrt(3)*L2*sin(x[5]))*0.5-(L2*cos(x[5])*sin(x[6]))*0.5   (L2*sin(x[5]))*0.5+(sqrt(3)*L2*cos(x[5])*sin(x[6]))*0.5   L2*cos(x[5])*cos(x[6])   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0
                0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   -L1*cos(x[4])*sin(x[5])*sin(x[6])*(L2*M3+LC2*M2)   0.0   sin(x[5])^2*(J2+L2^2*M3+LC2^2*M2)   0.0   0.0   0.0   -(L2*cos(x[6])*sin(x[5]))*0.5   (sqrt(3)*L2*cos(x[6])*sin(x[5]))*0.5   -L2*sin(x[5])*sin(x[6])   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0
                0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   J1+L1^2*(M2+M3)+LC1^2*M1   L1*(L2*M3+LC2*M2)*(sin(x[7])*sin(x[8])+cos(x[7])*cos(x[8])*cos(x[9]))   -L1*cos(x[7])*sin(x[8])*sin(x[9])*(L2*M3+LC2*M2)   0.0   0.0   0.0   -(sqrt(3)*L1*sin(x[7]))*0.5   (L1*sin(x[7]))*0.5   L1*cos(x[7])   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0
                0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   L1*(L2*M3+LC2*M2)*(sin(x[7])*sin(x[8])+cos(x[7])*cos(x[8])*cos(x[9]))   J2+L2^2*M3+LC2^2*M2   0.0   0.0   0.0   0.0   -(L2*cos(x[8])*sin(x[9]))*0.5-(sqrt(3)*L2*sin(x[8]))*0.5   (L2*sin(x[8]))*0.5-(sqrt(3)*L2*cos(x[8])*sin(x[9]))*0.5   L2*cos(x[8])*cos(x[9])   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0
                0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   -L1*cos(x[7])*sin(x[8])*sin(x[9])*(L2*M3+LC2*M2)   0.0   sin(x[8])^2*(J2+L2^2*M3+LC2^2*M2)   0.0   0.0   0.0   -(L2*cos(x[9])*sin(x[8]))*0.5   -(sqrt(3)*L2*cos(x[9])*sin(x[8]))*0.5   -L2*sin(x[8])*sin(x[9])   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0
                0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0
                0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0
                0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0
                0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0
                0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0
                0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0
                0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0
                0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0
                0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0
                0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0
                0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0
                0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0
                0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0
                0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0
                0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0
            ]
            # Extends Fdx to contain equations for disturbance model. Hard-coded for pendulum disturbance model
            return [Fdx_tmp         zeros(33,2)   zeros(33,1)
                  zeros(2,33)  [1.0 0.0; 0.0 1.0] zeros(2,1)
                  zeros(1,33)       zeros(1,2)    zeros(1,1)]
        end
    end
    term_func(t) = (λ_func(t)')*Fdx(dx_func(t,1:2num_dyn_vars_adj),x_func(t,1:2num_dyn_vars_adj))*z_func(t)[num_dyn_vars_adj+nxw+nw+1:end]

    # term_func(t) = (λ_func(t)')*Fdx(dx_func(t),x_func(t))*x_func(t)[num_dyn_vars_adj+1:end]   DYN CASE

    println("End values of trajectories: ", integral_sens_1[end], "; ", int_with_lambda_2[end], "; ", post_partial_3[end], "+", -term_func.(N*Ts)+term_func(0.0), "=", post_partial_3[end]-term_func.(N*Ts)+term_func(0.0), "; ", final_expression_4[end], "+", -term_func.(N*Ts)+term_func(0.0), "=", final_expression_4[end]-term_func.(N*Ts)+term_func(0.0))

    deb_rem_6 = [stepstep_sol.u[ind][6] for ind=eachindex(stepstep_sol.u)]

    # PREMISES: integral_sens_1 should match gradient on original sum-cost well. 
    # int_with_lambda_2 should be similar, with a little bit of drift, otherwise something is wrong with our forward solution
    # DEBUGGING: post_partial_3 can look a little different, since partial integration seems to have an unintuittive effect on numerical errors.
    # However, the final value should match integral_sens_1 as well as possible, otherwise we will have issues.
    # If adjoint solution is correct, post_partial_3 and final_expression_4 should match well, since that's where we replace lambda-expressions.
    # NOTE: If you seem to get large numerical mismatch, try to disable disturbances and use a smoother input signal, e.g. a sinusoid. If this reduces the numerical
    # mismatch, then perhaps the earlier mismatch is just inherent to the very quickly changing input signals
    # I think it also should be expected that sum-cost and integral-cost will differ somewhat when inputs are so irregular

    plot(integral_sens_1, label="int_cost_sens")
    plot!(int_with_lambda_2, label="int_sens_with_lam")
    plot!(post_partial_3-term_func.(0.0:Ts:N*Ts).+term_func(0.0), label="post_partial")
    plot!(final_expression_4-term_func.(0.0:Ts:N*Ts).+term_func(0.0), label="adj_final")

    # # ------------------ EXTRA: COMPARING SUM AND INTEGRAL COST -----------------------

    # Comparing sum and integral costs:
    intcosttraj = [stepstep_sol.u[ind][5] for ind=eachindex(stepstep_sol.u)]
    sumcosttraj = [(1/(N+1))*sum( ( Y[1:y_len*ind] - Y1[1:y_len*ind,1] ).^2 ) for ind=1:N+1]
    plot(intcosttraj)
    plot!(sumcosttraj)


    # Comparing numerical cost gradient with integral cost gradient
    sumcosttraj1 = [(1/N)*sum( ( Y[1:y_len*ind] - Y1[1:y_len*ind,1] ).^2 ) for ind=1:N+1]
    sumcosttraj2 = [(1/N)*sum( ( Y[1:y_len*ind] - Y2[1:y_len*ind,1] ).^2 ) for ind=1:N+1]
    sumcostgrad = (sumcosttraj2-sumcosttraj1)/my_δ
    plot(integral_sens_1, label="int_cost_sens")
    plot!(sumcostgrad, label="sum_cost_numsens")
end

function adjoint_dist_debug_new(expid::String)
    # ------------------------------------------------------------------------
    # ------------------------------ Setup -----------------------------------
    # ------------------------------------------------------------------------
    exp_data, isws = get_experiment_data(expid)
    W_meta = exp_data.W_meta
    Y = exp_data.Y
    N = size(Y,1)÷y_len-1
    Nw = exp_data.Nw
    u = exp_data.u
    nxw = W_meta.nx
    n_in = W_meta.n_in
    n_out = W_meta.n_out
    n_tot = nxw*n_in
    η_true = W_meta.η
    dist_par_inds = W_meta.free_par_inds
    free_pars_true = vcat(free_dyn_pars_true, W_meta.η[dist_par_inds])

    @assert (num_dyn_pars == 0 && length(dist_par_inds) == 1) "Only debugging with respect to zero dynamical parameters and 1 disturbance parameter is supported"

    Zm = [randn(Nw, n_tot) for m = 1:1]
    # ---------------------------------------------------------------------------------------------------
    # ------------------ Forward solution of nominal system with forward sensitivity --------------------
    # ---------------------------------------------------------------------------------------------------
    
    η_true = exp_data.get_all_ηs(free_pars_true)
    dmdl_sens = discretize_ct_noise_model_with_sensitivities_alt(get_ct_disturbance_model(η_true, nxw, n_out), δ, dist_par_inds)
    # NOTE: OPTION 1: Use the rows below here for linear interpolation
    @time XWm_sens = simulate_noise_process_mangled(dmdl_sens, Zm)
    wmm_sens(m::Int) = mk_noise_interp(dmdl_sens.Cd, XWm_sens, m, δ)
    xwm_sens(m::Int) = mk_xw_interp(dmdl_sens.Cd, XWm_sens, m, δ)

    # Simulates a second solution without sensitivity to use for numerical estimate of gradient
    my_δ = 0.01
    Ts_exact = 0.0001 # Reducing below 0.0001 didn't seem to provide any additional benefit. Just using Ts also seems to work fine actually
    # # Original
    # @time sol_for1 = solvew_sens(u, wmm(1), free_dyn_pars_true, N)
    # Replacing solvew_sens with parts to carefully compute time impact of every one of them
    @time my_mdl = model_sens_to_use(φ0, u, wmm_sens(1), get_all_θs(free_dyn_pars_true))
    @time my_prob = problem(my_mdl, N, Ts)
    @time sol_for1 = solve(my_prob, saveat = 0:Ts:N*Ts, abstol = abstol, reltol = reltol, maxiters = maxiters)

    # TODO Solving once and then downsampling would be smarter rather than solving twice, but oh well
    ts_exact = 0:Ts_exact:N*Ts+Ts_exact/2
    sol_for_exact = solve(my_prob, saveat = ts_exact, abstol = abstol, reltol = reltol, maxiters = maxiters)

    η_2 = exp_data.get_all_ηs(free_pars_true.+my_δ)
    dmdl_2 = discretize_ct_noise_model(get_ct_disturbance_model(η_2, nxw, n_out), δ)
    # NOTE: OPTION 1: Use the rows below here for linear interpolation
    @time XWm_2 = simulate_noise_process_mangled(dmdl_2, Zm)
    wmm_2(m::Int) = mk_noise_interp(dmdl_2.Cd, XWm_2, m, δ)
    sol_for2 = solvew(u, wmm_2(1), free_dyn_pars_true, N)

    Y1, sens1 = h_comp(sol_for1, get_all_θs(free_dyn_pars_true))
    Y2 = h(sol_for2, get_all_θs(free_dyn_pars_true))
    Y1 = vcat(Y1...)
    Y2 = vcat(Y2...)
    sens1 = vcat(sens1...)

    cost1 = get_cost_value(Y[:,1], Y1[:,1:1])   # Reshaping vector Y1 to Matrix (I think they are vectors to begin with)
    cost2 = get_cost_value(Y[:,1], Y2[:,1:1])   # Reshaping vector Y2 to Matrix (I think they are vectors to begin with)

    for_est = first(get_cost_gradient(Y[:, 1], Y1[:,1:1], [sens1]))
    num_est = (cost2-cost1)/my_δ


    # ---------------------------------------------------------------------------------------------------
    # ------------------------------ Adjoint setup and gradient estimate --------------------------------
    # ---------------------------------------------------------------------------------------------------

    # Extracts entire state trajecotry from first forward solution. Adjoint version of state
    xvec1 = vcat(transpose(h_debug_with_sens(sol_for_exact, get_all_θs(free_dyn_pars_true)))...)   # transpose so that result is a matrix, with each row being one of the inner vectors (h_sens_deb returns a vector of vectors)

    # Creates functions for output, states, and their derivatives using interpolation.
    # y_func  = linear_interpolation_multivar(Y[:,1], Ts, y_len)
    y_func = extrapolate(scale(interpolate(transpose(reshape(Y[:,1], y_len, N+1)), (BSpline(interp_type), NoInterp())), 0.0:Ts:N*Ts, 1:y_len), Line())
    dy_est  = (Y[y_len+1:end,1]-Y[1:end-y_len,1])/Ts
    # dy_func = linear_interpolation_multivar(dy_est, Ts, y_len)
    dy_func = extrapolate(scale(interpolate(transpose(reshape(dy_est, y_len, N)), (BSpline(interp_type), NoInterp())), 0.0:Ts:(N-1)*Ts, 1:y_len), Line())
    # x_func  = get_mvar_cubic(ts_exact, xvec1)
    x_func = extrapolate(scale(interpolate(xvec1, (BSpline(interp_type), NoInterp())), ts_exact, 1:size(xvec1,2)), Line())
    # der_est  = get_der_est(ts_exact, x_func)
    der_est  = get_der_est2(ts_exact, x_func, size(xvec1, 2))
    dx_func = extrapolate(scale(interpolate(der_est, (BSpline(interp_type), NoInterp())), ts_exact[1:end-1], 1:size(der_est,2)), Line())
    # dx_func = get_mvar_cubic(ts_exact[1:end-1], der_est)

    # Computing xp0, initial conditions of derivative of x wrt to p
    mdl = model_to_use(φ0, u, wmm_sens(1), get_all_θs(free_dyn_pars_true))
    mdl_sens = model_sens_to_use(φ0, u, wmm_sens(1), get_all_θs(free_dyn_pars_true))
    n_mdl = length(mdl.x0)
    xp0 = reshape(f_sens_deb(mdl_sens.x0,get_all_θs(free_pars_true)), num_dyn_vars_adj, length(f_sens_deb(mdl_sens.x0,get_all_θs(free_pars_true)))÷num_dyn_vars_adj)
    (nx,np) = size(xp0)

    dmdl, B̃, B̃ηa = discretize_ct_noise_model_with_sensitivities_for_adj(get_ct_disturbance_model(η_true, nxw, n_out), δ, dist_par_inds)
    vm(m::Int) = mk_v_ZOH(Zm[m], δ)
    XWm_adj = simulate_noise_process_mangled(dmdl, Zm)
    xwm_adj(m::Int) = mk_xw_interp(dmdl.Cd, XWm_adj, m, δ)
    nw = size(dmdl.Cd, 1)
    mdl_adj, get_Gp, debugs = model_adj_to_use_dist_sens_new(u, wmm_sens(1), xwm_sens(1), get_all_θs(free_pars_true), N*Ts, x_func, x_func, y_func, dy_func, xp0, dx_func, dx_func, η_true, 1)
    adj_prob = problem_reverse(mdl_adj, N, Ts)
    adj_sol = solve(adj_prob, saveat = 0:Ts:N*Ts, abstol =  abstol, reltol = reltol,
        maxiters = maxiters)
    adj_sol_exact = solve(adj_prob, saveat = ts_exact, abstol =  abstol, reltol = reltol,
        maxiters = maxiters)

    Gp = first(get_Gp(adj_sol))

    println("Num: $num_est, for: $for_est, adj: $Gp")

    # # --------- Obtaining lambda functions  - VER 1 ---------
    λs = [adj_sol_exact.u[end-ind+1][1:nx] for ind=eachindex(adj_sol_exact.u)]
    λ_func = linear_interpolation_multivar(vcat(λs...), Ts_exact, length(λs[1]))
    # --------- More accurate lambda functions - VER 2 - ONLY FOR PENDULUM ---------
    # if model_id == PENDULUM
    #     λ_func, _ = solve_accurate_adjoint(N, Ts, x_func, dx_func, x_func, y_func, 0)
    # end

    # --------------------------------------------------------------------------------------------------------------
    # ------------------ BONUS: COMPARING ADJOINT SOLUTION TRAJECTORY TO FORWARD SENSITIVITY -----------------------
    # --------------------------------------------------------------------------------------------------------------

    (get_Gp_debug, get_term_debug) = debugs # Only get_term_debug seems used
    _, integral, term = get_Gp_debug(adj_sol)


    # Extracts beta and term trajectories from adjoint solution
    _, sens_deb = h_sens_deb(sol_for1, get_all_θs(free_dyn_pars_true))
    xps = hcat(sens_deb...)
    term_vec = get_term_debug(adj_sol, xps, 0:Ts:(N*Ts))    # ONLY USED AT VERY END. BUT VERY NICE TO HAVE :D
    beta = [adj_sol.u[end-ind+1][nx+1] for ind=1:length(adj_sol.u)]     # ALSO VERY NICE TO HAVE FOR DEBUGGING AND COMPARING TO INTEGRAL COST

    # TODO: Consider generalizing the exact pendulum-part for disturbance parameters
    # it currently only works for m, L, and k!!!!!!! <-------------------------------
    ##################################################################################
    # if model_id == PENDULUM
    #     function get_pend_betas(λs_func, x, dx_func, xps, N, Ts)
    #         times = 0.0:Ts:N*Ts

    #         mint(z,p,t) = [-λs_func(t)[3]*dx_func(t)[4] - λs_func(t)[4]*(dx_func(t)[5]+g)]
    #         Lint(z,p,t) = [2*λs_func(t)[2]*L]
    #         kint(z,p,t) = [-λs_func(t)[3]*abs(x(t)[4])*x(t)[4] - λs_func(t)[4]*abs(x(t)[5])*x(t)[5]]
    #         mprob = ODEProblem(mint, [0.0], (0.0, N*Ts), [])
    #         Lprob = ODEProblem(Lint, [0.0], (0.0, N*Ts), [])
    #         kprob = ODEProblem(kint, [0.0], (0.0, N*Ts), [])
    #         msol = DifferentialEquations.solve(mprob, Tsit5(), reltol=reltol, abstol=abstol, saveat=times)
    #         Lsol = DifferentialEquations.solve(Lprob, Tsit5(), reltol=reltol, abstol=abstol, saveat=times)
    #         ksol = DifferentialEquations.solve(kprob, Tsit5(), reltol=reltol, abstol=abstol, saveat=times)
    #         mvec = [msol.u[end][1]-msol.u[i][1] for i=eachindex(msol)]
    #         Lvec = [Lsol.u[end][1]-Lsol.u[i][1] for i=eachindex(Lsol)]
    #         kvec = [ksol.u[end][1]-ksol.u[i][1] for i=eachindex(ksol)]

    #         # ---------------- Getting term ----------------
    #         Fdx = t -> vcat([1   0   0          0   0   2x(t)[1]    0
    #                         0   1   0          0   0   2x(t)[2]    0
    #                         0   0   -x(t)[1]   m   0   0           0
    #                         0   0   -x(t)[2]   0   m   0           0], zeros(3,7))

    #         term = zeros(length(times))
    #         for ind=eachindex(adj_sol.u)
    #             term[ind] = ((λs_func(times[ind])')*Fdx(times[ind]))*xps[:,ind]
    #         end

    #         return mvec, Lvec, kvec, term
    #     end

    #     # NOTE: term only valid for parameter corresponding to xps
    #     βm, βL, βk, term_acc = get_pend_betas(λ_func, x_func, dx_func, xps, N, Ts)
    # end
    
    cost_grad_plot = [first(get_cost_gradient(Y[1:y_len*ind,1], Y1[1:y_len*ind,1:1], [sens1[1:y_len*ind,1:1]])) for ind=1:size(Y,1)÷y_len]  
    plot(beta[1].-beta .- term_vec.+term_vec[1], label="adj_est")
    # if model_id == PENDULUM
    #     # Interestingly enough this plot always seems to be the most off, less so with accurate version, but still
    #     plot!(βk[1].-βk .- term_acc.+term_acc[1], label="other_est(acc/other)")
    # end
    plot!(cost_grad_plot, label="for_est")

    # -----------------------------------------------------------------------------------------------------------------------
    # ------------------ BONUS: STEP-BY-STEP ANALYSIS OF ADJOINT SOLUTION, TO SEE WHERE ERRORS APPEAR -----------------------
    # -----------------------------------------------------------------------------------------------------------------------

    mydelta = 0.01
    dλ_func(t) = (λ_func(t+mydelta)-λ_func(t))/mydelta
    stepbystep_mdl = model_stepbystep_dist(u, wmm_sens(1), xwm_sens(1), vm(1), get_all_θs(free_dyn_pars_true), y_func, dy_func, x_func, dx_func, λ_func, dλ_func, deb_Fp, B̃, B̃ηa, η_true, N*Ts)

    stepstep_prob = problem(stepbystep_mdl, N, Ts)
    stepstep_sol = solve(stepstep_prob, saveat = 0:Ts:N*Ts, abstol = abstol, reltol = reltol, maxiters = maxiters)
    integral_sens_1 = [stepstep_sol.u[ind][1] for ind=eachindex(stepstep_sol.u)]
    int_with_lambda_2 = [stepstep_sol.u[ind][2] for ind=eachindex(stepstep_sol.u)]
    post_partial_3 = [stepstep_sol.u[ind][3] for ind=eachindex(stepstep_sol.u)]
    final_expression_4 = [stepstep_sol.u[ind][4] for ind=eachindex(stepstep_sol.u)]

    function Fdx(dx,x)
        if model_id == PENDULUM
            Fdx_tmp = vcat([1   0   0    0   0   2x[1]    0
                            0   1   0    0   0   2x[2]    0
                            0   0   -x[1]   m   0   0     0
                            0   0   -x[2]   0   m   0     0], zeros(3,7))
            return Fdx_tmp
        elseif model_id == DELTA
            Fdx_tmp =  [   # x and dx have to be adjoint versions of states, i.e. include model output as a state
                1.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   -L1*sin(x[1])   L1*cos(x[1])   0.0   -L1*sin(x[1])   L1*cos(x[1])   0.0   0.0   0.0
                0.0   1.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   L2*cos(x[2])*sin(x[3])   -L2*sin(x[2])   L2*cos(x[2])*cos(x[3])   L2*cos(x[2])*sin(x[3])   -L2*sin(x[2])   L2*cos(x[2])*cos(x[3])   0.0   0.0   0.0
                0.0   0.0   1.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   L2*cos(x[3])*sin(x[2])   0.0   -L2*sin(x[2])*sin(x[3])   L2*cos(x[3])*sin(x[2])   0.0   -L2*sin(x[2])*sin(x[3])   0.0   0.0   0.0
                0.0   0.0   0.0   1.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   -(sqrt(3)*L1*sin(x[4]))*0.5   -(L1*sin(x[4]))*0.5   -L1*cos(x[4])   0.0   0.0   0.0   0.0   0.0   0.0
                0.0   0.0   0.0   0.0   1.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   (L2*cos(x[5])*sin(x[6]))*0.5-(sqrt(3)*L2*sin(x[5]))*0.5   -(L2*sin(x[5]))*0.5-(sqrt(3)*L2*cos(x[5])*sin(x[6]))*0.5   -L2*cos(x[5])*cos(x[6])   0.0   0.0   0.0   0.0   0.0   0.0
                0.0   0.0   0.0   0.0   0.0   1.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   (L2*cos(x[6])*sin(x[5]))*0.5   -(sqrt(3)*L2*cos(x[6])*sin(x[5]))*0.5   L2*sin(x[5])*sin(x[6])   0.0   0.0   0.0   0.0   0.0   0.0
                0.0   0.0   0.0   0.0   0.0   0.0   1.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   (sqrt(3)*L1*sin(x[7]))*0.5   -(L1*sin(x[7]))*0.5   -L1*cos(x[7])   0.0   0.0   0.0
                0.0   0.0   0.0   0.0   0.0   0.0   0.0   1.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   (L2*cos(x[8])*sin(x[9]))*0.5+(sqrt(3)*L2*sin(x[8]))*0.5   (sqrt(3)*L2*cos(x[8])*sin(x[9]))*0.5-(L2*sin(x[8]))*0.5   -L2*cos(x[8])*cos(x[9])   0.0   0.0   0.0
                0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   1.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   (L2*cos(x[9])*sin(x[8]))*0.5   (sqrt(3)*L2*cos(x[9])*sin(x[8]))*0.5   L2*sin(x[8])*sin(x[9])   0.0   0.0   0.0
                0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   J1+L1^2*(M2+M3)+LC1^2*M1   L1*(L2*M3+LC2*M2)*(sin(x[1])*sin(x[2])+cos(x[1])*cos(x[2])*cos(x[3]))   -L1*cos(x[1])*sin(x[2])*sin(x[3])*(L2*M3+LC2*M2)   0.0   0.0   0.0   0.0   0.0   0.0   0.0   L1*sin(x[1])   -L1*cos(x[1])   0.0   L1*sin(x[1])   -L1*cos(x[1])   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0
                0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   L1*(L2*M3+LC2*M2)*(sin(x[1])*sin(x[2])+cos(x[1])*cos(x[2])*cos(x[3]))   J2+L2^2*M3+LC2^2*M2   0.0   0.0   0.0   0.0   0.0   0.0   0.0   -L2*cos(x[2])*sin(x[3])   L2*sin(x[2])   -L2*cos(x[2])*cos(x[3])   -L2*cos(x[2])*sin(x[3])   L2*sin(x[2])   -L2*cos(x[2])*cos(x[3])   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0
                0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   -L1*cos(x[1])*sin(x[2])*sin(x[3])*(L2*M3+LC2*M2)   0.0   sin(x[2])^2*(J2+L2^2*M3+LC2^2*M2)   0.0   0.0   0.0   0.0   0.0   0.0   -L2*cos(x[3])*sin(x[2])   0.0   L2*sin(x[2])*sin(x[3])   -L2*cos(x[3])*sin(x[2])   0.0   L2*sin(x[2])*sin(x[3])   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0
                0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   J1+L1^2*(M2+M3)+LC1^2*M1   L1*(L2*M3+LC2*M2)*(sin(x[4])*sin(x[5])+cos(x[4])*cos(x[5])*cos(x[6]))   -L1*cos(x[4])*sin(x[5])*sin(x[6])*(L2*M3+LC2*M2)   0.0   0.0   0.0   (sqrt(3)*L1*sin(x[4]))*0.5   (L1*sin(x[4]))*0.5   L1*cos(x[4])   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0
                0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   L1*(L2*M3+LC2*M2)*(sin(x[4])*sin(x[5])+cos(x[4])*cos(x[5])*cos(x[6]))   J2+L2^2*M3+LC2^2*M2   0.0   0.0   0.0   0.0   (sqrt(3)*L2*sin(x[5]))*0.5-(L2*cos(x[5])*sin(x[6]))*0.5   (L2*sin(x[5]))*0.5+(sqrt(3)*L2*cos(x[5])*sin(x[6]))*0.5   L2*cos(x[5])*cos(x[6])   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0
                0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   -L1*cos(x[4])*sin(x[5])*sin(x[6])*(L2*M3+LC2*M2)   0.0   sin(x[5])^2*(J2+L2^2*M3+LC2^2*M2)   0.0   0.0   0.0   -(L2*cos(x[6])*sin(x[5]))*0.5   (sqrt(3)*L2*cos(x[6])*sin(x[5]))*0.5   -L2*sin(x[5])*sin(x[6])   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0
                0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   J1+L1^2*(M2+M3)+LC1^2*M1   L1*(L2*M3+LC2*M2)*(sin(x[7])*sin(x[8])+cos(x[7])*cos(x[8])*cos(x[9]))   -L1*cos(x[7])*sin(x[8])*sin(x[9])*(L2*M3+LC2*M2)   0.0   0.0   0.0   -(sqrt(3)*L1*sin(x[7]))*0.5   (L1*sin(x[7]))*0.5   L1*cos(x[7])   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0
                0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   L1*(L2*M3+LC2*M2)*(sin(x[7])*sin(x[8])+cos(x[7])*cos(x[8])*cos(x[9]))   J2+L2^2*M3+LC2^2*M2   0.0   0.0   0.0   0.0   -(L2*cos(x[8])*sin(x[9]))*0.5-(sqrt(3)*L2*sin(x[8]))*0.5   (L2*sin(x[8]))*0.5-(sqrt(3)*L2*cos(x[8])*sin(x[9]))*0.5   L2*cos(x[8])*cos(x[9])   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0
                0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   -L1*cos(x[7])*sin(x[8])*sin(x[9])*(L2*M3+LC2*M2)   0.0   sin(x[8])^2*(J2+L2^2*M3+LC2^2*M2)   0.0   0.0   0.0   -(L2*cos(x[9])*sin(x[8]))*0.5   -(sqrt(3)*L2*cos(x[9])*sin(x[8]))*0.5   -L2*sin(x[8])*sin(x[9])   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0
                0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0
                0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0
                0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0
                0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0
                0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0
                0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0
                0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0
                0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0
                0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0
                0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0
                0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0
                0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0
                0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0
                0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0
                0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0
            ]
            return Fdx_tmp
        end
    end
    term_func(t) = (λ_func(t)')*Fdx(dx_func(t,1:2num_dyn_vars_adj),x_func(t,1:2num_dyn_vars_adj))*x_func(t, num_dyn_vars_adj+1:2num_dyn_vars_adj)

    println("End values of trajectories: ", integral_sens_1[end], "; ", int_with_lambda_2[end], "; ", post_partial_3[end], "+", -term_func.(N*Ts)+term_func(0.0), "=", post_partial_3[end]-term_func.(N*Ts)+term_func(0.0), "; ", final_expression_4[end], "+", -term_func.(N*Ts)+term_func(0.0), "=", final_expression_4[end]-term_func.(N*Ts)+term_func(0.0))

    deb_rem_6 = [stepstep_sol.u[ind][6] for ind=eachindex(stepstep_sol.u)]

    # PREMISES: integral_sens_1 should match gradient on original sum-cost well. 
    # int_with_lambda_2 should be similar, with a little bit of drift, otherwise something is wrong with our forward solution
    # DEBUGGING: post_partial_3 can look a little different, since partial integration seems to have an unintuittive effect on numerical errors.
    # However, the final value should match integral_sens_1 as well as possible, otherwise we will have issues.
    # If adjoint solution is correct, post_partial_3 and final_expression_4 should match well, since that's where we replace lambda-expressions.
    # NOTE: If you seem to get large numerical mismatch, try to disable disturbances and use a smoother input signal, e.g. a sinusoid. If this reduces the numerical
    # mismatch, then perhaps the earlier mismatch is just inherent to the very quickly changing input signals
    # I think it also should be expected that sum-cost and integral-cost will differ somewhat when inputs are so irregular

    plot(integral_sens_1, label="int_cost_sens")
    plot!(int_with_lambda_2, label="int_sens_with_lam")
    plot!(post_partial_3-term_func.(0.0:Ts:N*Ts).+term_func(0.0), label="post_partial")
    plot!(final_expression_4-term_func.(0.0:Ts:N*Ts).+term_func(0.0), label="adj_final")

    # # ------------------ EXTRA: COMPARING SUM AND INTEGRAL COST -----------------------

    # Comparing sum and integral costs:
    intcosttraj = [stepstep_sol.u[ind][5] for ind=eachindex(stepstep_sol.u)]
    sumcosttraj = [(1/(N+1))*sum( ( Y[1:y_len*ind] - Y1[1:y_len*ind,1] ).^2 ) for ind=1:N+1]
    plot(intcosttraj)
    plot!(sumcosttraj)


    # Comparing numerical cost gradient with integral cost gradient
    sumcosttraj1 = [(1/N)*sum( ( Y[1:y_len*ind] - Y1[1:y_len*ind,1] ).^2 ) for ind=1:N+1]
    sumcosttraj2 = [(1/N)*sum( ( Y[1:y_len*ind] - Y2[1:y_len*ind,1] ).^2 ) for ind=1:N+1]
    sumcostgrad = (sumcosttraj2-sumcosttraj1)/my_δ
    plot(integral_sens_1, label="int_cost_sens")
    plot!(sumcostgrad, label="sum_cost_numsens")
end

function for_sens_debug(expid::String, par_val::Float64, K::Int)
    exp_data, isws = get_experiment_data(expid)
    W_meta = exp_data.W_meta
    Y = exp_data.Y
    N = size(Y,1)÷y_len-1
    Nw = exp_data.Nw
    nx = W_meta.nx
    n_in = W_meta.n_in
    n_tot = nx*n_in
    dist_sens_inds = W_meta.free_par_inds

    Zm = [randn(Nw, n_tot) for m = 1:K]
    # @warn "Using zero disturbance here!"
    # Zm = [zeros(Nw, n_tot) for m = 1:K]

    # all_y_len = length(f_sens(ones(num_dyn_vars)))  

    myδ = 0.01
    Ysim, jacs = simulate_system_sens(exp_data, [par_val], K, dist_sens_inds, isws, Zm)
    Ysim2 = simulate_system(exp_data, [par_val.+myδ], K, dist_sens_inds, isws, Zm)
    ystacked = vcat(Y[:,1:K]...)
    # We can change f to get all states instead of only output. However, then we should ensure that
    # y_len is unchanged, since the true output Y assumes the unchanged y_len. Therefore, we use a 
    # separate my_y_len here so that we can ensure that y_len is unchanged
    my_y_len = length(f(ones(num_dyn_vars), get_all_θs(free_dyn_pars_true)))
    ysim_stacked = reshape(Ysim[:,1:K][:], my_y_len*(N+1)*K, 1)    # Reshapes to obtain a column Matrix
    ysim_stacked2 = reshape(Ysim2[:,1:K][:], my_y_len*(N+1)*K, 1)    # Reshapes to obtain a column Matrix
    jac_stacked = reshape(hcat(jacs[1:K]...)[:], my_y_len*(N+1)*K, 1)    # Reshapes to obtain a column Matrix

    num_y_grad = (ysim_stacked2-ysim_stacked)/myδ
    # Computing costs only works when f returns the same output as Y, and not e.g. the entire state vector
    if my_y_len == y_len
        cost = get_cost_value(ystacked, ysim_stacked)
        cost2 = get_cost_value(ystacked, ysim_stacked2)
        cost_grad = first(get_cost_gradient(ystacked, ysim_stacked, [jac_stacked]))
        num_cost_grad = (cost2-cost)/myδ
    else
        cost = NaN
        cost2 = NaN
        cost_grad = NaN
        num_cost = NaN
    end

    # for delta, something seems funky with computing the output, works fine if we just check sensitivity of every variable...
    return ysim_stacked, ysim_stacked2, jac_stacked, num_y_grad, cost_grad, num_cost_grad
end

# Single-realization cost wrt one parameter
function cost_det_plot_with_sens_1d(expid::String, par_val::Float64, myδ::Float64, num_steps::Int, K::Int)
    exp_data, isws = get_experiment_data(expid)
    W_meta = exp_data.W_meta
    Y = exp_data.Y
    N = size(Y,1)÷y_len-1
    Nw = exp_data.Nw
    nx = W_meta.nx
    n_in = W_meta.n_in
    n_tot = nx*n_in
    dist_sens_inds = W_meta.free_par_inds

    # @warn "Zm now generated for differentiation-first approach! It's all hard-coded right now"
    # Zm = [randn(Nw, 2n_tot) for m = 1:100]#K]
    Zm = [randn(Nw, n_tot) for m = 1:100]#K]
    ystacked = vcat(Y[:,1:K]...)

    par_range = par_val-num_steps*myδ:myδ:par_val+num_steps*myδ
    cost_vals = zeros(length(par_range))
    grad_vals = zeros(length(par_range))
    for (ind,par)=enumerate(par_range)
        # Ysim, jacs = simulate_system_sens(exp_data, [par], K, dist_sens_inds, isws, Zm)
        Ysim, jacs = simulate_system_sens(exp_data, [par], 100, dist_sens_inds, isws, Zm)
        # We can change f to get all states instead of only output. However, then we should ensure that
        # y_len is unchanged, since the true output Y assumes the unchanged y_len. Therefore, we use a 
        # separate my_y_len here so that we can ensure that y_len is unchanged
        my_y_len = length(f(ones(num_dyn_vars), get_all_θs(free_dyn_pars_true)))
        ysim_stacked = reshape(Ysim[:,1:K][:], my_y_len*(N+1)*K, 1)    # Reshapes to obtain a column Matrix
        jac_stacked = reshape(hcat(jacs[1:K]...)[:], my_y_len*(N+1)*K, 1)    # Reshapes to obtain a column Matrix

        # Computing costs only works when f returns the same output as Y, and not e.g. the entire state vector
        if my_y_len == y_len
            cost = get_cost_value(ystacked, ysim_stacked)
            cost_grad = first(get_cost_gradient(ystacked, ysim_stacked, [jac_stacked]))
        else
            throw(DimensionMismatch("my_y_len is not equal to y_len, can't compute cost values"))
        end
        cost_vals[ind] = cost
        grad_vals[ind] = cost_grad
    end

    return par_range, cost_vals, grad_vals
end

# NOTE: Only debugs gradient of cost function
function for_sens_debug_multipar(expid::String, par_vals::Vector{Float64}, K::Int)
    num_costs = zeros(length(par_vals))
    cost_grads = zeros(length(par_vals))

    for ind = eachindex(par_vals)
        _, _, _, _, cost_grads[ind], num_costs[ind] = for_sens_debug(expid, par_vals[ind], K)
    end

    return cost_grads, num_costs
end

function state_sens_debug(expid::String, par_val::Float64, K::Int)
    exp_data, isws = get_experiment_data(expid)
    W_meta = exp_data.W_meta
    Y = exp_data.Y
    N = size(Y,1)÷y_len-1
    Nw = exp_data.Nw
    nx = W_meta.nx
    n_in = W_meta.n_in
    n_tot = nx*n_in
    n_out = W_meta.n_out
    dist_sens_inds = W_meta.free_par_inds

    Zm = [randn(Nw, n_tot) for m = 1:K]


    η = exp_data.get_all_ηs([par_val])
    dmdl = discretize_ct_noise_model_with_sensitivities_alt(get_ct_disturbance_model(η, nx, n_out), δ, dist_sens_inds)
    # # NOTE: OPTION 1: Use the rows below here for linear interpolation
    XWm = simulate_noise_process_mangled(dmdl, Zm)
    wmm(m::Int) = mk_noise_interp(dmdl.Cd, XWm, m, δ)

    myδ = 0.01
    sol1 = solvew_sens(exp_data.u, t -> wmm(1)(t), [par_val], N)
    sol2 = solvew_sens(exp_data.u, t -> wmm(1)(t), [par_val.+myδ], N)
    mat1 = hcat(sol1.u...)
    mat2 = hcat(sol2.u...)
    x1 = mat1[1:num_dyn_vars, :]
    x1θ = mat1[num_dyn_vars+1:end, :]
    x2 = mat2[1:num_dyn_vars, :]
    x2θ = mat2[num_dyn_vars+1:end, :]
    y1 = hcat([f(mat1[:,ind], get_all_θs([par_val])) for ind=1:size(mat1,2)]...)
    y2 = hcat([f(mat2[:,ind], get_all_θs([par_val.+myδ])) for ind=1:size(mat2,2)]...)
    y1θ = hcat([f_sens(mat1[:,ind], get_all_θs([par_val])) for ind=1:size(mat1,2)]...)
    y2θ = hcat([f_sens(mat2[:,ind], get_all_θs([par_val.+myδ])) for ind=1:size(mat2,2)]...)

    num_xθ = (x2-x1)./myδ
    num_yθ = (y2-y1)./myδ

    return x1, x2, x1θ, x2θ, num_xθ, y1, y2, y1θ, y2θ, num_yθ
end

function disturbance_sensitivity_debug(expid::String)
    exp_data, isws = get_experiment_data(expid)
    W_meta = exp_data.W_meta
    nx = W_meta.nx
    dist_sens_inds = W_meta.free_par_inds
    n_in = W_meta.n_in
    n_out = W_meta.n_out
    N = size(exp_data.Y, 1)÷y_len-1
    n_tot = nx*n_in
    Nw = exp_data.Nw
    Zm = [randn(Nw, n_tot) for m = 1:M]

    η1 = W_meta.η
    η2 = copy(η1)
    η2[1] += 0.01

    dmdl1 = discretize_ct_noise_model_with_sensitivities_alt(get_ct_disturbance_model(η1, nx, n_out), δ, dist_sens_inds)
    dmdl2 = discretize_ct_noise_model_with_sensitivities_alt(get_ct_disturbance_model(η2, nx, n_out), δ, dist_sens_inds)
    XWm1 = simulate_noise_process_mangled(dmdl1, Zm)
    XWm2 = simulate_noise_process_mangled(dmdl2, Zm)
    wmm1 = mk_noise_interp(dmdl1.Cd, XWm1, 1, δ)
    wmm2 = mk_noise_interp(dmdl2.Cd, XWm2, 1, δ)
    dwm(t::Float64) = (wmm2(t)-wmm1(t))/0.01
    return wmm1, wmm2, dwm
end