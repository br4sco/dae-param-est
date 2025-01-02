include("../models.jl")
using .DynamicalModels: pendulum_new, pendulum_sensitivity_m

pend_model = let
    m = 0.3                   # [kg]
    L = 6.25                  # [m], gives period T = 5s (T ≅ 2√L) not accounting for friction.
    g = 9.81                  # [m/s^2]
    k = 6.25                  # [1/s^2]
    φ0 = 0.0                  # Initial angle of pendulum from negative y-axis
    
    # ------- The following fields are part of the informal interface for model metadata -------
    get_all_θs(pars::Vector{Float64}) = [pars[1], L, g, k]  # [m, L, g, k]
    free_dyn_pars_true = [m]#Array{Float64}(undef, 0)#[k]# True values of free parameters
    model_nominal = pendulum_new
    model_sens = pendulum_sensitivity_m
    σ_tmp = 0.002                                               # measurement noise variance
    f(x::Vector{Float64}, θ::Vector{Float64}) = x[7]        # Output function, returns the output given the state vector x

    (σ = σ_tmp, m = m, L = L, g = g, k = k, φ0 = 0, get_all_θs = get_all_θs, free_dyn_pars_true = free_dyn_pars_true, model_nominal = model_nominal, model_sens = model_sens, f=f)
end

# model_to_use = pendulum_new
# model_sens_to_use = pendulum_sensitivity_m

# # const free_dyn_pars_true = [m]#Array{Float64}(undef, 0)#[k]# True values of free parameters
# # const num_dyn_vars = 7
# # const num_dyn_vars_adj = 7 # For adjoint method, there might be additional state variables, since outputs need to be baked into the state. Though outputs are already baked in for pendulum
# # use_adjoint = false
# # use_new_adj = true
# # get_all_θs(pars::Vector{Float64}) = [pars[1], L, g, k]#[pars[1], L, pars[2], k]
# # # Each row corresponds to lower and upper bounds of a free dynamic parameter.
# # dyn_par_bounds = [0.01 1e4]#[0.01 1e4; 0.1 1e4; 0.1 1e4]#; 0.1 1e4] #Array{Float64}(undef, 0, 2)
# # @warn "The learning rate dimensiond doesn't deal with disturbance parameters in any nice way, other info comes from W_meta, and this part is hard coded"
# # const_learning_rate = [0.1]#[0.1, 1.0, 0.1]
# # model_sens_to_use = pendulum_sensitivity_m#_with_dist_sens_1#_sans_g_with_dist_sens_1#_with_dist_sens_3#pendulum_sensitivity_k_with_dist_sens_1#pendulum_sensitivity_sans_g#_full
# # model_to_use = pendulum_new
# # model_adj_to_use = my_pendulum_adjoint_konly_new
# # model_adj_to_use_dist_sens = my_pendulum_adjoint_distsensa1_new#_with_dist_sens_3
# # model_adj_to_use_dist_sens_new = my_pendulum_foradj_distsensa1
# # sgd_version_to_use = perform_SGD_adam_new
# # # Models for debug:
# # model_stepbystep = pendulum_adj_stepbystep_NEW#pendulum_adj_stepbystep_k#pendulum_adj_stepbystep_deb
# # model_stepbystep_dist = pendulum_adj_stepbystep_dist_new

# # Fpk = (x, dx) -> [0.; 0.; abs(x[4])*x[4]; abs(x[5])*x[5]; 0.; 0.; 0.;;]
# # Fpm = (x, dx) -> [.0; .0; dx[4]; dx[5]+g; .0; .0; .0;;]
# # FpL = (x, dx) -> [.0; .0; .0; .0; -2L; .0; .0;;]
# # deb_Fp = Fpk

# # f(x::Vector{Float64}, θ::Vector{Float64}) = x[7]               # applied on the state at each step
# # # f_sens should return a matrix/column vector with each row corresponding to a different output component and each column corresponding to a different parameter
# # f_sens(x::Vector{Float64}, θ::Vector{Float64})::Matrix{Float64} = [x[14];;]# x[21] x[28] x[35]]# x[42] x[49]]# x[28]]##[x[14] x[21] x[28] x[35] x[42]]   # NOTE: Hard-coded right now
# # # f_sens(x::Vector{Float64}, θ::Vector{Float64}) = [x[14], x[21], x[28]]                                                                                           #tuesday debug starting here
# # f_sens_deb(x::Vector{Float64}, θ::Vector{Float64}) = x[8:end]
# # f_debug(x::Vector{Float64}, θ::Vector{Float64}) = x[1:7]
# # # The purpose of a separate baseline function is only relevant for delta robot, because of parameter-dependent output function
# # f_sens_baseline(x::Vector{Float64}, θ::Vector{Float64})::Matrix{Float64} = f_sens(x::Vector{Float64}, θ::Vector{Float64})