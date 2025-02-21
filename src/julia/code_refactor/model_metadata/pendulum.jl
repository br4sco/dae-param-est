include("../minimizers.jl")
using .Minimizers: perform_ADAM

pend_model_data = let
    m = 0.3                   # [kg]
    L = 6.25                  # [m], gives period T = 5s (T ≅ 2√L) not accounting for friction.
    g = 9.81                  # [m/s^2]
    k = 6.25                  # [1/s^2]
    φ0 = 0.0                  # Initial angle of pendulum from negative y-axis
    
    num_dyn_vars = 7   # Number of variables in the nominal model

    # NOTE: If the sensitivity with respect to some disturbance parameters will also be considered,
    # this needs to be set in the get_disturbance_free_pars()-function in run_experiment.jl

    # ------- The following fields are part of the informal interface for model metadata -------
    get_all_pars(pars::Vector{Float64}) = [m, L, g, pars[1]]  # [m, L, g, k]
    free_dyn_pars_true = [k]#Array{Float64}(undef, 0)#[k]# True values of free parameters
    init_learning_rate = [1.0] # The initial learning rate for each component of free_dyn_pars_true
    # Left column contains lower bound for parameters, right column contains upper bound
    par_bounds = [0.1 1e4; 0.01 1e4]#[0.01 1e4; 0.1 1e4; 0.1 1e4]#; 0.1 1e4] #Array{Float64}(undef, 0, 2)
    model_nominal = pendulum
    model_sens = pendulum_forward_k_1dist                                 # For forward sensitivity
    model_adjoint = pendulum_adjoint_k_1dist                              # For adjoint sensitivity
    model_adjoint_odedist = pendulum_adjoint_k_1dist_ODEdist           # Adjoint when disturbances are given by an ODE incorporated into the DAE
    σ = 0.002                                               # measurement noise variance

    # Should return the initial sensitivity of all state variables, given parameters and initial input and disturbance
    function get_sens_init(θ::Vector{Float64}, u0::Vector{Float64}, w0::Vector{Float64})::Matrix{Float64}
        pend0, dpend0 = get_pendulum_initial(θ, u0[1], w0[1], φ0)
        # sm, _= get_pendulum_initial_msens(θ, u0[1], w0[1], φ0, pend0, dpend0)
        sk, _= get_pendulum_initial_ksens(θ, u0[1], w0[1], φ0, pend0, dpend0)
        sa, _= get_pendulum_initial_distsens(θ, u0[1], w0[1], φ0, pend0, dpend0)
        # The return value should be a matrix with state-component along the rows and parameter index along the columns
        hcat(sk, sa)
    end

    f(x::Vector{Float64}, θ::Vector{Float64}) = x[7]        # Output function, returns the output given the state vector x
    f_sens(x::Vector{Float64}, θ::Vector{Float64})::Matrix{Float64} = [x[14]   x[21]]# x[21] x[28] x[35] ;;]# x[42] x[49]]# x[28] ;;]##[x[14] x[21] x[28] x[35] x[42];;]   # Returns sensitivities of the output
    # THERE MUST BE BETTER WAY OF DOING THIS!!!
    # f_all_sens(x::Vector{Float64}, θ::Vector{Float64}) = x[8:end]            # Returns the sensitivity of all states, but only the sensitivities
    f_all_adj(x::Vector{Float64}, θ::Vector{Float64}) = x[1:num_dyn_vars]    # Returns all nominal states, including the model output
    dθ = length(free_dyn_pars_true)
    ny = length(f(ones(7), get_all_pars(free_dyn_pars_true)))
    # E.g. the Delta robot needs a slightly different implementation of ADAM that was hard to generalize. 
    # Therefore, we attach the used minimizer to the model through minimizer so that it can vary between models.
    minimizer = perform_ADAM

    # TODO: Is f_sens_base really not relevant for pendulum? Shouldn't it become relevant when we do ID of forward sens????? Edit: Im pretty sure it should be relevant, but do check it out
    (σ = σ, m = m, L = L, g = g, k = k, φ0 = .0, get_all_pars = get_all_pars, free_dyn_pars_true = free_dyn_pars_true, par_bounds = par_bounds,
        ny = ny, model_nominal = model_nominal, model_sens = model_sens, model_adjoint = model_adjoint, model_adjoint_odedist=model_adjoint_odedist,
        f=f, f_sens=f_sens, f_sens_baseline=f_sens, f_all_adj=f_all_adj, dθ = dθ, minimizer=minimizer, init_learning_rate=init_learning_rate, get_sens_init=get_sens_init)
end