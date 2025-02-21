include("../minimizers.jl")
using .Minimizers: perform_ADAM, perform_ADAM_deltaversion

delta_model_data = let
    L0 = 1.0
    L1 = 1.5
    L2 = 2.0
    L3 = 0.5
    LC1 = 0.75
    LC2 = 1.0
    M1 = 0.1
    M2 = 0.1
    M3 = 0.3
    J1 = 0.4
    J2 = 0.4
    g = 0.0     # This is the gravity-compensated delta robot model
    γ = 1.0

    num_dyn_vars = 30       # Number of variables in the nominal model
    num_dyn_vars_adj = 33   # Number of variables in model as used in adjoint method, i.e. when state contains model output

    # NOTE: If the sensitivity with respect to some disturbance parameters will also be considered,
    # this needs to be set in the get_disturbance_free_pars()-function in run_experiment.jl

    # ------- The following fields are part of the informal interface for model metadata -------
    get_all_pars(pars::Vector{Float64}) = vcat(pars[1:11], [g], pars[12])#[L0, L1, L2, L3, LC1, LC2, M1, M2, M3, J1, J2, g, γ]#Array{Float64}(undef, 0)
    free_dyn_pars_true = [L0, L1, L2, L3, LC1, LC2, M1, M2, M3, J1, J2, γ]#Array{Float64}(undef, 0)   # True values of free parameters
    # TODO: This is no good, with them being specified elsewhere............. Change that! Or how did pendulum do it????
    # The initial learning rate for each component of free_dyn_pars_true, as well as for each components of the free disturbance parameters η specified elsewhere
    init_learning_rate = [0.1, 0.1, 0.1, 0.05, 0.05, 0.1, 0.02, 0.02, 0.05, 0.05, 0.05, 0.2]
    # init_learning_rate = init_learning_rate[12]
    # Left column contains lower bound for parameters, right column contains upper bound
    par_bounds = hcat(fill(0.01, 12, 1), fill(1e4, 12, 1)) #Array{Float64}(undef, 0, 2)
    # par_bounds[3,1] = 1.0 # Setting lower bound for L2
    model_nominal = delta_robot
    model_sens = delta_forward_allpar_alldist                        # For forward sensitivity
    model_adjoint = delta_adjoint_allpar_alldist              # For adjoint sensitivity
    model_adjoint_odedist = delta_adjoint_allpar_alldist_ODEdist             # Adjoint when disturbances are given by an ODE incorporated into the DAE

    σ = 0.002                                               # measurement noise variance
    minimizer = perform_ADAM_deltaversion  

    # Output function, returns the output (position of end effector) given the state vector x
    f(x::Vector{Float64}, p::Vector{Float64}) = [p[3]*sin(x[2])*sin(x[3])
        p[2]*cos(x[1]) + p[3]*cos(x[2]) + p[1] - p[4]
        p[2]*sin(x[1]) + p[3]*sin(x[2])*cos(x[3])]

    # Returns all nominal states, including the model output
    f_all_adj(x::Vector{Float64}, p::Vector{Float64}) = vcat(x[1:num_dyn_vars], f(x, p))

    ##################################################################################################################################################
    # f_sens should return a matrix with each row corresponding to a different output component and each column corresponding to a different parameter
    ##################################################################################################################################################

    # sans_p-part
    f_sens_base(x::Vector{Float64}, p::Vector{Float64}, par_ind::Int)::Matrix{Float64} = 
        [p[3]*cos(x[2])*sin(x[3])*x[30*par_ind+2]+p[3]*cos(x[3])*sin(x[2])*x[30*par_ind+3]
        -p[2]*sin(x[1])*x[30*par_ind+1]-p[3]*sin(x[2])*x[30*par_ind+2]
        p[2]*cos(x[1])*x[30*par_ind+1]+p[3]*cos(x[2])*cos(x[3])*x[30*par_ind+2]-p[3]*sin(x[2])*sin(x[3])*x[30*par_ind+3];;]
    # p-parts
    f_sens_L0(x::Vector{Float64})::Matrix{Float64} = [0.0; 1.0; 0.0;;]
    f_sens_L1(x::Vector{Float64})::Matrix{Float64} = [0.0; cos(x[1]); sin(x[1]);;]
    f_sens_L2(x::Vector{Float64})::Matrix{Float64} = [sin(x[2])*sin(x[3]); cos(x[2]); cos(x[3])*sin(x[2]);;]
    f_sens_L3(x::Vector{Float64})::Matrix{Float64} = [0.0; -1.0; 0.0;;]
    f_sens_other(x::Vector{Float64})::Matrix{Float64} = zeros(3,1)

    # # Sensitivity wrt to L1. To create a column-matrix, make sure to use ;; at the end, e.g. [...;;]
    # f_sens(x::Vector{Float64}, p::Vector{Float64})::Matrix{Float64} = f_sens_base(x, p, 1)+f_sens_L1(x)

    # # Sensitivity wrt to whichever individual parameter except L0, L1, L2, L3, all others are the same
    # f_sens(x::Vector{Float64}, p::Vector{Float64})::Matrix{Float64} = f_sens_base(x, p, 1)+f_sens_other(x)

    # # Sensitivity wrt to [L1, M1, J1]
    # f_sens(x::Vector{Float64}, p::Vector{Float64})::Matrix{Float64} = hcat(f_sens_L1(x)+f_sens_base(x,p,1), f_sens_other(x)+f_sens_base(x,p,2), f_sens_other(x)+f_sens_base(x,p,3))

    # # Sensitivity wrt to γ and one disturbance parameter
    # f_sens(x::Vector{Float64}, p::Vector{Float64})::Matrix{Float64} = [f_sens_base(x, p, 1)+f_sens_other(x)    f_sens_base(x, p, 2)+f_sens_other(x)]

    # Sensitivity wrt to debug2-case parameters
    # f_sens(x::Vector{Float64}, p::Vector{Float64})::Matrix{Float64} = hcat(f_sens_base(x, p, 1)+f_sens_other(x), f_sens_base(x, p, 2)+f_sens_other(x))#, f_sens_base(x, p, 3)+f_sens_L2(x))#, 
    #     # f_sens_base(x, p, 4)+f_sens_L3(x), f_sens_base(x, p, 5)+f_sens_other(x), f_sens_base(x, p, 6)+f_sens_other(x), f_sens_base(x, p, 7)+f_sens_other(x),
    #     # f_sens_base(x, p, 8)+f_sens_other(x), f_sens_base(x, p, 9)+f_sens_other(x), f_sens_base(x, p, 10)+f_sens_other(x), f_sens_base(x, p, 11)+f_sens_other(x),
    #     # f_sens_base(x, p, 12)+f_sens_other(x))
    # f_sens(x::Vector{Float64}, p::Vector{Float64})::Matrix{Float64} = hcat(f_sens_base(x, p, 1)+f_sens_L1(x), f_sens_base(x, p, 2)+f_sens_other(x))

    # # Sensitivity for deb1 tests
    # f_sens(x::Vector{Float64}, p::Vector{Float64})::Matrix{Float64} = f_sens_base(x, p, 1)+f_sens_L2(x)

    # # Sensitivity wrt to one disturbance parameter
    # f_sens(x::Vector{Float64}, p::Vector{Float64})::Matrix{Float64} = f_sens_base(x, p, 1)+f_sens_other(x)

    # # Sensitivity wrt to all parameters
    # f_sens(x::Vector{Float64}, p::Vector{Float64})::Matrix{Float64} = hcat(f_sens_base(x, p, 1)+f_sens_L0(x), f_sens_base(x, p, 2)+f_sens_L1(x), f_sens_base(x, p, 3)+f_sens_L2(x), 
    #     f_sens_base(x, p, 4)+f_sens_L3(x), f_sens_base(x, p, 5)+f_sens_other(x), f_sens_base(x, p, 6)+f_sens_other(x), f_sens_base(x, p, 7)+f_sens_other(x),
    #     f_sens_base(x, p, 8)+f_sens_other(x), f_sens_base(x, p, 9)+f_sens_other(x), f_sens_base(x, p, 10)+f_sens_other(x), f_sens_base(x, p, 11)+f_sens_other(x),
    #     f_sens_base(x, p, 12)+f_sens_other(x))

    # # Sensitivity wrt to all disturbance parameters
    # f_sens(x::Vector{Float64}, p::Vector{Float64})::Matrix{Float64} = hcat(f_sens_base(x, p, 1)+f_sens_other(x), f_sens_base(x, p, 2)+f_sens_other(x), f_sens_base(x, p, 3)+f_sens_other(x), 
    #     f_sens_base(x, p, 4)+f_sens_other(x), f_sens_base(x, p, 5)+f_sens_other(x), f_sens_base(x, p, 6)+f_sens_other(x), f_sens_base(x, p, 7)+f_sens_other(x),
    #     f_sens_base(x, p, 8)+f_sens_other(x), f_sens_base(x, p, 9)+f_sens_other(x), f_sens_base(x, p, 10)+f_sens_other(x), f_sens_base(x, p, 11)+f_sens_other(x),
    #     f_sens_base(x, p, 12)+f_sens_other(x))

    # Sensitivity wrt to all dynamical parameters AND all disturbance parameters
    f_sens(x::Vector{Float64}, p::Vector{Float64})::Matrix{Float64} = hcat(f_sens_base(x, p, 1)+f_sens_L0(x), f_sens_base(x, p, 2)+f_sens_L1(x), f_sens_base(x, p, 3)+f_sens_L2(x), 
        f_sens_base(x, p, 4)+f_sens_L3(x), f_sens_base(x, p, 5)+f_sens_other(x), f_sens_base(x, p, 6)+f_sens_other(x), f_sens_base(x, p, 7)+f_sens_other(x),
        f_sens_base(x, p, 8)+f_sens_other(x), f_sens_base(x, p, 9)+f_sens_other(x), f_sens_base(x, p, 10)+f_sens_other(x), f_sens_base(x, p, 11)+f_sens_other(x),
        f_sens_base(x, p, 12)+f_sens_other(x), f_sens_base(x, p, 13)+f_sens_other(x), f_sens_base(x, p, 14)+f_sens_other(x), f_sens_base(x, p, 15)+f_sens_other(x), 
        f_sens_base(x, p, 16)+f_sens_other(x), f_sens_base(x, p, 17)+f_sens_other(x), f_sens_base(x, p, 18)+f_sens_other(x), f_sens_base(x, p, 19)+f_sens_other(x),
        f_sens_base(x, p, 20)+f_sens_other(x), f_sens_base(x, p, 21)+f_sens_other(x), f_sens_base(x, p, 22)+f_sens_other(x), f_sens_base(x, p, 23)+f_sens_other(x),
        f_sens_base(x, p, 24)+f_sens_other(x))

    # BASELINE: SHOULD NOT INCLUDE DISTURBANCE PARAMETERS, SINCE BASELINE METHOD CANNOT IDENTIFY THEM ANYWAY
    # Sensitivity wrt to all dynamical parameters
    f_sens_baseline(x::Vector{Float64}, p::Vector{Float64})::Matrix{Float64} = hcat(f_sens_base(x, p, 1)+f_sens_L0(x), f_sens_base(x, p, 2)+f_sens_L1(x), f_sens_base(x, p, 3)+f_sens_L2(x), 
    f_sens_base(x, p, 4)+f_sens_L3(x), f_sens_base(x, p, 5)+f_sens_other(x), f_sens_base(x, p, 6)+f_sens_other(x), f_sens_base(x, p, 7)+f_sens_other(x),
    f_sens_base(x, p, 8)+f_sens_other(x), f_sens_base(x, p, 9)+f_sens_other(x), f_sens_base(x, p, 10)+f_sens_other(x), f_sens_base(x, p, 11)+f_sens_other(x),
    f_sens_base(x, p, 12)+f_sens_other(x))
    # # Sensitivity wrt to whichever individual parameter except L0, L1, L2, L3, all others are the same
    # f_sens_baseline(x::Vector{Float64}, p::Vector{Float64})::Matrix{Float64} = f_sens_base(x, p, 1)+f_sens_other(x)
    # # Sensitivity wrt whichever parameters I felt like while debugging (γ)
    # f_sens_baseline(x::Vector{Float64}, p::Vector{Float64})::Matrix{Float64} = hcat(f_sens_base(x, p, 1)+f_sens_other(x))

    # Should return the initial sensitivity of all state variables, given parameters and initial input and disturbance
    # NOTE: Used for adjoint method, so state variables must contain model output too.
    function get_sens_init(θ::Vector{Float64}, u0::Vector{Float64}, w0::Vector{Float64})::Matrix{Float64}
        z0dyn, dz0dyn, did = get_delta_initial_with_mats(θ, u0, w0)
        zL0, _ = get_delta_initial_L0sens(θ, z0dyn, dz0dyn, did)
        zL1, _ = get_delta_initial_L1sens(θ, z0dyn, dz0dyn, did)
        zL2, _ = get_delta_initial_L2sens(θ, z0dyn, dz0dyn, did)
        zL3, _ = get_delta_initial_L3sens(θ, z0dyn, dz0dyn, did)
        zLC1, _ = get_delta_initial_LC1sens(θ, z0dyn, dz0dyn, did)
        zLC2, _ = get_delta_initial_LC2sens(θ, z0dyn, dz0dyn, did)
        zM1, _ = get_delta_initial_M1sens(θ, z0dyn, dz0dyn, did)
        zM2, _ = get_delta_initial_M2sens(θ, z0dyn, dz0dyn, did)
        zM3, _ = get_delta_initial_M3sens(θ, z0dyn, dz0dyn, did)
        zJ1, _ = get_delta_initial_J1sens(θ, z0dyn, dz0dyn, did)
        zJ2, _ = get_delta_initial_J2sens(θ, z0dyn, dz0dyn, did)
        zγ, _ = get_delta_initial_γsens(θ, z0dyn, dz0dyn, did)
        z0dist = zeros(num_dyn_vars_adj)  # Assuming parameter-independent initial value of w
    
        z0 = vcat(z0dyn, zL0, zL1, zL2, zL3, zLC1, zLC2, zM1, zM2, zM3, zJ1, zJ2, zγ)

        # The return value should be a matrix with state-component along the rows and parameter index along the columns
        hcat(
            vcat(zL0, f_sens_base(z0, θ, 1)+f_sens_L0(z0)),
            vcat(zL1, f_sens_base(z0, θ, 2)+f_sens_L1(z0)),
            vcat(zL2, f_sens_base(z0, θ, 3)+f_sens_L2(z0)),
            vcat(zL3, f_sens_base(z0, θ, 4)+f_sens_L3(z0)),
            vcat(zLC1, f_sens_base(z0, θ, 5)+f_sens_other(z0)),
            vcat(zLC2, f_sens_base(z0, θ, 6)+f_sens_other(z0)),
            vcat(zM1, f_sens_base(z0, θ, 7)+f_sens_other(z0)),
            vcat(zM2, f_sens_base(z0, θ, 8)+f_sens_other(z0)),
            vcat(zM3, f_sens_base(z0, θ, 9)+f_sens_other(z0)),
            vcat(zJ1, f_sens_base(z0, θ, 10)+f_sens_other(z0)),
            vcat(zJ2, f_sens_base(z0, θ, 11)+f_sens_other(z0)),
            vcat(zγ, f_sens_base(z0, θ, 12)+f_sens_other(z0)),
            repeat(z0dist, 1, 12)
        )
        # reshape(vcat(zγ, f_sens(z0, θ)), :, 1)
    end

    dθ = length(free_dyn_pars_true)
    ny = length(f(ones(30), get_all_pars(free_dyn_pars_true)))

    (σ = σ, L0 = L0, L1 = L1, L2 = L2, L3 = L3, LC1 = LC1, LC2 = LC2, M1 = M1, M2 = M2, M3 = M3, J1 = J1, J2 = J2, γ = γ,
    get_all_pars = get_all_pars, free_dyn_pars_true = free_dyn_pars_true, par_bounds = par_bounds,
    ny = ny, model_nominal = model_nominal, model_sens = model_sens, model_adjoint = model_adjoint, model_adjoint_odedist=model_adjoint_odedist,
    f=f, f_sens=f_sens, f_sens_baseline=f_sens, f_all_adj=f_all_adj, dθ = dθ, minimizer=minimizer, init_learning_rate=init_learning_rate, get_sens_init=get_sens_init)
end