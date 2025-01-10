include("../models.jl")
include("../minimizers.jl")
using .DynamicalModels: delta_robot_gc, delta_robot_gc_γsens
using .Minimizers: perform_ADAM, perform_ADAM_deltaversion

delta_mdl = let
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

    # ------- The following fields are part of the informal interface for model metadata -------
    get_all_θs(pars::Vector{Float64}) = [L0, L1, L2, L3, LC1, LC2, M1, M2, M3, J1, J2, g, pars[1]]#[L0, L1, L2, L3, LC1, LC2, M1, M2, M3, J1, J2, γ]#Array{Float64}(undef, 0)
    free_dyn_pars_true = [γ]#Array{Float64}(undef, 0)   # True values of free parameters
    init_learning_rate = [0.1] # The initial learning rate for each component of free_dyn_pars_true
    # Left column contains lower bound for parameters, right column contains upper bound
    par_bounds = [0.01 1e4] #Array{Float64}(undef, 0, 2)
    model_nominal = delta_robot_gc
    model_sens = delta_robot_gc_γsens
    σ = 0.002                                               # measurement noise variance

    # Output function, returns the output (position of end effector) given the state vector x
    f(x::Vector{Float64}, θ::Vector{Float64}) = [θ[3]*sin(x[2])*sin(x[3])
        θ[2]*cos(x[1]) + θ[3]*cos(x[2]) + θ[1] - θ[4]
        θ[2]*sin(x[1]) + θ[3]*sin(x[2])*cos(x[3])]

    ##################################################################################################################################################
    # f_sens should return a matrix with each row corresponding to a different output component and each column corresponding to a different parameter
    ##################################################################################################################################################

    # sans_p-part
    f_sens_base(x::Vector{Float64}, θ::Vector{Float64}, par_ind::Int)::Matrix{Float64} = 
        [θ[3]*cos(x[2])*sin(x[3])*x[30*par_ind+2]+θ[3]*cos(x[3])*sin(x[2])*x[30*par_ind+3]
        -θ[2]*sin(x[1])*x[30*par_ind+1]-θ[3]*sin(x[2])*x[30*par_ind+2]
        θ[2]*cos(x[1])*x[30*par_ind+1]+θ[3]*cos(x[2])*cos(x[3])*x[30*par_ind+2]-θ[3]*sin(x[2])*sin(x[3])*x[30*par_ind+3];;]
    # p-parts
    f_sens_L0(x::Vector{Float64})::Matrix{Float64} = [0.0; 1.0; 0.0;;]
    f_sens_L1(x::Vector{Float64})::Matrix{Float64} = [0.0; cos(x[1]); sin(x[1]);;]
    f_sens_L2(x::Vector{Float64})::Matrix{Float64} = [sin(x[2])*sin(x[3]); cos(x[2]); cos(x[3])*sin(x[2]);;]
    f_sens_L3(x::Vector{Float64})::Matrix{Float64} = [0.0; -1.0; 0.0;;]
    f_sens_other(x::Vector{Float64})::Matrix{Float64} = zeros(3,1)

    # # Sensitivity wrt to L1. To create a column-matrix, make sure to use ;; at the end, e.g. [...;;]
    # f_sens(x::Vector{Float64}, θ::Vector{Float64})::Matrix{Float64} = f_sens_base(x, θ, 1)+f_sens_L1(x)

    # Sensitivity wrt to whichever individual parameter except L0, L1, L2, L3, all others are the same
    f_sens(x::Vector{Float64}, θ::Vector{Float64})::Matrix{Float64} = f_sens_base(x, θ, 1)+f_sens_other(x)

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

    # # Sensitivity wrt to all parameters
    # f_sens(x::Vector{Float64}, θ::Vector{Float64})::Matrix{Float64} = hcat(f_sens_base(x, θ, 1)+f_sens_L0(x), f_sens_base(x, θ, 2)+f_sens_L1(x), f_sens_base(x, θ, 3)+f_sens_L2(x), 
    #     f_sens_base(x, θ, 4)+f_sens_L3(x), f_sens_base(x, θ, 5)+f_sens_other(x), f_sens_base(x, θ, 6)+f_sens_other(x), f_sens_base(x, θ, 7)+f_sens_other(x),
    #     f_sens_base(x, θ, 8)+f_sens_other(x), f_sens_base(x, θ, 9)+f_sens_other(x), f_sens_base(x, θ, 10)+f_sens_other(x), f_sens_base(x, θ, 11)+f_sens_other(x),
    #     f_sens_base(x, θ, 12)+f_sens_other(x))

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
    # # Sensitivity wrt to all dynamical parameters
    # f_sens_baseline(x::Vector{Float64}, θ::Vector{Float64})::Matrix{Float64} = hcat(f_sens_base(x, θ, 1)+f_sens_L0(x), f_sens_base(x, θ, 2)+f_sens_L1(x), f_sens_base(x, θ, 3)+f_sens_L2(x), 
    # f_sens_base(x, θ, 4)+f_sens_L3(x), f_sens_base(x, θ, 5)+f_sens_other(x), f_sens_base(x, θ, 6)+f_sens_other(x), f_sens_base(x, θ, 7)+f_sens_other(x),
    # f_sens_base(x, θ, 8)+f_sens_other(x), f_sens_base(x, θ, 9)+f_sens_other(x), f_sens_base(x, θ, 10)+f_sens_other(x), f_sens_base(x, θ, 11)+f_sens_other(x),
    # f_sens_base(x, θ, 12)+f_sens_other(x))
    # Sensitivity wrt to whichever individual parameter except L0, L1, L2, L3, all others are the same
    f_sens_baseline(x::Vector{Float64}, θ::Vector{Float64})::Matrix{Float64} = f_sens_base(x, θ, 1)+f_sens_other(x)
    # # Sensitivity wrt whichever parameters I felt like while debugging (γ)
    # f_sens_baseline(x::Vector{Float64}, θ::Vector{Float64})::Matrix{Float64} = hcat(f_sens_base(x, θ, 1)+f_sens_other(x))

    dθ = length(free_dyn_pars_true)
    ny = length(f(ones(30), get_all_θs(free_dyn_pars_true)))
    minimizer = perform_ADAM#_deltaversion  

    (L0 = L0, L1 = L1, L2 = L2, L3 = L3, LC1 = LC1, LC2 = LC2, M1 = M1, M2 = M2, M3 = M3, J1 = J1, J2 = J2, γ = γ,
        σ = σ, get_all_θs = get_all_θs, free_dyn_pars_true = free_dyn_pars_true, par_bounds = par_bounds, ny = ny,
        model_nominal = model_nominal, model_sens = model_sens, f=f, f_sens=f_sens, f_sens_baseline=f_sens, dθ = dθ,
        minimizer=minimizer, init_learning_rate=init_learning_rate)
end