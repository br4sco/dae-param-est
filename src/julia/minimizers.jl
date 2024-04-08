function perform_SGD_adam(
    get_grad_estimate::Function,
    pars0::Vector{Float64},                        # Initial guess for parameters
    bounds::Matrix{Float64},                      # Parameter bounds
    learning_rate::Function=learning_rate_vec;
    tol::Float64=1e-6,
    maxiters=200,
    verbose=false,
    betas::Vector{Float64} = [0.9, 0.999])   # betas are the decay parameters of moment estimates
    # betas::Vector{Float64} = [0.5, 0.999])   # betas are the decay parameters of moment estimates

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
    pars0::Vector{Float64},                        # Initial guess for parameters
    bounds::Matrix{Float64},                      # Parameter bounds
    learning_rate::Function=learning_rate_vec_red;
    tol::Float64=1e-6,
    maxiters=200,
    verbose=false,
    betas::Vector{Float64} = [0.9, 0.999],
    # λ = 0.5)   # betas are the decay parameters of moment estimates   # NOTE: I used to use this, then I started deubbing adjoint on 17/10/2023
    λ = 1-1e-4)   # betas are the decay parameters of moment estimates
    # betas::Vector{Float64} = [0.5, 0.999])   # betas are the decay parameters of moment estimates

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
        # New way, of repeating simulation on convergence failure
        grad_est = zeros(size(pars0))
        succeeded = false
        maxtries = 10
        ind = 1
        # while !succeeded && ind <= maxtries
        #     try
        #         grad_est = get_grad_estimate(pars, M_rate(t))
        #         succeeded = true
        #     catch ex
        #         println("Attempt $ind failed with error:")
        #         println(ex)
        #         ind += 1
        #     end
        # end
        # if ind == 11
        #     throw(ErrorException("Failed all $maxtries attempts to obtain gradient estimate for parameter values $(pars)"))
        # end
        # Original way of doing it
        grad_est = get_grad_estimate(pars, M_rate(t))   # TODO: THIS USES GLOBAL FUNCTION M_rate(t), ISN'T THAT A WEIRD CHOICE?

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

# VERSION WITH PARAMETER-VARYING BOUNDS
# Also adds time-varying beta_1, dependent on the new hyper-parameter λ
function perform_SGD_adam_new_deltaversion(
    get_grad_estimate::Function,
    pars0::Vector{Float64},                        # Initial guess for parameters
    bounds::Matrix{Float64},                      # Parameter bounds as function of current parameters
    learning_rate::Function=learning_rate_vec_red;
    tol::Float64=1e-6,
    maxiters=200,
    verbose=false,
    betas::Vector{Float64} = [0.9, 0.999],
    # λ = 0.5)   # betas are the decay parameters of moment estimates   # NOTE: I used to use this, then I started deubbing adjoint on 17/10/2023
    λ = 1-1e-4)   # betas are the decay parameters of moment estimates
    # betas::Vector{Float64} = [0.5, 0.999])   # betas are the decay parameters of moment estimates

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
        # New way, of repeating simulation on convergence failure
        grad_est = zeros(size(pars0))
        succeeded = false
        maxtries = 10
        ind = 1
        # while !succeeded && ind <= maxtries
        #     try
        #         grad_est = get_grad_estimate(pars, M_rate(t))
        #         succeeded = true
        #     catch ex
        #         println("Attempt $ind failed with error:")
        #         println(ex)
        #         ind += 1
        #     end
        # end
        # if ind == 11
        #     throw(ErrorException("Failed all $maxtries attempts to obtain gradient estimate for parameter values $(pars)"))
        # end
        # Original way of doing it
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
        bounds[4,:] = [max(pars[1]-pars[3], 0.01), pars[1]+pars[3]] # Constrains max(0.01, L0-L2) <= L3 <= L0+L2
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
function project_on_bounds!(vec::Vector{Float64}, bounds::Matrix{Float64})
    low_inds = findall(vec .< bounds[:,1])
    high_inds = findall(vec .> bounds[:,2])
    vec[low_inds] = bounds[low_inds, 1]
    vec[high_inds] = bounds[high_inds, 2]
    nothing
end