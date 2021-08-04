include("data-gen4.jl")

exp_id = "multiparam_experiment"

# NOTE: THIS FILE IS DEPRACATED, WILL PROBABLY BE DELETED IN THE FUTURE.
# THE MOST RECENT FILE FOR MULTIPARAMETER EXPERIMENTS, AT THE TIME OF WRITING
# THIS COMMENTS, IS MIMO_experiment.jl

#!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
# 8  THREADS !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
#!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!

Y = calc_Y()

diff_θ = 0.5*ones(size(θ0))
diff_η = 0.5*ones(size(η0))

fit, trace = get_fit(Y, θ0+diff_θ, η0+diff_η)

try
    write_custom(exp_id, "opt_pars", coef(fit))
catch e
    @warn "Error while storing optimal parameters, make sure to do it manually"
    println(e)
end

try
    write_custom(exp_id, "true_pars", [θ0; η0])
catch e
    @warn "Error while storing true parameters, make sure to do it manually"
    println(e)
end

try
    write_custom(exp_id, "init_pars", [θ0+diff_θ; η0+diff_η])
catch e
    @warn "Error while storing initial parameters, make sure to do it manually"
    println(e)
end

dx_trace = zeros(length(θ0)+length(η0), length(trace)-1)
g_norm_trace = zeros(length(trace)-1)   # Infinity norm of gradient
for i=1:length(trace)-1
    dx_trace[:,i]   = trace[i+1].metadata["dx"]
    g_norm_trace[i] = trace[i+1].metadata["g(x)"] # Infinity norm of gradient
end

try
    write_custom(exp_id, "dx_trace", dx_trace)
catch e
    @warn "Error while storing dx trace, make sure to do it manually"
    println(e)
end

try
    write_custom(exp_id, "gnorm_trace", g_norm_trace)
catch e
    @warn "Error while storing g_norm trace, make sure to do it manually"
    println(e)
end
