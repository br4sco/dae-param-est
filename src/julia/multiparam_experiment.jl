include("data-gen4.jl")

exp_id = "multiparam_experiment"

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
for i=1:length(trace)-1
    dx_trace[:,i] = trace[i+1].metadata["dx"]
end

try
    write_custom(exp_id, "dx_trace", dx_trace)
catch e
    @warn "Error while storing dx trace, make sure to do it manually"
    println(e)
end
