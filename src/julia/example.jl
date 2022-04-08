using LaTeXStrings, Plots, StatsPlots
include("run_experiment.jl")

function get_estimates(expid::String)
    # Computes optimal parameters for baseline (output error) method
    # and the proposed method, using data from the experiment given by expid,
    # initial parameter values [0.5, 4.25, 4.25] and a transient of 500 steps
    opt_pars_oe, opt_pars_proposed, avg_pars_proposed,
        trace_proposed, trace_gradient, durations =
        get_estimates(expid, [0.5, 4.25, 4.25], 500);

    m_base = opt_pars_oe[1,:]
    L_base = opt_pars_oe[2,:]
    k_base = opt_pars_oe[3,:]
    m_prop = avg_pars_proposed[1,:]
    L_prop = avg_pars_proposed[2,:]
    k_prop = avg_pars_proposed[3,:]
    return (m_base, L_base, k_base), (m_prop, L_prop, k_prop)
end

# TODO: CHANGE THIS COMMENT!!!

"""
    `thetahat_boxplots(outputs, Ns, N_trans)`
Produces two boxplots, comparing the output error method to the proposed method
for a single parameter. oe_pars and prop_pars should be arrays where
oe_pars[iₑ,jₙ] and prop_pars[iₑ,jₙ] contain the iₑ:th estimate of the considered
parameter for the jₙ:th value of N. By default, iₑ=1,...,100. Ns should be an
array of all the values of N for which parameter estimates are provided
"""
function thetahat_boxplots(oe_pars::AbstractArray{Float64}, prop_pars::AbstractArray{Float64}, Ns::Array{Int64,1})
  # θhatbs =
  #   map(N -> calc_theta_hats(outputs.θs, outputs.Y, outputs.Yb, N_trans, N), Ns)
  # θhatms =
  #   map(N -> calc_theta_hats(outputs.θs, outputs.Y, outputs.Ym, N_trans, N), Ns)
  # θhats = hcat(θhatbs..., θhatms...)
  # labels = reshape([map(N -> "bl $(N)", Ns); map(N -> "m $(N)", Ns)], (1, :))
  # idxs = 1:(2length(Ns))
  θhats = hcat(oe_pars, prop_pars)
  idxs = 1:2size(oe_pars,2)
  labels = reshape([map(N -> "oe $(N)", Ns); map(N -> "prop $(N)", Ns)], (1, :))
  p = boxplot(
    θhats,
    xticks = (idxs, labels),
    label = "",
    ylabel = L"\hat{\theta}",
    notch = false,
  )
end

# TODO: Function to get comparison between grid search and proposed method?
