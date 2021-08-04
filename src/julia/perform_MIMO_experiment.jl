include("MIMO_experiment")

Y = calc_Y()
perform_experiments(Y, vcat(θ0, w_scale))
