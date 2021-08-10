include("MIMO_experiment.jl")

try
    Y = calc_Y()
    perform_experiments(Y, vcat(θ0, w_scale))
catch e
    p = joinpath(data_dir, "error_msg.txt")
    file = open(p, "w")
    write(file, sprint(showerror, backtrace()))
    close(file)
end
