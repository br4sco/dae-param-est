include("data-gen.jl")

Ym = calc_mean_Y()
try
    write_mean_Y("paper_experiment_50000", Ym)
catch e
    @warn "Error while storing Ym, make sure to do it manually"
    println(e)
end
