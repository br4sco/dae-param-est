include("data-gen.jl")

Ym = calc_mean_Y()
try
    write_mean_Y("idtsheta100000", Ym)
catch e
    @warn "Error while storing Ym, make sure to do it manually"
    println(e)
end
