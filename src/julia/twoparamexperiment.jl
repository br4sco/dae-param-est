include("data-gen3.jl")

Y = calc_Y()
try
    write_Y("id2params10000", Y)
catch e
    @warn "Error while storing Y, make sure to do it manually"
end

reset_isws!(isws)

Ym = calc_mean_Y()

try
    write_mean_Y("id2params10000", Ym)
catch e
    @warn "Error while storing Ym, make sure to do it manually"
    println(e)
end
