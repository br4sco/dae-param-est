include("data-gen.jl")

Ym = calc_mean_Y()
write_mean_Y("idtheta100000", Ym)
