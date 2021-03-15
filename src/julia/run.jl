include("data-gen.jl")

const exp = "L_06_10_d"

mk_exp_dir(exp)

Y = calc_Y()
write_Y(exp, Y)

write_theta(exp)

Yb = calc_baseline_Y()
write_baseline_Y(exp, Yb)

Ym = calc_mean_Y()
write_mean_Y(exp, Ym)

write_meta_data(exp)
