using CSV
using DataFrames
using Plots
using LaTeXStrings
using Glob

include("helpers.jl")

function post_process(dir, id)
  cdf = glob(joinpath("data", dir, "run_$(id)*cost_data.csv"))
  odf = glob(joinpath("data", dir, "run_$(id)*output_data.csv"))
  mdf = glob(joinpath("data", dir, "run_$(id)*meta_data.csv"))

  data = CSV.File(cdf[1]) |> DataFrame
  data_extra = CSV.File(odf[1]) |> DataFrame
  meta_data = CSV.File(mdf[1]) |> DataFrame

  print(meta_data)

  pl1 = plot_costs(data.θ, data.cost, data.cost_baseline, meta_data.θ0)
  pl2 = plot_outputs(data_extra.y, data_extra.yhat, data_extra.yhat_baseline,
                     meta_data.N[1], meta_data.Ts[1], meta_data.N_trans[1])

  plot(pl1, pl2, layout = (2, 1), legend = true)
end

function post_process_dir(dir)
  mdp = joinpath("data", dir, "*meta_data.csv")
  dp = joinpath("data", dir, "*cost_data.csv")

  mdsp = glob(mdp)
  dsp = glob(dp)

  mds = map(f -> CSV.File(f) |> DataFrame, mdsp)
  ds = map(f -> CSV.File(f) |> DataFrame, dsp)

  n = length(ds)

  print(mds[1])

  θ0 = mds[1].θ0
  θs = ds[1].θ
  nθ = length(θs)

  cost_baseline = zeros(nθ, n)
  cost = zeros(nθ, n)

  for (i, d) in enumerate(ds)
    cost_baseline[:, i] .+= d.cost_baseline
    cost[:, i] .+= d.cost
  end

  mean_cost = mean(cost, dims = 2)

  pl = plot()
  pl = plot!(pl, θs, mean_cost, ribbon = var(cost, dims = 2),
            fillalpha = .5, label="our attempt", color=:black)

  pl = plot!(pl, θs, mean(cost_baseline, dims = 2), ribbon = var(cost_baseline, dims = 2),
             fillalpha = .5, label="baseline", color=:red)

  vline!(pl, [θ0], linecolor=:green, lines = :dot, label="θ0")
end

function collect_dir_data(dir)
  mdp = joinpath("data", dir, "*meta_data.csv")
  dp = joinpath("data", dir, "*cost_data.csv")

  mdsp = glob(mdp)
  dsp = glob(dp)

  mds = map(f -> CSV.File(f) |> DataFrame, mdsp)
  ds = map(f -> CSV.File(f) |> DataFrame, dsp)

  n = length(ds)

  print(mds[1])

  θ0 = mds[1].θ0
  θs = ds[1].θ
  nθ = length(θs)

  cost_baseline = zeros(nθ, n)
  cost = zeros(nθ, n)

  for (i, d) in enumerate(ds)
    cost_baseline[:, i] .+= d.cost_baseline
    cost[:, i] .+= d.cost
  end

  θs, cost, cost_baseline
end

function baseline_mean_min()
  Y = calc_m(m -> calc_y_true(m), ms_true)
  ys_bl = map(calc_yhat_bl, θs)
  is =
    mapslices(argmin,
              vcat(map(yhat -> mean((Y .- yhat).^2, dims = 1), ys_bl)...),
              dims = 1)
  θs[is]
end

err = (baseline_mean_min() - θ0)^2
print(mean(err))
print(var(err))
