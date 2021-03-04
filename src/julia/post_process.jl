using CSV
using DataFrames
using Plots
using LaTeXStrings

include("helpers.jl")

M = 1000
N = 2000
id = 8

function post_process(M, N, id)
  filename = "run_$(id)_$(M)_$(N)"

  data = CSV.File(joinpath("data", "$(filename)_data.csv")) |> DataFrame
  data_extra = CSV.File(joinpath("data", "$(filename)_data_extra.csv")) |> DataFrame
  meta_data = CSV.File(joinpath("data", "$(filename)_meta_data.csv")) |> DataFrame

  print(meta_data)

  pl1 = plot_costs(data.θ, data.cost, data.cost_baseline, meta_data.θ0)
  pl2 = plot_outputs(data_extra.y, data_extra.yhat, data_extra.yhat_baseline)
  plot(pl1, pl2, layout = (2, 1), legend = true)
end

post_process(M, N, id)
