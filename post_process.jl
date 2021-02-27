using CSV
using DataFrames
using Plots
using LaTeXStrings

M = 1000
N = 100

filename = "run_$(M)_$(N)"

data = CSV.File("$(filename)_data.csv") |> DataFrame
meta_data = CSV.File("$(filename)_meta_data.csv") |> DataFrame

function plot_costs()

  pl = plot(xlabel=L"\theta", ylabel=L"\texttt{cost}(\theta)")
  plot!(pl, data.θ, data.cost_baseline, label="baseline", linecolor=:red)
  plot!(pl, data.θ, data.cost,
        label="our attempt at M=$(meta_data.M), N=$(meta_data.N)",
        linecolor=:black)

  vline!(pl, [meta_data.θ0], linecolor=:gray, label="θ0")
  pl
end

plot_costs()
