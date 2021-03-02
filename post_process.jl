using CSV
using DataFrames
using Plots
using LaTeXStrings

M = 5000
N = 100

filename = "run_5_$(M)_$(N)"

data = CSV.File("$(filename)_data.csv") |> DataFrame
meta_data = CSV.File("$(filename)_meta_data.csv") |> DataFrame

function plot_costs()

  pl = plot(
    xlabel=L"\theta",
    ylabel=L"\texttt{cost}(\theta)",
    title = "u_scale = $(meta_data.u_scale), w_scale = $(meta_data.w_scale), N = $(meta_data.N), M = $(meta_data.M)"
  )

  plot!(pl, data.θ, data.cost_baseline, label="baseline", linecolor=:red)
  plot!(pl, data.θ, data.cost, label="our attempt", linecolor=:black)

  vline!(pl, [meta_data.θ0], linecolor=:gray, lines = :dot, label="θ0")
  pl
end

print(meta_data)
plot_costs()
