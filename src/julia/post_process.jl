using CSV
using DataFrames
using Plots
using LaTeXStrings

include("helpers.jl")

M = 500
N = 500
id = 6

filename = "run_$(id)_$(M)_$(N)"

data = CSV.File(joinpath("data", "$(filename)_data.csv")) |> DataFrame
meta_data = CSV.File(joinpath("data", "$(filename)_meta_data.csv")) |> DataFrame

print(meta_data)

plot_costs(data.θ, data.cost, data.cost_baseline, meta_data.θ0)
