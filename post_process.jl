include("estimation.jl")
using LaTeXStrings
using JLD

function plot_experiment(file)
  e = load(file)
  runs = e["runs"]

  θ = first(runs)["theta"]
  snr = first(runs)["snr"]
  name = first(map(x-> x["name"], filter(x -> x["name"] != "baseline", runs)))
  bl = vcat(map(x -> x["thetahat"],
                filter(x -> x["name"] == "baseline", runs))...)
  @info "baseline: $(bl)"

  res = vcat(map(x -> x["thetahat"],
                 filter(x -> x["name"] != "baseline", runs))...)
  @info "name: $(res)"

  p = plot(1:length(res),
           res,
           seriestype = :scatter,
           label = name,
           xlabel="run",
           ylabel=L"\hat{\theta}",
           legend=:bottomleft,
           title="SNR = $(snr)")

  plot!(p, bl, seriestype = :scatter, label = "baseline")
  plot!(p, [θ], seriestype=:hline, label=L"\theta")
end

function plot_baseline()
  e = load("baseline.jld")
  wscales = e["wscales"]
  θ = first(first(wscales))["theta"]
  n = length(first(wscales))
  x = map(xs -> vcat(map(x -> x["thetahat"], xs)...), wscales)
  s = map(xs -> first(map(x -> x["snr"], xs)), wscales)
  p = plot(s,
           map(mean, x),
           seriestype = :scatter,
           yerror = map(std, x),
           xaxis=:log,
           xlabel="SNR",
           label=L"\hat{\theta}",
           title="baseline over $(n) runs")

  plot!(p, [θ], seriestype=:hline, label=L"\theta")
end
