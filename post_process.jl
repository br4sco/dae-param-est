include("estimation.jl")
using LaTeXStrings
using JLD

d = load("data.jld")
θs = d["thetas"]
costs0 = d["costs0"]

function plot_cost_over_theta(data, wscale, snr)
  p = plot(
    xlabel=L"\theta", ylabel=L"\Sigma_k (\hat{y}_k - y_k)^2",
    title="wscale = $(wscale), snr = $(snr)")

  plot!(p, [1.], seriestype=:vline, label=L"\theta")
  plot!(p, θs, costs0, label=L"w(t)=0")
  for d in data
    plot!(p, θs, d["costs"], label="M = $(d["M"])")
  end
end

function plot_cost_cost0_diff(data, wscale, snr)
  p = plot(
    xlabel=L"\theta", ylabel=L"|cost_M-cost0|",
    title="wscale = $(wscale), snr = $(snr)", yaxis=:log)

  plot!(p, [1.], seriestype=:vline, label=L"\theta")
  for d in data
    plot!(p, θs, abs.(d["costs"] - costs0), label="M = $(d["M"])")
  end
end

function print_theta(data, wscale)
  print("wscale = $(wscale)\n")
  for d in data
    print("M = $(d["M"]), theta = $(d["theta"]), theta0 = $(d["theta0"]), thetahat = $(d["thetahat"])\n")
  end
end

p1 = problem1(0.02, 1)
snr1 = sum(p1.ys.^2) / sum(p1.ws.^2)

p2 = problem1(0.2, 1)
snr2 = sum(p2.ys.^2) / sum(p2.ws.^2)


d02 = filter(x -> x["wscale"] == 0.02, d["runs"])
d2 = filter(x -> x["wscale"] == 0.2, d["runs"])

plot_cost_over_theta(d02, 0.02, snr1)
savefig("cost_over_theta_02.svg")

plot_cost_over_theta(d2, 0.2, snr2)
savefig("cost_over_theta_2.svg")

plot_cost_cost0_diff(d02, 0.02, snr1)
savefig("cost_diff_over_theta_02.svg")

plot_cost_cost0_diff(d2, 0.2, snr2)
savefig("cost_diff_over_theta_2.svg")

print_theta(d02, 0.02)
print_theta(d2, 0.2)
