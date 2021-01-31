include("estimation.jl")
using LaTeXStrings
using JLD

function plot_signal_to_noise_ratio(ys, ws, wscale)
  plot(
    ys.^2 ./ ws.^2, yaxis=:log, xlabel="time (steps)", ylabel=L"y^2/w^2",
    title="wscale = $(wscale)")
end

function plot_yhat_over_theta(data, wscale)
  p = plot(
    xlabel=L"\theta", ylabel=L"\Sigma_k (\hat{y}_k - y_k)^2",
    title="wscale = $(wscale)")

  plot!(p, [1.], seriestype=:vline, label=L"\theta")
  for d in data
    plot!(p, d["thetas"], d["yhats"], label="M = $(d["M"])")
  end
end

function print_theta(data, wscale)
  print("wscale = $(wscale)\n")
  for d in data
    print("M = $(d["M"]), theta = $(d["theta"]), theta0 = $(d["theta0"]), thetahat = $(d["thetahat"])\n")
  end
end

p1 = problem1(0.02, 1)
plot_signal_to_noise_ratio(p1.ys, p1.ws, 0.02)
savefig("signal_to_noise_02.svg")

p2 = problem1(0.2, 1)
plot_signal_to_noise_ratio(p2.ys, p2.ws, 0.2)
savefig("signal_to_noise_2.svg")

d = load("data.jld")
d02 = filter(x -> x["wscale"] == 0.02, d["data"])
d2 = filter(x -> x["wscale"] == 0.2, d["data"])

plot_yhat_over_theta(d02, 0.02)
savefig("cost_over_theta_02.svg")

plot_yhat_over_theta(d2, 0.2)
savefig("cost_over_theta_2.svg")

print_theta(d02, 0.02)
print_theta(d2, 0.2)
