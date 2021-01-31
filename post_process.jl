include("estimation.jl")

function plot_signal_to_noise_ratio(ys, ws, wscale)
  plot(ys.^2 ./ ws.^2, yaxis=:log, xlabel="time (steps)", ylabel=L"y^2/w^2", title="wscale = $(wscale)" )
end

p1 = problem1(0.02, 1)
plot_signal_to_noise_ratio(p1.ys, p1.ws, 0.02)
savefig("signal_to_noise_02")

p2 = problem1(0.2, 1)
plot_signal_to_noise_ratio(p2.ys, p2.ws, 0.2)
savefig("signal_to_noise_2")

d = load("data.jld")
d2 = filter(x -> x["wscale"] == 0.2, d["data"])
d02 = filter(x -> x["wscale"] == 0.02, d["data"])

p = plot(xlabel=L"\theta", ylabel=L"\Sigma_k (\hat{y}_k - y_k)^2")
for d in d2
  plot!(p, d["yhats"], label = "M = $(d["M"])")
end
# savefig("cost_fun")
display(p)
