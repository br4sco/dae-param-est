function plot_M(f, ts, ms)
  p = plot()
  for m = ms
    plot!(p, f(m), ts, linecolor=:gray, linealpha=0.5, label="")
  end
  p
end

function plot_costs(θs, costs, costs_baseline, θ0)
  pl = plot(
    xlabel=L"\theta",
    ylabel=L"\texttt{cost}(\theta)")

  plot!(pl, θs, costs_baseline, label="baseline", linecolor=:red)
  n_min_b = argmin(costs_baseline)
  plot!(pl, [θs[n_min_b]], [costs_baseline[n_min_b]],
    color=:red, seriestype = :scatter, label = "", alpha = 0.5)

  plot!(pl, θs, costs, label="our attempt", linecolor=:black)
  n_min = argmin(costs)
  plot!(pl, [θs[n_min]], [costs[n_min]],
    color=:black, seriestype = :scatter, label = "", alpha = 0.5)

  vline!(pl, [θ0], linecolor=:gray, lines = :dot, label="θ0")
  pl
end

function plot_outputs(y, yhat, yhat_baseline, N, Ts, N_trans)
  T = N * Ts
  ts = 0.0:Ts:T

  pl = plot(xlabel="time [s]")

  plot!(pl, ts, [yhat y], fillrange=[y yhat],
    fillalpha=0.2, c=:yellow, label="")

  plot!(pl, ts, [yhat_baseline y], fillrange=[y yhat_baseline],
    fillalpha=0.2, c=:blue, label="")

  plot!(pl, ts, y, linecolor = :red, linewidth = 1, label = "true trajectory")
  plot!(pl, ts, yhat, linecolor = :green, linewidth = 1, label = "mean")
  plot!(pl, ts, yhat_baseline, linecolor = :blue, linewidth = 1, label = "baseline")
  vline!(pl, [N_trans * Ts], linecolor=:gray, lines = :dot, label="")
end
