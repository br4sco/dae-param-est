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
    ylabel=L"\texttt{cost}(\theta)",
    title = "u_scale = $(meta_data.u_scale), w_scale = $(meta_data.w_scale), N = $(meta_data.N), M = $(meta_data.M)"
  )

  plot!(pl, θs, costs_baseline, label="baseline", linecolor=:red)
  plot!(pl, θs, costs, label="our attempt", linecolor=:black)

  vline!(pl, [θ0], linecolor=:gray, lines = :dot, label="θ0")
  pl
end

function plot_outputs(y, yhat, yhat_baseline)
  pl = plot(xlabel="time [s]",
            title = "u_scale = $(u_scale), w_scale = $(w_scale), N = $(N)")

  plot!(pl, [yhat y], fillrange=[y yhat],
    fillalpha=0.2, c=:yellow, label="")

  plot!(pl, [yhat_baseline y], fillrange=[y yhat_baseline],
    fillalpha=0.2, c=:blue, label="")

  plot!(pl, y, linecolor = :red, linewidth = 1, label = "true trajectory")
  plot!(pl, yhat, linecolor = :green, linewidth = 1, label = "mean")
  plot!(pl, yhat_baseline, linecolor = :blue, linewidth = 1, label = "baseline")
end
