

function plot_costs(cs_baseline, cs)
  pl = plot(xlabel=L"\theta", ylabel=L"\texttt{cost}(\theta)")
  plot!(pl, θs, cs_baseline, label="baseline", linecolor=:red)
  plot!(pl, θs, cs, label="our attempt at M=$(M), N=$(N)", linecolor=:black)
  vline!(pl, [θ0], linecolor=:gray, label="θ0")
  pl
end

plot_costs(cs_baseline, cs)
