function plot_M(f, ts, ms)
  p = plot()
  for m = ms
    plot!(p, f(m), ts, linecolor=:gray, linealpha=0.5, label="")
  end
  p
end
