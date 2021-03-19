using DelimitedFiles
using Plots
using CSV
using DataFrames
using StatsPlots
using Statistics
using LaTeXStrings

const N_trans = 500

const data_dir = "data/experiemnts"
# const expid = "L_06_25_alsvin_high"
const expid = "candidate_4"

const atol = 1e-7
const rtol = 1e-4

exp_path(id) = joinpath(data_dir, id)

function read_Y(expid)
  p = joinpath(exp_path(expid), "Y.csv")
  readdlm(p, ',')
end

function read_theta(expid)
  p = joinpath(exp_path(expid), "theta.csv")
  readdlm(p, ',')
end

function read_baseline_Y(expid)
  p = joinpath(exp_path(expid), "Yb.csv")
  readdlm(p, ',')
end

function read_mean_Y(expid)
  p = joinpath(exp_path(expid), "Ym.csv")
  readdlm(p, ',')
end

function read_meta_data(expid)
  p = joinpath(exp_path(expid), "meta_data.csv")
  CSV.File(p) |> DataFrame
end

errY(Y) = map(x -> max(x, atol), rtol * Y)
cost(y::Array{Float64, 1}, yhat::Array{Float64, 1}) = mean((y - yhat).^2)
cost_err(y::Array{Float64, 1}, yhat::Array{Float64, 1}, Δyhat::Array{Float64, 1}) =
  2mean(map(abs, (y - yhat) .* Δyhat))

const md = read_meta_data(expid)
const Y = read_Y(expid)
const Yb = read_baseline_Y(expid)
const Ym = read_mean_Y(expid)
# const Ym = zeros(size(Yb))

const ΔYb = errY(Yb)
const ΔYm = errY(Ym)

const N = size(Y, 1)
const Ts = md.Ts[1]
const ts = 0:Ts:(Ts * (N - 1))

const θs = read_theta(expid)
const nθ = length(θs)
const i0 = Int(ceil(nθ / 2))
const θ0 = θs[i0]

plot_y_at_theta0(m::Int, start = 1, stop = N) =
  plot(ts[start:stop],
       [Y[start:stop, m], Yb[start:stop, i0], Ym[start:stop, i0]],
       labels = ["true" "baseline" "mean"])

calc_costs(Yhat::Array{Float64, 2}, N::Int) =
  mapslices(y ->
            mapslices(yhat -> cost(y[N_trans:N], yhat[N_trans:N]),
                      Yhat,
                      dims = 1)[:],
            Y,
            dims = 1)

function calc_costs_err(Yhat::Array{Float64, 2}, N::Int)
  ΔYhat = errY(Yhat)
  nerr = size(Yhat, 2)
  E = size(Y, 2)
  err = ones(nerr, E) * Inf
  for i = 1:nerr
    for e = 1:E
      err[i, e] =
        cost_err(Y[N_trans:N, e], Yhat[N_trans:N, i], ΔYhat[N_trans:N, i])
    end
  end
  err
end


const costsb = calc_costs(Yb, N)
const Δcostsb = calc_costs_err(Yb, N)
const costsm = calc_costs(Ym, N)
const Δcostsm = calc_costs_err(Ym, N)

function plot_costm(m::Int)

  p = plot(θs, costsm[:, m],
           ribbon = Δcostsm[:, m],
           label = "realization $(m)",
           xlabel = L"\theta",
           ylabel = L"cost(\theta)",
           title = "mean")
  vline!(p, [θ0], linestyle = :dash, label = "θ0")
end

function plot_costb(m::Int)
  p = plot(θs, costsb[:, m],
           ribbon = Δcostsb[:, m],
           label = "realization $(m)",
           xlabel = L"\theta",
           ylabel = L"cost(\theta)",
           title = "baseline")
  vline!(p, [θ0], linestyle = :dash, label = "θ0")
end

function calc_theta_hats(Y::Array{Float64, 2}, N::Int)
  cost = calc_costs(Y, N)
  argmincost = mapslices(argmin, cost, dims = 1)[:]
  θs[argmincost]
end

calc_θhatsb(N) = calc_theta_hats(Yb, N)
calc_Δθhatsb(N) = map(abs, calc_theta_hats(Yb + ΔYb, N) - calc_θhatsb(N))
calc_θhatsm(N) = calc_theta_hats(Ym, N)
calc_Δθhatsm(N) = map(abs, calc_theta_hats(Ym + ΔYm, N) - calc_θhatsm(N))

function write_theta_hats(Ns::Array{Int, 1})
  θhatbs = hcat(map(N -> calc_theta_hats(Yb, N), Ns)...)
  θhatms = hcat(map(N -> calc_theta_hats(Ym, N), Ns)...)
  writedlm(joinpath(exp_path(expid), "theta_hat_baseline.csv"), θhatbs, ',')
  writedlm(joinpath(exp_path(expid), "theta_hat_mean.csv"), θhatms, ',')
end

function plot_thetahat_density_b!(p, N)
  θhat = calc_theta_hats(Yb, N)
  # density!(p, θhat, label = "baseline", linecolor = :red)
  histogram!(p, θhat, label = "baseline", color = :red, alpha = 0.5)
  vline!(p, [mean(θhat)], label = "", linestyle = :dash, linecolor = :red)
end

function plot_thetahat_density_m!(p, N)
  θhat = calc_theta_hats(Ym, N)
  # density!(p, θhatm, label = "mean", linecolor = :black)
  histogram!(p, θhat, label = "mean", color = :black, alpha = 0.5)
  vline!(p, [mean(θhat)], label = "", linestyle = :dash, linecolor = :black)
end

function plot_thetahat_density(N)
  p = plot(xlabel = L"\hat{\theta}",
           ylabel = L"density[\hat{\theta}]",
           title = "N = $(N)")

  plot_thetahat_density_b!(p, N)
  plot_thetahat_density_m!(p, N)
  vline!(p, [θ0], linestyle = :dash, label = "θ0")
end

function plot_thetahat_boxs(Ns)
  θhatbs = map(N -> calc_theta_hats(Yb, N), Ns)
  θhatms = map(N -> calc_theta_hats(Ym, N), Ns)
  θhats = hcat(θhatbs..., θhatms...)
  labels = reshape([map(N -> "bl $(N)", Ns); map(N -> "m $(N)", Ns)], (1, :))
  idxs = 1:(2length(Ns))
  p = boxplot(θhats,
              xticks = (idxs, labels),
              label = "",
              ylabel = L"\hat{\theta}",
              notch = false)

  hline!(p, [θ0], label = L"\theta_0", linestyle = :dot, linecolor = :gray)
end

println(md)
