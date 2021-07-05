include("simulation.jl")
include("noise_generation.jl")
include("noise_interpolation.jl")

seed = 1234
Random.seed!(seed)

const data_dir = joinpath("data", "experiments")
exp_path(id) = joinpath(data_dir, id)
const experiment_id = "comparisons"
const Ts = 0.1
const δ1 = 0.05
const δ2 = 5.0
const factor = Int(δ2/δ1)
const T = 100
const Nw1 = ceil(Int, T/δ1)
const Nw2 = ceil(Int, T/δ2)
const Nw_extra = 200
const N = ceil(Int, T/Ts)
const Q = 2000
const W = 100
const M = 1000#500

const nx = 2
const A = [0.0 1.0; -4^2 -0.8]
const B = reshape([0.0 1.0], (2,1))
const C = [1.0 0.0]

const x0 = zeros(nx)
const dmdl1 = discretize_ct_noise_model(A, B, C, δ1, x0)
const dmdl2 = discretize_ct_noise_model(A, B, C, δ2, x0)

const Zu = [randn(Nw1+Nw_extra, nx)]
const Zm = [randn(Nw1+Nw_extra, nx) for m=1:M]
# const Z2 = [Z1[1][1:factor:end, :]]
# const Z2 = [randn(Nw2+Nw_extra, nx)]

mangle_XW(XW::Array{Array{Float64, 1}, 2}) =
  hcat([vcat(XW[:, m]...) for m = 1:size(XW,2)]...)

const XWup  = simulate_noise_process(dmdl1, Zu)
const XWd1p = simulate_noise_process(dmdl1, Zm)
# const XWd2p = simulate_noise_process(dmdl2, Z2)
const XWd2p = XWd1p[1:factor:end, :]
const XWu = mangle_XW(XWup)
const XWd1 = mangle_XW(XWd1p)
const XWd2 = mangle_XW(XWd2p)

function interpx(xl::AbstractArray{Float64, 1},
                 xu::AbstractArray{Float64, 1},
                 t::Float64,
                 δ::Float64,
                 n::Int)

  xl .+ (t - (n - 1) * δ) .* (xu .- xl) ./ δ
end

function mk_noise_interp1(A::Array{Float64, 2},
                             B::Array{Float64, 2},
                             C::Array{Float64, 2},
                             XW::Array{Float64, 2},
                             m::Int)

  let
    nx = size(A, 1)
    function w(t::Float64)
      n = Int(floor(t / δ1)) + 1

      k = (n - 1) * nx + 1

      # xl = view(XW, k:(k + nx - 1), m)
      # xu = view(XW, (k + nx):(k + 2nx - 1), m)

      xl = XW[k:(k + nx - 1), m]
      xu = XW[(k + nx):(k + 2nx - 1), m]

      first(C * interpx(xl, xu, t, δ1, n))
    end
  end
end

function mk_noise_interp2(A::Array{Float64, 2},
                             B::Array{Float64, 2},
                             C::Array{Float64, 2},
                             XW::Array{Float64, 2},
                             m::Int)

  let
    nx = size(A, 1)
    function w(t::Float64)
      n = Int(floor(t / δ2)) + 1

      k = (n - 1) * nx + 1

      # xl = view(XW, k:(k + nx - 1), m)
      # xu = view(XW, (k + nx):(k + 2nx - 1), m)

      xl = XW[k:(k + nx - 1), m]
      xu = XW[(k + nx):(k + 2nx - 1), m]

      first(C * interpx(xl, xu, t, δ2, n))
    end
  end
end

function mk_newer_noise_interp1(A::Array{Float64, 2},
                               B::Array{Float64, 2},
                               C::Array{Float64, 2},
                               isw::InterSampleWindow,
                               m::Int)

   let
       function w(t::Float64)
           xw_temp = noise_inter(t, δ1, A, B, view(XWd1p, :, m), isw)
           return first(C*xw_temp)
       end
   end
end

function mk_noise_interp2(A::Array{Float64, 2},
                             B::Array{Float64, 2},
                             C::Array{Float64, 2},
                             XW::Array{Float64, 2},
                             m::Int)

  let
    nx = size(A, 1)
    function w(t::Float64)
      n = Int(floor(t / δ2)) + 1

      k = (n - 1) * nx + 1

      # xl = view(XW, k:(k + nx - 1), m)
      # xu = view(XW, (k + nx):(k + 2nx - 1), m)

      xl = XW[k:(k + nx - 1), m]
      xu = XW[(k + nx):(k + 2nx - 1), m]
      first(C * interpx(xl, xu, t, δ2, n))
    end
  end
end

function mk_newer_noise_interp2(A::Array{Float64, 2},
                               B::Array{Float64, 2},
                               C::Array{Float64, 2},
                               isw::InterSampleWindow,
                               m::Int)

   let
       function w(t::Float64)
           xw_temp = noise_inter(t, δ2, A, B, view(XWd2p, :, m), isw)
           return first(C*xw_temp)
       end
   end
end

isws = [initialize_isw(Q, W, nx, true) for m=1:M]

wmd1(m::Int) = mk_noise_interp1(A, B, w_scale.*C, XWd1, m)
wmd2(m::Int) = mk_noise_interp2(A, B, w_scale.*C, XWd2, m)
wmn1(m::Int) = mk_newer_noise_interp1(A, B, w_scale.*C, isws[m], m)
wmn2(m::Int) = mk_newer_noise_interp2(A, B, w_scale.*C, isws[m], m)
u(t::Float64) = mk_noise_interp1(A, B, C, XWu, 1)(t)

# === MODEL (AND DATA) PARAMETERS ===
const σ = 0.002                 # observation noise variance
const u_scale = 0.2             # input scale
# const u_scale = 10.0            # input scale larger
const u_bias = 0.0              # input bias
const w_scale = 0.6             # noise scale
# const w_scale = 5.0             # noise scale larger

const m = 0.3                   # [kg]
const L = 6.25                  # [m], gives period T = 5s (T ≅ 2√L) not
                                # accounting for friction.
const g = 9.81                  # [m/s^2]
const k = 0.05                  # [1/s^2]

const φ0 = 0. / 8              # Initial angle of pendulum from negative y-axis

# === OUTPUT FUNCTIONS ===
f(x::Array{Float64, 1}) = x[7]            # applied on the state at each step
h(sol) = apply_outputfun(f, sol)          # for our model
h_baseline(sol) = apply_outputfun(f, sol) # for the baseline method

# === MODEL REALIZATION AND SIMULATION ===
h(sol) = apply_outputfun(f, sol)          # for our model
const θ0 = L                    # true value of θ
# mk_θs(θ::Float64) = [m, L, g, θ]
realize_model(w::Function, N::Int) =
  problem(pendulum(φ0, t -> u_scale * u(t) + u_bias, w, [m, L, g, k]), N, Ts)

  # === SOLVER PARAMETERS ===
  const abstol = 1e-8
  const reltol = 1e-5
  const maxiters = Int64(1e8)

solvew(w::Function, N::Int; kwargs...) =
    solve(realize_model(w, N),
          saveat=0:Ts:(N*Ts),
          abstol = abstol,
          reltol = reltol,
          maxiters = maxiters;
          kwargs...)

calc_y_2e(m::Int) = solvew(t -> wmn2(m)(t), N) |> h
calc_y_2l(m::Int) = solvew(t -> wmd2(m)(t), N) |> h

function get_times_rng_2(num::Int)
    # num = number of samples per interval
    times_rng = zeros(num*Nw2, M)
    dts = [[δ2*rand() for i=1:num] for i=1:Nw2, j=1:M]
    times_temp = 0.0:δ2:Nw2*δ2
    for i=1:Nw2
        for j=1:M
            times_rng[(i-1)*num+1:i*num, j] = dts[i,j] .+ times_temp[i]
            # for k=1:samples_per_interval
            #     times_rng[k+(i-1)*samples_per_interval, j] = times_temp[i]+dts[i,j][k]
            # end
        end
    end
    return times_rng
end

samples_per_interval = 10
times_uni_2 = 0.0:δ2/samples_per_interval:Nw2*δ2
times_rng_2 = get_times_rng_2(samples_per_interval)
get_w_2e_uni(m::Int) = [wmn2(m)(t) for t=times_uni_2]
get_w_2l_uni(m::Int) = [wmd2(m)(t) for t=times_uni_2]
get_w_2e_rng(m::Int) = [wmn2(m)(t) for t=times_rng_2[:,m]]
get_w_2l_rng(m::Int) = [wmd2(m)(t) for t=times_rng_2[:,m]]

function get_ws_2le(M::Int=M, use_rng::Bool=false, use_parallel::Bool=true)
    reset_isws!(isws)
    ms = collect(1:M)
    if use_parallel
        if use_rng
            Wme = solve_in_parallel(get_w_2e_rng, ms)
            Wml = solve_in_parallel(get_w_2l_rng, ms)
        else
            Wme = solve_in_parallel(get_w_2e_uni, ms)
            Wml = solve_in_parallel(get_w_2l_uni, ms)
        end
    else
        Wme = zeros(length(times_uni_2), M)
        Wml = zeros(length(times_uni_2), M)
        for m=ms
            if use_rng
                Wme[:,m] = get_w_2e_rng(m)
                Wml[:,m] = get_w_2l_rng(m)
            else
                Wme[:,m] = get_w_2e_uni(m)
                Wml[:,m] = get_w_2l_uni(m)
            end
        end
    end
    return Wme, Wml
end

function plot_w_means_2le(use_rng::Bool = false, save_to_file::Bool=false)
    Wme, Wml = get_ws_2le(M, use_rng)
    mean_e = mean(Wme, dims=2)
    mean_l = mean(Wml, dims=2)
    pl = plot()
    if use_rng
        plot!(mean_e, label="Exact Sampling")
        plot!(mean_l, label="Linear Sampling")
    else
        plot!(times_uni_2, mean_e, label="Exact Sampling")
        plot!(times_uni_2, mean_l, label="Linear Sampling")
    end
    if save_to_file
        p = joinpath(exp_path(experiment_id), "means_w_el.csv")
        writedlm(p, data, ",")
        mse = mean((mean_e-mean_l).^2)
        p = joinpath(exp_path(experiment_id), "mse_w_el.csv")
        writedlm(p, mse, ",")
    end

    display(pl)
    return Wme, Wml
end

function get_ys_2le(M::Int=M)
    reset_isws!(isws)
    ms = collect(1:M)
    Yme = solve_in_parallel(calc_y_2e, ms)
    Yml = solve_in_parallel(calc_y_2l, ms)
    return Yme, Yml
end

function plot_y_means_2le(to_show::Int = M, save_to_file::Bool=false)
    Yme, Yml = get_ys_2le(to_show)
    Ye = mean(Yme[:, 1:to_show], dims=2)
    Yl = mean(Yml[:, 1:to_show], dims=2)
    times = 0:Ts:(N*Ts)
    if save_to_file
        p = joinpath(exp_path(experiment_id), "means_2el.csv")
        writedlm(p, hcat(times, Ye, Yl), ",")
    end
    pl = plot()
    plot!(times, Ye, label="Exact sampling")
    plot!(times, Yl, label="Linear sampling")
    display(pl)
    return Yme, Yml
end

function get_mse_from_file(file_name)
    p = joinpath(exp_path(experiment_id), file_name)
    data = readdlm(p, ',')
    mean((data[:,2] - data[:,3]).^2)
    # mse = mean((data[:,2] - data[:,3]).^2)
    # p = joinpath(exp_path(experiment_id), "mse.csv")
    # writedlm(p, mse, ",")
end

function write_mse_from_files(file_names, labels)
    mses = zeros(size(file_names))
    for (i, file_name) in enumerate(file_names)
        mses[i] = get_mse_from_file(file_name)
    end
    p = joinpath(exp_path(experiment_id), "mses.csv")
    writedlm(p, hcat(labels, mses), ",")
end
