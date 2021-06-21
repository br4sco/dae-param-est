include("simulation.jl")
include("noise_generation.jl")
include("noise_interpolation.jl")

seed = 1234
Random.seed!(seed)

const Ts = 0.1
const δ1 = 0.05
const δ2 = 0.5
const factor = Int(δ2/δ1)
const T = 100
const Nw1 = Int(T/δ1)
const Nw2 = Int(T/δ2)
const Nw_extra = 100
const N = Int(T/Ts)
const Q = 500
const W  = 100

const nx = 2
const A = [0.0 1.0; -4^2 -0.8]
const B = reshape([0.0 1.0], (2,1))
const C = [1.0 0.0]

const x0 = zeros(nx)
const dmdl1 = discretize_ct_noise_model(A, B, C, δ1, x0)
const dmdl2 = discretize_ct_noise_model(A, B, C, δ2, x0)

const Zu = [randn(Nw1+Nw_extra, nx)]
const Z1 = [randn(Nw1+Nw_extra, nx)]
# const Z2 = [Z1[1][1:factor:end, :]]
# const Z2 = [randn(Nw2+Nw_extra, nx)]

mangle_XW(XW::Array{Array{Float64, 1}, 2}) =
  hcat([vcat(XW[:, m]...) for m = 1:size(XW,2)]...)

const XWup  = simulate_noise_process(dmdl1, Zu)
const XWd1p = simulate_noise_process(dmdl1, Z1)
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
                             XW::Array{Float64, 2})

  let
    nx = size(A, 1)
    function w(t::Float64)
      n = Int(floor(t / δ1)) + 1

      k = (n - 1) * nx + 1

      # xl = view(XW, k:(k + nx - 1), m)
      # xu = view(XW, (k + nx):(k + 2nx - 1), m)

      xl = XW[k:(k + nx - 1), 1]
      xu = XW[(k + nx):(k + 2nx - 1), 1]

      first(C * interpx(xl, xu, t, δ1, n))
    end
  end
end

function mk_noise_interp2(A::Array{Float64, 2},
                             B::Array{Float64, 2},
                             C::Array{Float64, 2},
                             XW::Array{Float64, 2})

  let
    nx = size(A, 1)
    function w(t::Float64)
      n = Int(floor(t / δ2)) + 1

      k = (n - 1) * nx + 1

      # xl = view(XW, k:(k + nx - 1), m)
      # xu = view(XW, (k + nx):(k + 2nx - 1), m)

      xl = XW[k:(k + nx - 1), 1]
      xu = XW[(k + nx):(k + 2nx - 1), 1]

      first(C * interpx(xl, xu, t, δ2, n))
    end
  end
end

function mk_newer_noise_interp1(A::Array{Float64, 2},
                               B::Array{Float64, 2},
                               C::Array{Float64, 2},
                               isw::InterSampleWindow)

   let
       function w(t::Float64)
           xw_temp = noise_inter(t, δ1, A, B, view(XWd1p, :, 1), isw)
           return first(C*xw_temp)
       end
   end
end

function mk_noise_interp2(A::Array{Float64, 2},
                             B::Array{Float64, 2},
                             C::Array{Float64, 2},
                             XW::Array{Float64, 2})

  let
    nx = size(A, 1)
    function w(t::Float64)
      n = Int(floor(t / δ2)) + 1

      k = (n - 1) * nx + 1

      # xl = view(XW, k:(k + nx - 1), m)
      # xu = view(XW, (k + nx):(k + 2nx - 1), m)

      xl = XW[k:(k + nx - 1), 1]
      xu = XW[(k + nx):(k + 2nx - 1), 1]
      first(C * interpx(xl, xu, t, δ2, n))
    end
  end
end

function mk_newer_noise_interp2(A::Array{Float64, 2},
                               B::Array{Float64, 2},
                               C::Array{Float64, 2},
                               isw::InterSampleWindow)

   let
       function w(t::Float64)
           xw_temp = noise_inter(t, δ2, A, B, view(XWd2p, :, 1), isw)
           return first(C*xw_temp)
       end
   end
end

isw = initialize_isw(Q, W, nx, true)

wmd1(t::Float64) = mk_noise_interp1(A, B, w_scale.*C, XWd1)(t)
wmd2(t::Float64) = mk_noise_interp2(A, B, w_scale.*C, XWd2)(t)
wmn1(t::Float64) = mk_newer_noise_interp1(A, B, w_scale.*C, isw)(t)
wmn2(t::Float64) = mk_newer_noise_interp2(A, B, w_scale.*C, isw)(t)
u(t::Float64) = mk_noise_interp1(A, B, C, XWu)(t)

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

# Gets y when using exact interpolation for both δ1 and δ2
function get_ys_12()
    reset_isw!(isw)
    y1 = solvew(wmn1, N) |> h
    reset_isw!(isw)
    y2 = solvew(wmn2, N) |> h
    return y1, y2
end

# Gets y when using both linear and exact interpolation for δ2
function get_ys_2le()
    reset_isw!(isw)
    ye = solvew(wmn2, N) |> h
    yl = solvew(wmd2, N) |> h
    return ye, yl
end

# Gets w when using exact interpolation for both δ1 and δ2
function get_ws_12(step1::Float64=δ1, step2::Float64=δ2,
                t_start::Float64=0.0, t_end::Float64=N*Ts)
    reset_isw!(isw)
    w1 = [wmn1(t) for t=t_start:step1:t_end]
    reset_isw!(isw)
    w2 = [wmn2(t) for t=t_start:step2:t_end]
    return w1, w2
end

# Gets w when using both linear and exact interpolation for δ2
function get_ws_2le(step::Float64 = δ2, t_start::Float64=0.0, t_end::Float64=N*Ts)
    reset_isw!(isw)
    we = [wmn2(t) for t=t_start:step:t_end]
    wl = [wmd2(t) for t=t_start:step:t_end]
    return we, wl
end

# Plots w when using exact interpolation for both δ1 and δ2
function plot_ws_12(show_scatter::Bool = false, step1::Float64=δ1, step2::Float64=δ2,
                 t_start::Float64=0.0, t_end::Float64=N*Ts)

    t0_1 = t_start - rem(t_start, δ1)
    t0_2 = t_start - rem(t_start, δ2)
    times1 = t_start:step1:t_end
    times2 = t_start:step2:t_end
    reset_isw!(isw)
    w1 = [wmn1(t) for t=times1]
    reset_isw!(isw)
    w2 = [wmn2(t) for t=times2]
    reset_isw!(isw)
    w1_fixed = [wmn1(t) for t = t0_1:δ1:t_end]
    reset_isw!(isw)
    w2_fixed = [wmn2(t) for t = t0_2:δ2:t_end]
    pl = plot()
    if show_scatter
        scatter!(pl, t0_1:δ1:t_end, w1_fixed)
        scatter!(pl, t0_2:δ2:t_end, w2_fixed, markershape=:star5)
    end
    plot!(pl, times1, w1)
    plot!(pl, times2, w2)
end

# Plots w when using both linear and exact interpolation for δ2
function plot_ws_2le(show_scatter::Bool = false, step::Float64=δ2,
                    t_start::Float64=0.0, t_end::Float64=N*Ts)
    if Q < δ2/step
        @warn "Q is chosen too small to successfully store all inter-sample values for this step size. Needs at least Q=$(round(Int, δ2/step))"
    end

    t0 = t_start - rem(t_start, δ2)
    times = t_start:step:t_end
    reset_isw!(isw)
    we = [wmn2(t) for t=times]
    wl = [wmd2(t) for t=times]
    w_fixed = [wmd2(t) for t = t0:δ2:t_end]
    pl = plot()
    if show_scatter
        scatter!(pl, t0:δ2:t_end, w_fixed, markershape=:star5, label="Fixed samples")
    end
    plot!(pl, times, we, label="Exact method")
    plot!(pl, times, wl, label="Linear method")
end

# Plots y when using exact interpolation for both δ1 and δ2
function plot_ys_12(percent_start::Float64 = 0.0, percent_end::Float64=1.0)
    y1, y2 = get_ys()
    times = 0:Ts:(N*Ts)
    len = length(times)
    start_ind = max(round(Int, percent_start*len), 1)
    end_ind   = min(round(Int, percent_end * len), len)
    pl = plot()
    plot!(times[start_ind:end_ind], y1[start_ind:end_ind], label="High sampling frequency")
    plot!(times[start_ind:end_ind], y2[start_ind:end_ind], label="Low sampling frequency")
end

# Plots y when using both linear and exact interpolation for δ2
function plot_ys_2le(percent_start::Float64 = 0.0, percent_end::Float64=1.0)
    ye, yl = get_ys_2le()
    times = 0:Ts:(N*Ts)
    len = length(times)
    start_ind = max(round(Int, percent_start*len), 1)
    end_ind   = min(round(Int, percent_end * len), len)
    pl = plot()
    plot!(times[start_ind:end_ind], ye[start_ind:end_ind], label="Exact sampling")
    plot!(times[start_ind:end_ind], yl[start_ind:end_ind], label="Linear sampling")
end
