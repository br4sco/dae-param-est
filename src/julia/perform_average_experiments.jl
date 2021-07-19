include("simulation.jl")
include("noise_generation.jl")
include("noise_interpolation.jl")

seed = 1234
Random.seed!(seed)

const data_dir = joinpath("data", "experiments")
exp_path(id) = joinpath(data_dir, id)
const experiment_id = "small_input_experiments_x10"
const Ts = 0.1
const δs = [0.05, 0.5, 1.0, 2.0, 5.0]
# const δs = [5.0]
# const δs = [0.05, 0.1, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 4.5, 5.0]
const δ_min = minimum(δs)
const factors = [Int(δ/δ_min) for δ = δs]
const T = 100
const Nw_min = ceil(Int, T/δ_min)
const Nws = [ceil(Int, T/δ) for δ = δs]
const Nw_extra = 200
const N = ceil(Int, T/Ts)
const Q = 2000
const W = 100
const M = 500

const nx = 2
const A = [0.0 1.0; -4^2 -0.8]
const B = reshape([0.0 1.0], (2,1))
const C = [1.0 0.0]

const x0 = zeros(nx)
const dmdl = discretize_ct_noise_model(A, B, C, δ_min, x0)

const Zu = [randn(Nw_min+Nw_extra, nx)]
const Zm = [randn(Nw_min+Nw_extra, nx) for m=1:M]
# const Z2 = [Z1[1][1:factor:end, :]]
# const Z2 = [randn(Nw2+Nw_extra, nx)]

mangle_XW(XW::Array{Array{Float64, 1}, 2}) =
  hcat([vcat(XW[:, m]...) for m = 1:size(XW,2)]...)

const XWup  = simulate_noise_process(dmdl, Zu)
const XWdp = simulate_noise_process(dmdl, Zm)
# const XWd2p = simulate_noise_process(dmdl2, Z2)
# const XWd2p = XWd1p[1:factor:end, :]
const XWu = mangle_XW(XWup)
const XWd = mangle_XW(XWdp)
# const XWd2 = mangle_XW(XWd2p)

function interpx(xl::AbstractArray{Float64, 1},
                 xu::AbstractArray{Float64, 1},
                 t::Float64,
                 δ::Float64,
                 n::Int)

  xl .+ (t - (n - 1) * δ) .* (xu .- xl) ./ δ
end

function mk_noise_interp(A::Array{Float64, 2},
                          B::Array{Float64, 2},
                          C::Array{Float64, 2},
                          XW::Array{Float64, 2},
                          m::Int,
                          δ::Float64)

  let
    nx = size(A, 1)
    function w(t::Float64)
      n = Int(floor(t / δ)) + 1

      k = (n - 1) * nx + 1

      # xl = view(XW, k:(k + nx - 1), m)
      # xu = view(XW, (k + nx):(k + 2nx - 1), m)

      xl = XW[k:(k + nx - 1), m]
      xu = XW[(k + nx):(k + 2nx - 1), m]

      first(C * interpx(xl, xu, t, δ, n))
    end
  end
end

function mk_newer_noise_interp(A::Array{Float64, 2},
                               B::Array{Float64, 2},
                               C::Array{Float64, 2},
                               XWdp::Array{Array{Float64, 1},2},
                               isw::InterSampleWindow,
                               m::Int,
                               δ::Float64)

   let
       function w(t::Float64)
           xw_temp = noise_inter(t, δ, A, B, view(XWdp, :, m), isw)
           return first(C*xw_temp)
       end
   end
end

isws = [initialize_isw(Q, W, nx, true) for m=1:M]
u(t::Float64) = mk_noise_interp(A, B, C, u_scale.*XWu, 1, δ_min)(t)

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
const θ0 = L                    # true value of θ
# mk_θs(θ::Float64) = [m, L, g, θ]
realize_model(w::Function, N::Int) =
  problem(pendulum(φ0, t -> u(t) + u_bias, w, [m, L, g, k]), N, Ts)

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

function compare_Y_means(sims_per_δ::Int=1)
    ms = collect(1:M)
    times = 0:Ts:(N*Ts)
    Yes = [zeros(N+1, sims_per_δ) for j=1:length(δs)]
    Yls = [zeros(N+1, sims_per_δ) for j=1:length(δs)]
    mses = zeros(length(δs), sims_per_δ)
    for (i, δ) in enumerate(δs)
        println("Running experiments for δ=$δ")
        XWdp_i = XWdp[1:factors[i]:end, :]
        XWd_i  = mangle_XW(XWdp_i)
        wmd(m::Int) = mk_noise_interp(A, B, w_scale.*C, XWd_i, m, δ)
        wmn(m::Int) = mk_newer_noise_interp(A, B, w_scale.*C, XWdp_i, isws[m], m, δ)
        calc_y_e(m::Int) = solvew(t -> wmn(m)(t), N) |> h
        calc_y_l(m::Int) = solvew(t -> wmd(m)(t), N) |> h
        for j=1:sims_per_δ
            println("Running realization $j out of $sims_per_δ")
            reset_isws!(isws)
            Yme = solve_in_parallel(calc_y_e, ms)
            Yml = solve_in_parallel(calc_y_l, ms)
            Yes[i][:,j] = mean(Yme, dims=2)
            Yls[i][:,j] = mean(Yml, dims=2)
            mses[i,j] = mean((Yes[i][:,j]-Yls[i][:,j]).^2)
        end

        δ_string = replace("$δ", "." => "p")
        p = joinpath(exp_path(experiment_id), "means_M$(M)_$(δ_string)_el.csv")
        try
            writedlm(p, hcat(times, Yes[i], Yls[i]), ",")
        catch e
            @warn "Failed storing means of Ym and Yl for δ=$δ, make sure to do it manually"
        end
    end

    p = joinpath(exp_path(experiment_id), "mses_M$(M)_el.csv")
    try
        writedlm(p, hcat(δs, mses), ",")
    catch e
        @warn "Failed storing mses, make sure to do it manually"
    end

    p = joinpath(exp_path(experiment_id), "metadata_M$(M)_el.csv")
    try
        file = open(p, "w")
        write(file, "Deltas: $δs, w_cale: $w_scale, u_scale: $u_scale, Ts: $Ts, T: $T")
        close(file)
    catch e
        @warn "Failed storing metadata, make sure to do it manually"
    end

    return Yes, Yls, mses
end

# function compare_Y_variance(num_sims::Int=10, num_realizations::Int=M)
#     ms = collect(1:M)
#     times = 0:Ts:(N*Ts)
#
#
#
#
#     Yes = [zeros(N, length(δs)) for j=1:num_sims]
#     Yls = [zeros(N, length(δs)) for j=1:num_sims]
#     for (i, δ) i enumerate(δs)
#         XWdp_i = XWdp[1:factors[i]:end, :]
#         XWd_i  = mangle_XW(XWdp_i)
#         wmd(m::Int) = mk_noise_interp(A, B, w_scale.*C, XWd_i, m)
#         wmn(m::Int) = mk_newer_noise_interp(A, B, w_scale.*C, XWdp_i, isws[m], m)
#         calc_y_e(m::Int) = solvew(t -> wmn(m)(t), N) |> h
#         calc_y_l(m::Int) = solvew(t -> wmd(m)(t), N) |> h
#
#         # SHOULD BE SAME FUNCTION!!!
#
#         for j=1:num_sims
#             reset_isws!(isws)
#             Yme = solve_in_parallel(calc_y_e, ms)
#             Yml = solve_in_parallel(calc_y_l, ms)
#             Ye = mean(Yme, dims=2)
#             Yl = mean(Yml, dims=2)
#             Yes[j][:,i] = Ye
#             Yls[j][:,i] = Yl
#
#             δ_string = repace("$δ", "." => "p")
#             p = joinpath(exp_path(experiment_id), "means_M$(M)_$(δ_string)_el.csv")
#             try
#                 writedlm(p, hcat(times, Ye, Yl), ",")
#             catch e
#                 @warn "Failed storing Ym and Yl for δ=$δ, make sure to do it manually"
#             end
#         end
#     end
#
#     p = joinpath(exp_path(experiment_id), "metadata_M$(M)_$(δ_string)_el.csv")
#     try
#         writedlm(p, "Deltas: $δs, w_cale: $w_scale, u_scale: $u_scale, Ts: $Ts, T: $T", ",")
#     catch e
#         @warn "Failed storing metadata, make sure to do it manually"
#     end
#
#     return Yes, Yls, mses
# end
