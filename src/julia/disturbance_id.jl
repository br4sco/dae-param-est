using DelimitedFiles, ProgressMeter, DifferentialEquations, Interpolations, Plots
include("noise_generation.jl")


const data_dir = "C:/Programming/dae-param-est/src/julia/data/experiments/"
const Ts = 0.1
const Nw = Int(1e3)


function load_disturbance_metadata(expid::String)::DisturbanceMetaData
    W_meta_raw, W_meta_names =
        readdlm(joinpath(data_dir, expid*"/meta_W.csv"), ',', header=true)

    nx = Int(W_meta_raw[1,1])
    n_in = Int(W_meta_raw[1,2])
    n_out = Int(W_meta_raw[1,3])
    η_true = W_meta_raw[:,4]

    # n_tot = nx*n_in
    # dη = length(η_true)
    # a_vec = η_true[1:nx]
    # C_true = reshape(η_true[nx+1:end], (n_out, n_tot))

    return DisturbanceMetaData(nx, n_in, n_out, η_true, zeros(0))
end

function generate_w(z::Matrix{Float64}, dmdl::DT_SS_Model, nx::Int)
    XW_vec = simulate_noise_process_mangled(dmdl, [z])[:]
    w_vec = zeros(length(XW_vec)÷nx)
    for i=eachindex(w_vec)
        # NOTE: SCALAR_OUTPUT is assumed
        w_vec[i] = first(dmdl.Cd*XW_vec[(i-1)*nx+1:i*nx])
    end
    return w_vec
end

function generate_ws_in_parallel(zs::Array{Matrix{Float64}},  dmdl::DT_SS_Model, nx::Int, irange::UnitRange{Int64})
    len = length(irange)
    p = Progress(len, 1, "Running $len simulations...", 50)
    ws = zeros(Nw, len)
    Threads.@threads for i=irange
        w = generate_w(zs[i], dmdl, nx)
        ws[:,i] += w
        next!(p)
    end
    return ws
end

# Can be used to compare w for different values of all parameters
function get_costs(exp_id::String)
    M = 1
    W_meta = load_disturbance_metadata(exp_id)
    zs = [randn(Nw, W_meta.nx*W_meta.n_in) for m=1:M]    # n_tot = nx*n_in
    z_meas = randn(Nw, W_meta.nx*W_meta.n_in)    # n_tot = nx*n_in

    # η from  5k_u2w6_from_Alsvin is [0.8, 16, 0.0, 0.6], corresponding to
    # [η₁, η₂, 0.0, η₃], the first two being a-parameters and the last c-parameters
    cmdl_true = get_ct_disturbance_model(W_meta.η, W_meta.nx, W_meta.n_out)
    dmdl_true = discretize_ct_noise_model(cmdl_true, Ts)

    @info "Getting started..."
    # Each column of ws_true is a realization of w
    ws_true = generate_ws_in_parallel(zs, dmdl_true, W_meta.nx, 1:M)
    # w_true = generate_w(zs[1], dmdl_true, W_meta.nx)
    @info "Generated true w..."
    w_meas = generate_w(z_meas, dmdl_true, W_meta.nx)
    @info "Generated measured w..."

    a1range = 0.45:0.01:0.85
    a2range = 15:0.1:18
    crange  = 0.3:0.01:0.65

    # For comparing mean and variance accuracy of w^2 process
    meand_a1fixed = [zeros(length(a2range), length(crange)) for a1=a1range]
    meand_a2fixed = [zeros(length(a1range), length(crange)) for a2=a2range]
    meand_cfixed  = [zeros(length(a1range), length(a2range)) for c=crange]
    vard_a1fixed = [zeros(length(a2range), length(crange)) for a1=a1range]
    vard_a2fixed = [zeros(length(a1range), length(crange)) for a2=a2range]
    vard_cfixed  = [zeros(length(a1range), length(a2range)) for c=crange]

    cost_a1fixed = [zeros(length(a2range), length(crange)) for a1=a1range]
    cost_a2fixed = [zeros(length(a1range), length(crange)) for a2=a2range]
    cost_cfixed  = [zeros(length(a1range), length(a2range)) for c=crange]
    # Denotes middle index for fixed parameter. Not really useful anymore
    a1ind = (length(a1range)-1)÷2+1
    a2ind = (length(a2range)-1)÷2+1
    cind  = (length(crange)-1)÷2+1

    mincost  = Inf
    minmeand = Inf
    minvard  = Inf
    mininds_cost  = [0,0,0]
    mininds_meand = [0,0,0]
    mininds_vard  = [0,0,0]

    for (i1,a1) = enumerate(a1range)
        for (i2, a2) = enumerate(a2range)
            for (ic, c) = enumerate(crange)
                η = [a1, a2, 0.0, c]
                cmdl = get_ct_disturbance_model(η, W_meta.nx, W_meta.n_out)
                dmdl = discretize_ct_noise_model(cmdl, Ts)
                w = generate_w(zs[1], dmdl, W_meta.nx)
                # ws = generate_ws_in_parallel(zs, dmdl, W_meta.nx, 1:M)
                # cost = mean((ws_true[:,1].^2-w.^2).^2)
                cost = mean((w_meas.^2-w.^2).^2)
                meand = abs(mean(w_meas.^2)-mean(w.^2))
                vard  = abs(var(w_meas.^2)-var(w.^2))
                cost_a1fixed[i1][i2,ic] = cost
                cost_a2fixed[i2][i1,ic] = cost
                cost_cfixed[ic][i1,i2] = cost
                @info "Finished computation for parameters $i1, $i2, $ic out of $(length(a1range)), $(length(a2range)), $(length(crange))"
                meand_a1fixed[i1][i2,ic] = meand
                meand_a2fixed[i2][i1,ic] = meand
                meand_cfixed[ic][i1,i2] = meand
                vard_a1fixed[i1][i2,ic] = vard
                vard_a2fixed[i2][i1,ic] = vard
                vard_cfixed[ic][i1,i2] = vard

                if cost < mincost
                    mincost = cost
                    mininds_cost = [i1, i2, ic]
                end
                if meand < minmeand
                    minmeand = meand
                    mininds_meand = [i1, i2, ic]
                end
                if vard < minvard
                    minvard = vard
                    mininds_vard = [i1, i2, ic]
                end
            end
        end
    end

    # COMPARE CT SPECTRA??? THEY REPRESENT SECOND ORDER CHARACTERISTICS!!!

    inds = (a1ind, a2ind, cind)
    ranges = (a1range, a2range, crange)
    meands = (meand_a1fixed, meand_a2fixed, meand_cfixed)
    vards  = (vard_a1fixed, vard_a2fixed, vard_cfixed)
    mininds = (mininds_cost, mininds_meand, mininds_vard)
    minvals = (mincost, minmeand, minvard)
    return cost_a1fixed, cost_a2fixed, cost_cfixed, inds, ranges, meands, vards, mininds, minvals
    # return w_log, cost_true, cost_meas
end

# Can be used to compare w for different values of parameters a1 and a2
function get_2par_costs(exp_id::String)
    M = 1000
    W_meta = load_disturbance_metadata(exp_id)
    zs = [randn(Nw, W_meta.nx*W_meta.n_in) for m=1:M]    # n_tot = nx*n_in
    z_meas = randn(Nw, W_meta.nx*W_meta.n_in)    # n_tot = nx*n_in
    # zs_meas = [randn(Nw, W_meta.nx*W_meta.n_in) for e=1:E]    # n_tot = nx*n_in # TODO: We could implement for E>1, but maybe do that later

    # η from  5k_u2w6_from_Alsvin is [0.8, 16, 0.0, 0.6], corresponding to
    # [η₁, η₂, 0.0, η₃], the first two being a-parameters and the last c-parameters
    cmdl_true = get_ct_disturbance_model(W_meta.η, W_meta.nx, W_meta.n_out)
    dmdl_true = discretize_ct_noise_model(cmdl_true, Ts)

    @info "Getting started..."
    # Each column of ws_true_mdl is a realization of w.
    ws_true_mdl = generate_ws_in_parallel(zs, dmdl_true, W_meta.nx, 1:M)
    # w_true = generate_w(zs[1], dmdl_true, W_meta.nx)
    @info "Generated w using true model (but simulated noise)..."

    w_meas = generate_w(z_meas, dmdl_true, W_meta.nx)
    # ws_meas = generate_ws_in_parallel(zs_meas, dmld_true, W_meta.nx)
    @info "Generated measured w (simulated using \"measured\" noise)..."

    # True values, a1=0.8, a2 = 16
    a1range = 0.5:0.1:25.0
    a2range = 3:0.5:40
    # a1range = 0.45:0.01:0.85
    # a2range = 15:0.1:18


    meand = zeros(length(a1range), length(a2range))
    vard  = zeros(length(a1range), length(a2range))
    cost = zeros(length(a1range), length(a2range))

    mincost  = Inf
    minmeand = Inf
    minvard  = Inf
    minind_cost  = [0, 0]
    minind_meand = [0, 0]
    minind_vard  = [0, 0]

    # TODO: WHY ONLY OVER ONE REALIZATION? OR WHAT IS zs[1]?????????????? Yeah, only one right now...

    for (i1,a1) = enumerate(a1range)
        for (i2, a2) = enumerate(a2range)
                η = [a1, a2, 0.0, 0.6]
                cmdl = get_ct_disturbance_model(η, W_meta.nx, W_meta.n_out)
                dmdl = discretize_ct_noise_model(cmdl, Ts)
                # Case for M > 1
                ws = generate_ws_in_parallel(zs, dmdl, W_meta.nx, 1:M)
                cost[i1,i2] = mean((w_meas.^2 .- ws.^2).^2) # Takes mean over both time and M realizations of w
                meand[i1,i2] = mean(abs.(mean(w_meas.^2).-mean(ws.^2, dims=1)))    # For each realization m, computes bias of the mean over time. Saves mean over all m
                vard[i1,i2]  = mean(abs.(var(w_meas.^2).-var(ws.^2, dims=1)))       # For each realization m, computes bias of the variance over time. Saves mean over all m
                # # Case equivalent to M=1
                # w = generate_w(zs[1], dmdl, W_meta.nx)
                # cost[i1,i2] = mean((w_meas.^2-w.^2).^2)
                # meand[i1,i2] = abs(mean(w_meas.^2)-mean(w.^2))
                # vard[i1,i2]  = abs(var(w_meas.^2)-var(w.^2))
                @info "Finished computation for parameters $i1, $i2 out of $(length(a1range)), $(length(a2range))"

                if cost[i1,i2] < mincost
                    mincost = cost[i1,i2]
                    minind_cost = [i1, i2]
                end
                if meand[i1,i2] < minmeand
                    minmeand = meand[i1,i2]
                    minind_meand = [i1, i2]
                end
                if vard[i1,i2] < minvard
                    minvard = vard[i1,i2]
                    minind_vard = [i1, i2]
                end
        end
    end

    maxval = 104.262975
    function f_a1_to_a2(a1)
        if a1 <= (4*maxval)^(0.25)
            return maxval/(a1^2) + (a1^2)/4
        else
            return sqrt(maxval)
        end
    end

    maxw2 = 10.295
    f_a1_to_a2_w(a1) = maxw2 + (a1^2)/2

    myvar = (0.6^2)/(2*2.9*14.5)
    f_a1_to_a2_σ(a1) = (0.6^2)/(myvar*2*a1)

    # y = f_a1_to_a2.(a1range);
    # z = 0.0064*ones(size(y));
    # scatter!(a1range, y, z)

    # y2 = f_a1_to_a2_w.(a1range);
    # z2 = 0.0064*ones(size(y2));
    # scatter!(a1range, y2, z2)

    # y3 = f_a1_to_a2_σ.(a1range);
    # z3 = 0.0064*ones(size(y3));
    # scatter!(a1range, y3, z3)

    # - Computing parameter curve where we expect the minima to be from theory -
    mysum = mean(w_meas.^2)
    f_a1_to_a2_opt(a1) = 3*(0.6^2)/(2*a1*mysum)
    # y4 = f_a1_to_a2_opt.(a1range);
    # z4 = 0.0064*ones(size(y4));
    # scatter!(a1range, y4, z4)

    # COMPARE CT SPECTRA??? THEY REPRESENT SECOND ORDER CHARACTERISTICS!!!
    mininds = (minind_cost, minind_meand, minind_vard)
    minvals = (mincost, minmeand, minvard)
    return cost, a1range, a2range, meand, vard, mininds, minvals, mysum
    # return cost_a1fixed, cost_a2fixed, cost_cfixed, inds, ranges, meands, vards, mininds, minvals
    # # return w_log, cost_true, cost_meas
end

# Can be used to compare w for different values of a single parameter
function get_1par_costs(exp_id::String)
    M = 10
    W_meta = load_disturbance_metadata(exp_id)
    zs = [randn(Nw, W_meta.nx*W_meta.n_in) for m=1:M]    # n_tot = nx*n_in
    z_meas = randn(Nw, W_meta.nx*W_meta.n_in)    # n_tot = nx*n_in

    # η from  5k_u2w6_from_Alsvin is [0.8, 16, 0.0, 0.6], corresponding to
    # [η₁, η₂, 0.0, η₃], the first two being a-parameters and the last c-parameters
    cmdl_true = get_ct_disturbance_model(W_meta.η, W_meta.nx, W_meta.n_out)
    dmdl_true = discretize_ct_noise_model(cmdl_true, Ts)

    @info "Getting started..."
    # Each column of ws_true is a realization of w
    ws_true = generate_ws_in_parallel(zs, dmdl_true, W_meta.nx, 1:M)
    # w_true = generate_w(zs[1], dmdl_true, W_meta.nx)
    @info "Generated true w..."
    w_meas = generate_w(z_meas, dmdl_true, W_meta.nx)
    @info "Generated measured w..."

    # a1range = 0.4:0.1:1.0
    a1range = 0.4:0.2:0.8
    # w_log = [zeros(size(ws_true)) for j=a1range]
    cost_true = zeros(M, length(a1range))
    cost_meas = zeros(M, length(a1range))
    for (i,a1) = enumerate(a1range)
        η = vcat([a1], W_meta.η[2:end])

        cmdl = get_ct_disturbance_model(η, W_meta.nx, W_meta.n_out)
        dmdl = discretize_ct_noise_model(cmdl, Ts)
        # w = generate_w(zs[1], dmdl, W_meta.nx)
        ws = generate_ws_in_parallel(zs, dmdl, W_meta.nx, 1:M)
        # w_log[i] = ws
        cost_true[:,i] = mean((ws_true-ws).^2, dims=1)
        cost_meas[:,i] = mean((w_meas.-ws).^2, dims=1)
        @info "Finished computation for parameter $i out of $(length(a1range))"
    end

    # return cost_true, cost_meas
    return w_log, cost_true, cost_meas
end

#######################################################################
######### TRY ABSTRACT MATRIX, ABSTRACT ARRAY, ETC!!!!!!!!!!! #########
#######################################################################

# function get_ode_sol(η::AbstractVector{Float64}, nx::Int, nout::Int, z::AbstractVector{Float64}, trange::AbstractVector{Float64}, )
#     cmdl_true = get_ct_disturbance_model(η, nx, n_out)
#     dmdl_true = discretize_ct_noise_model(cmdl_true, Ts)
#     w_true = generate_w(z, dmdl_true, nx)
#     w_func_true = linear_interpolation(trange, w_true, extrapolation_bc=Line())
# end

function some_ode_business(exp_id::String)
    M = 100
    t_begin=0.0
    t_end=Nw*Ts
    tspan = (t_begin,t_end)
    trange = 0:Ts:(Nw-1)*Ts
    x_init=[-pi/2, 0]
    # x_init=[0.0, 0.0]
    W_meta = load_disturbance_metadata(exp_id)
    z = randn(Nw, W_meta.nx*W_meta.n_in)    # n_tot = nx*n_in
    zs = [randn(Nw, W_meta.nx*W_meta.n_in) for i=1:M]
    a_true = 0.3

    # η from  5k_u2w6_from_Alsvin is [0.8, 16, 0.0, 0.6], corresponding to
    # [η₁, η₂, 0.0, η₃], the first two being a-parameters and the last c-parameters
    cmdl_true = get_ct_disturbance_model(W_meta.η, W_meta.nx, W_meta.n_out)
    dmdl_true = discretize_ct_noise_model(cmdl_true, Ts)

    w_true = generate_w(z, dmdl_true, W_meta.nx)
    w_func_true = linear_interpolation(trange, w_true, extrapolation_bc=Line())

    wmean = mean(w_true.^2)

    ode_fn_template(x,p,t,a,w) = [x[2], -a*sin(x[1]) + 2*(wmean-w(t)^2)]

    ode_fn_true(x,p,t) = ode_fn_template(x,p,t, a_true, w_func_true)
    prob_true = ODEProblem(ode_fn_true, x_init, tspan, [0.0]) # Last argument is just dummy parameter, not used
    sol_true  = solve(prob_true, Tsit5(), reltol=1e-8, abstol=1e-8, saveat=trange)
    θtrue     = [sol_true.u[i][1] for i=eachindex(sol_true.u)]

    # # a1range = 0.5:0.3:1.1
    # # a2range = 14:2:18
    # # crange  = 0.2:0.4:1.0
    a1range = 0.45:0.025:0.85
    a2range = 15:0.25:18
    crange  = 0.3:0.025:0.65
    # a1range = 0.45:0.05:0.85
    # a2range = 15:0.5:18
    # crange  = 0.3:0.05:0.65

    cost_a1fixed = [zeros(length(a2range), length(crange)) for a1=a1range]
    cost_a2fixed = [zeros(length(a1range), length(crange)) for a2=a2range]
    cost_cfixed  = [zeros(length(a1range), length(a2range)) for c=crange]

    # function get_ode_fn(η::AbstractVector{Float64}, nx::Int, n_out::Int)
    #     cmdl = get_ct_disturbance_model(η, nx, n_out)
    #     dmdl = discretize_ct_noise_model(cmdl, Ts)
    #
    #     w = generate_w(z, dmdl, nx)
    #     w_func = linear_interpolation(trange, w, extrapolation_bc=Line())
    #
    #     return (x,p,t) -> ode_fn_template(x, p, t, a_true, w_func)
    # end

    for (i1, a1) = enumerate(a1range)
        for (i2, a2) = enumerate(a2range)
            for (ic, c) = enumerate(crange)
                η = [a1, a2, 0.0, c]

                cmdl = get_ct_disturbance_model(η, W_meta.nx, W_meta.n_out)
                dmdl = discretize_ct_noise_model(cmdl, Ts)

                function solve_realization(m::Int)
                    w = generate_w(zs[m], dmdl, W_meta.nx)
                    w_func = linear_interpolation(trange, w, extrapolation_bc=Line())

                    ode_fn(x,p,t) = ode_fn_template(x, p, t, a_true, w_func)
                    prob = ODEProblem(ode_fn, x_init, tspan, [0.0])
                    solve(prob, Tsit5(), reltol=1e-8, abstol=1e-8, saveat=trange, maxiters=Int(1e8))
                end

                θs = solve_ode_in_parallel(solve_realization, 1:M)
                θm = mean(θs, dims=2)
                cost = mean((θm-θtrue).^2)

                # w = generate_w(z, dmdl, W_meta.nx)
                # w_func = linear_interpolation(trange, w, extrapolation_bc=Line())
                # ode_fn(x,p,t) = ode_fn_template(x,p,t, a_true, w_func)
                # prob = ODEProblem(ode_fn, x_init, tspan, [0.0]) # Last argument is just dummy parameter, not used
                # sol = solve(prob, Tsit5(), reltol=1e-8, abstol=1e-8, saveat=trange, maxiters=Int(1e8))
                # θ = [sol.u[i][1] for i=eachindex(sol.u)]
                # cost = mean((θ-θtrue).^2)
                cost_a1fixed[i1][i2,ic] = cost
                cost_a2fixed[i2][i1,ic] = cost
                cost_cfixed[ic][i1,i2] = cost
                @info "Finished computation for parameters $i1, $i2, $ic out of $(length(a1range)), $(length(a2range)), $(length(crange)), cost: $cost"
            end
        end
    end

    costs = (cost_a1fixed, cost_a2fixed, cost_cfixed)
    ranges = (a1range, a2range, crange)

    return costs, ranges
end

function solve_ode_in_parallel(sol_func::Function, ms::AbstractVector{Int})
    M = length(ms)
    p = Progress(M, 1, "Running $M simulations...", 50)
    θs = zeros(Nw, M)
    Threads.@threads for i = 1:M
        sol = sol_func(ms[i])
        θ = [sol.u[i][1] for i=eachindex(sol.u)]
        θs[:,i] += θ
        next!(p)
    end
    return θs
end


# function solve_in_parallel(solve, is)
#   M = length(is)
#   p = Progress(M, 1, "Running $(M) simulations...", 50)
#   y1 = solve(is[1])
#   Y = zeros(length(y1), M)
#   Y[:, 1] += y1
#   next!(p)
#   Threads.@threads for m = 2:M
#       y = solve(is[m])
#       Y[:, m] .+= y
#       next!(p)
#   end
#   Y
# end
