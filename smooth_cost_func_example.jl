using Random, Distributions, Statistics, Plots
include("noise_generation.jl")
include("smooth_cost_func_helper_functions.jl")
include("new_noise_interpolation.jl")

Random.seed!(1234)

T  = 20         # Number of seconds of collected data
N  = 5          # Number of collected data points
Nw = 100        # Number of uniform samples of noise
M  = 1          # Number of simulated noise realizations
Nθ = 50         # Number of points cost funciton is plotted at
θ  = 0.5        # True parameters θ
θl = 0.1        # Lower bound for possible θ
θu = 1.0        # Upper bound for possible θ
j = 1
# j==0 => our proposed noise generation
# j==1 => unconditional noise generation

function A_θ(θ::Float64)
    return [0 1; -1 -θ]
end

function generate_w(t_vec::Array{Float64, 1}, noise_mdl)
    N = size(t_vec)[1]
    nw = size(noise_mdl.A)[1]
    xw = fill(NaN, (N+1, nw))
    w = fill(NaN, (N+1, 1))
    # TODO: ISN'T IT A BIT WEIRD THAT THIS STARTS AT 0...?
    # WHY WOULD TRANSITION FROM X0 TO X1 BE DETERMINISTIC?
    xw[1,:] = noise_mdl.x0
    w[1,:] = noise_mdl.C*xw[1,:]
    t = 0.0
    for k = 1:size(t_vec)[1]
        δ = t_vec[k] - t
        Φ, Γ, Λ = discretize_noise_model(noise_mdl, δ)
        xw[k+1,:] = Φ*xw[k:k,:]' + Γ*randn(Float64, (nw,1))
        w[k+1, :] = Λ*xw[k+1,:]
        t = t_vec[k]
    end
    return w
end

A  = A_θ(θ)
B  = reshape([0.; 1.], (2,1))
D = reshape([0.; 0.1], (2,1))
C  = reshape([1. 0.], (1,2))

Aw = [0 -(4^2);
      1 -(2*4*0.1);];
Bw = reshape([0.5; 0.0], (2,1))
Cw = reshape([0. 1.], (1,2))

θ_range = θl:((θu-θl)/Nθ):θu
t_vec = sort(rand(Uniform(0,T), (N,)))

n = size(A)[1]
nw = size(Aw)[1]
sys = Special_CT_SS_Model(A, B, C, D, zeros(n,))
noise_mdl = CT_SS_Model(Aw, Bw, Cw, zeros(nw,))
# u = randn((N, 1))
u = reshape([3*sin(x) for x=0.0:0.1:N*0.1], (N+1,1))
v = 0.05*randn((N+1, 1))

z_uniform_all, z_inter_all = generate_noise_new(Nw, M, 1, nw)
Awd, Bwd, Cwd = discretize_noise_model(noise_mdl, T/Nw)
dn_mdl = DT_SS_Model(Awd, Bwd, Cwd, zeros(nw,), T/Nw)
xe_processes = simulate_noise_process_new(dn_mdl, z_uniform_all)

w_true = generate_w(t_vec, noise_mdl)

# True system trajectory, plus measurement noise
y, _, x = simulate_system_exactly(sys, noise_mdl, t_vec, u, v, w_true)

cost_func0 = fill(NaN, Nθ)
cost_func1 = fill(NaN, Nθ)
for i=1:Nθ
    local θ = θ_range[i]
    mdl = Special_CT_SS_Model(A_θ(θ), B, C, D, zeros(n,))
    y_mat1 = fill(NaN, N+1, M)
    y_mat0 = fill(NaN, N+1, M)
    for m=1:M

        xe0 = fill(NaN, N+1, nw)
        e0  = fill(NaN, N+1, 1)
        # TODO: ISN'T IT A BIT WEIRD THAT THIS STARTS AT 0...?
        # WHY WOULD TRANSITION FROM X0 TO X1 BE DETERMINISTIC?
        xe0[1,:] = zeros(nw,1)
        e0[1,:] = noise_mdl.C*xe0[1,:]
        function get_xe(t::Float64)
            noise_inter(t, T/Nw, noise_mdl.A, noise_mdl.B, xe_processes[:,m],
                        z_inter_all[m], zeros(Int64, Nw), noise_mdl.x0)
        end

        for k=1:1:size(t_vec)[1]
            xe0[k+1,:] = get_xe(t_vec[k])
            e0[k+1,:]  = noise_mdl.C*xe0[k+1:k+1,:]'
        end

        e1 = generate_w(t_vec, noise_mdl)

        y_hat0, _, x_hat0= simulate_system_exactly(mdl,
                                                 noise_mdl,
                                                 t_vec,
                                                 u,
                                                 zeros(size(v)),
                                                 e0)
         y_hat1, _, x_hat1= simulate_system_exactly(mdl,
                                                  noise_mdl,
                                                  t_vec,
                                                  u,
                                                  zeros(size(v)),
                                                  e1)
        y_mat1[:,m] = y_hat1
        y_mat0[:,m] = y_hat0
    end
    y_bar1 = mean(y_mat1, dims=2)
    cost_func1[i] = mean((y-y_bar1).^2, dims=1)[1,1]
    y_bar0 = mean(y_mat0, dims=2)
    cost_func0[i] = mean((y-y_bar0).^2, dims=1)[1,1]
end

plot(θ_range, cost_func0, xlabel="θ", ylabel="cost", label="Exact noise interpolation")
plot!(θ_range, cost_func1, label="Unconditional noise sampling")
# savefig("./cost_plot.png")
# TODO: Add display()-call?
