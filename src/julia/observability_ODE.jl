using DifferentialEquations
using Plots

N = 200
Ts = 0.1

g = 9.81
L = 6.25

# ode_fn(x,p,t) = [x[2], -(g/L)*sin(x[1])]
ode_fn(x,p,t) = [x[2], -(p[2]/p[1])*sin(x[1])]

t_begin=0.0
t_end=N*Ts
tspan = (t_begin,t_end)
trange = 0:Ts:N*Ts
x_init=[-pi/2, 0]

prob_true = ODEProblem(ode_fn, x_init, tspan, [L, g])
num_sol = solve(prob_true, Tsit5(), reltol=1e-8, abstol=1e-8, saveat=trange)

Y = num_sol.u

# Computing cost function
num_steps = 5
step_size = 0.1
vec = [1,1]./sqrt(2)

range = 1-num_steps*step_size:step_size:1+num_steps*step_size
Ylog = zeros(length(trange), length(range))

for i=eachindex(range)
    pars = range[i]*[L,g]
    prob = ODEProblem(ode_fn, x_init, tspan, pars)
    sol = solve(prob, Tsit5(), reltol=1e-8, abstol=1e-8, saveat=trange)
    Ylog[:,i] = [sol.u[i][1] for i=eachindex(sol.u)]
end

# plot(num_sol.t, an_sol.(num_sol.t),
#     linewidth=2, ls=:dash,
#     title="ODE 1st order IVP solved by D.E. package",
#     xaxis="t", yaxis="x",
#     label="analytical",
#     legend=true)
# plot!(num_sol,
#     linewidth=1,
#     label="numerical")
