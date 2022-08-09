using DiffEqSensitivity, OrdinaryDiffEq, Zygote, ForwardDiff

function fiip(res, xp,x,p,t)
    m = p[1]
    L = p[2]
    g = p[3]
    k = p[4]
    res[1] = xp[1] - x[4] + 2xp[6]*x[1]
    res[2] = xp[2] - x[5] + 2xp[6]*x[2]
    res[3] = m*xp[4] - xp[3]*x[1] + k*abs(x[4])*x[4]# - ut[1] - wt[1]^2
    res[4] = m*xp[5] - xp[3]*x[2] + k*abs(x[5])*x[5] + m*g
    res[5] = x[1]^2 + x[2]^2 - L^2
    res[6] = x[4]*x[1] + x[5]*x[2]
    # Equation for obtaining angle
    res[7] = x[7] - atan(x[1] / -x[2])
end


x1_0 = L * sin(Φ)
x2_0 = -L * cos(Φ)
dx3_0 = m*g/x2_0
dx4_0 = -g*tan(Φ)
x0 = vcat([x1_0, x2_0], zeros(4), [atan(x1_0 / -x2_0)])
xp0 = vcat([0., 0., dx3_0, dx4_0], zeros(3))
dvars = vcat(fill(true, 6), [false])
p = [0.3, 6.25, 9.81, 6.25]

prob = DAEProblem(fiip, xp0, x0, (0, N*Ts), p, differential_vars=dvars)
sol = solve(prob)#,Tsit5())

# function sum_of_solution(x)
#     _prob = remake(prob,u0=x[1:2],p=x[3:end])
#     sum(solve(_prob,Tsit5(),reltol=1e-6,abstol=1e-6,saveat=0.1))
# end
# dx = ForwardDiff.gradient(sum_of_solution,[u0;p])

function sum_of_solution(up0,u0,p)
  _prob = remake(prob,du0=up0,u0=u0,p=p)
  sol = solve(_prob,reltol=1e-6,abstol=1e-6,saveat=0.1,sensealg=ForwardDiffSensitivity())
  sol.u[1][1]
end
dup01,du01,dp1 = Zygote.gradient(sum_of_solution,xp0,x0,p)
