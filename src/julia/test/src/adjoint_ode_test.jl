using DiffEqSensitivity, OrdinaryDiffEq, Zygote, ForwardDiff

function fiip(du,u,p,t)
  du[1] = dx = p[1]*u[1] - p[2]*u[1]*u[2]
  du[2] = dy = -p[3]*u[2] + p[4]*u[1]*u[2]
end
p = [1.5,1.0,3.0,1.0]; u0 = [1.0;1.0]
prob = ODEProblem(fiip,u0,(0.0,10.0),p)
sol = solve(prob,Tsit5())

# function sum_of_solution(x)
#     _prob = remake(prob,u0=x[1:2],p=x[3:end])
#     sum(solve(_prob,Tsit5(),reltol=1e-6,abstol=1e-6,saveat=0.1))
# end
# dx = ForwardDiff.gradient(sum_of_solution,[u0;p])

function sum_of_solution(u0,p)
  _prob = remake(prob,u0=u0,p=p)
  # Ooooh, okay, the function they differentiate is simply the sum.
  # How on earth does that work? Don't they need the derivatives of the function?
  # Ooooh, maybe not for ode...? I'm not sure actually.
  # SURELY THEY WOULD NEED DERIVATIVES OF g????
  # Actually, if g doesn't depend explicitly on p, then no, they don't need them
  # for Gp. Buuuuut they are needed for adjoint system, wrt u!!!!
  # sum(solve(_prob,Tsit5(),reltol=1e-6,abstol=1e-6,saveat=0.1,sensealg=QuadratureAdjoint()))
  sol = solve(_prob,Tsit5(),reltol=1e-6,abstol=1e-6,saveat=0.1,sensealg=QuadratureAdjoint())
  sol.u[50][1]
end
du01,dp1 = Zygote.gradient(sum_of_solution,u0,p)
# Yeah, there seem to be some black ODE-magic going on...
# Could probably work for our DAEs, buuut it might be hard to adapt, since we
# need independent realizations and all that, which we cannot get in this
# high-level interface
