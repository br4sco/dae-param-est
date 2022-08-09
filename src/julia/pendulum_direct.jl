## Adapted from  doc/libsundials-serial-dev/examples/ida/serial/idaRoberts_dns.c and
##               sundialsTB/ida/examples_ser/midasRoberts_dns.m

## /*
##  * -----------------------------------------------------------------
##  * $Revision: 1.2 $
##  * $Date: 2009/01/21 21:46:40 $
##  * -----------------------------------------------------------------
##  * Programmer(s): Allan Taylor, Alan Hindmarsh and
##  *                Radu Serban @ LLNL
##  * -----------------------------------------------------------------
##  * This simple example problem for IDA, due to Robertson,
##  * is from chemical kinetics, and consists of the following three
##  * equations:
##  *
##  *      dy1/dt = -.04*y1 + 1.e4*y2*y3
##  *      dy2/dt = .04*y1 - 1.e4*y2*y3 - 3.e7*y2**2
##  *         0   = y1 + y2 + y3 - 1
##  *
##  * on the interval from t = 0.0 to t = 4.e10, with initial
##  * conditions: y1 = 1, y2 = y3 = 0.
##  *
##  * While integrating the system, we also use the rootfinding
##  * feature to find the points at which y1 = 1e-4 or at which
##  * y3 = 0.01.
##  *
##  * The problem is solved with IDA using IDADENSE for the linear
##  * solver, with a user-supplied Jacobian. Output is printed at
##  * t = .4, 4, 40, ..., 4e10.
##  * -----------------------------------------------------------------
##  */

using Sundials

# Mohamed's DAE
θ = -0.2
function resrob(tres, yy_nv, yp_nv, rr_nv, user_data)
    x = convert(Vector, yy_nv)
    xp = convert(Vector, yp_nv)
    res = convert(Vector, rr_nv)
    ut = sin(tres)
    res[1] = xp[1] + θ*x[1] + ut
    res[2] = (xp[1]^2+1)x[2] + 2

    return Sundials.IDA_SUCCESS
end


# # ODE
# a = -1
# b = 0.52
# function resrob(tres, yy_nv, yp_nv, rr_nv, user_data)
#     x = convert(Vector, yy_nv)
#     xp = convert(Vector, yp_nv)
#     res = convert(Vector, rr_nv)
#     ut = sin(tres)
#     res[1] = xp[1] - a*x[1]-b*ut
#     res[2] = xp[2] - a*x[1]
#
#     return Sundials.IDA_SUCCESS
# end

# # PENDULUM
# ## Define the system residual function.
# function resrob(tres, yy_nv, yp_nv, rr_nv, user_data)
#     x = convert(Vector, yy_nv)
#     xp = convert(Vector, yp_nv)
#     res = convert(Vector, rr_nv)
#
#     wt = [0.0]#w(t)
#     ut = [0.0]#u(t)
#     # Dynamic Equations
#     res[1] = xp[1] - x[4] + 2xp[6]*x[1]
#     res[2] = xp[2] - x[5] + 2xp[6]*x[2]
#     res[3] = m*xp[4] - xp[3]*x[1] + k*abs(x[4])*x[4] - ut[1] - wt[1]^2
#     res[4] = m*xp[5] - xp[3]*x[2] + k*abs(x[5])*x[5] + m*g
#     res[5] = x[1]^2 + x[2]^2 - L^2
#     res[6] = x[4]*x[1] + x[5]*x[2]
#     # Equation for obtaining angle
#     res[7] = x[7] - atan(x[1] / -x[2])
#
#     # rr[1] = -0.04 * yy[1] + 1.0e4 * yy[2] * yy[3]
#     # rr[2] = -rr[1] - 3.0e7 * yy[2] * yy[2] - yp[2]
#     # rr[1] -= yp[1]
#     # rr[3] = yy[1] + yy[2] + yy[3] - 1.0
#     return Sundials.IDA_SUCCESS
# end

# #TANK
# ## Define the system residual function.
# function resrob(tres, yy_nv, yp_nv, rr_nv, user_data)
#     x = convert(Vector, yy_nv)
#     xp = convert(Vector, yp_nv)
#     res = convert(Vector, rr_nv)
#
#     wt = [0.0]#w(t)
#     ut = zeros(6)#u(t)
#     # Dynamic Equations
#     res[1] = xp[1] - ut[1] + ut[4]
#     res[2] = xp[2] - ut[1]*(ut[2]-x[2])/x[1] + k0*exp(-k1/x[4])x[2]
#     res[3] = xp[3] + ut[1]x[3]/x[1] - k0*exp(-k1/x[4])x[2]
#     res[4] = x[6] - k3*x[5]/x[1] - ut[1]*(ut[3]-x[4])/x[1] + k0*k2*exp(-k1/x[4])*x[2]
#     res[5] = xp[4] + k3*x[5]/k4 - ut[5]*(ut[6]-x[4])/k4 - wt[1]^2
#     res[6] = x[6] - xp[4]
#
#     # rr[1] = -0.04 * yy[1] + 1.0e4 * yy[2] * yy[3]
#     # rr[2] = -rr[1] - 3.0e7 * yy[2] * yy[2] - yp[2]
#     # rr[1] -= yp[1]
#     # rr[3] = yy[1] + yy[2] + yy[3] - 1.0
#     return Sundials.IDA_SUCCESS
# end

function my_own_residual_test(x, xp)
    res = zeros(7)
    wt = [0.0]#w(t)
    ut = [0.0]#u(t)
    # Dynamic Equations
    res[1] = xp[1] - x[4] + 2xp[6]*x[1]
    res[2] = xp[2] - x[5] + 2xp[6]*x[2]
    res[3] = m*xp[4] - xp[3]*x[1] + k*abs(x[4])*x[4] - ut[1] - wt[1]^2
    res[4] = m*xp[5] - xp[3]*x[2] + k*abs(x[5])*x[5] + m*g
    res[5] = x[1]^2 + x[2]^2 - L^2
    res[6] = x[4]*x[1] + x[5]*x[2]
    # Equation for obtaining angle
    res[7] = x[7] - atan(x[1] / -x[2])
    return res
end

const m = 0.3                   # [kg]
const L = 6.25                  # [m], gives friction-less period T = 5s (T ≅ 2√L)
const g = 9.81                  # [m/s^2]
const k = 6.25                  # [1/s^2]
const φ0 = 0.0
# DEBUG
@warn "Initializing input and disturbances to zero"
nout = 12
t0 = 0.0

# FOR MOHAMED'S DAE
neq = 2
x0 = 1.0
yy0 = [x0; x0]
yp0 = [-θ*x0; -2/((θ*x0)^2+1)]


# # FOR ODE
# neq = 2
# x0 = 2.0
# yy0 = [x0; x0]
# yp0 = [a*x0; a*x0]


# # FOR PENDULUM !!!!!!!!!!!!!!!!!!!
# const u0 = 0.0
# const w0 = 0.0
# neq = 7
# x1_0 = L * sin(φ0)
# x2_0 = -L * cos(φ0)
# dx3_0 = m*g/x2_0
# dx4_0 = -g*tan(φ0) + (u0 + w0^2)/m
# yy0 = vcat([x1_0, x2_0], zeros(4), [atan(x1_0 / -x2_0)])
# yp0 = vcat([0., 0., dx3_0, dx4_0], zeros(3))


# # FOR TANK REACTOR !!!!!!!!!!!!!!!!!!!!
# neq = 6
# const u0 = zeros(6)
# const w0 = 0.0
# const V0 = 10.0
# const T0 = 293.15
# const k0 = 1.0
# const k1 = 1.0
# const k2 = 0.02
# const k3 = 1.0
# const k4 = 1.0
# x5_0 = (V0*u0[5]*(u0[6]-T0) - k4*u0[1]*(u0[3]-T0))/((V0+k4)k3)
# xp4_0 = -k3*x5_0/k4 + u0[5]*(u0[6]-T0)/k4
# xp6_0 = -k3*x5_0/V0^2 - u0[1]*xp4_0/V0 - u0[1]*(u0[3]-T0)/V0^2 + k0*k2*u0[1]*u0[2]exp(-k1/T0)/V0
#
# yy0 = [V0; 0.; 0.; T0; x5_0; xp4_0]
# yp0 = [u0[1]-u0[4]; u0[1]u0[2]/V0; 0.; xp4_0; 0.; xp6_0]

# yy0 = [1.0, 0.0, 0.0]
# yp0 = [-0.04, 0.04, 0.0]
rtol = 1e-4
# avtol = [1e-8, 1e-8, 1e-6]
avtol = [1e-3, 1e-3, 1e-3]
tout1 = 0.4

# my_res = my_own_residual_test(yy0, yp0)
# @info "my_res: $my_res"
# hey = hop + hiya

mem = Sundials.IDACreate()
Sundials.@checkflag Sundials.IDAInit(mem, resrob, t0, yy0, yp0)
Sundials.@checkflag Sundials.IDASVtolerances(mem, rtol, avtol)

# ## Call IDARootInit to specify the root function grob with 2 components
# Sundials.@checkflag Sundials.IDARootInit(mem, 2, grob)

## Call IDADense and set up the linear solver.
# A = Sundials.SUNDenseMatrix(length(y0), length(y0))
# LS = Sundials.SUNLinSol_Dense(y0, A)
A = Sundials.SUNDenseMatrix(length(yy0), length(yy0))   # MY OWN TRY
LS = Sundials.SUNLinSol_Dense(yy0, A)                   # MY OWN TRY
Sundials.@checkflag Sundials.IDADlsSetLinearSolver(mem, LS, A)

#TODO: NOTE: Maybe we are supposed to manually pass Jacobian? Or nah, we didn't
# do that for indirect interface. But myabe worth a try to see if it fixed convergence...?

iout = 0
tout = tout1
tret = [1.0]

while iout < nout
    yy = similar(yy0)   # MY: The similar() is an inbuilt function in julia which is used to create an uninitialized mutable array with the given element type and size, based upon the given source array.
    yp = similar(yp0)
    retval = Sundials.IDASolve(mem, tout, tret, yy, yp, Sundials.IDA_NORMAL)
    println("T=", tout, ", Y=", yy)
    if retval == Sundials.IDA_ROOT_RETURN   # MY: ARE THEY TRYING TO FIND CONSISTENT INITIAL CONDITIONS OR SOMETHING...?
        rootsfound = zeros(Cint, 2)
        Sundials.@checkflag Sundials.IDAGetRootInfo(mem, rootsfound)
        println("roots=", rootsfound)
    elseif retval == Sundials.IDA_SUCCESS
        global iout += 1
        global tout += 0.1
    else
        @warn "YEAHED MESSED UP"
        break
    end
end

Sundials.SUNLinSolFree_Dense(LS)
Sundials.SUNMatDestroy_Dense(A)
