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

# ## Define the system residual function.
# function resrob(tres, yy_nv, yp_nv, rr_nv, user_data)
#     yy = convert(Vector, yy_nv)
#     yp = convert(Vector, yp_nv)
#     rr = convert(Vector, rr_nv)
#     rr[1] = -0.04 * yy[1] + 1.0e4 * yy[2] * yy[3]
#     rr[2] = -rr[1] - 3.0e7 * yy[2] * yy[2] - yp[2]
#     rr[1] -= yp[1]
#     rr[3] = yy[1] + yy[2] + yy[3] - 1.0
#     return Sundials.IDA_SUCCESS
# end

## Define the system residual function.
# function resrob(tres, yy_nv, yp_nv, rr_nv, user_data)
#     yy = convert(Vector, yy_nv)
#     yp = convert(Vector, yp_nv)
#     rr = convert(Vector, rr_nv)
#     rr[1] = yp[1] + yy[1]
#     rr[2] = yy[2] - yy[1]
#     return Sundials.IDA_SUCCESS
# end

# function resrob(tres, yy_nv, yp_nv, rr_nv, user_data)
#     yy = convert(Vector, yy_nv)
#     yp = convert(Vector, yp_nv)
#     rr = convert(Vector, rr_nv)
#     rr[1] = yp[1] + yy[1]
#     rr[2] = yy[2] - yy[1]
#     m = 0.3
#     L = 6.25
#     k = 6.25
#     g = 9.8
#
#     rr[1] = yp[1] - yy[4] + 2yp[6]*yy[1]
#     rr[2] = yp[2] - yy[5] + 2yp[6]*yy[2]
#     rr[3] = m*yp[4] - yp[3]*yy[1] + k*abs(yy[4])*yy[4]# - ut[1] - wt[1]^2
#     rr[4] = m*yp[5] - yp[3]*yy[2] + k*abs(yy[5])*yy[5] + m*g
#     rr[5] = yy[1]^2 + yy[2]^2 - L^2
#     rr[6] = yy[4]*yy[1] + yy[5]*yy[2]
#     # Equation for obtaining angle
#     rr[7] = yy[7] - atan(yy[1] / -yy[2])
#     return Sundials.IDA_SUCCESS
# end

function resrob(tres, yy_nv, yp_nv, rr_nv, user_data)
    yy = convert(Vector, yy_nv)
    yp = convert(Vector, yp_nv)
    rr = convert(Vector, rr_nv)
    rr[1] = yp[1] + yy[1]
    rr[2] = yy[2] - yy[1]
    m = 0.3
    L = 6.25
    k = 6.25
    g = 9.8

    rr[1] = yp[1] - yy[3]
    rr[2] = yp[2] - yy[4]
    rr[3] = m*yp[3] + k*abs(yy[3])*yy[3]# - ut[1] - wt[1]^2
    rr[4] = m*yp[4] + k*abs(yy[4])*yy[4] + m*g
    return Sundials.IDA_SUCCESS
end

# ## Root function routine. Compute functions g_i(t,y) for i = 0,1.
# function grob(t, yy_nv, yp_nv, gout_ptr, user_data)
#     yy = convert(Vector, yy_nv)
#     gout = Sundials.asarray(gout_ptr, (2,))
#     gout[1] = yy[1] - 0.0001
#     gout[2] = yy[3] - 0.01
#     return Sundials.IDA_SUCCESS
# end

## Define the Jacobian function. BROKEN - JJ is wrong
function jacrob(Neq, tt, cj, yy, yp, resvec, JJ, user_data, tempv1, tempv2, tempv3)
    JJ = pointer_to_array(convert(Ptr{Float64}, JJ), (3, 3))
    JJ[1, 1] = -0.04 - cj
    JJ[2, 1] = 0.04
    JJ[3, 1] = 1.0
    JJ[1, 2] = 1.0e4 * yy[3]
    JJ[2, 2] = -1.0e4 * yy[3] - 6.0e7 * yy[2] - cj
    JJ[3, 2] = 1.0
    JJ[1, 3] = 1.0e4 * yy[2]
    JJ[2, 3] = -1.0e4 * yy[2]
    JJ[3, 3] = 1.0
    return Sundials.IDA_SUCCESS
end


m = 0.3
L = 6.25
k = 6.25
g = 9.8
Φ = 0.3#pi/8
u0 = 0#u(0.0)[1]
w0 = 0#w(0.0)[1]

# FOR PARTIAL PENDULUM
y1 = L*sin(Φ)
y2 = -L*cos(Φ)
x0 = [y1, y2, 0, 0]
xp0 = [0, 0, 0, -g]

# # FOR COMPLETE PENDULUM
# x1_0 = L * sin(Φ)
# x2_0 = -L * cos(Φ)
# dx3_0 = m*g/x2_0
# dx4_0 = -g*tan(Φ) + (u0 + w0^2)/m
# x0 = vcat([x1_0, x2_0], zeros(4), [atan(x1_0 / -x2_0)])
# xp0 = vcat([0., 0., dx3_0, dx4_0], zeros(3))

neq = 4#7
nout = 2000
t0 = 0.0
yy0 = x0
yp0 = xp0
rtol = 1e-4
avtol = [1e-8, 1e-14, 1e-6]
tout1 = 0.4

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

iout = 0
δ = 1e-3
tout = δ#0.01#tout1
tret = [1.0]

sol_mat = NaN*ones(nout, neq)

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
        sol_mat[iout, :] = yy
        # global tout *= 10.0
        global tout += δ
    end
end

y4 = sol_mat[:,4]
yp4 = zeros(length(y4))
for i=1:length(y4)
    if i == 1
        yp4[i] = (y4[i]-yy0[4])/δ
    else
        yp4[i] = (y4[i]-y4[i-1])/δ
    end
end
# WE CLEARLY SEE THAT THE EQUATIONS ARE NOT SATISFIED!!!!!! THE LAST EQUATION AT LEAST
# WHY DO WE GET SUCH INACCURATE SOLUTIONS? AND WHY IS IT 50-50 IF IT EVEN CONVERGES???
# DID I FORGET TO CLEAR SOME MEMORY OR SOMETHING?
# println(sol_mat)

Sundials.SUNLinSolFree_Dense(LS)
Sundials.SUNMatDestroy_Dense(A)
