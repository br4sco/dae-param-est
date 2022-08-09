using Sundials
include("simulation.jl")

const abstol = 1e-8#1e-9
const reltol = 1e-5#1e-6
const maxiters = Int64(1e8)

function get_derivative_estimate(vec, y0, h)
    dev = zeros(length(vec))
    for i=1:length(vec)
        if i==1
            dev[i] = (vec[1]-y0)/h
        else
            dev[i] = (vec[i]-vec[i-1])/h
        end
    end
    return dev
end

## Define the system residual function.
# neq = 3
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

L = 6.25
m = 0.3
k = 6.25
g = 9.81
neq = 4
function res(yy, yp)
    res = zeros(length(yy))
    res[1] = yp[1] - yy[3]
    res[2] = yp[2] - yy[4]
    res[3] = m*yp[3] + k*abs(yy[3])*yy[3]
    res[4] = m*yp[4] + k*abs(yy[4])*yy[4]
    return res
end
function resrob(tres, yy_nv, yp_nv, rr_nv, user_data)
    yy = convert(Vector, yy_nv)
    yp = convert(Vector, yp_nv)
    rr = convert(Vector, rr_nv)
    temp = res(yy, yp)
    rr[1] = temp[1]
    rr[2] = temp[2]
    rr[3] = temp[3]
    rr[4] = temp[4]
    return Sundials.IDA_SUCCESS
end

## Root function routine. Compute functions g_i(t,y) for i = 0,1.
function grob(t, yy_nv, yp_nv, gout_ptr, user_data)
    yy = convert(Vector, yy_nv)
    gout = Sundials.asarray(gout_ptr, (2,))
    gout[1] = yy[1] - 0.0001
    gout[2] = yy[3] - 0.01
    return Sundials.IDA_SUCCESS
end

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

nout = 12
t0 = 0.0
# # For DNS origninal
# yy0 = [1.0, 0.0, 0.0]
# yp0 = [-0.04, 0.04, 0.0]
# For simplified pendulum
Φ = pi/4
yy0 = [L*sin(Φ), -L*cos(Φ), 0, 0]
yp0 = [0, 0, 0, -g]

# # FOR HIGH LEVEL, WE USE
# const abstol = 1e-8
# const reltol = 1e-5
# # MAYBE THAT CAUSES A DIFFERENCE? NOO, PROBLEM SEEMS TO PERSIST...

rtol = 1e-4
avtol = [1e-8, 1e-14, 1e-6]
tout1 = 0.4

mem = Sundials.IDACreate()
Sundials.@checkflag Sundials.IDAInit(mem, resrob, t0, yy0, yp0)
Sundials.@checkflag Sundials.IDASVtolerances(mem, rtol, avtol)

## Call IDARootInit to specify the root function grob with 2 components
Sundials.@checkflag Sundials.IDARootInit(mem, 2, grob)

## Call IDADense and set up the linear solver.
# A = Sundials.SUNDenseMatrix(length(y0), length(y0))
# LS = Sundials.SUNLinSol_Dense(y0, A)
A = Sundials.SUNDenseMatrix(length(yy0), length(yy0))   # MY OWN TRY
LS = Sundials.SUNLinSol_Dense(yy0, A)                   # MY OWN TRY
Sundials.@checkflag Sundials.IDADlsSetLinearSolver(mem, LS, A)

N = 2000
Ts = 0.01

times = 0:Ts:((N+1)*Ts) # +1 just so we don't get overflow in line 89

iout = 0
tout = times[iout+2]#tout1
tret = [1.0]

sol_mat = zeros(N, length(yy0))

while iout < N
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
        global tout = times[iout+2]
        sol_mat[iout,:] = yy
    end
    # println("sum: $(sum(yy))")
end

Sundials.SUNLinSolFree_Dense(LS)
Sundials.SUNMatDestroy_Dense(A)



# ------------------------ HIGH LEVEL VERSION -----------------------

realize_model() = problem(
    Roberts_dns_simple_pend(),
    N,
    Ts,
)

# times = [0.4, 4.0, 40, 400, 4000, 40000, 400000, 4e6, 4e7, 4e8, 4e9, 4e10];

sol = solve(
  realize_model(),
  saveat = 0:Ts:(N*Ts), # times,
  abstol = abstol,
  reltol = reltol,
  maxiters = maxiters
)


low1 = vcat([yy0[1]], sol_mat[:,1]);
low2 = vcat([yy0[2]], sol_mat[:,2]);
low3 = vcat([yy0[3]], sol_mat[:,3]);
dlow1 = get_derivative_estimate(low1, yy0[1], Ts)
dlow2 = get_derivative_estimate(low2, yy0[2], Ts)
dlow3 = get_derivative_estimate(low3, yy0[3], Ts)
high1 = [sol.u[i][1] for i=1:length(sol.u)]
high2 = [sol.u[i][2] for i=1:length(sol.u)]
high3 = [sol.u[i][3] for i=1:length(sol.u)]
dhigh1 = get_derivative_estimate(high1, yy0[1], Ts)
dhigh2 = get_derivative_estimate(high2, yy0[2], Ts)
dhigh3 = get_derivative_estimate(high3, yy0[3], Ts)

if neq > 3
    low4 = vcat([yy0[4]], sol_mat[:,4]);
    high4 = [sol.u[i][4] for i=1:length(sol.u)]
    dlow4 = get_derivative_estimate(low4, yy0[4], Ts)
    dhigh4 = get_derivative_estimate(high4, yy0[4], Ts)

    lowres = zeros(length(low4), neq)
    highres = zeros(length(low4), neq)
    for i=1:length(low4)
        yylow = [low1[i], low2[i], low3[i], low4[i]]
        yyhigh = [high1[i], high2[i], high3[i], high4[i]]
        yplow = [dlow1[i], dlow2[i], dlow3[i], dlow4[i]]
        yphigh = [dhigh1[i], dhigh2[i], dhigh3[i], dhigh4[i]]
        lowres[i,:] = res(yylow, yplow)
        highres[i,:] = res(yyhigh, yphigh)
    end
end
