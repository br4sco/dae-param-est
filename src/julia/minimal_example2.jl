using Sundials, DifferentialEquations

# ======================= Defining useful variables ============================
N = 2000    # Number of samples of output computed
Ts = 0.01   # Time between samples

times = 0:Ts:((N+1)*Ts) # +1 just so we don't get overflow in our while loop

# ===========  Functions and structs for using high-level interface ============
struct Model
  f!::Function                  # residual function
  jac!::Function               # jacobian function
  x0::Array{Float64, 1}         # initial values of x
  xp0::Array{Float64, 1}        # initial values of x'
  dvars::Array{Bool, 1}         # bool array indicating differential variables
  ic_check::Array{Float64, 1}   # residual at t0 (DEBUG)
end

function dns_model()::Model
    # the residual function
    function f!(res, xp, x, θ, t)
        res[1] = -0.04 * x[1] + 1.0e4 * x[2] * x[3]
        res[2] = -res[1] - 3.0e7 * x[2] * x[2] - xp[2]
        res[1] -= xp[1]
        res[3] = x[1] + x[2] + x[3] - 1.0
        nothing
    end

    # Setting consistent initial conditions
    x0 = [1.0, 0.0, 0.0]
    xp0 = [-0.04, 0.04, 0.0]

    dvars = [true, true, false]

    r0 = zeros(length(x0))
    f!(r0, xp0, x0, [], 0.0)

    # t -> 0.0 is just a dummy function, not to be used
    Model(f!, t -> 0.0, x0, xp0, dvars, r0)
end

const abstol = 1e-8
const reltol = 1e-5
const maxiters = Int64(1e8)
# =========== Functions and variables for using low level interface ============

function resrob(tres, yy_nv, yp_nv, rr_nv, user_data)
    yy = convert(Vector, yy_nv)
    yp = convert(Vector, yp_nv)
    rr = convert(Vector, rr_nv)
    rr[1] = -0.04 * yy[1] + 1.0e4 * yy[2] * yy[3]
    rr[2] = -rr[1] - 3.0e7 * yy[2] * yy[2] - yp[2]
    rr[1] -= yp[1]
    rr[3] = yy[1] + yy[2] + yy[3] - 1.0
    return Sundials.IDA_SUCCESS
end

t0 = 0.0
yy0 = [1.0, 0.0, 0.0]
yp0 = [-0.04, 0.04, 0.0]
rtol = 1e-4
avtol = [1e-8, 1e-14, 1e-6]
tout1 = 0.4

# ============== Solving equations using low-level interface ===================

mem = Sundials.IDACreate()
Sundials.@checkflag Sundials.IDAInit(mem, resrob, t0, yy0, yp0)
Sundials.@checkflag Sundials.IDASVtolerances(mem, rtol, avtol)

## Call IDADense and set up the linear solver.
A = Sundials.SUNDenseMatrix(length(yy0), length(yy0))
LS = Sundials.SUNLinSol_Dense(yy0, A)
Sundials.@checkflag Sundials.IDADlsSetLinearSolver(mem, LS, A)

iout = 0
tout = times[2]    # First sample must be after time=0
tret = [1.0]

# Matrix for storing solutions
sol_mat = zeros(N, length(yy0))

while iout < N
    yy = similar(yy0)   # MY: The similar() is an inbuilt function in julia which is used to create an uninitialized mutable array with the given element type and size, based upon the given source array.
    yp = similar(yp0)
    retval = Sundials.IDASolve(mem, tout, tret, yy, yp, Sundials.IDA_NORMAL)
    println("T=", tout, ", Y=", yy)
    if retval == Sundials.IDA_SUCCESS
        global iout += 1
        global tout = times[iout+2]
        sol_mat[iout,:] = yy
    end
end

Sundials.SUNLinSolFree_Dense(LS)
Sundials.SUNMatDestroy_Dense(A)


# ============= Solving equations using high-level interface ===================

mdl = dns_model()
prob = DAEProblem(mdl.f!, mdl.xp0, mdl.x0, (0, N*Ts), [], differential_vars=mdl.dvars)

sol = solve(
  prob,
  saveat = times[1:end-1],
  abstol = abstol,
  reltol = reltol,
  maxiters = maxiters
)

# ============= CHECKING RESIDUALS OF SOLUTIONS ==================
# This section can safely be ignored, it's only providing helpful tools for
# realizing that the low-level and high-level solutions are both correct

function res(x, xp)
    res = zeros(length(x))
    res[1] = -0.04 * x[1] + 1.0e4 * x[2] * x[3]
    res[2] = -res[1] - 3.0e7 * x[2] * x[2] - xp[2]
    res[1] -= xp[1]
    res[3] = x[1] + x[2] + x[3] - 1.0
    return res
end

function get_derivative_estimate(vec, yp0, h)
    dev = zeros(length(vec))
    for i=1:length(vec)
        if i==1
            dev[i] = yp0#(vec[1]-y0)/h
        else
            dev[i] = (vec[i]-vec[i-1])/h
        end
    end
    return dev
end

low1 = vcat([yy0[1]], sol_mat[:,1]);
low2 = vcat([yy0[2]], sol_mat[:,2]);
low3 = vcat([yy0[3]], sol_mat[:,3]);
dlow1 = get_derivative_estimate(low1, yp0[1], Ts)
dlow2 = get_derivative_estimate(low2, yp0[2], Ts)
dlow3 = get_derivative_estimate(low3, yp0[3], Ts)
high1 = [sol.u[i][1] for i=1:length(sol.u)]
high2 = [sol.u[i][2] for i=1:length(sol.u)]
high3 = [sol.u[i][3] for i=1:length(sol.u)]
dhigh1 = get_derivative_estimate(high1, yp0[1], Ts)
dhigh2 = get_derivative_estimate(high2, yp0[2], Ts)
dhigh3 = get_derivative_estimate(high3, yp0[3], Ts)

# Stores residual values for low-level method
lowres = zeros(length(low3), 3)
# Stores residual values for high-level method
highres = zeros(length(low3), 3)

for i=1:length(low3)
    yylow = [low1[i], low2[i], low3[i]]
    yyhigh = [high1[i], high2[i], high3[i]]
    yplow = [dlow1[i], dlow2[i], dlow3[i]]
    yphigh = [dhigh1[i], dhigh2[i], dhigh3[i]]
    lowres[i,:] = res(yylow, yplow)
    highres[i,:] = res(yyhigh, yphigh)
end
