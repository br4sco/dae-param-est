struct Model
  f!::Function                  # residual function
  jac!::Function               # jacobian function
  x0::Vector{Float64}         # initial values of x
  dx0::Vector{Float64}        # initial values of x'
  dvars::Array{Bool, 1}         # bool array indicating differential variables
  ic_check::Vector{Float64}   # residual at t0 (DEBUG)
end

struct Model_ode
    f::Function             # ODE Function
    x0::Vector{Float64}   # initial values of x
end