import DifferentialEquations
import Sundials
import ProgressMeter
include("models.jl")
using .DynamicalModels: Model, Model_ode
# include("Model_type.jl")

# function problem(m::Model, N::Int, Ts::Float64)
#   T = N * Ts
#   # ff = DAEFunction(m.f!, jac = m.jac!)
#   # ff = DAEFunction{true,true}(m.f!)
#   DAEProblem(m.f!, m.dx0, m.x0, (0, T), [], differential_vars=m.dvars)
# end

function problem_reverse(m::Model, N::Int, Ts::Float64)
    T = N * Ts
    # ff = DAEFunction(m.f!, jac = m.jac!)
    # ff = DAEFunction{true,true}(m.f!)
    DAEProblem(m.f!, m.dx0, m.x0, (T, 0), [], differential_vars=m.dvars)
end

function problem_ode(m::Model_ode, N::Int, Ts::Float64)
    T = N * Ts
    ODEProblem(m.f, m.x0, (0,T), [])
end

# function solve(prob; kwargs...)
#   DifferentialEquations.solve(prob, IDA(); kwargs...)
# end

# If these principles are not achieved yet, then that should be fixed
####################################################################
### DESIGN PRINCIPLES BEHIND apply_outputfun AND solve-FUNCTIONS ###
####################################################################
# sol.u is a vector of states (which are themselves vectors)
# After applying output function, we return a vector of outputs (i.e. a vector of vector or a vector of numbers, depending on dimensionality of output)
# After applying sensitivity function, we return a vector of matrices. The rows of the matrices represent different output components, and the columns represent different parameters

# in solve_in_parallel-functions:
# we apply vcat(*...) on the output vector of vectors/numbers. Regardless of if it's vectors or scalars on the inside, this operation will return a vector. For mutlidimensional outputs, 
#       all components of the output for a given time will be stacked next to each other.
# we apply vcat(*...) on the sensitivity vector of matrices. The operation returns a matrix, where rows correspond to time/output component and columns correspond to different parameters. 
#       For the rows, all components of the output for a given time will be stacked next to each other.
####################################################################

function apply_outputfun(h, sol)
    if sol.retcode != :Success
        throw(ErrorException("Solution retcode: $(sol.retcode)"))   # Can change this into a println() instead of a throw() to only print error message without crashing
        # println(ErrorException("Solution retcode: $(sol.retcode)"))
    end
    # NOTE: There are alternative, more recommended, ways of accessing solution
    # than through sol.u: https://diffeq.sciml.ai/stable/basics/solution/
    map(h, sol.u)
end

function apply_two_outputfun(h1, h2, sol)
    if sol.retcode != :Success
        throw(ErrorException("Solution retcode: $(sol.retcode)"))
    end
    # NOTE: There are alternative, more recommended, ways of accessing solution
    # than through sol.u: https://diffeq.sciml.ai/stable/basics/solution/
    map(h1, sol.u), map(h2, sol.u)
end

# This one I use for sure, I'm not so sure about the others
function solve_in_parallel(solve_func, is)
    M = length(is)
    p = ProgressMeter.Progress(M, 1, "Running $(M) simulations...", 50)
    y1 = solve_func(is[1])
    ny = length(y1[1])
    Y = zeros(ny*length(y1), M)
    Y[:, 1] += vcat(y1...)
    ProgressMeter.next!(p)
    Threads.@threads for m = 2:M
        y = solve_func(is[m])
        Y[:, m] += vcat(y...)
        ProgressMeter.next!(p)
    end
    Y
end

# More efficient than original, mostly relevant for small M
function solve_in_parallel2(solve_func, is, ny, N)
    M = length(is)
    p = Progress(M, 1, "Running $(M) simulations...", 50)
    # y1 = solve_func(is[1])
    # ny = length(y1[1])
    # Y = zeros(ny*length(y1), M)
    Y = zeros(ny*N, M)
    # Y[:, 1] += vcat(y1...)
    # next!(p)
    Threads.@threads for m = 1:M
        y = solve_func(is[m])
        Y[:, m] += vcat(y...)
        next!(p)
    end
    Y
end

# Returns matrix, with np rows and M columns
function solve_adj_in_parallel(solve_func, is)
  M = length(is)
  p = Progress(M, 1, "Running $(M) simulations...", 50)
  Gp1 = solve_func(is[1])
  Gps = zeros(length(Gp1), M)
  Gps[:, 1] .+= Gp1
  next!(p)
  Threads.@threads for m = 2:M
      Gp = solve_func(is[m])
      Gps[:, m] .+= Gp
      next!(p)
  end
  Gps
end

# Returns matrix, with np rows and M columns
# More efficient than original, mostly relevant for small M
function solve_adj_in_parallel2(solve_func, is, np)
    M = length(is)
    p = Progress(M, 1, "Running $(M) simulations...", 50)
    # Gp1 = solve_func(is[1])
    # Gps = zeros(length(Gp1), M)
    Gps = zeros(np, M)
    # Gps[:, 1] .+= Gp1
    # next!(p)
    Threads.@threads for m = 1:M
        Gp = solve_func(is[m])
        Gps[:, m] .+= Gp
        next!(p)
    end
    Gps
end

# Handles multivariate outputs by flattening the output
function solve_in_parallel_sens(solve_func, is)
    M = length(is)
    p = ProgressMeter.Progress(M, 1, "Running $(M) simulations...", 50)
    y1, sens1 = solve_func(is[1])
    ny = length(y1[1])
    Ysens = [Matrix{Float64}(undef, ny*length(sens1), length(sens1[1])÷ny) for m=1:M]
    Y = zeros(ny*length(y1), M)
    Y[:,1] += vcat(y1...)   # Flattens the array
    Ysens[1] = vcat(sens1...)
    ProgressMeter.next!(p)
    Threads.@threads for m = 2:M
        y, sens = solve_func(is[m])
        Y[:,m] += vcat(y...)   # Flattens the array
        Ysens[m] = vcat(sens...)
        ProgressMeter.next!(p)
    end
    Y, Ysens
end

# Handles multivariate outputs by flattening the output
# More efficient than original, mostly relevant for small M
function solve_in_parallel_sens2(solve_func, is, ny, np, N)
    M = length(is)
    p = Progress(M, 1, "Running $(M) simulations...", 50)
    # y1, sens1 = solve_func(is[1])
    # ny = length(y1[1])
    # Ysens = [Matrix{Float64}(undef, ny*length(sens1), length(sens1[1])÷ny) for m=1:M]
    # Y = zeros(ny*length(y1), M)
    Ysens = [Matrix{Float64}(undef, ny*(N+1), np) for m=1:M]
    Y = zeros(ny*(N+1), M)
    # Y[:,1] += vcat(y1...)   # Flattens the array
    # Ysens[1] = vcat(sens1...)
    # next!(p)
    Threads.@threads for m = 1:M
        y, sens = solve_func(is[m])
        Y[:,m] += vcat(y...)   # Flattens the array
        Ysens[m] = vcat(sens...)
        next!(p)
    end
    Y, Ysens
end

# Could the uses of this be replace by just solve_in_parallel? That would be nice
function solve_in_parallel_sens_debug(solve_func, is, yind, sensinds, sampling_ratio)
    M = length(is)
    p = Progress(M, 1, "Running $(M) simulations...", 50)
    y1 = solve_func(is[1])
    ny = length(y1[1])

    # Rows of inner matrix are assumed to be different values of t, columns of
    # matrix are assumed to be different elements of the vector-valued process

    # Inner matrix is size of state vector times number of time samples. Or the other way around....?
    matout = [Matrix{Float64}(undef, length(y1), ny) for m=1:M]
    Y = zeros(length(y1[1:sampling_ratio:end,:]), M)
    sens = [zeros(length(y1[1:sampling_ratio:end,:]), length(sensinds)) for m=1:M]
    matout[1] = transpose(hcat(y1...))     # TODO: Would using matout[1][:,:] = transpose(hcat(y1...)) instead avoid allocating new memory for matout[1]?
    Y[:,1]    = matout[1][1:sampling_ratio:end, yind]
    sens[1]   = matout[1][1:sampling_ratio:end, sensinds]
    next!(p)
    Threads.@threads for m = 2:M
        y1 = solve_func(is[m])
        matout[m] = transpose(hcat(y1...))
        Y[:,m]  = matout[m][1:sampling_ratio:end, yind]
        sens[m] = matout[m][1:sampling_ratio:end, sensinds]
        next!(p)
    end
    matout, Y, sens
end

# NOTE: This has a very similar structure to solve_in_parallel_sens_debug, but implemented a little differently. Might be worth at some occation
# to evaluate which function is better written and make sure to write the two in a similar way
function solve_in_parallel_debug(solve_func, is, yind, sampling_ratio)
    M = length(is)
    p = Progress(M, 1, "Running $(M) simulations...", 50)
    # the rows of out correspond to different times, and the columns to different state components
    out = transpose(hcat(solve_func(is[1])...))
    matout = [Matrix{Float64}(undef, size(out)) for m=1:M]
    Y = zeros(size(out[1:sampling_ratio:end,:], 1), M)
    matout[1] = out
    Y[:,1]    = out[1:sampling_ratio:end, yind]
    next!(p)
    Threads.@threads for m = 2:M
        out = transpose(hcat(solve_func(is[m])...))
        matout[m] = out
        Y[:,m]  = out[1:sampling_ratio:end, yind]
        next!(p)
    end
    matout, Y
end

# NOTE: This has a very similar structure to solve_in_parallel_sens_debug, but implemented a little differently. Might be worth at some occation
# to evaluate which function is better written and make sure to write the two in a similar way
# More efficient than original, mostly relevant for small M
function solve_in_parallel_debug2(solve_func, is, yind, sampling_ratio, ny, N)
    M = length(is)
    p = Progress(M, 1, "Running $(M) simulations...", 50)
    # the rows of out correspond to different times, and the columns to different state components
    # out = transpose(hcat(solve_func(is[1])...))
    # matout = [Matrix{Float64}(undef, size(out)) for m=1:M]
    matout = [Matrix{Float64}(undef, N, ny) for m=1:M]
    # Y = zeros(size(out[1:sampling_ratio:end,:], 1), M)  
    Y = zeros(N÷sampling_ratio+1, M)
    # out has N rows, then we downsample with sampling_ratio. We should get N/sampling_ration rows, but what if that isn't an integer?
    # Do we round up or down? Assume every 10th sample. Less than 10 samples, we have 1 element, 10-19, 2 elements, 20-29 3 elements
    # So it should be N/sampling_ratio rounded down + 1. ÷ truncates to an integer, i.e. rounds down, so let's just use that!
    # matout[1] = out
    # Y[:,1]    = out[1:sampling_ratio:end, yind]
    # next!(p)
    Threads.@threads for m = 1:M
        out = transpose(hcat(solve_func(is[m])...))
        matout[m] = out
        Y[:,m]  = out[1:sampling_ratio:end, yind]
        next!(p)
    end
    matout, Y
end