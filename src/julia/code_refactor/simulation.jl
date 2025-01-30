import DifferentialEquations as DE
import Sundials
import ProgressMeter

function problem_reverse(m::Model, N::Int, Ts::Float64)::DAEProblem
    T = N * Ts
    # ff = DAEFunction(m.f!, jac = m.jac!)
    # ff = DAEFunction{true,true}(m.f!)
    DAEProblem(m.f!, m.dx0, m.x0, (T, 0), [], differential_vars=m.dvars)
end

function problem_ode(m::Model_ode, N::Int, Ts::Float64)::ODEProblem
    T = N * Ts
    ODEProblem(m.f, m.x0, (0,T), [])
end


# TODO: Check how we're doing with these principles
# If these principles are not achieved yet, then that should be fixed
####################################################################
### DESIGN PRINCIPLES BEHIND apply_outputfun AND solve-FUNCTIONS ###
####################################################################
# sol.u is a vector of states (which are themselves vectors)
# After applying output function, we return a vector of outputs (i.e. a vector of vectors or a vector of numbers, depending on dimensionality of output)
# After applying sensitivity function, we return a vector of matrices. The rows of the matrices represent different output components, and the columns represent different parameters

# in solve_in_parallel-functions:
# we apply vcat(*...) on the output vector of vectors/numbers. Regardless of if it's vectors or scalars on the inside, this operation will return a vector. For mutlidimensional outputs, 
#       all components of the output for a given time will be stacked next to each other.
# we apply vcat(*...) on the sensitivity vector of matrices. The operation returns a matrix, where rows correspond to time/output component and columns correspond to different parameters. 
#       For the rows, all components of the output for a given time will be stacked next to each other.
####################################################################

function apply_outputfun(h::Function, sol::DE.DAESolution)::Vector  # Type of element of output depends on the type of output sample, could be a vector or a scalar
    if sol.retcode != :Success
        throw(ErrorException("Solution retcode: $(sol.retcode)"))   # Can change this into a println() instead of a throw() to only print error message without crashing
        # println(ErrorException("Solution retcode: $(sol.retcode)"))
    end
    # NOTE: There are alternative, more recommended, ways of accessing solution
    # than through sol.u: https://diffeq.sciml.ai/stable/basics/solution/
    map(h, sol.u)
end

function apply_two_outputfun(h1::Function, h2::Function, sol::DE.DAESolution)
    if sol.retcode != :Success
        throw(ErrorException("Solution retcode: $(sol.retcode)"))
    end
    # NOTE: There are alternative, more recommended, ways of accessing solution
    # than through sol.u: https://diffeq.sciml.ai/stable/basics/solution/
    map(h1, sol.u), map(h2, sol.u)
end

function solve_in_parallel(solve_func::Function, is::UnitRange{Int64}, ny::Int64, N::Int64)::Matrix{Float64}
    M = length(is)
    p = ProgressMeter.Progress(M, 1, "Running $(M) simulations...", 50)
    Y = zeros(ny*N, M)
    Threads.@threads for m = 1:M
        y = solve_func(is[m])
        Y[:, m] += vcat(y...)
        ProgressMeter.next!(p)
    end
    Y
end

# Handles multivariate outputs by flattening the output
function solve_in_parallel_sens(solve_func::Function, is::UnitRange{Int64}, ny::Int64, nθ::Int64, N::Int64)::Tuple{Matrix{Float64}, Vector{Matrix{Float64}}}
    M = length(is)
    p = ProgressMeter.Progress(M, 1, "Running $(M) simulations...", 50)
    Ysens = [Matrix{Float64}(undef, ny*(N+1), nθ) for _=1:M]
    Y = zeros(ny*(N+1), M)
    Threads.@threads for m = 1:M
        y, sens = solve_func(is[m])
        Y[:,m] += vcat(y...)   # Flattens the array
        Ysens[m] = vcat(sens...)
        ProgressMeter.next!(p)
    end
    Y, Ysens
end

# Treats the entire state as output
function solve_in_parallel_stateout(solve_func::Function, is::UnitRange{Int64}, ny::Int64, N::Int64)::Vector{Matrix{Float64}}
    M = length(is)
    p = ProgressMeter.Progress(M, 1, "Running $(M) simulations...", 50)
    # the rows of out correspond to different times, and the columns to different state components
    # out = transpose(hcat(solve_func(is[1])...))
    matout = [Matrix{Float64}(undef, N, ny) for m=1:M]
    Threads.@threads for m = 1:M        # TODO: I should be able to iterate m=is, instead of m=1:M, no?
        matout[m] = transpose(hcat(solve_func(is[m])...))
        ProgressMeter.next!(p)
    end
    matout
end

# Returns matrix, with np rows and M columns
function solve_adj_in_parallel(solve_func::Function, is::UnitRange{Int64}, np::Int64)::Matrix{Float64}
    M = length(is)
    p = ProgressMeter.Progress(M, 1, "Running $(M) simulations...", 50)
    # Gθ1 = solve_func(is[1])
    # Gθs = zeros(length(Gθ1), M)
    Gθs = zeros(np, M)
    # Gθs[:, 1] .+= Gθ1
    # next!(p)
    Threads.@threads for m = 1:M
        Gθ = solve_func(is[m])
        Gθs[:, m] .+= Gθ
        ProgressMeter.next!(p)
    end
    Gθs
end