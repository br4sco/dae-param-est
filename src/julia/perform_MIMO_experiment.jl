using Dates
include("MIMO_experiment.jl")

function write_Y_and_metadata(Y)
    datetime_str = replace(string(now()), ":" => "_")
    metadata = "θ0: $(string(mk_θs(θ0)))
η0: $(string(η0))
nx: $nx
n_out: $n_out
n_in: $n_in
n_u: $n_u
T: $T
δ: $δ_min
Nw: $Nw_max
N: $N
Ts: $Ts
W: $W
Q: $Q"
    p_Y = joinpath(data_dir, ("Y"*datetime_str)*".csv")
    p_metadata = joinpath(data_dir, ("Y_metadata"*datetime_str)*".txt")
    writedlm(p_Y, Y, ",")
    file_meta = open(p_metadata, "w")
    write(file_meta, metadata)
    close(file_meta)
end

load_Y = true
# Y_filename = "Y2021-09-16T18_27_51.584.csv"         # Large dataset
# Y_filename = "Y2021-09-29T17_32_17.022.csv"         # Medium dataset
# Y_filename = "Y2021-09-30T08_59_05.213.csv"         # Medium dataset 2
Y_filename = "Y2021-09-29T17_27_35.631.csv"       # Small dataset

try
    if load_Y
        local p = joinpath(data_dir, Y_filename)
        println("Loading Y...")
        Y = readdlm(p, ',')
        if size(Y,1) != N+1
            throw(DimensionMismatch("N is set to $N, but length of loaded data Y was $(size(Y,1)) (should be $(N+1))"))
        end
        println("Successfully loaded Y")
    else
        println("Generating Y...")
        Y = calc_Y()
        try
            write_Y_and_metadata(Y)
            println("Stored Y and metadata")
        catch
            println("Failed storing Y and metadata")
        end
    end
    perform_experiments(Y, vcat(θ0, w_scale))
    println("Finished performing experiments")
catch e
    # print(sprint(showerror, e, catch_backtrace()))
    local p = joinpath(data_dir, "error_msg.txt")
    file = open(p, "w")
    write(file, sprint(showerror, e, catch_backtrace()))
    close(file)
    throw(e)
end
