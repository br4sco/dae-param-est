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
δ: $(δ_min)
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

try
    Y = calc_Y()
    try
        write_Y_and_metadata(Y)
    catch
        print("Failed storing Y and metadata")
    end
    perform_experiments(Y, vcat(θ0, w_scale))
catch e
    print(e)
    p = joinpath(data_dir, "error_msg.txt")
    file = open(p, "w")
    write(file, sprint(showerror, e, catch_backtrace()))
    close(file)
end
