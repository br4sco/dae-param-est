# dae-param-est
This repository contains the code to the numerical experiment discussed in
(TO BE ADDED, SUBMITTED TO CDC 2022), which proposes a method for consistent parameter
estimation using stochastic approximation for Differential Algebraic Equations (DAEs) subject to process
disturbances.


## Prerequisites
The experiment is implemented in [julia](https://docs.julialang.org/en/v1/), see
their homepage for installations instructions for your platform.

You will need to install some additional julia packages. The julia repl
(`julia`) will notify you which packages are missing when you include files
(i.e. `include("<file>.jl")`) and how to install them. To start julia with
multiple threads (recommended), start julia with the following command `julia
--threads n`, where `n` is the number of threads you want to use.

## Run the experiment
The main experiment is defined in
[src/julia/run_experiment.jl](src/julia/run_experiment.jl). Additionally, the
noise model and noise generation is defined in
[src/julia/noise_generation.jl](src/julia/noise_generation.jl). The simulation scripts and
physical model is defined in [src/julia/simulation.jl](src/julia/simulation.jl).

### Download or generate noise
You can either generate the data yourself or download the data used in our experiment from
https://kth-my.sharepoint.com/:f:/g/personal/robbj_ug_kth_se/Eoy33BAr42JPnTzK1MGQ2qcBP2sjLWb-fs0cAFC2YrUK5Q?e=I5bAHp (This link is only valid for 180 days, if it doesn't work then please e-mail robbj@kth.se, since that means we have forgotten to update it). Place these files in
[src/julia/data/experiments](src/julia/data/experiments) (Note that these files are quite large). You can also generate noise yourself using
the functions defined in
[src/julia/noise_generation.jl](src/julia/noise_generation.jl). You should generate and
save four files in total:

1. A `Nw✕1` noise matrix serving as the input u(t).
2. Meta-data corresponding to the above noise matrix
3. A `Nw✕E` noise matrix representing the process disturbance, where `E` is the
   number of used data-sets
4. Meta-data corresponding to the above noise matrix


The parameter `Nw` defines the number of steps of the noise sequence, and which has to be long enough for running the simulation.  Specifically, `Nw*δ > N*Ts` must hold, where `δ` is the sampling time of the noise generation, `N` and `Ts` is the number of steps and sampling time of the simulation, respectively.

To use the same disturbance model as described in the paper, in a julia repl in [src/julia](src/julia) run:

```{julia}
using DelimitedFiles, CSV, DataFrames
include("noise_generation.jl")

XW, W, meta_W = get_filtered_noise(disturbance_model_3, δ, E, Nw, scale=0.6)  # Generated process disturbance and meta-data
XU, U, meta_U = get_filtered_noise(disturbance_model_3, δ, 1, Nw, scale=0.2)  # Generates control input and meta-data
meta_Y = DataFrame(Ts=0.1, N=500)	# It is convenient to also generate metadata for system output
writedlm("data/experiments/expid/XW_T.csv", transpose(XW), ',')
writedlm("data/experiments/expid/U.csv", U, ',')
CSV.write("data/experiments/expid/meta_W.csv", meta_W)
CSV.write("data/experiments/expid/meta_U.csv", meta_U)
CSV.write("data/experiments/expid/meta_Y.csv", meta_Y)
```
This stores the necessary files in ```src/julia/experiments/expid```. For the experiments in the paper `δ = 0.01`, `E=100`, and `Nw=10*N` was used.

### Run the experiment
To run the experiment do
```
julia --threads n
```

in [src/julia](src/julia) to open the julia repl.

Then, in the repl, include the experiment script

```{julia}
include("run_experiment.jl")
```
You will probably have to install a number of dependencies pointed out by julia.
After that, you can run the estimation experiment over the `E` data-sets found in the folder ```src/julia/experiments/expid```, by writing

```{julia}
opt_pars_baseline, opt_pars_proposed, avg_pars_proposed, trace_base, trace_proposed, trace_gradient, durations = get_estimates("expid", [0.5, 4.25, 4.25], 500)
```

The results can be interpreted as follows. ```opt_pars_baseline[i,e]``` is the optimal value of parameter `i` found by the output error method for the data-set `e`. Similarly, ```avg_pars_proposed[i,e]``` is the optimal value of parameter `i` found by the proposed method for the data-set `e`.
