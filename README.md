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
2. Metadata corresponding to the above noise matrix
3. A `Nw✕E` noise matrix representing the process disturbance, where `E` is the
   number of used data-sets
4. Metadata corresponding to the above noise matrix


The parameter `Nw` defines the number of steps of the noise sequence, and which has to be long enough for running the simulation.  Specifically, `Nw*δ > N*Ts` must hold, where `δ` is the sampling time of the noise generation, `N` and `Ts` is the number of steps and sampling time of the simulation, respectively.

To use the same disturbance model as described in the paper, in a julia repl in [src/julia](src/julia) run:

```{julia}
using DelimitedFiles, CSV
include("noise_generation.jl")

XW, W, meta_W = get_filtered_noise(disturbance_model_3, δ, E, Nw, scale=0.6)  # Generated process disturbance and meta-data
XU, U, meta_U = get_filtered_noise(disturbance_model_3, δ, 1, Nw, scale=0.2)  # Generates control input and meta-data
writedlm("data/experiments/expid/XW.csv", XW, ',')
writedlm("data/experiments/expid/U.csv", U, ',')
CSV.write("data/experiments/expid/meta_W.csv", meta_W)
CSV.write("data/experiments/expid/meta_U.csv", meta_U)
```
This stores the necessary files in ```src/julia/experiments/expid```. For the experiments in the paper `δ = 0.01`, `E=100`, and `Nw=10*N` was used.

### Run experiment
The simples way to perform an experiment is to do
```
julia --threads n
```

in [src/julia](src/julia) to open the julia repl. Then run

```{julia}
include("example.jl")
oe_out, prop_out = get_estimates(expid)
```

where ```expid``` is the name (String) of the directory containing the experiment data. Note that this code might take some time because it performs 100 Monte-Carlo experiments to estimate the statistics of the different estimators. ```oe_out``` is a tuple containing three arrays, where the i:th array contains all 100 estimates of the i:th parameter obtained using the output error method neglecting process disturbances. ```prop_out``` similarly contains the corresponding estimates obtained from the proposed method. To visualize the results using box-plots, for the first parameters, you can call

```{julia}
thetahat_boxplots(oe_out[1], prop_out[1], Ns)
```

where ```Ns``` is an array with the number of data-samples N used in the experiment ```expid``` (it's only used for labeling of the axes). If you want to compare the results for different values of N, you can visualize the results from several experiments as follows:

```{julia}
include("example.jl")
oe_out1, prop_out1 = get_estimates(expid1)
oe_out2, prop_out2 = get_estimates(expid2)
oe_out3, prop_out3 = get_estimates(expid3)

i = 1
oe_pars = hcat(oe_out1[i], oe_out2[i], oe_out3[i])
prop_pars = hcat(prop_out1[i], prop_out2[i], prop_out3[i])
thetahat_boxplots(oe_pars, prop_pars)
```

where you can change the value of ```i``` to visualize the result for a different parameter.
