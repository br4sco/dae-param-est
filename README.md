# dae-param-est
A Parameter Estimation Experiment for DAEs  
For the code used in our [paper published at CDC 2021](https://www.diva-portal.org/smash/get/diva2:1539989/FULLTEXT01.pdf), please see the branch "cdc".  
For the code used in our [paper submitted to CDC 2022](https://kth.diva-portal.org/smash/get/diva2:1914575/FULLTEXT01.pdf), please see the branch "CDC22".\
This main branch contains the code for the [Licentiate thesis](https://kth.diva-portal.org/smash/get/diva2:1914575/FULLTEXT01.pdf) of Robert Bereza.


## Prerequisites
The experiment is implemented in [julia](https://docs.julialang.org/en/v1/), see
their homepage for installations instructions for your platform.

You will need to install some additional julia packages. The julia repl
(`julia`) will notify you which packages are missing when you include files
(i.e. `include("<file>.jl")`) and how to install them. To start julia with
multiple threads (recommended), start julia with the following command `julia
--threads n`, where `n` is the number of threads you want to use. We have used `n=8`.

## Run the experiment
The main experiment is defined in
[src/julia/run_experiment.jl](src/julia/run_experiment.jl). Additionally, the
noise model and noise generation is defined in
[src/julia/noise_generation.jl](src/julia/noise_generation.jl). The
physical model is defined in [src/julia/models.jl](src/julia/models.jl).

### Minimal example

Many experiment metaparameters have to be set in the code and are perhaps therefore not easy to edit. The current metaparameters are set so that it is straightforward to run a single experiment for identifying the parameter `k` of the pendulum model. Download the experiment data titled *500_u2w6_from_Alsvin* from [https://kth-my.sharepoint.com/:f:/g/personal/robbj_ug_kth_se/EqMa_VdRU89GtYkNFiPfe-sB6O4B19RKqXIo494jX9mcGA?e=wP89k9](https://kth-my.sharepoint.com/:f:/g/personal/robbj_ug_kth_se/EqMa_VdRU89GtYkNFiPfe-sB6O4B19RKqXIo494jX9mcGA?e=wP89k9) (This link is only valid for 180 days, if it doesn't work then please e-mail robbj@kth.se, since that means we have forgotten to update it). You can also generate your own noise using the instructions under the next heading. In that case, use `N=500`. Place the data in [src/julia/data/experiments](src/julia/data/experiments).

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
After that, you can run the estimation experiment over the data-set found in the folder ```src/julia/experiments/500_u2w6_from_Alsvin```, by writing

```{julia}
opt_pars_baseline, opt_pars_proposed, avg_pars_proposed, trace_base, trace_proposed, trace_gradient, durations = get_estimates("500_u2w6_from_Alsvin", [4.25], 0)
```
The experiment for the baseline method will run first, followed by the experiment for the proposed method. Only a single experiment is performed for each method, and the initial parameter estimate is set to `4.25`, while the true value is `6.25`.

### Generate noise
If you don't want to use the downloaded data, you can also generate noise yourself using
the functions defined in
[src/julia/noise_generation.jl](src/julia/noise_generation.jl). You should generate and
save four files in total:

1. A `Nw✕1` noise matrix serving as the input u(t).
2. Meta-data corresponding to the above noise matrix
3. A `Nw✕E` noise matrix representing the process disturbance, where `E` is the
   number of used data-sets
4. Meta-data corresponding to the above noise matrix


The parameter `Nw` defines the number of steps of the noise sequence, and which has to be long enough for running the simulation.  Specifically, `Nw*δ > N*Ts` must hold, where `δ` is the sampling time of the noise generation, `N` and `Ts` is the number of steps and sampling time of the simulation, respectively.

To use the disturbance model chosen for the pendulum, in a julia repl in [src/julia](src/julia) run:

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

This stores the necessary files in ```src/julia/experiments/expid```. For the experiments in the thesis `δ = 0.01`, `E=100`, and `Nw=10*N` was used.
To use the same name for the experiment as above, make sure to replace `expid` by `500_u2w6_from_Alsvin`. Otherwise, change the experiment name in the call to `get_estimates()`.

