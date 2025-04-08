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
Before running your first ever experiment, it is important to install the necessary packages, or the code will crash. One can attempt to run the code and from the error message see which package needs to be installed, but you can also simply upfront run
```{julia}
using Pkg
pkg_list = ["LsqFit", "DataFrames", "ControlSystems", "Interpolations", "DifferentialEquations", "CSV", "Sundials", "ProgressMeter", "DelimitedFiles", "LinearAlgebra"];
Pkg.add(pkg_list)
```
The main experiment is defined in
[src/julia/run_experiment.jl](src/julia/run_experiment.jl). Additionally, the
noise model and noise generation is defined in
[src/julia/noise_generation.jl](src/julia/noise_generation.jl). The DAE models can be found in [src/julia/models.jl](src/julia/models.jl), while 
the user selects which of those models to use in the corresponding model metadata-file, found in the [src/julia/model_metadata](src/julia/model_metadata).

### Minimal example

Many experiment metaparameters have to be set in the code and are perhaps therefore not easy to edit. The current metaparameters are set so that it is straightforward to run a single experiment for identifying the parameter `k` of the pendulum model. Download the experiment data titled *500_u2w6_from_Alsvin* from [https://kth-my.sharepoint.com/:f:/g/personal/robbj_ug_kth_se/EqMa_VdRU89GtYkNFiPfe-sB6O4B19RKqXIo494jX9mcGA?e=wP89k9](https://kth-my.sharepoint.com/:f:/g/personal/robbj_ug_kth_se/EqMa_VdRU89GtYkNFiPfe-sB6O4B19RKqXIo494jX9mcGA?e=wP89k9) (This link is only valid for 180 days, if it doesn't work then please e-mail robbj@kth.se, since that means we have forgotten to update it). You can also generate your own noise using the instructions under the next heading. In that case, use `N=500`. Place the data in [src/julia/data/experiments](src/julia/data/experiments).


The parameter `Nw` defines the number of steps of the noise sequence, and which has to be long enough for running the simulation.  Specifically, `Nw*δ >= N*Ts` must hold, where `δ` is the sampling time of the noise generation, `N` and `Ts` is the number of steps and sampling time of the simulation, respectively.

To use the same disturbance model as described in the paper, in a julia repl in [src/julia](src/julia) run:

```{julia}
using DataFrames
include("run_experiment.jl")
using .NoiseGeneration: get_filtered_noise, disturbance_model_5, get_multisine_data   # disturbance_model_5 is for delta robot

XW, Wmat, meta_W = get_filtered_noise(disturbance_model_5, δ, E, Nw, scale=0.6, p_scale=500.0)  # Generated process disturbance and meta-data
_, meta_U = get_multisine_data(50, 3)
meta_Y = DataFrame(Ts=10δ, N=Nw÷10)	# It is convenient to also generate metadata for system output
writedlm("data/experiments/expid/XW_T.csv", transpose(XW), ',')
CSV.write("data/experiments/expid/meta_W.csv", meta_W)
CSV.write("data/experiments/expid/meta_U.csv", meta_U)
CSV.write("data/experiments/expid/meta_Y.csv", meta_Y)
```
This stores the necessary files in ```src/julia/experiments/expid```. For the experiments in the paper `δ = 1e-5` (for delta robot, `δ = 0.01` for pendulum), `E=100`, and `Nw=10*N` were used.

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
After that, you can run the estimation experiment over the data-set found in the folder ```src/julia/experiments/500_u2w6_from_Alsvin```, by writing

```{julia}
exp_data, isws = get_experiment_data("expid")
opt_pars_proposed, trace_proposed, trace_gradient = get_estimates(free_pars, exp_id, isws)
# TODO: Add line for computing baseline results too
```

The results can be interpreted as follows. ```opt_pars_proposed[i,e]``` is the optimal value of parameter `i` found by the proposed method for the data-set `e`.
