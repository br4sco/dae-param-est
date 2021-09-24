# dae-param-est
This repository contains the code to the numerical experiment discussed in
(TODO: insert paper link here), which proposes a method for consistent parameter
estimation for Differential Algebraic Equations (DAEs) subject to process
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
[src/julia/noise_model.jl](src/julia/noise_model.jl). The simulation scripts and
physical model is defined in [src/julia/simulation.jl](src/julia/simulation.jl).

### Download or generate noise
You can either download the noise used in our experiment from
https://kth.box.com/v/dae-param-est-noise. Place these files in
[src/julia/data/experiment](src/julia/data/experiment) (be advised, this
experiment consumes a lot of memory). You can also generate noise yourself using
the functions defined in
[src/julia/noise_model.jl](src/julia/noise_model.jl). You should generate and
save four noise matrices in total.

1. A `K✕1` noise matrix serving as the input u(t).
2. Two `K✕n` noise matrices for the true system. The true system is simulated in batches of two.
3. A `K✕M` noise matrix for the proposed method, where `M` is number of noise
   realizations the method should average over.

The parameter `K` defines the number of steps of the noise sequence whose length
should surpass the length you plan to run the simulation with some small
margin. I.e. `K✕δ > N✕Ts`, where `δ` is the sampling time of the noise
generation, `N` and `Ts` is the number of steps and sampling time of the
simulation, respectively.

To use the linear filter described in the paper, in a julia repl in [src/julia](src/julia) run:
```
include("noise_model.jl")
```
Then
```
WS = gen_filtered_noise_1(M, δ, K)
```
For each matrix discussed above with the appropriate arguments. In julia you can save a matrix to a coma separated file using `writedlm`, e.g.
```{julia}
writedlm("data/experiment/data.csv", WS, ',')
```

If you generate your own noise you need to update
[src/julia/run_experiment.jl](src/julia/run_experiment.jl) to point to the
correct files.

### Run the experiment
To run the experiment do
```
julia --threads n
```

in [src/julia](src/julia) to open the julia repl.

Then, in the repl, include the experiment script

```{julia}
include("run-experiment.jl")
```

You will probably have to install a number of dependency's pointed out by julia.

When the you have successfully included the experiment script, create an experiment folder by calling
```{julia}
mk_exp_dir(expid)
```
where `expid` is a string identifying your experiment. 

Then generate outputs of the true system and the two different methods by calling
```{julia}
outputs = get_outputs(expid)
```
*Note that intermediate results are saved to files in the specified experiment directory and subsequent calls to `get_outputs` will load those results rather then re-running the simulation.* 

This will take a while depending on the size of your noise matrices. Finally, you can issue:
```
thetahat_boxplots(outputs, Ns, N_trans)
```
To construct a boxplot that summarizes the statistics of your experiment. The parameter `Ns` is an array of simulation steps `N` where you wish to compare the baseline method and the proposed method, and `N_trans` is the number of steps of the transient.

