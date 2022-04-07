# dae-param-est
A Parameter Estimation Experiment for DAEs  
For the code used in our [paper published at CDC 2021](https://www.diva-portal.org/smash/get/diva2:1539989/FULLTEXT01.pdf), please see the branch "cdc".
For the code used in our paper submitted to CDC 2022, please see the branch "CDC22".

# Prerequisites
The simulation routine is implemented in
[julia](https://docs.julialang.org/en/v1/), see their homepage for
installations instructions on your platform.

You will need to install some additional julia packages. The julia repl
(`julia`) will notify you which packages are missing when you include the
simulation implementation `include("simulation.jl")` and how to install them.
