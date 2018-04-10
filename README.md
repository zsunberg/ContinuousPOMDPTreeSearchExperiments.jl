# ContinuousPOMDPTreeSearchExperiments

Experiments for POMCPOW

> Zachary N. Sunberg and Mykel J. Kochenderfer. “Online algorithms for POMDPs with continuous state, action, and observation spaces”. In: International Conference on Automated Planning and Scheduling (ICAPS). 2018.

Paper available at https://arxiv.org/abs/1709.06196.

The POMCPOW implementation with basic documentation is available at https://github.com/JuliaPOMDP/POMCPOW.jl.

This requires Julia version 0.6.

To reproduce the results in the table, get the dependencies with `Pkg.build("ContinuousPOMDPTreeSearchExperiments")` and then run the scripts in `icaps_2018` with `table` in their names.

Scripts for producing the other results are also in the `icaps_2018` directory.

Since Julia is still in a state of flux, a tarball with julia linux binaries and all the packages that the experiments depend on is here https://github.com/zsunberg/ContinuousPOMDPTreeSearchExperiments.jl/releases/download/Final_Experiments/pomcpow_reproduction.tar.gz.
