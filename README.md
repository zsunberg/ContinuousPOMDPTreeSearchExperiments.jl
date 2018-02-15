# ContinuousPOMDPTreeSearchExperiments

[![Build Status](https://travis-ci.org/zsunberg/ContinuousPOMDPTreeSearchExperiments.jl.svg?branch=master)](https://travis-ci.org/zsunberg/ContinuousPOMDPTreeSearchExperiments.jl)

[![Coverage Status](https://coveralls.io/repos/zsunberg/ContinuousPOMDPTreeSearchExperiments.jl/badge.svg?branch=master&service=github)](https://coveralls.io/github/zsunberg/ContinuousPOMDPTreeSearchExperiments.jl?branch=master)

[![codecov.io](http://codecov.io/github/zsunberg/ContinuousPOMDPTreeSearchExperiments.jl/coverage.svg?branch=master)](http://codecov.io/github/zsunberg/ContinuousPOMDPTreeSearchExperiments.jl?branch=master)

Experiments for POMCPOW

> Zachary N. Sunberg and Mykel J. Kochenderfer. “POMCPOW: an online algorithm for POMDPs with continuous state, action, and observation spaces”. In: International Conference on Automated Planning and Scheduling (ICAPS). 2018.

Paper available at https://arxiv.org/abs/1709.06196.

The POMCPOW implementation with basic documentation is available at https://github.com/JuliaPOMDP/POMCPOW.jl.

This requires Julia version 0.6.

To reproduce the results in the table, get the dependencies with `Pkg.build("ContinuousPOMDPTreeSearchExperiments")` and then run the scripts in `icaps_2018` with `table` in their names.

Scripts for producing the other results are also in the `icaps_2018` directory.
