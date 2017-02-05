using POMDPs

POMDPs.add("POMDPToolbox")
POMDPs.add("POMCP")

Pkg.clone("https://github.com/zsunberg/LightDarkPOMDPs.jl")
Pkg.clone("https://github.com/zsunberg/ParticleFilters.jl")
