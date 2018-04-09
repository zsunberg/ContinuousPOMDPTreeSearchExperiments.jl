using POMDPs

POMDPs.add("BasicPOMCP")
POMDPs.add("QMDP")
POMDPs.add("POMCPOW")
POMDPs.add("DESPOT")
POMDPs.add("DiscreteValueIteration")
POMDPs.add("ARDESPOT")

try Pkg.clone("https://github.com/slundberg/PmapProgressMeter.jl.git") end

try Pkg.clone("https://github.com/zsunberg/LightDarkPOMDPs.jl") end
try Pkg.clone("https://github.com/zsunberg/VDPTag.jl.git") end
try Pkg.clone("https://github.com/zsunberg/VDPTag2.jl.git") end
try Pkg.clone("https://github.com/zsunberg/LaserTag.jl.git") end
try Pkg.clone("https://github.com/zsunberg/SubHunt.jl.git") end
