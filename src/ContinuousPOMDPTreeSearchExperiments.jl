module ContinuousPOMDPTreeSearchExperiments

using POMDPs
using BasicPOMCP
using POMCPOW
using Parameters
using LightDarkPOMDPs
using VDPTag
using StaticArrays
using POMDPToolbox
using ParticleFilters
using ControlSystems
using Plots
using Distributions
using LaserTag
# using DESPOT
using ARDESPOT
using QMDP

# import DESPOT: bounds, init_bounds
import ARDESPOT: bounds, init_bounds
import POMCPOW.n_children

export
    RadiusRandom,
    AdaptiveRadiusRandom,
    SimpleFeedback,
    SymmetricNormalResampler,
    MinPopResampler,
    ObsAdaptiveParticleFilter,
    ObsAdaptiveSRFilter,
    LightDarkLQRSolver,
    LightDarkLQRPolicy,
    ModeAugmentedBelief,
    InfoGatherHeur,
    InfoGatherUpdater,
    OneStepValue,
    LaserBounds,
    InevitableInit


include("policies.jl")
include("updaters.jl")
include("action_gen.jl")

include("sr_filter.jl")
include("bounds_and_heuristics.jl")

n_children(h::BeliefNode) = length(h.children)

immutable OneStepValue end
BasicPOMCP.estimate_value(o::OneStepValue, pomdp::POMDP, s, h, steps) = reward(pomdp, s)

end # module
