module ContinuousPOMDPTreeSearchExperiments

using POMDPs
using POMCP
using POMCPOW
using Parameters
using LightDarkPOMDPs
using StaticArrays
using POMDPToolbox
using ParticleFilters
using GenerativeModels
using ControlSystems
using Plots

import POMCPOW.n_children

export
    RadiusRandom,
    AdaptiveRadiusRandom,
    SimpleFeedback,
    SymmetricNormalResampler,
    MinPopResampler,
    ObsAdaptiveParticleFilter,
    LightDarkLQRSolver,
    LightDarkLQRPolicy,
    ModeAugmentedBelief,
    InfoGatherHeur,
    InfoGatherUpdater,
    OneStepValue

include("policies.jl")
include("updaters.jl")

n_children(h::BeliefNode) = length(h.children)

immutable OneStepValue end
POMCP.estimate_value(o::OneStepValue, pomdp::POMDP, s, h, steps) = reward(pomdp, s)

end # module
