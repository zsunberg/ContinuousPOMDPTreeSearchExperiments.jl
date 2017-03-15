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
    InfoGatherUpdater


@with_kw type RadiusRandom <: Policy
    radius::Float64     = 1.0
    rng::AbstractRNG    = Base.GLOBAL_RNG
end

POMCP.next_action(gen::RadiusRandom, pomdp::POMDP, b, h) = rand_in_radius(gen.rng, gen.radius)
function rand_in_radius(rng::AbstractRNG, radius::Float64)
    r = radius*rand(rng)
    angle = 2*pi*rand(rng)
    return r*@SVector([cos(angle), sin(angle)])
end

POMDPs.action(p::RadiusRandom, b) = rand_in_radius(p.rng, p.radius)
POMDPs.updater(p::RadiusRandom) = VoidUpdater()

@with_kw type AdaptiveRadiusRandom <: Policy
    max_radius::Float64     = 1.0
    to_zero_first::Bool     = true
    to_light_second::Bool   = true
    rng::AbstractRNG        = Base.GLOBAL_RNG
end

function POMCP.next_action(gen::AdaptiveRadiusRandom, pomdp::POMDP, b, h::BeliefNode)
    if gen.to_zero_first && n_children(h) < 1
        m = mean(b)
        n = norm(m)
        return clip_to_max(-m, gen.max_radius)
    elseif gen.to_light_second && n_children(h) < 2
        m = mean(b)
        n = norm(m)
        return clip_to_max(Vec2(pomdp.min_noise_loc-m[1], 0.0), gen.max_radius)
    else
        rad = min(gen.max_radius)
        return rand_in_radius(gen.rng, rad)
    end
    #=
    @show h.node
    @show h.tree.tried[h.node]
    @show h.tree.total_n[h.node]
    @show h.tree.n[first(h.tree.tried[h.node])]
    =#
end

n_children(h::BeliefNode) = length(h.children)

#=
POMDPs.action(p::AdaptiveRadiusRandom, b) = rand_in_radius(p.rng, min(p.max_radius, p.radius_scale*norm(mean(b))))
POMDPs.updater
=#

@with_kw type SimpleFeedback <: Policy
    gain::Float64 = 1.0
    max_radius::Float64 = 1.0
end

function clip_to_max(s::AbstractVector{Float64}, max::Float64)
    nrm = norm(s)
    if nrm > max
        return max/nrm*s
    else
        return s
    end
end

POMDPs.action(p::SimpleFeedback, b::SymmetricNormal2) = clip_to_max(-p.gain*b.mean, p.max_radius)
POMDPs.action(p::SimpleFeedback, b) = clip_to_max(-p.gain*mean(b), p.max_radius)
POMDPs.action(p::SimpleFeedback, o::Vec2) = clip_to_max(-p.gain*o, p.max_radius)

@with_kw immutable SymmetricNormalResampler
    n::Int
    std::Float64
end

function ParticleFilters.resample(r::SymmetricNormalResampler, b::WeightedParticleBelief, rng::AbstractRNG)
    collection = resample(LowVarianceResampler(r.n), b, rng)
    ps = particles(collection)
    for i in 1:r.n 
        ps[i] += r.std*randn(rng, 2)
    end
    return collection
end

@with_kw immutable MinPopResampler
    n::Int
    min_pop::Int
    std::Float64
end

function ParticleFilters.resample(r::MinPopResampler, b, rng::AbstractRNG)
    collection = resample(LowVarianceResampler(r.n), b, rng)
    ps = particles(collection)
    nu = length(unique(ps))
    if r.min_pop > nu
        is = rand(rng, 1:r.n, r.min_pop - nu)
        for i in is
            ps[i] += r.std*randn(rng, 2)
        end
    end
    @show length(unique(ps))
    @show length(ps)
    return collection
end


immutable ObsAdaptiveParticleFilter{S} <: Updater{ParticleFilters.ParticleCollection}
    pomdp::POMDP{S}
    resample::Any
    max_frac_replaced::Float64
    rng::AbstractRNG
end

POMDPs.initialize_belief{S}(up::ObsAdaptiveParticleFilter{S}, d::Any) = resample(up.resample, d, up.rng)
POMDPs.update(up::ObsAdaptiveParticleFilter, b, a, o) = update(up, resample(up.resample, b, up.rng), a, o)

function POMDPs.update{S}(up::ObsAdaptiveParticleFilter{S}, b::ParticleFilters.ParticleCollection, a, o)
    ps = particles(b)
    pm = Array(S, 0)
    wm = Array(Float64, 0)
    sizehint!(pm, n_particles(b))
    sizehint!(wm, n_particles(b))
    all_terminal = true
    for i in 1:n_particles(b)
        s = ps[i]
        if !isterminal(up.pomdp, s)
            all_terminal = false
            sp = generate_s(up.pomdp, s, a, up.rng)
            push!(pm, sp)
            od = observation(up.pomdp, s, a, sp)
            push!(wm, pdf(od, o))
        end
    end
    if all_terminal
        warn("All states in particle collection were terminal.")
        return initialize_belief(up, initial_state_distribution(up.pomdp))
    end

    pc = resample(up.resample, WeightedParticleBelief{S}(pm, wm, sum(wm), nothing), up.rng)
    ps = particles(pc)
    for i in 1:length(ps)
        ps[i] += 0.001*randn(up.rng, 2)
    end

    od = observation(up.pomdp, a, o) # will only work for LightDark
    frac_replaced = up.max_frac_replaced*max(0.0, 1.0 - maximum(wm)/pdf(od, o))
    n_replaced = floor(Int, frac_replaced*length(ps))
    is = randperm(up.rng, length(ps))[1:n_replaced]
    for i in is
        ps[i] = o + LightDarkPOMDPs.obs_std(up.pomdp, o[1])*randn(up.rng, 2)
    end
    return pc
end



immutable LightDarkLQRSolver <: Solver end

function POMDPs.solve(solver::LightDarkLQRSolver, pomdp::LightDark2D)
    A = eye(2)
    B = eye(2)
    K = dlqr(A, B, pomdp.Q, pomdp.R)
    return LightDarkLQRPolicy(K)
end

immutable LightDarkLQRPolicy <: Policy
    K::AbstractMatrix{Float64}
end

POMDPs.action(p::LightDarkLQRPolicy, s::Vec2) = -p.K*s
POMDPs.action(p::LightDarkLQRPolicy, b) = -p.K*mean(b)

immutable InfoGatherHeur <: Policy
    info_x::Float64
    std_thresh::Float64
    homing::Policy
    exploring_up::Updater
    homing_up::Updater
end

# for now, mode can be :exploring or :homing
immutable ModeAugmentedBelief{B}
    mode::Symbol
    b::B
end

@recipe function f(b::ModeAugmentedBelief) b.b end

function POMDPs.action(h::InfoGatherHeur, b::ModeAugmentedBelief)
    if b.mode == :homing
        return action(h.homing, b.b)
    else
        target = Vec2(h.info_x, 0.0)
        diff = mean(b.b) - target
        return action(h.homing, diff)
    end
end

POMDPs.updater(h::InfoGatherHeur) = InfoGatherUpdater(h.std_thresh, h.exploring_up, h.homing_up)

immutable InfoGatherUpdater <: Updater{ModeAugmentedBelief}
    std_thresh::Float64
    exploring_up::Updater
    homing_up::Updater
end

POMDPs.initialize_belief(up::InfoGatherUpdater, b) = ModeAugmentedBelief(:exploring, b)

function POMDPs.update(up::InfoGatherUpdater, b::ModeAugmentedBelief, a, o)
    if b.mode != :homing
        bnew = update(up.exploring_up, b.b, a, o)
        std = det(cov(cat(2, particles(bnew)...)'))^(1/4)
        if std < up.std_thresh
            return ModeAugmentedBelief(:homing, bnew)
        else
            return ModeAugmentedBelief(:exploring, bnew)
        end
    else
        bnew = update(up.homing_up, b.b, a, o)
        return ModeAugmentedBelief(:homing, bnew)
    end
end


end # module
