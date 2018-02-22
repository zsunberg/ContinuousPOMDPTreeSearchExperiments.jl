type LaserBounds{P<:LaserTagPOMDP}
    ubp::QMDPPolicy{P, Int}
    lbp::RandomPolicy
    max_depth::Int
    lb_not_same::Float64

    LaserBounds{P}() where P = new()
end

#=
function LaserBounds(p::LaserTagPOMDP)
    sol = QMDPSolver(max_iterations=1000)
    pol = solve(sol, p)
    lb_not_same = -p.step_cost/(1-discount(p))
    return LaserBounds{typeof(p)}(pol, lb)
end
=#

function bounds(l::LaserBounds, p::LaserTagPOMDP, b::ScenarioBelief)
    ps = particles(b)
    if !all(isterminal(p, s) for s in ps) && !isnull(previous_obs(b))
        if get(previous_obs(b)) == LaserTag.D_SAME_LOC
            @assert all(!isterminal(p, s) && s.robot==s.opponent for s in ps)
        else
            @assert any(isterminal(p, s) || s.robot!=s.opponent for s in ps)
        end
    end
    steps = l.max_depth-b.depth
    lb = ARDESPOT.branching_sim(p, l.lbp, b, steps)/length(b.scenarios)
    return bounds(l, p, ps, lb)
end

function bounds(l::LaserBounds, p::LaserTagPOMDP, particles, provided_lb=nothing)
    if all(isterminal(p, s) for s in particles)
        lb = 0.0
    elseif all(!isterminal(p, s) && s.robot==s.opponent for s in particles)
        lb = l.ubp.pomdp.tag_reward
    elseif !isa(provided_lb, Void)
        lb = provided_lb
    elseif length(particles) == 1
        lb = state_value(l.ubp, first(particles))
    else
        lb = l.lb_not_same
    end

    vsum = 0.0
    for s in particles
        vsum += state_value(l.ubp, s)
    end
    ub = vsum/length(particles)
    # if lb > ub
    #     warn("lb > ub")
    #     @show particles
    # end
    return lb, vsum/length(particles)
end

function init_bounds(l::LaserBounds, p::LaserTagPOMDP, max_depth::Int, rng=Base.GLOBAL_RNG)
    sol = QMDPSolver(max_iterations=1000)
    l.ubp = solve(sol, p)
    l.lbp = solve(RandomSolver(rng), p)
    l.max_depth = max_depth
    l.lb_not_same = -p.step_cost/(1-discount(p))
    return l
end

init_bounds(l::LaserBounds, p::LaserTagPOMDP, s::DESPOTSolver) = init_bounds(l, p, s.D, s.rng)
# DESPOT.init_bounds(l::LaserBounds, p::LaserTagPOMDP, c::DESPOT.DESPOTConfig) = init_bounds(l, p, c.search_depth)
# 
# function DESPOT.bounds{S}(l::LaserBounds, p::LaserTagPOMDP, b::Vector{DESPOT.DESPOTParticle{S}}, ::DESPOT.DESPOTConfig)
#     return bounds(l, p, p.state for p in b)
# end
# 
# DESPOT.default_action(l::LaserBounds, pomdp::POMDP, particles, c) = default_action(particles)

function default_action(particles)
    if all(s.robot==s.opponent for s in particles)
        return LaserTag.TAG_ACTION
    else
        warn("non-tag default")
        return rand(1:4)
    end
end

struct NoGapTag end

function ARDESPOT.default_action(ngt::NoGapTag, b, ex)
    if isa(ex, NoGap)
        return default_action(particles(b))
    else
        rethrow(ex)
    end
    return -1
end

function nogap_tag(b, ex)
    if isa(ex, NoGap)
        @assert all(!s.terminal && s.robot==s.opponent for s in iterator(b))
        # @assert ng.value == l.ubp.pomdp.tag_reward
        return LaserTag.TAG_ACTION
    else
        rethrow(ex)
    end
end


immutable InevitableInit end 

POMCPOW.init_N(m::InevitableInit, pomdp::LaserTagPOMDP, h::BeliefNode, a::Int) = h.node == 1 ? 0 : 1

function POMCPOW.init_V(m::InevitableInit, pomdp::LaserTagPOMDP, h::BeliefNode, a::Int)
    # only works for POMCPOW now because of node access
    if a == LaserTag.TAG_ACTION
        if h.node == 1
            return 0.0
        elseif h.tree.o_labels[h.node] == LaserTag.C_SAME_LOC
            return pomdp.tag_reward
        else
            return -pomdp.tag_reward - pomdp.step_cost
        end
    else
        return -pomdp.step_cost
    end
end

struct RootToNextMLFirst
    rng::MersenneTwister
end

function MCTS.next_action(gen::RootToNextMLFirst, p::VDPTagPOMDP, b, node)
    if isroot(node) && n_children(node) < 1
        target_sum=MVector(0.0, 0.0)
        agent_sum=MVector(0.0, 0.0)
        for s in particles(b::ParticleCollection)
            target_sum += s.target
            agent_sum += s.agent
        end
        next = VDPTag.next_ml_target(mdp(p), target_sum/n_particles(b))
        diff = next-agent_sum/n_particles(b)
        return TagAction(false, atan2(diff[2], diff[1]))
    else
        return rand(gen.rng, actions(p))
    end
end

MCTS.next_action(gen::RootToNextMLFirst, p::GenerativeBeliefMDP, b, node) = next_action(gen, p.pomdp, b, node)

# XXXD
# function MCTS.next_action(gen::RootToNextMLFirst, p::VDPTag.DiscreteVDPTagProblem, b, node)
#     if isroot(node) && n_children(node) < 1
#         cpc = ParticleCollection([convert_s(state_type(cproblem(p)), s, p) for s in b.particles])
#         ca = next_action(gen, cproblem(p), cpc, node)
#         return convert_a(action_type(p), ca, p)
#     else
#         return rand(gen.rng, actions(p))
#     end
# end

struct ModeMDP <: Policy 
    vi::ValueIterationPolicy
end

struct ModeMDPSolver <: Solver
    vi::ValueIterationSolver
end

ModeMDPSolver(;kwargs...) = ModeMDPSolver(ValueIterationSolver(;kwargs...))

POMDPs.solve(sol::ModeMDPSolver, pomdp::POMDP) = ModeMDP(solve(sol.vi, pomdp))

POMDPs.action(p::ModeMDP, b) = action(p.vi, mode(b))

struct VDPUpper end

function ARDESPOT.ubound(ub::VDPUpper, pomdp::POMDP, b::ScenarioBelief)
    if all(isterminal(pomdp, s) for s in particles(b))
        return 0.0
    else
        return mdp(cproblem(pomdp)).tag_reward
    end
end

struct SampleRollout{P<:Policy, RNG<:AbstractRNG}
    policy::P
    rng::RNG
end

function MCTS.estimate_value(ro::SampleRollout, mdp::GenerativeBeliefMDP, b, depth::Int)
    sfo = SolvedFORollout(ro.policy, ro.rng)
    return rollout(sfo, mdp.pomdp, rand(ro.rng, b), depth)
end
