type LaserBounds{P<:LaserTagPOMDP}
    ubp::QMDPPolicy{P, Int}
    lb_not_same::Float64

    LaserBounds{P}() where P = new()
    LaserBounds{P}(p, lb) where P = new(p, lb)
end

function LaserBounds(p::LaserTagPOMDP)
    sol = QMDPSolver(max_iterations=1000)
    pol = solve(sol, p)
    lb_not_same = -p.step_cost/(1-discount(p))
    return LaserBounds{typeof(p)}(pol, lb)
end

function bounds(l::LaserBounds, p::LaserTagPOMDP, b::ScenarioBelief)
    no = previous_obs(b)
    if all(isterminal(p, s) for s in iterator(b))
        lb = 0.0
    elseif !isnull(previous_obs(b)) && get(previous_obs(b)) == LaserTag.D_SAME_LOC
        lb = l.ubp.pomdp.tag_reward
    else
        lb = l.lb_not_same
    end
    vsum = 0.0
    for s in iterator(b)
        vsum += state_value(l.ubp, s)
    end
    ub = vsum/length(b.scenarios)
    if lb > ub
        @show b.scenarios
    end
    return lb, vsum/length(b.scenarios)
end

function init_bounds(l::LaserBounds, p::LaserTagPOMDP, ::DESPOTSolver)
    sol = QMDPSolver(max_iterations=1000)
    l.ubp = solve(sol, p)
    l.lb_not_same = -p.step_cost/(1-discount(p))
    return l
end


#=
function bounds{S}(l::LaserBounds, p::LaserTagPOMDP, b::Vector{DESPOTParticle{S}}, ::DESPOTConfig)
    bv = zeros(n_states(p))
    for dp in b 
        bv[state_index(p, dp.state)] += dp.weight
    end
    bv ./= sum(bv)
    return l.lb, value(l.ubp, DiscreteBelief(bv))
end

function init_bounds(l::LaserBounds, p::LaserTagPOMDP, config::DESPOTConfig)
    sol = QMDPSolver(max_iterations=1000)
    l.ubp = solve(sol, p)
    l.lb = -p.step_cost/(1-discount(p))
    return l
end
=#

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
