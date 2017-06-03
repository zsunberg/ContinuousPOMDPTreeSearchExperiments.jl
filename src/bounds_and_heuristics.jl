type LaserBounds{M}
    ubp::QMDPPolicy{LaserTagPOMDP{M}, Int}
    lb::Float64

    LaserBounds() = new()
    LaserBounds(p, lb) = new(p, lb)
end

function LaserBounds{M}(p::LaserTagPOMDP{M})
    sol = QMDPSolver(max_iterations=1000)
    pol = solve(sol, p)
    lb = -p.step_cost/(1-discount(p))
    return LaserBounds{}(pol, lb)
end

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

immutable InevitableInit end 

init_N(m::InevitableInit, pomdp::LaserTagPOMDP, h::BeliefNode, a::Int) = h.node == 1 ? 0 : 1

function init_V(m::InevitableInit, pomdp::LaserTagPOMDP, h::BeliefNode, a::Int)
    # only works for POMCPOW now because of node access
    if a == TAG_ACTION
        if h.node == 1
            return 0.0
        elseif h.tree.o_labels[h.node] == C_SAME_LOC
            return pomdp.tag_reward
        else
            return -pomdp.tag_reward - pomdp.step_cost
        end
    else
        return -pomdp.step_cost
    end
end
