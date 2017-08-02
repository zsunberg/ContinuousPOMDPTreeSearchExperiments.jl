type LaserBounds{P<:LaserTagPOMDP}
    ubp::QMDPPolicy{P, Int}
    lb::Float64

    LaserBounds{P}() where P = new()
    LaserBounds{P}(p, lb) where P = new(p, lb)
end

function LaserBounds(p::LaserTagPOMDP)
    sol = QMDPSolver(max_iterations=1000)
    pol = solve(sol, p)
    lb = -p.step_cost/(1-discount(p))
    return LaserBounds{typeof(p)}(pol, lb)
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
