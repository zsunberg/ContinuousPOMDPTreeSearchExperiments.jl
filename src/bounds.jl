type LaserBounds
    ubp::QMDPPolicy{LaserTagPOMDP, Int}
    lb::Float64

    LaserBounds() = new()
    LaserBounds(p, lb) = new(p, lb)
end

function LaserBounds(p::LaserTagPOMDP)
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
