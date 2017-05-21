immutable LaserBounds
    ubp::QMDPPolicy
    lb::Float64
end

function LaserBounds(p::LaserTagPOMDP)
    sol = QMDPSolver(max_iterations=1000)
    pol = solve(sol, p)
    lb = -p.step_cost/(1-discount(p))
    return LaserBounds(pol, lb)
end

function bounds{S}(l::LaserBounds, p::LaserTagPOMDP, b::Vector{DESPOTParticle{S}}, ::DESPOTConfig)
    bv = zeros(n_states(p))
    for dp in b 
        bv[state_index(p, dp.state)] += dp.weight
    end
    bv ./= sum(bv)
    return l.lb, value(l.ubp, DiscreteBelief(bv))
end
