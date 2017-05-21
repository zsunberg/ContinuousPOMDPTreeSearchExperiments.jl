using LaserTag
using Plots
using POMDPToolbox
using ContinuousPOMDPTreeSearchExperiments
using Plots
using POMDPs
using QMDP
using ProfileView
using ParticleFilters

using DESPOT
import DESPOT: bounds


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

ro = MoveTowards()

p = gen_lasertag(rng=MersenneTwister(4))

solver = DESPOTSolver{LTState,
                      Int,
                      CMeas,
                      LaserBounds,
                      MersenneStreamArray}(bounds = LaserBounds(p),
                                           random_streams=MersenneStreamArray(MersenneTwister(1)),
                                           rng=MersenneTwister(3),
                                           next_state=LTState([1,1], [1,1], false),
                                           curr_obs=CMeas(),
                                           time_per_move=-1.0,
                                           max_trials=100_000
                                          )


policy = solve(solver, p)

@time action(policy, initial_state_distribution(p))

hr = HistoryRecorder(max_steps=5, show_progress=true)
filter = SIRParticleFilter(p, 100_000)

simulate(hr, p, policy, filter)
