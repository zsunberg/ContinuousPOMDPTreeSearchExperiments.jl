using ContinuousPOMDPTreeSearchExperiments
using Base.Test

using POMDPModels
using POMDPToolbox
using LaserTag
using ParticleFilters
using QMDP

srand(4)

sld = SimpleLightDark()
@test isterminal(sld, sld.radius+1)
p = solve(LDHSolver(), sld)
filter = SIRParticleFilter(sld, 1000)
for (s, b, a, r, sp, o) in stepthrough(sld, p, filter, "sbarspo", max_steps=100)
    @show (s, a, r, sp, o)
    @show mean(b)
end

qp = solve(QMDPSolver(), sld, verbose=true)
filter = SIRParticleFilter(sld, 1000)
for (s, b, a, r, sp, o) in stepthrough(sld, qp, filter, "sbarspo", max_steps=100)
    @show (s, a, r, sp, o)
    @show mean(b)
end

m = LQG1D(1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 3)
filter = Kalman1D(m)
policy = LinearFeedback(1/2)
rsum = 0.0
for (s, b, a, r, sp, o) in stepthrough(m, policy, filter, "sbarspo", max_steps=100)
    @show (s, a, r, sp, o)
    @show mean(b)
    rsum += r
end
@show rsum
