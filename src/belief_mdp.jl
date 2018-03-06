struct MeanRewardBeliefMDP{P<:POMDP, R, S, B, A} <: MDP{B, A}
    pomdp::P
    resample::R
    max_frac_replaced::Float64
    _pm::Vector{S}
    _rm::Vector{Float64}
    _wm::Vector{Float64}
end

function MeanRewardBeliefMDP(pomdp::P, resampler, max_frac_replaced) where {P<:POMDP}
    S = state_type(pomdp)
    b0 = resample(resampler, initial_state_distribution(pomdp), MersenneTwister(14))
    MeanRewardBeliefMDP{P, typeof(resampler), S, typeof(b0), action_type(pomdp)}(
        pomdp,
        resampler,
        max_frac_replaced,
        S[],
        Float64[],
        Float64[]
       )
end

function generate_sr(bmdp::MeanRewardBeliefMDP, b, a, rng::AbstractRNG)
    s = rand(rng, filter(s->!isterminal(bmdp.pomdp, s), particles(b)))
    sp, o = generate_so(bmdp.pomdp, s, a, rng)

    if n_particles(b) > 2*bmdp.resample.n
        b = resample(bmdp.resample, b, rng)
    end

    ps = particles(b)
    pm = bmdp._pm
    rm = bmdp._rm
    wm = bmdp._wm
    resize!(pm, 0)
    resize!(rm, 0)
    resize!(wm, 0)

    all_terminal = true
    for i in 1:n_particles(b)
        s = ps[i]
        if !isterminal(bmdp.pomdp, s)
            all_terminal = false
            sp, r = generate_sr(bmdp.pomdp, s, a, rng)
            push!(pm, sp)
            push!(rm, r)
            od = observation(bmdp.pomdp, s, a, sp)
            push!(wm, pdf(od, o))
        end
    end
    ws = sum(wm)
    if all_terminal || ws < eps(1.0/length(wm))
        # warn("All states in particle collection were terminal.")
        return resample(bmdp.resample, reset_distribution(bmdp.pomdp, b, a, o), rng), 0.0
    end

    pc = resample(bmdp.resample, WeightedParticleBelief{state_type(bmdp.pomdp)}(pm, wm, ws, nothing), rng)
    ps = particles(pc)

    mpw = max_possible_weight(bmdp.pomdp, a, o)
    frac_replaced = bmdp.max_frac_replaced*max(0.0, 1.0 - maximum(wm)/mpw)
    n_replaced = floor(Int, frac_replaced*length(ps))
    is = randperm(rng, length(ps))[1:n_replaced]
    for i in is
        ps[i] = new_particle(bmdp.pomdp, b, a, o, rng)
    end

    return pc, dot(wm, rm)/ws
end

function initial_state(bmdp::MeanRewardBeliefMDP, rng::AbstractRNG)
    return resample(bmdp.resample, initial_state_distribution(bmdp.pomdp), rng)
end

actions(bmdp::MeanRewardBeliefMDP{P,U,B,A}, b::B) where {P,U,B,A} = actions(bmdp.pomdp, b)
actions(bmdp::MeanRewardBeliefMDP) = actions(bmdp.pomdp)

isterminal(bmdp::MeanRewardBeliefMDP, b) = all(isterminal(bmdp.pomdp, s) for s in iterator(b))

discount(bmdp::MeanRewardBeliefMDP) = discount(bmdp.pomdp)
