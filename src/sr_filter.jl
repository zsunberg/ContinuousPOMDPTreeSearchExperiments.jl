immutable ObsAdaptiveSRFilter{S} <: Updater
    pomdp::POMDP{S}
    resample::Any
    max_frac_replaced::Float64
    rng::AbstractRNG
end

function POMDPs.initialize_belief{S}(up::ObsAdaptiveSRFilter{S}, d::Any)
    sc = resample(up.resample, d, up.rng)
    return ParticleCollection{Tuple{S,Float64}}([(p,0.0) for p in particles(sc)])
end
POMDPs.update(up::ObsAdaptiveSRFilter, b, a, o) = update(up, initialize_belief(up, b), a, o)

function POMDPs.update{S}(up::ObsAdaptiveSRFilter{S}, b::ParticleFilters.ParticleCollection{Tuple{S,Float64}}, a, o)
    if n_particles(b) > 2*up.resample.n
        b = resample(up.resample, b, up.rng)
    end

    ps = particles(b)
    pm = Array{Tuple{S,Float64}}(0)
    wm = Array{Float64}(0)
    sizehint!(pm, n_particles(b))
    sizehint!(wm, n_particles(b))
    all_terminal = true
    for i in 1:n_particles(b)
        sr = ps[i]
        s = first(sr)
        if !isterminal(up.pomdp, s)
            all_terminal = false
            sp, r = generate_sr(up.pomdp, s, a, up.rng)
            push!(pm, (sp,r))
            od = observation(up.pomdp, s, a, sp)
            push!(wm, pdf(od, o))
        end
    end
    ws = sum(wm)
    if all_terminal || sum(wm) == 0.0
        # warn("All states in particle collection were terminal.")
        return initialize_belief(up, reset_distribution(up.pomdp, a, o))
    end

    pc = resample(up.resample, WeightedParticleBelief{Tuple{S,Float64}}(pm, wm, ws, nothing), up.rng)
    ps = particles(pc)
    # for i in 1:length(ps)
    #     ps[i] += 0.001*randn(up.rng, 2)
    # end

    mpw = max_possible_weight(up.pomdp, a, o)
    frac_replaced = up.max_frac_replaced*max(0.0, 1.0 - maximum(wm)/mpw)
    n_replaced = floor(Int, frac_replaced*length(ps))
    is = randperm(up.rng, length(ps))[1:n_replaced]
    for i in is
        ps[i] = (new_particle(up.pomdp, a, o, up.rng), 0.0)
    end
    return pc
end

