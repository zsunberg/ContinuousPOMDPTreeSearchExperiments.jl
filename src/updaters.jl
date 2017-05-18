@with_kw immutable SymmetricNormalResampler
    n::Int
    std::Float64
end

function ParticleFilters.resample(r::SymmetricNormalResampler, b::WeightedParticleBelief, rng::AbstractRNG)
    collection = resample(LowVarianceResampler(r.n), b, rng)
    ps = particles(collection)
    for i in 1:r.n 
        ps[i] += r.std*randn(rng, 2)
    end
    return collection
end

@with_kw immutable MinPopResampler
    n::Int
    min_pop::Int
    std::Float64
end

function ParticleFilters.resample(r::MinPopResampler, b, rng::AbstractRNG)
    collection = resample(LowVarianceResampler(r.n), b, rng)
    ps = particles(collection)
    nu = length(unique(ps))
    if r.min_pop > nu
        is = rand(rng, 1:r.n, r.min_pop - nu)
        for i in is
            ps[i] += r.std*randn(rng, 2)
        end
    end
    @show length(unique(ps))
    @show length(ps)
    return collection
end


immutable ObsAdaptiveParticleFilter{S} <: Updater{ParticleFilters.ParticleCollection{S}}
    pomdp::POMDP{S}
    resample::Any
    max_frac_replaced::Float64
    rng::AbstractRNG
end

POMDPs.initialize_belief{S}(up::ObsAdaptiveParticleFilter{S}, d::Any) = resample(up.resample, d, up.rng)
POMDPs.update(up::ObsAdaptiveParticleFilter, b, a, o) = update(up, resample(up.resample, b, up.rng), a, o)

function POMDPs.update{S}(up::ObsAdaptiveParticleFilter{S}, b::ParticleFilters.ParticleCollection, a, o)
    if n_particles(b) > 2*up.resample.n
        b = resample(up.resample, b, up.rng)
    end

    ps = particles(b)
    pm = Array(S, 0)
    wm = Array(Float64, 0)
    sizehint!(pm, n_particles(b))
    sizehint!(wm, n_particles(b))
    all_terminal = true
    for i in 1:n_particles(b)
        s = ps[i]
        if !isterminal(up.pomdp, s)
            all_terminal = false
            sp = generate_s(up.pomdp, s, a, up.rng)
            push!(pm, sp)
            od = observation(up.pomdp, s, a, sp)
            push!(wm, pdf(od, o))
        end
    end
    ws = sum(wm)
    if all_terminal || sum(wm) == 0.0
        # warn("All states in particle collection were terminal.")
        return initialize_belief(up, reset_distribution(up.pomdp, b, a, o))
    end

    pc = resample(up.resample, WeightedParticleBelief{S}(pm, wm, ws, nothing), up.rng)
    ps = particles(pc)
    # for i in 1:length(ps)
    #     ps[i] += 0.001*randn(up.rng, 2)
    # end

    mpw = max_possible_weight(up.pomdp, a, o)
    frac_replaced = up.max_frac_replaced*max(0.0, 1.0 - maximum(wm)/mpw)
    n_replaced = floor(Int, frac_replaced*length(ps))
    is = randperm(up.rng, length(ps))[1:n_replaced]
    for i in is
        ps[i] = new_particle(up.pomdp, a, o, up.rng)
    end
    return pc
end

# POMCPOW.belief_type(::Type{ObsAdaptiveParticleFilter{Vec2}}, ::Type{LightDark2DTarget}) = POWNodeBelief{Vec2, Vec2, Vec2, LightDark2DTarget}

function max_possible_weight(pomdp::AbstractLD2, a, o)
    od = observation(pomdp, a, o) # will only work for LightDark
    return pdf(od, o)    
end

function new_particle(pomdp::AbstractLD2, a, o, rng)
    return o + LightDarkPOMDPs.obs_std(pomdp, o[1])*randn(rng, 2)
end

reset_distribution(p::POMDP, b, a, o) = initial_state_distribution(p)

#=
max_possible_weight(pomdp::PowseekerPOMDP, a::GPSOrAngle, o::SkierObs) = 0.0

function new_particle(pomdp::PowseekerPOMDP, a::GPSOrAngle, o::SkierObs, rng::AbstractRNG)
    if a.gps
        return SkierState(o.time, get(o.pos)+pomdp.gps_std*randn(rng, 1), 2*pi*rand(rng))
    else
        is = initial_state(pomdp, rng)
        return SkierState(o.time, is.pos, is.psi)
    end
end

reset_distribution(p::PowseekerPOMDP, a::GPSOrAngle, o::SkierObs) = SkierUnif(o.time, mdp(p).xlim, mdp(p).ylim)
=#

max_possible_weight(pomdp::VDPTagPOMDP, a::TagAction, o::Float64) = 0.0

new_particle(pomdp::VDPTagPOMDP, a::TagAction, o::Float64) = error("shouldn't get here")


function reset_distribution(p::LaserTagPOMDP, b::ParticleCollection, a, o)
    warn("Resetting Particle Filter Distribution")
    rob = first(particles(b)).robot
    nextrob = LaserTagPOMDP.add_if_inside(p.floor, rob, LaserTagPOMDP.ACTION_DIRS[a])
    if o == C_SAME_LOC
        return ParticleCollection{LTState}([LTState(nextrob, nextrob, false)])
    else
        return LaserTagPOMDP.LTInitialBelief(nextrob, p.floor)
    end
end

max_possible_weight(pomdp::LaserTagPOMDP, a::Int, o) = 0.0

new_particle(pomdp::LaserTagPOMDP, a::Int, o) = error("tried to generate a new particle (shouldn't get here)")

#=
max_possible_weight(pomdp::LaserTagPOMDP, a::Int, o::Float64) = max(1.0, pdf(Normal(0.0, pomdp.return_std), 0.0))

function new_particle(pomdp::VDPTagPOMDP, a::Int, o::Float64)
    if o == C_SAME_LOC
        return LTState(
end
=#


#=
function POMCPOW.init_node_belief(::ObsAdaptiveParticleFilter, p::LightDark2DTarget, s::Vec2, a::Vec2, o::Vec2, sp::Vec2)
    POWNodeBelief(p, s, a, o, sp)
end

function POMCPOW.push_weighted!(b::POWNodeBelief, up::ObsAdaptiveParticleFilter, s::Vec2, sp::Vec2)
    od = observation(b.model, s, b.a, sp)
    w = pdf(od, b.o)
    ood = observation(b.model, b.a, b.o)
    frac_replaced = up.max_frac_replaced*max(0.0, 1.0 - w/pdf(ood, b.o))
    insert!(b.dist, sp, w*(1.0-frac_replaced))
    sp2 = rand(up.rng, ood)
    insert!(b.dist, sp2, w*frac_replaced)
end
=#
