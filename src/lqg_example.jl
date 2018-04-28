struct Staged{T}
    x::T
    k::Int
end

Base.:+(a::Staged, b::Staged) = Staged(a.x+b.x, a.k)
Base.:*(a::Staged, b::Number) = Staged(a.x*b, a.k)
Base.:/(a::Staged, b::Number) = Staged(a.x/b, a.k)

struct LQG1D <: POMDP{Staged{Float64}, Float64, Float64}
    a::Float64 # state transition
    b::Float64 # control
    c::Float64 # observation
    q::Float64 # state cost
    r::Float64 # control cost
    m::Float64 # process noise standard deviation
    n::Float64 # process noise standard deviation
    p::Float64 # initial distribution standard deviation
    k::Int     # number of stages
end

discount(::LQG1D) = 1.0
isterminal(m::LQG1D, s::Staged{Float64}) = s.k == m.k 

rand(rng::AbstractRNG, d::Staged{D}) where D = Staged(rand(rng, d.x), d.k)
mean(d::Staged) = Staged(mean(d.x), d.k)

transition(m::LQG1D, s::Staged{Float64}, u::Float64) = Staged(Normal(m.a*s.x + m.b*u, m.m), s.k+1)
observation(m::LQG1D, s::Staged{Float64}) = Normal(m.c*s.x, m.n)
function reward(m::LQG1D, s::Staged{Float64}, u::Float64, sp::Staged{Float64})
    c = m.q*s.x^2 + m.r*u^2
    if isterminal(m, sp)
        c += m.q*sp.x^2
    end
    return -c
end
initial_state_distribution(m::LQG1D) = Staged(Normal(0.0, m.p), 1)

# function MCTS.next_action(gen::RandomActionGenerator, m::LQG1D, b, snode::MCTS.AbstractStateNode)
#     rs = rand(gen.rng, b)
#     x = rs.x
#     return rand(gen.rng, Normal(-1/2*x, 1.0))
# end

function MCTS.next_action(gen::RandomActionGenerator, m::LQG1D, b, snode::MCTS.AbstractStateNode)
    ms = mean(b)
    if ms.k == 1
        return 0.0
    else
        x = ms.x
        return rand(gen.rng, Normal(-1/2*x, 0.01))
    end
end

struct LinearFeedback <: Policy
    k::Float64
end

function action(p::LinearFeedback, b::Staged)
    return -p.k*mean(b.x)
end

struct Kalman1D <: Updater
    m::LQG1D
end

function update(kf::Kalman1D, b::Staged{D}, a::Float64, o::Float64) where D <: Normal
    m = kf.m
    propped = m.a*mean(b.x) + m.b*a
    propped_var = m.a^2*var(b.x) + m.m^2
    variance = propped_var - propped_var^2*m.c^2/(m.c^2*propped_var + m.n)
    xhat = propped + propped + variance*m.c/m.n*(o - m.c*propped)
    return Staged(Normal(xhat, sqrt(variance)), b.k+1)
end
