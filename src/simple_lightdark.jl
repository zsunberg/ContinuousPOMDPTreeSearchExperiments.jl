@with_kw struct SimpleLightDark <: POMDPs.POMDP{Int,Int,Float64}
    discount::Float64       = 1.0
    correct_r::Float64      = 10.0
    incorrect_r::Float64    = -10.0
    light_loc::Int          = 10
    radius::Int             = 30
end
discount(p::SimpleLightDark) = p.discount
isterminal(p::SimpleLightDark, s::Number) = !(s in -p.radius:p.radius)

const ACTIONS = [-10, -1, 0, 1, 10]
actions(p::SimpleLightDark) = ACTIONS
n_actions(p::SimpleLightDark) = length(actions(p))
const ACTION_INDS = Dict(a=>i for (i,a) in enumerate(actions(SimpleLightDark())))
action_index(p::SimpleLightDark, a::Int) = ACTION_INDS[a]

states(p::SimpleLightDark) = -p.radius:p.radius + 1
n_states(p::SimpleLightDark) = length(states(p))
state_index(p::SimpleLightDark, s::Int) = s+p.radius+1

function transition(p::SimpleLightDark, s::Int, a::Int) 
    if a == 0
        return SparseCat(SVector(p.radius+1), SVector(1.0))
    else
        return SparseCat(SVector(clamp(s+a, -p.radius, p.radius)), SVector(1.0))
    end
end

observation(p::SimpleLightDark, sp) = Normal(sp, abs(sp - p.light_loc) + 0.0001)

function reward(p::SimpleLightDark, s, a)
    if a == 0
        return s == 0 ? p.correct_r : p.incorrect_r
    else
        return -1.0
    end
end

function initial_state_distribution(p::SimpleLightDark)
    return SparseCat(-p.radius:p.radius, ones(2*p.radius+1))
end

struct LDHeuristic <: Policy
    p::SimpleLightDark
    q::QMDPPolicy{SimpleLightDark, Int}
    std_thresh::Float64
end

struct LDHSolver <: Solver
    q::QMDPSolver
    std_thresh::Float64
end

LDHSolver(;std_thresh::Float64=0.1, kwargs...) = LDHSolver(QMDPSolver(;kwargs...), std_thresh)

solve(sol::LDHSolver, pomdp::SimpleLightDark) = LDHeuristic(pomdp, solve(sol.q, pomdp), sol.std_thresh)

action(p::LDHeuristic, s::Int) = action(p.q, s)
Base.srand(p::LDHeuristic, s) = p

function action(p::LDHeuristic, b::AbstractParticleBelief)
    s = std(particles(b))
    if s <= p.std_thresh
        return action(p.q, b)
    else
        m = mean(particles(b))
        ll = p.p.light_loc
        if m == ll
            return -1*Int(sign(ll))
        elseif abs(m-ll) >= 10 
            return -10*Int(sign(m-ll))
        else
            return -Int(sign(m-ll))
        end
    end
end
