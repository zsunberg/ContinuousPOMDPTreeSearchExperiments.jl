@with_kw mutable struct PDPWSolver <: AbstractPOMCPSolver
    rng::AbstractRNG        = Base.GLOBAL_RNG
    max_depth::Int          = 20
    c::Float64              = 1.0
    tree_queries::Int       = 1000
    max_time::Float64       = Inf
    k_action::Float64       = 10.0
    alpha_action::Float64   = 1/2
    k_observation::Float64  = 10.0
    alpha_observation::Float64 = 1/2
    enable_action_pw::Bool      = false
    check_repeat_obs::Bool      = false
    check_repeat_act::Bool      = false
    next_action::Any        = RandomActionGenerator(rng)
    default_action::Any     = ExceptionRethrow()
    estimate_value::Any     = RolloutEstimator(RandomSolver(rng))
end

struct PDPWTree{A,O,S}
    # for each observation-terminated history
    total_n::Vector{Int}
    children::Vector{Vector{Int}}
    o_labels::Vector{O}
    B::Vector{Vector{S}}

    # o_lookup::Dict{Tuple{Int, O}, Int}

    # for each action-terminated history
    n::Vector{Int}
    v::Vector{Float64}
    ha_children::Vector{Vector{Int}}
    a_labels::Vector{A}
end

function PDPWTree(pomdp::POMDP, b::ParticleCollection, enable_apw::Bool, sz::Int=1000)
    A = action_type(pomdp)
    O = obs_type(pomdp)
    S = state_type(pomdp)
    sz = min(100_000, sz)
    if enable_apw
        return PDPWTree{A,O,S}(sizehint!(Int[0], sz),
                               sizehint!(Vector{Int}[Int[]], sz),
                               sizehint!(Vector{O}(1), sz),
                               sizehint!([particles(b)], sz),

                               sizehint!(Int[], sz),
                               sizehint!(Float64[], sz),
                               sizehint!(Vector{Int}[], sz),
                               sizehint!(A[], sz)
                             )
    else
        acts = collect(iterator(actions(pomdp)))
        return PDPWTree{A,O,S}(sizehint!(Int[0], sz),
                              sizehint!(Vector{Int}[collect(1:length(acts))], sz),
                              sizehint!(Vector{O}(1), sz),
                              sizehint!([particles(b)], sz),

                              sizehint!(zeros(Int, length(acts)), sz),
                              sizehint!(zeros(Float64, length(acts)), sz),
                              sizehint!(fill(Int[], length(acts)), sz),
                              sizehint!(acts, sz)
                             )
    end
end    

function insert_obs_node!(t::PDPWTree, pomdp::POMDP, ha::Int, o, s, enable_apw::Bool)
    push!(t.total_n, 0)
    if enable_apw
        push!(t.children, Int[])
    else
        push!(t.children, sizehint!(Int[], n_actions(pomdp)))
    end
    push!(t.B, [s])
    push!(t.o_labels, o)
    hao = length(t.total_n)
    # t.o_lookup[(ha, o)] = hao
    if !enable_apw
        for a in iterator(actions(pomdp))
            n = insert_action_node!(t, hao, a)
            push!(t.children[hao], n)
        end
    end
    return hao
end

function insert_action_node!(t::PDPWTree, h::Int, a)
    push!(t.n, 0)
    push!(t.v, 0.0)
    push!(t.ha_children, Int[])
    push!(t.a_labels, a)
    return length(t.n)
end

struct PDPWObsNode{A,O} <: BasicPOMCP.BeliefNode
    tree::PDPWTree{A,O}
    node::Int
end

MCTS.isroot(n::PDPWObsNode) = n.node == 1
MCTS.n_children(n::PDPWObsNode) = length(n.tree.children[n.node])

mutable struct PDPWPlanner{P, SE, NA, RNG} <: Policy
    solver::PDPWSolver
    problem::P
    solved_estimator::SE
    next_action::NA
    rng::RNG
    _best_node_mem::Vector{Int}
    _tree::Nullable
end

function PDPWPlanner(solver::PDPWSolver, pomdp::POMDP)
    se = BasicPOMCP.convert_estimator(solver.estimate_value, solver, pomdp)
    @assert solver.check_repeat_obs == false
    @assert solver.check_repeat_act == false
    return PDPWPlanner(solver, pomdp, se, solver.next_action, solver.rng, Int[], Nullable())
end

solve(solver::PDPWSolver, pomdp::POMDP) = PDPWPlanner(solver, pomdp)

Base.srand(p::PDPWPlanner, seed) = srand(p.rng, seed)

function action(p::PDPWPlanner, b)
    local a::action_type(p.problem)
    try
        tree = PDPWTree(p.problem, b, p.solver.enable_action_pw, p.solver.tree_queries)
        a = search(p, b, tree)
        p._tree = Nullable(tree)
    catch ex
        # Note: this might not be type stable, but it shouldn't matter too much here
        a = convert(action_type(p.problem), BasicPOMCP.default_action(p.solver.default_action, p.problem, b, ex))
    end
    return a
end

function search(p::PDPWPlanner, b, t::PDPWTree)
    all_terminal = true
    start_us = CPUtime_us()
    for i in 1:p.solver.tree_queries
        if CPUtime_us() - start_us >= 1e6*p.solver.max_time
            break
        end
        s = rand(p.rng, b)
        if !POMDPs.isterminal(p.problem, s)
            simulate(p, s, PDPWObsNode(t, 1), p.solver.max_depth)
            all_terminal = false
        end
    end

    if all_terminal
        throw(AllSamplesTerminal(b))
    end

    h = 1
    best_node = first(t.children[h])
    best_v = t.v[best_node]
    @assert !isnan(best_v)
    for node in t.children[h][2:end]
        if t.v[node] >= best_v
            best_v = t.v[node]
            best_node = node
        end
    end

    return t.a_labels[best_node]
end


function simulate(p::PDPWPlanner, s, hnode::PDPWObsNode, steps::Int)
    if steps == 0 || isterminal(p.problem, s)
        return 0.0
    end
    
    t = hnode.tree
    h = hnode.node

    ka = p.solver.k_action
    aa = p.solver.alpha_action
    if p.solver.enable_action_pw && length(t.children[h]) <= ka*t.total_n[h]^aa
        a = next_action(p.next_action, p.problem, ParticleCollection(t.B[h]), hnode)
        n = insert_action_node!(t, h, a)
        push!(t.children[h], n)
    end

    ltn = log(t.total_n[h])
    best_nodes = empty!(p._best_node_mem)
    best_criterion_val = -Inf
    for node in t.children[h]
        n = t.n[node]
        if n == 0 && ltn <= 0.0
            criterion_value = t.v[node]
        elseif n == 0 && t.v[node] == -Inf
            criterion_value = Inf
        else
            criterion_value = t.v[node] + p.solver.c*sqrt(ltn/n)
        end
        if criterion_value > best_criterion_val
            best_criterion_val = criterion_value
            empty!(best_nodes)
            push!(best_nodes, node)
        elseif criterion_value == best_criterion_val
            push!(best_nodes, node)
        end
    end
    ha = rand(p.rng, best_nodes)
    a = t.a_labels[ha]

    if length(t.ha_children[ha]) <= p.solver.k_observation*t.n[ha]^p.solver.alpha_observation
        sp, o, r = generate_sor(p.problem, s, a, p.rng)

        hao = insert_obs_node!(t, p.problem, ha, o, sp, p.solver.enable_action_pw)
        v = BasicPOMCP.estimate_value(p.solved_estimator,
                                      p.problem,
                                      sp,
                                      PDPWObsNode(t, hao),
                                      steps-1)
        R = r + discount(p.problem)*v
    else
        hao = rand(p.rng, t.ha_children[ha])
        sp = rand(p.rng, t.B[hao])
        o = o_labels[hao]
        r = reward(s, a, sp)
        R = r + discount(p.problem)*simulate(p, sp, PDPWObsNode(t, hao), steps-1)
    end

    t.total_n[h] += 1
    t.n[ha] += 1
    t.v[ha] += (R-t.v[ha])/t.n[ha]

    return R
end
