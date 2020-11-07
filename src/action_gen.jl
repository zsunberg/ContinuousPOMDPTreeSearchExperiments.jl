#=
struct GPSFirst
    rng::MersenneTwister
end

function POMCP.next_action(gen::GPSFirst, pomdp::PowseekerPOMDP, b, h)
    if n_children(h) == 0
        return GPSOrAngle(true, 0.0)
    else
        return GPSOrAngle(false, 2*pi*rand(gen.rng))
    end
end
=#
