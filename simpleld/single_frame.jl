using Plots
using ContinuousPOMDPTreeSearchExperiments

gr()

pomdp = SimpleLightDark()

tmax = 80
smin = -10
smax = 20
T = linspace(0.0, tmax)
S = linspace(-1.0, 21.0)
inv_grays = cgrad([RGB(1.0, 1.0, 1.0),RGB(0.0,0.0,0.0)])

p = contour(T, S, (t,s)->abs(s-pomdp.light_loc),
        bg_inside=:black,
        fill=true,
        xlim=(0, tmax),
        ylim=(smin, smax),
        color=inv_grays,
        # xlabel="Time",
        # ylabel="State",
        cbar=false,
        legend=:topright,
       )

plot!(p, [0, tmax], [0, 0], linewidth=2, color="green", label="", line=:dash)
@show fname = tempname()*".svg"
savefig(p, fname)
run(`inkscape $fname`)
