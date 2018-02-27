using VDPTag2
using Plots

pyplot()

pomdp = VDPTagPOMDP(mdp=VDPTagMDP(barriers=CardinalBarriers(0.2, 2.8)))

quiver(pomdp)
