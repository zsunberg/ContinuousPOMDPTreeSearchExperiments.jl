using DataFrames

d = Pkg.dir("ContinuousPOMDPTreeSearchExperiments", "icaps_2018","data")

solver_order = ["pomcpow", "qmdp", "pomcp-dpw", "despot", "pft", "d_pomcp", "d_despot"]

data = Dict(
    "lightdark" => Dict(
        "heuristic_01" => (24.469, 0.854),
        "despot_01" => (6.659, 1.267),
        "qmdp" => (5.287, 1.248),
        "pomcpow" => (62.179, 0.511),
        "pomcpdpw" => (5.297, 1.254),
        "pft" => (57.123, 0.396),
        "d_pomcp" => (64.496, 0.383),
        "d_despot" => (52.163, 1.346),
        "heuristic_1" => (28.216, 0.898),
        "pomcp" => (4.491, 1.238),
    ) 
)


