filenames = Dict{String, String}()

tests = ["lasertag",
         "simpleld",
         "subhunt",
         "vdpbarrier"]

for t in tests
    try
        println("\n$(uppercase(t))\n")
        include(t*"_table.jl")
        filenames[t] = filename
    catch ex
        showerror(ex)
    end
end

for dt in ["subhunt_discretization.jl", "ld_discretization.jl"]
    try
        println("\n$(uppercase(dt))\n")
        include(dt)
        filenames[dt] = filename
    catch ex
        showerror(ex)
    end
end


