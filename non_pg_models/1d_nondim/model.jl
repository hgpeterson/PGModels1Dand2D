struct Model{T, V}
    S::T
    N::T
    v₀::T
    Δt::T
    z::V
    ν::V
    κ::V
    r::V
    canonical::Bool
end

function Model(S, v₀, N, Δt, z, ν, κ, r; canonical=false)
    nz = length(z)
    τ_A = 1/S
    H = z[end]
    τ_S = H

    # log properties
    ofile = joinpath(out_dir, "out.txt")
    open(ofile, "w") do f
        write(f, "Nondimensional 1D model with Parameters:\n\n")
        write(f, @sprintf("nz  = %1.5e\n", nz))
        write(f, @sprintf("τ_A = %1.5e\n", τ_A))
        write(f, @sprintf("τ_S = %1.5e\n", τ_S))
        write(f, @sprintf("H   = %1.5e\n", H))
        write(f, @sprintf("S   = %1.5e\n", S))
        write(f, @sprintf("v₀ = %1.5e\n", v₀))
        write(f, @sprintf("N   = %1.5e\n", N))
        write(f, @sprintf("Δt  = %1.5e\n", Δt))
        write(f, string("\nCanonical: ", canonical, "\n"))
        write(f, @sprintf("τ_A/τ_S  = %1.5e\n", τ_A/τ_S))
    end
    println("Wrote '$ofile' with contents:")
    open(ofile, "r") do f
        while !eof(f)
            println(readline(f))
        end
    end

    # make sure S, N, v₀, and Δt̃ are same type
    S, N, v₀, Δt = promote(S, N, v₀, Δt)

    # make sure z, ν, and κ are same type
    z = collect(z)

    return Model(S, N, v₀, Δt, z, ν, κ,r, canonical)
end