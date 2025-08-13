using PyPlot
using PyCall
using JLD2

plt.style.use(joinpath(@__DIR__, "../../plots.mplstyle"))
close("all")
pygui(false)

pl = pyimport("matplotlib.pylab")
inset_locator = pyimport("mpl_toolkits.axes_grid1.inset_locator")
pc = 1/6 # pica

"""
    profile_plot(datafiles)

Plot profiles from JLD2 snapshot files in the `datafiles` list.
"""
function profile_plot(datafiles; fname=joinpath(out_dir, "profiles.png"))
    # init plot
    fig, ax = subplots(1, 3, figsize=(33pc, 33pc*1.62/3), sharey=true)


    ax[1].set_xlabel(L"Cross-slope flow $\tilde{u}$")
    ax[1].set_ylabel(L"Vertical coordinate $\tilde{z}$")

    ax[2].set_xlabel(L"Along-slope flow $\tilde{v}$")

    ax[3].set_xlabel(L"Stratification $N^2 + \partial_{\tilde z} \tilde b$")

    subplots_adjust(bottom=0.2, top=0.9, left=0.1, right=0.9, wspace=0.2, hspace=0.6)

    for a in ax
        a.ticklabel_format(style="sci", axis="x", scilimits=(-3, 3))
    end

    # color map
    colors = pl.cm.viridis(range(1, 0, length=size(datafiles, 1)-1))

    # # zoomed z
    # ax[1].set_ylim([0, 10])

    # plot data from `datafiles`
    for i ∈ eachindex(datafiles)
        # load
        d = jldopen(datafiles[i], "r")
        u = d["u"]
        v = d["v"]
        b = d["b"]
        Px = d["Px"]
        t = d["t"]
        model = d["model"]
        close(d)
        z = model.z
        N = model.N

        # stratification
        bz = differentiate(b, z)

        # colors and labels
        if t == Inf
            label = "Steady state"
        else
            label = string(L"$\tilde{t}/\tau_A$ = ", Int64(round(t/τ_A)))
        end
        if i == 1
            color = "r"
        else
            color = colors[i-1, :]
        end

        # plot
        ax[1].plot(u, z, c=color)
        ax[2].plot(v, z, c=color)
        ax[2].axvline(Px, lw=1.0, c=color, ls="--", label=(i == length(datafiles) ? L"\partial_{\tilde x} \tilde P" : ""))
        ax[3].plot(N^2 .+ bz,  z, c=color, label=label)
    end

    ax[2].legend()
    ax[3].legend()

    savefig(fname)
    println(fname)
    plt.close()
end
