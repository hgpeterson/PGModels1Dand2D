using SparseArrays
using LinearAlgebra
using JLD2

@doc raw"""
    LHS, RHS, rhs = build_linear_system(model::Model)
   

Set up the matrices and vectors for the non-dimensional 1D evolution equations.

The equations are:
```math
u_t = v - Px + S*b + (ν*u_z)_z
v_t = -u + (ν*v_z)_z
b_t = -u + [κ*(N² + b_z)]_z
```
where ``u``, ``v``, ``b`` are the non-dimensional cross-slope flow, along-slope flow, and 
buoyancy perturbation, respectively, and ``Px`` is the non-dimensional barotropic
pressure gradient. 

Boundary conditions are
```math
u = v = 0 \quad \text{at} \quad z = 0,
b_z = -N² \quad \text{at} \quad z = 0,
dz(u) = dz(v) = dz(b) = 0 \quad \text{at} \quad z = z_{top}
```
If `canonical=true`, the the barotropic pressure gradient is constant in time, equal to `v₀`. 
Otherwise, ``Px`` is set by the transport constraint ``∫ u dz = 0``.

They dynamics are determined by two non-dimensional numbers: the slope Burger number 
``S`` and background stratification ``N²``. ``ν`` and ``κ`` are 
the non-dimensional turbulent viscosity and diffusivity, respectively.

These equations are discretized using finite differences in the vertical direction, leading
to the matrix equation
```math
x_t = A*x + rhs
```
where ``x`` is the vector of variables.
We use a Crank-Nicolson time-stepping scheme,
```math
(x^(n+1) - x^n)/Δt = 1/2*[A*x^n + A*x^(n+1)] + rhs
```    
so that 
```math
LHS = I/Δt - 1/2*A,
RHS = I/Δt + 1/2*A.
```   
"""
function build_linear_system(model::Model)
    # unpack
    S = model.S
    N = model.N
    ν = model.ν
    κ = model.κ
    z = model.z
    r = model.r
    Δt = model.Δt
    canonical = model.canonical
    v₀ = model.v₀

    # setup
    nz, ndof, imap = get_dof(z)
    LHS = Tuple{Int64,Int64,Float64}[]  
    RHS = Tuple{Int64,Int64,Float64}[]  
    rhs = zeros(ndof)

    # Main loop, insert stencil in matrices for each node point
    for j=2:nz-1
        # dz stencil
        fd_z = mkfdstencil(z[j-1:j+1], z[j], 1)
        ν_z = sum(fd_z.*ν[j-1:j+1])
        κ_z = sum(fd_z.*κ[j-1:j+1])

        # dzz stencil
        fd_zz = mkfdstencil(z[j-1:j+1], z[j], 2)

        # 1st eqtn: u_t = v - Px + S*b + (ν*u_z)_z -r*u
        row = imap[1, j]
        # time derivative
        add_to_matrix!(LHS, row, row, 1/Δt)
        add_to_matrix!(RHS, row, row, 1/Δt)
        # first term RHS: v
        add_to_CN_matrices!(LHS, RHS, row, imap[2, j], 1)
        # second term RHS: -Px
        add_to_CN_matrices!(LHS, RHS, row, ndof, -1)
        # third term RHS: S*b
        add_to_CN_matrices!(LHS, RHS, row, imap[3, j], S)
        # fourth term RHS: (ν*u_z)_z = ν_z*u_z + ν*u_zz
        add_to_CN_matrices!(LHS, RHS, row, imap[1, j-1], ν_z*fd_z[1] + ν[j]*fd_zz[1])
        add_to_CN_matrices!(LHS, RHS, row, imap[1, j],   ν_z*fd_z[2] + ν[j]*fd_zz[2])
        add_to_CN_matrices!(LHS, RHS, row, imap[1, j+1], ν_z*fd_z[3] + ν[j]*fd_zz[3])
        # fifth term RHS: -r*u
        add_to_CN_matrices!(LHS, RHS, row, imap[1, j], -r[j])
        
        # 2nd eqtn: v_t = -u + (ν*v_z)_z - r*v
        row = imap[2, j]
        # time derivative
        add_to_matrix!(LHS, row, row, 1/Δt)
        add_to_matrix!(RHS, row, row, 1/Δt)
        # first term RHS: -u
        add_to_CN_matrices!(LHS, RHS, row, imap[1, j], -1)
        # second term RHS: (ν*v_z)_z = ν_z*v_z + ν*v_zz
        add_to_CN_matrices!(LHS, RHS, row, imap[2, j-1], ν_z*fd_z[1] + ν[j]*fd_zz[1])
        add_to_CN_matrices!(LHS, RHS, row, imap[2, j],   ν_z*fd_z[2] + ν[j]*fd_zz[2])
        add_to_CN_matrices!(LHS, RHS, row, imap[2, j+1], ν_z*fd_z[3] + ν[j]*fd_zz[3])
        # third term RHS: -r*v
        add_to_CN_matrices!(LHS, RHS, row, imap[2, j], -r[j])
        # rhs[row] = -r[j]*v₀ # not needed?

        # 3rd eqtn: b_t = -u + [κ*(N² + b_z)]_z - r*b
        row = imap[3, j]
        # time derivative
        add_to_matrix!(LHS, row, row, 1/Δt)
        add_to_matrix!(RHS, row, row, 1/Δt)
        # first term RHS: -u
        add_to_CN_matrices!(LHS, RHS, row, imap[1, j], -1)
        # second term RHS: [κ*(N² + b_z)]_z = N²κ_z + κ_z*b_z + κ*b_zz
        add_to_CN_matrices!(LHS, RHS, row, imap[3, j-1], (κ_z*fd_z[1] + κ[j]*fd_zz[1]))
        add_to_CN_matrices!(LHS, RHS, row, imap[3, j],   (κ_z*fd_z[2] + κ[j]*fd_zz[2]))
        add_to_CN_matrices!(LHS, RHS, row, imap[3, j+1], (κ_z*fd_z[3] + κ[j]*fd_zz[3]))
        rhs[row] = N^2*κ_z
        # third term RHS: -r*b
        # add_to_CN_matrices!(LHS, RHS, row, imap[3, j], -r)
    end

    # Boundary Conditions: Bottom
    # u = 0
    row = imap[1, 1] 
    add_to_matrix!(LHS, row, row, 1.0)
    # v = 0
    row = imap[2, 1] 
    add_to_matrix!(LHS, row, row, 1.0)
    # b_z = -N^2
    row = imap[3, 1] 
    fd_z = mkfdstencil(z[1:3], z[1], 1)
    add_to_matrix!(LHS, row, imap[3, 1], fd_z[1])
    add_to_matrix!(LHS, row, imap[3, 2], fd_z[2])
    add_to_matrix!(LHS, row, imap[3, 3], fd_z[3])
    rhs[row] = -N^2

    # Boundary Conditions: Top
    fd_z = mkfdstencil(z[nz-2:nz], z[nz], 1)
    # dz(u) = 0
    row = imap[1, nz] 
    add_to_matrix!(LHS, row, imap[1, nz-2], fd_z[1])
    add_to_matrix!(LHS, row, imap[1, nz-1], fd_z[2])
    add_to_matrix!(LHS, row, imap[1, nz],   fd_z[3])
    # dz(v) = 0
    row = imap[2, nz] 
    add_to_matrix!(LHS, row, imap[2, nz-2], fd_z[1])
    add_to_matrix!(LHS, row, imap[2, nz-1], fd_z[2])
    add_to_matrix!(LHS, row, imap[2, nz],   fd_z[3])
    # dz(b) = 0
    row = imap[3, nz]
    add_to_matrix!(LHS, row, imap[3, nz-2], fd_z[1])
    add_to_matrix!(LHS, row, imap[3, nz-1], fd_z[2])
    add_to_matrix!(LHS, row, imap[3, nz],   fd_z[3])

    # transport constraint
    row = ndof
    if canonical == true
        # canonical 1D: Px = constant in time (equal to v₀)
        add_to_matrix!(LHS, row, ndof, 1)
        rhs[row] = v₀
    else
        # transport-constrained 1D: Px such that ∫ u dz = 0
        for j=1:nz-1
            # trapezoidal rule: (u_j+1 + u_j)/2 * Δz_j
            add_to_matrix!(LHS, row, imap[1, j],   (z[j+1] - z[j])/2)
            add_to_matrix!(LHS, row, imap[1, j+1], (z[j+1] - z[j])/2)
        end
    end

    # Create CSC sparse matrices from matrix elements
    LHS = sparse((x->x[1]).(LHS), (x->x[2]).(LHS), (x->x[3]).(LHS), ndof, ndof)
    RHS = sparse((x->x[1]).(RHS), (x->x[2]).(RHS), (x->x[3]).(RHS), ndof, ndof)

    return lu(LHS), RHS, rhs
end
function get_dof(z)
    nz = length(z)
    ndof = 3*nz + 1 # number of degrees of freedom (u, v, b, Px)
    imap = reshape(1:(ndof-1), 3, nz)    
    return nz, ndof, imap
end
function add_to_matrix!(A, row, col, val)
    push!(A, (row, col, val))
end
function add_to_CN_matrices!(LHS, RHS, row, col, val)
    add_to_matrix!(LHS, row, col, -1/2*val)
    add_to_matrix!(RHS, row, col, +1/2*val)
end

"""
    u, v, b, Px = evolve(model::Model; t_final, t_save)

Solve 1D equations until `t_final`, saving every `t_save`.
"""
function evolve(model::Model; t_final, t_save)
    # unpack
    Δt = model.Δt
    z = model.z
    v₀ = model.v₀

    nz, ndof, imap = get_dof(z)

    # timestep
    n_steps = Int64(ceil(t_final/Δt))
    n_save = Int64(floor(t_save/Δt))

    # get matrices and vectors
    LHS, RHS, rhs = build_linear_system(model)

    # initial condition
    t = 0
    sol = zeros(ndof)
    # start with far-field geostrophic flow
    sol[imap[2, :]] .= v₀
    sol[ndof] = v₀
    u = @view sol[imap[1, :]]
    v = @view sol[imap[2, :]]
    b = @view sol[imap[3, :]]
    Px = @view sol[ndof]

    # save initial condition
    ofile = joinpath(out_dir, @sprintf("checkpoint%03d.jld2", 0))
    jldsave(ofile; u, v, b, Px, t, model)
    @info "Saved checkpoint to '$ofile'"
    i_save = 1

    # main loop
    for i=1:n_steps
        t += Δt

        if BT12 # Benthusyen and Thomas 2012 mixing scheme
            # update ν, κ based on current state
            model = set_ν_κ_BT12!(model, u, v, b, z)

            # rebuild the linear system with updated mixing coefficients
            LHS, RHS, rhs = build_linear_system(model)

            # debug
            if i % 10 == 0 && BT12_debug
                plot_κ_stratification(model, b; i)
            end
        end

        # solve linear system
        ldiv!(sol, LHS, RHS*sol + rhs)

        # log
        if i % n_save == 0
            # save data
            ofile = joinpath(out_dir, @sprintf("checkpoint%03d.jld2", i_save))
            jldsave(ofile; u, v, b, Px, t, model)
            @info "Saved '$ofile'"
            i_save += 1
        end
    end

    return u, v, b, Px
end

function set_ν_κ_BT12!(model, u, v, b, z)
    du_dz = differentiate(u, z)
    dv_dz = differentiate(v, z)
    db_dz = differentiate(b, z)

    # Richardson number
    Ri = (N^2 .+ db_dz) ./ (du_dz.^2 .+ dv_dz.^2 .+ 1e-12)

    # Benthusyen and Thomas 2012 mixing scheme
    κ_new = similar(model.κ)
    nz = length(z)
    for j in 1:nz
        if Ri[j] <= 0.2
            κ_new[j] = κ_b
        elseif Ri[j] < 0.3
            κ_new[j] = 10 * (model.κ[j] - κ_b) * (Ri[j] - 0.2) + κ_b
        else
            κ_new[j] = model.κ[j]
        end
    end

    # smooth κ_new with a 5-point moving average
    κ_smooth = copy(κ_new)
    for j in 3:nz-2
        κ_smooth[j] = (κ_new[j-2] + κ_new[j-1] + κ_new[j] + κ_new[j+1] + κ_new[j+2]) / 5
    end
    # keep boundaries unsmoothed
    κ_smooth[1] = κ_new[1]
    κ_smooth[2] = κ_new[2]
    κ_smooth[nz-1] = κ_new[nz-1]
    κ_smooth[nz] = κ_new[nz]

    # set ν, κ in model
    model.ν .= κ_smooth
    model.κ .= κ_smooth

    return model
end

function plot_κ_stratification(model, b; i=0)
    # unpack
    z = model.z
    κ = model.κ

    # init plot
    fig, ax = subplots(1, 2, figsize=(4, 3.2), sharey=true)
    ax[1].set_xlabel(L"Diffusivity $\tilde{\kappa}$")
    ax[2].set_xlabel(L"Stratification $N^2 + \partial_\tilde{z} \tilde{b}$")
    ax[1].set_ylabel(L"Vertical coordinate $\tilde{z}$")

    # plot
    ax[1].plot(κ, z)
    db_dz = differentiate(b, z)
    ax[2].plot(N^2 .+ db_dz, z)
    
    # save figure
    fname = joinpath(out_dir, @sprintf("kappa_strat%04d.png", i))
    savefig(fname)
    @info "Saved '$fname'"
    plt.close()
end