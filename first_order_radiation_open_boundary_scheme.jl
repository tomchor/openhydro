using Oceananigans.Grids: xspacing, yspacing, zspacing
using Oceananigans.BoundaryConditions: Open, getbc
import Oceananigans.BoundaryConditions: _fill_west_open_halo!, _fill_east_open_halo!, _fill_south_open_halo!, _fill_north_open_halo!, _fill_bottom_open_halo!, _fill_north_open_halo!

struct FirstOrderRadiation{FT}
    relaxation_timescale :: FT
    c⁻¹
end

FOROBC = BoundaryCondition{<:Open{<:FirstOrderRadiation}}

function FirstOrderRadiationOpenBoundaryCondition(val = nothing; relaxation_timescale = Inf, c⁻¹ = nothing, kwargs...)
    classification = Open(FirstOrderRadiation(relaxation_timescale, c⁻¹))
    return BoundaryCondition(classification, val; kwargs...)
end

@inline function relax(l, m, c, bc, grid, clock, model_fields)
    Δt = clock.last_stage_Δt
    τ = bc.classification.matching_scheme.relaxation_timescale

    Δt̄ = min(1, Δt / τ)
    cₑₓₜ = getbc(bc, l, m, grid, clock, model_fields)

    Δc =  ifelse(isnothing(bc.condition)||!isfinite(clock.last_stage_Δt),
                 0, (cₑₓₜ - c) * Δt̄)

    return c + Δc
end

const C = Center()

@inline function _fill_west_open_halo!(j, k, grid, c, bc::FOROBC, loc, clock, model_fields)
    gradient_free_c = @inbounds bc.classification.matching_scheme.c⁻¹[1, j, k]
    @inbounds c[1, j, k] = relax(j, k, gradient_free_c, bc, grid, clock, model_fields)

    return nothing
end

@inline function _fill_east_open_halo!(j, k, grid, c, bc::FOROBC, loc, clock, model_fields)
    i = grid.Nx + 1

    gradient_free_c = @inbounds bc.classification.matching_scheme.c⁻¹[1, j, k]
    @inbounds c[i, j, k] = relax(j, k, gradient_free_c, bc, grid, clock, model_fields)

    return nothing
end

@inline function _fill_south_open_halo!(i, k, grid, c, bc::FOROBC, loc, clock, model_fields)
    gradient_free_c = @inbounds bc.classification.matching_scheme.c⁻¹[1, j, k]
    @inbounds c[i, 1, k] = relax(i, k, gradient_free_c, bc, grid, clock, model_fields)

    return nothing
end

@inline function _fill_north_open_halo!(i, k, grid, c, bc::FOROBC, loc, clock, model_fields)
    j = grid.Ny + 1

    gradient_free_c = @inbounds bc.classification.matching_scheme.c⁻¹[1, j, k]
    @inbounds c[i, j, k] = relax(i, k, gradient_free_c, bc, grid, clock, model_fields)

    return nothing
end

@inline function _fill_bottom_open_halo!(i, j, grid, c, bc::FOROBC, loc, clock, model_fields)
    gradient_free_c = @inbounds bc.classification.matching_scheme.c⁻¹[1, j, k]
    @inbounds c[i, j, 1] = relax(i, j, gradient_free_c, bc, grid, clock, model_fields)

    return nothing
end

@inline function _fill_top_open_halo!(i, j, grid, c, bc::FOROBC, loc, clock, model_fields)
    k = grid.Nz + 1

    gradient_free_c = @inbounds bc.classification.matching_scheme.c⁻¹[1, j, k]
    @inbounds c[i, j, k] = relax(i, j, gradient_free_c, bc, grid, clock, model_fields)

    return nothing
end

using Oceananigans: prognostic_fields
function update_first_order_radiation_matching_scheme!(sim)
    fields = prognostic_fields(sim.model)
    for (field_name, field) in zip(keys(fields), values(fields))
        bcs = field.boundary_conditions
        bcs.east   isa FOROBC && (interior(sim.model.velocities.u.boundary_conditions.east.classification.matching_scheme.c⁻¹,   1, :, :) .= interior(sim.model.velocities.u, grid.Nx, :, :))
        bcs.west   isa FOROBC && (interior(sim.model.velocities.u.boundary_conditions.west.classification.matching_scheme.c⁻¹,   1, :, :) .= interior(sim.model.velocities.u, 2, :, :))
        bcs.south  isa FOROBC && (interior(sim.model.velocities.v.boundary_conditions.south.classification.matching_scheme.c⁻¹,  :, 1, :) .= interior(sim.model.velocities.v, :, grid.Ny, :))
        bcs.north  isa FOROBC && (interior(sim.model.velocities.v.boundary_conditions.north.classification.matching_scheme.c⁻¹,  :, 1, :) .= interior(sim.model.velocities.v, :, 2, :))
        bcs.bottom isa FOROBC && (interior(sim.model.velocities.w.boundary_conditions.bottom.classification.matching_scheme.c⁻¹, :, :, 1) .= interior(sim.model.velocities.w, :, :, grid.Nz))
        bcs.top    isa FOROBC && (interior(sim.model.velocities.w.boundary_conditions.top.classification.matching_scheme.c⁻¹,    :, :, 1) .= interior(sim.model.velocities.w, :, :, 2))
    end
end
