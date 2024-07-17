using Oceananigans.Grids: xspacing, yspacing, zspacing
using Oceananigans.BoundaryConditions: Open, getbc
import Oceananigans.BoundaryConditions: _fill_west_open_halo!, _fill_east_open_halo!, _fill_south_open_halo!, _fill_north_open_halo!, _fill_bottom_open_halo!, _fill_north_open_halo!

struct SecondOrderRadiation{FT}
    relaxation_timescale :: FT
    c⁻¹
    c⁻²
end

SOROBC = BoundaryCondition{<:Open{<:SecondOrderRadiation}}

function SecondOrderRadiationOpenBoundaryCondition(val = nothing; relaxation_timescale = Inf, c⁻¹ = nothing, c⁻² = nothing, kwargs...)
    classification = Open(SecondOrderRadiation(relaxation_timescale, c⁻¹, c⁻²))
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

@inline function _fill_west_open_halo!(j, k, grid, c, bc::SOROBC, loc, clock, model_fields)
    gradient_free_c = @inbounds 2 * model_fields.u⁻¹[2, j, k] - model_fields.u⁻²[3, j, k]
    @inbounds c[1, j, k] = relax(j, k, gradient_free_c, bc, grid, clock, model_fields)

    return nothing
end

@inline function _fill_east_open_halo!(j, k, grid, c, bc::SOROBC, loc, clock, model_fields)
    i = grid.Nx + 1

    gradient_free_c = @inbounds 2 * model_fields.u⁻¹[i - 1, j, k] - model_fields.u⁻²[i - 2, j, k]
    #gradient_free_c = @inbounds 2 * bc.classification.matching_scheme.c⁻¹[1, j, k] - bc.classification.matching_scheme.c⁻²[1, j, k]
    @inbounds c[i, j, k] = relax(j, k, gradient_free_c, bc, grid, clock, model_fields)

    return nothing
end

@inline function _fill_south_open_halo!(i, k, grid, c, bc::SOROBC, loc, clock, model_fields)
    gradient_free_c = @inbounds 2 * model_fields.v⁻¹[i, 2, k] - model_fields.v⁻²[i, 3, k]
    @inbounds c[i, 1, k] = relax(i, k, gradient_free_c, bc, grid, clock, model_fields)

    return nothing
end

@inline function _fill_north_open_halo!(i, k, grid, c, bc::SOROBC, loc, clock, model_fields)
    j = grid.Ny + 1

    gradient_free_c = @inbounds 2 * model_fields.v⁻¹[i, j - 1, k] - model_fields.v⁻²[i, j - 2, k]
    @inbounds c[i, j, k] = relax(i, k, gradient_free_c, bc, grid, clock, model_fields)

    return nothing
end

@inline function _fill_bottom_open_halo!(i, j, grid, c, bc::SOROBC, loc, clock, model_fields)
    gradient_free_c = @inbounds 2 * model_fields.w⁻¹[i, j, 2] - model_fields.w⁻²[i, j, 3]
    @inbounds c[i, j, 1] = relax(i, j, gradient_free_c, bc, grid, clock, model_fields)

    return nothing
end

@inline function _fill_top_open_halo!(i, j, grid, c, bc::SOROBC, loc, clock, model_fields)
    k = grid.Nz + 1

    gradient_free_c = @inbounds 2 * model_fields.w⁻¹[i, j, k - 1] - model_fields.w⁻²[i, j, k - 2]
    @inbounds c[i, j, k] = relax(i, j, gradient_free_c, bc, grid, clock, model_fields)

    return nothing
end

using Oceananigans: prognostic_fields
function update_second_order_radiation_matching_scheme!(sim)
    fields = prognostic_fields(sim.model)
    for (field_name, field) in zip(keys(fields), values(fields))
        bcs = field.boundary_conditions
        bcs.east   isa SOROBC && (interior(bcs.east.classification.matching_scheme.c⁻²,   1, :, :) .= interior(bcs.east.classification.matching_scheme.c⁻¹, 1, :, :); # This needs to be on grid.Nx-1
                                  interior(bcs.east.classification.matching_scheme.c⁻¹,   1, :, :) .= interior(field, grid.Nx, :, :))
        bcs.west   isa SOROBC && (interior(bcs.west.classification.matching_scheme.c⁻¹,   1, :, :) .= interior(field, 2, :, :))
        bcs.south  isa SOROBC && (interior(bcs.south.classification.matching_scheme.c⁻¹,  :, 1, :) .= interior(field, :, grid.Ny, :))
        bcs.north  isa SOROBC && (interior(bcs.north.classification.matching_scheme.c⁻¹,  :, 1, :) .= interior(field, :, 2, :))
        bcs.bottom isa SOROBC && (interior(bcs.bottom.classification.matching_scheme.c⁻¹, :, :, 1) .= interior(field, :, :, grid.Nz))
        bcs.top    isa SOROBC && (interior(bcs.top.classification.matching_scheme.c⁻¹,    :, :, 1) .= interior(field, :, :, 2))
    end
end
