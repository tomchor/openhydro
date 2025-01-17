using Oceananigans.Grids: xspacing, yspacing, zspacing
using Oceananigans.BoundaryConditions: Open, getbc
import Oceananigans.BoundaryConditions: _fill_west_open_halo!, _fill_east_open_halo!, _fill_south_open_halo!, _fill_north_open_halo!, _fill_bottom_open_halo!, _fill_north_open_halo!

"""
Supercripts are time offsets
Subscripts are spatial offsets
"""
struct SecondOrderRadiation{FT}
    relaxation_timescale :: FT
    c⁻¹₋₁
    c⁻¹₋₂
    c⁻²₋₂
end

SOROBC = BoundaryCondition{<:Open{<:SecondOrderRadiation}}

function SecondOrderRadiationOpenBoundaryCondition(val = nothing; relaxation_timescale = Inf, c⁻¹₋₁ = nothing, c⁻¹₋₂ = nothing, c⁻²₋₂, kwargs...)
    classification = Open(SecondOrderRadiation(relaxation_timescale, c⁻¹₋₁, c⁻¹₋₂, c⁻²₋₂))
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
    gradient_free_c = @inbounds 2 * bc.classification.matching_scheme.c⁻¹₋₁[1, j, k] - bc.classification.matching_scheme.c⁻²₋₂[1, j, k]
    @inbounds c[1, j, k] = relax(j, k, gradient_free_c, bc, grid, clock, model_fields)

    return nothing
end

@inline function _fill_east_open_halo!(j, k, grid, c, bc::SOROBC, loc, clock, model_fields)
    i = grid.Nx + 1

    gradient_free_c = @inbounds 2 * bc.classification.matching_scheme.c⁻¹₋₁[1, j, k] - bc.classification.matching_scheme.c⁻²₋₂[1, j, k]
    @inbounds c[i, j, k] = relax(j, k, gradient_free_c, bc, grid, clock, model_fields)

    return nothing
end

@inline function _fill_south_open_halo!(i, k, grid, c, bc::SOROBC, loc, clock, model_fields)
    gradient_free_c = @inbounds 2 * bc.classification.matching_scheme.c⁻¹₋₁[1, j, k] - bc.classification.matching_scheme.c⁻²₋₂[1, j, k]
    @inbounds c[i, 1, k] = relax(i, k, gradient_free_c, bc, grid, clock, model_fields)

    return nothing
end

@inline function _fill_north_open_halo!(i, k, grid, c, bc::SOROBC, loc, clock, model_fields)
    j = grid.Ny + 1

    gradient_free_c = @inbounds 2 * bc.classification.matching_scheme.c⁻¹₋₁[1, j, k] - bc.classification.matching_scheme.c⁻²₋₂[1, j, k]
    @inbounds c[i, j, k] = relax(i, k, gradient_free_c, bc, grid, clock, model_fields)

    return nothing
end

@inline function _fill_bottom_open_halo!(i, j, grid, c, bc::SOROBC, loc, clock, model_fields)
    gradient_free_c = @inbounds 2 * bc.classification.matching_scheme.c⁻¹₋₁[1, j, k] - bc.classification.matching_scheme.c⁻²₋₂[1, j, k]
    @inbounds c[i, j, 1] = relax(i, j, gradient_free_c, bc, grid, clock, model_fields)

    return nothing
end

@inline function _fill_top_open_halo!(i, j, grid, c, bc::SOROBC, loc, clock, model_fields)
    k = grid.Nz + 1

    gradient_free_c = @inbounds 2 * bc.classification.matching_scheme.c⁻¹₋₁[1, j, k] - bc.classification.matching_scheme.c⁻²₋₂[1, j, k]
    @inbounds c[i, j, k] = relax(i, j, gradient_free_c, bc, grid, clock, model_fields)

    return nothing
end

using Oceananigans: prognostic_fields
function update_second_order_radiation_matching_scheme!(sim)
    fields = prognostic_fields(sim.model)
    for (field_name, field) in zip(keys(fields), values(fields))
        bcs = field.boundary_conditions
        bcs.west   isa SOROBC && (interior(bcs.west.classification.matching_scheme.c⁻²₋₂,   1, :, :) .= interior(bcs.west.classification.matching_scheme.c⁻¹₋₂, 1, :, :);
                                  interior(bcs.west.classification.matching_scheme.c⁻¹₋₂,   1, :, :) .= interior(field, 3, :, :);
                                  interior(bcs.west.classification.matching_scheme.c⁻¹₋₁,   1, :, :) .= interior(field, 2, :, :);)
        bcs.east   isa SOROBC && (interior(bcs.east.classification.matching_scheme.c⁻²₋₂,   1, :, :) .= interior(bcs.east.classification.matching_scheme.c⁻¹₋₂, 1, :, :);
                                  interior(bcs.east.classification.matching_scheme.c⁻¹₋₂,   1, :, :) .= interior(field, grid.Nx - 1, :, :);
                                  interior(bcs.east.classification.matching_scheme.c⁻¹₋₁,   1, :, :) .= interior(field, grid.Nx,     :, :))

        bcs.south  isa SOROBC && (interior(bcs.south.classification.matching_scheme.c⁻²₋₂,  :, 1, :) .= interior(bcs.south.classification.matching_scheme.c⁻¹₋₂, :, 1, :);
                                  interior(bcs.south.classification.matching_scheme.c⁻¹₋₂,  :, 1, :) .= interior(field, :, 3, :);
                                  interior(bcs.south.classification.matching_scheme.c⁻¹₋₁,  :, 1, :) .= interior(field, :, 2, :))
        bcs.north  isa SOROBC && (interior(bcs.north.classification.matching_scheme.c⁻²₋₂,  :, 1, :) .= interior(bcs.north.classification.matching_scheme.c⁻¹₋₂, :, 1, :);
                                  interior(bcs.north.classification.matching_scheme.c⁻¹₋₂,  :, 1, :) .= interior(field, :, grid.Nx - 1, :);
                                  interior(bcs.north.classification.matching_scheme.c⁻¹₋₁,  :, 1, :) .= interior(field, :, grid.Nx,     :))

        bcs.bottom isa SOROBC && (interior(bcs.bottom.classification.matching_scheme.c⁻²₋₂, :, :, 1) .= interior(bcs.bottom.classification.matching_scheme.c⁻¹₋₂, :, :, 1);
                                  interior(bcs.bottom.classification.matching_scheme.c⁻¹₋₂, :, :, 1) .= interior(field, :, :, 3);
                                  interior(bcs.bottom.classification.matching_scheme.c⁻¹₋₁, :, :, 1) .= interior(field, :, :, 2))
        bcs.top    isa SOROBC && (interior(bcs.top.classification.matching_scheme.c⁻²₋₂,    :, :, 1) .= interior(bcs.top.classification.matching_scheme.c⁻¹₋₂, :, 1, :);
                                  interior(bcs.top.classification.matching_scheme.c⁻¹₋₂,    :, :, 1) .= interior(field, :, :, grid.Nx - 1);
                                  interior(bcs.top.classification.matching_scheme.c⁻¹₋₁,    :, :, 1) .= interior(field, :, :, :, grid.Nz))
    end
end
