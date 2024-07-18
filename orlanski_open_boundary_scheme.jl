using Oceananigans.Grids: xspacing, yspacing, zspacing
using Oceananigans.BoundaryConditions: Open, getbc
import Oceananigans.BoundaryConditions: _fill_west_open_halo!, _fill_east_open_halo!, _fill_south_open_halo!, _fill_north_open_halo!, _fill_bottom_open_halo!, _fill_north_open_halo!

"""
Supercripts are time offsets
Subscripts are spatial offsets
"""
struct Orlanski{FT}
    relaxation_timescale :: FT
    c⁻¹₋₁
    c⁻¹₋₂
    c⁻²₋₁
    c⁻²₋₂
end

OOBC = BoundaryCondition{<:Open{<:Orlanski}}

function OrlanskiOpenBoundaryCondition(val = nothing; relaxation_timescale = Inf, c⁻¹₋₁ = nothing, c⁻¹₋₂ = nothing, c⁻²₋₁ = nothing, c⁻²₋₂ = nothing, kwargs...)
    classification = Open(Orlanski(relaxation_timescale, c⁻¹₋₁, c⁻¹₋₂, c⁻²₋₁, c⁻²₋₂))
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

@inline function constrain_outflow(l, m, cₑₓₜ, cᵢₙₜ, cₘₐₓ, Cᵩ_norm)
    return @inbounds ifelse(Cᵩ_norm <= 0,
                            cₑₓₜ,
                            ifelse(Cᵩ_norm <= 1,
                                   cᵢₙₜ,
                                   cₘₐₓ))

end

const C = Center()

@inline function _fill_west_open_halo!(j, k, grid, c, bc::OOBC, loc, clock, model_fields)
    gradient_free_c = @inbounds 2 * bc.classification.matching_scheme.c⁻¹₋₁[1, j, k] - bc.classification.matching_scheme.c⁻²₋₂[1, j, k]
    @inbounds c[1, j, k] = relax(j, k, gradient_free_c, bc, grid, clock, model_fields)

    return nothing
end

@inline function _fill_east_open_halo!(j, k, grid, c, bc::OOBC, loc, clock, model_fields)
    i = grid.Nx + 1
    Cᵩ_max = xspacing(i - 2, j, k, grid, Center(), Center(), Center()) / (2 * clock.last_stage_Δt)
    Cᵩ = (c[i - 1, j, k] - bc.classification.matching_scheme.c⁻²₋₁[1, j, k]) * (Cᵩ_max / 2) /
         (c[i - 1, j, k] + bc.classification.matching_scheme.c⁻²₋₁[1, j, k] - bc.classification.matching_scheme.c⁻¹₋₂[1, j, k])
    Cᵩ_norm = Cᵩ / Cᵩ_max

    cₑₓₜ = getbc(bc, j, k, grid, clock, model_fields)
    cᵢₙₜ = @inbounds (1 - Cᵩ_norm) * bc.classification.matching_scheme.c⁻¹₋₂[1, j, k] / (1 + Cᵩ_norm) + (2 * Cᵩ_norm * c[i - 1, j, k]) / (1 + Cᵩ_norm)
    cₘₐₓ = @inbounds bc.classification.matching_scheme.c⁻¹₋₁[1, j, k]
    @inbounds c[i, j, k] = constrain_outflow(j, k, cₑₓₜ, cᵢₙₜ, cₘₐₓ, Cᵩ_norm)

    return nothing
end

@inline function _fill_south_open_halo!(i, k, grid, c, bc::OOBC, loc, clock, model_fields)
    gradient_free_c = @inbounds 2 * bc.classification.matching_scheme.c⁻¹₋₁[1, j, k] - bc.classification.matching_scheme.c⁻²₋₂[1, j, k]
    @inbounds c[i, 1, k] = relax(i, k, gradient_free_c, bc, grid, clock, model_fields)

    return nothing
end

@inline function _fill_north_open_halo!(i, k, grid, c, bc::OOBC, loc, clock, model_fields)
    j = grid.Ny + 1

    gradient_free_c = @inbounds 2 * bc.classification.matching_scheme.c⁻¹₋₁[1, j, k] - bc.classification.matching_scheme.c⁻²₋₂[1, j, k]
    @inbounds c[i, j, k] = relax(i, k, gradient_free_c, bc, grid, clock, model_fields)

    return nothing
end

@inline function _fill_bottom_open_halo!(i, j, grid, c, bc::OOBC, loc, clock, model_fields)
    gradient_free_c = @inbounds 2 * bc.classification.matching_scheme.c⁻¹₋₁[1, j, k] - bc.classification.matching_scheme.c⁻²₋₂[1, j, k]
    @inbounds c[i, j, 1] = relax(i, j, gradient_free_c, bc, grid, clock, model_fields)

    return nothing
end

@inline function _fill_top_open_halo!(i, j, grid, c, bc::OOBC, loc, clock, model_fields)
    k = grid.Nz + 1

    gradient_free_c = @inbounds 2 * bc.classification.matching_scheme.c⁻¹₋₁[1, j, k] - bc.classification.matching_scheme.c⁻²₋₂[1, j, k]
    @inbounds c[i, j, k] = relax(i, j, gradient_free_c, bc, grid, clock, model_fields)

    return nothing
end

using Oceananigans: prognostic_fields
function update_orlanski_matching_scheme!(sim)
    fields = prognostic_fields(sim.model)
    for (field_name, field) in zip(keys(fields), values(fields))
        bcs = field.boundary_conditions
        bcs.west   isa OOBC && (interior(bcs.west.classification.matching_scheme.c⁻²₋₂,   1, :, :) .= interior(bcs.west.classification.matching_scheme.c⁻¹₋₂, 1, :, :);
                                interior(bcs.west.classification.matching_scheme.c⁻¹₋₂,   1, :, :) .= interior(field, 3, :, :);
                                interior(bcs.west.classification.matching_scheme.c⁻¹₋₁,   1, :, :) .= interior(field, 2, :, :);)
        bcs.east   isa OOBC && (interior(bcs.east.classification.matching_scheme.c⁻²₋₂,   1, :, :) .= interior(bcs.east.classification.matching_scheme.c⁻¹₋₂, 1, :, :);
                                interior(bcs.east.classification.matching_scheme.c⁻²₋₁,   1, :, :) .= interior(bcs.east.classification.matching_scheme.c⁻¹₋₁, 1, :, :);
                                interior(bcs.east.classification.matching_scheme.c⁻¹₋₂,   1, :, :) .= interior(field, grid.Nx - 1, :, :);
                                interior(bcs.east.classification.matching_scheme.c⁻¹₋₁,   1, :, :) .= interior(field, grid.Nx,     :, :))

        bcs.south  isa OOBC && (interior(bcs.south.classification.matching_scheme.c⁻²₋₂,  :, 1, :) .= interior(bcs.south.classification.matching_scheme.c⁻¹₋₂, :, 1, :);
                                interior(bcs.south.classification.matching_scheme.c⁻¹₋₂,  :, 1, :) .= interior(field, :, 3, :);
                                interior(bcs.south.classification.matching_scheme.c⁻¹₋₁,  :, 1, :) .= interior(field, :, 2, :))
        bcs.north  isa OOBC && (interior(bcs.north.classification.matching_scheme.c⁻²₋₂,  :, 1, :) .= interior(bcs.north.classification.matching_scheme.c⁻¹₋₂, :, 1, :);
                                interior(bcs.north.classification.matching_scheme.c⁻¹₋₂,  :, 1, :) .= interior(field, :, grid.Nx - 1, :);
                                interior(bcs.north.classification.matching_scheme.c⁻¹₋₁,  :, 1, :) .= interior(field, :, grid.Nx,     :))

        bcs.bottom isa OOBC && (interior(bcs.bottom.classification.matching_scheme.c⁻²₋₂, :, :, 1) .= interior(bcs.bottom.classification.matching_scheme.c⁻¹₋₂, :, :, 1);
                                interior(bcs.bottom.classification.matching_scheme.c⁻¹₋₂, :, :, 1) .= interior(field, :, :, 3);
                                interior(bcs.bottom.classification.matching_scheme.c⁻¹₋₁, :, :, 1) .= interior(field, :, :, 2))
        bcs.top    isa OOBC && (interior(bcs.top.classification.matching_scheme.c⁻²₋₂,    :, :, 1) .= interior(bcs.top.classification.matching_scheme.c⁻¹₋₂, :, 1, :);
                                interior(bcs.top.classification.matching_scheme.c⁻¹₋₂,    :, :, 1) .= interior(field, :, :, grid.Nx - 1);
                                interior(bcs.top.classification.matching_scheme.c⁻¹₋₁,    :, :, 1) .= interior(field, :, :, :, grid.Nz))
    end
end
