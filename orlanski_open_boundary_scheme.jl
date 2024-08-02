using Oceananigans.Grids: xspacing, yspacing, zspacing
using Oceananigans.BoundaryConditions: Open, getbc
import Oceananigans.BoundaryConditions: _fill_west_open_halo!, _fill_east_open_halo!, _fill_south_open_halo!, _fill_north_open_halo!, _fill_bottom_open_halo!, _fill_north_open_halo!

"""
Supercripts are time offsets
Subscripts are spatial offsets
"""
struct Orlanski{FT}
    relaxation_timescale :: FT
    ϕ⁻¹
    ϕ⁻¹₋₁
    ϕ⁻¹₋₂
    ϕ⁻²₋₁
end

OOBC = BoundaryCondition{<:Open{<:Orlanski}}

function OrlanskiOpenBoundaryCondition(val = nothing; relaxation_timescale = Inf, ϕ⁻¹ = nothing, ϕ⁻¹₋₁ = nothing, ϕ⁻¹₋₂ = nothing, ϕ⁻²₋₁ = nothing, kwargs...)
    classification = Open(Orlanski(relaxation_timescale, ϕ⁻¹, ϕ⁻¹₋₁, ϕ⁻¹₋₂, ϕ⁻²₋₁))
    return BoundaryCondition(classification, val; kwargs...)
end

@inline function relax(l, m, ϕ, bc, grid, clock, model_fields)
    Δt = clock.last_stage_Δt
    τ = bc.classification.matching_scheme.relaxation_timescale

    Δt̄ = min(1, Δt / τ)
    ϕₑₓₜ = getbc(bc, l, m, grid, clock, model_fields)

    Δc =  ifelse(isnothing(bc.condition)||!isfinite(clock.last_stage_Δt),
                 0, (ϕₑₓₜ - ϕ) * Δt̄)

    return ϕ + Δc
end

@inline function constrain_outflow(l, m, grid, bc, cᵢₙₜ, ϕₘₐₓ, Cᵩ_norm, clock, model_fields)
    ϕₑₓₜ = getbc(bc, l, m, grid, clock, model_fields)
    return @inbounds ifelse(Cᵩ_norm <= 0,
                            ϕₑₓₜ,
                            ifelse(Cᵩ_norm <= 1,
                                   cᵢₙₜ,
                                   ϕₘₐₓ))
end

const C = Center()

@inline function _fill_west_open_halo!(j, k, grid, ϕ, bc::OOBC, loc, clock, model_fields)
    i = 1
    Cᵩ_max = xspacing(2, j, k, grid, C, C, C) / (2 * clock.last_stage_Δt)
    Cᵩ = (ϕ[i + 1, j, k] - bc.classification.matching_scheme.ϕ⁻²₋₁[1, j, k]) * (Cᵩ_max / 2) /
         (ϕ[i + 1, j, k] + bc.classification.matching_scheme.ϕ⁻²₋₁[1, j, k] - bc.classification.matching_scheme.ϕ⁻¹₋₂[1, j, k])
    Cᵩ_norm = Cᵩ / Cᵩ_max

    cᵢₙₜ = @inbounds (1 - Cᵩ_norm) * bc.classification.matching_scheme.ϕ⁻¹[1, j, k] / (1 + Cᵩ_norm) + (2 * Cᵩ_norm * ϕ[i + 1, j, k]) / (1 + Cᵩ_norm)
    ϕₘₐₓ = @inbounds bc.classification.matching_scheme.ϕ⁻¹₋₁[1, j, k]
    @inbounds ϕ[i, j, k] = constrain_outflow(j, k, grid, bc, cᵢₙₜ, ϕₘₐₓ, Cᵩ_norm, clock, model_fields)

    return nothing
end

@inline function _fill_east_open_halo!(j, k, grid, ϕ, bc::OOBC, loc, clock, model_fields)
    i = grid.Nx + 1
    Cᵩ_max = xspacing(i - 2, j, k, grid, C, C, C) / (2 * clock.last_stage_Δt)
    Cᵩ = (ϕ[i - 1, j, k] - bc.classification.matching_scheme.ϕ⁻²₋₁[1, j, k]) * (Cᵩ_max / 2) /
         (ϕ[i - 1, j, k] + bc.classification.matching_scheme.ϕ⁻²₋₁[1, j, k] - bc.classification.matching_scheme.ϕ⁻¹₋₂[1, j, k])
    Cᵩ_norm = Cᵩ / Cᵩ_max
    cᵢₙₜ = @inbounds (1 - Cᵩ_norm) * bc.classification.matching_scheme.ϕ⁻¹[1, j, k] / (1 + Cᵩ_norm) + (2 * Cᵩ_norm * ϕ[i - 1, j, k]) / (1 + Cᵩ_norm)
    ϕₘₐₓ = @inbounds bc.classification.matching_scheme.ϕ⁻¹₋₁[1, j, k]
    @inbounds ϕ[i, j, k] = constrain_outflow(j, k, grid, bc, cᵢₙₜ, ϕₘₐₓ, Cᵩ_norm, clock, model_fields)
    return nothing
end

@inline function _fill_south_open_halo!(i, k, grid, ϕ, bc::OOBC, loc, clock, model_fields)
    j = 1
    Cᵩ_max = yspacing(i, 2, k, grid, C, C, C) / (2 * clock.last_stage_Δt)
    Cᵩ = (ϕ[i, j + 1, k] - bc.classification.matching_scheme.ϕ⁻²₋₁[i, 1, k]) * (Cᵩ_max / 2) /
         (ϕ[i, j + 1, k] + bc.classification.matching_scheme.ϕ⁻²₋₁[i, 1, k] - bc.classification.matching_scheme.ϕ⁻¹₋₂[i, 1, k])
    Cᵩ_norm = Cᵩ / Cᵩ_max

    cᵢₙₜ = @inbounds (1 - Cᵩ_norm) * bc.classification.matching_scheme.ϕ⁻¹[i, 1, k] / (1 + Cᵩ_norm) + (2 * Cᵩ_norm * ϕ[i, j + 1, k]) / (1 + Cᵩ_norm)
    ϕₘₐₓ = @inbounds bc.classification.matching_scheme.ϕ⁻¹₋₁[i, 1, k]
    @inbounds ϕ[i, j, k] = constrain_outflow(i, k, grid, bc, cᵢₙₜ, ϕₘₐₓ, Cᵩ_norm, clock, model_fields)

    return nothing
end

@inline function _fill_north_open_halo!(i, k, grid, ϕ, bc::OOBC, loc, clock, model_fields)
    j = grid.Ny + 1
    Cᵩ_max = yspacing(i, j - 2, k, grid, C, C, C) / (2 * clock.last_stage_Δt)
    Cᵩ = (ϕ[i, j - 1, k] - bc.classification.matching_scheme.ϕ⁻²₋₁[i, 1, k]) * (Cᵩ_max / 2) /
         (ϕ[i, j - 1, k] + bc.classification.matching_scheme.ϕ⁻²₋₁[i, 1, k] - bc.classification.matching_scheme.ϕ⁻¹₋₂[i, 1, k])
    Cᵩ_norm = Cᵩ / Cᵩ_max

    cᵢₙₜ = @inbounds (1 - Cᵩ_norm) * bc.classification.matching_scheme.ϕ⁻¹[i, 1, k] / (1 + Cᵩ_norm) + (2 * Cᵩ_norm * ϕ[i, j - 1, k]) / (1 + Cᵩ_norm)
    ϕₘₐₓ = @inbounds bc.classification.matching_scheme.ϕ⁻¹₋₁[i, 1, k]
    @inbounds ϕ[i, j, k] = constrain_outflow(i, k, grid, bc, cᵢₙₜ, ϕₘₐₓ, Cᵩ_norm, clock, model_fields)

    return nothing
end

@inline function _fill_bottom_open_halo!(i, j, grid, ϕ, bc::OOBC, loc, clock, model_fields)
    gradient_free_c = @inbounds 2 * bc.classification.matching_scheme.ϕ⁻¹₋₁[1, j, k] - bc.classification.matching_scheme.ϕ⁻²₋₂[1, j, k]
    @inbounds ϕ[i, j, 1] = relax(i, j, gradient_free_c, bc, grid, clock, model_fields)

    return nothing
end

@inline function _fill_top_open_halo!(i, j, grid, ϕ, bc::OOBC, loc, clock, model_fields)
    k = grid.Nz + 1

    gradient_free_c = @inbounds 2 * bc.classification.matching_scheme.ϕ⁻¹₋₁[1, j, k] - bc.classification.matching_scheme.ϕ⁻²₋₂[1, j, k]
    @inbounds ϕ[i, j, k] = relax(i, j, gradient_free_c, bc, grid, clock, model_fields)

    return nothing
end

using Oceananigans: prognostic_fields
function update_orlanski_matching_scheme!(sim)
    fields = prognostic_fields(sim.model)
    i = grid.Nx + 1
    j = grid.Ny + 1
    k = grid.Nz + 1
    for (field_name, field) in zip(keys(fields), values(fields))
        bcs = field.boundary_conditions
        bcs.west   isa OOBC && (interior(bcs.west.classification.matching_scheme.ϕ⁻²₋₁,   1, :, :) .= interior(bcs.west.classification.matching_scheme.ϕ⁻¹₋₁, 1, :, :);
                                interior(bcs.west.classification.matching_scheme.ϕ⁻¹₋₂,   1, :, :) .= interior(field, 3, :, :);
                                interior(bcs.west.classification.matching_scheme.ϕ⁻¹₋₁,   1, :, :) .= interior(field, 2, :, :);
                                interior(bcs.west.classification.matching_scheme.ϕ⁻¹,     1, :, :) .= interior(field, 1, :, :))
        bcs.east   isa OOBC && (interior(bcs.east.classification.matching_scheme.ϕ⁻²₋₁,   1, :, :) .= interior(bcs.east.classification.matching_scheme.ϕ⁻¹₋₁, 1, :, :);
                                interior(bcs.east.classification.matching_scheme.ϕ⁻¹₋₂,   1, :, :) .= interior(field, i - 2, :, :);
                                interior(bcs.east.classification.matching_scheme.ϕ⁻¹₋₁,   1, :, :) .= interior(field, i - 1, :, :);
                                interior(bcs.east.classification.matching_scheme.ϕ⁻¹,     1, :, :) .= interior(field, i,     :, :))

        bcs.south  isa OOBC && (interior(bcs.south.classification.matching_scheme.ϕ⁻²₋₁,  :, 1, :) .= interior(bcs.south.classification.matching_scheme.ϕ⁻¹₋₁, :, 1, :);
                                interior(bcs.south.classification.matching_scheme.ϕ⁻¹₋₂,  :, 1, :) .= interior(field, :, 3, :);
                                interior(bcs.south.classification.matching_scheme.ϕ⁻¹₋₁,  :, 1, :) .= interior(field, :, 2, :);
                                interior(bcs.south.classification.matching_scheme.ϕ⁻¹,    :, 1, :) .= interior(field, :, 1, :))
        bcs.north  isa OOBC && (interior(bcs.north.classification.matching_scheme.ϕ⁻²₋₁,  :, 1, :) .= interior(bcs.north.classification.matching_scheme.ϕ⁻¹₋₁, :, 1, :);
                                interior(bcs.north.classification.matching_scheme.ϕ⁻¹₋₂,  :, 1, :) .= interior(field, :, j - 2, :);
                                interior(bcs.north.classification.matching_scheme.ϕ⁻¹₋₁,  :, 1, :) .= interior(field, :, j - 1, :);
                                interior(bcs.north.classification.matching_scheme.ϕ⁻¹,    :, 1, :) .= interior(field, :, j,     :))

        bcs.bottom isa OOBC && (interior(bcs.bottom.classification.matching_scheme.ϕ⁻²₋₁, :, :, 1) .= interior(bcs.bottom.classification.matching_scheme.ϕ⁻¹₋₂, :, :, 1);
                                interior(bcs.bottom.classification.matching_scheme.ϕ⁻¹₋₂, :, :, 1) .= interior(field, :, :, 3);
                                interior(bcs.bottom.classification.matching_scheme.ϕ⁻¹₋₁, :, :, 1) .= interior(field, :, :, 2))
        bcs.top    isa OOBC && (interior(bcs.top.classification.matching_scheme.ϕ⁻²₋₁,    :, :, 1) .= interior(bcs.top.classification.matching_scheme.ϕ⁻¹₋₁, :, :, 1);
                                interior(bcs.top.classification.matching_scheme.ϕ⁻¹₋₂,    :, :, 1) .= interior(field, :, :, k - 2);
                                interior(bcs.top.classification.matching_scheme.ϕ⁻¹₋₁,    :, :, 1) .= interior(field, :, :, k - 1))
    end
end
