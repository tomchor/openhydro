using Oceananigans.Grids: xspacing, yspacing, zspacing
using Oceananigans.BoundaryConditions: Open, getbc
import Oceananigans.BoundaryConditions: _fill_west_open_halo!, _fill_east_open_halo!, _fill_south_open_halo!, _fill_north_open_halo!, _fill_bottom_open_halo!, _fill_north_open_halo!

struct FirstOrderRadiation{FT}
    relaxation_timescale :: FT
    ϕ⁻¹
end

FOROBC = BoundaryCondition{<:Open{<:FirstOrderRadiation}}

function FirstOrderRadiationOpenBoundaryCondition(val = nothing; relaxation_timescale = Inf, ϕ⁻¹ = nothing, kwargs...)
    classification = Open(FirstOrderRadiation(relaxation_timescale, ϕ⁻¹))
    return BoundaryCondition(classification, val; kwargs...)
end

@inline function relax(l, m, ϕ, bc, grid, clock, model_fields)
    Δt = clock.last_stage_Δt
    τ = bc.classification.matching_scheme.relaxation_timescale

    Δt̄ = min(1, Δt / τ)
    ϕₑₓₜ = getbc(bc, l, m, grid, clock, model_fields)

    Δϕ =  ifelse(isnothing(bc.condition)||!isfinite(clock.last_stage_Δt),
                 0, (ϕₑₓₜ - ϕ) * Δt̄)

    return ϕ + Δϕ
end

const C = Center()

@inline function _fill_west_open_halo!(j, k, grid, ϕ, bc::FOROBC, loc, clock, model_fields)
    gradient_free_c = @inbounds bc.classification.matching_scheme.ϕ⁻¹[1, j, k]
    @inbounds ϕ[1, j, k] = relax(j, k, gradient_free_c, bc, grid, clock, model_fields)

    return nothing
end

@inline function _fill_east_open_halo!(j, k, grid, ϕ, bc::FOROBC, loc, clock, model_fields)
    i = grid.Nx + 1

    gradient_free_c = @inbounds bc.classification.matching_scheme.ϕ⁻¹[1, j, k]
    @inbounds ϕ[i, j, k] = relax(j, k, gradient_free_c, bc, grid, clock, model_fields)

    return nothing
end

@inline function _fill_south_open_halo!(i, k, grid, ϕ, bc::FOROBC, loc, clock, model_fields)
    gradient_free_c = @inbounds bc.classification.matching_scheme.ϕ⁻¹[i, 1, k]
    @inbounds ϕ[i, 1, k] = relax(i, k, gradient_free_c, bc, grid, clock, model_fields)

    return nothing
end

@inline function _fill_north_open_halo!(i, k, grid, ϕ, bc::FOROBC, loc, clock, model_fields)
    j = grid.Ny + 1

    gradient_free_c = @inbounds bc.classification.matching_scheme.ϕ⁻¹[i, 1, k]
    @inbounds ϕ[i, j, k] = relax(i, k, gradient_free_c, bc, grid, clock, model_fields)

    return nothing
end

@inline function _fill_bottom_open_halo!(i, j, grid, ϕ, bc::FOROBC, loc, clock, model_fields)
    gradient_free_c = @inbounds bc.classification.matching_scheme.ϕ⁻¹[i, j, 1]
    @inbounds ϕ[i, j, 1] = relax(i, j, gradient_free_c, bc, grid, clock, model_fields)

    return nothing
end

@inline function _fill_top_open_halo!(i, j, grid, ϕ, bc::FOROBC, loc, clock, model_fields)
    k = grid.Nz + 1

    gradient_free_c = @inbounds bc.classification.matching_scheme.ϕ⁻¹[i, j, 1]
    @inbounds ϕ[i, j, k] = relax(i, j, gradient_free_c, bc, grid, clock, model_fields)

    return nothing
end

using Oceananigans: prognostic_fields
function update_first_order_radiation_matching_scheme!(sim)
    fields = prognostic_fields(sim.model)
    for (field_name, field) in zip(keys(fields), values(fields))
        bcs = field.boundary_conditions
        bcs.west   isa FOROBC && (interior(bcs.west.classification.matching_scheme.ϕ⁻¹,   1, :, :) .= interior(field, 2, :, :))
        bcs.east   isa FOROBC && (interior(bcs.east.classification.matching_scheme.ϕ⁻¹,   1, :, :) .= interior(field, grid.Nx, :, :))
        bcs.south  isa FOROBC && (interior(bcs.south.classification.matching_scheme.ϕ⁻¹,  :, 1, :) .= interior(field, :, 2, :))
        bcs.north  isa FOROBC && (interior(bcs.north.classification.matching_scheme.ϕ⁻¹,  :, 1, :) .= interior(field, :, grid.Ny, :))
        bcs.bottom isa FOROBC && (interior(bcs.bottom.classification.matching_scheme.ϕ⁻¹, :, :, 1) .= interior(field, :, :, 2))
        bcs.top    isa FOROBC && (interior(bcs.top.classification.matching_scheme.ϕ⁻¹,    :, :, 1) .= interior(field, :, :, grid.Nz))
    end
end
