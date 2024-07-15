using Oceananigans.Grids: xspacing, yspacing, zspacing
using Oceananigans.BoundaryConditions: Open, getbc
import Oceananigans.BoundaryConditions: _fill_west_open_halo!, _fill_east_open_halo!, _fill_south_open_halo!, _fill_north_open_halo!, _fill_bottom_open_halo!, _fill_north_open_halo!

struct SecondOrderRadiation{FT}
    relaxation_timescale :: FT
end

SOROBC = BoundaryCondition{<:Open{<:SecondOrderRadiation}}

function SecondOrderRadiationOpenBoundaryCondition(val = nothing; relaxation_timescale = Inf, kwargs...)
    classification = Open(SecondOrderRadiation(relaxation_timescale))
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