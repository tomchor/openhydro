using Oceananigans.Grids: xspacing, yspacing, zspacing
using Oceananigans.BoundaryConditions: Open, getbc
import Oceananigans.BoundaryConditions: _fill_west_open_halo!, _fill_east_open_halo!, _fill_south_open_halo!, _fill_north_open_halo!, _fill_bottom_open_halo!, _fill_north_open_halo!

struct FlatExtrapolation2{FT}
    relaxation_timescale :: FT
end

FEOBC2 = BoundaryCondition{<:Open{<:FlatExtrapolation2}}

function FlatExtrapolation2OpenBoundaryCondition(val = nothing; relaxation_timescale = Inf, kwargs...)
    classification = Open(FlatExtrapolation2(relaxation_timescale))
    
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

@inline function _fill_west_open_halo!(j, k, grid, c, bc::FEOBC2, loc, clock, model_fields)
    Δx₁ = xspacing(1, j, k, grid, C, C, C)
    Δx₂ = xspacing(2, j, k, grid, C, C, C)
    Δx₃ = xspacing(3, j, k, grid, C, C, C)

    spacing_factor = Δx₁ / (Δx₂ + Δx₃)

    gradient_free_c = @inbounds c[3, j, k] - (c[2, j, k] - c[4, j, k]) * spacing_factor

    @inbounds c[1, j, k] = relax(j, k, gradient_free_c, bc, grid, clock, model_fields)

    return nothing
end

@inline function _fill_east_open_halo!(j, k, grid, c, bc::FEOBC2, loc, clock, model_fields)
    i = grid.Nx + 1

    Δx₁ = xspacing(i-1, j, k, grid, C, C, C)
    Δx₂ = xspacing(i-3, j, k, grid, C, C, C)
    Δx₃ = xspacing(i-3, j, k, grid, C, C, C)

    spacing_factor = Δx₁ / (Δx₂ + Δx₃)

    gradient_free_c = @inbounds c[i - 2, j, k] - (c[i - 1, j, k] - c[i - 3, j, k]) * spacing_factor

    @inbounds c[i, j, k] = relax(j, k, gradient_free_c, bc, grid, clock, model_fields)

    return nothing
end

@inline function _fill_south_open_halo!(i, k, grid, c, bc::FEOBC2, loc, clock, model_fields)
    Δy₁ = yspacing(i, 1, k, grid, C, C, C)
    Δy₂ = yspacing(i, 2, k, grid, C, C, C)
    Δy₃ = yspacing(i, 3, k, grid, C, C, C)

    spacing_factor = Δy₁ / (Δy₂ + Δy₃)

    gradient_free_c = c[i, 3, k] - (c[i, 2, k] - c[i, 4, k]) * spacing_factor

    @inbounds c[i, 1, k] = relax(i, k, gradient_free_c, bc, grid, clock, model_fields)
    
    return nothing
end

@inline function _fill_north_open_halo!(i, k, grid, c, bc::FEOBC2, loc, clock, model_fields)
    j = grid.Ny + 1

    Δy₁ = yspacing(i, j-1, k, grid, C, C, C)
    Δy₂ = yspacing(i, j-3, k, grid, C, C, C)
    Δy₃ = yspacing(i, j-3, k, grid, C, C, C)

    spacing_factor = Δy₁ / (Δy₂ + Δy₃)

    gradient_free_c = @inbounds c[i, j - 2, k] - (c[i, j - 1, k] - c[i, j - 3, k]) * spacing_factor

    @inbounds c[i, j, k] = relax(i, k, gradient_free_c, bc, grid, clock, model_fields)

    return nothing
end

@inline function _fill_bottom_open_halo!(i, j, grid, c, bc::FEOBC2, loc, clock, model_fields)
    Δz₁ = zspacing(i, j, 1, grid, C, C, C)
    Δz₂ = zspacing(i, j, 2, grid, C, C, C)
    Δz₃ = zspacing(i, j, 3, grid, C, C, C)

    spacing_factor = Δz₁ / (Δz₂ + Δz₃)

    gradient_free_c = @inbounds c[i, j, 3] - (c[i, k, 2] - c[i, j, 4]) * spacing_factor

    @inbounds c[i, j, 1] = relax(i, j, gradient_free_c, bc, grid, clock, model_fields)

    return nothing
end

@inline function _fill_top_open_halo!(i, j, grid, c, bc::FEOBC2, loc, clock, model_fields)
    k = grid.Nz + 1

    Δz₁ = zspacing(i, j, k-1, grid, C, C, C)
    Δz₂ = zspacing(i, j, k-3, grid, C, C, C)
    Δz₃ = zspacing(i, j, k-3, grid, C, C, C)

    spacing_factor = Δz₁ / (Δz₂ + Δz₃)

    gradient_free_c = @inbounds c[i, j, k - 2] - (c[i, j, k - 1] - c[i, j, k - 3]) * spacing_factor

    @inbounds c[i, j, k] = relax(i, j, gradient_free_c, bc, grid, clock, model_fields)

    return nothing
end
