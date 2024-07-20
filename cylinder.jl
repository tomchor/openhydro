using Oceananigans, CairoMakie
using Oceananigans.BoundaryConditions: FlatExtrapolationOpenBoundaryCondition
include("first_order_radiation_open_boundary_scheme.jl")
include("second_order_radiation_open_boundary_scheme.jl")
include("orlanski_open_boundary_scheme.jl")

@kwdef struct Cylinder{FT}
    D :: FT = 1.
   x₀ :: FT = 0.
   y₀ :: FT = 0.
end

@inline (cylinder::Cylinder)(x, y) = ifelse((x - cylinder.x₀)^2 + (y - cylinder.y₀)^2 < (cylinder.D/2)^2, 1, 0)

architecture = CPU()

# model parameters
Re = 200
U = 1
D = 1.
resolution = D / 10
T = 20 / U

# add extra downstream distance to see if the solution near the cylinder changes
ΔL = 0

cylinder = Cylinder(; D)

L = 10
y = (-L/2, +L/2) .* D

Ny = Int(10 / resolution)
Nx = Ny + Int(ΔL / resolution)

β = 0.2
x_faces(i) = (L + ΔL)/2 * (β * ((2 * (i - 1)) / Nx - 1)^3 + (2 * (i - 1)) / Nx - 1) / (β+1)

ν = abs(U) * D / Re
Δt = .3 * resolution / abs(U)

closure = ScalarDiffusivity(; ν, κ = ν)
grid = @show RectilinearGrid(architecture; topology = (Bounded, Periodic, Flat), size = (Nx, Ny), x = x_faces, y)

@inline u∞(y, t, p) = p.U * cos(t * 2π / p.T) * (1 + 0.01 * randn())
@inline v∞(y, t, p) = p.U * sin(t * 2π / p.T) * (1 + 0.01 * randn())

function run_cylinder(boundary_conditions; plot=true, simname = "")
    @info "Testing $simname"

    u_forcing = Relaxation(; rate = 1 / (2 * Δt), mask = cylinder)
    v_forcing = Relaxation(; rate = 1 / (2 * Δt), mask = cylinder) 

    global model = NonhydrostaticModel(; grid,
                                       closure,
                                       forcing = (u = u_forcing, v = v_forcing),
                                       auxiliary_fields = (u⁻¹ = Field{Face, Center, Center}(grid),
                                                           u⁻² = Field{Face, Center, Center}(grid),
                                                           ),
                                       boundary_conditions)

    @info "Constructed model"

    # initial noise to induce turbulance faster
    set!(model, u = U, v = (x, y) -> randn() * U * 0.01)

    @info "Set initial conditions"
    simulation = Simulation(model; Δt = Δt, stop_time = 50)

    #+++ Callbacks
    wizard = TimeStepWizard(cfl = 0.3)
    simulation.callbacks[:wizard] = Callback(wizard, IterationInterval(100))

    progress(sim) = @info "$(time(sim)) with Δt = $(prettytime(sim.Δt)) in $(prettytime(sim.run_wall_time))"
    simulation.callbacks[:progress] = Callback(progress, IterationInterval(1000))

    if boundary_conditions.u.east.classification.matching_scheme isa FirstOrderRadiation
        update_first_order_radiation_matching_scheme!(simulation)
        simulation.callbacks[:update_bc] = Callback(update_first_order_radiation_matching_scheme!, IterationInterval(1))
    elseif boundary_conditions.u.east.classification.matching_scheme isa SecondOrderRadiation
        update_second_order_radiation_matching_scheme!(simulation); update_second_order_radiation_matching_scheme!(simulation)
        simulation.callbacks[:update_bc] = Callback(update_second_order_radiation_matching_scheme!, IterationInterval(1))
    elseif boundary_conditions.u.east.classification.matching_scheme isa Orlanski
        update_orlanski_matching_scheme!(simulation); update_orlanski_matching_scheme!(simulation)
        simulation.callbacks[:update_bc] = Callback(update_orlanski_matching_scheme!, IterationInterval(1))
    end

    function update_aux_fields!(sim)
        sim.model.auxiliary_fields.u⁻² .= sim.model.auxiliary_fields.u⁻¹
        sim.model.auxiliary_fields.u⁻¹ .= sim.model.velocities.u
    end
    update_aux_fields!(simulation); update_aux_fields!(simulation);
    simulation.callbacks[:update_aux] = Callback(update_aux_fields!, IterationInterval(1))

    time_step!(model, 0.1)
    #---

    u, v, w = model.velocities
    outputs = (; model.velocities..., ζ = (@at (Center, Center, Center) ∂x(v) - ∂y(u)))
    filename = "cylinder_$(simname)_$(ΔL)_Re_$Re.jld2"
    simulation.output_writers[:velocity] = JLD2OutputWriter(model, outputs,
                                                            overwrite_existing = true, 
                                                            filename = filename,
                                                            schedule = TimeInterval(0.5),
                                                            with_halos = true)

    run!(simulation)

    if plot
        # load the results 
        ζ_ts = FieldTimeSeries(filename, "ζ")
        @info "Loaded results"

        # plot the results
        fig = Figure(size = (600, 600))
        ax = Axis(fig[1, 1], aspect = DataAspect())
        xc, yc, zc = nodes(ζ_ts, with_halos=true)

        n = Observable(1)

        ζ_plt = @lift ζ_ts[:, :, 1, $n].parent
        heatmap!(ax, collect(xc), collect(yc), ζ_plt, colorrange = (-2, 2), colormap = :roma)
        record(fig, "ζ_$filename.mp4", 1:length(ζ_ts.times), framerate = 12) do i;
            n[] = i
            i % 10 == 0 && @info "$(n.val) of $(length(ζ_ts.times))"
        end
    end
end

c⁻¹   = Field{Nothing, Center, Center}(grid)
c⁻¹₋₁ = Field{Nothing, Center, Center}(grid)
c⁻¹₋₂ = Field{Nothing, Center, Center}(grid)
c⁻²₋₁ = Field{Nothing, Center, Center}(grid)
c⁻²₋₂ = Field{Nothing, Center, Center}(grid)

u_fo = FirstOrderRadiationOpenBoundaryCondition(u∞,  parameters = (; U, T), relaxation_timescale=1, c⁻¹ = c⁻¹₋₁)
u_so = SecondOrderRadiationOpenBoundaryCondition(u∞, parameters = (; U, T), relaxation_timescale=1; c⁻¹₋₁, c⁻¹₋₂, c⁻²₋₂)
u_or = OrlanskiOpenBoundaryCondition(u∞,             parameters = (; U, T), relaxation_timescale=1; c⁻¹, c⁻¹₋₁, c⁻¹₋₂, c⁻²₋₁)

u_boundaries = FieldBoundaryConditions(east = u_fo,
                                       west = u_fo)

v_boundaries = FieldBoundaryConditions(east = GradientBoundaryCondition(0),
                                       west = GradientBoundaryCondition(0))


boundary_conditions = (u = u_boundaries, v = v_boundaries)

run_cylinder(boundary_conditions, simname = nameof(typeof(u_boundaries.east.classification.matching_scheme)))

