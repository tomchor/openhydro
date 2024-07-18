using Oceananigans, CairoMakie
using Oceananigans.BoundaryConditions: FlatExtrapolationOpenBoundaryCondition
include("first_order_radiation_open_boundary_scheme.jl")
include("second_order_radiation_open_boundary_scheme.jl")

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

# add extra downstream distance to see if the solution near the cylinder changes
extra_downstream = 0

cylinder = Cylinder(; D)

x = (-5, 5 + extra_downstream) .* D
y = (-5, 5) .* D

Ny = Int(10 / resolution)
Nx = Ny + Int(extra_downstream / resolution)

ν = U * D / Re
Δt = .3 * resolution / U

closure = ScalarDiffusivity(; ν, κ = ν)
grid = RectilinearGrid(architecture; topology = (Bounded, Periodic, Flat), size = (Nx, Ny), x, y)

@inline u∞(y, t, U) = U * (1 + 0.01 * randn())

function run_cylinder(u_east_bc; plot=true)
    bc_name = string(nameof(typeof(u_east_bc.classification.matching_scheme)))
    @info "Testing $bc_name"

    u_boundaries = FieldBoundaryConditions(east = u_east_bc,
                                           west = OpenBoundaryCondition(u∞, parameters = U))

    v_boundaries = FieldBoundaryConditions(east = GradientBoundaryCondition(0),
                                           west = GradientBoundaryCondition(0))

    u_forcing = Relaxation(; rate = 1 / (2 * Δt), mask = cylinder)
    v_forcing = Relaxation(; rate = 1 / (2 * Δt), mask = cylinder) 

    global model = NonhydrostaticModel(; grid,
                                       closure,
                                       forcing = (u = u_forcing, v = v_forcing),
                                       auxiliary_fields = (u⁻¹ = Field{Face, Center, Center}(grid),
                                                           u⁻² = Field{Face, Center, Center}(grid),
                                                           ),
                                       boundary_conditions = (u = u_boundaries, v = v_boundaries))

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

    if u_east_bc.classification.matching_scheme isa FirstOrderRadiation
        update_first_order_radiation_matching_scheme!(simulation)
        simulation.callbacks[:update_bc] = Callback(update_first_order_radiation_matching_scheme!, IterationInterval(1))
    elseif u_east_bc.classification.matching_scheme isa SecondOrderRadiation
        update_second_order_radiation_matching_scheme!(simulation); update_second_order_radiation_matching_scheme!(simulation)
        simulation.callbacks[:update_bc] = Callback(update_second_order_radiation_matching_scheme!, IterationInterval(1))
    end

    function update_aux_fields!(sim)
        sim.model.auxiliary_fields.u⁻² .= sim.model.auxiliary_fields.u⁻¹
        sim.model.auxiliary_fields.u⁻¹ .= sim.model.velocities.u
    end
    update_aux_fields!(simulation); update_aux_fields!(simulation);
    simulation.callbacks[:update_aux] = Callback(update_aux_fields!, IterationInterval(1))
    #---

    u, v, w = model.velocities
    outputs = (; model.velocities..., ζ = (@at (Center, Center, Center) ∂x(v) - ∂y(u)))
    filename = "cylinder_$(bc_name)_$(extra_downstream)_Re_$Re.jld2"
    simulation.output_writers[:velocity] = JLD2OutputWriter(model, outputs,
                                                            overwrite_existing = true, 
                                                            filename = filename,
                                                            schedule = TimeInterval(0.5),
                                                            with_halos = true)

    run!(simulation)

    if plot
        # load the results 
        u_ts = FieldTimeSeries(filename, "u")
        v_ts = FieldTimeSeries(filename, "v")
        ζ_ts = FieldTimeSeries(filename, "ζ")

        @info "Loaded results"

        # plot the results
        fig = Figure(size = (600, 600))
        ax = Axis(fig[1, 1], aspect = DataAspect())
        xc, yc, zc = nodes(ζ_ts, with_halos=true)

        n = Observable(1)

        ζ_plt = @lift ζ_ts[:, :, 1, $n].parent
        heatmap!(ax, collect(xc), collect(yc), ζ_plt, colorrange = (-2, 2), colormap = :roma)
        record(fig, "ζ_$filename.mp4", 1:length(u_ts.times), framerate = 12) do i;
            n[] = i
            i % 10 == 0 && @info "$(n.val) of $(length(u_ts.times))"
        end
    end
end

c⁻¹₋₁ = Field{Nothing, Center, Center}(grid)
c⁻¹₋₂ = Field{Nothing, Center, Center}(grid)
c⁻²₋₂ = Field{Nothing, Center, Center}(grid)

u_east_fo = FirstOrderRadiationOpenBoundaryCondition(U, relaxation_timescale=1, c⁻¹ = c⁻¹₋₁)
u_east_so = SecondOrderRadiationOpenBoundaryCondition(U, relaxation_timescale=1; c⁻¹₋₁, c⁻¹₋₂, c⁻²₋₂)
run_cylinder(u_east_so)


