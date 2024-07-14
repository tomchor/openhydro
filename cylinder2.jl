using Oceananigans, CairoMakie
using Oceananigans.BoundaryConditions: FlatExtrapolationOpenBoundaryCondition
include("first_order_radiation_open_boundary_scheme.jl")

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

closure = ScalarDiffusivity(;ν, κ = ν)

grid = RectilinearGrid(architecture; topology = (Bounded, Periodic, Flat), size = (Nx, Ny), x, y)

@inline u∞(y, t, U) = U * (1 + 0.01 * randn())

u_boundaries = FieldBoundaryConditions(east = FirstOrderRadiationOpenBoundaryCondition(Field{Nothing, Center, Center}(grid),
                                                                                       U, relaxation_timescale=1),
#u_boundaries = FieldBoundaryConditions(east = OpenBoundaryCondition(U),
#u_boundaries = FieldBoundaryConditions(east = FlatExtrapolationOpenBoundaryCondition(U),
                                       west = OpenBoundaryCondition(u∞, parameters = U))

v_boundaries = FieldBoundaryConditions(east = GradientBoundaryCondition(0),
                                       west = GradientBoundaryCondition(0))

Δt = .3 * resolution / U

u_forcing = Relaxation(; rate = 1 / (2 * Δt), mask = cylinder)
v_forcing = Relaxation(; rate = 1 / (2 * Δt), mask = cylinder) 

model = NonhydrostaticModel(; grid, 
                              closure, 
                              forcing = (u = u_forcing, v = v_forcing),
                              auxiliary_fields = (u⁻¹ = Field{Face, Center, Center}(grid),
                                                  v⁻¹ = Field{Center, Face, Center}(grid),
                                                  η⁻¹ = Field{Center, Center, Nothing}(grid)),
                              boundary_conditions = (u = u_boundaries, v = v_boundaries))

@info "Constructed model"

# initial noise to induce turbulance faster
set!(model, u = U, v = (x, y) -> randn() * U * 0.01)

@info "Set initial conditions"

simulation = Simulation(model; Δt = Δt, stop_time = 100)

wizard = TimeStepWizard(cfl = 0.3)
simulation.callbacks[:wizard] = Callback(wizard, IterationInterval(100))

progress(sim) = @info "$(time(sim)) with Δt = $(prettytime(sim.Δt)) in $(prettytime(sim.run_wall_time))"
simulation.callbacks[:progress] = Callback(progress, IterationInterval(1000))

function update_aux_fields!(sim)
    sim.model.auxiliary_fields.u⁻¹ .= sim.model.velocities.u
    sim.model.auxiliary_fields.v⁻¹ .= sim.model.velocities.v
end
simulation.callbacks[:update] = Callback(update_aux_fields!, IterationInterval(1))

u, v, w = model.velocities
outputs = (; model.velocities..., ζ = (@at (Center, Center, Center) ∂x(v) - ∂y(u)))
simulation.output_writers[:velocity] = JLD2OutputWriter(model, outputs,
                                                        overwrite_existing = true, 
                                                        filename = "cylinder_$(extra_downstream)_Re_$Re.jld2", 
                                                        schedule = TimeInterval(0.5),
                                                        with_halos = true)

run!(simulation)

# load the results 

u_ts = FieldTimeSeries("cylinder_$(extra_downstream)_Re_$Re.jld2", "u")
v_ts = FieldTimeSeries("cylinder_$(extra_downstream)_Re_$Re.jld2", "v")
ζ_ts = FieldTimeSeries("cylinder_$(extra_downstream)_Re_$Re.jld2", "ζ")

@info "Loaded results"

# plot the results

fig = Figure(size = (600, 600))

ax = Axis(fig[1, 1], aspect = DataAspect())

xc, yc, zc = nodes(ζ_ts, with_halos=true)

n = Observable(1)

ζ_plt = @lift ζ_ts[:, :, 1, $n].parent

heatmap!(ax, collect(xc), collect(yc), ζ_plt, colorrange = (-2, 2), colormap = :roma)

record(fig, "ζ_Re_$Re.mp4", 1:length(u_ts.times), framerate = 12) do i;
    n[] = i
    i % 10 == 0 && @info "$(n.val) of $(length(u_ts.times))"
end

