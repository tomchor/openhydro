using Oceananigans
using Oceananigans.BoundaryConditions: FlatExtrapolationOpenBoundaryCondition

N = 1024
L = 1000
grid = RectilinearGrid(topology = (Bounded, Flat, Bounded), size = (N, N÷64), x = (-L/2, +L/2), z = (-10, 0), halo = (3,3))

u_boundaries = FieldBoundaryConditions(east = FlatExtrapolationOpenBoundaryCondition(0),
                                       west = OpenBoundaryCondition(0))

η_bc = FieldBoundaryConditions(east = FlatExtrapolationOpenBoundaryCondition(0),
                                       west = OpenBoundaryCondition(0))

model = HydrostaticFreeSurfaceModel(; grid,
                                    momentum_advection = WENO(order=5),
                                    tracers = (),
                                    boundary_conditions = (u = u_boundaries, η = η_bc),
                                    buoyancy = nothing,)
η₀(x, z) = exp(-(x/10)^2)
set!(model, η=η₀)

g = model.free_surface.gravitational_acceleration
c = @show √(g*grid.Lz)
simulation = Simulation(model; Δt = c/2, stop_time = 200)

wizard = TimeStepWizard(cfl = 0.6)
simulation.callbacks[:wizard] = Callback(wizard, IterationInterval(100))

progress(sim) = @info "$(time(sim)) with Δt = $(prettytime(sim.Δt)) in $(prettytime(sim.run_wall_time))"
simulation.callbacks[:progress] = Callback(progress, IterationInterval(100))

filename = "height_impulse.jld2"
simulation.output_writers[:velocity] = JLD2OutputWriter(model, fields(model),
                                                        overwrite_existing = true, 
                                                        filename = filename, 
                                                        schedule = TimeInterval(2),
                                                        with_halos = true)

run!(simulation)

@info "Start plotting"
u_ts = FieldTimeSeries(filename, "u")
η_ts = FieldTimeSeries(filename, "η")

xf, yc, zc = nodes(u_ts, with_halos=true)
xc, yc, _ = nodes(η_ts, with_halos=true)

using CairoMakie
n = Observable(1)

u_plt = @lift u_ts[:, 1, :, $n]
u_plt = @lift u_ts[$n].data[:, 1, :].parent
η_plt = @lift η_ts[:, 1, grid.Nz+1, $n]

fig = Figure(size = (600, 600))
ax2 = Axis(fig[2, 1], xlabel="η [m/s]")
ax3 = Axis(fig[3, 1], xlabel="u [m/s]")

lines!(ax2, xc, η_plt)
heatmap!(ax3, collect(xf), collect(zc), u_plt, colormap = :balance, colorrange = (-1e-1, +1e-1))

record(fig, "height_impulse.mp4", 1:length(u_ts.times), framerate = 14) do i;
    n[] = i
    i % 10 == 0 && @info "$(n.val) of $(length(u_ts.times))"
end
