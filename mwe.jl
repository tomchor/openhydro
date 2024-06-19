using Oceananigans
grid = RectilinearGrid(topology = (Bounded, Flat, Bounded), size = (4, 4), extent = (1, 1))
u₀ = 1
u_bcs = FieldBoundaryConditions(east = OpenBoundaryCondition(u₀), west = OpenBoundaryCondition(u₀))
model = HydrostaticFreeSurfaceModel(; grid, boundary_conditions = (; u = u_bcs,))
set!(model, u = u₀)
time_step!(model, 0.1)
