module Birkhoff_Functions

export generate_ICs, standard_map, arnold_cat_map, evolve_system_to_N, save_evolutions, plot, calculate_WBN

using DynamicalSystems: DeterministicIteratedMap, trajectory
using Plots: savefig, scatter, plot, gr, plot!, display, savefig, default
using ZippedArrays: ZippedArray
using Random
using StaticArrays
using QuadGK
using StaticArrays: SVector
using BSplineKit

setprecision(BigFloat, 80)

### Standard Map ###
function standard_map(u, r, t)
    x, y = u
    x_n = mod2pi(x+y)
    y_n = mod2pi(y + r*sin(x + y))
    return SVector(BigFloat(x_n), BigFloat(y_n))
end

function henon_map(u, r, t)
    x, y = u
    x_n = x*cos(r)-(y-x^2)*sin(r)
    y_n = x*sin(r)+(y-x^2)*cos(r)
    return SVector(BigFloat(x_n), BigFloat(y_n))
end

### Area of Function ###
function initial_area_delta(delta_plotting, fixed_param_f)
    return quadgk(fixed_param_f, 0, delta_plotting-1, rtol=1e-20)
end

### Generate Initial Conditions ###
function generate_ICs(num_initial_conditions, precision_type, map_right_range, map_left_range)
    Random.seed!(0)
    # Function to generate a random tuple.
    random_tuple() = (big(rand() * (2*map_right_range) - map_left_range), big(rand() * (2*map_right_range) - map_left_range))
    # Create the matrix of random tuples.
    return [[random_tuple()] for _ in 1:num_initial_conditions] 
end

### Evolve Dynamical System up to N ###
function evolve_system_to_N(num_initial_conditions, map_type, r, evolutions, N)
    # Evolve the dynamical system.
    for i in 1:num_initial_conditions
        evolve_N = DeterministicIteratedMap(map_type, evolutions[i][length(evolutions[i])], r)
        map, _ = trajectory(evolve_N, N)
        tuple_map = ZippedArray(map[:,1], map[:,2])
        resize!(evolutions[i], N)
        evolutions[i] = tuple_map[1:N]
    end
    return evolutions
end;

### Saves Evolutions to Scatter Plot ###

function save_evolutions(scatter_dump, name)
    scatter_dump = [getindex.(scatter_dump, i) for i in 1:length(scatter_dump[1])]

    gr(aspect_ratio=1, legend=:none)
    fig = scatter(scatter_dump[1], scatter_dump[2]; markersize=0.3)
    savefig(name)
end

### Plotting the Quasiperiodic Orbits ###

function save_quasi_periodic_orbits(evolutions, quasi_orbits, N)
    quasi_pos = Vector{Tuple{BigFloat, BigFloat}}(undef, length(quasi_orbits)*N)
    N_zeros_quasi = Vector{Int}(undef, length(quasi_orbits)*N)

    # Add all evolution states on quasi orbits to vectors.
    for i in 1:length(quasi_orbits)
        for j in 1:N
            quasi_pos[(i-1)*N+j] = deepcopy(evolutions[quasi_orbits[i][1]][j])
            N_zeros_quasi[(i-1)*N+j] =  deepcopy(quasi_orbits[i][2])
        end
    end

    quasi_pos = [getindex.(quasi_pos, i) for i in 1:length(quasi_pos[1])]

    scatter(quasi_pos[1], quasi_pos[2]; zcolor = N_zeros_quasi, 
        markersize = 0.5, 
        markerstrokewidth = 0, 
        markershape = :circle, 
        color = :thermal, 
        legend = false, 
        colorbar = true,
        dpi=200 # Quality of image.
    )

    savefig("heatmap_quasi_orbits")
end


                                                                                    ### Standard Map Version ###

### Observable ###
function f(x)
    return sin(x[1]+x[2])
end;

### WBA ###
function WBA_weight(t, vertical_shift, width)
    if (t <= 0 || t >= 1)
        return vertical_shift
    end
    return (1-vertical_shift)*exp(-width/(t*(1-t)))+vertical_shift
end;

### Calculating Dig_T of Orbit ###
function calculate_WBA(evolutions, N, precision_type, vertical_shift, width)
    # Renormalise function weighting.
    normalise_weights = BigFloat(0.0)
    for j in 0:N-1
        normalise_weights += WBA_weight(j/N, vertical_shift, width)
    end

    orbit_convergence = Tuple{Int, precision_type}[]
    # Calculate convergence of trajectory.
    for i in 1:length(evolutions)
        WB_N = 0
        WB_2N = 0
        temp_WBN = Base.zeros(precision_type, Threads.nthreads())
        temp_WB2N = Base.zeros(precision_type, Threads.nthreads())
    
        # Calculating average - use threads as fastest instead of dot product.
        Threads.@threads for j in 1:N
            j -= 1
            # println("First: ", j, " where j/N = ", j/N, " where weighting is ", weighted_birkhoff_function(j/N, flat, vertical_shift, width))
            temp_WBN[Threads.threadid()] += WBA_weight(j/N, vertical_shift, width) * f(evolutions[i][j+1])      
        end
        Threads.@threads for j in N:2*N-1
            # println("Second: ", j, " where j-N/N = ", (j-N)/N, " where weighting is ", weighted_birkhoff_function((j-N)/N, flat, vertical_shift, width))
            temp_WB2N[Threads.threadid()] += WBA_weight((j-N)/N, vertical_shift, width) * f(evolutions[i][j+1])
        end

        WB_N = (1/normalise_weights) * sum(temp_WBN)
        WB_2N = (1/normalise_weights) * sum(temp_WB2N)
        # Testing for convergence
        absdigit = -log(abs(WB_N-WB_2N))
        reldigit = -log(abs(WB_N-WB_2N)/(0.5*abs(WB_N+WB_2N)))
        zeros = max(absdigit, reldigit, 0)
        push!(orbit_convergence, (i, zeros))
    end
    return orbit_convergence
end;

function evolve_single_orbit_N(singular_orbit, map, r, N)
    initial_condition = [[deepcopy(singular_orbit)]]
    return evolve_system_to_N(1, map, r, initial_condition, N)
end

### Computes the f() observed value for an orbit for N iterations
function compute_singular_observed_trajectory(evolutions, N, precision_type; vertical_shift, width)
    normalise_weights = BigFloat(0.0)
    for j in 0:N-1
        normalise_weights += WBA_weight(j/N, vertical_shift, width)
    end

    observable_vector = Vector{precision_type}(undef, N)
    for j in 1:N
        j -= 1
        observable_vector[j+1] = WBA_weight(j/N, vertical_shift, width) * f(evolutions[j+1]) * (1/normalise_weights)
    end
    return observable_vector
end

#https://en.wikipedia.org/wiki/Total_variation
function TV(f_x)
    total_variation = 0
    for i in 1:length(f_x)-1
        total_variation += abs(f_x[i+1] - f_x[i])
    end
    return total_variation
end

# https://math.stackexchange.com/questions/605594/finding-the-total-variation-of-3x2-2x3
# \int_a^b |f'(x)|dx
# function TV(x, f_x, order)
#     spl = interpolate(x, f_x, BSplineOrder(6))
#     D_i = Derivative(order) * spl
#     df = Vector{BigFloat}(undef, length(x))

#     for i in 1:length(x)
#         df[i] = abs(D_i(x[i]))
#     end

#     step_length = step(x)
#     # Birkhoff_Functions.plot!(main_plot, x, df)

#     return step_length/3*(abs(df[1])+2*sum(abs.(df[3:2:end-2]))+4*sum(abs.(df[2:2:end]))+abs(df[end]))
# end





### Evolve Dynamical System up to N ###
function evolve_system_to_N2(num_initial_conditions, map, r, evolutions, curr_N, N)
    curr_N += N
    old_length = length(evolutions[1])

    # Evolve the dynamical system.
    for i in 1:num_initial_conditions
        evolve_N = DeterministicIteratedMap(map, evolutions[i][length(evolutions[i])], r)
        map, _ = trajectory(evolve_N, N)
        tuple_map = ZippedArray(map[:,1], map[:,2])
        
        resize!(evolutions[i], N+old_length)

        evolutions[i][old_length+1:N+old_length] = tuple_map[2:N+1]
    end
    return evolutions, curr_N
end;

end