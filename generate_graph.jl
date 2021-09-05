using LightGraphs, SparseArrays, DataFrames, CSV

function convert_grid_to_list(dim1, dim2)
    g = Grid([dim1, dim2])
    sources = Int64[]
    destinations = Int64[]

    for e in edges(g)
        push!(sources, src(e))
        push!(destinations, dst(e))

        # Make it directed by commenting out the below
        #push!(sources, dst(e))
        #push!(destinations, src(e))
    end

    return sources, destinations
end

function generate_LP_matrix(sources, destinations, start_node, end_node)

    nodes = unique(union(sources, destinations))
    n_nodes = length(nodes)
    n_edges = length(sources)

    # Need to hard code incidence matrix!
    # A_mat = zeros(n_nodes, n_edges)
    # for i = 1:n_edges
    #     A_mat[sources[i], i] = -1
    #     A_mat[destinations[i], i] = 1
    # end

    # Hard code sparse node-edge incidence matrix!
    I_vec = [sources; destinations]
    J_vec = [collect(1:n_edges); collect(1:n_edges)]
    V_vec = [-ones(n_edges); ones(n_edges)]
    A_mat = sparse(I_vec, J_vec, V_vec)

    # Set up RHS
    b_vec = sparse(zeros(n_nodes))
    b_vec[start_node] = -1
    b_vec[end_node] = 1

    # Remove sparse
    A_mat = round.(Int, Matrix(A_mat))
    b_vec = round.(Int, Vector(b_vec))

    return A_mat, b_vec
end

grid_dim = 5

sources, destinations = convert_grid_to_list(grid_dim, grid_dim)
d_feasibleregion = length(sources)
A_mat, b_vec = generate_LP_matrix(sources, destinations, 1, grid_dim^2)
CSV.write("A_and_b.csv",  DataFrame([A_mat b_vec]), writeheader=false)
