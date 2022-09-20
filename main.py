import ant_colony
import numpy as np
import big_graph as bg
import sys
import time
import multiprocessing

def initpool(arr):
    global array
    array = arr

def change_array(graph ,changes):
    phero_common_arr_np = np.frombuffer(array.get_obj(), dtype=np.float64).reshape(graph.shape[0], graph.shape[1])
    for i in changes:
        phero_common_arr_np[i[0], i[1]] += changes[i] * graph.Qui * graph.weight_matrix[i[0]][i[1]]
        phero_common_arr_np[i[1], i[0]] += changes[i] * graph.Qui * graph.weight_matrix[i[1]][i[0]]
    graph.iteration += 1
    graph.phero_matrix = phero_common_arr_np
    #print(np.frombuffer(array.get_obj()))

def ant_solver(graph, iteration):
    """
    Main solver function for TSP (multiprocessor)
    :param graph: the graph object that need to be solved
    :param iteration: number of time the ants has to travel the graphs
    :return: a sorted list of edges by pheromone that make up the best tour
    """

    # Parallel computation with a pool of ant
    phero_common_arr = multiprocessing.Array('d', graph.phero_matrix.shape[0] * graph.phero_matrix.shape[1], lock=True)
    # Wrap X as a numpy array so that we can easily manipulate its data.
    phero_common_arr_np = np.frombuffer(phero_common_arr.get_obj()).reshape(graph.phero_matrix.shape)
    # Copy data to our shared array.
    np.copyto(phero_common_arr_np, graph.phero_matrix)
    # Processing pool
    initpool(phero_common_arr)
    pool = multiprocessing.Pool(processes=30)
    while graph.iteration < iteration:
        # This dictionary saves the pheromone deposited by all ants travelling the graph in one iteration
        phero_changes = dict()
        # Compile a parameter list to be fed to multiple processors
        temp = list()
        for i in range(graph.shape[0]):
            temp.append(graph)
        all_ants = zip(temp, range(graph.shape[0]))
        # Now let the ants start moving on every vertex of the graph
        all_ant_phero_changes = pool.starmap(ant_colony.travelling_ant_warpper, all_ants)
        for ant in all_ant_phero_changes:
            for edge in ant:
                if edge in phero_changes:
                    phero_changes[edge] += ant[edge]
                else:
                    phero_changes[edge] = ant[edge]
        graph.phero_matrix = np.where(graph.phero_matrix != 0, graph.phero_matrix * (1 - graph.phi), 0)
        change_array(graph, phero_changes)
    return graph.ant_harvester()

def test_cases_big_graph(size, density):
    print(str(size) + " vertices graph: ")
    matrix = bg.big_graph(size, density)
    start_time = time.time()
    test_graph = ant_colony.Graph(matrix, 1.2, 1.4, 0.3, 3, 5)
    edges = ant_solver(test_graph, 5)
    print("--- %s seconds ---" % (time.time() - start_time))

if __name__ == '__main__':
    print("Just a usual test case with 4 vertices: ")
    matrix = np.array([[0., 7., 8., 9.], [7., 0., 6., 8.], [8., 6., 0., 2.], [9., 8., 2., 0.]])
    test_graph = ant_colony.Graph(matrix, 2, 3, 0.1, 3, 5)
    edges = ant_solver(test_graph, 10)
    print(edges)

    print("Weight sensitive test: ")
    matrix = np.array([[0., 9.997, 9.998, 9.999], [9.997, 0., 9.996, 9.998], [9.998, 9.996, 0., 9.992], [9.999, 9.998, 9.992, 0.]])
    test_graph = ant_colony.Graph(matrix, 2, 3, 0.1, 3, 5)
    edges = ant_solver(test_graph, 10)
    print(edges)

    print("5 vertices and has closed eulerian tour: ")
    matrix = np.array([[0., 7., 0., 15., 23.], [7., 0, 7., 5., 9.], [0., 7., 0., 2., 0.], [15., 5., 2., 0., 3.],
                       [23., 9., 0., 3., 0.]])
    test_graph = ant_colony.Graph(matrix, 2, 3, 0.1, 3, 5)
    edges = ant_solver(test_graph, 10)
    print(edges)

    print("5 vertices and no eulerian tour: ")
    matrix = np.array([[0., 3., 0., 0., 0.], [3., 0, 4., 0., 6.], [0., 4., 0., 4., 0.], [0., 0., 4., 0., 5.],
                       [0., 6., 0., 5., 0.]])
    test_graph = ant_colony.Graph(matrix, 2, 3, 0.1, 3, 5)
    edges = ant_solver(test_graph, 10)
    print(edges)

    print("5 vertices disconnected graph: ")
    matrix = np.array(
        [[0., 2., 0., 9., 0.], [2., 0, 0., 5., 0.], [0., 0., 0., 0., 3.], [9., 5., 0., 0., 0.], [0., 0., 3., 0., 0.]])
    test_graph = ant_colony.Graph(matrix, 2, 3, 0.1, 3, 5)
    edges = ant_solver(test_graph, 10)
    print(edges)

    # Big Graph testing
    # Set recursion limit to allow big graph test
    sys.setrecursionlimit(999999)

    test_cases_big_graph(400, 0.9)
