import ant_colony
import numpy as np
import big_graph as bg
import sys, time


def ant_solver(graph, iteration):
    """
    Main solver function for TSP
    :param graph: the graph object that need to be solved
    :param iteration: number of time the ants has to travel the graphs
    :return: a sorted list of edges by pheromone that make up the best tour
    """
    while graph.iteration < iteration:
        # This dictionary saves the pheromone deposited by all ants travelling the graph in one iteration
        phero_changes = dict()
        # Now let's the ants start moving on every vertices of the graph
        for i in range(graph.verticles_no):
            temp_changes = graph.traveling_ant(i, dict(), list())
            for k in temp_changes:
                if k in phero_changes:
                    phero_changes[k] += temp_changes[k]
                else:
                    phero_changes[k] = temp_changes[k]
        graph.update_pheromone(phero_changes)
    return graph.ant_harvester()


if __name__ == '__main__':
    print("Just a usual test case with 4 vertices: ")
    matrix = np.array([[0., 7., 8., 9.], [7., 0., 6., 8.], [8., 6., 0., 2.], [9., 8., 2., 0.]])
    graph01 = ant_colony.Graph(matrix, 2, 3, 0.1, 3, 5)
    edges = ant_solver(graph01, 10)
    print(edges)

    print("Weight sensitive test: ")
    matrix = np.array([[0., 9.997, 9.998, 9.999], [9.997, 0., 9.996, 9.998], [9.998, 9.996, 0., 9.992], [9.999, 9.998, 9.992, 0.]])
    graph01 = ant_colony.Graph(matrix, 2, 3, 0.1, 3, 5)
    edges = ant_solver(graph01, 10)
    print(edges)

    print("5 vertices and some edges not available: ")
    matrix = np.array([[0., 7., 0., 15., 23.], [7., 0, 7., 5., 9.], [0., 7., 0., 2., 0.], [15., 5., 2., 0., 3.],
                       [23., 9., 0., 3., 0.]])
    graph01 = ant_colony.Graph(matrix, 2, 3, 0.1, 3, 5)
    edges = ant_solver(graph01, 10)
    print(edges)

    print("5 vertices disconnected graph: ")
    matrix = np.array(
        [[0., 2., 0., 9., 0.], [2., 0, 0., 5., 0.], [0., 0., 0., 0., 3.], [9., 5., 0., 0., 0.], [0., 0., 3., 0., 0.]])
    graph01 = ant_colony.Graph(matrix, 2, 3, 0.1, 3, 5)
    edges = ant_solver(graph01, 10)
    print(edges)

    # Big Graph testing
    # Set recursion limit to allow big graph test
    sys.setrecursionlimit(999999)

    print("Dense 50 vertices graph: ")
    matrix = bg.big_graph(50, 0.9)
    start_time = time.time()
    graph01 = ant_colony.Graph(matrix, 1.2, 1.4, 0.3, 3, 5)
    edges = ant_solver(graph01, 5)
    print("--- %s seconds ---" % (time.time() - start_time))

    print("Sparse 50 vertices graph: ")
    matrix = bg.big_graph(50, 0.3)
    start_time = time.time()
    graph01 = ant_colony.Graph(matrix, 1.2, 1.4, 0.3, 3, 5)
    edges = ant_solver(graph01, 5)
    print("--- %s seconds ---" % (time.time() - start_time))

    print("Dense 100 vertices graph: ")
    matrix = bg.big_graph(100, 0.9)
    start_time = time.time()
    graph01 = ant_colony.Graph(matrix, 1.2, 1.4, 0.3, 3, 5)
    edges = ant_solver(graph01, 5)
    print("--- %s seconds ---" % (time.time() - start_time))

    print("Sparse 100 vertices graph: ")
    matrix = bg.big_graph(100, 0.3)
    start_time = time.time()
    graph01 = ant_colony.Graph(matrix, 1.2, 1.4, 0.3, 3, 5)
    edges = ant_solver(graph01, 5)
    print("--- %s seconds ---" % (time.time() - start_time))

    print("Dense 200 vertices graph: ")
    matrix = bg.big_graph(200, 0.9)
    start_time = time.time()
    graph01 = ant_colony.Graph(matrix, 1.2, 1.4, 0.3, 3, 5)
    edges = ant_solver(graph01, 5)
    print("--- %s seconds ---" % (time.time() - start_time))

    print("Sparse 200 vertices graph: ")
    matrix = bg.big_graph(200, 0.3)
    start_time = time.time()
    graph01 = ant_colony.Graph(matrix, 1.2, 1.4, 0.3, 3, 5)
    edges = ant_solver(graph01, 5)
    print("--- %s seconds ---" % (time.time() - start_time))