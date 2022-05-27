import ant_colony
import numpy as np
import big_graph as bg
import sys
import time
from multiprocessing import Pool


def parallel_ant_solver(graph, iteration):
    """
    Main solver function for TSP (multiprocessor)
    :param graph: the graph object that need to be solved
    :param iteration: number of time the ants has to travel the graphs
    :return: a sorted list of edges by pheromone that make up the best tour
    """
    # Parallel computation with a pool of ant
    pool = Pool()
    while graph.iteration < iteration:
        # This dictionary saves the pheromone deposited by all ants travelling the graph in one iteration
        phero_changes = dict()
        # Compile a parameter list to be fed to multiple processors
        temp = list()
        for i in range(graph.verticles_no):
            temp.append(graph)
        all_ants = zip(temp, range(graph.verticles_no))
        # Now let's the ants start moving on every vertices of the graph
        all_ant_phero_changes = pool.starmap(ant_colony.travelling_ant_warpper, all_ants)
        for ant in all_ant_phero_changes:
            for edge in ant:
                if edge in phero_changes:
                    phero_changes[edge] += ant[edge]
                else:
                    phero_changes[edge] = ant[edge]
        graph.update_pheromone(phero_changes)
    return graph.ant_harvester()


def serial_ant_solver(graph, iteration):
    """
    Main solver function for TSP (1 processor)
    :param graph: the graph object that need to be solved
    :param iteration: number of time the ants has to travel the graphs
    :return: a sorted list of edges by pheromone that make up the best tour
    """
    while graph.iteration < iteration:
        # This dictionary saves the pheromone deposited by all ants travelling the graph in one iteration
        phero_changes = dict()
        # Now let's the ants start moving on every vertices of the graph
        for i in range(graph.verticles_no):
            ant = ant_colony.travelling_ant_warpper(graph, i)
            for edge in ant:
                if edge in phero_changes:
                    phero_changes[edge] += ant[edge]
                else:
                    phero_changes[edge] = ant[edge]
        graph.update_pheromone(phero_changes)
    return graph.ant_harvester()


def ant_solver(graph, iteration, computation='parallel'):
    """
    A wrapper function for ant solver for user to choose between parallel and serial computation
    :param graph: the graph object that need to be solved
    :param iteration: number of time the ants has to travel the graphs
    :param computation: "serial" or "parallel" accepted, default parallel
    :return:
    """
    if computation == 'serial':
        return serial_ant_solver(graph, iteration)
    return parallel_ant_solver(graph, iteration)


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

    print("5 vertices and has closed eulerian tour: ")
    matrix = np.array([[0., 7., 0., 15., 23.], [7., 0, 7., 5., 9.], [0., 7., 0., 2., 0.], [15., 5., 2., 0., 3.],
                       [23., 9., 0., 3., 0.]])
    graph01 = ant_colony.Graph(matrix, 2, 3, 0.1, 3, 5)
    edges = ant_solver(graph01, 10)
    print(edges)

    print("5 vertices and no eulerian tour: ")
    matrix = np.array([[0., 3., 0., 0., 0.], [3., 0, 4., 0., 6.], [0., 4., 0., 4., 0.], [0., 0., 4., 0., 5.],
                       [0., 6., 0., 5., 0.]])
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

    print("Dense 400 vertices graph: ")
    matrix = bg.big_graph(400, 0.9)
    start_time = time.time()
    graph01 = ant_colony.Graph(matrix, 1.2, 1.4, 0.3, 3, 5)
    edges = ant_solver(graph01, 5)
    print("--- %s seconds ---" % (time.time() - start_time))

    print("Sparse 400 vertices graph: ")
    matrix = bg.big_graph(400, 0.3)
    start_time = time.time()
    graph01 = ant_colony.Graph(matrix, 1.2, 1.4, 0.3, 3, 5)
    edges = ant_solver(graph01, 5)
    print("--- %s seconds ---" % (time.time() - start_time))
