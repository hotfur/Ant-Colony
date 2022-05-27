import numpy as np

"""
Big matrices helper file
@author Vuong Kha Sieu
@date: 9/5/2022
"""


def big_graph(vertices, density):
    """
    Generate a very big graph for use with TSP Ant Colony algorithm
    :param vertices: the number of vertices of the graph
    :param density: the density of the graph from 0 to 1. 1 is a graph with n^2 edges and 0 is a graph with no edge
    :return: a numpy matrix
    """
    # Because random modify both the upper and lower halves of the matrix, thus resulting in a bias toward sparser
    # matrix, we has to adjust the density constant to offset this effect.
    density = np.sqrt(density)
    matrix = np.where(np.identity(vertices) != 0, 0, np.random.rand(vertices, vertices))
    matrix *= np.random.choice([0, 1], (vertices, vertices), p=[1-density, density])
    matrix *= np.transpose(matrix)
    return matrix

"""TEST CASE"""

#print(big_graph(4,0.1))