import ant_colony
import numpy as np

# just a usual test case
matrix = np.array([[0.,7.,8.,9.],[7.,0.,6.,8.],[8.,6.,0.,2.],[9.,8.,2.,0.]])
graph01 = ant_colony.Graph(matrix, 2, 3, 0.1, 3, 5)
edges = ant_colony.ant_solver(graph01, 40)
print(edges)

#weight sensitive test
matrix = np.array([[0.,9.7,9.8,9.9],[9.7,0.,9.6,9.8],[9.8,9.6,0.,9.2],[9.9,9.8,9.2,0.]])
graph01 = ant_colony.Graph(matrix, 2, 10, 0.1, 3, 5)
edges = ant_colony.ant_solver(graph01, 40)
print(edges)

#5 vertex and some edge not available
matrix = np.array([[0., 7., 0., 15., 23.], [7., 0, 7., 5., 9.], [0., 7., 0., 2., 0.], [15., 5., 2., 0., 3.], [23., 9., 0., 3., 0.]])
graph01 = ant_colony.Graph(matrix, 2, 1.1, 0.1, 3, 5)
edges = ant_colony.ant_solver(graph01, 1000)
print(edges)

#5 verticles disconnected graph
matrix = np.array([[0., 2., 0., 9., 0.], [2., 0, 0., 5., 0.], [0., 0., 0., 0., 3.], [9., 5., 0., 0., 0.], [0., 0., 3., 0., 0.]])
graph01 = ant_colony.Graph(matrix, 2, 1.1, 0.1, 3, 5)
edges = ant_colony.ant_solver(graph01, 1000)
print(edges)