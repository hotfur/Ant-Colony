# Ant-Colony
## Short Introduction
A Python implementation of ant colony optimization algorithm to solve the Travelling Salesman Problem on n vertices graph.
## Poster and detailed report
Please see poster.pdf and report.docx respectively.
## Code instruction
### Code Structure
There is three module in this repo:
- main.py: contains code for single-core or multi-core ACO and test cases.
- ant_colony.py: codes for all mathematics works in ACO, which include a graph class and methods for update and print the pheromone level.
- big_graph: generate a random nxn adjacency matrix of a graph with adjustable edge density for benchmarking
### Usage Instruction
This program do not come with an interface. To use it, please add your input graph as an adjacency matrix (using numpy array) directly into the
test case section of the main code. The output will be in form of a descending-ordered list by pheromone level containing all edges that is on
the Eulerian path.
## Authors
Authors:
- Vuong Kha Sieu
- Nguyen Thanh Long
