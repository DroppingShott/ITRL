import numpy as np
import matplotlib as mpl
import sys

# Import the necessary classes
from actionValue import ActionValue as av 
from mazeSimulation import runMazeSim


# maze_layout = np.array(
#     [
#         [1, 1, 1, 1, 1],
#         [1, 2, 0, 0, 1],
#         [1, 0, 1, 8, 1],
#         [1, 0, 0, 9, 1],
#         [1, 1, 1, 1, 1],
#     ]
# )

# Create simple maze layout where 1 represents a wall and 0 represents a path, 
# 2 represents the start, 8 represents the sub-goal, and 9 represents the end goal
maze_layout = np.array(
    [
        [1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
        [1, 2, 1, 0, 0, 0, 1, 0, 0, 1],
        [1, 0, 1, 0, 1, 0, 1, 0, 1, 1],
        [1, 0, 0, 0, 1, 0, 0, 0, 0, 1],
        [1, 1, 1, 0, 1, 1, 1, 1, 0, 1],
        [1, 0, 1, 8, 0, 0, 0, 1, 0, 1],
        [1, 0, 0, 0, 1, 1, 0, 0, 0, 1],
        [1, 1, 1, 0, 1, 0, 1, 1, 0, 1],
        [1, 0, 0, 0, 0, 0, 1, 9, 0, 1],
        [1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
    ]
)


print(maze_layout)
# Run the simulation 100 times
AVclass = av(maze_layout)

for i in range(10000):
    runMazeSim(maze_layout, AVclass)

AVclass.pretty_print()