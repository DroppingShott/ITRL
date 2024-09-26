import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib

matplotlib.use("Agg")

maze_layout = np.array(
    [
        [1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
        [1, 0, 1, 0, 0, 0, 1, 0, 0, 1],
        [1, 0, 1, 0, 1, 0, 1, 0, 1, 1],
        [1, 0, 0, 0, 1, 0, 0, 0, 0, 1],
        [1, 1, 1, 0, 1, 1, 1, 1, 0, 1],
        [1, 0, 1, 0, 0, 0, 0, 1, 0, 1],
        [1, 0, 0, 0, 1, 1, 0, 0, 0, 1],
        [1, 1, 1, 0, 1, 0, 1, 1, 0, 1],
        [1, 0, 0, 0, 0, 0, 1, 0, 0, 1],
        [1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
    ]
)

start_pos = (1, 1)
sub_goal_pos = (5, 3)
end_goal_pos = (8, 6)


def plot_maze_with_path_sns(maze, path, start, sub_goal, end):
    maze_with_path = maze.copy().astype(float)
    for i, j in path:
        maze_with_path[i, j] = 0.5
    maze_with_path[start] = -0.5
    maze_with_path[sub_goal] = 0.7
    maze_with_path[end] = 1.2
    sns.heatmap(
        maze_with_path,
        cmap="coolwarm",
        cbar=False,
        linewidths=0.5,
        linecolor="black",
        annot=True,
        fmt="",
    )
    plt.savefig("maze_visualization_seaborn.png")


optimal_path = [(1, 1), (1, 3), (3, 3), (5, 3), (6, 3), (8, 6)]

plot_maze_with_path_sns(
    maze_layout, optimal_path, start_pos, sub_goal_pos, end_goal_pos
)
