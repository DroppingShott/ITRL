import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch
from actionValue import ActionValue as av  # Your action value class
from epsilonGreedy import epsilon_greedy_action as ega  # Your epsilon greedy action selection
from mazeSimulation import runMazeSim

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
# Assuming AVclass is already trained and contains the Q-values (after running several episodes)
AVclass = av(maze_layout)  # Initialize or load the trained ActionValue class
for i in range(10000):
    runMazeSim(maze_layout, AVclass)
start_pos = (1, 1)
sub_goal_pos = (5, 3)
end_goal_pos = (8, 6)

def plot_maze_with_q_values(maze_layout, optimal_Qtable, start_pos, sub_goal_pos, end_goal_pos):
    maze_with_path = maze.copy().astype(float)
    maze_with_path[start_pos] = -0.5
    maze_with_path[sub_goal_pos] = 0.7
    maze_with_path[end_goal_pos] = 1.2

    fig, ax = plt.subplots(figsize=(8, 8))
    sns.heatmap(
        maze_with_path,
        cmap="coolwarm",
        cbar=False,
        linewidths=0.5,
        linecolor="black",
        annot=False,
        ax=ax
    )

    # Annotate Q-values for each state
    for state, actions in Qtable.items():
        x, y = state
        if maze[x, y] != 1:  # Non-wall states
            q_value_str = "\n".join([f"{a[0].upper()}: {v:.1f}" for a, v in actions.items()])
            ax.text(y + 0.5, x + 0.5, q_value_str, ha='center', va='center', fontsize=8, color='black')

    plt.savefig("maze_with_q_values.png")
    plt.show()


# Assuming AVclass has already run 10000  episodes and learned the optimal policy
optimal_Qtable = AVclass.action_values  # This contains the Q-values for each state

plot_maze_with_q_values(maze_layout, optimal_Qtable, start_pos, sub_goal_pos, end_goal_pos)

