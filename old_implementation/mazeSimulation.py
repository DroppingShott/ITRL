import sys

import matplotlib as mpl
import numpy as np

# Import the necessary classes
from old_implementation.actionValue import ActionValue as av
from old_implementation.epsilonGreedy import epsilon_greedy_action as ega

sub_goal = False
end_goal = False


def getReward(state):
    global sub_goal
    global end_goal

    if state == 8:
        if not sub_goal:
            sub_goal = True
            return 20
        else:
            return -10  # Disincentivize revisiting the sub-goal
    elif state == 9:
        if not end_goal:
            end_goal = True
            return 100
    elif state == 0:
        return -2  # Walking path penalty
    elif state == 2:  # Disincentivize revisiting the start
        return -10
    else:
        raise ValueError("INVALID STATE || This state should never be reached")
        sys.exit(1)


def move_agent(state, action, maze):
    x, y = state
    if action == "up":
        x -= 1
    elif action == "down":
        x += 1
    elif action == "left":
        y -= 1
    elif action == "right":
        y += 1

    # Ensure valid movement within maze boundaries
    if 0 <= x < len(maze) and 0 <= y < len(maze[0]) and maze[x][y] != 1:
        return (x, y)
    return state


# Single episode of the maze simulation
def runMazeSim(
    maze_layout, AVclass, max_epsilon=1.0, min_epsilon=0.01, decay_rate=0.001, episode=1
):
    global sub_goal
    global end_goal

    state = (1, 1)  # Start position
    path = [state]

    # Compute epsilon for this episode (gradual decay)
    epsilon = min_epsilon + (max_epsilon - min_epsilon) * np.exp(-decay_rate * episode)

    while maze_layout[state] != 9:
        # Get the next action using epsilon-greedy strategy with decaying epsilon
        action = ega(state, AVclass.action_values, epsilon)

        next_state = move_agent(state, action, maze_layout)

        # Get the reward for the next state
        reward = getReward(maze_layout[next_state])
        path.append(next_state)

        # Update the Q-value for the state-action pair
        AVclass.update_q_value(state, action, reward, next_state, alpha=0.8, gamma=0.95)

        state = next_state

    sub_goal = False
    end_goal = False

    return path
