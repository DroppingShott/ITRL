import numpy as np

# Function to get possible actions for a given state
def getPossibleActions(state, maze_layout):
    actions = []
    x, y = state

    if x - 1 >= 0 and maze_layout[x - 1][y] != 1:
        actions.append("up")
    if x + 1 < len(maze_layout) and maze_layout[x + 1][y] != 1:
        actions.append("down")
    if y - 1 >= 0 and maze_layout[x][y - 1] != 1:
        actions.append("left")
    if y + 1 < len(maze_layout[0]) and maze_layout[x][y + 1] != 1:
        actions.append("right")
    
    return actions

class ActionValue:
    def __init__(self, input_maze):
        self.maze = input_maze
        self.action_values = self.initialize_action_values(input_maze)

    # Action values initialization function
    def initialize_action_values(self, maze):
        actions = []

        Qtable = {}

        for x in range(maze.shape[0]):
            for y in range(maze.shape[1]):
                if maze[x][y] != 1:  # Non-wall states
                    actions = getPossibleActions((x, y), maze)
                
                    if maze[x][y] == 9:  # Termination state
                        Qtable[(x, y)] = {action: 300.0 for action in actions}  # High value for terminal state
                    else:
                        Qtable[(x, y)] = {action: 0.0 for action in actions}  # Initialize other states to 0

        return Qtable
    
    def pretty_print(self):
        print("\nAction Values:")
        for state, actions in self.action_values.items():
            action_str = ', '.join(f"{action}: {value:.2f}" for action, value in actions.items())
            print(f"State {state}: {action_str}")

    def get_action_value(self, state, action):
        return self.action_values.get(state, {}).get(action, None)
    
    def update_q_value(self, state, action, reward, next_state, alpha, gamma):
        # Get the current Q-value for the state-action pair
        current_q_value = self.get_action_value(state, action) #

        # Get the maximum Q-value for the next state
        max_future_q_value = max(self.action_values.get(next_state, {}).values(), default=0.0)

        # Update the Q-value using the Q-learning formula
        new_q_value = current_q_value + alpha * (reward + gamma * max_future_q_value - current_q_value)

        # Set the new Q-value in the action values table
        self.action_values[state][action] = new_q_value
