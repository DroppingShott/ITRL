import numpy as np

def getPossibleActions(state, maze_layout):
    # Define possible actions for each state
    actions = []
    x, y = state

    # Check if agent can move up (not out of bounds and the state above is not a wall)
    if x - 1 >= 0 and maze_layout[x - 1][y] != 1:
        actions.append("up")
    
    # Check if agent can move down (not out of bounds and the state below is not a wall)
    if x + 1 < len(maze_layout) and maze_layout[x + 1][y] != 1:
        actions.append("down")
    
    # Check if agent can move left (not out of bounds and the state to the left is not a wall)
    if y - 1 >= 0 and maze_layout[x][y - 1] != 1:
        actions.append("left")
    
    # Check if agent can move right (not out of bounds and the state to the right is not a wall)
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
                if maze[x][y] != 1:  # Check for non-wall states
                    actions = getPossibleActions((x, y), maze)
                    # Add to dictionary where key is state and value another dictionary {action: value}
                    Qtable[(x, y)] = {action: 0.0 for action in actions}  # Initialize action values to 0
        return Qtable

    def get_action_value(self, state, action):
        # Return the action value for a specific state and action
        return self.action_values.get(state, {}).get(action, None)
    
    def pretty_print(self):
        print("\nAction Values:")
        for state, actions in self.action_values.items():
            action_str = ', '.join(f"{action}: {value:.2f}" for action, value in actions.items())
            print(f"State {state}: {action_str}")
