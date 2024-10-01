import numpy as np
import matplotlib as mpl

# Create simple maze layout where 1 represents a wall and 0 represents a path, 
# 2 represents the start, 8 represents the sub-goal, and 9 represents the end goal

maze_layout = np.array(
    [
        [1, 1, 1, 1, 1],
        [1, 2, 0, 8, 1],
        [1, 0, 1, 0, 1],
        [1, 0, 0, 9, 1],
        [1, 1, 1, 1, 1],
    ]
)



def cumulative_reward(rewards, gamma=0.9):
    G_t = 0
    for i, reward in enumerate(rewards):
        G_t += (gamma ** i) * reward
    return G_t


def getReward(state):
    # Define rewards for each state
    if state == 8:
        return 50
    elif state == 9:
        return 100
    elif state == 0:
        return -1
    else:
        # print("Invalid state")
        return -5
    
def move_agent(state, action, maze):
    # Define movement rules (up, down, left, right)
    x, y = state
    if action == 'up': x -= 1
    elif action == 'down': x += 1
    elif action == 'left': y -= 1
    elif action == 'right': y += 1
    
    # Ensure valid movement within maze boundaries
    if 0 <= x < len(maze_layout) and 0 <= y < len(maze_layout[0]) and maze[x][y] != 1:
        return (x, y)  # New state, no wall hit
    return state

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



def runMazeSim(maze_layout):
    state = (1, 1)  # Start position
    path = [state]
    rewards = [getReward(maze_layout[state])]
    # print(rewards)
    while maze_layout[state] != 9:
        # get Possible actions
        actions = getPossibleActions(state, maze_layout)
        
        # Select random action
        action = np.random.choice(actions)

        # Move agent
        state = move_agent(state, action, maze_layout)
        path.append(state)
        rewards.append(getReward(maze_layout[state]))
        # print(f"Current state: {state}, Action: {action}")

    print(f"cummalitive reward: {cumulative_reward(rewards)}")
    print("END FOUND")

print(maze_layout)
runMazeSim(maze_layout)