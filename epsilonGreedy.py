import numpy as np

# This is an implementation of the basic epsilon-greedy action selection algorithm. 
def epsilon_greedy_action(state, Qtable, epsilon):
    possible_actions = Qtable[state].keys()  
    random_value = np.random.rand() 

    if random_value < epsilon:
        action = np.random.choice(list(possible_actions)) 
    else:
        action = max(possible_actions, key=lambda a: Qtable[state][a]) 

    return action
