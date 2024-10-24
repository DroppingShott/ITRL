# pylint: disable=import-error
import random

import gym
import numpy as np


class QLearningAgent:
    """Q-learning agent for solving the maze environment."""

    def __init__(
        self,
        env,
        alpha=0.8,
        gamma=0.95,
        epsilon=1.0,
        epsilon_decay=0.99,
        min_epsilon=0.05,
    ):  # pylint: disable=too-many-arguments
        """
        Initialize the Q-learning agent with the environment and hyperparameters.
        """
        self.env = env
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay
        self.min_epsilon = min_epsilon
        self.q_table = np.zeros((env.size, env.size, env.action_space.n))

        # Initialize lists for tracking training performance
        self.steps_per_episode = []
        self.rewards_per_episode = []

    def choose_action(self, state):
        """
        Epsilon-greedy action selection: choose a random action with probability epsilon
        or the best-known action (exploit) with probability 1 - epsilon.
        """
        if random.uniform(0, 1) < self.epsilon:
            return self.env.action_space.sample()  # Explore
        return np.argmax(self.q_table[state[0], state[1]])  # Exploit

    def update_q_value(self, state, action, reward, next_state) -> None:
        """
        Update the Q-value using the Q-learning formula.
        """
        best_next_action = np.argmax(self.q_table[next_state[0], next_state[1]])
        td_target = (
            reward
            + self.gamma * self.q_table[next_state[0], next_state[1], best_next_action]
        )
        td_error = td_target - self.q_table[state[0], state[1], action]
        self.q_table[state[0], state[1], action] += self.alpha * td_error

    def train(self, num_episodes=1500, log_interval=10) -> np.array:
        """
        Train the Q-learning agent by running episodes in the environment.
        """
        for episode in range(num_episodes):
            state, _ = self.env.reset()
            done = False
            steps = 0
            total_reward = 0

            while not done:
                action = self.choose_action(state)
                next_state, reward, done, _, _ = self.env.step(action)

                self.update_q_value(state, action, reward, next_state)

                state = next_state

                total_reward += reward
                steps += 1
            # Track episode metrics
            self.steps_per_episode.append(steps)
            self.rewards_per_episode.append(total_reward)

            # Decay epsilon
            self.epsilon = max(self.min_epsilon, self.epsilon * self.epsilon_decay)

            if episode % log_interval == 0:
                print(f"Episode {episode}: epsilon={self.epsilon}")

        print("Training complete!")
        return self.q_table

    def play_game(self, q_table=None, max_steps=10000, num_episodes=1, render=True):
        """
        Play the game after training, using the learned Q-table.
        """
        if q_table is None:
            q_table = self.q_table

        for episode in range(num_episodes):
            state, _ = self.env.reset()
            done = False
            path = [state]

            print(f"EPISODE {episode + 1}")
            print("****************************************************")

            for step in range(max_steps):
                if render:
                    self.env.render()

                action = np.argmax(q_table[state[0], state[1]])
                print(f"Step {step + 1}: Agent at {state}, taking action {action}")

                next_state, reward, done, _, _ = self.env.step(action)

                if np.array_equal(next_state, self.env.sub_goal):
                    print(f"Agent reached the sub goal in {step + 1} steps!")

                path.append(next_state)

                if done:
                    if render:
                        self.env.render()
                    print(f"Agent reached the goal in {step + 1} steps!")
                    break

                state = next_state

            if not done:
                print(f"Agent did not reach the goal in {max_steps} steps.")
            return path  # Return the path for visualization

        self.env.close()

    # Helper functions to access performance data
    def get_training_data(self):
        return self.steps_per_episode, self.rewards_per_episode
