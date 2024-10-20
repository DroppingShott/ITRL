# pylint: disable=import-error
import gym

import maze_env
from q_learning_agent import QLearningAgent


def main():
    """Main function to run the Q-learning agent in the maze environment."""
    # Environment setup
    env = gym.make("Maze-v0")

    agent = QLearningAgent(
        env=env,
        alpha=0.8,  # Learning rate
        gamma=0.95,  # Discount factor
        epsilon=1.0,  # Initial exploration rate
        epsilon_decay=0.99,  # Exploration decay rate
        min_epsilon=0.05,  # Minimum exploration rate
    )

    # Train the agent
    q_table = agent.train(num_episodes=1500, log_interval=10)

    # Play the game using the trained Q-table
    agent.play_game(q_table, max_steps=10000, num_episodes=1)


if __name__ == "__main__":
    main()
