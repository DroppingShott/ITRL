# pylint: disable=import-error

import gym
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

import maze_env
from q_learning_agent import QLearningAgent

# Constants for sub-goal and end goal positions
SUB_GOAL = np.array([5, 3])
END_GOAL = np.array([8, 7])


def plot_maze_with_path(maze, path):
    """Plot the maze with the path, sub-goal, and end goal."""
    for step in path:
        maze[step[0], step[1]] = 0.5  # Path color

    maze[path[0][0], path[0][1]] = 0.75  # Start position color
    maze[SUB_GOAL[0], SUB_GOAL[1]] = 0.85  # Sub-goal color
    maze[END_GOAL[0], END_GOAL[1]] = 1.0  # End goal color

    plt.figure(figsize=(8, 8))
    sns.heatmap(maze, cmap="Blues", linewidths=0.1, linecolor="black", cbar=False)
    plt.title("Agent Path from Start to Goal with Sub-Goal")
    plt.show()


def plot_steps_per_episode(steps_per_episode):
    """Plot the number of steps per episode."""
    plt.figure(figsize=(10, 6))
    plt.plot(steps_per_episode, marker="o")
    plt.xlabel("Episode")
    plt.ylabel("Steps to Reach Goal")
    plt.title("Steps per Episode")
    plt.grid(True)
    plt.show()


def plot_rewards_per_episode(rewards_per_episode):
    """Plot the cumulative rewards per episode."""
    plt.figure(figsize=(10, 6))
    plt.plot(rewards_per_episode, marker="o")
    plt.xlabel("Episode")
    plt.ylabel("Cumulative Reward")
    plt.title("Rewards per Episode")
    plt.grid(True)
    plt.show()


def plot_q_value_heatmaps(q_table):
    """Plot the Q-value heatmaps for each action."""
    actions = ["UP", "DOWN", "LEFT", "RIGHT"]
    for i in range(4):
        plt.figure(figsize=(10, 10))
        sns.heatmap(q_table[:, :, i], cmap="coolwarm", annot=False)
        plt.title(f"Q-values for Action: {actions[i]}")
        plt.show()


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

    print("Training complete.")
    print("Q-table:")
    print(q_table)

    # Play the game and get the path the agent took
    path = agent.play_game(q_table, num_episodes=1, render=False)

    # Retrieve the training data for visualizations
    steps_per_episode, rewards_per_episode = agent.get_training_data()

    # Initialize the maze with zeros
    maze = np.zeros((env.size, env.size))

    # Plot the maze with the path, sub-goal, and end goal
    plot_maze_with_path(maze, path)

    # Plot the steps per episode
    plot_steps_per_episode(steps_per_episode)

    # Plot the rewards per episode
    plot_rewards_per_episode(rewards_per_episode)

    # Plot the Q-value heatmaps for each action
    plot_q_value_heatmaps(q_table)


if __name__ == "__main__":
    main()
