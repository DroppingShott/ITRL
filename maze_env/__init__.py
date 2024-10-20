from gym.envs.registration import register

from maze_env.maze_env import MazeEnv

__all__ = ["MazeEnv"]


register(
    id="Maze-v0",
    entry_point="maze_env.maze_env:MazeEnv",
    max_episode_steps=3000,
)
