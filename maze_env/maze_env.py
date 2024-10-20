# pylint: disable=import-error
from typing import Any

import gym
import numpy as np
from gym import spaces


class MazeEnv(gym.Env):
    metadata = {"render_modes": ["rgb_array"], "render_fps": 4}

    def __init__(self):
        super(MazeEnv, self).__init__()
        self.size = 10

        self.maze = np.array(
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

        self.start_pos = np.array([1, 1])
        self.sub_goal = np.array([5, 3])
        self.end_goal = np.array([8, 7])
        self.subgoal_reached = False

        self.observation_space = spaces.MultiDiscrete([self.size, self.size])

        self.action_space = spaces.Discrete(4)

        self._action_to_direction = {
            0: np.array([0, 1]),  # Move right
            1: np.array([1, 0]),  # Move down
            2: np.array([0, -1]),  # Move left
            3: np.array([-1, 0]),  # Move up
        }

        self._agent_location = np.copy(self.start_pos)

    def reset(self) -> tuple[Any, dict]:
        """Resets the environment to the initial state."""
        self._agent_location = np.copy(self.start_pos)
        self.subgoal_reached = False
        return (
            np.array(self._agent_location),
            {},
        )

    def step(self, action) -> tuple[np.array, float, bool, bool, dict]:
        """Moves the agent in the maze."""

        direction = self._action_to_direction[action]
        new_position = self._agent_location + direction

        # Ensure the agent stays within grid bounds
        new_position = np.clip(new_position, 0, self.size - 1)

        # Ensure the new position is not a wall
        if self.maze[tuple(new_position)] == 0:
            self._agent_location = new_position

        terminated = False
        reward = -1  # Small penalty for each step

        if np.array_equal(self._agent_location, self.end_goal) and self.subgoal_reached:
            reward = 100  # Reward for reaching the end-goal
            terminated = True

        # Rewards and penalties
        if np.array_equal(self._agent_location, self.sub_goal):
            reward = 2  # Reward for reaching subgoal
            self.subgoal_reached = True

        return np.array(self._agent_location), reward, terminated, False, {}

    def render(self, mode="rgb_array") -> Any:
        """Renders the current state of the environment as a grid array."""
        if mode == "rgb_array":
            return self._render_frame()

    def _render_frame(self) -> Any:
        """Generate a simple grid representation (3D numpy array for 'rgb_array')."""
        img = (
            np.ones((self.size, self.size, 3), dtype=np.uint8) * 255
        )  # White background

        # Walls in black
        for x in range(self.size):
            for y in range(self.size):
                if self.maze[x, y] == 1:
                    img[x, y] = [0, 0, 0]

        # Sub-goal in green
        img[self.sub_goal[0], self.sub_goal[1]] = [0, 255, 0]

        # End-goal in red
        img[self.end_goal[0], self.end_goal[1]] = [255, 0, 0]

        # Agent in blue
        img[self._agent_location[0], self._agent_location[1]] = [
            0,
            0,
            255,
        ]

        return img
