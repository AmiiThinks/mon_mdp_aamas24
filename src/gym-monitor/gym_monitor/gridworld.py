from contextlib import closing
import numpy as np
import gymnasium as gym
from gymnasium.error import DependencyNotInstalled
from typing import Optional
from io import StringIO

LEFT = 0
DOWN = 1
RIGHT = 2
UP = 3

# these are both identifiers and reward values
EMPTY = 0
GOOD = 1
BAD = -10

# rendering colors
RED = (255, 0, 0)
GREEN = (0, 255, 0)
BLUE = (0, 0, 255)
ORANGE = (255, 175, 0)
WHITE = (255, 255, 255)
BLACK = (0, 0, 0)

INT_TO_ANSI = {
    EMPTY: b"E",
    GOOD: b"G",
    BAD: b"B",
}

GRIDS = {
    "3x3_easy": [
        [EMPTY, EMPTY, GOOD],
        [EMPTY, EMPTY, EMPTY],
        [EMPTY, EMPTY, EMPTY],
    ],
    "3x3_medium": [
        [EMPTY, BAD, GOOD],
        [EMPTY, BAD, EMPTY],
        [EMPTY, EMPTY, EMPTY],
    ],
    "2x3_easy": [
        [EMPTY, EMPTY, GOOD],
        [EMPTY, EMPTY, EMPTY],
    ],
    "2x3_medium": [
        [EMPTY, BAD, GOOD],
        [EMPTY, EMPTY, EMPTY],
    ],
}


def _move(row, col, a, nrow, ncol):
    if a == LEFT:
        col = max(col - 1, 0)
    elif a == DOWN:
        row = min(row + 1, nrow - 1)
    elif a == RIGHT:
        col = min(col + 1, ncol - 1)
    elif a == UP:
        row = max(row - 1, 0)
    else:
        raise ValueError("illegal action")
    return (row, col)


class Gridworld(gym.Env):
    """
    Gridworld where the agent has to reach a goal while avoid penalty cells.

    ## Grid
    The grid is defined by a 2D array of integers. It is possible to define
    custom grids.

    ## Action Space
    The action shape is discrete in the range `{0, 3}`.

    - 0: Move left
    - 1: Move down
    - 2: Move right
    - 3: Move up

    ## Observation Space
    The action shape is discrete in the range `{0, n_rows * n_cols - 1}`.
    Each integer denotes a cell. For a 3x3 grid:

     0 1 2
     3 4 5
     6 7 8

    ## Starting State
    The episode starts with the agent at the top-left cell.

    ## Transition
    By default, the transition is deterministic. It can be made stochastic by
    passing 'random_action_prob'. This is the probability that the action will
    be random. For example, if random_action_prob=0.1, there is a 10% chance
    that instead of doing LEFT / RIGHT / ... as passed in self.step(action)
    the agent will do a random action.

    ## Rewards
    - Reaching the goal: +1
    - Walking over penalty cells: -10
    - Otherwise: 0

    Gaussian noise can be added to the rewards by passing 'reward_noise_std'.

    ## Episode End
    The episode ends if the following happens:

    - Termination:
        1. The goal is reached.

    - Truncation:
        1. The length of the episode is 50.

    ## Rendering
    Human mode renders the environment as a grid with colored cells.

    - Black: empty cells
    - Green: goal
    - Red: penalty cells
    - Blue: agent
    - Orange arrow: last action

    """

    metadata = {
        "render_modes": ["human", "rgb_array", "ansi", "binary"],
        "render_fps": 30,
    }

    def __init__(
        self,
        render_mode: Optional[str] = None,
        grid: Optional[str] = "3x3_medium",
        random_action_prob: Optional[float] = 0.0,
        reward_noise_std: Optional[float] = 0.0,
        **kwargs,
    ):
        self.grid_key = grid
        self.grid = np.asarray(GRIDS[self.grid_key])
        self.random_action_prob = random_action_prob
        self.reward_noise_std = reward_noise_std

        self.n_rows, self.n_cols = self.grid.shape
        self.observation_space = gym.spaces.Discrete(self.n_cols * self.n_rows)
        self.action_space = gym.spaces.Discrete(4)
        self.agent_pos = None
        self.last_action = None

        self.render_mode = render_mode
        self.window_surface = None
        self.clock = None
        self.window_size = (
            min(64 * self.n_cols, 512),
            min(64 * self.n_rows, 512)
        )  # fmt: skip
        self.cell_size = (
            self.window_size[0] // self.n_cols,
            self.window_size[1] // self.n_rows,
        )  # fmt: skip

    def get_state(self):
        return np.ravel_multi_index(self.agent_pos, (self.n_rows, self.n_cols))

    def reset(self, seed: int = None, **kwargs):
        super().reset(seed=seed, **kwargs)
        self.grid = np.asarray(GRIDS[self.grid_key])
        self.agent_pos = (0, 0)
        self.last_action = None
        self.last_pos = None
        return self.get_state(), {}

    def step(self, action: int):
        self.last_pos = self.agent_pos

        if self.np_random.random() < self.random_action_prob:
            action = self.action_space.sample()

        self.agent_pos = _move(
            self.agent_pos[0], self.agent_pos[1], action, self.n_rows, self.n_cols
        )

        reward = self.grid[self.agent_pos] + self.np_random.normal() * self.reward_noise_std

        terminated = False
        if self.grid[self.agent_pos] == GOOD:
            terminated = True

        self.last_action = action

        return self.get_state(), reward, terminated, False, {}

    def render(self):
        if self.render_mode is None:
            assert self.spec is not None
            gym.logger.warn(
                "You are calling render method without specifying any render mode. "
                "You can specify the render_mode at initialization, "
                f'e.g. gym.make("{self.spec.id}", render_mode="rgb_array")'
            )
            return
        if self.render_mode == "ansi":
            return self._render_text()
        elif self.render_mode == "binary":
            return self._render_binary()
        else:  # self.render_mode in {"human", "rgb_array"}:
            return self._render_gui(self.render_mode)

    def _render_binary(self):
        obs = np.zeros(self.grid.shape, dtype=np.uint8)
        obs[self.agent_pos] = 1
        return obs

    def _render_gui(self, mode):
        try:
            import pygame
        except ImportError as e:
            raise DependencyNotInstalled(
                "pygame is not installed, run `pip install gymnasium[toy-text]`"
            ) from e

        if self.window_surface is None:
            pygame.init()
            if mode == "human":
                pygame.display.init()
                pygame.display.set_caption("Gridworld")
                self.window_surface = pygame.display.set_mode(self.window_size)
            elif mode == "rgb_array":
                self.window_surface = pygame.Surface(self.window_size)

        assert (
            self.window_surface is not None
        ), "Something went wrong with pygame. This should never happen."  # fmt: skip

        if self.clock is None:
            self.clock = pygame.time.Clock()

        grid = self.grid.tolist()
        assert (
            isinstance(grid, list)
        ), f"grid should be a list or an array, got {grid}"  # fmt: skip

        # draw tiles
        for y in range(self.n_rows):
            for x in range(self.n_cols):
                pos = (x * self.cell_size[0], y * self.cell_size[1])
                rect = pygame.Rect(pos, self.cell_size)

                if grid[y][x] == GOOD:
                    pygame.draw.rect(self.window_surface, GREEN, rect)
                elif grid[y][x] == BAD:
                    pygame.draw.rect(self.window_surface, RED, rect)
                elif grid[y][x] == EMPTY or grid[y][x]:
                    pygame.draw.rect(self.window_surface, BLACK, rect)

                # draw agent
                if (y, x) == self.agent_pos:
                    pygame.draw.ellipse(self.window_surface, BLUE, rect)

            # draw action arrow
            if self.last_pos is not None:
                x = self.last_pos[1]
                y = self.last_pos[0]
                a = self.last_action

                pos = (
                    x * self.cell_size[0] + self.cell_size[0] / 2,
                    y * self.cell_size[1] + self.cell_size[1] / 2,
                )

                if a == LEFT:
                    end_pos = (pos[0] - self.cell_size[0] / 4, pos[1])
                    arrow_points = (
                        (end_pos[0], end_pos[1] - self.cell_size[1] / 5),
                        (end_pos[0], end_pos[1] + self.cell_size[1] / 5),
                        (end_pos[0] - self.cell_size[0] / 5, end_pos[1]),
                    )
                elif a == DOWN:
                    end_pos = (pos[0], pos[1] + self.cell_size[1] / 4)
                    arrow_points = (
                        (end_pos[0] - self.cell_size[0] / 5, end_pos[1]),
                        (end_pos[0] + self.cell_size[0] / 5, end_pos[1]),
                        (end_pos[0], end_pos[1] + self.cell_size[1] / 5),
                    )
                elif a == RIGHT:
                    end_pos = (pos[0] + self.cell_size[0] / 4, pos[1])
                    arrow_points = (
                        (end_pos[0], end_pos[1] - self.cell_size[1] / 5),
                        (end_pos[0], end_pos[1] + self.cell_size[1] / 5),
                        (end_pos[0] + self.cell_size[0] / 5, end_pos[1]),
                    )
                elif a == UP:
                    end_pos = (pos[0], pos[1] - self.cell_size[1] / 4)
                    arrow_points = (
                        (end_pos[0] - self.cell_size[0] / 5, end_pos[1]),
                        (end_pos[0] + self.cell_size[0] / 5, end_pos[1]),
                        (end_pos[0], end_pos[1] - self.cell_size[1] / 5),
                    )
                else:
                    raise ValueError("illegal action")

                pygame.draw.polygon(self.window_surface, ORANGE, (pos, end_pos), 10)
                pygame.draw.polygon(self.window_surface, ORANGE, arrow_points, 0)

        if mode == "human":
            pygame.event.pump()
            pygame.display.update()
            self.clock.tick(self.metadata["render_fps"])

        elif mode == "rgb_array":
            return np.transpose(
                np.array(pygame.surfarray.pixels3d(self.window_surface)), axes=(1, 0, 2)
            )

        else:
            raise NotImplementedError

    def _render_text(self):
        grid = self.grid.tolist()
        outfile = StringIO()

        grid = [[INT_TO_ANSI[c].decode("utf-8") for c in line] for line in grid]
        grid[self.agent_pos[0]][self.agent_pos[1]] = gym.utils.colorize(
            grid[self.agent_pos[0]][self.agent_pos[1]], "red", highlight=True
        )
        if self.last_action is not None:
            outfile.write(f"  ({['Left', 'Down', 'Right', 'Up'][self.last_action]})\n")
        else:
            outfile.write("\n")
        outfile.write("\n".join("".join(line) for line in grid) + "\n")

        with closing(outfile):
            return outfile.getvalue()

    def close(self):
        if self.window_surface is not None:
            import pygame

            pygame.display.quit()
            pygame.quit()
