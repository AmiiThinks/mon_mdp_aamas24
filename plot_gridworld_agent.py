# ruff: noqa: F403, F405
from matplotlib import pyplot as plt
import matplotlib
import numpy as np
import seaborn as sns
import os
import pygame

from src.plot_utils import *  # noqa: F405


def plot_agent(critic, env, savepath=None):
    """
    Plot and save the Q-tables and the greedy policy of an agent trained on a
    Gridworld.
    """

    if critic._strategy in ["q_joint", "q_sequential"]:
        q_env = critic._q_env._table
        q_mon = critic._q_mon._table

        filepath = None
        if savepath is not None:
            filepath = os.path.join(savepath, critic._strategy + "_q_mon.png")
        plot_q(q_mon, filepath)

        filepath = None
        if savepath is not None:
            filepath = os.path.join(savepath, critic._strategy + "_q_env.png")
        plot_q(q_env[:, None, :, None], filepath)

        if critic._strategy in ["q_sequential"]:
            n_senv, n_smon, n_aenv, n_amon = q_mon.shape
            actions_env = [[a[0] for a in np.argwhere(q_env[i] == q_env[i].max())]
                for i in range(n_senv)]  # fmt: skip
            actions = [[[(a_env, a_mon[0])
                for a_env in actions_env[i]
                for a_mon in np.argwhere(q_mon[i, j, a_env] == q_mon[i, j, a_env].max())]
                for i in range(n_senv)]
                for j in range(n_smon)]  # fmt: skip

        else:
            q = q_mon + q_env[:, None, :, None]
            n_senv, n_smon, n_aenv, n_amon = q.shape
            actions = [[[tuple(a) for a in np.argwhere(q[i, j] == q[i, j].max())]
                for i in range(n_senv)]
                for j in range(n_smon)]  # fmt: skip

    else:
        q = critic._q_joint._table
        filepath = None
        if savepath is not None:
            filepath = os.path.join(savepath, critic._strategy + "_q_joint.png")
        plot_q(q, filepath)

        n_senv, n_smon, n_aenv, n_amon = q.shape
        actions = [[[tuple(a) for a in np.argwhere(q[i, j] == q[i, j].max())]
            for i in range(n_senv)]
            for j in range(n_smon)]  # fmt: skip

        if critic._strategy in ["reward_model", "oracle_with_reward_model"]:
            r_env = critic._r_env._table[:, None, :, None]
            filepath = None
            if savepath is not None:
                filepath = os.path.join(savepath, critic._strategy + "_r_env.png")
            plot_q(r_env, filepath)

    filepath = os.path.join(savepath, critic._strategy + "_policy.png")
    plot_pi(env, actions, filepath)


def plot_pi(env, actions_list, filepath=None):
    """
    Plot a policy returned for Gridworlds.
    actions_list must be a list of actions for every state. For example,

        >>> actions[0]     <--- monitor state index
        [[(3, 0)],         <--- list of tuples of (env. action, mon. action) for env. state 0
        [(2, 1)],          <--- list for env. state 1
        [(0, 1), (1, 1)],  <--- list for env. state 2
        ...

    Means that the policy:
     - In mon. state 0 and env. state 0 executes env. action 3 and mon. action 0,
     - In mon. state 0 and env. state 1 executes env. action 2 and mon. action 1,
     - In mon. state 0 and env. state 2 executes env. action 0 and mon. action 1,
       or env. action 1 and mon. action 1, both with equal probability,
     - ... and so on, for all states.
    """

    fig, axs = make_subplots(1, len(actions_list), width_per_plot=2, height_per_plot=2)

    render_mode = env.unwrapped.render_mode
    agent_pos = env.unwrapped.agent_pos
    last_pos = env.unwrapped.last_pos
    env.unwrapped.render_mode = "rgb_array"
    env.unwrapped.agent_pos = None
    env.unwrapped.last_pos = None
    cell_size = env.unwrapped.cell_size
    n_rows, n_cols = env.unwrapped.grid.shape
    ORANGE = (255, 175, 0)
    WHITE = (255, 255, 255)

    for i, actions in enumerate(actions_list):
        env.render()
        surf = env.unwrapped.window_surface

        for y in range(n_rows):
            for x in range(n_cols):
                pos = (x * cell_size[0], y * cell_size[1])

                for a_joint in actions[np.ravel_multi_index((y, x), (n_rows, n_cols))]:
                    pos = (
                        x * cell_size[0] + cell_size[0] / 2,
                        y * cell_size[1] + cell_size[1] / 2,
                    )

                    if a_joint[0] == 0:  # left
                        end_pos = (pos[0] - cell_size[0] / 4, pos[1])
                        arrow_points = (
                            (end_pos[0], end_pos[1] - cell_size[1] / 5),
                            (end_pos[0], end_pos[1] + cell_size[1] / 5),
                            (end_pos[0] - cell_size[0] / 5, end_pos[1]),
                        )
                        arrow_width = -(-cell_size[1] // 6)
                    elif a_joint[0] == 1:  # down
                        end_pos = (pos[0], pos[1] + cell_size[1] / 4)
                        arrow_points = (
                            (end_pos[0] - cell_size[0] / 5, end_pos[1]),
                            (end_pos[0] + cell_size[0] / 5, end_pos[1]),
                            (end_pos[0], end_pos[1] + cell_size[1] / 5),
                        )
                        arrow_width = -(-cell_size[0] // 6)
                    elif a_joint[0] == 2:  # right
                        end_pos = (pos[0] + cell_size[0] / 4, pos[1])
                        arrow_points = (
                            (end_pos[0], end_pos[1] - cell_size[1] / 5),
                            (end_pos[0], end_pos[1] + cell_size[1] / 5),
                            (end_pos[0] + cell_size[0] / 5, end_pos[1]),
                        )
                        arrow_width = -(-cell_size[1] // 6)
                    elif a_joint[0] == 3:  # up
                        end_pos = (pos[0], pos[1] - cell_size[1] / 4)
                        arrow_points = (
                            (end_pos[0] - cell_size[0] / 5, end_pos[1]),
                            (end_pos[0] + cell_size[0] / 5, end_pos[1]),
                            (end_pos[0], end_pos[1] - cell_size[1] / 5),
                        )
                        arrow_width = -(-cell_size[0] // 6)
                    else:
                        raise ValueError("illegal action")

                    if a_joint[1] == 0:
                        pygame.draw.polygon(surf, ORANGE, (pos, end_pos), arrow_width)
                        pygame.draw.polygon(surf, ORANGE, arrow_points, 0)
                    elif a_joint[1] == 1:
                        pygame.draw.polygon(surf, WHITE, (pos, end_pos), 5).scale_by(0.5)
                        pygame.draw.polygon(surf, WHITE, arrow_points, 5).scale_by(0.5)
                    else:
                        raise ValueError("illegal action")

        rgb_array = np.transpose(np.array(pygame.surfarray.pixels3d(surf)), (1, 0, 2))
        axs[0][i].imshow(rgb_array)
        axs[0][i].set_title(f"Monitor State {i}")
        axs[0][i].axes.xaxis.set_visible(False)
        axs[0][i].axes.yaxis.set_visible(False)

    env.unwrapped.render_mode = render_mode
    env.unwrapped.agent_pos = agent_pos
    env.unwrapped.last_pos = last_pos

    if filepath is None:
        plt.show()
    else:
        plt.draw()
        plt.savefig(filepath, bbox_inches="tight", pad_inches=0, dpi=1500)


def plot_q_alt(q, filepath=None):
    """
    Plot a Q-table of shape (n_env_states, n_mon_states, n_env_actions, n_mon_actions).
    The image is a grid of heatmaps where
    - Outer rows are monitor states,
    - Outer columns are environment states,
    - Inner rows are monitor actions,
    - Inner columns are environment actions.
    Therefore, every heatmap has the value of all (env_action, mon_action) pairs.
    The pair with the highest value is highlighted in red.
    """

    n_senv, n_smon, n_aenv, n_amon = q.shape
    original_q = q.copy()
    nonz = q != 0
    q[nonz] = np.sign(q[nonz]) * np.log(np.abs(q[nonz]) + 1.0)  # for better colormaps
    fig, axs = make_subplots(n_smon, n_senv, width_per_plot=n_aenv, height_per_plot=n_amon)

    for i in range(n_smon):
        for j in range(n_senv):
            sns.heatmap(
                q[j, i, :, :].T,
                ax=axs[i][j],
                cmap="plasma",
                cbar=False,
                annot=original_q[j, i, :, :].T,
                fmt=".3f",
                vmin=q.min(),
                vmax=q.max(),
            )

            for ii in range(n_amon):
                for jj in range(n_aenv):
                    if q[j, i, jj, ii] == q[j, i, :, :].max():
                        highlight_cell(axs[i][j], jj, ii, color="r", linewidth=3)

            if i == 0:
                axs[i][j].set_title(f"Env. State {j}", pad=10)

            if j == 0:
                axs[i][j].set_ylabel("Mon. Action")
            else:
                for tick in axs[i][j].yaxis.get_major_ticks():
                    tick.tick1line.set_visible(False)
                    tick.tick2line.set_visible(False)
                    tick.label1.set_visible(False)
                    tick.label2.set_visible(False)

            if i != n_smon - 1:
                axs[i][j].xaxis.set_ticks([])
            else:
                axs[i][j].set_xlabel("Env. Action")

            if j == n_senv - 1:
                axs[i][j].yaxis.set_label_position("right")

                # invisible axis for extra label
                ax = fig.add_subplot(n_smon, 1, i + 1, frame_on=False)
                ax.tick_params(labelcolor="none", bottom=False, left=False)
                ax.set_ylabel(
                    f"Mon. State {i}",
                    labelpad=10,
                    fontsize=matplotlib.rcParams["axes.titlesize"],
                )
                ax.grid(False)

            bbox = axs[i][j].get_position()
            rect = matplotlib.patches.Rectangle(
                (0, bbox.y0), 1, bbox.height, color="0.7", zorder=-1
            )
            fig.add_artist(rect)

    if filepath is None:
        plt.show()
    else:
        plt.draw()
        plt.savefig(filepath, bbox_inches="tight", pad_inches=0, dpi=1500)


def plot_q(q, filepath=None):
    """
    Plot a Q-table of shape (n_env_states, n_mon_states, n_env_actions, n_mon_actions).
    The image is a grid of heatmaps where
    - Outer rows are monitor states,
    - Outer columns are monitor actions,
    - Inner rows are environment states,
    - Inner columns are environment actions.
    Therefore, every row across every column has the value of all (env_action, mon_action) pairs.
    The pair with the highest value is highlighted in red.
    """

    n_senv, n_smon, n_aenv, n_amon = q.shape
    original_q = q.copy()
    nonz = q != 0
    q[nonz] = np.sign(q[nonz]) * np.log(np.abs(q[nonz]) + 1.0)  # for better colormaps
    fig, axs = make_subplots(n_smon, n_amon, width_per_plot=n_aenv, height_per_plot=n_senv)

    for i in range(n_smon):
        for j in range(n_amon):
            sns.heatmap(
                q[:, i, :, j],
                ax=axs[i][j],
                cmap="plasma",
                cbar=False,
                annot=original_q[:, i, :, j],
                fmt=".3f",
                vmin=q.min(),
                vmax=q.max(),
            )

            for ii in range(n_senv):
                for jj in range(n_aenv):
                    if q[ii, i, jj, j] == q[ii, i, :, :].max():
                        highlight_cell(axs[i][j], jj, ii, color="r", linewidth=2)

            if i == 0:
                axs[i][j].set_title(f"Mon. Action {j}", pad=10)

            if j == 0:
                axs[i][j].set_ylabel("Env. State")
            else:
                for tick in axs[i][j].yaxis.get_major_ticks():
                    tick.tick1line.set_visible(False)
                    tick.tick2line.set_visible(False)
                    tick.label1.set_visible(False)
                    tick.label2.set_visible(False)

            if i != n_smon - 1:
                axs[i][j].xaxis.set_ticks([])
            else:
                axs[i][j].set_xlabel("Env. Action")

            if j == 0:
                # axs[i][j].yaxis.set_label_position("right")

                # invisible axis for extra label
                ax = fig.add_subplot(n_smon, 1, i + 1, frame_on=False)
                ax.tick_params(labelcolor="none", bottom=False, left=False)
                ax.set_ylabel(
                    f"Mon. State {i}",
                    labelpad=12,
                    fontsize=matplotlib.rcParams["axes.titlesize"],
                )
                ax.grid(False)

            bbox = axs[i][j].get_position()
            rect = matplotlib.patches.Rectangle(
                (0, bbox.y0), 1, bbox.height, color="0.7", zorder=-1
            )
            fig.add_artist(rect)

    if filepath is None:
        plt.show()
    else:
        plt.draw()
        plt.savefig(filepath, bbox_inches="tight", pad_inches=0, dpi=1500)
