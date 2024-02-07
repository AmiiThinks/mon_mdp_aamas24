import gymnasium
from gymnasium import spaces
import numpy as np
from abc import abstractmethod


class Monitor(gymnasium.Wrapper):
    """
    Generic class for monitors that DO NOT depend on the environment state.
    Monitors that DO depend on the environment state need to be customized
    according to the environment.

    Args:
        env (gymnasium.Env): the Gymnasium environment.
    """

    @abstractmethod
    def _monitor_step(self, action, env_reward):
        pass

    def step(self, action):
        """
        This type of monitors DO NOT depend on the environment state.
        Therefore, we first execute self.env.step() and then self._monitor_step().
        Everything else works as in classic Gymnasium environments, but state,
        actions, and rewards are dictionaries. That is, the agent expects

            actions = {"env": action_env, "mon": action_mon}

        and returns

            state = {"env": state_env, "mon": state_mon}
            reward = {"env": reward_env, "mon": reward_mon, "proxy": reward_proxy}

        Terminate, truncated, and info remain the same as self.env.step().
        """
        (
            env_obs,
            env_reward,
            env_terminated,
            env_truncated,
            env_info,
        ) = self.env.step(action["env"])

        (
            monitor_obs,
            proxy_reward,
            monitor_reward,
        ) = self._monitor_step(action, env_reward)

        obs = {"env": env_obs, "mon": monitor_obs}
        reward = {"env": env_reward, "mon": monitor_reward, "proxy": proxy_reward}
        terminated = env_terminated
        truncated = env_truncated

        return obs, reward, terminated, truncated, env_info


class FullMonitor(Monitor):
    """
    This monitor always shows the true reward, regardless of its state and action.
    The monitor reward is always 0.
    This is a 'trivial Mon-MDP', i.e., it is equivalent to a classic MDP.

    Args:
        env (gymnasium.Env): the Gymnasium environment.
    """

    def __init__(self, env, **kwargs):
        gymnasium.Wrapper.__init__(self, env)
        self.action_space = spaces.Dict({
            "env": env.action_space,
            "mon": spaces.Discrete(1),
        })  # fmt: skip
        self.observation_space = spaces.Dict({
            "env": env.observation_space,
            "mon": spaces.Discrete(1),
        })  # fmt: skip

    def reset(self, seed=None, **kwargs):
        self.action_space.seed(seed)
        self.observation_space.seed(seed)
        env_obs, env_info = self.env.reset(seed=seed, **kwargs)
        monitor_obs = 0  # default monitor state, always on
        return {"env": env_obs, "mon": monitor_obs}, env_info

    def _monitor_step(self, action, env_reward):
        monitor_reward = 0.0
        proxy_reward = env_reward
        monitor_obs = 0
        return monitor_obs, proxy_reward, monitor_reward


class StatefulBinaryMonitor(Monitor):
    """
    Simple monitor where the action is "turn on monitor" / "do nothing".
    The monitor state is also binary ("monitor on" / "monitor off").
    The monitor reward is a constant penalty given if the monitor is on or turned on.

    The monitor can turn off itself randomly at every time step (default probability is 0).
    If the monitor is on or being turned on, the true reward is observed.

    Args:
        env (gymnasium.Env): the Gymnasium environment,
        monitor_cost (float): cost for monitor being active,
        monitor_reset_prob (float): probability of the monitor resetting itself.
    """

    def __init__(self, env, monitor_cost=0.2, monitor_reset_prob=0.0, **kwargs):
        gymnasium.Wrapper.__init__(self, env)
        self.action_space = spaces.Dict({
            "env": env.action_space,
            "mon": spaces.Discrete(2),
        })  # fmt: skip
        self.observation_space = spaces.Dict({
            "env": env.observation_space,
            "mon": spaces.Discrete(2),
        })  # fmt: skip
        self.monitor_state = 0  # off
        self.monitor_reset_prob = monitor_reset_prob
        self.monitor_cost = monitor_cost

    def reset(self, seed=None, **kwargs):
        self.action_space.seed(seed)
        self.observation_space.seed(seed)
        env_obs, env_info = self.env.reset(seed=seed, **kwargs)
        self.monitor_state = 0  # monitor starts off
        return {"env": env_obs, "mon": self.monitor_state}, env_info

    def _monitor_step(self, action, env_reward):
        if action["mon"] == 1:
            self.monitor_state = 1
        elif action["mon"] == 0:
            pass
        else:
            raise ValueError("illegal monitor action")

        if self.monitor_state == 1:
            proxy_reward = env_reward
            monitor_reward = -self.monitor_cost
        else:
            proxy_reward = np.nan
            monitor_reward = 0.0

        if self.np_random.random() < self.monitor_reset_prob:
            self.monitor_state = 0
        monitor_obs = self.monitor_state

        return monitor_obs, proxy_reward, monitor_reward


class StatelessBinaryMonitor(Monitor):
    """
    Simple monitor where the action is "turn on monitor" / "do nothing".
    The monitor is always off. The reward is seen only when the agent asks for it.
    The monitor reward is a constant penalty given if the agent asks to see the reward.

    Args:
        env (gymnasium.Env): the Gymnasium environment,
        monitor_cost (float): cost for asking the monitor for rewards.
    """

    def __init__(self, env, monitor_cost=0.2, **kwargs):
        gymnasium.Wrapper.__init__(self, env)
        self.action_space = spaces.Dict({
            "env": env.action_space,
            "mon": spaces.Discrete(2),
        })  # fmt: skip
        self.observation_space = spaces.Dict({
            "env": env.observation_space,
            "mon": spaces.Discrete(1),
        })  # fmt: skip
        self.monitor_state = 0  # off
        self.monitor_cost = monitor_cost

    def reset(self, seed=None, **kwargs):
        self.action_space.seed(seed)
        self.observation_space.seed(seed)
        env_obs, env_info = self.env.reset(seed=seed, **kwargs)
        self.monitor_state = 0  # monitor always off
        return {"env": env_obs, "mon": self.monitor_state}, env_info

    def _monitor_step(self, action, env_reward):
        if action["mon"] == 1:
            proxy_reward = env_reward
            monitor_reward = -self.monitor_cost
        else:
            proxy_reward = np.nan
            monitor_reward = 0.0
        monitor_obs = self.monitor_state
        return monitor_obs, proxy_reward, monitor_reward


class NMonitor(Monitor):
    """
    There are N monitors. At every time step, a random monitor is on.
    If the agent's action matches the monitor state, the agent observes the
    environment reward but receives a negative monitor reward.
    Otherwise it does not observe the environment reward, but receives a smaller
    positive monitor reward.
    For example, if state = 2 and action = 2, then the agent observes the environment
    reward and gets reward_monitor = -0.2.
    If state = 2 and action != 2, the agent does not observe the reward but
    gets reward_monitor = 0.001.

    Args:
        env (gymnasium.Env): the Gymnasium environment,
        n_monitors (int): number of monitors,
        monitor_cost (float): cost for observing the reward,
        monitor_bonus (float): reward for not observing the reward.
    """

    def __init__(self, env, n_monitors=5, monitor_cost=0.2, monitor_bonus=0.001, **kwargs):
        gymnasium.Wrapper.__init__(self, env)
        self.action_space = spaces.Dict({
            "env": env.action_space,
            "mon": spaces.Discrete(n_monitors),
        })  # fmt: skip
        self.observation_space = spaces.Dict({
            "env": env.observation_space,
            "mon": spaces.Discrete(n_monitors),
        })  # fmt: skip
        self.monitor_state = self.observation_space["mon"].sample()
        self.monitor_cost = monitor_cost
        self.monitor_bonus = monitor_bonus

    def reset(self, seed=None, **kwargs):
        self.action_space.seed(seed)
        self.observation_space.seed(seed)
        env_obs, env_info = self.env.reset(seed=seed, **kwargs)
        self.monitor_state = self.observation_space["mon"].sample()
        return {"env": env_obs, "mon": self.monitor_state}, env_info

    def _monitor_step(self, action, env_reward):
        assert (
            action["mon"] < self.action_space["mon"].n
        ), "illegal monitor action"  # fmt: skip

        if action["mon"] == self.monitor_state:
            proxy_reward = env_reward
            monitor_reward = -self.monitor_cost
        else:
            proxy_reward = np.nan
            monitor_reward = self.monitor_bonus

        self.monitor_state = self.observation_space["mon"].sample()
        monitor_obs = self.monitor_state

        return monitor_obs, proxy_reward, monitor_reward


class LevelMonitor(Monitor):
    """
    The monitor has N levels, from 0 to N - 1.
    The initial level is 0, and it increases if the agent's action matches the
    current level.
    For example, if state = 2 and action = 2, then next_state = 3.
    If the agent executes the wrong action, the level resets to 0.
    Action N does nothing.
    Environment rewards will become visible only when the monitor level is max.
    Leveling up the monitor is costly, but once the monitor is maxed observing
    rewards is cost-free.

    Args:
        env (gymnasium.Env): the Gymnasium environment,
        n_levels (int): number of levels,
        monitor_cost (float): cost for leveling up the monitor state.
    """

    def __init__(self, env, n_levels=4, monitor_cost=0.2, **kwargs):
        gymnasium.Wrapper.__init__(self, env)
        self.action_space = spaces.Dict({
            "env": env.action_space,
            "mon": spaces.Discrete(n_levels + 1),  # last action is "do nothing"
        })  # fmt: skip
        self.observation_space = spaces.Dict({
            "env": env.observation_space,
            "mon": spaces.Discrete(n_levels),
        })  # fmt: skip
        self.monitor_state = 0
        self.monitor_cost = monitor_cost

    def reset(self, seed=None, **kwargs):
        self.action_space.seed(seed)
        self.observation_space.seed(seed)
        env_obs, env_info = self.env.reset(seed=seed, **kwargs)
        self.monitor_state = 0
        return {"env": env_obs, "mon": self.monitor_state}, env_info

    def _monitor_step(self, action, env_reward):
        assert (
            action["mon"] < self.action_space["mon"].n
        ), "illegal monitor action"  # fmt: skip

        monitor_reward = 0.0
        proxy_reward = np.nan

        if self.monitor_state == self.observation_space["mon"].n - 1:
            proxy_reward = env_reward

        if action["mon"] == self.action_space["mon"].n - 1:
            pass  # last action is "do nothing"
        else:
            monitor_reward = -self.monitor_cost  # pay cost
            if action["mon"] == self.monitor_state:
                self.monitor_state += 1  # raise level
                if self.monitor_state > self.observation_space["mon"].n - 1:  # level is already max
                    self.monitor_state = self.observation_space["mon"].n - 1
            else:
                self.monitor_state = 0  # reset level

        monitor_obs = self.monitor_state

        return monitor_obs, proxy_reward, monitor_reward


class LimitedTimeMonitor(Monitor):
    """
    The monitor is on at the beginning of the episode and the agent sees
    rewards for free.
    At every step, there is a small probability that the monitor goes off.
    If it goes off, it stays off until the end of the episode.

    Args:
        env (gymnasium.Env): the Gymnasium environment,
        monitor_reset_prob (float): probability of the monitor resetting itself.
    """

    def __init__(self, env, monitor_reset_prob=0.2, **kwargs):
        gymnasium.Wrapper.__init__(self, env)
        self.action_space = spaces.Dict({
            "env": env.action_space,
            "mon": spaces.Discrete(2),
        })  # fmt: skip
        self.observation_space = spaces.Dict({
            "env": env.observation_space,
            "mon": spaces.Discrete(2),
        })  # fmt: skip
        self.monitor_state = 1
        self.monitor_reset_prob = monitor_reset_prob

    def reset(self, seed=None, **kwargs):
        self.action_space.seed(seed)
        self.observation_space.seed(seed)
        env_obs, env_info = self.env.reset(seed=seed, **kwargs)
        self.monitor_state = 1  # monitor starts on
        return {"env": env_obs, "mon": self.monitor_state}, env_info

    def _monitor_step(self, action, env_reward):
        monitor_reward = 0.0
        if self.monitor_state == 1:
            proxy_reward = env_reward
        else:
            proxy_reward = np.nan

        if self.np_random.random() < self.monitor_reset_prob:
            self.monitor_state = 0
        monitor_obs = self.monitor_state

        return monitor_obs, proxy_reward, monitor_reward


class LimitedUseMonitor(Monitor):
    """
    The monitor has a battery that is consumed whenever it is on.
    The state of the monitor is (battery level, monitor on/off).
    The battery level goes from 0, 1, 2, ..., N.
    Every time step, if the monitor is on the battery goes down by 1.
    When the battery level reaches 0 the monitor stays off.
    The agent can turn the monitor on / off or do nothing.

    Args:
        env (gymnasium.Env): the Gymnasium environment,
        max_battery (int): how many time steps the monitor can stay on.
    """

    def __init__(self, env, max_battery=5, **kwargs):
        gymnasium.Wrapper.__init__(self, env)
        self.action_space = spaces.Dict({
            "env": env.action_space,
            "mon": spaces.Discrete(3),  # turn on/off, do nothing
        })  # fmt: skip
        self.observation_space = spaces.Dict({
            "env": env.observation_space,
            "mon": spaces.Discrete((max_battery + 1) * 2),  # battery levels, monitor on/off
        })  # fmt: skip
        self.max_battery = max_battery
        self.monitor_state = 0
        self.monitor_battery = max_battery

    def reset(self, seed=None, **kwargs):
        self.action_space.seed(seed)
        self.observation_space.seed(seed)
        env_obs, env_info = self.env.reset(seed=seed, **kwargs)
        self.monitor_state = 0  # monitor starts off
        self.monitor_battery = self.max_battery  # battery full
        monitor_obs = np.ravel_multi_index(
            (self.monitor_battery, self.monitor_state), (self.max_battery + 1, 2)
        )
        return {"env": env_obs, "mon": monitor_obs}, env_info

    def _monitor_step(self, action, env_reward):
        proxy_reward = np.nan
        monitor_reward = 0.0

        if self.monitor_state == 1:
            proxy_reward = env_reward
            self.monitor_battery -= 1

        if action["mon"] == 0:  # turn on
            self.monitor_state = 1
        elif action["mon"] == 1:  # turn off
            self.monitor_state = 0
        elif action["mon"] == 2:  # do nothing
            pass
        else:
            raise ValueError("illegal monitor action")

        if self.monitor_battery < 0:
            self.monitor_battery = 0

        if self.monitor_battery == 0:
            self.monitor_state = 0

        monitor_obs = np.ravel_multi_index(
            (self.monitor_battery, self.monitor_state), (self.max_battery + 1, 2)
        )

        return monitor_obs, proxy_reward, monitor_reward


class LimitedUseBonusMonitor(LimitedUseMonitor):
    """
    Like LimitedUseMonitor, but terminal states will give a bonus of +1 if the
    battery is depleted.
    The agent should learn to turn on the monitor such that the battery will be
    depleted by the time a terminal state is reached.
    """

    def step(self, action):
        obs, reward, terminated, truncated, info = super().step(action)
        if self.monitor_battery == 0 and terminated:
            reward["mon"] = 1.0
        return obs, reward, terminated, truncated, info


class ToySwitchMonitor(Monitor):
    """
    Monitor for the Gridworld.
    The monitor is turned on/off by doing DOWN in the BOTTOM-RIGHT cell.
    There are no explicit monitor actions.
    If the monitor is on, the agent receives negative monitor rewards and observes
    the environment rewards.
    The monitor can be already on at the beginning of the episode, so the
    optimal policy should turn it off to prevent negative monitor rewards.

    Args:
        env (gymnasium.Env): the Gymnasium environment,
        monitor_cost (float): cost for monitor being active,
        test (bool): if True, the initial monitor state will switch between
            on/off at every reset. That is, at the first reset() it will be 0,
            at the next reset() it will 1, then 0 again, then 1, and so on.
            This allows to test the environment over only 2 episodes, because
            everything else is deterministic.
    """

    def __init__(self, env, monitor_cost=0.2, test=False, **kwargs):
        gymnasium.Wrapper.__init__(self, env)
        self.action_space = spaces.Dict({
            "env": env.action_space,
            "mon": spaces.Discrete(1),  # do nothing
        })  # fmt: skip
        self.observation_space = spaces.Dict({
            "env": env.observation_space,
            "mon": spaces.Discrete(2),  # monitor on/off
        })  # fmt: skip
        n_rows, n_cols = env.unwrapped.n_rows, env.unwrapped.n_cols
        self.switch_cell_id = np.ravel_multi_index(
            (n_rows - 1, n_cols - 1), (n_rows, n_cols)
        )  # bottom-right
        self.monitor_state = 0
        self.monitor_cost = monitor_cost
        self.initial_state = None
        if test:
            self.initial_state = 0

    def reset(self, seed=None, **kwargs):
        self.action_space.seed(seed)
        self.observation_space.seed(seed)
        env_obs, env_info = self.env.reset(seed=seed, **kwargs)
        if self.initial_state is None:  # monitor randomly on/off at beginning
            self.monitor_state = self.observation_space["mon"].sample()
        else:  # next reset start with switched monitor state
            self.monitor_state = self.initial_state
            self.initial_state = abs(self.initial_state - 1)
        return {"env": env_obs, "mon": self.monitor_state}, env_info

    def step(self, action):
        env_obs = self.env.unwrapped.get_state()
        (
            env_next_obs,
            env_reward,
            env_terminated,
            env_truncated,
            env_info,
        ) = self.env.step(action["env"])

        proxy_reward = np.nan
        monitor_reward = 0.0

        if action["env"] == 1 and env_obs == self.switch_cell_id:  # down in bottom-right cell
            if self.monitor_state == 1:
                self.monitor_state = 0
            elif self.monitor_state == 0:
                self.monitor_state = 1

        if self.monitor_state == 1:
            proxy_reward = env_reward
            monitor_reward = -self.monitor_cost

        monitor_obs = self.monitor_state

        obs = {"env": env_next_obs, "mon": monitor_obs}
        reward = {"env": env_reward, "mon": monitor_reward, "proxy": proxy_reward}
        terminated = env_terminated
        truncated = env_truncated

        return obs, reward, terminated, truncated, env_info
