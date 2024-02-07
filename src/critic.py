import numpy as np
from abc import ABC, abstractmethod
from omegaconf import DictConfig

import src.parameter as parameter
from src.approximator import MSETable, RunningMeanTable


def td_target(rwd: np.array, term: np.array, q_next: np.array, gamma: float):
    """
    Temporal-difference Q-Learning target. Vectorized.

    Args:
        rwd (np.array): r_t,
        term (np.array): True if s_t is terminal, False otherwise,
        q_next (np.array): max_a Q(s_{t+1}, a),
        gammma (float): discount factor,
    """

    return rwd + gamma * (1.0 - term) * q_next


class Critic(ABC):
    @abstractmethod
    def __init__(self, **kwargs):
        pass

    @abstractmethod
    def update(self, **kwargs):
        pass

    @abstractmethod
    def reset(self):
        pass

    def train(self):
        pass

    def eval(self):
        pass


class MonQCritic(Critic):
    """
    Generic class for Mon-MDP critics.
    Compared to classic critics, MonQCritic has multiple Q-functions, and a
    self._strategy attribute to define update rules and how to treat NaN rewards.
    """

    def __init__(
        self,
        strategy: str,
        gamma: float,
        lr: DictConfig,
        **kwargs,
    ):
        """
        Args:
            strategy (str): can be either "oracle", "reward_model", "q_sequential",
                "q_joint", "ignore", "reward_is_X", where X is a float.
            gamma (float): discount factor,
            lr (DictConfig): configuration to initialize the learning rate,
        """

        self.gamma = gamma
        self.lr = getattr(parameter, lr.id)(**lr)
        self._strategy = strategy
        self._rwd_consant = None
        if "reward_is_" in self._strategy:
            self._rwd_consant = float(self._strategy.split("is_")[-1])

        self._q_env = None
        self._q_mon = None
        self._q_joint = None

        self._q_env_target = None  # for computing Q(next_state) in the TD target
        self._q_mon_target = None
        self._q_joint_target = None

    def _parse_reward(
        self,
        obs_env, obs_mon,
        act_env, act_mon,
        rwd_env, rwd_proxy,
    ):  # fmt: skip
        """
        Parse NaN proxy rewards.
        """

        unobserv = np.isnan(rwd_proxy)
        observ = np.logical_not(unobserv)

        if self._strategy == "oracle":
            rwd_proxy = rwd_env
        elif self._strategy == "reward_model":
            if np.any(observ):
                self._r_env.update(
                    obs_env[observ], act_env[observ], target=rwd_proxy[observ],
                )  # fmt: skip
            rwd_proxy = self._r_env(obs_env, act_env)
        elif "reward_is_" in self._strategy:
            rwd_proxy[unobserv] = self._rwd_consant
        elif self._strategy == "oracle_with_reward_model":
            self._r_env.update(obs_env, act_env, target=rwd_env)
            rwd_proxy = self._r_env(obs_env, act_env)
        elif self._strategy in ["q_joint", "q_sequential", "ignore"]:
            pass
        else:
            raise ValueError("unknown update strategy")

        return rwd_proxy

    def update(
        self,
        obs_env, obs_mon,
        act_env, act_mon,
        rwd_env, rwd_mon, rwd_proxy,
        term,
        next_obs_env, next_obs_mon,
    ):  # fmt: skip
        """
        TD updates of the critics.
        """

        rwd_proxy = self._parse_reward(
            obs_env, obs_mon, act_env, act_mon, rwd_env, rwd_proxy,
        )  # fmt: skip

        unobserv = np.isnan(rwd_proxy)
        observ = np.logical_not(unobserv)
        skip = np.all(unobserv)

        error_env = np.array([0.0])
        error_mon = np.array([0.0])
        error_joint = np.array([0.0])

        if self._strategy in ["q_joint", "q_sequential"]:
            if not skip:
                q_env_next = self._q_env_target(next_obs_env[observ])
                target_env = td_target(
                    rwd_proxy[observ], term[observ], q_env_next.max(-1), self.gamma,
                )  # fmt: skip
                error_env = self._q_env.update(
                    obs_env[observ],
                    act_env[observ],
                    target=target_env,
                    stepsize=self.lr.value,
                )

            q_mon_next = self._q_mon_target(next_obs_env, next_obs_mon)
            if self._strategy in ["q_sequential"]:
                q_env_next = self._q_env(next_obs_env)
                mask = q_env_next == q_env_next.max(-1, keepdims=True)
                q_mon_next = np.where(mask[..., None], q_mon_next, -np.inf)

            target_mon = td_target(
                rwd_mon, term, q_mon_next.max((-2, -1)), self.gamma,
            )  # fmt: skip
            error_mon = self._q_mon.update(
                obs_env, obs_mon,
                act_env, act_mon,
                target=target_mon,
                stepsize=self.lr.value,
            )  # fmt: skip

        else:
            if not skip:
                rwd_joint = rwd_proxy + rwd_mon
                q_joint_next = self._q_joint_target(
                    next_obs_env[observ], next_obs_mon[observ],
                )  # fmt: skip
                target_joint = td_target(
                    rwd_joint[observ], term[observ], q_joint_next.max((-2, -1)), self.gamma,
                )  # fmt: skip
                error_joint = self._q_joint.update(
                    obs_env[observ], obs_mon[observ],
                    act_env[observ], act_mon[observ],
                    target=target_joint,
                    stepsize=self.lr.value,
                )  # fmt: skip

        self.lr.step()
        return (error_env.mean() + error_mon.mean() + error_joint.mean()).sum()

    def reset(self):
        self._q_env.reset()
        self._q_mon.reset()
        self._q_joint.reset()
        self._r_env.reset()


class MonQTableCritic(MonQCritic):
    """
    Instance of MonQCritic that uses tabular Q-function critics.
    """

    def __init__(
        self,
        n_obs_env: int,
        n_obs_mon: int,
        n_act_env: int,
        n_act_mon: int,
        q0: float = 0.0,
        **kwargs,
    ):
        MonQCritic.__init__(self, **kwargs)
        self.action_shape = (n_act_env, n_act_mon)

        self._r_env = RunningMeanTable(n_obs_env, n_act_env)

        self._q_env = MSETable(n_obs_env, n_act_env, init_value=q0)
        self._q_mon = MSETable(n_obs_env, n_obs_mon, n_act_env, n_act_mon, init_value=q0)
        self._q_joint = MSETable(n_obs_env, n_obs_mon, n_act_env, n_act_mon, init_value=q0)

        self._q_env_target = self._q_env  # with tabular Q we don't need a different target function
        self._q_mon_target = self._q_mon
        self._q_joint_target = self._q_joint

        self.reset()
