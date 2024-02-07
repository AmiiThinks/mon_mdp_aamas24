import numpy as np
from abc import ABC, abstractmethod
from omegaconf import DictConfig

from src.utils import random_argmax
import src.parameter as parameter
from src.critic import Critic


class Actor(ABC):
    def __init__(self, critic: Critic):
        self._critic = critic
        self._train = True
        self.reset()

    @abstractmethod
    def __call__(self, obs_env, obs_mon):
        """
        Draw one action in one state. Not vectorized.
        """
        pass

    def greedy_call(self, obs_env, obs_mon):
        """
        Draw the greedy action, i.e., the one maximizing the critic's estimate
        of the state-action value. Not vectorized.
        """

        if self._critic._strategy in ["q_sequential", "q_joint"]:
            q_env = self._critic._q_env(obs_env)
            q_mon = self._critic._q_mon(obs_env, obs_mon)
            if self._critic._strategy == "q_sequential":
                act_env = random_argmax(q_env)[0]
                act_mon = random_argmax(q_mon[act_env])[0]
                return (act_env, act_mon)
            else:
                q = q_mon + q_env[:, None]
                return tuple(random_argmax(q))
        else:
            q = self._critic._q_joint(obs_env, obs_mon)
            return tuple(random_argmax(q))

    @abstractmethod
    def update(self):
        pass

    @abstractmethod
    def reset(self):
        pass

    def eval(self):
        self._train = False

    def train(self):
        self._train = True


class MonEpsilonGreedy(Actor):
    def __init__(
        self,
        critic: Critic,
        eps: DictConfig,
        **kwargs,
    ):
        """
        Args:
            critic (Critic): the critic providing estimates of state-action values,
            eps (DictConfig): configuration to initialize the exploration coefficient
                epsilon,
        """

        self._eps = getattr(parameter, eps.id)(**eps)
        Actor.__init__(self, critic)

    def __call__(self, obs_env, obs_mon):
        if self._train and np.random.random() < self._eps.value:
            return tuple(np.random.randint(self._critic.action_shape))
        else:
            return self.greedy_call(obs_env, obs_mon)

    def update(self):
        self._eps.step()

    def reset(self):
        self._eps.reset()
