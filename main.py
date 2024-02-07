import gymnasium
import wandb
import hydra
from omegaconf import DictConfig, OmegaConf
import os
import numpy as np
from pprint import pprint  # noqa: F401

from src.utils import dict_to_id
from src.actor import MonEpsilonGreedy
from src.critic import MonQTableCritic
from src.experiment import MonExperiment
from src.wrappers import monitor_wrappers


@hydra.main(version_base=None, config_path="configs", config_name="default")
def run(cfg: DictConfig) -> None:
    config = OmegaConf.to_container(cfg, resolve=True, throw_on_missing=True)
    # pprint(config)

    group = dict_to_id(cfg.environment) + "/" + dict_to_id(cfg.monitor)
    wandb.init(
        group=group,
        config=config,
        settings=wandb.Settings(
            start_method="thread",
            _disable_stats=True,
            _disable_meta=True,
        ),
        **cfg.wandb,
    )

    if cfg.monitor.id in [
        "StatelessBinaryMonitor",
        "LimitedUseMonitor",
        "LimitedUseBonusMonitor",
    ]:  # these are fully deterministic monitors
        cfg.experiment.testing_episodes = 1
    if cfg.monitor.id in ["ToySwitchMonitor"]:  # on/off initial monitor state
        cfg.experiment.testing_episodes = 2

    env = gymnasium.make(**cfg.environment)
    cfg.environment.id = cfg.environment.id.replace("-Stochastic", "")  # test with determ. rewards
    cfg.environment.max_episode_steps = 10  # greedy policies need less than 10 steps to go to goal
    env_test = gymnasium.make(**cfg.environment)
    env = getattr(monitor_wrappers, cfg.monitor.id)(env, **cfg.monitor)
    env_test = getattr(monitor_wrappers, cfg.monitor.id)(env_test, test=True, **cfg.monitor)

    critic = MonQTableCritic(
        env.observation_space["env"].n,
        env.observation_space["mon"].n,
        env.action_space["env"].n,
        env.action_space["mon"].n,
        **cfg.agent.critic,
    )
    actor = MonEpsilonGreedy(critic, **cfg.agent.actor)
    experiment = MonExperiment(env, env_test, actor, critic, **cfg.experiment)

    return_train_history, return_test_history = experiment.train()

    if cfg.experiment.debugdir is not None:
        from plot_gridworld_agent import plot_agent

        savepath = os.path.join(
            cfg.experiment.debugdir,
            group,
            str(float(cfg.agent.critic.q0)),
        )
        os.makedirs(savepath, exist_ok=True)
        plot_agent(critic, env, savepath)

    if cfg.experiment.datadir is not None:
        filepath = os.path.join(
            cfg.experiment.datadir,
            group,
            str(float(cfg.agent.critic.q0)),
        )
        os.makedirs(filepath, exist_ok=True)
        strat = cfg.agent.critic.strategy
        seed = str(cfg.experiment.rng_seed)
        savepath = os.path.join(filepath, strat + "_train_" + seed)
        np.save(savepath, np.array(return_train_history))
        savepath = os.path.join(filepath, strat + "_test_" + seed)
        np.save(savepath, np.array(return_test_history))

    wandb.finish()


if __name__ == "__main__":
    run()
