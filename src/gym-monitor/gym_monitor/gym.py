from gymnasium.envs.registration import register


def register_envs():
    register(
        id="Gridworld-Easy-2x3-v0",
        entry_point="gym_monitor.gridworld:Gridworld",
        max_episode_steps=50,
        kwargs={
            "grid": "2x3_easy",
        },
    )

    register(
        id="Gridworld-Medium-2x3-v0",
        entry_point="gym_monitor.gridworld:Gridworld",
        max_episode_steps=50,
        kwargs={
            "grid": "2x3_medium",
        },
    )

    register(
        id="Gridworld-Easy-3x3-v0",
        entry_point="gym_monitor.gridworld:Gridworld",
        max_episode_steps=50,
        kwargs={
            "grid": "3x3_easy",
        },
    )

    register(
        id="Gridworld-Medium-3x3-v0",
        entry_point="gym_monitor.gridworld:Gridworld",
        max_episode_steps=50,
        kwargs={
            "grid": "3x3_medium",
        },
    )

    register(
        id="Gridworld-Easy-3x3-Stochastic-v0",
        entry_point="gym_monitor.gridworld:Gridworld",
        max_episode_steps=50,
        kwargs={
            "grid": "3x3_easy",
            "random_action_prob": 0.0,
            "reward_noise_std": 0.05,
        },
    )

    register(
        id="Gridworld-Medium-3x3-Stochastic-v0",
        entry_point="gym_monitor.gridworld:Gridworld",
        max_episode_steps=50,
        kwargs={
            "grid": "3x3_medium",
            "random_action_prob": 0.0,
            "reward_noise_std": 0.05,
        },
    )
