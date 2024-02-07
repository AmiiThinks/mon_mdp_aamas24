import numpy as np

smoothing_window = 5
testing_frequency = 10
consecutive_steps_for_convergence = 200
y_tick_pad = -20
savedir = "deterministic_main"

q_init_values = ["-10.0"]

mon_to_label = {
    "Gridworld-Easy-3x3-v0_mes50/iStatelessBinaryMonitor": "Simple",
    "Gridworld-Medium-3x3-v0_mes50/iStatelessBinaryMonitor": "Penalty",
    "Gridworld-Medium-3x3-v0_mes50/iToySwitchMonitor": "Button",
}

mon_to_opt = {
    "Gridworld-Easy-3x3-v0_mes50/iStatelessBinaryMonitor": 0.99,
    "Gridworld-Medium-3x3-v0_mes50/iStatelessBinaryMonitor": 0.9509,
    "Gridworld-Medium-3x3-v0_mes50/iToySwitchMonitor": 0.552,
}

alg_to_label = {
    "oracle_with_reward_model": "Oracle",
    "reward_model": "Rew. Model",
}

mon_to_xlim = {
    "Gridworld-Easy-3x3-v0_mes50/iStatelessBinaryMonitor": 400,
    "Gridworld-Medium-3x3-v0_mes50/iStatelessBinaryMonitor": 1200,
    "Gridworld-Medium-3x3-v0_mes50/iToySwitchMonitor": 1200,
}
mon_to_xticks = {
    "Gridworld-Easy-3x3-v0_mes50/iStatelessBinaryMonitor": [0, 200, 400],
    "Gridworld-Medium-3x3-v0_mes50/iStatelessBinaryMonitor": [0, 400, 800, 1200],
    "Gridworld-Medium-3x3-v0_mes50/iToySwitchMonitor": [0, 400, 800, 1200],
}

mon_to_ylim = {
    "Gridworld-Easy-3x3-v0_mes50/iStatelessBinaryMonitor": [0.4, 1.05],
    "Gridworld-Medium-3x3-v0_mes50/iStatelessBinaryMonitor": [-0.1, 1.05],
    "Gridworld-Medium-3x3-v0_mes50/iToySwitchMonitor": [-4.1, 1.1],
}
mon_to_yticks = {
    "Gridworld-Easy-3x3-v0_mes50/iStatelessBinaryMonitor": [0, 0.5, 1.0],
    "Gridworld-Medium-3x3-v0_mes50/iStatelessBinaryMonitor": [0, 0.5, 1.0],
    "Gridworld-Medium-3x3-v0_mes50/iToySwitchMonitor": [-4.0, -3.0, -2.0, -1.0, 0.0, 1.0],
}
