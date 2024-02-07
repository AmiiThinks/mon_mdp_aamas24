import numpy as np
from collections import defaultdict

smoothing_window = 5
testing_frequency = 10
consecutive_steps_for_convergence = 200
y_tick_pad = -2
savedir = "ablation_q0"

q_init_values = ["-10.0", "0.0", "1.0"]

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
    # 'oracle': 'Oracle',
    "reward_model": "Rew. Model",
    "q_sequential": "Sequential",
    "q_joint": "Joint",
    "ignore": r"Ignore $\bot$",
    "reward_is_0.": r"$\bot$ = 0",
}

mon_to_xlim = defaultdict(lambda: None)
mon_to_xticks = defaultdict(lambda: None)
mon_to_xlim = {
    "Gridworld-Easy-3x3-v0_mes50/iStatelessBinaryMonitor": 500,
    "Gridworld-Medium-3x3-v0_mes50/iStatelessBinaryMonitor": 2000,
    "Gridworld-Medium-3x3-v0_mes50/iToySwitchMonitor": 2000,
}
mon_to_xticks = {
    "Gridworld-Easy-3x3-v0_mes50/iStatelessBinaryMonitor": [0, 250, 500],
    "Gridworld-Medium-3x3-v0_mes50/iStatelessBinaryMonitor": np.arange(0, 2001, 500),
    "Gridworld-Medium-3x3-v0_mes50/iToySwitchMonitor": np.arange(0, 2001, 500),
}

mon_to_ylim = defaultdict(lambda: [None, None])
mon_to_yticks = defaultdict(lambda: None)
mon_to_ylim = {
    "Gridworld-Easy-3x3-v0_mes50/iStatelessBinaryMonitor": [-0.3, 1.05],
    "Gridworld-Medium-3x3-v0_mes50/iStatelessBinaryMonitor": [-22.0, 1.2],
    "Gridworld-Medium-3x3-v0_mes50/iToySwitchMonitor": [-16, 1.1],
}
mon_to_yticks = {
    "Gridworld-Easy-3x3-v0_mes50/iStatelessBinaryMonitor": [0, 0.5, 1.0],
    "Gridworld-Medium-3x3-v0_mes50/iStatelessBinaryMonitor": [-20, -10.0, 0.0],
    "Gridworld-Medium-3x3-v0_mes50/iToySwitchMonitor": [-15, -10, -5, 0],
}
