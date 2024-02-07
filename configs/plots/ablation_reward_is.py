import numpy as np
from collections import defaultdict

smoothing_window = 5
testing_frequency = 10
consecutive_steps_for_convergence = 200
y_tick_pad = -2
savedir = "ablation_reward_is"

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
    "reward_is_-10.": r"$\bot$ = -10",
    "reward_is_0.": r"$\bot$ = 0",
    "reward_is_1.": r"$\bot$ = 1",
}

mon_to_xlim = defaultdict(lambda: 10000)
mon_to_xticks = defaultdict(lambda: np.arange(0, 10001, 2000))

mon_to_ylim = {
    "Gridworld-Easy-3x3-v0_mes50/iStatelessBinaryMonitor": [-0.3, 1.05],
    "Gridworld-Medium-3x3-v0_mes50/iStatelessBinaryMonitor": [-21.0, 1.05],
    "Gridworld-Medium-3x3-v0_mes50/iToySwitchMonitor": [-16, -4],
}
mon_to_yticks = {
    "Gridworld-Easy-3x3-v0_mes50/iStatelessBinaryMonitor": [0, 0.5, 1.0],
    "Gridworld-Medium-3x3-v0_mes50/iStatelessBinaryMonitor": [-20, -10.0, 0.0],
    "Gridworld-Medium-3x3-v0_mes50/iToySwitchMonitor": [-15, -10, -5],
}
